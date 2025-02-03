#include "cuda_runtime.h"
#include "cooperative_groups.h"
namespace cg = cooperative_groups;

// Constants for optimization
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr float SCALE_FACTOR = 1.0f / sqrt(64.0f); // For head size 64

__device__ void apply_softmax(float* scores, int size) {
    float max_val = -INFINITY;
    for (int i = 0; i < size; i++) {
        max_val = max(max_val, scores[i]);
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        scores[i] = __expf(scores[i] - max_val);
        sum += scores[i];
    }
    
    for (int i = 0; i < size; i++) {
        scores[i] /= sum;
    }
}

__device__ void compute_qkv_transform(
    const float* __restrict__ input,
    const float* __restrict__ qkv_weights,
    float* shared_qkv,
    const int hidden_size,
    const int head_size
) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);
    
    // Use warp-level matrix multiplication
    #pragma unroll
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float q = 0.0f, k = 0.0f, v = 0.0f;
        
        // Vectorized memory access
        for (int j = 0; j < hidden_size; j += 4) {
            float4 inp = reinterpret_cast<const float4*>(&input[j])[0];
            float4 wq = reinterpret_cast<const float4*>(&qkv_weights[i * hidden_size + j])[0];
            float4 wk = reinterpret_cast<const float4*>(&qkv_weights[(hidden_size + i) * hidden_size + j])[0];
            float4 wv = reinterpret_cast<const float4*>(&qkv_weights[(2 * hidden_size + i) * hidden_size + j])[0];
            
            q += inp.x * wq.x + inp.y * wq.y + inp.z * wq.z + inp.w * wq.w;
            k += inp.x * wk.x + inp.y * wk.y + inp.z * wk.z + inp.w * wk.w;
            v += inp.x * wv.x + inp.y * wv.y + inp.z * wv.z + inp.w * wv.w;
        }
        
        // Warp-level reduction
        q = warp.shfl_down(q, 16);
        k = warp.shfl_down(k, 16);
        v = warp.shfl_down(v, 16);
        
        // Store results
        if (threadIdx.x % WARP_SIZE == 0) {
            shared_qkv[i] = q;
            shared_qkv[hidden_size + i] = k;
            shared_qkv[2 * hidden_size + i] = v;
        }
    }
}

__global__ void transformer_layer_fusion(
    const float* __restrict__ input,
    const float* __restrict__ qkv_weights,
    const float* __restrict__ ffn_weights,
    float* __restrict__ output,
    const int batch_size,
    const int seq_length,
    const int hidden_size,
    const int num_heads,
    const float early_stopping_threshold
) {
    extern __shared__ float shared_mem[];
    float* shared_qkv = shared_mem;
    float* attention_scores = &shared_mem[3 * hidden_size];
    
    const int head_size = hidden_size / num_heads;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    // Step 1: QKV Transform
    compute_qkv_transform(input, qkv_weights, shared_qkv, hidden_size, head_size);
    __syncthreads();
    
    // Step 2: Attention Scores
    const int scores_per_thread = (seq_length * seq_length) / blockDim.x;
    #pragma unroll
    for (int i = 0; i < scores_per_thread; i++) {
        const int idx = tid * scores_per_thread + i;
        const int q_idx = idx / seq_length;
        const int k_idx = idx % seq_length;
        
        if (idx < seq_length * seq_length) {
            float score = 0.0f;
            
            // Compute attention score
            for (int h = 0; h < head_size; h++) {
                score += shared_qkv[q_idx * head_size + h] * 
                        shared_qkv[hidden_size + k_idx * head_size + h];
            }
            attention_scores[idx] = score * SCALE_FACTOR;
        }
    }
    __syncthreads();
    
    // Step 3: Softmax
    if (tid < seq_length) {
        apply_softmax(&attention_scores[tid * seq_length], seq_length);
    }
    __syncthreads();
    
    // Step 4: Attention Output
    if (tid < hidden_size) {
        float out_val = 0.0f;
        for (int i = 0; i < seq_length; i++) {
            float attention_weight = attention_scores[tid * seq_length + i];
            out_val += attention_weight * shared_qkv[2 * hidden_size + i * head_size + (tid % head_size)];
        }
        output[bid * hidden_size + tid] = out_val;
    }
} 