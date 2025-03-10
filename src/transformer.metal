#include <metal_stdlib>
using namespace metal;

kernel void transformerKernel(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    // Compute softmax
    float max_val = -INFINITY;
    for (uint i = 0; i < 256; i++) {
        max_val = max(max_val, input[i]);
    }
    
    float sum = 0.0f;
    for (uint i = 0; i < 256; i++) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }
    
    for (uint i = 0; i < 256; i++) {
        output[i] /= sum;
    }
} 