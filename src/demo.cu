#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include "transformer_fusion.hpp"

// Utility function to generate random data
void generate_test_data(float* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Benchmark function
double benchmark_inference(TransformerFusion& fusion, 
                         float* input, 
                         float* qkv_weights,
                         float* ffn_weights,
                         float* output,
                         int num_iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        fusion.forward(input, qkv_weights, ffn_weights, output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds / num_iterations;
}

int main() {
    // Configuration
    TransformerConfig config{
        .batch_size = 1,
        .seq_length = 512,
        .hidden_size = 768,
        .num_heads = 12,
        .early_stopping_threshold = 0.95f
    };

    // Allocate memory
    size_t input_size = config.batch_size * config.seq_length * config.hidden_size;
    size_t qkv_size = 3 * config.hidden_size * config.hidden_size;
    size_t ffn_size = 4 * config.hidden_size * config.hidden_size;
    
    float *h_input, *h_qkv_weights, *h_ffn_weights, *h_output;
    float *d_input, *d_qkv_weights, *d_ffn_weights, *d_output;
    
    // Host memory allocation
    h_input = new float[input_size];
    h_qkv_weights = new float[qkv_size];
    h_ffn_weights = new float[ffn_size];
    h_output = new float[input_size];
    
    // Generate test data
    generate_test_data(h_input, input_size);
    generate_test_data(h_qkv_weights, qkv_size);
    generate_test_data(h_ffn_weights, ffn_size);
    
    // Device memory allocation
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_qkv_weights, qkv_size * sizeof(float));
    cudaMalloc(&d_ffn_weights, ffn_size * sizeof(float));
    cudaMalloc(&d_output, input_size * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qkv_weights, h_qkv_weights, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ffn_weights, h_ffn_weights, ffn_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create transformer fusion instance
    TransformerFusion fusion(config);
    
    // Warmup
    fusion.forward(d_input, d_qkv_weights, d_ffn_weights, d_output);
    
    // Benchmark
    int num_iterations = 100;
    double avg_latency = benchmark_inference(fusion, 
                                          d_input, 
                                          d_qkv_weights, 
                                          d_ffn_weights, 
                                          d_output, 
                                          num_iterations);
    
    std::cout << "Average latency: " << avg_latency << " ms" << std::endl;
    std::cout << "Throughput: " << 1000.0 / avg_latency << " inferences/second" << std::endl;
    
    // Cleanup
    delete[] h_input;
    delete[] h_qkv_weights;
    delete[] h_ffn_weights;
    delete[] h_output;
    
    cudaFree(d_input);
    cudaFree(d_qkv_weights);
    cudaFree(d_ffn_weights);
    cudaFree(d_output);
    
    return 0;
} 