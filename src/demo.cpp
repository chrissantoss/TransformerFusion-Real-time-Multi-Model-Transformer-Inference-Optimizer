#include <iostream>
#include <iomanip>
#include "transformer_fusion.hpp"

void print_array(const float* arr, int size, const std::string& name) {
    std::cout << name << " (first 10 elements):\n";
    for (int i = 0; i < std::min(10, size); i++) {
        std::cout << std::fixed << std::setprecision(4) << arr[i] << " ";
    }
    std::cout << "\n\n";
}

int main() {
    // Initialize transformer config
    TransformerConfig config;
    config.batch_size = 1;
    config.seq_length = 128;
    config.hidden_size = 768;
    config.num_heads = 12;
    config.early_stopping_threshold = 0.95f;

    std::cout << "Initializing Transformer with:\n";
    std::cout << "  Batch size: " << config.batch_size << "\n";
    std::cout << "  Sequence length: " << config.seq_length << "\n";
    std::cout << "  Hidden size: " << config.hidden_size << "\n";
    std::cout << "  Number of heads: " << config.num_heads << "\n\n";

    // Create transformer fusion instance
    TransformerFusion transformer(config);

    // Allocate input/output buffers
    const int input_size = config.batch_size * config.seq_length * config.hidden_size;
    const int weight_size = config.hidden_size * config.hidden_size;
    
    float* input = new float[input_size];
    float* qkv_weights = new float[weight_size * 3]; // Query, key, value weights
    float* ffn_weights = new float[weight_size * 2]; // FFN weights
    float* output = new float[input_size];

    // Initialize input and weights (simplified for demo)
    for (int i = 0; i < input_size; i++) {
        input[i] = 0.1f;
    }
    for (int i = 0; i < weight_size * 3; i++) {
        qkv_weights[i] = 0.01f;
    }
    for (int i = 0; i < weight_size * 2; i++) {
        ffn_weights[i] = 0.01f;
    }

    // Print initial values
    print_array(input, input_size, "Input");
    print_array(qkv_weights, weight_size * 3, "QKV Weights");

    std::cout << "Running transformer forward pass...\n";
    transformer.forward(input, qkv_weights, ffn_weights, output);

    // Print results
    print_array(output, input_size, "Output");

    // Cleanup
    delete[] input;
    delete[] qkv_weights;
    delete[] ffn_weights;
    delete[] output;

    std::cout << "Transformer demo completed successfully!\n";
    return 0;
}