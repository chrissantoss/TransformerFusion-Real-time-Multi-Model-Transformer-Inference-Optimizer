#pragma once

#include <string>
#include <vector>
#include <iostream>

class BenchmarkSuite {
public:
    void run_comprehensive_benchmark();
    void benchmark_sequence_lengths();
    void benchmark_batch_sizes();
    void benchmark_against_baseline();
    void benchmark_memory_efficiency();
    void generate_comparison_plots();
    void run_huggingface_baseline();
    void run_fusion_implementation();
    void compare_results();
    void save_plot_data(const std::string& filename);

private:
    std::vector<float> sequence_length_results_;
    std::vector<float> batch_size_results_;
    std::vector<float> baseline_results_;
    std::vector<float> fusion_results_;
    std::vector<float> memory_usage_results_;
}; 