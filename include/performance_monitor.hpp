#pragma once

#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

class PerformanceMonitor {
public:
    void record_latency(float ms);
    void record_memory_usage(size_t bytes);
    void generate_report();
    
private:
    std::vector<float> latencies_;
    std::vector<size_t> memory_usage_;
    
    float calculate_average(const std::vector<float>& values) {
        if (values.empty()) return 0.0f;
        return std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
    }
    
    float calculate_percentile(const std::vector<float>& values, float percentile) {
        if (values.empty()) return 0.0f;
        
        std::vector<float> sorted_values = values;
        std::sort(sorted_values.begin(), sorted_values.end());
        
        size_t index = static_cast<size_t>(std::ceil(percentile * sorted_values.size())) - 1;
        return sorted_values[index];
    }
    
    size_t get_peak_memory() {
        if (memory_usage_.empty()) return 0;
        return *std::max_element(memory_usage_.begin(), memory_usage_.end());
    }
    
    void plot_latency_distribution() {
        // Placeholder for plotting functionality
        std::cout << "Latency distribution plot would be generated here.\n";
    }
}; 