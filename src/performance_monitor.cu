#include "performance_monitor.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

void PerformanceMonitor::record_latency(float ms) {
    latencies_.push_back(ms);
}

void PerformanceMonitor::record_memory_usage(size_t bytes) {
    memory_usage_.push_back(bytes);
}

void PerformanceMonitor::generate_report() {
    // Generate detailed performance report
    std::cout << "\nPerformance Report:\n";
    std::cout << "Average Latency: " << calculate_average(latencies_) << " ms\n";
    std::cout << "99th Percentile Latency: " << calculate_percentile(latencies_, 0.99) << " ms\n";
    std::cout << "Peak Memory Usage: " << get_peak_memory() / 1024.0 / 1024.0 << " MB\n";
    
    // Generate plots using matplotlib-cpp or save data for external plotting
    plot_latency_distribution();
} 