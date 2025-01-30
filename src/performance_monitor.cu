class PerformanceMonitor {
public:
    void record_latency(float ms) {
        latencies_.push_back(ms);
    }
    
    void record_memory_usage(size_t bytes) {
        memory_usage_.push_back(bytes);
    }
    
    void generate_report() {
        // Generate detailed performance report
        std::cout << "\nPerformance Report:\n";
        std::cout << "Average Latency: " << calculate_average(latencies_) << " ms\n";
        std::cout << "99th Percentile Latency: " << calculate_percentile(latencies_, 0.99) << " ms\n";
        std::cout << "Peak Memory Usage: " << get_peak_memory() / 1024.0 / 1024.0 << " MB\n";
        
        // Generate plots using matplotlib-cpp or save data for external plotting
        plot_latency_distribution();
    }

private:
    std::vector<float> latencies_;
    std::vector<size_t> memory_usage_;
}; 