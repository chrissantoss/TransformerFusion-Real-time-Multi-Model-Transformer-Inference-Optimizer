class BenchmarkSuite {
public:
    void run_comprehensive_benchmark() {
        benchmark_sequence_lengths();
        benchmark_batch_sizes();
        benchmark_against_baseline();
        benchmark_memory_efficiency();
        generate_comparison_plots();
    }
    
    void benchmark_against_baseline() {
        // Compare against huggingface/transformer implementation
        run_huggingface_baseline();
        run_fusion_implementation();
        compare_results();
    }

    void save_plot_data(const std::string& filename) {
        // Implementation of save_plot_data method
    }
};

BenchmarkSuite suite;

// Run specific benchmarks
suite.benchmark_sequence_lengths();
suite.benchmark_batch_sizes();
suite.benchmark_against_baseline();

// Save plot data
suite.save_plot_data("benchmark_data.csv"); 