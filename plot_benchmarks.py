import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_benchmarks(csv_file):
    # Read benchmark data
    data = pd.read_csv(csv_file)
    
    # Set up plot style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Plot sequence length vs latency
    plt.subplot(2, 2, 1)
    sns.lineplot(x='sequence_length', y='latency', data=data)
    plt.title('Sequence Length vs Latency')
    plt.xlabel('Sequence Length')
    plt.ylabel('Latency (ms)')
    
    # Plot batch size vs throughput
    plt.subplot(2, 2, 2)
    sns.lineplot(x='batch_size', y='throughput', data=data)
    plt.title('Batch Size vs Throughput')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (inferences/sec)')
    
    # Plot memory usage
    plt.subplot(2, 2, 3)
    sns.lineplot(x='sequence_length', y='memory_usage', data=data)
    plt.title('Sequence Length vs Memory Usage')
    plt.xlabel('Sequence Length')
    plt.ylabel('Memory Usage (MB)')
    
    # Plot baseline comparison
    plt.subplot(2, 2, 4)
    sns.barplot(x='implementation', y='latency', data=data)
    plt.title('Implementation Comparison')
    plt.xlabel('Implementation')
    plt.ylabel('Latency (ms)')
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.show()

if __name__ == "__main__":
    plot_benchmarks('benchmark_data.csv') 