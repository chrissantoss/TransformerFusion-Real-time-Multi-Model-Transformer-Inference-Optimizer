#include "transformer_fusion.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>

class KVCacheManager {
public:
    KVCacheManager(const TransformerConfig& config) 
        : config_(config), cublas_handle_(nullptr) {
        cudaStreamCreate(&stream_);
        cublasCreate(&cublas_handle_);
        
        // Allocate pinned memory for cache
        cache_size_ = config.batch_size * config.seq_length * config.hidden_size * 2;
        cudaMallocHost(&host_cache_, cache_size_ * sizeof(float));
        cudaMalloc(&device_cache_, cache_size_ * sizeof(float));
        
        // Initialize cache metadata
        cache_entries_.reserve(config.seq_length);
        current_pos_ = 0;
        
        // Initialize pruning threshold
        pruning_threshold_ = 0.1f; // Configurable
    }
    
    void update_cache(const float* new_kv, int seq_pos, cudaStream_t stream = nullptr) {
        const size_t entry_size = config_.hidden_size * 2;
        const size_t offset = seq_pos * entry_size;
        
        // Asynchronous update
        cudaMemcpyAsync(device_cache_ + offset,
                       new_kv,
                       entry_size * sizeof(float),
                       cudaMemcpyDeviceToDevice,
                       stream ? stream : stream_);
        
        // Update metadata
        CacheEntry entry{
            .position = seq_pos,
            .timestamp = current_pos_++,
            .relevance_score = 1.0f
        };
        cache_entries_.push_back(entry);
        
        // Trigger pruning if needed
        if (current_pos_ > config_.seq_length * 0.8) {
            prune_cache();
        }
    }
    
    void prune_cache() {
        // Sort entries by relevance score
        std::sort(cache_entries_.begin(), cache_entries_.end(),
                 [](const CacheEntry& a, const CacheEntry& b) {
                     return a.relevance_score > b.relevance_score;
                 });
        
        // Keep top 80% entries
        const size_t keep_count = cache_entries_.size() * 0.8;
        cache_entries_.resize(keep_count);
        
        // Compact cache memory
        compact_cache();
    }
    
private:
    struct CacheEntry {
        int position;
        int timestamp;
        float relevance_score;
    };
    
    void compact_cache() {
        const size_t entry_size = config_.hidden_size * 2;
        
        // Create temporary buffer for compaction
        float* temp_buffer;
        cudaMalloc(&temp_buffer, cache_size_ * sizeof(float));
        
        // Copy valid entries to temporary buffer
        for (size_t i = 0; i < cache_entries_.size(); i++) {
            const size_t src_offset = cache_entries_[i].position * entry_size;
            const size_t dst_offset = i * entry_size;
            
            cudaMemcpyAsync(temp_buffer + dst_offset,
                           device_cache_ + src_offset,
                           entry_size * sizeof(float),
                           cudaMemcpyDeviceToDevice,
                           stream_);
        }
        
        // Swap buffers
        std::swap(temp_buffer, device_cache_);
        cudaFree(temp_buffer);
        
        // Update positions
        for (size_t i = 0; i < cache_entries_.size(); i++) {
            cache_entries_[i].position = i;
        }
        
        current_pos_ = cache_entries_.size();
    }
    
    TransformerConfig config_;
    float* host_cache_;
    float* device_cache_;
    size_t cache_size_;
    std::vector<CacheEntry> cache_entries_;
    int current_pos_;
    float pruning_threshold_;
    cudaStream_t stream_;
    cublasHandle_t cublas_handle_;
}; 