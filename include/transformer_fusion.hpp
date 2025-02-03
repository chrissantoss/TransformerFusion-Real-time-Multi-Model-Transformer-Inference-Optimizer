#include <cstdint>  // For int64_t
#include <cstring>  // For std::memcpy
#include <vector>   // For std::vector

struct TransformerConfig {
    int batch_size;
    int seq_length; 
    int hidden_size;
    int num_heads;
    float early_stopping_threshold;
};

class TransformerFusion {
public:
    TransformerFusion(const TransformerConfig& config);
    
    void forward(
        const float* input,
        const float* qkv_weights,
        const float* ffn_weights,
        float* output
    );
    
private:
    // Custom memory manager for KV cache with efficient memory allocation and pruning
    struct CacheEntry {
        float* key_cache;
        float* value_cache;
        bool is_valid;
        int64_t timestamp;
    };

    class KVCacheManager {
    public:
        KVCacheManager(int batch_size, int seq_length, int hidden_size, int num_heads)
            : batch_size_(batch_size), seq_length_(seq_length),
              hidden_size_(hidden_size), num_heads_(num_heads) {
            cache_entries_.resize(batch_size * seq_length);
        }

        void store(int batch_idx, int seq_idx, const float* key, const float* value) {
            CacheEntry& entry = get_entry(batch_idx, seq_idx);
            if (!entry.is_valid) {
                size_t cache_size = hidden_size_ * num_heads_;
                entry.key_cache = new float[cache_size];
                entry.value_cache = new float[cache_size];
                entry.is_valid = true;
            }
            size_t cache_size = hidden_size_ * num_heads_;
            std::memcpy(entry.key_cache, key, cache_size * sizeof(float));
            std::memcpy(entry.value_cache, value, cache_size * sizeof(float));
            entry.timestamp = current_timestamp_++;
        }

        bool retrieve(int batch_idx, int seq_idx, float* key_out, float* value_out) {
            CacheEntry& entry = get_entry(batch_idx, seq_idx);
            if (!entry.is_valid) {
                return false;
            }
            size_t cache_size = hidden_size_ * num_heads_;
            std::memcpy(key_out, entry.key_cache, cache_size * sizeof(float));
            std::memcpy(value_out, entry.value_cache, cache_size * sizeof(float));
            return true;
        }

        void clear() {
            for (auto& entry : cache_entries_) {
                if (entry.is_valid) {
                    delete[] entry.key_cache;
                    delete[] entry.value_cache;
                    entry.is_valid = false;
                }
            }
        }

    private:
        CacheEntry& get_entry(int batch_idx, int seq_idx) {
            return cache_entries_[batch_idx * seq_length_ + seq_idx];
        }

        std::vector<CacheEntry> cache_entries_;
        int batch_size_;
        int seq_length_;
        int hidden_size_;
        int num_heads_;
        int64_t current_timestamp_;
    };
    
    TransformerConfig config_;
    KVCacheManager kv_cache_;

    // Add these private member function declarations
    void matmul(const float* a, const float* b, float* c, int m, int n);
    void apply_attention(float* qkv, float* output);
}; 