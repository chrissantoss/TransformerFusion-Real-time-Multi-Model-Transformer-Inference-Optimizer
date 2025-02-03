#include <cmath>
#include <vector>

void apply_softmax(float* scores, int size) {
    // CPU implementation of softmax
    float max_val = -INFINITY;
    for (int i = 0; i < size; i++) {
        max_val = std::max(max_val, scores[i]);
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        scores[i] = std::exp(scores[i] - max_val);
        sum += scores[i];
    }
    
    for (int i = 0; i < size; i++) {
        scores[i] /= sum;
    }
}