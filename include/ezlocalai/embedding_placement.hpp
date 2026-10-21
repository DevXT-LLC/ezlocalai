#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ezlocalai {
struct EmbeddingMetadata {
    int64_t layers = 28, kv_heads = 8, key_length = 128, value_length = 128;
};
struct EmbeddingMemory {
    int64_t layers = 28;
    double total_gb = 0, fixed_gb = 0, per_layer_gb = 0, weights_gb = 0, kv_gb = 0, compute_gb = 0;
};
EmbeddingMetadata embedding_model_metadata(const std::string &path);
double embedding_cache_bytes(const std::string &cache);
EmbeddingMemory estimate_embedding_memory(const EmbeddingMetadata &metadata, uint64_t file_bytes,
    int context, int parallel, int ubatch, const std::string &cache);
EmbeddingMemory estimate_embedding_memory(const std::string &path, int context, int parallel, int ubatch, const std::string &cache);
int resolve_embedding_gpu_layers(const EmbeddingMemory &memory, std::optional<int> requested,
    bool force_cpu, bool cuda_available, double free_gb, double safety_gb);
std::vector<int> embedding_fallback_layers(int selected, int64_t layers);
double embedding_gpu_memory(const EmbeddingMemory &memory, int selected);
bool embedding_memory_error(std::string message);
// The reference's warmup predicate treats an empty KEEP_LOADED as disabled,
// unlike the pool retention predicate. Returned bodies are in inference order.
std::vector<std::string> embedding_warmup_inputs(bool cuda, int context, int ubatch);
}
