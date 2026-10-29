#pragma once
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ezlocalai {
enum class ModelType { Llm, VisionLlm, Tts, Stt, Image, Video, Music, Embedding, VoiceConversion };
const char *model_type_name(ModelType type);
double model_vram_estimate(ModelType type);
struct ModelResource {
    ModelType type;
    std::string name, device;
    double vram_gb = 0;
    bool in_use = false;
    uint64_t active_requests = 0;
    double last_used = 0;
};
struct CudaAllocatorStats {
    int index = 0;
    double allocated_gb = 0, reserved_gb = 0, max_allocated_gb = 0, free_gb = 0, total_gb = 0;
    std::string error;
};
struct CudaAllocatorProbes {
    std::function<bool()> available;
    std::function<int()> device_count;
    std::function<std::pair<double, double>(int)> memory_info;
    std::function<double(int)> allocated_bytes, reserved_bytes, max_allocated_bytes;
};
std::vector<CudaAllocatorStats> collect_cuda_allocator_stats(const CudaAllocatorProbes &probes);
// Separate probes preserve the reference's repeated live memory queries. The
// constructor caches capacity, while decisions and snapshots read free memory.
struct ResourceProbes {
    std::function<int()> gpu_count;
    std::function<std::vector<double>()> total_per_gpu, free_per_gpu;
    std::function<double(std::optional<int>)> free_vram;
    std::function<int()> primary_gpu;
    std::function<std::optional<int>()> secondary_gpu;
    std::function<std::vector<CudaAllocatorStats>()> allocator_stats;
    std::function<double()> system_ram, available_ram, now;
    std::function<std::string()> fallback_server;
    // Optional in-process allocator. Subprocess engines release their own
    // allocations when destroyed; LibTorch builds additionally clear its cache.
    std::function<void()> release_unused_cuda_memory;
};
ResourceProbes native_resource_probes();
struct ModelAdmission { bool can_load; std::string device, reason; };
struct ResourceStatus {
    int gpu_count;
    double total_vram_gb, free_vram_gb, system_ram_gb;
    std::vector<double> free_per_gpu_gb;
    std::vector<CudaAllocatorStats> torch_cuda;
    std::vector<ModelResource> loaded_models;
};
class ResourceManager {
public:
    explicit ResourceManager(ResourceProbes probes = native_resource_probes());
    bool is_model_loaded(ModelType type) const;
    bool is_model_in_use(ModelType type) const;
    uint64_t get_model_active_count(ModelType type) const;
    void mark_model_in_use(ModelType type, bool in_use = true);
    void register_model(ModelType type, std::string name, std::string device,
                        std::optional<double> vram_gb = std::nullopt, bool pinned = false);
    void unregister_model(ModelType type);
    double get_free_vram(std::optional<int> ordinal = std::nullopt) const;
    double get_total_free_vram() const;
    const std::vector<double> &gpu_capacity_gb() const { return per_gpu_capacity_; }
    void release_unused_cuda_memory() const { if (probes_.release_unused_cuda_memory) probes_.release_unused_cuda_memory(); }
    std::vector<double> get_per_gpu_free_vram() const;
    double get_free_ram() const;
    double current_time() const;
    ModelAdmission can_load_model(ModelType type, std::optional<double> required_vram = std::nullopt) const;
    std::vector<ModelType> get_models_to_unload(double required_vram, const std::vector<ModelType> &exclude = {}) const;
    std::vector<ModelResource> loaded_models() const;
    ResourceStatus get_status() const;
private:
    ResourceProbes probes_;
    int gpu_count_;
    double total_vram_, system_ram_;
    std::vector<double> per_gpu_capacity_;
    mutable std::mutex mutex_;
    // Replacing an existing type preserves insertion order, just like dict.
    std::vector<ModelResource> models_;
};
}
