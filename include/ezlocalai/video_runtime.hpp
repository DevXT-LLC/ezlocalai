#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ezlocalai {
struct CudaDevice {
    int ordinal = 0;
    std::string name;
    int major = 0, minor = 0;
    uint64_t total_memory = 0;
};
struct RankedCudaDevice { CudaDevice device; double score; };
// Query process-visible ordinals through CUDA, or HIP in an explicit HIP build,
// without allocating model memory. Keep PyTorch-compatible public CUDA naming.
std::vector<CudaDevice> visible_cuda_devices();
int visible_cuda_device_count();
// Router hardware metadata distinguishes ROCm from CUDA, unlike Torch devices.
std::string gpu_backend_name();
// The pinned llama engine exposes different device names for CUDA and HIP.
std::string llama_gpu_device_name(const std::string &logical_ordinal);
// Strict per-device probes; failures propagate independently so callers can
// preserve the source's distinct memory/capability and profile fallbacks.
std::pair<int, int> cuda_compute_capability(int ordinal);
CudaDevice cuda_device_properties(int ordinal);
// Selected device's SM count and maximum resident threads per SM. Used to
// reproduce CUDA reduction order without a toolkit dependency in the worker.
std::pair<int, int> cuda_multiprocessor_geometry(int ordinal);
struct CudaMemoryInfo { uint64_t free_bytes, total_bytes; };
// Strict probe used by ResourceManager: CUDA failures propagate to callers.
CudaMemoryInfo cuda_memory_info(int ordinal);
// Mirrors torch.cuda.mem_get_info on the process-visible ordinal, falling back
// to ordinal zero for an invalid index and returning zero if probing fails.
// Retains the primary context, then restores the caller's current context.
double cuda_free_memory_gb(int ordinal = 0);
std::array<int64_t, 3> gpu_generation_score(std::string name);
std::vector<RankedCudaDevice> rank_cuda_devices(const std::vector<CudaDevice> &devices);
std::optional<int> secondary_cuda_device(const std::vector<CudaDevice> &devices);
std::string video_device(std::optional<int> secondary, bool force_cpu = false,
                         bool can_load = true, const std::string &resource_device = "cuda");
bool video_should_unload_llm(bool resident_llm, std::optional<int> secondary, const std::function<bool()> &forwarding = {});
bool video_should_unload_llm(bool resident_llm, const std::function<std::optional<int>()> &secondary, const std::function<bool()> &forwarding = {});
bool video_should_reload_llm();
bool has_image_server_url();
bool video_requires_exclusive_worker(bool resident_llm, const std::function<std::optional<int>()> &secondary, const std::function<bool()> &forwarding = {});
bool image_should_unload_llm(bool resident_llm, bool image_loaded, std::optional<int> secondary, double free_gb, const std::function<bool()> &forwarding = {});
bool image_should_unload_llm(bool resident_llm, bool image_loaded,
    const std::function<std::optional<int>()> &secondary, const std::function<double()> &free_gb, const std::function<bool()> &forwarding = {});
bool image_should_unload_auxiliary(bool unload_llm, const std::vector<std::string> &model_sources);

// Order follows Pipes._unload_aux_models_for_video and its restore method.
inline constexpr std::array<const char *, 4> video_auxiliary_models{"tts", "stt", "embedding", "img"};
using VideoAuxiliaryMask = std::array<bool, 4>;
struct AuxiliaryResidency { bool loaded = false; int active_requests = 0; };
bool video_should_unload_auxiliary(bool unload_llm);
VideoAuxiliaryMask video_auxiliary_unload(const std::array<AuxiliaryResidency, 4> &models, bool unload_llm);
bool voice_should_preload(const std::string &service, const std::vector<std::string> &model_sources);
VideoAuxiliaryMask video_auxiliary_restore(const VideoAuxiliaryMask &unloaded,
    const std::vector<std::string> &model_sources, bool image_loaded);
}
