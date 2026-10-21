#include "ezlocalai/resource_manager.hpp"
#include "ezlocalai/video_runtime.hpp"
#include "ezlocalai/ezlocalai_c.h"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <locale>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace ezlocalai {
namespace {
constexpr double gib = 1073741824.0;
std::string fixed(double value, int digits) {
    std::ostringstream out; out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(digits) << value; return out.str();
}
std::string number(double value) {
    if (std::isnan(value)) return "nan";
    if (std::isinf(value)) return value < 0 ? "-inf" : "inf";
    char buffer[128]; const auto magnitude = std::abs(value);
    const auto result = magnitude == 0 || (magnitude >= 1e-4 && magnitude < 1e16)
        ? std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::fixed)
        : std::to_chars(buffer, buffer + sizeof(buffer), value);
    if (result.ec != std::errc()) throw std::runtime_error("Cannot format resource estimate");
    std::string text(buffer, result.ptr);
    if (text.find_first_of(".eE") == std::string::npos) text += ".0";
    return text;
}
double rounded(double value) { return std::stod(fixed(value, 3)); }
std::vector<double> free_per_gpu() {
    std::vector<double> values;
    for (int i = 0, count = visible_cuda_device_count(); i < count; ++i) values.push_back(cuda_memory_info(i).free_bytes / gib);
    return values;
}
}
const char *model_type_name(ModelType type) {
    switch (type) {
        case ModelType::Llm: return "llm"; case ModelType::VisionLlm: return "vision_llm";
        case ModelType::Tts: return "tts"; case ModelType::Stt: return "stt";
        case ModelType::Image: return "img"; case ModelType::Video: return "video";
        case ModelType::Music: return "music"; case ModelType::Embedding: return "embedding";
    }
    throw std::invalid_argument("Unknown model type");
}
double model_vram_estimate(ModelType type) {
    switch (type) {
        case ModelType::Llm: return 8; case ModelType::VisionLlm: return 6;
        case ModelType::Tts: return 4; case ModelType::Stt: return 2;
        case ModelType::Image: return 6; case ModelType::Video: return 12;
        case ModelType::Music: return 0; case ModelType::Embedding: return 2;
    }
    return 2;
}
std::vector<CudaAllocatorStats> collect_cuda_allocator_stats(const CudaAllocatorProbes &probes) {
    std::vector<CudaAllocatorStats> result;
    if (!probes.available()) return result;
    for (int i = 0, count = probes.device_count(); i < count; ++i) {
        CudaAllocatorStats stats; stats.index = i;
        try {
            const auto [free, total] = probes.memory_info(i);
            stats.allocated_gb = rounded(probes.allocated_bytes(i) / gib);
            stats.reserved_gb = rounded(probes.reserved_bytes(i) / gib);
            stats.max_allocated_gb = rounded(probes.max_allocated_bytes(i) / gib);
            stats.free_gb = rounded(free / gib); stats.total_gb = rounded(total / gib);
        } catch (...) { stats = {}; stats.index = i; stats.error = "CUDA memory statistics unavailable"; }
        result.push_back(std::move(stats));
    }
    return result;
}
ResourceProbes native_resource_probes() {
    ResourceProbes probes;
    probes.gpu_count = visible_cuda_device_count;
    probes.total_per_gpu = [] {
        std::vector<double> values;
        for (const auto &device : visible_cuda_devices()) values.push_back(std::floor(device.total_memory / gib));
        return values;
    };
    probes.free_per_gpu = free_per_gpu;
    probes.free_vram = [](std::optional<int> ordinal) {
        const auto count = visible_cuda_device_count(); if (!count) return 0.0;
        if (ordinal) return *ordinal < count ? cuda_memory_info(*ordinal).free_bytes / gib : 0.0;
        const auto values = free_per_gpu(); return std::accumulate(values.begin(), values.end(), 0.0);
    };
    probes.primary_gpu = [] { const auto ranked = rank_cuda_devices(visible_cuda_devices()); return ranked.empty() ? 0 : ranked.front().device.ordinal; };
    probes.secondary_gpu = [] { return secondary_cuda_device(visible_cuda_devices()); };
    probes.allocator_stats = [] {
        // Builds without LibTorch have no in-process caching allocator.
        return collect_cuda_allocator_stats({
            [] { return visible_cuda_device_count() > 0; }, visible_cuda_device_count,
            [](int i) { const auto memory = cuda_memory_info(i); return std::pair<double, double>{memory.free_bytes, memory.total_bytes}; },
            [](int) { return 0.0; }, [](int) { return 0.0; }, [](int) { return 0.0; }
        });
    };
    probes.system_ram = [] {
        std::ifstream in("/proc/meminfo"); std::string key, line; double kib = 0;
        while (in >> key) { if (key == "MemTotal:" && in >> kib) return kib * 1024 / gib; std::getline(in, line); }
        throw std::runtime_error("Cannot read total system memory");
    };
    probes.available_ram = [] {
        std::ifstream in("/proc/meminfo"); std::string key, line; double kib = 0;
        while (in >> key) { if (key == "MemAvailable:" && in >> kib) return kib * 1024 / gib; std::getline(in, line); }
        return 0.0;
    };
    probes.now = [] { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); };
    probes.fallback_server = [] { return std::string(ezl_getenv_default("FALLBACK_SERVER")); };
    return probes;
}
ResourceManager::ResourceManager(ResourceProbes probes) : probes_(std::move(probes)), gpu_count_(probes_.gpu_count()) {
    per_gpu_capacity_ = probes_.total_per_gpu(); total_vram_ = std::accumulate(per_gpu_capacity_.begin(), per_gpu_capacity_.end(), 0.0);
    system_ram_ = probes_.system_ram();
}
bool ResourceManager::is_model_loaded(ModelType type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::any_of(models_.begin(), models_.end(), [type](const auto &m) { return m.type == type; });
}
bool ResourceManager::is_model_in_use(ModelType type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &m : models_) if (m.type == type) return m.in_use; return false;
}
uint64_t ResourceManager::get_model_active_count(ModelType type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &m : models_) if (m.type == type) return m.active_requests; return 0;
}
void ResourceManager::mark_model_in_use(ModelType type, bool in_use) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &m : models_) if (m.type == type) {
        if (in_use) { ++m.active_requests; m.last_used = probes_.now(); }
        else if (m.active_requests) --m.active_requests;
        m.in_use = m.active_requests > 0; return;
    }
}
void ResourceManager::register_model(ModelType type, std::string name, std::string device, std::optional<double> vram) {
    std::lock_guard<std::mutex> lock(mutex_);
    ModelResource resource{type, std::move(name), std::move(device)};
    resource.vram_gb = resource.device.find("cuda") != std::string::npos ? vram.value_or(model_vram_estimate(type)) : 0;
    const auto found = std::find_if(models_.begin(), models_.end(), [type](const auto &m) { return m.type == type; });
    if (found != models_.end()) { resource.in_use = found->in_use; resource.active_requests = found->active_requests; }
    resource.last_used = probes_.now();
    if (found != models_.end()) *found = std::move(resource); else models_.push_back(std::move(resource));
}
void ResourceManager::unregister_model(ModelType type) {
    std::lock_guard<std::mutex> lock(mutex_);
    models_.erase(std::remove_if(models_.begin(), models_.end(), [type](const auto &m) { return m.type == type; }), models_.end());
}
double ResourceManager::get_free_vram(std::optional<int> ordinal) const { return probes_.free_vram(ordinal); }
double ResourceManager::get_free_ram() const { return probes_.available_ram(); }
double ResourceManager::current_time() const { return probes_.now(); }
double ResourceManager::get_total_free_vram() const {
    if (!gpu_count_) return 0;
    const auto free = probes_.free_per_gpu(); return std::accumulate(free.begin(), free.end(), 0.0);
}
std::vector<double> ResourceManager::get_per_gpu_free_vram() const {
    return gpu_count_ ? probes_.free_per_gpu() : std::vector<double>{};
}
ModelAdmission ResourceManager::can_load_model(ModelType type, std::optional<double> required_vram) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto required = required_vram.value_or(model_vram_estimate(type)), free = get_total_free_vram();
    if (free >= required + 1.0) {
        if (gpu_count_ == 1) return {true, "cuda", "Sufficient VRAM (" + fixed(free, 1) + "GB free)"};
        const auto per_gpu = probes_.free_per_gpu();
        probes_.primary_gpu(); const auto secondary = probes_.secondary_gpu();
        if (type != ModelType::Llm && secondary) {
            const auto index = *secondary < 0 ? static_cast<int>(per_gpu.size()) + *secondary : *secondary;
            const auto available = per_gpu.at(index);
            if (available >= required) return {true, "cuda:" + std::to_string(*secondary), "Secondary GPU " + std::to_string(*secondary) + " has " + fixed(available, 1) + "GB free (primary reserved for LLM)"};
        }
        if (per_gpu.empty()) throw std::runtime_error("max() arg is an empty sequence");
        const auto best = std::max_element(per_gpu.begin(), per_gpu.end()); const auto ordinal = std::to_string(best - per_gpu.begin());
        if (*best >= required) return {true, "cuda:" + ordinal, "GPU " + ordinal + " has " + fixed(*best, 1) + "GB free"};
        return {true, "cuda", "Using tensor split (" + fixed(free, 1) + "GB total free)"};
    }
    double idle = 0; std::string names;
    for (const auto &m : models_) if (!m.in_use && m.type != ModelType::Llm) {
        idle += m.vram_gb; if (!names.empty()) names += ", "; names += "'" + std::string(model_type_name(m.type)) + "'";
    }
    if (free + idle >= required + 1.0) return {true, "cuda", "Can free " + fixed(idle, 1) + "GB by unloading [" + names + "]"};
    if (type == ModelType::Tts || type == ModelType::Stt || type == ModelType::Embedding) return {true, "cpu", "Insufficient VRAM, using CPU (model supports CPU)"};
    if (type == ModelType::Image) return {true, "cuda", "Using sequential CPU offload for image generation"};
    if (type == ModelType::Video) return {true, "cuda", "Using sequential CPU offload for video generation"};
    if (!probes_.fallback_server().empty()) return {false, "fallback", "Insufficient VRAM (" + fixed(free, 1) + "GB), will use fallback server"};
    return {false, "none", "Insufficient VRAM (" + fixed(free, 1) + "GB free, need " + number(required) + "GB)"};
}
std::vector<ModelType> ResourceManager::get_models_to_unload(double required, const std::vector<ModelType> &exclude) const {
    std::lock_guard<std::mutex> lock(mutex_); std::vector<const ModelResource *> candidates;
    for (const auto &m : models_) if (m.type != ModelType::Llm && !m.in_use && m.vram_gb > 0 && std::find(exclude.begin(), exclude.end(), m.type) == exclude.end()) candidates.push_back(&m);
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto *a, const auto *b) { return a->last_used < b->last_used; });
    double freed = 0; std::vector<ModelType> result;
    for (const auto *m : candidates) { if (freed >= required) break; result.push_back(m->type); freed += m->vram_gb; }
    return result;
}
std::vector<ModelResource> ResourceManager::loaded_models() const { std::lock_guard<std::mutex> lock(mutex_); return models_; }
ResourceStatus ResourceManager::get_status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    ResourceStatus status; status.gpu_count = gpu_count_; status.total_vram_gb = total_vram_; status.system_ram_gb = system_ram_;
    status.free_vram_gb = get_total_free_vram(); status.free_per_gpu_gb = probes_.free_per_gpu();
    status.torch_cuda = probes_.allocator_stats(); status.loaded_models = models_; return status;
}
}
