#include "ezlocalai/video_runtime.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/ezlocalai_c.h"

#include <algorithm>
#include <array>
#include <dlfcn.h>
#include <memory>
#include <map>
#include <mutex>
#include <optional>
#include <regex>
#include <stdexcept>

namespace ezlocalai {
namespace {
std::string lower(std::string value) {
    for (char &c : value) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    return value;
}
bool on(const std::string &value) { return value == "true" || value == "1" || value == "yes" || value == "on"; }
bool off(const std::string &value) { return value == "false" || value == "0" || value == "no" || value == "off"; }
std::string setting(const char *name, const char *fallback = "") {
    const auto raw = ezl_getenv_default(name);
    return lower(strip_config_value(*raw ? raw : fallback));
}
bool media_keep_loaded() { return on(setting("MEDIA_KEEP_MODELS_LOADED", "false")); }
bool dedicated(const char *name) { return lower(ezl_getenv_default(name)) == "true"; }
bool delegated(const char *name) {
    std::string value = ezl_getenv_default(name);
    if (lower(value) == "true") return false;
    while (!value.empty() && value.back() == '/') value.pop_back();
    return !value.empty();
}
bool fast_voice_profile(const std::vector<std::string> &sources) {
    static const std::regex release(R"(minicpm5-2b(?:-gguf)?(?:[-:](?:q\d[\w]*|f16|bf16))?(?:\.gguf)?)", std::regex::icase);
    return !sources.empty() && std::all_of(sources.begin(), sources.end(), [&](std::string name) {
        name = strip_config_value(name);
        const auto slash = name.rfind('/'); if (slash != std::string::npos) name.erase(0, slash + 1);
        const auto replica = name.find('#'); if (replica != std::string::npos) name.resize(replica);
        return std::regex_match(name, release);
    });
}
struct Driver {
    using Result = int;
#ifdef EZLOCALAI_USE_HIP
    void *library = dlopen("libamdhip64.so", RTLD_NOW | RTLD_LOCAL);
#else
    void *library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
    Result (*init)(unsigned) = nullptr;
    Result (*count)(int *) = nullptr;
    Result (*device)(int *, int) = nullptr;
    Result (*name)(char *, int, int) = nullptr;
    Result (*compute)(int *, int *, int) = nullptr;
    Result (*attribute)(int *, int, int) = nullptr;
    Result (*memory)(size_t *, int) = nullptr;
    Result (*retain)(void **, int) = nullptr;
    Result (*release)(int) = nullptr;
    Result (*push)(void *) = nullptr;
    Result (*pop)(void **) = nullptr;
    Result (*free_memory)(size_t *, size_t *) = nullptr;
    Result (*pci_bus_id)(char *, int, int) = nullptr;
    std::mutex context_mutex;
    std::map<int, void *> primary_contexts;
    Driver() {
        if (!library) return;
#ifdef EZLOCALAI_USE_HIP
        // ROCm 6.4.1 hip_runtime_api.h: these driver-compatible entry points
        // use the same scalar/opaque-context ABI, but different symbol names.
        init = reinterpret_cast<decltype(init)>(dlsym(library, "hipInit"));
        count = reinterpret_cast<decltype(count)>(dlsym(library, "hipGetDeviceCount"));
        device = reinterpret_cast<decltype(device)>(dlsym(library, "hipDeviceGet"));
        name = reinterpret_cast<decltype(name)>(dlsym(library, "hipDeviceGetName"));
        compute = reinterpret_cast<decltype(compute)>(dlsym(library, "hipDeviceComputeCapability"));
        attribute = reinterpret_cast<decltype(attribute)>(dlsym(library, "hipDeviceGetAttribute"));
        memory = reinterpret_cast<decltype(memory)>(dlsym(library, "hipDeviceTotalMem"));
        retain = reinterpret_cast<decltype(retain)>(dlsym(library, "hipDevicePrimaryCtxRetain"));
        release = reinterpret_cast<decltype(release)>(dlsym(library, "hipDevicePrimaryCtxRelease"));
        push = reinterpret_cast<decltype(push)>(dlsym(library, "hipCtxPushCurrent"));
        pop = reinterpret_cast<decltype(pop)>(dlsym(library, "hipCtxPopCurrent"));
        free_memory = reinterpret_cast<decltype(free_memory)>(dlsym(library, "hipMemGetInfo"));
#else
        // These stable driver entry points avoid a CUDA toolkit build/link
        // dependency. Signatures follow cuda.h's CUDA Driver API declarations.
        init = reinterpret_cast<decltype(init)>(dlsym(library, "cuInit"));
        count = reinterpret_cast<decltype(count)>(dlsym(library, "cuDeviceGetCount"));
        device = reinterpret_cast<decltype(device)>(dlsym(library, "cuDeviceGet"));
        name = reinterpret_cast<decltype(name)>(dlsym(library, "cuDeviceGetName"));
        compute = reinterpret_cast<decltype(compute)>(dlsym(library, "cuDeviceComputeCapability"));
        attribute = reinterpret_cast<decltype(attribute)>(dlsym(library, "cuDeviceGetAttribute"));
        memory = reinterpret_cast<decltype(memory)>(dlsym(library, "cuDeviceTotalMem_v2"));
        retain = reinterpret_cast<decltype(retain)>(dlsym(library, "cuDevicePrimaryCtxRetain"));
        release = reinterpret_cast<decltype(release)>(dlsym(library, "cuDevicePrimaryCtxRelease_v2"));
        push = reinterpret_cast<decltype(push)>(dlsym(library, "cuCtxPushCurrent_v2"));
        pop = reinterpret_cast<decltype(pop)>(dlsym(library, "cuCtxPopCurrent_v2"));
        free_memory = reinterpret_cast<decltype(free_memory)>(dlsym(library, "cuMemGetInfo_v2"));
        pci_bus_id = reinterpret_cast<decltype(pci_bus_id)>(dlsym(library, "cuDeviceGetPCIBusId"));
#endif
    }
    ~Driver() {
        for (const auto &entry : primary_contexts) release(entry.first);
        if (library) dlclose(library);
    }
    bool available() const { return init && count && device && name && compute && memory; }
};
Driver &cuda_driver() { static Driver driver; return driver; }
#ifndef EZLOCALAI_USE_HIP
// NVML reports device memory without a CUDA context. cuMemGetInfo needs a
// retained primary context, which costs this process ~260 MiB of VRAM per GPU
// that Python's in-process engine does not pay separately.
struct Nvml {
    struct MemoryV2 { unsigned version; unsigned long long total, reserved, free, used; };
    struct MemoryV1 { unsigned long long total, free, used; };
    using Result = int;
    void *library = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    Result (*by_bus)(const char *, void **) = nullptr;
    Result (*memory_v2)(void *, MemoryV2 *) = nullptr;
    Result (*memory_v1)(void *, MemoryV1 *) = nullptr;
    bool ready = false;
    Nvml() {
        if (!library) return;
        const auto init = reinterpret_cast<Result (*)()>(dlsym(library, "nvmlInit_v2"));
        by_bus = reinterpret_cast<decltype(by_bus)>(dlsym(library, "nvmlDeviceGetHandleByPciBusId_v2"));
        memory_v2 = reinterpret_cast<decltype(memory_v2)>(dlsym(library, "nvmlDeviceGetMemoryInfo_v2"));
        memory_v1 = reinterpret_cast<decltype(memory_v1)>(dlsym(library, "nvmlDeviceGetMemoryInfo"));
        ready = init && by_bus && (memory_v2 || memory_v1) && init() == 0;
    }
    // Free bytes for a PCI bus ID, matching cuMemGetInfo: total less driver
    // reserve and every process's allocations. GB10 unified memory and other
    // unsupported devices return nothing, keeping the CUDA probe.
    std::optional<size_t> free_memory(const char *bus) const {
        void *handle = nullptr;
        if (!ready || by_bus(bus, &handle) != 0) return std::nullopt;
        if (memory_v2) {
            MemoryV2 value{static_cast<unsigned>(sizeof(MemoryV2) | (2u << 24)), 0, 0, 0, 0};
            if (memory_v2(handle, &value) == 0 && value.total) return static_cast<size_t>(value.free);
        }
        MemoryV1 value{};
        if (memory_v1 && memory_v1(handle, &value) == 0 && value.total) return static_cast<size_t>(value.free);
        return std::nullopt;
    }
};
const Nvml &nvml() { static const Nvml library; return library; }
#endif
void checked(int result, const char *operation) {
#ifdef EZLOCALAI_USE_HIP
    if (result) {
        std::string name(operation);
        if (name.compare(0, 2, "cu") == 0) name.replace(0, 2, "hip");
        if (name.size() > 3 && name.compare(name.size() - 3, 3, "_v2") == 0) name.resize(name.size() - 3);
        throw std::runtime_error(name + " failed with HIP error " + std::to_string(result));
    }
#else
    if (result) throw std::runtime_error(std::string(operation) + " failed with CUDA error " + std::to_string(result));
#endif
}
}
std::vector<CudaDevice> visible_cuda_devices() {
    const auto &driver = cuda_driver();
    int count = 0;
    if (!driver.available() || driver.init(0) || driver.count(&count) || count <= 0) return {};
    std::vector<CudaDevice> result;
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        int handle = 0; std::array<char, 256> name{}; size_t memory = 0;
        CudaDevice value; value.ordinal = ordinal;
        checked(driver.device(&handle, ordinal), "cuDeviceGet");
        checked(driver.name(name.data(), name.size(), handle), "cuDeviceGetName");
        checked(driver.compute(&value.major, &value.minor, handle), "cuDeviceComputeCapability");
        checked(driver.memory(&memory, handle), "cuDeviceTotalMem_v2");
        value.name = name.data(); value.total_memory = memory;
        result.push_back(std::move(value));
    }
    return result;
}
int visible_cuda_device_count() {
    const auto &driver = cuda_driver(); int count = 0;
    return driver.init && driver.count && !driver.init(0) && !driver.count(&count) ? std::max(0, count) : 0;
}
std::string llama_gpu_device_name(const std::string &logical_ordinal) {
#ifdef EZLOCALAI_USE_HIP
    return "ROCm" + logical_ordinal;
#else
    return "CUDA" + logical_ordinal;
#endif
}
std::string gpu_backend_name() {
#ifdef EZLOCALAI_USE_HIP
    return "rocm";
#else
    return "cuda";
#endif
}
std::pair<int, int> cuda_compute_capability(int ordinal) {
    // torch.cuda.get_device_capability reads the complete properties object.
    const auto properties = cuda_device_properties(ordinal);
    return {properties.major, properties.minor};
}
CudaDevice cuda_device_properties(int ordinal) {
    const auto &driver = cuda_driver();
    if (!driver.init || !driver.device || !driver.name || !driver.compute || !driver.memory) throw std::runtime_error("CUDA properties unavailable");
    checked(driver.init(0), "cuInit");
    int handle, major, minor; std::array<char, 256> name{}; size_t memory;
    checked(driver.device(&handle, ordinal), "cuDeviceGet");
    checked(driver.name(name.data(), name.size(), handle), "cuDeviceGetName");
    checked(driver.compute(&major, &minor, handle), "cuDeviceComputeCapability");
    checked(driver.memory(&memory, handle), "cuDeviceTotalMem_v2");
    return {ordinal, name.data(), major, minor, memory};
}
CudaMemoryInfo cuda_memory_info(int ordinal) {
    auto &driver = cuda_driver();
    const auto count = visible_cuda_device_count();
    if (!count || !driver.device || !driver.retain || !driver.release || !driver.push || !driver.pop || !driver.free_memory) throw std::runtime_error("CUDA memory probe unavailable");
    if (ordinal < 0 || ordinal >= count) throw std::out_of_range("Invalid CUDA device ordinal");
    std::lock_guard<std::mutex> lock(driver.context_mutex);
    int device = 0; checked(driver.device(&device, ordinal), "cuDeviceGet");
#ifndef EZLOCALAI_USE_HIP
    if (driver.pci_bus_id && driver.memory) {
        char bus[32] = {};
        size_t total = 0;
        if (driver.pci_bus_id(bus, sizeof(bus), device) == 0 && driver.memory(&total, device) == 0)
            if (const auto free = nvml().free_memory(bus)) return {*free, total};
    }
#endif
    auto found = driver.primary_contexts.find(device);
    if (found == driver.primary_contexts.end()) {
        void *context = nullptr; checked(driver.retain(&context, device), "cuDevicePrimaryCtxRetain");
        try { found = driver.primary_contexts.emplace(device, context).first; }
        catch (...) { driver.release(device); throw; }
    }
    checked(driver.push(found->second), "cuCtxPushCurrent");
    size_t free = 0, total = 0; const auto status = driver.free_memory(&free, &total);
    void *previous = nullptr; const auto restored = driver.pop(&previous);
    checked(status, "cuMemGetInfo"); checked(restored, "cuCtxPopCurrent");
    return {free, total};
}
std::pair<int, int> cuda_multiprocessor_geometry(int ordinal) {
    const auto& driver = cuda_driver();
    if (!driver.init || !driver.device || !driver.attribute)
        throw std::runtime_error("CUDA device attributes unavailable");
    checked(driver.init(0), "cuInit");
    int handle = 0, sms = 0, threads = 0;
    checked(driver.device(&handle, ordinal), "cuDeviceGet");
#ifdef EZLOCALAI_USE_HIP
    // hip_runtime_api.h: MultiprocessorCount=63, MaxThreadsPerMultiProcessor=57.
    constexpr int multiprocessor_count = 63, max_threads_per_multiprocessor = 57;
#else
    // cuda.h: MULTIPROCESSOR_COUNT=16, MAX_THREADS_PER_MULTIPROCESSOR=39.
    constexpr int multiprocessor_count = 16, max_threads_per_multiprocessor = 39;
#endif
    checked(driver.attribute(&sms, multiprocessor_count, handle), "cuDeviceGetAttribute");
    checked(driver.attribute(&threads, max_threads_per_multiprocessor, handle), "cuDeviceGetAttribute");
    if (sms <= 0 || threads < 512) throw std::runtime_error("Unsupported CUDA reduction geometry");
    return {sms, threads};
}
double cuda_free_memory_gb(int ordinal) {
    const auto count = visible_cuda_device_count();
    if (!count) return 0;
    if (ordinal < 0 || ordinal >= count) ordinal = 0;
    try { return static_cast<double>(cuda_memory_info(ordinal).free_bytes) / (1024.0 * 1024.0 * 1024.0); }
    catch (...) { return 0; }
}
std::array<int64_t, 3> gpu_generation_score(std::string name) {
    // cuDeviceGetName returns ASCII. Preserve Pipes.py's actual parsing,
    // including its first-two-digits tier slice and whole-name TI detection.
    for (char &c : name) if (c >= 'a' && c <= 'z') c -= 'a' - 'A';
    static const std::regex datacenter(R"(\b([AH])(\d{2,3})\b)"), consumer(R"(\b(?:RTX|GTX)\s*(\d)(\d{2,3})\b)"), quadro(R"(QUADRO\s*(?:RTX\s*)?(\d+))");
    std::smatch match;
    std::array<int64_t, 3> score{};
    if (std::regex_search(name, match, datacenter)) return {match[1] == "H" ? 100 : 90, std::stoll(match[2]), 0};
    if (std::regex_search(name, match, consumer)) {
        score[0] = std::stoll(match[1]) * 10;
        const auto tier = match[2].str(); score[1] = std::stoll(tier.size() == 3 ? tier.substr(0, 2) : tier);
        score[2] = name.find("TI") != std::string::npos ? 5 : name.find("SUPER") != std::string::npos ? 3 : 0;
    }
    if (std::regex_search(name, match, quadro)) {
        const auto number = std::stoll(match[1]); score[0] = 35; score[1] = number / (number >= 1000 ? 100 : 10);
    }
    return score;
}
std::vector<RankedCudaDevice> rank_cuda_devices(const std::vector<CudaDevice> &devices) {
    std::vector<RankedCudaDevice> result;
    for (const auto &device : devices) {
        const auto score = gpu_generation_score(device.name);
        result.push_back({device, static_cast<double>(score[0] * 10000 + score[1] * 100 + score[2] * 10 + device.major * 10 + device.minor)
            + static_cast<double>(device.total_memory) / (1024.0 * 1024.0 * 1024.0) / 100.0});
    }
    std::stable_sort(result.begin(), result.end(), [](const auto &a, const auto &b) { return a.score > b.score; });
    return result;
}
std::optional<int> secondary_cuda_device(const std::vector<CudaDevice> &devices) {
    const auto ranking = rank_cuda_devices(devices);
    if (ranking.size() < 2) return std::nullopt;
    return ranking[1].device.ordinal;
}
std::string video_device(std::optional<int> secondary, bool force_cpu, bool can_load, const std::string &resource_device) {
    if (force_cpu) return "cpu";
    if (secondary) return "cuda:" + std::to_string(*secondary);
    if (!can_load && resource_device == "fallback") return "cuda";
    return resource_device == "cpu" ? "cpu" : "cuda";
}
bool video_should_unload_llm(bool resident_llm, std::optional<int> secondary, const std::function<bool()> &forwarding) {
    return video_should_unload_llm(resident_llm, [secondary] { return secondary; }, forwarding);
}
bool video_should_unload_llm(bool resident_llm, const std::function<std::optional<int>()> &secondary, const std::function<bool()> &forwarding) {
    const auto mode = lower(strip_config_value(ezl_getenv_fallback("VIDEO_UNLOAD_LLM_DURING_GENERATION", "auto")));
    if (off(mode)) return false;
    if (on(mode)) return true;
    if (on(lower(strip_config_value(ezl_getenv_fallback("MEDIA_KEEP_MODELS_LOADED", "false"))))) return false;
    if ((forwarding ? forwarding() : delegated("IMAGE_SERVER")) || secondary()) return false;
    return resident_llm;
}
bool video_should_reload_llm() {
    return !off(lower(strip_config_value(ezl_getenv_fallback("VIDEO_RELOAD_LLM_AFTER_GENERATION", "true"))));
}
bool video_requires_exclusive_worker(bool resident_llm, const std::function<std::optional<int>()> &secondary, const std::function<bool()> &forwarding) {
    return !media_keep_loaded() || video_should_unload_llm(resident_llm, secondary, forwarding);
}
bool has_image_server_url() { return delegated("IMAGE_SERVER"); }
bool image_should_unload_llm(bool resident_llm, bool image_loaded, std::optional<int> secondary, double free_gb, const std::function<bool()> &forwarding) {
    return image_should_unload_llm(resident_llm, image_loaded, [secondary] { return secondary; }, [free_gb] { return free_gb; }, forwarding);
}
bool image_should_unload_llm(bool resident_llm, bool image_loaded,
    const std::function<std::optional<int>()> &secondary, const std::function<double()> &free_gb, const std::function<bool()> &forwarding) {
    auto mode = lower(strip_config_value(ezl_getenv_fallback("IMAGE_UNLOAD_LLM_DURING_GENERATION", "auto")));
    if (off(mode)) return false;
    if (on(mode)) return true;
    if (media_keep_loaded() || (forwarding ? forwarding() : delegated("IMAGE_SERVER")) || secondary() || image_loaded || !resident_llm) return false;
    double required = 6;
    if (!parse_configuration_float(ezl_getenv_fallback("IMAGE_MODEL_MIN_FREE_GB", "6"), &required)) required = 6;
    required = std::max(0.0, required);
    return free_gb() < required + 1.0;
}
bool image_should_unload_auxiliary(bool unload_llm, const std::vector<std::string> &sources) {
    return !(media_keep_loaded() && !unload_llm) && (unload_llm || !fast_voice_profile(sources));
}
bool video_should_unload_auxiliary(bool unload_llm) {
    return !media_keep_loaded() || unload_llm;
}
VideoAuxiliaryMask video_auxiliary_unload(const std::array<AuxiliaryResidency, 4> &models, bool unload_llm) {
    VideoAuxiliaryMask result{};
    if (!video_should_unload_auxiliary(unload_llm)) return result;
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = models[i].loaded && (i == 3 ? models[i].active_requests <= 0 : models[i].active_requests == 0);
    return result;
}
bool voice_should_preload(const std::string &service, const std::vector<std::string> &sources) {
    const bool fast = fast_voice_profile(sources);
    const bool preload = lower(ezl_getenv_default("LAZY_LOAD_VOICE")) == "false" || dedicated("VOICE_SERVER") || fast || media_keep_loaded();
    if (!preload) return false;
    std::string key = service; for (auto &c : key) if (c >= 'a' && c <= 'z') c -= 'a' - 'A';
    key += "_UNLOAD_LLM_DURING_GENERATION";
    // The per-service fallback is the raw generic value, before stripping.
    std::string generic = ezl_getenv_fallback("VOICE_UNLOAD_LLM_DURING_GENERATION", "auto");
    if (generic.empty()) generic = "auto";
    std::string mode = ezl_getenv_fallback(key.c_str(), generic.c_str());
    if (mode.empty()) mode = generic;
    mode = lower(strip_config_value(mode));
    if (off(mode)) return true;
    if (on(mode)) return delegated("VOICE_SERVER");
    if (media_keep_loaded() || delegated("VOICE_SERVER") || delegated("TEXT_SERVER")) return true;
    const bool text_mode = dedicated("TEXT_SERVER") || (!dedicated("IMAGE_SERVER") && !dedicated("VOICE_SERVER"));
    if (dedicated("VOICE_SERVER") && !text_mode) return true;
    return fast || sources.empty();
}
VideoAuxiliaryMask video_auxiliary_restore(const VideoAuxiliaryMask &unloaded,
    const std::vector<std::string> &sources, bool image_loaded) {
    VideoAuxiliaryMask result{};
    for (size_t i = 0; i < 2; ++i) {
        result[i] = unloaded[i] && voice_should_preload(video_auxiliary_models[i], sources)
            && setting(i == 0 ? "TTS_ENABLED" : "STT_ENABLED", "false") == "true" && !delegated("VOICE_SERVER");
    }
    result[2] = unloaded[2] && setting("EMBEDDING_ENABLED", "true") == "true" && !delegated("EMBEDDING_SERVER")
        && setting("EMBEDDING_KEEP_LOADED", "true") == "true";
    const std::string image_model = ezl_getenv_default("IMG_MODEL");
    result[3] = unloaded[3] && !image_loaded && setting("IMAGE_ENABLED", "false") == "true"
        && !image_model.empty() && lower(image_model) != "none";
    return result;
}
}
