#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef EZLOCALAI_TEST_HIP
#define cuInit hipInit
#define cuDeviceGetCount hipGetDeviceCount
#define cuDeviceGet hipDeviceGet
#define cuDeviceGetName hipDeviceGetName
#define cuDeviceComputeCapability hipDeviceComputeCapability
#define cuDeviceTotalMem_v2 hipDeviceTotalMem
#define cuDeviceGetAttribute hipDeviceGetAttribute
#define cuDevicePrimaryCtxRetain hipDevicePrimaryCtxRetain
#define cuDevicePrimaryCtxRelease_v2 hipDevicePrimaryCtxRelease
#define cuCtxPushCurrent_v2 hipCtxPushCurrent
#define cuCtxPopCurrent_v2 hipCtxPopCurrent
#define cuMemGetInfo_v2 hipMemGetInfo
#endif

namespace {
struct Device { std::string name; int major, minor; size_t memory, free; };
std::vector<Device> devices() {
    std::vector<Device> all;
    if (const char *path = std::getenv("EZLOCALAI_TEST_CUDA_DEVICES")) {
        std::ifstream input(path);
        for (std::string line; std::getline(input, line);) {
            std::istringstream row(line); Device device{};
            std::getline(row, device.name, '\t'); row >> device.major >> device.minor >> device.memory;
            if (!(row >> device.free)) device.free = device.memory;
            all.push_back(device);
        }
    }
#ifdef EZLOCALAI_TEST_HIP
    const char *visible = std::getenv("HIP_VISIBLE_DEVICES");
    // ROCm 6.4.1 rocdevice.cpp uses the nonempty HIP mask, otherwise CUDA.
    if (!visible || !*visible) visible = std::getenv("CUDA_VISIBLE_DEVICES");
    if (!visible || !*visible) return all;
#else
    const char *visible = std::getenv("CUDA_VISIBLE_DEVICES");
#endif
    if (!visible) return all;
    std::vector<Device> result; std::istringstream mask(visible);
    for (std::string item; std::getline(mask, item, ',');) {
        int ordinal = -1; try { ordinal = std::stoi(item); } catch (...) { break; }
        if (ordinal < 0 || static_cast<size_t>(ordinal) >= all.size()) break;
        result.push_back(all[ordinal]);
    }
    return result;
}
bool fail(const char *operation) { const auto *value = std::getenv("EZLOCALAI_TEST_CUDA_FAIL"); return value && std::string(value) == operation; }
thread_local std::vector<void *> contexts;
}
extern "C" int cuInit(unsigned flags) { return flags || fail("init") ? 100 : 0; }
extern "C" int cuDeviceGetCount(int *count) { if (fail("count")) return 3; *count = devices().size(); return 0; }
extern "C" int cuDeviceGet(int *device, int ordinal) {
    if (fail("get") || ordinal < 0 || static_cast<size_t>(ordinal) >= devices().size()) return 101;
    *device = ordinal + 100; return 0;
}
extern "C" int cuDeviceGetName(char *name, int size, int device) {
    if (fail("name")) return 999;
    const auto value = devices().at(device - 100).name;
    const auto length = std::min(static_cast<size_t>(size - 1), value.size()); std::memcpy(name, value.data(), length); name[length] = 0; return 0;
}
extern "C" int cuDeviceComputeCapability(int *major, int *minor, int device) {
    if (fail("compute")) return 999;
    const auto value = devices().at(device - 100); *major = value.major; *minor = value.minor; return 0;
}
extern "C" int cuDeviceTotalMem_v2(size_t *memory, int device) {
    if (fail("memory")) return 999;
    *memory = devices().at(device - 100).memory; return 0;
}
extern "C" int cuDeviceGetAttribute(int *value, int attribute, int device) {
    if (fail("attribute")) return 999;
    const auto gpu = devices().at(device - 100);
#ifdef EZLOCALAI_TEST_HIP
    if (attribute == 63) *value = gpu.major * 10;
    else if (attribute == 57) *value = 1536;
#else
    if (attribute == 16) *value = gpu.major * 10;
    else if (attribute == 39) *value = 1536;
#endif
    else return 1;
    return 0;
}
extern "C" int cuDevicePrimaryCtxRetain(void **context, int device) {
    if (fail("retain")) return 999;
    *context = reinterpret_cast<void *>(static_cast<intptr_t>(device)); return 0;
}
extern "C" int cuDevicePrimaryCtxRelease_v2(int) { return fail("release") ? 999 : 0; }
extern "C" int cuCtxPushCurrent_v2(void *context) {
    if (fail("push")) return 999;
    contexts.push_back(context); return 0;
}
extern "C" int cuCtxPopCurrent_v2(void **context) {
    if (fail("pop") || contexts.empty()) return 999;
    *context = contexts.back(); contexts.pop_back(); return 0;
}
extern "C" int cuMemGetInfo_v2(size_t *free, size_t *total) {
    if (fail("free") || contexts.empty()) return 999;
    const auto value = devices().at(reinterpret_cast<intptr_t>(contexts.back()) - 100);
    *free = value.free; *total = value.memory; return 0;
}
