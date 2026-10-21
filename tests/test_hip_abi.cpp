#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include <iostream>
#include <type_traits>

// Keep the dynamically loaded adapter checked against the packaged SDK.
static_assert(std::is_same_v<hipDevice_t, int>);
static_assert(sizeof(hipError_t) == sizeof(int));
static_assert(std::is_pointer_v<hipCtx_t> && sizeof(hipCtx_t) == sizeof(void *));
static_assert(hipDeviceAttributeMultiprocessorCount == 63);
static_assert(hipDeviceAttributeMaxThreadsPerMultiProcessor == 57);
#define CHECK_API(name, ...) static_assert(std::is_same_v<decltype(&name), hipError_t (*)(__VA_ARGS__)>)
CHECK_API(hipInit, unsigned int);
CHECK_API(hipGetDeviceCount, int *);
CHECK_API(hipDeviceGet, hipDevice_t *, int);
CHECK_API(hipDeviceGetName, char *, int, hipDevice_t);
CHECK_API(hipDeviceComputeCapability, int *, int *, hipDevice_t);
CHECK_API(hipDeviceGetAttribute, int *, hipDeviceAttribute_t, int);
CHECK_API(hipDeviceTotalMem, size_t *, hipDevice_t);
CHECK_API(hipDevicePrimaryCtxRetain, hipCtx_t *, hipDevice_t);
CHECK_API(hipDevicePrimaryCtxRelease, hipDevice_t);
CHECK_API(hipCtxPushCurrent, hipCtx_t);
CHECK_API(hipCtxPopCurrent, hipCtx_t *);
CHECK_API(hipMemGetInfo, size_t *, size_t *);
#undef CHECK_API

int main() {
    auto *library = dlopen("libamdhip64.so", RTLD_NOW | RTLD_LOCAL);
    if (!library) { std::cerr << dlerror() << '\n'; return 1; }
    const char *symbols[] = {"hipInit", "hipGetDeviceCount", "hipDeviceGet", "hipDeviceGetName",
        "hipDeviceComputeCapability", "hipDeviceGetAttribute", "hipDeviceTotalMem",
        "hipDevicePrimaryCtxRetain", "hipDevicePrimaryCtxRelease", "hipCtxPushCurrent",
        "hipCtxPopCurrent", "hipMemGetInfo"};
    for (const auto *symbol : symbols) {
        if (!dlsym(library, symbol)) {
            std::cerr << "Missing packaged HIP symbol: " << symbol << '\n';
            dlclose(library); return 1;
        }
    }
    dlclose(library);
    std::cout << "HIP headers, attribute values and all 12 runtime symbols match the device adapter\n";
}
