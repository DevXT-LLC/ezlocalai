#include "ezlocalai/video_runtime.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

namespace {
void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
void environment(const char *name, const char *value) { setenv(name, value, 1); }
}

int main() {
    using namespace ezlocalai;
    namespace fs = std::filesystem;
    const auto path = fs::temp_directory_path() / ("ezlocalai-hip-" + std::to_string(getpid()));
    struct Cleanup { fs::path path; ~Cleanup() { std::error_code error; fs::remove(path, error); } } cleanup{path};
    try {
        constexpr uint64_t gib = 1024ULL * 1024 * 1024;
        std::ofstream(path) << "AMD Radeon RX 7900 XTX\t11\t0\t" << 24 * gib << '\t' << 20 * gib
            << "\nAMD Instinct MI300X\t9\t4\t" << 192 * gib << '\t' << 180 * gib << '\n';
        environment("EZLOCALAI_TEST_CUDA_DEVICES", path.c_str());
        unsetenv("EZLOCALAI_TEST_CUDA_FAIL");
        // HIP owns device visibility, including its CUDA-compatible mask.
        unsetenv("CUDA_VISIBLE_DEVICES");
        unsetenv("HIP_VISIBLE_DEVICES");
        require(visible_cuda_device_count() == 2, "HIP enumeration uses its own driver and visibility");
        auto devices = visible_cuda_devices();
        require(devices.size() == 2 && devices[0].name == "AMD Radeon RX 7900 XTX" &&
            devices[1].total_memory == 192 * gib, "HIP names and capacities");
        environment("CUDA_VISIBLE_DEVICES", "-1");
        require(visible_cuda_device_count() == 0, "HIP supports the CUDA visibility alias");
        environment("HIP_VISIBLE_DEVICES", "1,0");
        devices = visible_cuda_devices();
        require(devices[0].ordinal == 0 && devices[0].name == "AMD Instinct MI300X" &&
            devices[1].ordinal == 1 && devices[1].total_memory == 24 * gib, "HIP logical ordinal remapping");
        require(cuda_compute_capability(0) == std::make_pair(9, 4), "HIP compute properties");
        require(cuda_multiprocessor_geometry(0) == std::make_pair(90, 1536), "HIP attribute enum values");
        require(cuda_memory_info(0).free_bytes == 180 * gib &&
            cuda_memory_info(1).free_bytes == 20 * gib, "HIP memory follows selected logical device");
        auto first = std::async(std::launch::async, [&] {
            for (int i = 0; i < 100; ++i) require(cuda_memory_info(0).free_bytes == 180 * gib, "first HIP context");
        });
        auto second = std::async(std::launch::async, [&] {
            for (int i = 0; i < 100; ++i) require(cuda_memory_info(1).free_bytes == 20 * gib, "second HIP context");
        });
        first.get(); second.get();
        environment("EZLOCALAI_TEST_CUDA_FAIL", "free");
        bool failed = false;
        try { (void)cuda_memory_info(0); } catch (const std::exception &error) {
            failed = std::string(error.what()).find("hipMemGetInfo failed with HIP error") != std::string::npos;
        }
        require(failed, "HIP errors identify the actual API");
        unsetenv("EZLOCALAI_TEST_CUDA_FAIL");
        require(cuda_memory_info(1).free_bytes == 20 * gib, "HIP memory failure restores context");
        for (const auto *mask : {"", "-1"}) {
            environment("HIP_VISIBLE_DEVICES", mask);
            require(visible_cuda_device_count() == 0 && visible_cuda_devices().empty() &&
                cuda_free_memory_gb() == 0, "hidden HIP devices are not advertised");
        }
        unsetenv("CUDA_VISIBLE_DEVICES");
        environment("HIP_VISIBLE_DEVICES", "");
        require(visible_cuda_device_count() == 2, "empty HIP and CUDA masks retain AMD runtime defaults");
        environment("HIP_VISIBLE_DEVICES", "0");
        environment("EZLOCALAI_TEST_CUDA_FAIL", "init");
        require(visible_cuda_device_count() == 0 && visible_cuda_devices().empty(), "failed HIP initialization");
        unsetenv("EZLOCALAI_TEST_CUDA_FAIL");
        environment("EZLOCALAI_TEST_CUDA_FAIL", "name");
        failed = false;
        try { (void)cuda_device_properties(0); } catch (const std::exception &) { failed = true; }
        require(failed && visible_cuda_device_count() == 1, "HIP property failure does not erase device count");
        unsetenv("EZLOCALAI_TEST_CUDA_FAIL");
        std::cout << "HIP discovery, logical devices, memory/context isolation, errors and visibility passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
