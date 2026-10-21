#include "ezlocalai/process.hpp"
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Architecture {
    int number;
    std::string suffix;
    bool operator<(const Architecture &other) const {
        return number == other.number ? suffix < other.suffix : number < other.number;
    }
};

void check(bool value, const std::string &message) {
    if (!value) throw std::runtime_error(message);
}

// NVIDIA's binary compatibility rules: ordinary cubins stay within a major
// version; ordinary PTX can JIT forward. Architecture-specific "a" code is
// exact-target only; family-specific "f" code stays within its major version.
// https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/
bool compatible(const Architecture &code, int device, bool ptx) {
    if (code.suffix == "a") return code.number == device;
    if (code.number > device) return false;
    if (code.suffix == "f" || !ptx) return code.number / 10 == device / 10;
    return true;
}

std::set<Architecture> architectures(const std::string &output) {
    static const std::regex pattern(R"(\.sm_([0-9]+)([af]?)\.)");
    std::set<Architecture> result;
    for (std::sregex_iterator it(output.begin(), output.end(), pattern), end; it != end; ++it)
        result.insert({std::stoi((*it)[1]), (*it)[2]});
    return result;
}

std::string describe(const std::set<Architecture> &codes) {
    std::string result;
    for (const auto &code : codes) {
        if (!result.empty()) result += ",";
        result += "sm_" + std::to_string(code.number) + code.suffix;
    }
    return result.empty() ? "none" : result;
}

void elf(const std::string &path, int expected_machine) {
    std::ifstream file(path, std::ios::binary);
    std::array<unsigned char, 20> header{};
    file.read(reinterpret_cast<char *>(header.data()), header.size());
    check(file.gcount() == static_cast<std::streamsize>(header.size()) &&
              header[0] == 0x7f && header[1] == 'E' && header[2] == 'L' && header[3] == 'F' &&
              header[4] == 2 && header[5] == 1,
          path + ": expected a little-endian ELF64 binary");
    check((header[18] | header[19] << 8) == expected_machine,
          path + ": wrong host architecture for package profile");
}

void inspect(const std::string &tool, const std::string &path, const std::vector<int> &devices,
             int machine, bool require_native) {
    elf(path, machine);
    const auto cubins = ezlocalai::run_process({tool, "--list-elf", path}, true, true);
    check(cubins.status == 0, path + ": could not inspect CUDA cubins");
    const auto native = architectures(cubins.output);
    std::set<Architecture> ptx;
    if (!require_native) {
        const auto intermediate = ezlocalai::run_process({tool, "--list-ptx", path}, true, true);
        check(intermediate.status == 0, path + ": could not inspect CUDA PTX");
        ptx = architectures(intermediate.output);
    }
    check(!native.empty() || !ptx.empty(), path + ": no CUDA kernels found");
    for (int device : devices) {
        const auto covers = [device](const auto &codes, bool is_ptx) {
            return std::any_of(codes.begin(), codes.end(), [=](const auto &code) {
                return compatible(code, device, is_ptx);
            });
        };
        const bool native_target = require_native
            ? std::any_of(native.begin(), native.end(), [device](const auto &code) {
                  return code.number == device;
              })
            : covers(native, false);
        check(native_target || (!require_native && covers(ptx, true)),
              path + ": no target for sm_" + std::to_string(device) + " (cubins=" +
                  describe(native) + "; PTX=" + describe(ptx) + ")");
    }
    std::cout << path << ": cubins=" << describe(native) << "; PTX=" << describe(ptx) << '\n';
}

void policy_cases() {
    const std::string sample = "ELF file 1: lib.1.sm_86.cubin\nELF file 2: lib.2.sm_120a.cubin\n";
    check(architectures(sample).size() == 2, "cuobjdump inventory parsing");
    check(compatible({86, ""}, 89, false), "Ampere cubin forward minor compatibility");
    check(!compatible({86, ""}, 120, false), "Ampere cubin must not satisfy Blackwell");
    check(compatible({86, ""}, 120, true), "ordinary PTX can JIT across major versions");
    check(!compatible({120, "a"}, 121, false) && !compatible({120, "a"}, 121, true),
          "sm_120a must not satisfy GB10, including PTX");
    check(compatible({120, ""}, 121, false) && compatible({120, "f"}, 121, true),
          "generic and family code can satisfy higher minor versions");
    check(!compatible({121, ""}, 120, true) && !compatible({90, "f"}, 120, true),
          "newer PTX and unrelated family PTX must not satisfy older/other devices");
}
} // namespace

int main(int argc, char **argv) {
    try {
        policy_cases();
        if (argc == 2 && std::string(argv[1]) == "--self-test") return 0;
        check(argc == 10, "usage: test_cuda_architectures PROFILE CUOBJDUMP LLAMA SD MUSIC CT2 TORCH WORKER TTS");
        const std::string profile = argv[1];
        check(profile == "cuda" || profile == "gb10" || profile == "jetson", "unknown CUDA package profile");
        const bool gb10 = profile == "gb10";
        const bool jetson = profile == "jetson";
        const int machine = gb10 || jetson ? 183 : 62; // EM_AARCH64 / EM_X86_64.
        const std::vector<int> devices = jetson ? std::vector<int>{87} :
            gb10 ? std::vector<int>{121} : std::vector<int>{86, 89, 120};
        // Own engines must contain every explicitly configured native target.
        // Pinned third-party libraries may use NVIDIA's cubin/PTX compatibility.
        // Tegra does not inherit desktop cubin compatibility. The Jetson image
        // builds every CUDA library for Orin, including LibTorch/CTranslate2.
        for (int i = 3; i <= 7; ++i) inspect(argv[2], argv[i], devices, machine, jetson || i <= 5);
        elf(argv[8], machine);
        elf(argv[9], machine);
        std::cout << profile << " package architecture inventory passed; hardware execution remains a separate gate\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
