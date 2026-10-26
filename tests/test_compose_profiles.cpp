#include "ezlocalai/process.hpp"
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: test_compose_profiles ROOT FIXTURE DOCKER");
        const std::filesystem::path root = argv[1];
        std::ifstream input(argv[2]);
        const auto fixture = nlohmann::json::parse(input);
        size_t comparisons = 0, corrections = 0;
        for (const auto &profile : fixture.at("profiles")) {
            for (const auto &scenario : profile.at("cases")) {
                // Never read the user's .env or inherit exported model/auth
                // settings. The capture uses the same empty/synthetic inputs.
                std::vector<std::string> command{"/usr/bin/env", "-i",
                    std::string("PATH=") + (std::getenv("PATH") ? std::getenv("PATH") : "/usr/bin:/bin")};
                for (const auto &[key, value] : scenario.at("input").items())
                    command.push_back(key + "=" + value.get<std::string>());
                const bool override_swap = scenario.at("name") == "overridden";
                for (const auto &key : {"LLM_PROMPT_CACHE_SWAP_MIB", "LLM_PROMPT_CACHE_SWAP_MARGIN_MIB"}) {
                    if (override_swap) command.push_back(std::string(key) + "=2048");
                    else if (scenario.at("name") == "empty") command.push_back(std::string(key) + '=');
                }
                command.insert(command.end(), {argv[3], "compose", "--env-file", "/dev/null",
                    "-f", (root / profile.at("file").get<std::string>()).string(),
                    "config", "--no-env-resolution", "--format", "json"});
                const auto response = ezlocalai::run_process(command, true);
                if (response.status != 0) throw std::runtime_error("Compose configuration failed");
                const auto service = nlohmann::json::parse(response.output).at("services").at("ezlocalai");
                const auto actual = service.at("environment");
                if (actual.at("LLM_PROMPT_CACHE_SWAP_MIB") != (override_swap ? "2048" : "0") ||
                    actual.at("LLM_PROMPT_CACHE_SWAP_MARGIN_MIB") != (override_swap ? "2048" : "1024"))
                    throw std::runtime_error("native bounded-swap settings must pass through every profile");
                comparisons += 2;
                for (const auto &[key, expected] : scenario.at("expected").items()) {
                    auto wanted = expected;
                    if (profile.contains("default_corrections") && profile.at("default_corrections").contains(key) &&
                        (!scenario.at("input").contains(key) || scenario.at("input").at(key) == "")) {
                        const auto &correction = profile.at("default_corrections").at(key);
                        if (expected != correction.at("source") || correction.at("reason").get<std::string>().empty())
                            throw std::runtime_error("default correction no longer matches the captured source");
                        wanted = correction.at("native");
                        ++corrections;
                    }
                    if (!actual.contains(key) || actual.at(key) != wanted)
                        throw std::runtime_error(profile.at("name").get<std::string>() + "/" +
                            scenario.at("name").get<std::string>() + ": setting differs: " + key);
                    ++comparisons;
                }
                const bool gb10 = profile.at("name") == "gb10";
                const bool jetson = profile.at("name") == "jetson";
                const bool rpi = profile.at("name") == "rpi";
                const bool rocm = profile.at("name") == "rocm";
                const bool cpu = profile.at("name") == "cpu" || rpi;
                if (service.at("platform") != (gb10 || jetson || rpi ? "linux/arm64" : "linux/amd64"))
                    throw std::runtime_error("wrong platform for deployment profile");
                const auto args = service.at("build").at("args");
                if (!cpu && !rocm && args.at("EZLOCALAI_CUDA_FLEET_PROFILE") != profile.at("name"))
                    throw std::runtime_error("fleet architecture gate is not enabled by default");
                if ((gb10 || jetson || rpi) && args.at("EZLOCALAI_PACKAGE_TEST_EXCLUDE") != "")
                    throw std::runtime_error("native ARM64 builds must run the full package gate");
                const nlohmann::json::json_pointer devices("/deploy/resources/reservations/devices");
                if (cpu && (actual.at("CUDA_VISIBLE_DEVICES") != "" ||
                            (service.contains(devices) && !service.at(devices).empty())))
                    throw std::runtime_error("CPU profile must hide CUDA and omit GPU reservations");
                if (jetson && service.at("runtime") != "nvidia")
                    throw std::runtime_error("Jetson must retain the source NVIDIA container runtime");
                if (rocm) {
                    if (args.at("EZLOCALAI_HIP_ARCHITECTURES") != "gfx1100" ||
                        args.at("EZLOCALAI_PACKAGE_TEST_EXCLUDE") != "" ||
                        actual.contains("CUDA_VISIBLE_DEVICES") ||
                        service.contains(devices))
                        throw std::runtime_error("ROCm profile must use its HIP target, full gate and AMD visibility");
                    const auto expected_devices = nlohmann::json::array({
                        {{"source", "/dev/kfd"}, {"target", "/dev/kfd"}, {"permissions", "rwm"}},
                        {{"source", "/dev/dri"}, {"target", "/dev/dri"}, {"permissions", "rwm"}}});
                    if (service.at("devices") != expected_devices ||
                        service.at("group_add") != nlohmann::json::array({"video", "render"}))
                        throw std::runtime_error("ROCm device access differs from the source profile");
                }
            }
        }
        std::cout << comparisons << " Compose settings checked (unset, empty and overridden), including "
                  << corrections << " documented default corrections; "
                     "platform and default build gates passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
