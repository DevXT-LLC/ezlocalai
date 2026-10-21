#include "ezlocalai/process.hpp"
#include "fixture_ports.hpp"
#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;
using Json = nlohmann::json;
void check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
struct Child {
    pid_t pid = -1;
    ~Child() {
        if (pid <= 0) return;
        kill(pid, SIGTERM);
        int status;
        for (int i = 0; i < 100; ++i) {
            if (waitpid(pid, &status, WNOHANG) == pid) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
    }
};
int main(int argc, char **argv) {
    const auto original = fs::current_path();
    fs::path directory;
    try {
        check(argc == 2, "worker executable required");
        const auto binary = fs::absolute(argv[1]);
        const auto library_path = std::getenv("LD_LIBRARY_PATH") ? std::getenv("LD_LIBRARY_PATH") : "";
        const std::string libraries(library_path);
        const int port = fixture_ports();
        std::string pattern = "/tmp/ezlocalai-dotenv-startup-XXXXXX";
        const auto created = mkdtemp(pattern.data());
        check(created != nullptr, "temporary directory");
        directory = created;
        fs::current_path(directory);
        clearenv();
        setenv("PATH", "/usr/bin:/bin", 1);
        setenv("LD_LIBRARY_PATH", libraries.c_str(), 1);
        setenv("TEST_PORT", std::to_string(port).c_str(), 1);
        setenv("SECRET_SUFFIX", "process", 1);
        {
            std::ofstream file(".env");
            file << "HOST=127.0.0.1\nPORT=${TEST_PORT}\n"
                "SECRET_SUFFIX=file\nEZLOCALAI_API_KEY='fixture-${SECRET_SUFFIX}'\n"
                "PRESET_MODEL=initial-model\nDEFAULT_MODEL=${PRESET_MODEL}\n"
                "PRESET_MODEL=final-model\nDEFAULT_MODEL=${PRESET_MODEL}\n"
                "EZLOCALAI_NATIVE_BACKEND=echo\nEZLOCALAI_PRELOAD_MODEL_ON_START=false\n"
                "EZLOCALAI_MODEL_STATUS_ON_START=false\nCUDA_VISIBLE_DEVICES=\nN_PARALLEL=1\n"
                "TTS_ENABLED=false\nSTT_ENABLED=false\nEMBEDDING_ENABLED=false\n"
                "IMAGE_ENABLED=false\nVIDEO_ENABLED=false\nMUSIC_ENABLED=false\nROUTER_URL=\n";
        }
        {
            Child worker;
            worker.pid = fork();
            check(worker.pid >= 0, "fork worker");
            if (!worker.pid) {
                const int fd = open("worker.log", O_WRONLY | O_CREAT | O_TRUNC, 0600);
                if (fd < 0) _exit(126);
                dup2(fd, 1); dup2(fd, 2); close(fd);
                execl(binary.c_str(), binary.c_str(), nullptr);
                _exit(127);
            }
            const auto url = "http://127.0.0.1:" + std::to_string(port);
            bool ready = false;
            for (int i = 0; i < 100; ++i) {
                if (ezlocalai::run_process({"/usr/bin/curl", "-fsS", "--max-time", "1", url + "/health"}, true, true).status == 0) { ready = true; break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            check(ready, "worker did not bind interpolated port");
            auto denied = ezlocalai::run_process({"/usr/bin/curl", "-sS", "--max-time", "5", "-o", "/dev/null", "-w", "%{http_code}", url + "/v1/models"}, true, true);
            check(denied.status == 0 && denied.output == "401", "dotenv API key not enforced");
            auto result = ezlocalai::run_process({"/usr/bin/curl", "-fsS", "--max-time", "5", "-H", "Authorization: Bearer fixture-process", "-H", "Content-Type: application/json", "-d", R"({"messages":[{"role":"user","content":"ok"}]})", url + "/v1/chat/completions"}, true, true);
            check(result.status == 0, "interpolated API key was rejected");
            check(Json::parse(result.output).at("model") == "final-model", "dotenv model profile/duplicate precedence not used");
        }
        { std::ofstream file(".env", std::ios::binary); file << "KEY=" << static_cast<char>(0xff) << "\n"; }
        auto invalid = ezlocalai::run_process_to_files({binary.string(), "--help"}, "invalid.log", "", 5, true);
        std::ifstream invalid_log("invalid.log");
        const std::string diagnostic{std::istreambuf_iterator<char>(invalid_log), std::istreambuf_iterator<char>()};
        check(invalid.status != 0 && diagnostic.find("Failed to load .env") != std::string::npos, "invalid UTF-8 did not stop startup");
        setenv("PYTHON_DOTENV_DISABLED", "true", 1);
        check(ezlocalai::run_process({binary.string(), "--help"}, true, true).status == 0, "dotenv disable switch ignored at startup");
        fs::current_path(original);
        fs::remove_all(directory);
        std::cout << "dotenv startup: port, model, auth, process precedence, invalid encoding and disable switch passed\n";
        return 0;
    } catch (const std::exception &error) {
        fs::current_path(original);
        std::cerr << error.what() << "\nArtifacts: " << directory << '\n';
        return 1;
    }
}
