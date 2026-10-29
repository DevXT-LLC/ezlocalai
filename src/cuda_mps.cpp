#include "ezlocalai/cuda_mps.hpp"
#include "ezlocalai/process.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace ezlocalai {
std::optional<int> llm_mps_client_priority(const std::string &setting, size_t index) {
    if (setting.empty()) return std::nullopt;
    std::vector<std::optional<int>> values;
    size_t begin = 0;
    for (;;) {
        const auto end = setting.find(',', begin);
        auto value = setting.substr(begin, end == std::string::npos ? end : end - begin);
        const auto first = value.find_first_not_of(" \t\r\n");
        value = first == std::string::npos ? "" : value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
        if (value.empty()) values.push_back(std::nullopt);
        else if (value == "0" || value == "1") values.push_back(value[0] - '0');
        else throw std::invalid_argument("LLM_CUDA_MPS_CLIENT_PRIORITY entries must be 0, 1 or empty");
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return values[std::min(index, values.size() - 1)];
}

namespace {
bool enabled() {
    const char *raw = std::getenv("EZLOCALAI_CUDA_MPS");
    std::string value = raw ? raw : "";
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
    if (value.empty() || value == "false" || value == "0") return false;
    if (value == "true" || value == "1") return true;
    throw std::runtime_error("EZLOCALAI_CUDA_MPS must be true or false");
}
}

void CudaMpsService::command(const std::string &text, double timeout) {
    const auto input = directory_ + "/command";
    { std::ofstream file(input); file << text << '\n'; }
    const auto result = run_process_to_files({"nvidia-cuda-mps-control"}, directory_ + "/reply",
        directory_ + "/command-error", timeout, false, input);
    if (result.status != 0 || result.timed_out)
        throw std::runtime_error("MPS control failed: " + text);
}

CudaMpsService::CudaMpsService() {
    if (!enabled()) return;
    // MPS renumbers numeric device selections. Reusing e.g. "2,0" in its
    // clients would select the wrong ordinal or fail. Stable GPU UUIDs avoid
    // that ambiguity without expanding the operator's visible device set.
    if (const char *visible = std::getenv("CUDA_VISIBLE_DEVICES")) {
        if (!*visible) throw std::runtime_error("Managed MPS requires a visible CUDA device");
        std::istringstream devices(visible);
        std::string device;
        size_t index = 0;
        while (std::getline(devices, device, ',')) {
            if (device != std::to_string(index++) && device.find("GPU-") != 0 && device.find("MIG-") != 0)
                throw std::runtime_error("Managed MPS requires CUDA_VISIBLE_DEVICES GPU UUIDs or unchanged ordinals 0,1,...");
        }
    }
    // Existing external MPS configurations remain supported by leaving the
    // managed option off. Never commandeer an external daemon or pipe directory.
    for (const char *name : {"CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY"})
        if (std::getenv(name)) throw std::runtime_error(std::string("Managed MPS requires unset ") + name);
    char pattern[] = "/tmp/ezlocalai-mps-XXXXXX";
    const auto directory = mkdtemp(pattern);
    if (!directory) throw std::system_error(errno, std::generic_category(), "create private MPS directory");
    directory_ = directory;
    try {
        std::filesystem::create_directory(directory_ + "/log");
        if (setenv("CUDA_MPS_PIPE_DIRECTORY", directory_.c_str(), 1) != 0 ||
            setenv("CUDA_MPS_LOG_DIRECTORY", (directory_ + "/log").c_str(), 1) != 0)
            throw std::system_error(errno, std::generic_category(), "configure MPS environment");
        // Foreground mode gives us an owned, waitable child. Daemon mode double
        // forks and leaves orphaned/zombie processes when the worker is PID 1.
        posix_spawn_file_actions_t actions;
        int error = posix_spawn_file_actions_init(&actions);
        if (error) throw std::system_error(error, std::generic_category(), "MPS spawn actions");
        error = posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO,
            (directory_ + "/startup.log").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (!error) error = posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
#if defined(__GLIBC__)
#if __GLIBC_PREREQ(2, 34)
        if (!error) error = posix_spawn_file_actions_addclosefrom_np(&actions, 3);
#endif
#endif
        char executable[] = "nvidia-cuda-mps-control", foreground[] = "-f";
        char *args[] = {executable, foreground, nullptr};
        if (!error) error = posix_spawnp(&daemon_pid_, executable, &actions, nullptr, args, environ);
        posix_spawn_file_actions_destroy(&actions);
        if (error) throw std::system_error(error, std::generic_category(), "MPS launch");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        bool ready = false;
        bool listening = false;
        do {
            int status = 0;
            if (waitpid(daemon_pid_, &status, WNOHANG) == daemon_pid_) {
                daemon_pid_ = -1;
                throw std::runtime_error("MPS daemon exited during startup");
            }
            if (!listening) {
                // The foreground controller may not have created its socket yet.
                try { command("get_server_list", 1); listening = true; }
                catch (const std::exception &) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
                command("start_server -uid " + std::to_string(getuid()), 2);
            }
            command("get_server_list", 2);
            std::ifstream file(directory_ + "/reply");
            long long server = 0;
            if (file >> server && server > 1) { ready = true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } while (std::chrono::steady_clock::now() < deadline);
        if (!ready) throw std::runtime_error("MPS server did not become ready");
        std::cout << "Private CUDA MPS server ready; logs: " << directory_ << '\n' << std::flush;
    } catch (const std::exception &error) {
        // Preserve diagnostics on failure; do not silently benchmark ordinary
        // CUDA execution when the operator explicitly requested MPS.
        const auto diagnostic = std::string(error.what()) + "; logs: " + directory_;
        stop();
        throw std::runtime_error(diagnostic);
    }
}

void CudaMpsService::stop() noexcept {
    if (daemon_pid_ > 0) {
        try { command("quit", 3); }
        catch (const std::exception &error) { std::cerr << "MPS shutdown: " << error.what() << '\n'; }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        int status = 0;
        for (;;) {
            const auto done = waitpid(daemon_pid_, &status, WNOHANG);
            if (done == daemon_pid_ || (done < 0 && errno == ECHILD)) break;
            if (std::chrono::steady_clock::now() >= deadline) {
                // Only our recorded child, never a PID read from an external MPS server.
                kill(daemon_pid_, SIGKILL);
                while (waitpid(daemon_pid_, &status, 0) < 0 && errno == EINTR) {}
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        daemon_pid_ = -1;
    }
    if (!directory_.empty()) {
        unsetenv("CUDA_MPS_PIPE_DIRECTORY");
        unsetenv("CUDA_MPS_LOG_DIRECTORY");
    }
}

CudaMpsService::~CudaMpsService() { stop(); }

int CudaMpsService::supervise(const std::vector<std::string> &arguments,
                            const std::function<bool()> &stopping) {
    if (!active() || arguments.empty()) throw std::logic_error("MPS supervisor requires an active service and worker command");
    std::vector<char *> args;
    for (const auto &arg : arguments) args.push_back(const_cast<char *>(arg.c_str()));
    args.push_back(nullptr);
    const char *raw = std::getenv("EZLOCALAI_CUDA_MPS");
    const std::string previous = raw ? raw : "true";
    // The child joins this server through the inherited private pipe directory.
    // It must not start another supervisor. Keep operator configuration in the
    // parent intact; only the child receives the disabled bootstrap option.
    if (setenv("EZLOCALAI_CUDA_MPS", "false", 1) != 0)
        throw std::system_error(errno, std::generic_category(), "configure MPS child");
    pid_t worker = -1;
    const auto error = posix_spawnp(&worker, args[0], nullptr, nullptr, args.data(), environ);
    setenv("EZLOCALAI_CUDA_MPS", previous.c_str(), 1);
    if (error) throw std::system_error(error, std::generic_category(), "launch MPS worker");
    bool signalled = false;
    for (;;) {
        int status = 0;
        const auto done = waitpid(worker, &status, WNOHANG);
        if (done == worker) return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
        if (done < 0 && errno != EINTR) throw std::system_error(errno, std::generic_category(), "wait for MPS worker");
        if (!signalled && stopping()) { kill(worker, SIGTERM); signalled = true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

} // namespace ezlocalai
