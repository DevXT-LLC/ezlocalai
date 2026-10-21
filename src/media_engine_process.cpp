#include "ezlocalai/media_engine_process.hpp"
#include "ezlocalai/media_worker_protocol.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <filesystem>
namespace ezlocalai {
MediaEngineProcess::MediaEngineProcess(std::vector<std::string> command) : command_(std::move(command)) {
    if (command_.empty()) throw std::invalid_argument("empty media worker command");
    for (const auto &arg : command_) if (arg.find('\0') != std::string::npos) throw std::invalid_argument("embedded null byte");
}
MediaEngineProcess::~MediaEngineProcess() { stop(); }
pid_t MediaEngineProcess::pid() const { std::lock_guard<std::mutex> lock(mutex_); return pid_; }
void MediaEngineProcess::start_locked() {
    if (pid_ > 0) return;
    auto args = command_;
    args.insert(args.end(), {"--ezlocalai-worker-fd", "3", "--ezlocalai-parent-pid", std::to_string(getpid())});
    std::vector<char *> argv; for (auto &arg : args) argv.push_back(arg.data()); argv.push_back(nullptr);
    int sockets[2];
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets)) throw std::system_error(errno, std::generic_category(), "media socketpair");
    posix_spawn_file_actions_t actions;
    int error = posix_spawn_file_actions_init(&actions);
    const bool initialized = !error;
    // Duplication clears CLOEXEC even if the source socket was allocated as 3.
    if (!error) error = posix_spawn_file_actions_adddup2(&actions, sockets[1], 3);
    if (!error) error = posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0);
    if (!error) error = posix_spawn_file_actions_addopen(&actions, 1, "/dev/null", O_WRONLY, 0);
    if (!error) error = posix_spawn_file_actions_adddup2(&actions, 1, 2);
#if defined(__GLIBC__) && __GLIBC_PREREQ(2, 34)
    if (!error) error = posix_spawn_file_actions_addclosefrom_np(&actions, 4);
#else
    if (!error && sockets[0] != 3) error = posix_spawn_file_actions_addclose(&actions, sockets[0]);
    if (!error && sockets[1] != 3) error = posix_spawn_file_actions_addclose(&actions, sockets[1]);
#endif
    pid_t child = -1;
    if (!error) error = posix_spawnp(&child, argv[0], &actions, nullptr, argv.data(), environ);
    if (initialized) posix_spawn_file_actions_destroy(&actions);
    close(sockets[1]);
    if (error) { close(sockets[0]); throw std::system_error(error, std::generic_category(), "media worker spawn"); }
    socket_ = sockets[0]; pid_ = child;
}
void MediaEngineProcess::stop_locked(bool immediate) {
    if (socket_ >= 0) { shutdown(socket_, SHUT_RDWR); close(socket_); socket_ = -1; }
    if (pid_ <= 0) return;
    const auto child = pid_; pid_ = -1;
    int status;
    if (!immediate) for (int i = 0; i < 100; ++i) {
        const auto done = waitpid(child, &status, WNOHANG);
        if (done == child || (done < 0 && errno == ECHILD)) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    kill(child, SIGKILL);
    while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
}
void MediaEngineProcess::stop() { std::lock_guard<std::mutex> lock(mutex_); stop_locked(); }
FileProcessResult MediaEngineProcess::request(const std::vector<std::string> &arguments, bool preload,
                                             const std::string &log_path, double timeout_seconds) {
    if (!std::isfinite(timeout_seconds)) throw std::invalid_argument("invalid media worker timeout");
    for (const auto &arg : arguments) if (arg.find('\0') != std::string::npos) throw std::invalid_argument("embedded null byte");
    if (log_path.find('\0') != std::string::npos) throw std::invalid_argument("embedded null byte");
    const auto payload = nlohmann::json{{"version", 1}, {"args", arguments}, {"preload", preload},
        {"log", std::filesystem::absolute(log_path).string()}}.dump();
    if (payload.size() > media_wire::max_message) throw std::length_error("media worker message too large");
    std::lock_guard<std::mutex> lock(mutex_);
    start_locked();
    try {
        const auto until = media_wire::deadline(timeout_seconds);
        media_wire::send(socket_, payload, until);
        std::string reply;
        if (!media_wire::receive(socket_, reply, until)) throw std::runtime_error("media worker exited before reply");
        const auto result = nlohmann::json::parse(reply);
        if (result.at("version") != 1 || !result.at("status").is_number_integer()) throw std::runtime_error("invalid media worker response");
        const int status = result.at("status");
        if (status) stop_locked(true);
        return {status, false};
    } catch (const media_wire::Timeout &) { stop_locked(true); return {-SIGKILL, true}; }
    catch (...) { stop_locked(true); throw; }
}
}
