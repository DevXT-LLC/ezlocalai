#pragma once

#include <cerrno>
#include <chrono>
#include <csignal>
#include <fcntl.h>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include <thread>

extern char **environ;

namespace ezlocalai {

struct ProcessResult {
    int status;
    std::string output;
};

struct FileProcessResult {
    int status = 0;
    bool timed_out = false;
};

// File-backed output avoids retaining model logs or downloaded weights in RAM.
// Timeout cleanup waits for the child, so callers can safely release its files
// and restore GPU residency before returning to request admission.
inline FileProcessResult run_process_to_files(const std::vector<std::string> &arguments,
                                              const std::string &stdout_path,
                                              const std::string &stderr_path,
                                              double timeout_seconds = -1,
                                              bool merge_stderr = false,
                                              const std::string &stdin_path = "",
                                              bool restore_signals = false) {
    if (arguments.empty()) throw std::invalid_argument("empty process command");
    std::vector<char *> argv;
    for (const auto &argument : arguments) argv.push_back(const_cast<char *>(argument.c_str()));
    argv.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    int error = posix_spawn_file_actions_init(&actions);
    if (error) throw std::system_error(error, std::generic_category(), "spawn file actions");
    if (!stdin_path.empty()) error = posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, stdin_path.c_str(), O_RDONLY, 0);
    if (!error && !stdout_path.empty()) error = posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, stdout_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (!error && merge_stderr) error = posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
    else if (!error && !stderr_path.empty()) error = posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, stderr_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
#if defined(__GLIBC__)
#if __GLIBC_PREREQ(2, 34)
    if (!error) error = posix_spawn_file_actions_addclosefrom_np(&actions, 3);
#endif
#endif
    pid_t pid = -1;
    posix_spawnattr_t attributes;
    bool attributes_ready = false;
    if (!error && restore_signals) {
        error = posix_spawnattr_init(&attributes);
        attributes_ready = !error;
        sigset_t defaults;
        sigemptyset(&defaults);
        sigaddset(&defaults, SIGPIPE);
#ifdef SIGXFZ
        sigaddset(&defaults, SIGXFZ);
#endif
#ifdef SIGXFSZ
        sigaddset(&defaults, SIGXFSZ);
#endif
        if (!error) error = posix_spawnattr_setsigdefault(&attributes, &defaults);
        if (!error) error = posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETSIGDEF);
    }
    if (!error) error = posix_spawnp(&pid, argv[0], &actions, attributes_ready ? &attributes : nullptr, argv.data(), environ);
    if (attributes_ready) posix_spawnattr_destroy(&attributes);
    posix_spawn_file_actions_destroy(&actions);
    if (error) throw std::system_error(error, std::generic_category(), arguments[0]);
    const auto started = std::chrono::steady_clock::now();
    FileProcessResult result;
    int status = 0;
    for (;;) {
        const auto done = waitpid(pid, &status, WNOHANG);
        if (done == pid) break;
        if (done < 0 && errno != EINTR) throw std::system_error(errno, std::generic_category(), "waitpid");
        if (timeout_seconds >= 0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >= timeout_seconds) {
            kill(pid, SIGKILL);
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
            result.timed_out = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    result.status = WIFEXITED(status) ? WEXITSTATUS(status) : -WTERMSIG(status);
    return result;
}

// Execute an argument vector directly. Model paths and environment values must
// never be interpreted as shell programs.
inline ProcessResult run_process(const std::vector<std::string> &arguments, bool capture = false, bool quiet_stderr = false) {
    if (arguments.empty()) throw std::invalid_argument("empty process command");
    std::vector<char *> argv;
    for (const auto &argument : arguments) argv.push_back(const_cast<char *>(argument.c_str()));
    argv.push_back(nullptr);

    int pipefd[2] = {-1, -1};
    if (capture) {
#ifdef __linux__
        // Concurrent spawns must not inherit another capture's write end,
        // otherwise EOF waits for an unrelated child to finish. dup2 below
        // deliberately makes only the child's STDOUT survive its exec.
        if (pipe2(pipefd, O_CLOEXEC) != 0) throw std::system_error(errno, std::generic_category(), "pipe");
#else
        if (pipe(pipefd) != 0) throw std::system_error(errno, std::generic_category(), "pipe");
        if (fcntl(pipefd[0], F_SETFD, FD_CLOEXEC) < 0 || fcntl(pipefd[1], F_SETFD, FD_CLOEXEC) < 0) {
            const int saved = errno; close(pipefd[0]); close(pipefd[1]);
            throw std::system_error(saved, std::generic_category(), "capture pipe close-on-exec");
        }
#endif
    }
    posix_spawn_file_actions_t actions;
    int error = posix_spawn_file_actions_init(&actions);
    if (error) {
        if (capture) { close(pipefd[0]); close(pipefd[1]); }
        throw std::system_error(error, std::generic_category(), "spawn file actions");
    }
    if (!error && capture) {
        error = posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO);
        if (!error) error = posix_spawn_file_actions_addclose(&actions, pipefd[0]);
        if (!error) error = posix_spawn_file_actions_addclose(&actions, pipefd[1]);
    }
    if (!error && quiet_stderr) error = posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    pid_t pid = -1;
    if (!error) error = posix_spawnp(&pid, argv[0], &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    if (capture) close(pipefd[1]);
    if (error) {
        if (capture) close(pipefd[0]);
        throw std::system_error(error, std::generic_category(), arguments[0]);
    }
    ProcessResult result{0, {}};
    int read_error = 0;
    if (capture) {
        char buffer[8192];
        for (;;) {
            const auto count = read(pipefd[0], buffer, sizeof(buffer));
            if (count > 0) result.output.append(buffer, static_cast<size_t>(count));
            else if (count < 0 && errno == EINTR) continue;
            else {
                if (count < 0) read_error = errno;
                break;
            }
        }
        close(pipefd[0]);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) throw std::system_error(errno, std::generic_category(), "waitpid");
    }
    if (read_error) throw std::system_error(read_error, std::generic_category(), "read process output");
    result.status = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
    return result;
}

inline void exec_process(const std::vector<std::string> &arguments) {
    if (arguments.empty()) throw std::invalid_argument("empty process command");
    std::vector<char *> argv;
    for (const auto &argument : arguments) argv.push_back(const_cast<char *>(argument.c_str()));
    argv.push_back(nullptr);
    execvp(argv[0], argv.data());
    throw std::system_error(errno, std::generic_category(), arguments[0]);
}

} // namespace ezlocalai
