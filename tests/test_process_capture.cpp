#include <spawn.h>
#include <fcntl.h>
#include <dirent.h>
#include <unistd.h>
#include <set>
#include <stdexcept>
#include <iostream>
#include <string>

namespace {
std::set<int> descriptors() {
    auto *directory = opendir("/proc/self/fd");
    if (!directory) throw std::runtime_error("open descriptor directory");
    const int own = dirfd(directory);
    std::set<int> result;
    while (const auto *entry = readdir(directory)) {
        char *end = nullptr;
        const long fd = std::strtol(entry->d_name, &end, 10);
        if (end != entry->d_name && !*end && fd != own) result.insert(static_cast<int>(fd));
    }
    closedir(directory); return result;
}
std::set<int> original;
int capture_descriptors = 0;
bool all_cloexec = true;
int observed_spawnp(pid_t *pid, const char *file, const posix_spawn_file_actions_t *actions,
        const posix_spawnattr_t *attributes, char *const argv[], char *const envp[]) {
    // Inspect the real run_process pipe immediately before its actual spawn.
    // This observes the race window without relying on concurrent scheduling.
    for (const int fd : descriptors()) if (!original.count(fd)) {
        ++capture_descriptors;
        const int flags = fcntl(fd, F_GETFD);
        all_cloexec = all_cloexec && flags >= 0 && (flags & FD_CLOEXEC);
    }
    return ::posix_spawnp(pid, file, actions, attributes, argv, envp);
}
}
#define posix_spawnp observed_spawnp
#include "ezlocalai/process.hpp"
#undef posix_spawnp

int main() {
    try {
        original = descriptors();
        const auto result = ezlocalai::run_process({"sh", "-c", "printf capture-ok"}, true, true);
        if (result.status || result.output != "capture-ok")
            throw std::runtime_error("explicit child stdout must survive exec and preserve output");
        if (capture_descriptors != 2 || !all_cloexec)
            throw std::runtime_error("capture descriptors can leak into a concurrent child before spawn");
        if (descriptors() != original) throw std::runtime_error("capture descriptors leaked in parent");
        std::cout << "Captured command pipes are atomically close-on-exec; explicit stdout and parent cleanup pass\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
