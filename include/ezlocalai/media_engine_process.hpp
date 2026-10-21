#pragma once
#include "ezlocalai/process.hpp"
#include <memory>
#include <mutex>
namespace ezlocalai {
// A private inherited socket carries requests; stdout/stderr remain file-backed.
// Failed or timed-out requests terminate and reap the engine before returning.
class MediaEngineProcess {
public:
    explicit MediaEngineProcess(std::vector<std::string> command);
    ~MediaEngineProcess();
    MediaEngineProcess(const MediaEngineProcess &) = delete;
    MediaEngineProcess &operator=(const MediaEngineProcess &) = delete;
    FileProcessResult request(const std::vector<std::string> &arguments, bool preload,
                              const std::string &log_path, double timeout_seconds);
    void stop();
    pid_t pid() const;
private:
    void start_locked();
    void stop_locked(bool immediate = false);
    std::vector<std::string> command_;
    mutable std::mutex mutex_;
    pid_t pid_ = -1;
    int socket_ = -1;
};
}
