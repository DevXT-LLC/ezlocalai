#pragma once

#include <string>
#include <vector>
#include <functional>
#include <optional>
#include <sys/types.h>

namespace ezlocalai {

// Per-model CSV, using the last value for later models like other LLM settings.
// Empty means inherit NVIDIA's global CUDA_MPS_CLIENT_PRIORITY.
std::optional<int> llm_mps_client_priority(const std::string &setting, size_t index);

// Opt-in, worker-owned MPS server. A CUDA-free supervisor owns the live service;
// its worker exits before teardown. Never targets another worker's MPS server.
class CudaMpsService {
public:
    CudaMpsService();
    ~CudaMpsService();
    CudaMpsService(const CudaMpsService &) = delete;
    CudaMpsService &operator=(const CudaMpsService &) = delete;
    bool active() const { return daemon_pid_ > 0; }
    // The supervisor never initializes CUDA itself. Its worker must fully exit
    // (including CUDA global destructors) before the MPS server can quit.
    int supervise(const std::vector<std::string> &arguments, const std::function<bool()> &stopping);
private:
    std::string directory_;
    pid_t daemon_pid_ = -1;
    void command(const std::string &text, double timeout);
    void stop() noexcept;
};

} // namespace ezlocalai
