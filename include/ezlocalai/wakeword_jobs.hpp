#pragma once
#include "ezlocalai/wakeword_model.hpp"
#include "ezlocalai/wakeword_tts.hpp"
#include <filesystem>
#include <optional>

namespace ezlocalai {
struct WakewordJobOperations {
    std::function<WakewordTtsOperations(const VoiceCancellation &)> samples;
    std::function<std::shared_ptr<WakewordTrainableModel>(const std::string &word)> trainer;
    std::function<std::vector<std::string>()> load;
};
// One worker serializes training and its RNG/GPU allocations. Persisted jobs
// are loaded for status/cancellation, not silently restarted after a crash.
class WakewordJobs {
public:
    WakewordJobs(std::filesystem::path root,WakewordJobOperations operations);
    ~WakewordJobs();
    WakewordJobs(const WakewordJobs &)=delete;
    WakewordJobs &operator=(const WakewordJobs &)=delete;
    // JSON snapshot plus whether a job was created (false for an active word).
    std::pair<std::string,bool> create(const std::string &word,int64_t samples=500,int64_t epochs=50,int64_t batch=32);
    std::optional<std::string> get(const std::string &id) const;
    std::optional<std::string> active(const std::string &word) const;
    std::vector<std::string> list(const std::string &status="",int64_t limit=100) const;
    bool cancel(const std::string &id);
    void stop();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
