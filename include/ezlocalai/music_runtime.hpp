#pragma once

#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace ezlocalai {

std::string ace_step_internal_url();
std::string ace_step_server_url();
std::string ace_step_models_dir();
std::string ace_step_binary_path();
std::vector<std::string> selected_ace_step_model_files();
std::vector<std::string> build_ace_step_server_command();
bool music_should_unload_llm();
bool music_should_reload_llm();

struct MusicPrecacheOperations {
    std::function<void(const std::string &)> mkdir;
    std::function<bool(const std::string &)> exists;
    std::function<void(const std::string &, const std::string &, const std::string &)> download;
};
// Startup preparation follows precache.py: failures are logged and do not
// prevent the worker from starting. Each existing path is reused without a Hub
// request; external servers own their own model directory.
void precache_music_models(MusicPrecacheOperations operations = {});

// Like MUSIC.py's module-level Popen, this tracks the child across lazy client
// creation. A health timeout does not kill the child or prevent generation.
class AceStepRuntime {
public:
    int start(const std::function<bool(const std::string &)> &healthy,
              const std::function<double()> &startup_timeout);
    int process_id() const;
private:
    bool running_locked();
    mutable std::mutex mutex_;
    int pid_ = -1;
    bool reaped_ = false;
};

AceStepRuntime &ace_step_runtime();

} // namespace ezlocalai
