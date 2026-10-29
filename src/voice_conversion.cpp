#include "ezlocalai/voice_conversion.hpp"
#include "ezlocalai/voice_audio.hpp"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
extern char **environ;
namespace ezlocalai {
namespace {
using Json = nlohmann::json;
std::string env(const char *key, const std::string &fallback) {
    auto p = getenv(key);
    return p && *p ? p : fallback;
}
int integer(const char *key, int fallback, int min, int max) {
    auto s = env(key, std::to_string(fallback));
    size_t used = 0;
    int n = std::stoi(s, &used);
    if (used != s.size() || n < min || n > max)
        throw std::invalid_argument(std::string("Invalid ") + key);
    return n;
}
} // namespace
VoiceConversionConfig voice_conversion_config() {
    VoiceConversionConfig c;
    auto enabled = env("MEANVC2_ENABLED", "false");
    std::transform(enabled.begin(), enabled.end(), enabled.begin(),
                   [](unsigned char c) { return c >= 'A' && c <= 'Z' ? char(c + 32) : char(c); });
    c.enabled = enabled == "true" || enabled == "1" || enabled == "yes" || enabled == "on";
    c.executable = env("MEANVC2_BIN", "/usr/local/bin/ezlocalai-meanvc2");
    c.model = env("MEANVC2_MODEL_PATH",
                  env("MODELS_PATH", "models") + "/MeanVC2-GGUF/meanvc2-120ms-40ms-fp32.gguf");
    c.voices = env("MEANVC2_VOICES_DIR", "voices");
    c.backend = env("MEANVC2_BACKEND", "cpu");
    if (!c.enabled)
        return c;
    if (c.backend != "cpu" && c.backend != "cuda" && c.backend != "hip")
        throw std::invalid_argument("MEANVC2_BACKEND must be cpu, cuda or hip");
    c.device = integer("MEANVC2_DEVICE", 0, 0, 63);
    c.threads = integer("MEANVC2_THREADS", 4, 1, 64);
    c.startup_timeout = integer("MEANVC2_STARTUP_TIMEOUT", 180, 1, 600);
    c.chunk_timeout = integer("MEANVC2_CHUNK_TIMEOUT", 15, 1, 120);
    return c;
}
bool voice_conversion_ready(const VoiceConversionConfig &c) {
    std::error_code ec;
    return c.enabled && access(c.executable.c_str(), X_OK) == 0 &&
           std::filesystem::is_regular_file(c.model, ec);
}
double voice_conversion_free_ram(double host_available_gb, const std::string &root) {
    // Keep the reference's host RAM reporting, but do not start a new 1.6GB
    // engine when the container has less headroom than the host advertises.
    std::ifstream maximum(root + "/memory.max");
    std::string text;
    if (!(maximum >> text) || text == "max")
        return host_available_gb;
    uint64_t limit = 0, used = 0, reclaimable = 0;
    try {
        size_t end = 0;
        limit = std::stoull(text, &end);
        if (end != text.size())
            return 0;
    } catch (const std::exception &) {
        return 0;
    }
    std::ifstream current(root + "/memory.current");
    if (!(current >> used))
        return 0;
    std::ifstream stat(root + "/memory.stat");
    std::string key;
    uint64_t value;
    while (stat >> key >> value)
        if (key == "inactive_file")
            reclaimable = value;
    const double available =
        std::max(0., double(limit) - double(used) + double(std::min(reclaimable, used))) / 1073741824.;
    return std::min(host_available_gb, available);
}
std::vector<std::string> voice_conversion_voices(const VoiceConversionConfig &c) {
    std::vector<std::string> voices;
    std::error_code ec;
    for (const auto &entry : std::filesystem::directory_iterator(c.voices, ec)) {
        if (entry.path().extension() != ".wav")
            continue;
        const auto id = entry.path().stem().string();
        if (id == "native")
            continue;
        try {
            voice_conversion_reference(c, id);
            voices.push_back(id);
        } catch (const std::exception &) {
        }
    }
    std::sort(voices.begin(), voices.end());
    return voices;
}
std::string voice_conversion_reference(const VoiceConversionConfig &c, const std::string &voice) {
    namespace fs = std::filesystem;
    if (voice.empty() || voice == "." || voice == ".." || voice.size() > 128 ||
        voice.find_first_of("/\\\0", 0, 3) != std::string::npos)
        throw std::invalid_argument("Invalid realtime voice ID");
    std::error_code ec;
    const auto base = fs::canonical(c.voices, ec);
    if (ec)
        throw std::invalid_argument("Voice directory unavailable");
    const auto filename =
        voice.size() >= 4 && voice.substr(voice.size() - 4) == ".wav" ? voice : voice + ".wav";
    const auto path = fs::canonical(base / filename, ec);
    if (ec || path.parent_path() != base || !fs::is_regular_file(path, ec) ||
        fs::file_size(path, ec) > 32 * 1024 * 1024 || ec)
        throw std::invalid_argument("Unknown or oversized realtime reference voice");
    return path.string();
}
class VoiceConversion::Impl {
  public:
    struct Job {
        uint64_t generation;
        std::string op, pcm, metadata;
    };
    VoiceConversionConfig config;
    std::function<bool()> external_stop;
    std::atomic<bool> stopping{false};
    std::atomic<uint64_t> epoch{1};
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<Job> jobs;
    std::deque<VoiceConversionResult> results;
    size_t input_bytes = 0, output_bytes = 0;
    std::string failure, pending;
    std::thread producer;
    pid_t pid = -1;
    int fd = -1;
    static constexpr size_t budget = 1024 * 1024;
    bool stopped() const { return stopping.load() || (external_stop && external_stop()); }
    void wait_io(short events, std::chrono::steady_clock::time_point deadline) {
        while (!stopped()) {
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("MeanVC2 engine timed out");
            pollfd p{fd, events, 0};
            const auto rc = poll(&p, 1, 20);
            if (rc < 0 && errno != EINTR)
                throw std::runtime_error("MeanVC2 pipe poll failed");
            if (rc > 0 && (p.revents & (events | POLLHUP | POLLERR)))
                return;
        }
        throw std::runtime_error("MeanVC2 stopped");
    }
    Json receive(double seconds) {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(int(seconds * 1000));
        for (;;) {
            const auto end = pending.find('\n');
            if (end != std::string::npos) {
                auto v = Json::parse(pending.substr(0, end));
                pending.erase(0, end + 1);
                if (!v.is_object() || v.contains("error"))
                    throw std::runtime_error("MeanVC2 engine failure: " +
                                             v.value("error", std::string("invalid reply")));
                return v;
            }
            if (pending.size() > 256 * 1024)
                throw std::runtime_error("MeanVC2 output exceeds block limit");
            wait_io(POLLIN, deadline);
            char bytes[8192];
            auto n = recv(fd, bytes, sizeof(bytes), 0);
            if (n > 0)
                pending.append(bytes, size_t(n));
            else if (n == 0)
                throw std::runtime_error("MeanVC2 engine exited");
            else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
                throw std::runtime_error("MeanVC2 pipe read failed");
        }
    }
    Json exchange(const Job &job) {
        const auto line = Json{{"op", job.op}, {"pcm_s16", voice_base64_encode(job.pcm)}}.dump() + "\n";
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(int(config.chunk_timeout * 1000));
        size_t at = 0;
        while (at < line.size()) {
            wait_io(POLLOUT, deadline);
            auto n = send(fd, line.data() + at, line.size() - at, MSG_NOSIGNAL);
            if (n > 0)
                at += size_t(n);
            else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
                throw std::runtime_error("MeanVC2 pipe write failed");
        }
        return receive(config.chunk_timeout);
    }
    void cleanup() noexcept {
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
        if (pid > 0) {
            int status = 0;
            auto rc = waitpid(pid, &status, WNOHANG);
            if (rc == 0) {
                kill(pid, SIGTERM);
                for (int i = 0; i < 20; ++i) {
                    rc = waitpid(pid, &status, WNOHANG);
                    if (rc != 0)
                        break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                if (rc == 0) {
                    kill(pid, SIGKILL);
                    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
                    }
                }
            }
            pid = -1;
        }
    }
    Impl(const VoiceConversionConfig &c, const std::string &reference, std::function<bool()> stop)
        : config(c), external_stop(std::move(stop)) {
        try {
            int pair[2];
            if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, pair))
                throw std::runtime_error("MeanVC2 socketpair failed");
            fd = pair[0];
            posix_spawn_file_actions_t actions;
            int error = posix_spawn_file_actions_init(&actions);
            const bool valid = !error;
            if (!error)
                error = posix_spawn_file_actions_adddup2(&actions, pair[1], STDIN_FILENO);
            if (!error)
                error = posix_spawn_file_actions_adddup2(&actions, pair[1], STDOUT_FILENO);
#if defined(__GLIBC__) && __GLIBC_PREREQ(2, 34)
            if (!error)
                error = posix_spawn_file_actions_addclosefrom_np(&actions, 3);
#else
            if (!error)
                error = posix_spawn_file_actions_addclose(&actions, pair[0]);
            if (!error)
                error = posix_spawn_file_actions_addclose(&actions, pair[1]);
#endif
            std::vector<std::string> args{c.executable,
                                          c.model,
                                          reference,
                                          c.backend,
                                          std::to_string(c.device),
                                          std::to_string(c.threads)},
                environment;
            for (char **p = environ; *p; ++p)
                if (std::string(*p).rfind("GGML_BACKEND_PATH=", 0) != 0)
                    environment.emplace_back(*p);
            environment.push_back("GGML_BACKEND_PATH=" +
                                  std::filesystem::path(c.executable).parent_path().string());
            std::vector<char *> argv, envp;
            for (auto &a : args)
                argv.push_back(a.data());
            argv.push_back(nullptr);
            for (auto &e : environment)
                envp.push_back(e.data());
            envp.push_back(nullptr);
            if (!error)
                error = posix_spawn(&pid, c.executable.c_str(), &actions, nullptr, argv.data(), envp.data());
            if (valid)
                posix_spawn_file_actions_destroy(&actions);
            close(pair[1]);
            if (error)
                throw std::runtime_error("Cannot start native MeanVC2 engine");
            if (fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK) < 0)
                throw std::runtime_error("MeanVC2 pipe flags failed");
            auto ready = receive(c.startup_timeout);
            if (!ready.value("ready", false) || ready.value("sample_rate", 0) != 24000)
                throw std::runtime_error("MeanVC2 invalid readiness");
            producer = std::thread([this] { run(); });
        } catch (...) {
            cleanup();
            throw;
        }
    }
    ~Impl() {
        stopping = true;
        changed.notify_all();
        if (producer.joinable())
            producer.join();
        cleanup();
    }
    void run() {
        try {
            for (;;) {
                Job job;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    changed.wait(lock, [&] { return stopped() || !jobs.empty(); });
                    if (stopped())
                        return;
                    job = std::move(jobs.front());
                    jobs.pop_front();
                    input_bytes -= job.pcm.size() + job.metadata.size() + 64;
                }
                auto reply = exchange(job);
                if (!reply.value("ok", false) || reply.value("sample_rate", 0) != 24000)
                    throw std::runtime_error("Invalid MeanVC2 audio reply");
                if (job.op == "reset")
                    continue;
                VoiceConversionResult result{job.generation, job.metadata,
                                             voice_base64_decode(reply.at("pcm_s16").get<std::string>()),
                                             job.op == "finish"};
                if (result.pcm.size() % 2 || result.pcm.size() > 128 * 1024)
                    throw std::runtime_error("Invalid MeanVC2 PCM block");
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [&] {
                    return stopped() || job.generation != epoch ||
                           output_bytes + result.pcm.size() + result.metadata.size() + 64 <= budget;
                });
                if (stopped())
                    return;
                if (job.generation != epoch)
                    continue;
                output_bytes += result.pcm.size() + result.metadata.size() + 64;
                results.push_back(std::move(result));
            }
        } catch (const std::exception &e) {
            std::lock_guard<std::mutex> lock(mutex);
            if (!stopped())
                failure = e.what();
        }
    }
    void enqueue(Job job) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!failure.empty())
            throw std::runtime_error(failure);
        auto bytes = job.pcm.size() + job.metadata.size() + 64;
        if (input_bytes + bytes > budget)
            throw std::runtime_error("MeanVC2 input queue capacity exceeded");
        input_bytes += bytes;
        jobs.push_back(std::move(job));
        changed.notify_all();
    }
};
VoiceConversion::VoiceConversion(const VoiceConversionConfig &c, const std::string &r,
                                 const std::function<bool()> &s)
    : impl_(std::make_unique<Impl>(c, r, s)) {}
VoiceConversion::~VoiceConversion() = default;
uint64_t VoiceConversion::generation() const { return impl_->epoch.load(); }
void VoiceConversion::audio(const std::string &p, const std::string &m) {
    if (p.size() % 2 || p.size() > 7680)
        throw std::runtime_error("Invalid voice input PCM block");
    impl_->enqueue({generation(), "audio", p, m});
}
void VoiceConversion::finish(const std::string &m) { impl_->enqueue({generation(), "finish", "", m}); }
void VoiceConversion::cancel() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    ++impl_->epoch;
    impl_->jobs.clear();
    impl_->results.clear();
    impl_->input_bytes = impl_->output_bytes = 0;
    impl_->jobs.push_back({generation(), "reset", "", ""});
    impl_->input_bytes = 64;
    impl_->changed.notify_all();
}
bool VoiceConversion::pop(VoiceConversionResult &r) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->failure.empty())
        throw std::runtime_error(impl_->failure);
    if (impl_->results.empty())
        return false;
    r = std::move(impl_->results.front());
    impl_->results.pop_front();
    impl_->output_bytes -= r.pcm.size() + r.metadata.size() + 64;
    impl_->changed.notify_all();
    return true;
}
size_t VoiceConversion::queued_bytes() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->input_bytes;
}
} // namespace ezlocalai
