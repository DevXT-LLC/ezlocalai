#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
namespace ezlocalai {
struct VoiceConversionConfig {
    bool enabled = false;
    std::string executable, model, voices, backend = "cpu";
    int device = 0, threads = 4;
    double startup_timeout = 180, chunk_timeout = 15;
};
VoiceConversionConfig voice_conversion_config();
std::string voice_conversion_reference(const VoiceConversionConfig &, const std::string &voice);
bool voice_conversion_ready(const VoiceConversionConfig &);
std::vector<std::string> voice_conversion_voices(const VoiceConversionConfig &);
double voice_conversion_free_ram(double host_available_gb, const std::string &cgroup_root = "/sys/fs/cgroup");
struct VoiceConversionResult {
    uint64_t generation = 0;
    std::string metadata, pcm;
    bool finished = false;
};
// One native engine per converted call. Socket and process access remain on the
// producer thread; the relay's cancellation never waits for an inference chunk.
class VoiceConversion {
  public:
    VoiceConversion(const VoiceConversionConfig &, const std::string &reference,
                    const std::function<bool()> &stopped);
    ~VoiceConversion();
    VoiceConversion(const VoiceConversion &) = delete;
    VoiceConversion &operator=(const VoiceConversion &) = delete;
    uint64_t generation() const;
    void audio(const std::string &pcm, const std::string &metadata);
    void finish(const std::string &metadata);
    void cancel();
    bool pop(VoiceConversionResult &);
    size_t queued_bytes() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ezlocalai
