#pragma once
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>
namespace ezlocalai {
using SttSpeakerFeature = std::array<float, 40>;
std::vector<float> stt_diarization_audio(const std::string &path);
SttSpeakerFeature stt_diarization_feature(const std::vector<float> &audio);
std::vector<int> stt_diarization_labels(const std::vector<SttSpeakerFeature> &features, std::optional<int> speakers = {});
class SttDiarizer {
public:
    explicit SttDiarizer(std::function<double()> now = {}, std::function<std::string()> uuid = {}, double ttl = 86400);
    ~SttDiarizer();
    std::string diarize(const std::string &path, const std::string &segments,
        const std::string &speakers = "null", const std::string &session = "null");
private:
    struct State;
    std::unique_ptr<State> state_;
};
// One process-wide store, shared across models and pool reloads, as in STT.py.
std::string stt_diarize(const std::string &path, const std::string &segments,
    const std::string &speakers, const std::string &session);
}
