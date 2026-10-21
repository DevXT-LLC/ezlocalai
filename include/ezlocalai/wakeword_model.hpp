#pragma once
#include "ezlocalai/ctts_audio.hpp"
#include <functional>
#include <memory>
#include <map>
#include <string>
#include <utility>

namespace ezlocalai {
class WakewordModel {
public:
    virtual ~WakewordModel() = default;
    virtual std::pair<bool, double> predict(const std::string &audio_bytes) = 0;
};
struct WakewordTrainingOptions {
    int64_t epochs = 50, batch_size = 32;
    double learning_rate = 0.001, validation_split = 0.2;
    // May throw to stop between samples/batches without interrupting a kernel.
    std::function<void()> check_cancel;
};
using WakewordMetrics = std::map<std::string, double>;
using WakewordHistory = std::map<std::string, std::vector<double>>;
using WakewordProgress = std::function<void(int64_t epoch, int64_t total, const WakewordMetrics &metrics)>;
using WakewordSavedFiles = std::vector<std::pair<std::string, std::string>>;
class WakewordTrainableModel : public WakewordModel {
public:
    virtual WakewordHistory train(const std::vector<std::string> &positive, const std::vector<std::string> &negative,
        const WakewordTrainingOptions &options = {}, const WakewordProgress &progress = {}) = 0;
    // Native torch.save-compatible model.pt and config.json. Export adapters
    // add ONNX/ESPDL separately; this method does not claim those artifacts.
    virtual void save_checkpoint(const std::string &directory) = 0;
    virtual void export_onnx(const std::string &directory) = 0;
    // Checkpoint/config are mandatory; optional export failures retain them.
    virtual WakewordSavedFiles save(const std::string &directory) = 0;
};
std::shared_ptr<WakewordTrainableModel> wakeword_torch_create(const std::string &config_json, const std::string &device = "");
using WakewordModelFactory = std::function<std::shared_ptr<WakewordModel>(const std::string &directory)>;
// Empty device selects the source's CUDA-if-available policy. No Python runtime.
WakewordModelFactory wakeword_torch_factory(const std::string &device = "");
// CPU torchaudio MFCC features, before the predictor's fixed 150-frame crop.
CttsAudioArray wakeword_audio_features(const std::string &audio_bytes, int sample_rate = 16000, int n_mfcc = 40);
}
