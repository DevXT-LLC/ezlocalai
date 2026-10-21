// MFCC operations follow torchaudio 2.9.1; see licenses/torchaudio.txt.
#include "ezlocalai/wakeword_model.hpp"
#include "ezlocalai/ctts_torch.hpp"
#include "ezlocalai/torch_checkpoint.hpp"
#include "ezlocalai/wakeword_paths.hpp"
#include "wakeword_checkpoint.hpp"
#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Context.h>
#include "torch_gpu_runtime.hpp"
#include <caffe2/serialize/inline_container.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/rnn.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace ezlocalai {
namespace {
using Json = nlohmann::json;
void require(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
std::string default_device() {
#ifdef EZLOCALAI_TORCH_GPU
    if (torch_gpu::device_count() > 0) {
        // Retain libtorch_cuda under --as-needed so its device/kernel
        // registrations are present even in a wake-word-only executable.
        (void)torch_gpu::device_properties(0);
        return "cuda";
    }
#endif
    return "cpu";
}
struct Config {
    int64_t n_mfcc = 40, sample_rate = 16000, hidden_size = 64, num_layers = 2;
    double dropout = 0.3;
    bool gru = false;
    Json data;
    static Json read(const std::string &directory) {
        std::ifstream input(wakeword_contained_path(directory, "config.json"));
        Json data; input >> data; return data;
    }
    explicit Config(const std::string &directory) : Config(read(directory)) {}
    explicit Config(Json input) : data(std::move(input)) {
        require(data.is_object() && data.contains("word"), "Invalid wake-word configuration");
        const std::set<std::string> keys{"word", "n_mfcc", "sample_rate", "max_length_sec", "model_type", "hidden_size", "num_layers", "dropout"};
        for (auto it = data.begin(); it != data.end(); ++it) require(keys.count(it.key()), "Unknown wake-word configuration field");
        const auto integer = [&](const char *key, int64_t fallback) {
            if (!data.contains(key)) return fallback;
            require(data[key].is_number_integer(), "Invalid integer in wake-word configuration");
            return data[key].get<int64_t>();
        };
        n_mfcc = integer("n_mfcc", n_mfcc); sample_rate = integer("sample_rate", sample_rate);
        hidden_size = integer("hidden_size", hidden_size); num_layers = integer("num_layers", num_layers);
        if (data.contains("dropout")) { require(data["dropout"].is_number(), "Invalid dropout"); dropout = data["dropout"].get<double>(); }
        gru = data.value("model_type", Json("cnn")) == "gru";
        require(sample_rate > 0 && sample_rate <= INT32_MAX && n_mfcc > 0 && n_mfcc <= 80 && hidden_size > 0,
                "Invalid wake-word model dimensions");
        const Json defaults{{"n_mfcc", 40}, {"sample_rate", 16000}, {"max_length_sec", 1.5},
            {"model_type", "cnn"}, {"hidden_size", 64}, {"num_layers", 2}, {"dropout", 0.3}};
        for (auto it = defaults.begin(); it != defaults.end(); ++it) if (!data.contains(it.key())) data[it.key()] = it.value();
    }
};
class Network : public torch::nn::Module {
public:
    virtual at::Tensor forward(at::Tensor input) = 0;
};
// Default Python torch.optim.Adam 2.9.1 operations. The C++ frontend uses
// mul/add for the first moment; Python uses lerp, which rounds differently.
// See licenses/pytorch-LICENSE. WakeWordTrainer uses the default beta/epsilon,
// no weight decay/AMSGrad, and CUDA foreach dispatch with CPU step tensors.
class Adam {
    struct State { at::Tensor first, second, step; };
    std::vector<at::Tensor> parameters_;
    std::vector<State> state_;
public:
    double learning_rate;
    Adam(std::vector<at::Tensor> parameters, double rate) : parameters_(std::move(parameters)), state_(parameters_.size()), learning_rate(rate) {
        require(rate >= 0, "Invalid learning rate");
    }
    void zero_grad() { for (auto &parameter : parameters_) parameter.mutable_grad() = at::Tensor(); }
    void step() {
        at::NoGradGuard guard;
        std::vector<at::Tensor> parameters, gradients, first, second, steps;
        for (size_t i = 0; i < parameters_.size(); ++i) {
            const auto &parameter = parameters_[i]; if (!parameter.grad().defined()) continue;
            auto &state = state_[i];
            if (!state.step.defined()) { state.step = at::zeros({}, at::kFloat); state.first = at::zeros_like(parameter); state.second = at::zeros_like(parameter); }
            parameters.push_back(parameter); gradients.push_back(parameter.grad()); first.push_back(state.first); second.push_back(state.second); steps.push_back(state.step);
        }
        if (parameters.empty()) return;
        const double beta1 = .9, beta2 = .999, epsilon = 1e-8;
        if (parameters.front().is_cuda()) {
            at::_foreach_add_(steps, at::scalar_tensor(1.0, at::kFloat), 1.0);
            at::_foreach_lerp_(first, gradients, 1 - beta1);
            at::_foreach_mul_(second, beta2); at::_foreach_addcmul_(second, gradients, gradients, 1 - beta2);
            std::vector<at::Scalar> step_sizes, corrections;
            for (const auto &step : steps) {
                const auto n = step.item<double>();
                step_sizes.emplace_back(-(learning_rate / (1 - std::pow(beta1, n))));
                corrections.emplace_back(std::pow(1 - std::pow(beta2, n), .5));
            }
            auto denominators = at::_foreach_sqrt(second);
            at::_foreach_div_(denominators, corrections); at::_foreach_add_(denominators, epsilon);
            at::_foreach_addcdiv_(parameters, first, denominators, step_sizes);
        } else for (size_t i = 0; i < parameters.size(); ++i) {
            steps[i].add_(1); first[i].lerp_(gradients[i], 1 - beta1);
            second[i].mul_(beta2).addcmul_(gradients[i], gradients[i], 1 - beta2);
            const auto n = steps[i].item<double>();
            const auto step_size = learning_rate / (1 - std::pow(beta1, n));
            const auto correction = std::pow(1 - std::pow(beta2, n), .5);
            const auto denominator = (second[i].sqrt() / correction).add_(epsilon);
            parameters[i].addcdiv_(first[i], denominator, -step_size);
        }
    }
};
class Cnn final : public Network {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::Dropout dropout{nullptr};
public:
    explicit Cnn(const Config &config) {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).padding(1)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(64));
        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)));
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(64));
        fc1 = register_module("fc1", torch::nn::Linear(64 * (config.n_mfcc / 8) * (150 / 8), config.hidden_size));
        dropout = register_module("dropout", torch::nn::Dropout(config.dropout));
        fc2 = register_module("fc2", torch::nn::Linear(config.hidden_size, 1));
    }
    at::Tensor forward(at::Tensor x) override {
        if (x.dim() == 3) x = x.unsqueeze(1);
        x = at::max_pool2d(at::relu(bn1(conv1(x))), {2, 2}, {2, 2});
        x = at::max_pool2d(at::relu(bn2(conv2(x))), {2, 2}, {2, 2});
        x = at::max_pool2d(at::relu(bn3(conv3(x))), {2, 2}, {2, 2});
        x = dropout(at::relu(fc1(x.flatten(1))));
        return at::sigmoid(fc2(x)).squeeze(-1);
    }
};
class Gru final : public Network {
    torch::nn::GRU gru{nullptr};
    torch::nn::Linear fc{nullptr};
public:
    explicit Gru(const Config &config) {
        gru = register_module("gru", torch::nn::GRU(torch::nn::GRUOptions(config.n_mfcc, config.hidden_size)
            .num_layers(config.num_layers).batch_first(true).bidirectional(true)
            .dropout(config.num_layers > 1 ? config.dropout : 0)));
        fc = register_module("fc", torch::nn::Linear(config.hidden_size * 2, 1));
    }
    at::Tensor forward(at::Tensor x) override {
        x = std::get<0>(gru(x.transpose(1, 2))).select(1, -1);
        return at::sigmoid(fc(x)).squeeze(-1);
    }
};
std::map<std::string, at::Tensor> load_checkpoint(const std::string &path) {
    caffe2::serialize::PyTorchStreamReader archive(path);
    const auto record = archive.getRecord("data.pkl");
    const auto state = parse_torch_checkpoint_pickle(std::string_view(static_cast<const char *>(std::get<0>(record).get()), std::get<1>(record)));
    std::string byteorder = "little";
    if (archive.hasRecord("byteorder")) {
        const auto order = archive.getRecord("byteorder");
        byteorder.assign(static_cast<const char *>(std::get<0>(order).get()), std::get<1>(order));
    }
    require(byteorder == "little" || byteorder == "big", "Invalid checkpoint byte order");
    const uint16_t endian_probe = 1;
    const bool swap = (byteorder == "little") != (*reinterpret_cast<const uint8_t *>(&endian_probe) == 1);
    const std::map<std::string, at::ScalarType> types{
        {"float32", at::kFloat}, {"float64", at::kDouble}, {"float16", at::kHalf}, {"bfloat16", at::kBFloat16},
        {"int64", at::kLong}, {"int32", at::kInt}, {"int16", at::kShort}, {"int8", at::kChar},
        {"uint8", at::kByte}, {"bool", at::kBool}, {"complex64", at::kComplexFloat}, {"complex128", at::kComplexDouble}};
    std::map<std::string, at::Tensor> storages, tensors;
    for (const auto &[name, metadata] : state) {
        const auto dtype = types.at(metadata.scalar_type);
        auto storage = storages.find(metadata.storage_key);
        if (storage == storages.end()) {
            const auto size = c10::elementSize(dtype);
            require(uint64_t(metadata.storage_elements) <= std::numeric_limits<size_t>::max() / size, "Checkpoint storage overflow");
            const auto data = archive.getRecord("data/" + metadata.storage_key);
            const auto bytes = size_t(metadata.storage_elements) * size;
            require(std::get<1>(data) == bytes, "Checkpoint storage size mismatch");
            auto tensor = at::empty({metadata.storage_elements}, at::TensorOptions().dtype(dtype));
            if (bytes) std::memcpy(tensor.mutable_data_ptr(), std::get<0>(data).get(), bytes);
            if (swap) {
                const auto component_size = c10::isComplexType(dtype) ? size / 2 : size;
                auto *ptr = static_cast<uint8_t *>(tensor.mutable_data_ptr());
                for (size_t i = 0; i < bytes; i += component_size) std::reverse(ptr + i, ptr + i + component_size);
            }
            storage = storages.emplace(metadata.storage_key, std::move(tensor)).first;
        }
        require(storage->second.scalar_type() == dtype && storage->second.numel() == metadata.storage_elements,
                "Conflicting checkpoint storage metadata");
        tensors.emplace(name, at::as_strided(storage->second, metadata.shape, metadata.strides, metadata.storage_offset));
    }
    return tensors;
}
at::Tensor audio_features(const std::string &bytes, int sample_rate, int n_mfcc) {
    require(sample_rate > 0 && n_mfcc > 0 && n_mfcc <= 80, "Invalid MFCC configuration");
    auto [audio, source_rate] = soundfile_mono_float32(bytes);
    if (source_rate != sample_rate) audio = ctts_torch_resample(audio, source_rate, sample_rate);
    auto waveform = at::from_blob(audio.samples.data(), {1, int64_t(audio.samples.size())}, at::kFloat).clone();
    waveform = waveform / (waveform.abs().max() + 1e-8);
    const auto options = at::TensorOptions().dtype(at::kFloat);
    const auto window = at::hann_window(400, true, options);
    auto spectrum = at::stft(waveform, 512, 160, 400, window, true, "reflect", false, true, true).abs().pow(2);
    const auto frequencies = at::linspace(0, sample_rate / 2, 257, options);
    const double mel_max = 2595.0 * std::log10(1 + double(sample_rate / 2) / 700.0);
    const auto mel_points = at::linspace(0, mel_max, 82, options);
    const auto points = 700.0 * (at::pow(10.0, mel_points / 2595.0) - 1.0);
    const auto differences = points.slice(0, 1) - points.slice(0, 0, -1);
    const auto slopes = points.unsqueeze(0) - frequencies.unsqueeze(1);
    const auto down = -slopes.slice(1, 0, -2) / differences.slice(0, 0, -1);
    const auto up = slopes.slice(1, 2) / differences.slice(0, 1);
    const auto filterbank = at::maximum(at::zeros({1}, options), at::minimum(down, up));
    auto mel = at::matmul(spectrum.transpose(-1, -2), filterbank).transpose(-1, -2);
    mel = 10.0 * at::log10(at::clamp_min(mel, 1e-10));
    mel = at::maximum(mel, mel.amax() - 80.0);
    const double pi = std::acos(-1.0);
    auto dct = at::cos(pi / 80.0 * (at::arange(80, options) + 0.5) * at::arange(n_mfcc, options).unsqueeze(1));
    dct[0] *= 1.0 / std::sqrt(2.0); dct *= std::sqrt(2.0 / 80.0);
    return at::matmul(mel.transpose(-1, -2), dct.transpose(0, 1)).transpose(-1, -2).squeeze(0);
}
class Model final : public WakewordTrainableModel {
    Config config_;
    at::Device device_;
    std::shared_ptr<Network> network_;
public:
    Model(Config config, const std::string &device)
        : config_(std::move(config)), device_(device.empty() ? default_device() : device) {
        at::NoGradGuard guard;
        if (config_.gru) network_ = std::make_shared<Gru>(config_);
        else network_ = std::make_shared<Cnn>(config_);
        network_->to(device_);
    }
    Model(const std::string &directory, const std::string &device) : Model(Config(directory), device) {
        at::NoGradGuard guard;
        auto tensors = load_checkpoint(wakeword_contained_path(directory, "model.pt").string());
        const auto copy = [&](const auto &items) {
            for (const auto &item : items) {
                const auto found = tensors.find(item.key());
                require(found != tensors.end(), "Missing checkpoint parameter: " + item.key());
                require(found->second.sizes() == item.value().sizes(), "Checkpoint parameter shape mismatch: " + item.key());
                item.value().copy_(found->second); tensors.erase(found);
            }
        };
        copy(network_->named_parameters()); copy(network_->named_buffers());
        require(tensors.empty(), "Unexpected checkpoint parameters");
        network_->eval();
    }
    std::pair<bool, double> predict(const std::string &bytes) override {
        at::NoGradGuard guard;
        if (network_->is_training()) network_->eval();
        auto features = audio_features(bytes, config_.sample_rate, config_.n_mfcc);
        if (features.size(1) < 150) features = at::cat({features, at::zeros({features.size(0), 150 - features.size(1)}, features.options())}, 1);
        else features = features.slice(1, 0, 150);
        const auto confidence = network_->forward(features.unsqueeze(0).to(device_)).item<double>();
        return {confidence > 0.5, confidence};
    }
    void save_checkpoint(const std::string &directory) override { save_wakeword_checkpoint(directory, *network_, config_.data); }
    void export_onnx(const std::string &directory) override {
        // save() in the source draws its export example on CPU, even for a
        // CUDA trainer, before transferring it to the model's device.
        (void)at::randn({1, config_.n_mfcc, 150}).to(device_);
        // The installed Python/Torch 2.9.1 exporter cannot trace CUDA GRU's
        // cuDNN flatten-weight operation. Its save() retains only pt/config.
        // Preserve that device-dependent artifact contract, including RNG use.
        if (device_.is_cuda() && config_.gru)
            throw std::runtime_error("CUDA GRU ONNX export is unavailable in the reference backend");
        export_wakeword_onnx(directory, *network_, config_.data);
    }
    WakewordSavedFiles save(const std::string &directory) override {
        save_checkpoint(directory);
        const auto root = std::filesystem::path(directory);
        WakewordSavedFiles files{{"pytorch", (root / "model.pt").string()}, {"config", (root / "config.json").string()}};
        try { export_onnx(directory); files.emplace_back("onnx", (root / "model.onnx").string()); }
        catch (const std::exception &error) { std::cerr << "Could not export ONNX: " << error.what() << '\n'; }
        return files;
    }
    WakewordHistory train(const std::vector<std::string> &positive, const std::vector<std::string> &negative,
                         const WakewordTrainingOptions &options, const WakewordProgress &progress) override {
        if (options.check_cancel) options.check_cancel();
        const int64_t total = positive.size() + negative.size();
        require(std::isfinite(options.validation_split) && options.validation_split >= 0 && options.validation_split <= 1, "Invalid validation split");
        const auto validation_size = int64_t(total * options.validation_split), training_size = total - validation_size;
        const auto indices = at::randperm(total, at::TensorOptions().dtype(at::kLong));
        require(training_size > 0 && options.batch_size > 0, "Invalid training dataset or batch size");
        Adam optimizer(network_->parameters(), options.learning_rate);
        WakewordHistory history{{"train_loss", {}}, {"val_loss", {}}, {"train_acc", {}}, {"val_acc", {}}};
        double scheduler_best = std::numeric_limits<double>::infinity(); int bad_epochs = 0;
        for (int64_t epoch = 0; epoch < options.epochs; ++epoch) {
            if (options.check_cancel) options.check_cancel();
            WakewordMetrics metrics;
            for (bool training : {true, false}) {
                network_->train(training);
                const auto count = training ? training_size : validation_size;
                // Python DataLoader draws a base seed even for a sequential
                // validation iterator. RandomSampler then seeds a private CPU
                // generator with another draw from the global generator.
                (void)at::empty({}, at::kLong).random_().item<int64_t>();
                at::Tensor order;
                if (training) {
                    const auto seed = at::empty({}, at::kLong).random_().item<int64_t>();
                    const auto generator = at::detail::createCPUGenerator(seed);
                    order = at::randperm(count, generator, at::TensorOptions().dtype(at::kLong));
                } else order = at::arange(count, at::TensorOptions().dtype(at::kLong));
                double loss_sum = 0; int64_t correct = 0, batches = 0, seen = 0;
                const auto run = [&] {
                    for (int64_t start = 0; start < count; start += options.batch_size) {
                        if (options.check_cancel) options.check_cancel();
                        std::vector<at::Tensor> inputs; std::vector<float> labels;
                        const auto end = std::min(count, start + options.batch_size);
                        for (int64_t position = start; position < end; ++position) {
                            if (options.check_cancel) options.check_cancel();
                            const auto offset = order[position].item<int64_t>() + (training ? 0 : training_size);
                            const auto index = indices[offset].item<int64_t>();
                            const bool is_positive = index < int64_t(positive.size());
                            const auto &audio = is_positive ? positive[index] : negative[index - positive.size()];
                            auto features = audio_features(audio, config_.sample_rate, config_.n_mfcc);
                            if (features.size(1) < 150) features = at::cat({features, at::zeros({features.size(0), 150 - features.size(1)}, features.options())}, 1);
                            else features = features.slice(1, 0, 150);
                            inputs.push_back(features); labels.push_back(is_positive ? 1.0f : 0.0f);
                        }
                        const auto features = at::stack(inputs).to(device_);
                        const auto targets = at::tensor(labels, at::TensorOptions().dtype(at::kFloat)).to(device_);
                        if (training) optimizer.zero_grad();
                        const auto outputs = network_->forward(features);
                        const auto loss = at::binary_cross_entropy(outputs, targets, {}, at::Reduction::Mean);
                        if (options.check_cancel) options.check_cancel();
                        if (training) { loss.backward(); optimizer.step(); }
                        loss_sum += loss.item<double>();
                        correct += ((outputs > 0.5).to(at::kFloat) == targets).sum().item<int64_t>();
                        seen += labels.size(); ++batches;
                    }
                };
                if (training) run(); else { at::NoGradGuard guard; run(); }
                require(batches > 0 && seen > 0, "division by zero in wake-word training");
                metrics[training ? "train_loss" : "val_loss"] = loss_sum / batches;
                metrics[training ? "train_acc" : "val_acc"] = double(correct) / seen;
            }
            // Python ReduceLROnPlateau uses double metrics, relative threshold
            // 1e-4, patience 5, factor .5, no cooldown, minimum LR 0, eps 1e-8.
            if (options.check_cancel) options.check_cancel();
            const auto validation_loss = metrics.at("val_loss");
            if (validation_loss < scheduler_best * (1.0 - 1e-4)) { scheduler_best = validation_loss; bad_epochs = 0; }
            else ++bad_epochs;
            if (bad_epochs > 5) {
                const auto old = optimizer.learning_rate, next = old * .5;
                if (old - next > 1e-8) optimizer.learning_rate = next;
                bad_epochs = 0;
            }
            for (const auto &[name, value] : metrics) history.at(name).push_back(value);
            if (progress) progress(epoch + 1, options.epochs, metrics);
        }
        // Source state_dict().copy() is a shallow copy: its purported best
        // state keeps sharing live parameter/buffer storage. Restoring it is
        // a no-op, so retaining the final weights reproduces that behavior.
        return history;
    }
};
}
WakewordModelFactory wakeword_torch_factory(const std::string &device) {
    return [device](const std::string &directory) { return std::make_shared<Model>(directory, device); };
}
std::shared_ptr<WakewordTrainableModel> wakeword_torch_create(const std::string &config_json, const std::string &device) {
    return std::make_shared<Model>(Config(Json::parse(config_json)), device);
}
CttsAudioArray wakeword_audio_features(const std::string &bytes, int sample_rate, int n_mfcc) {
    at::NoGradGuard guard;
    const auto tensor = audio_features(bytes, sample_rate, n_mfcc).contiguous();
    CttsAudioArray result; result.shape.assign(tensor.sizes().begin(), tensor.sizes().end());
    result.samples.resize(tensor.numel());
    if (!result.samples.empty()) std::memcpy(result.samples.data(), tensor.const_data_ptr<float>(), result.samples.size() * sizeof(float));
    return result;
}
}
