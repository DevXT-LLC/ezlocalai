#include "ezlocalai/wakeword_model.hpp"
#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
void check(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
Json read_json(const fs::path &path) { std::ifstream input(path); check(bool(input), "Missing reference: " + path.string()); return Json::parse(input); }
std::string rng_hash() {
    const auto state = at::detail::getDefaultCPUGenerator().get_state(); unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(static_cast<const unsigned char *>(state.const_data_ptr()), state.nbytes(), digest);
    std::ostringstream output; for (auto byte : digest) output << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte); return output.str();
}
std::vector<float> run(Ort::Session &session, std::vector<float> &input, int64_t batch, int64_t mfcc) {
    std::array<int64_t, 3> shape{batch, mfcc, 150};
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto value = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), shape.data(), shape.size());
    const char *inputs[] = {"input"}, *outputs[] = {"output"};
    auto result = session.Run(Ort::RunOptions{nullptr}, inputs, &value, 1, outputs, 1);
    check(result[0].GetTensorTypeAndShapeInfo().GetShape() == std::vector<int64_t>{batch}, "ONNX output shape");
    const auto *data = result[0].GetTensorData<float>(); return {data, data + batch};
}
}
int main(int argc, char **argv) {
    try {
        check(argc >= 2, "Wake-word reference root required"); const fs::path root = fs::path(argv[1]) / (argc > 3 ? argv[3] : "exports");
        const std::string device = argc > 2 ? argv[2] : "cpu";
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "wakeword-export"); Ort::SessionOptions options;
        options.SetIntraOpNumThreads(2); options.SetInterOpNumThreads(1);
        Json results = Json::array();
        for (const auto &row : read_json(root / "reference.json")) {
            const auto name = row.at("name").get<std::string>(); const auto source = root / name;
            const auto target = source / (device == "cpu" ? "native" : "native_cuda");
            const auto config = read_json(source / "config.json");
            const bool training = row.value("training", false);
            at::manual_seed(427);
            auto model = training ? ezlocalai::wakeword_torch_create(config.dump(), device) : std::dynamic_pointer_cast<ezlocalai::WakewordTrainableModel>(ezlocalai::wakeword_torch_factory(device)(source.string()));
            check(bool(model), "Trainable model missing");
            if (row.contains("rng_before")) check(rng_hash() == row.at("rng_before"), name + " pre-export RNG state");
            const auto files = model->save(target.string());
            if (row.contains("rng_after")) check(rng_hash() == row.at("rng_after"), name + " post-export RNG state");
            std::vector<std::string> formats; for (const auto &file : files) formats.push_back(file.first);
            check(Json(formats) == row.at("formats"), name + " saved format ordering");
            check(read_json(target / "config.json") == config, name + " exported configuration");
            if (formats.size() == 2) {
                check(!fs::exists(target / "model.onnx"), name + " unexpected ONNX artifact");
                results.push_back(Json{{"name", name}, {"device", device}, {"optional_onnx_failure", true}});
                std::cout << name << " optional ONNX failure retained checkpoint/config\n"; continue;
            }
            Ort::Session reference(env, (source / "model.onnx").c_str(), options), native(env, (target / "model.onnx").c_str(), options);
            double maximum = 0; size_t comparisons = 0, rejections = 0, stochastic = 0;
            for (const int64_t batch : {1, 2, 3, 8}) for (const float amplitude : {0.f, 0.1f, 1.f, 10.f}) {
                const auto mfcc = config.at("n_mfcc").get<int64_t>(); std::vector<float> input(batch * mfcc * 150);
                for (size_t i = 0; i < input.size(); ++i) input[i] = amplitude * std::sin(double(i) * 0.013);
                std::vector<float> expected, actual; bool source_error = false, native_error = false;
                try { expected = run(reference, input, batch, mfcc); } catch (const Ort::Exception &) { source_error = true; }
                try { actual = run(native, input, batch, mfcc); } catch (const Ort::Exception &) { native_error = true; }
                check(source_error == native_error, name + " batch rejection mismatch");
                if (source_error) { ++rejections; continue; }
                for (size_t i = 0; i < actual.size(); ++i) {
                    check(std::isfinite(actual[i]) && actual[i] >= 0 && actual[i] <= 1, name + " invalid probability");
                    if (training) { ++stochastic; continue; }
                    const auto error = std::abs(double(actual[i]) - expected[i]); maximum = std::max(maximum, error);
                    check(std::isfinite(actual[i]) && error <= 2e-6, name + " ONNX inference mismatch: " + std::to_string(error)); ++comparisons;
                }
            }
            results.push_back(Json{{"name", name}, {"device", device}, {"comparisons", comparisons}, {"batch_rejections", rejections}, {"stochastic_probabilities", stochastic}, {"max_absolute_error", maximum}});
            std::cout << name << " " << comparisons << " ONNX probabilities compared, " << rejections << " batch rejections, " << stochastic << " stochastic probabilities, max error " << maximum << '\n';
        }
        // Optional artifact errors must preserve mandatory checkpoint/config;
        // a checkpoint write error must still reach the caller.
        auto model = std::dynamic_pointer_cast<ezlocalai::WakewordTrainableModel>(ezlocalai::wakeword_torch_factory(device)((root / "cnn").string()));
        const auto failure = root / "native_failure"; fs::remove_all(failure); fs::create_directories(failure / "model.onnx");
        const auto files = model->save(failure.string());
        check(files.size() == 2 && fs::is_regular_file(failure / "model.pt") && fs::is_regular_file(failure / "config.json"), "Optional export failure lost checkpoint");
        fs::remove_all(failure); fs::create_directories(failure / "model.pt"); bool failed = false;
        try { model->save(failure.string()); } catch (const std::exception &) { failed = true; }
        check(failed, "Mandatory checkpoint failure was swallowed"); fs::remove_all(failure);
        std::ofstream(root / (device == "cpu" ? "native-results.json" : "native-cuda-results.json")) << results.dump(2) << '\n';
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
