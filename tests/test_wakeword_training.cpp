#include "ezlocalai/wakeword_model.hpp"
#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
void check(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
std::string read(const fs::path &path) { std::ifstream input(path, std::ios::binary); check(bool(input), "Missing evidence: " + path.string()); return {std::istreambuf_iterator<char>(input), {}}; }
std::string hash(const std::string &bytes) {
    unsigned char digest[SHA256_DIGEST_LENGTH]; SHA256(reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size(), digest);
    std::ostringstream output; for (auto byte : digest) output << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte); return output.str();
}
std::string rng() {
    const auto tensor = at::detail::getDefaultCPUGenerator().get_state();
    return hash(std::string(static_cast<const char *>(tensor.const_data_ptr()), tensor.nbytes()));
}
}
int main(int argc, char **argv) {
    try {
        check(argc >= 2, "Training reference root required"); const fs::path root(argv[1]);
        const std::string mode = argc > 2 ? argv[2] : "cpu";
        const bool deterministic = mode == "cuda-deterministic", capture = mode == "cuda-capture";
        const std::string device = mode == "cpu" ? "cpu" : "cuda";
        if (deterministic) { at::globalContext().setDeterministicAlgorithms(true, false); at::globalContext().setDeterministicCuDNN(true); }
        const auto training = root / (device == "cpu" ? "training" : deterministic ? "training_cuda_deterministic" : "training_cuda");
        const auto fixture = Json::parse(read(training / "reference.json")); Json actual = Json::array();
        for (const auto &row : fixture.at("cases")) {
            const auto name = row.at("name").get<std::string>(); const auto folder = training / name;
            check(hash(read(folder / "initial.pt")) == row.at("initial_sha256"), name + " initial reference hash");
            check(hash(read(folder / "model.pt")) == row.at("model_sha256"), name + " trained reference hash");
            at::manual_seed(row.at("seed").get<uint64_t>());
            auto model = ezlocalai::wakeword_torch_create(row.at("config").dump(), device);
            check(rng() == row.at("initial_rng"), name + " initialization RNG state");
            model->save_checkpoint((folder / "native_initial").string());
            ezlocalai::WakewordTrainingOptions options;
            options.epochs = row.at("epochs"); options.batch_size = row.at("batch_size"); options.learning_rate = row.at("learning_rate"); options.validation_split = row.at("validation_split");
            std::vector<std::string> positive, negative;
            for (const auto &path : row.at("positive")) positive.push_back(read(root / path.get<std::string>()));
            for (const auto &path : row.at("negative")) negative.push_back(read(root / path.get<std::string>()));
            Json progress = Json::array();
            const auto history = model->train(positive, negative, options, [&](int64_t epoch, int64_t total, const auto &metrics) {
                progress.push_back(Json{{"epoch", epoch}, {"total", total}, {"metrics", metrics}});
            });
            const auto final_rng = rng();
            model->save_checkpoint((folder / "native_final").string());
            const auto prediction = model->predict(read(root / "sine.wav"));
            actual.push_back(Json{{"name", name}, {"history", history}, {"progress", progress}, {"final_rng", final_rng}, {"prediction", Json::array({prediction.first, prediction.second})}});
            std::ofstream(training / "native-results.json") << actual.dump(2);
            check(final_rng == row.at("final_rng"), name + " training RNG state");
            for (const auto &[metric, values] : history) {
                const auto &expected = row.at("history").at(metric); check(values.size() == expected.size(), name + " history length");
                for (size_t i = 0; i < values.size(); ++i)
                    check(capture || std::abs(values[i] - expected.at(i).get<double>()) < 2e-6, name + " " + metric + " epoch " + std::to_string(i + 1) + ": " + std::to_string(values[i]) + " expected " + expected.at(i).dump());
            }
            check(progress.size() == row.at("progress").size(), name + " progress callback count");
            for (size_t i = 0; i < progress.size(); ++i) {
                check(progress[i].at("epoch") == row.at("progress")[i].at("epoch") && progress[i].at("total") == row.at("progress")[i].at("total"), name + " progress ordering");
            }
            check(capture || (prediction.first == row.at("prediction")[0] && std::abs(prediction.second - row.at("prediction")[1].get<double>()) < 2e-6), name + " trained prediction");
            const auto loaded = ezlocalai::wakeword_torch_factory(device)((folder / "native_final").string());
            check(loaded->predict(read(root / "sine.wav")) == prediction, name + " native checkpoint reload");
            std::cout << name << (capture ? " nondeterministic training captured; RNG and checkpoint reload checked\n" : " training, RNG, callback and checkpoint reload matched\n");
        }
        // Exercise the real tensor trainer, not just the job adapter: stop
        // before work and during the first epoch, before any epoch callback.
        struct Cancelled {};
        for (int boundary : {1, 3, 10}) {
            auto model = ezlocalai::wakeword_torch_create(fixture.at("cases").at(0).at("config").dump(), device);
            ezlocalai::WakewordTrainingOptions options;options.epochs=20;options.batch_size=1;
            int checks=0,epochs=0;bool cancelled=false;
            options.check_cancel=[&]{if(++checks==boundary)throw Cancelled{};};
            const std::vector<std::string> samples(4,read(root/"sine.wav"));
            try{model->train(samples,samples,options,[&](auto,auto,const auto &){++epochs;});}
            catch(const Cancelled &){cancelled=true;}
            check(cancelled&&checks==boundary&&epochs==0,"tensor training cancels within first epoch");
        }
        std::cout << "Tensor training cancellation before work and within a batch passed on " << device << '\n';
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
