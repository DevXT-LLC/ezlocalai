#include "ezlocalai/whisper_model.hpp"
#include "ezlocalai/stt_runtime.hpp"
#include "whisper_transcription_test.hpp"
using namespace transcription_test;
Json configuration(const WhisperFeatureConfiguration &value) {
    return {{"feature_size",value.feature_size},{"sampling_rate",value.sampling_rate},{"hop_length",value.hop_length},{"chunk_length",value.chunk_length},{"n_fft",value.n_fft}};
}
int main(int argc,char **argv) { try {
    expect(argc == 2,"fixture directory"); const fs::path fixtures(argv[1]); Temporary temporary; size_t files = 0, feature_cases = 0; double maximum_error = 0;
    for (const bool legacy : {false,true}) {
        const std::string profile = legacy ? "legacy" : "modern"; const auto rows = records(fixtures/("whisper_preprocessor_" + profile + ".jsonl")); expect(rows.size() == 33,"all preprocessor loading cases");
        for (size_t i = 1; i < rows.size(); ++i) {
            const auto &row = rows[i]; const auto path = temporary.path/"preprocessor_config.json"; fs::remove_all(path);
            if (row.at("kind") == "directory") fs::create_directory(path);
            else if (row.at("kind") == "file") { const auto bytes = unhex(row.at("hex")); std::ofstream(path,std::ios::binary).write(bytes.data(),bytes.size()); }
            Json failure, kwargs, config;
            try { kwargs = Json::parse(whisper_feature_kwargs(temporary.path)); config = configuration(whisper_feature_configuration(kwargs.dump())); }
            catch (const SttError &error) { failure = {{"category",error.category},{"message",error.what()}}; }
            const auto label = profile + " file " + std::to_string(i - 1);
            if (row.contains("kwargs")) compare_json(kwargs,row.at("kwargs"),label + ".kwargs");
            compare_json(failure,row.value("error",Json()),label + ".error"); if (failure.is_null()) compare_json(config,row.at("configuration"),label + ".configuration"); ++files;
        }
        const auto audio = packed(fixtures/"whisper_preprocessor_assets/audio.f32z",16003); const auto features = records(fixtures/("whisper_preprocessor_features_" + profile + ".jsonl")); expect(features.size() == 10,"all configured features");
        for (size_t i = 0; i < features.size(); ++i) {
            const auto &row = features[i]; const auto config = whisper_feature_configuration(row.at("config").dump()); Json failure; WhisperFeatures actual;
            try { actual = whisper_log_mel(audio,config.feature_size,legacy ? config.chunk_length * config.sampling_rate : 160,config.sampling_rate,config.n_fft,config.hop_length,legacy ? WhisperFeatureProfile::faster_whisper_1_0_3 : WhisperFeatureProfile::faster_whisper_1_2_0); }
            catch (const std::invalid_argument &error) { failure = {{"category","ValueError"},{"message",error.what()}}; }
            const auto label = profile + " features " + std::to_string(i); compare_json(failure,row.value("error",Json()),label + ".error");
            if (failure.is_null()) {
                expect(actual.bands == row.at("bands") && actual.frames == row.at("frames"),label + " shape");
                const auto expected = packed(fixtures/"whisper_preprocessor_assets"/row.at("file").get<std::string>(),actual.values.size());
                expect(hash(std::string(reinterpret_cast<const char *>(expected.data()),expected.size()*sizeof(float))) == row.at("sha256"),label + " golden identity");
                double error = 0; for (size_t n = 0; n < expected.size(); ++n) error = std::max(error,std::abs(double(actual.values[n]) - expected[n])); maximum_error = std::max(maximum_error,error);
                expect(error <= 2e-7,label + " feature error " + std::to_string(error));
            }
            ++feature_cases;
        }
    }
    std::cout << files << " source preprocessor loading cases and " << feature_cases << " configured feature cases pass; maximum error " << maximum_error << '\n'; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
