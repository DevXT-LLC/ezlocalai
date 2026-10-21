#include "whisper_transcription_test.hpp"
#include "ezlocalai/whisper_model.hpp"
using namespace transcription_test;
struct FakeEngine {
    Json script, events = Json::array(); size_t generated = 0; int encoded = 0, detected = 0; bool anomaly = false;
    WhisperEngineOperations operations(bool multilingual) {
        WhisperEngineOperations out; out.bands = 2; out.multilingual = multilingual;
        out.encode = [this](const WhisperFeatures &features) {
            const auto bytes = std::string(reinterpret_cast<const char *>(features.values.data()),features.values.size() * sizeof(float));
            events.push_back({{"kind","encode"},{"id",++encoded},{"bands",features.bands},{"frames",features.frames},{"sha256",hash(bytes)}});
            return std::make_shared<int>(encoded);
        };
        out.generate = [this](const WhisperEncoded &value,const std::vector<uint32_t> &prompt,const ezl_whisper_options &request) {
            const auto suppressed = request.suppress_count ? std::vector<int>(request.suppress_tokens,request.suppress_tokens + request.suppress_count) : std::vector<int>{};
            events.push_back({{"kind","generate"},{"encoded",*std::static_pointer_cast<int>(value)},{"prompt",prompt},{"options",{
                {"beam_size",request.beam_size},{"patience",request.patience},{"length_penalty",request.length_penalty},{"repetition_penalty",request.repetition_penalty},
                {"no_repeat_ngram_size",request.no_repeat_ngram_size},{"max_length",request.max_length},{"sampling_topk",request.sampling_topk},
                {"sampling_temperature",request.sampling_temperature},{"num_hypotheses",request.num_hypotheses},{"max_initial_timestamp_index",request.max_initial_timestamp_index},
                {"suppress_blank",request.suppress_blank},{"suppress_tokens",suppressed}}}});
            if (generated >= script.size()) throw std::runtime_error("script exhausted"); const auto &item = script[generated++];
            return WhisperGeneration{item.at("tokens").get<std::vector<uint32_t>>(),item.at("score").get<double>(),item.at("no_speech_prob").get<double>()};
        };
        out.align = [this](const WhisperEncoded &value,const std::vector<uint32_t> &start,const std::vector<uint32_t> &tokens,size_t frames,int64_t width) {
            events.push_back({{"kind","align"},{"encoded",*std::static_pointer_cast<int>(value)},{"start",start},{"tokens",tokens},{"frames",frames},{"width",width}});
            WhisperAlignment result;
            for (size_t i = 0; i <= tokens.size(); ++i) {
                const int64_t at = std::min(std::max<int64_t>(0,static_cast<int64_t>(frames) / 2 - 1),static_cast<int64_t>((anomaly ? 50 : 0) + i * (anomaly ? 60 : 7)));
                result.indices.emplace_back(i,at); result.indices.emplace_back(i,at + 1);
                if (i < tokens.size()) result.probabilities.push_back(static_cast<float>(anomaly ? .05 : .9 - i * .002));
            }
            return result;
        };
        out.detect_language = [this](const WhisperEncoded &value) {
            events.push_back({{"kind","language"},{"encoded",*std::static_pointer_cast<int>(value)}});
            return detected++ % 2 ? WhisperLanguages{{"<|es|>",.75},{"<|en|>",.25}} : WhisperLanguages{{"<|en|>",.75},{"<|es|>",.25}};
        };
        return out;
    }
};
int main(int argc,char **argv) { try {
    expect(argc == 2,"transcription fixture directory"); const fs::path fixtures(argv[1]); Temporary directory; const auto models = tokenizers(fixtures,directory.path);
    size_t total = 0;
    for (const bool legacy : {false,true}) {
        const std::string profile = legacy ? "legacy" : "modern"; auto cases = records(fixtures/("whisper_transcription_policy_" + profile + ".jsonl"));
        expect(cases.size() == 361 && cases[0]["profile"] == profile,"complete controller fixture " + profile);
        const auto configured = records(fixtures/("whisper_preprocessor_segments_" + profile + ".jsonl")); expect(configured.size() == 41,"configured controller cases"); cases.insert(cases.end(),configured.begin() + 1,configured.end());
        for (size_t i = 1; i < cases.size(); ++i) {
            const auto &row = cases[i]; const auto label = profile + " case " + std::to_string(i - 1); const auto key = row.at("tokenizer").get<std::string>();
            const auto settings = options(row.at("options"),legacy); const auto tokenizer = models.at(key).configured(key == "multi","transcribe","en");
            FakeEngine fake; fake.script = row.at("script"); fake.anomaly = row.value("anomaly",false); auto engine = fake.operations(key == "multi");
            if (row.contains("config")) engine.feature_configuration = whisper_feature_configuration(row.at("config").dump());
            Json actual; std::string error;
            try {
                if (row["kind"] == "segments") {
                    WhisperFeatures features; features.bands = row.at("bands"); features.frames = row.at("content_frames").get<size_t>() + (legacy ? engine.feature_configuration.value_or(WhisperFeatureConfiguration{}).maximum_frames() : 1); features.values.resize(features.bands * features.frames);
                    for (size_t band = 0; band < features.bands; ++band) for (size_t frame = 0; frame < features.frames; ++frame) features.values[band * features.frames + frame] = (band % 17) * .125 + (frame % 97) * .03125;
                    actual = segments(whisper_generate_segments(features,tokenizer,settings,engine,legacy ? WhisperTranscriptionProfile::faster_whisper_1_0_3 : WhisperTranscriptionProfile::faster_whisper_1_2_0));
                } else {
                    const auto result = whisper_decode_with_fallback(std::make_shared<int>(17),row.at("prompt").get<std::vector<uint32_t>>(),tokenizer,settings,engine);
                    actual = {{"tokens",result.result.tokens},{"score",result.result.score},{"no_speech_prob",result.result.no_speech_prob},{"avg_logprob",result.avg_logprob},{"temperature",result.temperature},{"compression_ratio",result.compression_ratio}};
                }
            } catch (const std::exception &exception) { error = exception.what(); }
            compare_json(error,row.value("error",std::string{}),label + ".error");
            compare_json(fake.events,row.at("events"),label + ".events");
            if (error.empty()) compare_json(actual,row.at(row["kind"] == "segments" ? "segments" : "result"),label + ".result"); ++total;
        }
    }
    std::cout << total << " transcription controller cases match segment output and complete engine call traces\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
