#include "whisper_transcription_test.hpp"
#include "ezlocalai/whisper_model.hpp"
using namespace transcription_test;
struct BatchHandle { int id; size_t count; };
struct FakeBatch {
    Json script, events = Json::array(); size_t generated = 0, calls = 0; int encoded = 0, detected = 0; bool anomaly = false, fail = false;
    WhisperEngineOperations operations(bool multilingual) {
        WhisperEngineOperations out; out.bands = 2; out.multilingual = multilingual;
        out.encode_batch = [this](const std::vector<WhisperFeatures> &features) {
            std::string bytes; for (const auto &feature : features) bytes.append(reinterpret_cast<const char *>(feature.values.data()),feature.values.size() * sizeof(float));
            events.push_back({{"kind","encode_batch"},{"id",++encoded},{"batch",features.size()},{"bands",features.at(0).bands},{"frames",features.at(0).frames},{"sha256",hash(bytes)}});
            return std::make_shared<BatchHandle>(BatchHandle{encoded,features.size()});
        };
        out.generate_batch = [this](const WhisperEncoded &value,const std::vector<std::vector<uint32_t>> &prompts,const ezl_whisper_options &request) {
            const auto suppressed = request.suppress_count ? std::vector<int>(request.suppress_tokens,request.suppress_tokens + request.suppress_count) : std::vector<int>{};
            events.push_back({{"kind","generate_batch"},{"encoded",std::static_pointer_cast<BatchHandle>(value)->id},{"prompts",prompts},{"options",{
                {"beam_size",request.beam_size},{"patience",request.patience},{"length_penalty",request.length_penalty},{"repetition_penalty",request.repetition_penalty},
                {"no_repeat_ngram_size",request.no_repeat_ngram_size},{"max_length",request.max_length},{"sampling_topk",request.sampling_topk},
                {"sampling_temperature",request.sampling_temperature},{"num_hypotheses",request.num_hypotheses},{"max_initial_timestamp_index",request.max_initial_timestamp_index},
                {"suppress_blank",request.suppress_blank},{"suppress_tokens",suppressed}}}});
            if (++calls == 2 && fail) throw std::runtime_error("scripted batch failure");
            std::vector<WhisperGeneration> results;
            for (size_t i = 0; i < prompts.size(); ++i) {
                if (generated >= script.size()) throw std::runtime_error("script exhausted"); const auto &item = script[generated++];
                results.push_back({item.at("tokens").get<std::vector<uint32_t>>(),item.at("score").get<double>(),item.at("no_speech_prob").get<double>()});
            }
            return results;
        };
        out.align_batch = [this](const WhisperEncoded &value,const std::vector<uint32_t> &start,const std::vector<std::vector<uint32_t>> &tokens,const std::vector<size_t> &frames,int64_t width) {
            events.push_back({{"kind","align_batch"},{"encoded",std::static_pointer_cast<BatchHandle>(value)->id},{"start",start},{"tokens",tokens},{"frames",frames},{"width",width}});
            std::vector<WhisperAlignment> results;
            for (size_t item = 0; item < tokens.size(); ++item) {
                WhisperAlignment result;
                for (size_t i = 0; i <= tokens[item].size(); ++i) {
                    const int64_t at = std::min(std::max<int64_t>(0,static_cast<int64_t>(frames[item]) / 2 - 1),static_cast<int64_t>((anomaly ? 50 : 0) + i * (anomaly ? 60 : 7)));
                    result.indices.emplace_back(i,at); result.indices.emplace_back(i,at + 1);
                    if (i < tokens[item].size()) result.probabilities.push_back(static_cast<float>(anomaly ? .05 : .9 - i * .002));
                }
                results.push_back(std::move(result));
            }
            return results;
        };
        out.detect_language_batch = [this](const WhisperEncoded &value) {
            const auto batch = std::static_pointer_cast<BatchHandle>(value); events.push_back({{"kind","language_batch"},{"encoded",batch->id}});
            std::vector<WhisperLanguages> results;
            for (size_t i = 0; i < batch->count; ++i) results.push_back(detected++ % 2 ? WhisperLanguages{{"<|es|>",.75},{"<|en|>",.25}} : WhisperLanguages{{"<|en|>",.75},{"<|es|>",.25}});
            return results;
        };
        return out;
    }
};
int main(int argc,char **argv) { try {
    expect(argc == 2,"batched transcription fixture directory"); const fs::path fixtures(argv[1]); Temporary directory; const auto models = tokenizers(fixtures,directory.path);
    auto cases = records(fixtures/"whisper_batched_policy.jsonl"); expect(cases.size() == 301 && cases[0]["version"] == "1.2.0","complete batch policy fixture"); size_t calls = 0;
    const auto configured = records(fixtures/"whisper_preprocessor_batch.jsonl"); expect(configured.size() == 31,"configured batch cases"); cases.insert(cases.end(),configured.begin() + 1,configured.end());
    for (size_t row_index = 1; row_index < cases.size(); ++row_index) {
        const auto &row = cases[row_index]; const auto label = "batch case " + std::to_string(row_index - 1); const auto key = row.at("tokenizer").get<std::string>();
        const auto settings = options(row.at("options"),false); const auto tokenizer = models.at(key).configured(key == "multi","transcribe","en");
        FakeBatch fake; fake.script = row.at("script"); fake.anomaly = row.at("anomaly"); fake.fail = row.at("fail"); auto engine = fake.operations(key == "multi");
        if (row.contains("config")) engine.feature_configuration = whisper_feature_configuration(row.at("config").dump());
        WhisperBatchedPipeline pipeline(engine);
        std::vector<WhisperFeatures> features(row.at("count").get<size_t>()); std::vector<WhisperAudioChunk> chunks;
        for (size_t i = 0; i < features.size(); ++i) {
            auto &feature = features[i]; feature.bands = row.at("bands"); feature.frames = row.at("frames"); feature.values.resize(feature.bands * feature.frames);
            for (size_t band = 0; band < feature.bands; ++band) for (size_t frame = 0; frame < feature.frames; ++frame) feature.values[band * feature.frames + frame] = (band % 17) * .125 + (frame % 97) * .03125 + i * .25;
            WhisperAudioChunk chunk; chunk.offset = row.at("chunks")[i].at("offset"); chunk.duration = row.at("chunks")[i].at("duration"); chunks.push_back(std::move(chunk));
        }
        for (const auto &call : row.at("calls")) {
            Json actual; std::string error;
            try { actual = segments(pipeline.generate_segments(features,tokenizer,chunks,settings,row.at("batch_size"))); }
            catch (const std::exception &exception) { error = exception.what(); }
            compare_json(error,call.value("error",std::string{}),label + ".error");
            if (error.empty()) compare_json(actual,call.at("segments"),label + ".segments");
            compare_json(pipeline.last_speech_timestamp(),call.at("last_speech"),label + ".last_speech"); ++calls;
        }
        compare_json(fake.events,row.at("events"),label + ".events");
    }
    std::cout << cases.size() - 1 << " batch policy cases, " << calls << " calls, match segments, engine traces and persistent alignment state\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
