#include "whisper_transcription_test.hpp"
using namespace transcription_test;
int main(int argc,char **argv) { try {
    expect(argc == 6,"fixture, English/multilingual Whisper and modern/legacy VAD directories"); const fs::path fixtures(argv[1]);
    const std::map<std::string,fs::path> paths{{"en",argv[2]},{"multi",argv[3]}};
    for (const auto &profile : {"modern","legacy"}) {
        const auto header = records(fixtures/(std::string("whisper_transcription_model_") + profile + ".jsonl")).at(0);
        for (const auto &model : paths) for (const auto &item : header.at("models").at(model.first).items()) expect(hash(read(model.second/item.key())) == item.value(),"pinned " + model.first + " " + item.key());
    }
    for (const auto &row : rows(fixtures/"whisper_vad_model_cases.tsv")) if (row[0] == "H") expect(hash(read(fs::path(row[1] == "modern" ? argv[4] : argv[5])/row[2])) == row[3],"pinned VAD weights");
    std::map<std::string,WhisperNativeEngine> engines; std::map<std::string,WhisperTokenizer> tokenizers;
    for (const auto &model : paths) { engines.emplace(model.first,WhisperNativeEngine(model.second.string())); tokenizers.emplace(model.first,WhisperTokenizer((model.second/"tokenizer.json").string(),model.first == "multi")); }
    WhisperVadModel modern(WhisperVadProfile::faster_whisper_1_2_0,argv[4]), legacy_model(WhisperVadProfile::faster_whisper_1_0_3,argv[5]);
    size_t total = 0;
    for (const bool legacy : {false,true}) {
        const std::string profile = legacy ? "legacy" : "modern"; const auto cases = records(fixtures/("whisper_transcription_model_" + profile + ".jsonl")); expect(cases.size() == 17,"complete real-model fixtures");
        for (size_t i = 1; i < cases.size(); ++i) {
            const auto &row = cases[i]; const auto label = profile + " real case " + std::to_string(i - 1); const auto key = row.at("model").get<std::string>();
            const auto audio = packed(fixtures/"whisper_transcription_assets"/row.at("audio").get<std::string>(),row.at("samples"));
            const auto settings = options(row.at("options"),legacy); WhisperTranscription actual; std::string error;
            try { actual = whisper_transcribe(audio,tokenizers.at(key),settings,engines.at(key).operations(),legacy ? WhisperTranscriptionProfile::faster_whisper_1_0_3 : WhisperTranscriptionProfile::faster_whisper_1_2_0,
                [&](const std::vector<float> &input,const WhisperVadOptions &vad) { return (legacy ? legacy_model : modern).filter(input,vad); }); }
            catch (const std::exception &exception) { error = exception.what(); }
            compare_json(error,row.value("error",std::string{}),label + ".error");
            if (error.empty()) { compare_json(segments(actual.segments),row.at("segments"),label + ".segments",1e-4); compare_json(info(actual),row.at("info"),label + ".info",1e-4); }
            std::cout << label << " matches\n" << std::flush; ++total;
        }
    }
    std::cout << total << " native full transcription cases match exact tokens/text/timestamps and reference scores\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
