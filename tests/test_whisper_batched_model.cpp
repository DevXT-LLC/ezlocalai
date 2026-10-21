#include "whisper_transcription_test.hpp"
#include <cblas.h>
using namespace transcription_test;
int main(int argc,char **argv) { try {
    expect(argc == 6,"fixture, English/multilingual Whisper, modern VAD directories and BLAS thread count"); const fs::path fixtures(argv[1]);
    const std::map<std::string,fs::path> paths{{"en",argv[2]},{"multi",argv[3]}}; const auto cases = records(fixtures/(std::string("whisper_batched_model_threads_") + argv[5] + ".jsonl"));
    expect(cases.size() == 26,"complete batched real-model fixtures");
    expect(cases[0].at("openblas_threads") == openblas_get_num_threads(),"native and reference BLAS thread counts must match");
    for (const auto &model : paths) for (const auto &item : cases[0].at("models").at(model.first).items()) expect(hash(read(model.second/item.key())) == item.value(),"pinned " + model.first + " " + item.key());
    for (const auto &row : rows(fixtures/"whisper_vad_model_cases.tsv")) if (row[0] == "H" && row[1] == "modern") expect(hash(read(fs::path(argv[4])/row[2])) == row[3],"pinned VAD weights");
    std::map<std::string,WhisperBatchedPipeline> pipelines; std::map<std::string,WhisperTokenizer> tokenizers;
    for (const auto &model : paths) { pipelines.emplace(model.first,WhisperBatchedPipeline(WhisperNativeEngine(model.second.string()).operations())); tokenizers.emplace(model.first,WhisperTokenizer((model.second/"tokenizer.json").string(),model.first == "multi")); }
    WhisperVadModel vad_model(WhisperVadProfile::faster_whisper_1_2_0,argv[4]);
    for (size_t i = 1; i < cases.size(); ++i) {
        const auto &row = cases[i]; const auto label = "batch real case " + std::to_string(i - 1); const auto key = row.at("model").get<std::string>();
        const auto audio = packed(fixtures/"whisper_transcription_assets"/row.at("audio").get<std::string>(),row.at("samples"));
        const auto settings = batch_options(row.at("options")); WhisperTranscription actual; std::string error;
        try { actual = pipelines.at(key).transcribe(audio,tokenizers.at(key),settings,[&](const std::vector<float> &input,const WhisperVadOptions &vad) { return vad_model.speech_timestamps(input,vad); }); }
        catch (const std::exception &exception) { error = exception.what(); }
        compare_json(error,row.value("error",std::string{}),label + ".error");
        if (error.empty()) { compare_json(segments(actual.segments),row.at("segments"),label + ".segments",1e-4); compare_json(info(actual),row.at("info"),label + ".info",1e-4); }
        compare_json(pipelines.at(key).last_speech_timestamp(),row.at("last_speech"),label + ".last_speech"); std::cout << label << " matches\n" << std::flush;
    }
    std::cout << cases.size() - 1 << " native batched transcriptions match tokens, text, times, language and alignment state\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
