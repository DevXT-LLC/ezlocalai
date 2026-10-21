#include "ezlocalai/whisper_features.hpp"
#include "ezlocalai/whisper_kernel.h"
#include "ezlocalai/whisper_vad.hpp"
#include "ezlocalai/whisper_tokenizer.hpp"
#include "whisper_test_data.hpp"
#include <cmath>
#include <iostream>
using namespace whisper_test;
using namespace ezlocalai;
struct Result { std::vector<uint64_t> tokens; float score = 0, no_speech = 0; size_t calls = 0; };
void result(const uint64_t *tokens,size_t count,float score,float no_speech,void *user) {
    auto &out = *static_cast<Result *>(user); out.tokens.assign(tokens,tokens + count);
    out.score = score; out.no_speech = no_speech; ++out.calls;
}
int main(int argc,char **argv) { try {
    expect(argc == 5,"speech fixture, Whisper model and two VAD model directories");
    const fs::path fixture(argv[1]); const auto cases = rows(fixture);
    expect(!cases.empty() && cases[0].size() == 3 && cases[0][0] == "H","speech model identity fixture");
    expect(hash(read(fs::path(argv[2])/"model.bin")) == cases[0][1],"pinned speech model weights");
    expect(hash(read(fs::path(argv[2])/"tokenizer.json")) == cases[0][2],"pinned speech tokenizer");
    const WhisperTokenizer tokenizer((fs::path(argv[2])/"tokenizer.json").string(),false);
    for (const auto &row : rows(fixture.parent_path()/"whisper_vad_model_cases.tsv")) if (row[0] == "H")
        expect(hash(read(fs::path(row[1] == "modern" ? argv[3] : argv[4])/row[2])) == row[3],"pinned VAD model bytes");
    WhisperVadModel modern(WhisperVadProfile::faster_whisper_1_2_0,argv[3]);
    WhisperVadModel legacy(WhisperVadProfile::faster_whisper_1_0_3,argv[4]);
    char error[4096]{};
    struct Handle { void *value; ~Handle() { ezl_whisper_kernel_destroy(value); } }
        model{ezl_whisper_kernel_create(argv[2],"cpu","int8",4,error,sizeof(error))};
    expect(model.value,std::string("create Whisper: ") + error);
    size_t count = 0;
    for (const auto &row : cases) {
        if (row[0] == "H") continue;
        expect(row[0] == "S" && row.size() == 16,"speech fixture columns"); ++count;
        const auto label = row[1] + " " + row[2] + " beam " + row[7]; const bool old = row[1] == "legacy";
        auto options = whisper_vad_options(old ? WhisperVadProfile::faster_whisper_1_0_3 : WhisperVadProfile::faster_whisper_1_2_0);
        options.min_silence_duration_ms = 500; options.speech_pad_ms = 200;
        const auto audio = packed(fixture.parent_path()/"whisper_vad_assets"/row[2],std::stoul(row[3]));
        const auto filtered = (old ? legacy : modern).filter(audio,options);
        const auto chunks = integers<int64_t>(row[13]); expect(chunks.size() == filtered.chunks.size() * 2,label + " kept chunk count");
        for (size_t i = 0; i < filtered.chunks.size(); ++i)
            expect(filtered.chunks[i].start == chunks[i * 2] && filtered.chunks[i].end == chunks[i * 2 + 1],label + " kept chunk boundaries");
        expect(filtered.audio.size() == std::stoul(row[4]),label + " filtered audio length");
        const auto bytes = filtered.audio.empty() ? std::string{} : std::string(reinterpret_cast<const char *>(filtered.audio.data()),filtered.audio.size() * sizeof(float));
        expect(hash(bytes) == row[5],label + " exact filtered audio samples");
        const auto features = old ? whisper_log_mel_legacy(filtered.audio) : whisper_log_mel(filtered.audio);
        expect(features.frames == std::stoul(row[6]),label + " feature frames");
        const auto padded = whisper_feature_window(features,0,features.frames);
        const auto tokens = whisper_prompt(tokenizer,tokenizer.encode(row[15]));
        const std::vector<uint64_t> prompt(tokens.begin(),tokens.end()); const auto expected = integers<uint64_t>(row[10]);
        const auto suppressed = whisper_suppressed_tokens(tokenizer,std::vector<int>{-1},old ? WhisperTokenizerProfile::faster_whisper_1_0_3 : WhisperTokenizerProfile::faster_whisper_1_2_0).value();
        expect(prompt == integers<uint64_t>(row[8]) && suppressed == integers<int>(row[9]),label + " native prompt/suppression policy");
        ezl_whisper_options request; request.beam_size = std::stoul(row[7]); request.suppress_tokens = suppressed.data(); request.suppress_count = suppressed.size();
        Result actual;
        expect(!ezl_whisper_kernel_generate(model.value,padded.values.data(),padded.bands,padded.frames,prompt.data(),prompt.size(),&request,result,&actual,error,sizeof(error)),label + " generation: " + error);
        if (actual.calls != 1 || actual.tokens != expected) {
            std::cerr << label << " actual tokens:"; for (auto token : actual.tokens) std::cerr << ' ' << token;
            std::cerr << "\nexpected:"; for (auto token : expected) std::cerr << ' ' << token; std::cerr << '\n';
            throw std::runtime_error(label + " decoder token mismatch");
        }
        expect(std::abs(actual.score - std::stof(row[11])) <= 0.0001f && std::abs(actual.no_speech - std::stof(row[12])) <= 0.0001f,label + " score/no-speech probability");
        expect(tokenizer.decode(std::vector<uint32_t>(actual.tokens.begin(),actual.tokens.end())) == row[14],label + " native decoded text");
    }
    expect(count == 16,"complete VAD-to-Whisper cases");
    std::cout << count << " native VAD/filter/features/tokenizer/CTranslate2 cases match reference audio, token and text outputs\n"; return 0;
} catch (const std::exception &exception) { std::cerr << exception.what() << '\n'; return 1; } }
