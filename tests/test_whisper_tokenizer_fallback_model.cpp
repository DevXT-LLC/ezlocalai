#include "ezlocalai/stt_whisper.hpp"
#include "whisper_transcription_test.hpp"
#include <openssl/evp.h>
#include <cblas.h>
using namespace transcription_test;
int main(int argc,char **argv) { try {
    expect(argc == 6,"fixtures, English/multilingual Whisper and modern/legacy VAD directories");
    const auto fixtures = fs::absolute(argv[1]); const std::map<std::string,fs::path> paths{{"en",fs::absolute(argv[2])},{"multi",fs::absolute(argv[3])}};
    const auto modern_vad = fs::absolute(argv[4]), legacy_vad = fs::absolute(argv[5]); Temporary temporary;
    fs::create_directory(temporary.path/"outputs"); const auto cache = temporary.path/"tokenizer-cache"; fs::create_directory(cache);
    struct RestoreDirectory { fs::path path = fs::current_path(); ~RestoreDirectory() { fs::current_path(path); } } restore;
    fs::current_path(temporary.path); setenv("HF_HUB_OFFLINE","1",1); setenv("HF_HUB_CACHE",cache.c_str(),1);
    const auto audio = read(fixtures/"stt_runtime_assets/speech.wav"); std::string encoded(4 * ((audio.size() + 2) / 3) + 1,'\0');
    const auto length = EVP_EncodeBlock(reinterpret_cast<unsigned char *>(encoded.data()),reinterpret_cast<const unsigned char *>(audio.data()),audio.size()); expect(length >= 0,"base64 audio"); encoded.resize(length);
    size_t constructors = 0, calls = 0;
    for (const bool legacy : {false,true}) {
        const std::string profile = legacy ? "legacy" : "modern"; const auto cases = records(fixtures/("whisper_tokenizer_fallback_model_" + profile + ".jsonl")); expect(cases.size() == 5,"complete fallback real-model fixtures");
        expect(cases[0].at("source_sha256") == records(fixtures/"stt_runtime_cases.jsonl")[0].at("source_sha256"),"matching STT source");
        expect(openblas_get_num_threads() == cases[0].at("openblas_threads"),"matching BLAS threads");
        for (const auto &item : cases[0].at("assets").items()) expect(hash(read(fixtures/"stt_runtime_assets"/item.key())) == item.value(),"pinned audio");
        for (const auto &model : paths) {
            for (const auto &item : cases[0].at("models").at(model.first).items()) expect(hash(read(model.second/item.key())) == item.value(),"pinned model " + item.key());
            const auto &meta = cases[0].at("fallback_tokenizers").at(model.first); const auto compressed = read(fixtures/"whisper_tokenizer_fallback_assets"/meta.at("file").get<std::string>());
            const auto size = meta.at("size").get<size_t>(); std::string raw(size + 1,'\0'); uLongf count = raw.size();
            expect(uncompress(reinterpret_cast<Bytef *>(raw.data()),&count,reinterpret_cast<const Bytef *>(compressed.data()),compressed.size()) == Z_OK && count == size,"decompress fallback tokenizer");
            raw.resize(size); expect(hash(raw) == meta.at("sha256"),"pinned fallback tokenizer");
            const auto root = cache/("models--openai--whisper-tiny" + std::string(model.first == "en" ? ".en" : ""));
            const auto snapshot = root/"snapshots"/meta.at("revision").get<std::string>(); fs::create_directories(snapshot); fs::create_directories(root/"refs");
            std::ofstream(root/"refs/main") << meta.at("revision").get<std::string>(); std::ofstream(snapshot/"tokenizer.json",std::ios::binary).write(raw.data(),raw.size());
        }
        for (const auto &row : rows(fixtures/"whisper_vad_model_cases.tsv")) if (row[0] == "H" && row[1] == profile)
            expect(hash(read((legacy ? legacy_vad : modern_vad)/row[2])) == row[3],"pinned VAD model");
        for (size_t i = 1; i < cases.size(); ++i) {
            const auto &row = cases[i]; const auto key = row.at("model").get<std::string>(), kind = row.at("kind").get<std::string>();
            const auto label = profile + " " + key + " tokenizer " + kind; const auto folder = temporary.path/(profile + "-" + key + "-" + kind); fs::create_directory(folder);
            for (const auto &name : {"model.bin","config.json","vocabulary.txt"}) fs::create_symlink(paths.at(key)/name,folder/name);
            if (kind == "directory") fs::create_directory(folder/"tokenizer.json");
            SttWhisperConfiguration configuration; configuration.profile = legacy ? WhisperTranscriptionProfile::faster_whisper_1_0_3 : WhisperTranscriptionProfile::faster_whisper_1_2_0; configuration.vad_directory = legacy ? legacy_vad : modern_vad;
            SttRuntimeOperations platform; platform.cuda_available = [] { return false; }; platform.cpu_count = [threads = cases[0].at("cpu_threads").get<int64_t>()] { return threads * 2; };
            platform.working_directory = [&] { return temporary.path.string(); };
            SttRuntime model(folder.string(),stt_whisper_operations(configuration,platform)); ++constructors;
            expect(model.device() == "cpu" && model.has_audio(),label + " real CPU/PortAudio");
            for (const auto &call : row.at("calls")) {
                auto request = call.at("request"); request["base64_audio"] = encoded; Json value,failure;
                try { value = Json::parse(model.transcribe(request.dump())); }
                catch (const SttError &error) { failure = {{"category",error.category},{"message",error.what()}}; }
                catch (const std::exception &error) { failure = {{"category","RuntimeError"},{"message",error.what()}}; }
                compare_json(failure,call.value("error",Json()),label + ".error"); if (failure.is_null()) compare_json(value,call.at("value"),label + ".value");
                expect(fs::is_empty(temporary.path/"outputs"),label + " audio cleanup"); ++calls;
            }
            model.close(); expect(!model.has_model() && !model.has_audio(),label + " close");
            expect(!fs::is_regular_file(folder/"tokenizer.json"),label + " local tokenizer remains absent");
            expect(!fs::exists(temporary.path/"models/models--openai--whisper-tiny") && !fs::exists(temporary.path/"models/models--openai--whisper-tiny.en"),"fallback uses default Hub cache independently of model root");
            std::cout << label << " matches\n" << std::flush;
        }
    }
    std::cout << constructors << " fallback constructors and " << calls << " real-model wrapper calls match\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
