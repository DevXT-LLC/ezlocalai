#include "ezlocalai/stt_whisper.hpp"
#include "whisper_transcription_test.hpp"
#include <openssl/evp.h>
#include <cblas.h>
using namespace transcription_test;
int main(int argc,char **argv) { try {
    expect(argc == 5,"fixture, English Whisper and modern/legacy VAD directories");
    const auto fixtures = fs::absolute(argv[1]), model_path = fs::absolute(argv[2]);
    const auto modern_vad = fs::absolute(argv[3]), legacy_vad = fs::absolute(argv[4]);
    Temporary temporary; fs::create_directory(temporary.path/"outputs"); const auto folder = temporary.path/"model"; fs::create_directory(folder);
    for (const auto &name : {"model.bin","config.json","tokenizer.json","vocabulary.txt"}) fs::create_symlink(model_path/name,folder/name);
    struct RestoreDirectory { fs::path path = fs::current_path(); ~RestoreDirectory() { fs::current_path(path); } } restore;
    fs::current_path(temporary.path);
    const auto audio = read(fixtures/"stt_runtime_assets/speech.wav"); std::string encoded(4 * ((audio.size() + 2) / 3) + 1,'\0');
    const auto size = EVP_EncodeBlock(reinterpret_cast<unsigned char *>(encoded.data()),reinterpret_cast<const unsigned char *>(audio.data()),audio.size());
    expect(size >= 0,"encode reference audio"); encoded.resize(size); size_t total = 0, constructors = 0;
    for (const bool legacy : {false,true}) {
        const std::string profile = legacy ? "legacy" : "modern"; const auto cases = records(fixtures/("whisper_preprocessor_model_" + profile + ".jsonl"));
        expect(cases.size() == 10,"complete configured model fixtures");
        for (const auto &item : cases[0].at("models").at("en").items()) expect(hash(read(model_path/item.key())) == item.value(),"pinned model " + item.key());
        for (const auto &item : cases[0].at("assets").items()) expect(hash(read(fixtures/"stt_runtime_assets"/item.key())) == item.value(),"pinned audio " + item.key());
        expect(cases[0].at("source_sha256") == records(fixtures/"stt_runtime_cases.jsonl")[0].at("source_sha256"),"matching wrapper source identity");
        expect(openblas_get_num_threads() == cases[0].at("openblas_threads"),"reference BLAS thread count");
        for (const auto &row : rows(fixtures/"whisper_vad_model_cases.tsv")) if (row[0] == "H" && row[1] == profile)
            expect(hash(read((legacy ? legacy_vad : modern_vad)/row[2])) == row[3],"pinned VAD model");
        for (size_t i = 1; i < cases.size(); ++i) {
            const auto &row = cases[i]; const auto label = profile + " configured model " + std::to_string(i - 1);
            std::ofstream(folder/"preprocessor_config.json") << row.at("config").get<std::string>();
            SttWhisperConfiguration configuration; configuration.profile = legacy ? WhisperTranscriptionProfile::faster_whisper_1_0_3 : WhisperTranscriptionProfile::faster_whisper_1_2_0;
            configuration.vad_directory = legacy ? legacy_vad : modern_vad;
            SttRuntimeOperations platform; platform.cuda_available = [] { return false; };
            platform.cpu_count = [threads = cases[0].at("cpu_threads").get<int64_t>()] { return threads * 2; };
            platform.working_directory = [&] { return temporary.path.string(); };
            Json failure; std::unique_ptr<SttRuntime> model;
            try { model = std::make_unique<SttRuntime>(folder.string(),stt_whisper_operations(configuration,platform)); }
            catch (const SttError &error) { failure = {{"category",error.category},{"message",error.what()}}; }
            catch (const std::exception &error) { failure = {{"category","RuntimeError"},{"message",error.what()}}; }
            compare_json(failure,row.value("error",Json()),label + ".constructor"); ++constructors;
            if (!model) continue;
            expect(model->device() == "cpu" && model->has_audio(),label + " native CPU and PortAudio handles");
            for (const auto &call : row.at("calls")) {
                auto request = call.at("request"); request["base64_audio"] = encoded; Json value; failure = Json();
                try { value = Json::parse(model->transcribe(request.dump())); }
                catch (const SttError &error) { failure = {{"category",error.category},{"message",error.what()}}; }
                catch (const std::exception &error) { failure = {{"category","RuntimeError"},{"message",error.what()}}; }
                compare_json(failure,call.value("error",Json()),label + ".call.error");
                if (failure.is_null()) compare_json(value,call.at("value"),label + ".call.value");
                expect(fs::is_empty(temporary.path/"outputs"),label + " output cleanup"); ++total;
            }
            model->close(); expect(!model->has_model() && !model->has_audio(),label + " close");
            std::cout << label << " matches\n" << std::flush;
        }
    }
    std::cout << constructors << " configured constructors and " << total << " real-model calls match reference output and errors\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
