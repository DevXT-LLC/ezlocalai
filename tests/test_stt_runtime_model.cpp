#include "ezlocalai/stt_whisper.hpp"
#include "whisper_transcription_test.hpp"
#include <openssl/evp.h>
#include <cblas.h>
using namespace transcription_test;
std::string encode(const std::string &input) {
    std::string result(4 * ((input.size() + 2) / 3) + 1,'\0'); const auto size = EVP_EncodeBlock(reinterpret_cast<unsigned char *>(result.data()),reinterpret_cast<const unsigned char *>(input.data()),input.size());
    expect(size >= 0,"encode test audio"); result.resize(size); return result;
}
int main(int argc,char **argv) { try {
    expect(argc == 6 || (argc == 7 && std::string(argv[6]) == "--gpu"),"fixture, English/multilingual Whisper and modern/legacy VAD directories, optional --gpu"); const auto fixtures = fs::absolute(argv[1]);
    const bool gpu = argc == 7;
    const std::map<std::string,fs::path> paths{{"en",fs::absolute(argv[2])},{"multi",fs::absolute(argv[3])}};
    const auto modern_vad = fs::absolute(argv[4]), legacy_vad = fs::absolute(argv[5]);
    Temporary temporary; fs::create_directory(temporary.path/"outputs"); size_t total = 0;
    struct RestoreDirectory { fs::path path = fs::current_path(); ~RestoreDirectory() { fs::current_path(path); } } restore;
    fs::current_path(temporary.path); setenv("HF_HUB_OFFLINE","1",1);
    for (const auto &model : paths) {
        const auto alias = model.first == "en" ? "tiny.en" : "tiny";
        const auto root = temporary.path/"models"/("models--Systran--faster-whisper-" + std::string(alias)), snapshot = root/"snapshots"/std::string(40,'a');
        fs::create_directories(snapshot); fs::create_directories(root/"refs"); std::ofstream(root/"refs/main") << std::string(40,'a');
        for (const auto &file : fs::directory_iterator(model.second)) if (file.is_regular_file()) fs::create_symlink(file.path(),snapshot/file.path().filename());
    }
    for (const bool legacy : {false,true}) {
        if (legacy && gpu) continue;
        const std::string profile = legacy ? "legacy" : gpu ? "modern_gpu" : "modern"; const auto cases = records(fixtures/("stt_runtime_model_" + profile + ".jsonl")); expect(cases.size() == 16,"complete runtime real-model fixtures");
        for (const auto &model : paths) for (const auto &item : cases[0].at("models").at(model.first).items()) expect(hash(read(model.second/item.key())) == item.value(),"pinned " + model.first + " " + item.key());
        for (const auto &item : cases[0].at("assets").items()) expect(hash(read(fixtures/"stt_runtime_assets"/item.key())) == item.value(),"pinned audio " + item.key());
        expect(cases[0].at("source_sha256") == records(fixtures/"stt_runtime_cases.jsonl").at(0).at("source_sha256"),"matching wrapper source identity");
        expect(openblas_get_num_threads() == cases[0].at("openblas_threads"),"matching reference BLAS thread count");
        expect(cases[0].at("pyaudio_version") == "0.2.14" && cases[0].at("audio_initialization") == "real PortAudio","real reference audio initialization");
        if (gpu) expect(cases[0].at("torch") == "2.9.1+cu128" && cases[0].at("compute") == "float16","real pinned GPU reference");
        for (const auto &row : rows(fixtures/"whisper_vad_model_cases.tsv")) if (row[0] == "H" && row[1] == (legacy ? "legacy" : "modern"))
            expect(hash(read((legacy ? legacy_vad : modern_vad)/row[2])) == row[3],"pinned runtime VAD model");
        std::map<std::string,std::unique_ptr<SttRuntime>> models;
        for (const auto &model : paths) {
            SttWhisperConfiguration configuration; configuration.profile = legacy ? WhisperTranscriptionProfile::faster_whisper_1_0_3 : WhisperTranscriptionProfile::faster_whisper_1_2_0; configuration.vad_directory = legacy ? legacy_vad : modern_vad;
            SttRuntimeOperations platform;
            if (!gpu) { platform.cuda_available = [] { return false; }; platform.cpu_count = [threads = cases[0].at("cpu_threads").get<int64_t>()] { return threads * 2; }; }
            platform.working_directory = [&] { return temporary.path.string(); };
            const auto name = (legacy == (model.first == "en")) ? model.second.string() : model.first == "en" ? "tiny.en" : "tiny";
            models.emplace(model.first,std::make_unique<SttRuntime>(name,stt_whisper_operations(configuration,platform)));
            expect(models.at(model.first)->has_audio(),"real PortAudio initialized by default");
            expect(models.at(model.first)->device() == (gpu ? "cuda" : "cpu"),"expected native inference device");
        }
        for (size_t i = 1; i < cases.size(); ++i) {
            const auto &row = cases[i]; const auto label = profile + " runtime case " + std::to_string(i - 1); auto request = row.at("request");
            request["base64_audio"] = encode(read(fixtures/"stt_runtime_assets"/row.at("audio").get<std::string>())); Json value, failure;
            try { value = Json::parse(models.at(row.at("model").get<std::string>())->transcribe(request.dump())); }
            catch (const SttError &error) { failure = {{"category",error.category},{"message",error.what()}}; }
            catch (const std::exception &error) { failure = {{"category","RuntimeError"},{"message",error.what()}}; }
            compare_json(failure,row.value("error",Json()),label + ".error"); if (failure.is_null()) compare_json(value,row.at("value"),label + ".value");
            if (gpu) expect(models.at(row.at("model").get<std::string>())->device() == "cuda",label + " remains on GPU");
            expect(fs::is_empty(temporary.path/"outputs"),label + " output-file cleanup"); std::cout << label << " matches\n" << std::flush; ++total;
        }
        for (auto &model : models) { model.second->close(); model.second->close(); expect(!model.second->has_model() && !model.second->has_batch() && !model.second->has_audio(),"closed native handles"); }
    }
    std::cout << total << " actual-model STT wrapper calls match exact text, JSON, errors and real temporary-file cleanup\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
