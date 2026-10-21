#include "stt_api_test.hpp"
#include "ezlocalai/stt_whisper.hpp"
#include <cblas.h>
int main(int argc,char **argv) { try {
    expect(argc == 5 || argc == 6,"fixtures, English model, modern/legacy VAD directories and optional fixture prefix"); const auto fixtures = fs::absolute(argv[1]),model = fs::absolute(argv[2]),modern_vad = fs::absolute(argv[3]),legacy_vad = fs::absolute(argv[4]); const std::string prefix = argc == 6 ? argv[5] : "stt_api_model";
    Temporary temporary; struct Restore { fs::path path = fs::current_path(); ~Restore() { fs::current_path(path); } } restore; fs::current_path(temporary.path); fs::create_directory("outputs"); size_t total = 0;
    for (bool legacy : {false,true}) {
        const std::string profile = legacy ? "legacy" : "modern"; const auto cases = records(fixtures/(prefix + "_" + profile + ".jsonl")); expect(cases.size() == cases[0].at("cases").get<size_t>() + 1,"complete model API fixtures");
        expect(openblas_get_num_threads() == cases[0].at("openblas_threads"),"matching BLAS thread count");
        for (const auto &item : cases[0].at("models").at("en").items()) expect(hash(read(model/item.key())) == item.value(),"pinned API model " + item.key());
        for (const auto &item : cases[0].at("assets").items()) expect(hash(read(fixtures/"stt_runtime_assets"/item.key())) == item.value(),"pinned API audio " + item.key());
        expect(cases[0].at("api_sha256") == records(fixtures/"stt_api_cases.jsonl")[0].at("source_sha256"),"matching API source");
        expect(cases[0].at("pool_sha256") == records(fixtures/"stt_api_cases.jsonl")[0].at("pool_sha256"),"matching pool/guard source");
        expect(cases[0].at("source_sha256") == records(fixtures/"stt_runtime_cases.jsonl")[0].at("source_sha256"),"matching STT source");
        for (const auto &row : rows(fixtures/"whisper_vad_model_cases.tsv")) if (row[0] == "H" && row[1] == profile) expect(hash(read((legacy ? legacy_vad : modern_vad)/row[2])) == row[3],"pinned API VAD model");
        for (size_t i = 1; i < cases.size(); ++i) {
            Environment env; env.config = cases[i].at("config"); env.real = true; SttWhisperConfiguration config; config.profile = legacy ? WhisperTranscriptionProfile::faster_whisper_1_0_3 : WhisperTranscriptionProfile::faster_whisper_1_2_0;
            config.vad_directory = legacy ? legacy_vad : modern_vad; config.resolve_model = [model](const auto &,const auto &) { return model.string(); };
            SttRuntimeOperations platform; platform.cuda_available = [] { return false; }; platform.cpu_count = [] { return 8; };
            env.native = stt_whisper_pool_operations(config,std::make_shared<ResourceManager>(),platform);
            total += run_api_case(env,cases[i],profile + " actual API " + std::to_string(i),fixtures); expect(fs::is_empty(temporary.path/"outputs"),"API audio-file cleanup");
            std::cout << profile << " API lifecycle " << i << " matches\n" << std::flush;
        }
    }
    std::cout << total << " native HTTP requests with actual models match\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
