#include "stt_pipeline_test.hpp"
#include "ezlocalai/stt_whisper.hpp"
#include <cblas.h>
int main(int argc,char **argv) { try {
    expect(argc == 5,"fixtures, English model and modern/legacy VAD directories"); const auto fixtures = fs::absolute(argv[1]),model = fs::absolute(argv[2]),modern_vad = fs::absolute(argv[3]),legacy_vad = fs::absolute(argv[4]);
    Temporary temporary; struct Restore { fs::path path = fs::current_path(); ~Restore() { fs::current_path(path); } } restore; fs::current_path(temporary.path); fs::create_directory("outputs"); size_t total = 0;
    for (bool legacy : {false,true}) {
        const std::string profile = legacy ? "legacy" : "modern"; const auto cases = records(fixtures/("stt_pipeline_model_" + profile + ".jsonl")); expect(cases.size() == cases[0].at("cases").get<size_t>()+1,"complete pipeline model fixtures");
        expect(openblas_get_num_threads() == cases[0].at("openblas_threads"),"matching pipeline BLAS threads");
        expect(cases[0].at("pool_sha256") == records(fixtures/"stt_pipeline_cases.jsonl")[0].at("source_sha256"),"matching pipeline source");
        expect(cases[0].at("source_sha256") == records(fixtures/"stt_runtime_cases.jsonl")[0].at("source_sha256"),"matching STT source");
        for (const auto &item : cases[0].at("models").at("en").items()) expect(hash(read(model/item.key())) == item.value(),"pinned pipeline model");
        for (const auto &item : cases[0].at("assets").items()) expect(hash(read(fixtures/"stt_runtime_assets"/item.key())) == item.value(),"pinned pipeline audio");
        for (const auto &row : rows(fixtures/"whisper_vad_model_cases.tsv")) if (row[0] == "H" && row[1] == profile) expect(hash(read((legacy ? legacy_vad : modern_vad)/row[2])) == row[3],"pinned pipeline VAD model");
        for (size_t i = 1; i < cases.size(); ++i) {
            Environment env; env.config = cases[i].at("config"); env.real = true; SttWhisperConfiguration config; config.profile = legacy ? WhisperTranscriptionProfile::faster_whisper_1_0_3 : WhisperTranscriptionProfile::faster_whisper_1_2_0; config.vad_directory = legacy ? legacy_vad : modern_vad; config.resolve_model = [model](const auto &,const auto &) { return model.string(); };
            SttRuntimeOperations platform; platform.cuda_available = [] { return false; }; platform.cpu_count = [] { return 8; }; env.native = stt_whisper_pool_operations(config,std::make_shared<ResourceManager>(),platform);
            total += run_pipeline_case(env,cases[i],profile + " pipeline " + std::to_string(i),fixtures); expect(fs::is_empty(temporary.path/"outputs"),"pipeline audio cleanup");
            std::cout << profile << " pipeline " << i << " matches\n" << std::flush;
        }
    }
    std::cout << total << " real-model pipeline calls match\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
