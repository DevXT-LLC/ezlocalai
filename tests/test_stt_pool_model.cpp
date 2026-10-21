#include "stt_pool_test.hpp"
#include "ezlocalai/stt_whisper.hpp"
#include <cblas.h>
int main(int argc,char **argv) { try {
    expect(argc == 5,"fixtures, Whisper English, modern/legacy VAD directories");
    const auto fixtures = fs::absolute(argv[1]), model = fs::absolute(argv[2]);
    const auto modern_vad = fs::absolute(argv[3]), legacy_vad = fs::absolute(argv[4]); Temporary temporary;
    struct Restore { fs::path path = fs::current_path(); ~Restore() { fs::current_path(path); } } restore;
    fs::current_path(temporary.path); fs::create_directory("outputs"); size_t actions = 0;
    for (bool legacy : {false,true}) {
        const std::string profile = legacy ? "legacy" : "modern"; const auto cases = records(fixtures/("stt_pool_model_" + profile + ".jsonl")); expect(cases.size() == 2,"complete model pool fixture");
        for (const auto &item : cases[0].at("models").at("en").items()) expect(hash(read(model/item.key())) == item.value(),"pinned pool model " + item.key());
        for (const auto &item : cases[0].at("assets").items()) expect(hash(read(fixtures/"stt_runtime_assets"/item.key())) == item.value(),"pinned pool audio " + item.key());
        expect(cases[0].at("pool_sha256") == records(fixtures/"stt_pool_cases.jsonl")[0].at("sha256"),"matching pool source");
        expect(cases[0].at("source_sha256") == records(fixtures/"stt_runtime_cases.jsonl")[0].at("source_sha256"),"matching STT source");
        expect(openblas_get_num_threads() == cases[0].at("openblas_threads"),"matching reference BLAS threads");
        expect(cases[0].at("audio_initialization") == "real PortAudio","reference native audio lifetime");
        for (const auto &row : rows(fixtures/"whisper_vad_model_cases.tsv")) if (row[0] == "H" && row[1] == profile) expect(hash(read((legacy ? legacy_vad : modern_vad)/row[2])) == row[3],"pinned pool VAD");
        Environment env(cases[1].at("config")); SttWhisperConfiguration config;
        config.profile = legacy ? WhisperTranscriptionProfile::faster_whisper_1_0_3 : WhisperTranscriptionProfile::faster_whisper_1_2_0;
        config.vad_directory = legacy ? legacy_vad : modern_vad;
        config.resolve_model = [model](const auto &,const auto &) { return model.string(); };
        SttRuntimeOperations platform; platform.cuda_available = [] { return false; }; platform.cpu_count = [] { return 8; };
        env.native = stt_whisper_pool_operations(config,env.resources,platform);
        SttPool pool("fixture-model",2,env.operations()); actions += run_pool_scenario(env,pool,cases[1],profile + " native model pool",fixtures);
        expect(fs::is_empty(temporary.path/"outputs"),"pool temporary audio cleanup");
        std::cout << profile << " actual pool lifecycle and inference match\n" << std::flush;
    }
    std::cout << actions << " actual-model pool actions match\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
