#include "stt_api_test.hpp"
#include "fixture_ports.hpp"
#include <csignal>
std::string encode_audio(const std::string &bytes) { std::string out(4*((bytes.size()+2)/3)+1,'\0'); const auto size = EVP_EncodeBlock(reinterpret_cast<unsigned char *>(out.data()),reinterpret_cast<const unsigned char *>(bytes.data()),bytes.size()); expect(size >= 0,"encode HTTP media"); out.resize(size); return out; }
void setup(const Json &settings,const fs::path &folder,int port) {
    for (const auto &entry : environment_defaults) setenv(std::string(entry.first).c_str(),std::string(entry.second).c_str(),1);
    for (const auto *key : {"LLAMA_CPP_SERVER_URL","LLAMA_CPP_CONTEXT","LLAMA_CPP_MODEL_PATHS","LLAMA_CPP_EXTRA_ARGS","LLM_FIXTURE_INFERENCE_RESULT","LLM_FIXTURE_INFERENCE_FILE","LLM_FIXTURE_FAIL_LAUNCH"}) unsetenv(key);
    for (const auto &entry : std::vector<std::pair<std::string,std::string>>{{"EZLOCALAI_DOTENV_PATH",""},{"EZLOCALAI_API_KEY",""},{"DEFAULT_MODEL","fixture-text"},{"WHISPER_MODEL","fixture-model"},{"GPU_LAYERS","0"},{"LLM_MAX_TOKENS","512"},{"LLM_PROMPT_CACHE_MIB","0"},{"SPECULATIVE_TYPE","none"},{"N_PARALLEL","2"},{"STT_N_PARALLEL","2"},{"WHISPER_CPP_BIN","/nonexistent-whisper-cli"},{"WHISPER_CPP_CLI","/nonexistent-whisper-cli"},{"LLAMA_CPP_SERVER_BIN",TEST_EMBEDDING_FIXTURE_BIN},{"LLAMA_CPP_SERVER_PORT",std::to_string(port)},{"LLAMA_CPP_MODEL_PATH",(folder/"model.gguf").string()},{"EMBEDDING_FIXTURE_DIR",folder.string()},{"LLAMA_CPP_START_TIMEOUT","3"},{"LLAMA_CPP_MMPROJ_PATH","projector.gguf"},{"ROUTER_OUTPUTS_DIR",(folder/"outputs").string()},{"VIDEO_MAX_FRAMES","4"},{"STT_WAIT_FOR_LLM_IDLE_TIMEOUT","0"}}) setenv(entry.first.c_str(),entry.second.c_str(),1);
    setenv("VOICE_SERVER",settings.at("voice_mode").get<bool>() ? "true" : "",1); setenv("LAZY_LOAD_VOICE",settings.at("preload").get<bool>() ? "false" : "true",1); setenv("STT_UNLOAD_LLM_DURING_GENERATION",settings.at("handoff").get<bool>() ? "true" : "false",1);
}
std::string request_body(const std::string &kind,const std::string &video) {
    if (kind == "completion") return R"({"prompt":"YQ==","audio_format":"wav"})";
    if (kind == "audio_format") return R"({"messages":[{"role":"user","content":"YQ=="}],"audio_format":"wav"})";
    const bool is_video = kind.find("video") != std::string::npos; const auto value = is_video ? video : "data:audio/wav;base64,YQ=="; const auto key = is_video ? "video_url" : "audio_url";
    Json message{{"role","user"}};
    message["content"] = Json::array({"before",Json{{key,value}},"after"});
    return Json{{"messages",Json::array({message})}}.dump();
}
int main(int argc,char **argv) { try {
    expect(argc == 3,"source and fixtures"); const auto fixtures = fs::absolute(argv[2]); const auto cases = records(fixtures/"stt_pipeline_http_cases.jsonl"); expect(hash(read(fs::path(argv[1])/"Pipes.py")) == cases[0].at("source_sha256"),"unchanged HTTP pipeline source"); expect(cases.size() == cases[0].at("cases").get<size_t>()+1,"complete HTTP cases");
    Temporary temporary; struct Restore { fs::path path = fs::current_path(); ~Restore() { fs::current_path(path); } } restore; fs::current_path(temporary.path);
    const auto video = "data:video/mp4;base64," + encode_audio(read(fixtures/"llm_video_assets/audio.mkv")); size_t total = 0;
    for (size_t i = 1; i < cases.size(); ++i) {
        const auto &scenario = cases[i]; const std::string kind = scenario.at("kind"); const bool is_video = kind.find("video") != std::string::npos; const auto folder = temporary.path/std::to_string(i); fs::create_directory(folder); std::ofstream(folder/"model.gguf") << "fixture";
        const auto port = fixture_ports(); const auto suffix = '-' + std::to_string(port); Environment env; env.config = scenario.at("config"); env.real = true; setup(env.config,folder,port);
        auto probes = native_resource_probes(); probes.gpu_count = [] { return 0; }; probes.total_per_gpu = [] { return std::vector<double>{}; }; probes.system_ram = [] { return 16.; }; probes.now = [] { return 100.; }; auto resources = std::make_shared<ResourceManager>(probes);
        std::vector<std::pair<bool,bool>> loaded_at_inference; bool expected_loaded = false;
        {
            ezl_env_config config{}; ezl_load_config(&config); config.request_timeout_ms = 15000;
            NativeServer server(config,std::make_unique<LlamaCppInferenceBackend>(config.default_model,resources),resources,[&](auto tracker) {
                auto ops = env.factory(tracker); auto load = ops.load; ops.load = [&,load](const auto &name) { auto model = load(name); auto infer = model->transcribe; model->transcribe = [&,infer](const auto &request) { loaded_at_inference.emplace_back(fs::exists(folder/("launch"+suffix)),expected_loaded); return infer(request); }; return model; }; return ops;
            });
            size_t step = 0;
            for (const auto &action : scenario.at("actions")) {
                if (action.value("recover",false)) { env.config.erase("infer_error"); env.config.erase("load_error"); }
                const auto &expected = scenario.at("outcomes")[step]; const auto before = fs::exists(folder/("seen"+suffix)) ? read(folder/("seen"+suffix)).size() : 0;
                expected_loaded = fs::exists(folder/("launch"+suffix)) || kind == "audio_format" || kind == "completion";
                const auto response = server.handle_request("POST",kind == "completion" ? "/v1/completions" : "/v1/chat/completions","",request_body(kind,video));
                const auto label = "managed pipeline " + std::to_string(i) + " step " + std::to_string(step); const bool failed = expected.contains("error");
                expect(response.status == (failed && !is_video ? 500 : 200),label + " status " + response.body);
                if (failed && !is_video) expect(response.body == R"({"detail":"Internal server error"})",label + " error body");
                compare_json(env.events,expected.at("events"),label + " events"); compare_json(env.state(server,*resources),expected.at("state"),label + " pool/resources");
                const auto seen = fs::exists(folder/("seen"+suffix)) ? read(folder/("seen"+suffix)).substr(before) : "";
                if (failed && !is_video) expect(seen.empty(),label + " error skips engine");
                else if (is_video) expect((seen.find("Video Audio Transcript") != std::string::npos) == !failed,label + " video transcript omitted on failure");
                else expect(seen.find("hello world") != std::string::npos,label + " transcript reaches engine");
                ++step; ++total;
            }
            for (const auto &phase : loaded_at_inference) expect(phase.first == phase.second,"source STT/LLM load phase " + kind);
            expect(server.handle_request("POST","/v1/chat/completions","",R"({"messages":[{"role":"user","content":"recovered"}]})").status == 200,"LLM slot recovery");
            ++total;
        }
        for (auto &work : env.pending) work();
        std::istringstream pids(read(folder/("launch"+suffix))); for (int pid; pids >> pid;) expect(kill(pid,0) != 0,"managed child reaped");
        expect(!fs::exists(folder/"outputs") || fs::is_empty(folder/"outputs"),"no pipeline audio artifacts");
    }
    // A get_response lease is already active when STT requests a shared-GPU
    // handoff. The source's zero-timeout case must fail before STT loads.
    for (const std::string kind : {"audio_format","video_inline"}) {
        const auto folder = temporary.path/("handoff-"+kind); fs::create_directory(folder); std::ofstream(folder/"model.gguf") << "fixture"; Environment env; env.config = {{"capacity",2},{"preload",false},{"voice_mode",false},{"handoff",true}}; setup(env.config,folder,fixture_ports());
        ezl_env_config config{}; ezl_load_config(&config); NativeServer server(config,std::make_unique<LlamaCppInferenceBackend>(config.default_model),{},[&](auto tracker) { return env.factory(tracker); });
        for (const auto &ignored : cases[0].at("ignored")) {
            expect(server.handle_request("POST","/v1/chat/completions","",ignored.at("body").dump()).status == 200,"source ignores audio/video fields outside list content");
            expect(env.events.empty(),"ignored fields never acquire STT"); ++total;
        }
        const auto response = server.handle_request("POST","/v1/chat/completions","",request_body(kind,video)); expect(response.status == (kind == "audio_format" ? 500 : 200),"active-LLM handoff status"); expect(env.events.empty(),"handoff timeout precedes STT load");
        expect(server.handle_request("POST","/v1/chat/completions","",R"({"messages":[{"role":"user","content":"recovered"}]})").status == 200,"handoff restores LLM admission"); total += 2;
    }
    std::cout << total << " managed LLM HTTP requests preserve native STT lifecycle, load order, video failure tolerance and handoff recovery\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
