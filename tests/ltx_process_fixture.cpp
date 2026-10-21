// Synthetic engine process: tests the worker boundary, not model inference.
#define main wan_fixture_main
#include "wan_process_fixture.cpp"
#undef main
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <unistd.h>
extern char **environ;
struct sd_ctx_params_t { bool eager_load = false; int ltx_offload_mode = 0; };
#define EZLOCALAI_LTX_OFFLOAD_POLICY 1
struct sd_ctx_t { int id; };
int context_sequence = 0;
bool sd_set_ltx_offload_mode(sd_ctx_t*,int mode) { return mode >= 0 && mode <= 2; }
sd_ctx_t *new_sd_ctx(const sd_ctx_params_t *) {
    std::ofstream("contexts.jsonl", std::ios::app) << nlohmann::json{{"event","load"},{"pid",getpid()}} << '\n';
    return new sd_ctx_t{++context_sequence};
}
void free_sd_ctx(sd_ctx_t *context) {
    if (context) std::ofstream("contexts.jsonl", std::ios::app) << nlohmann::json{{"event","free"},{"pid",getpid()}} << '\n';
    delete context;
}
void sd_set_log_callback(void *,void *) {} void sd_set_progress_callback(void *,void *) {}
enum preview_t { PREVIEW_NONE }; void sd_set_preview_callback(void *,preview_t,int,bool,bool,void *) {}
#include "sd_cli_worker.hpp"
void refresh_controls() {
    std::ifstream file(".ltx-fixture-controls.json"); if (!file) return;
    nlohmann::json controls; file >> controls; std::vector<std::string> keys;
    for (char **p=environ; *p; ++p) { const std::string entry=*p; const auto key=entry.substr(0,entry.find('='));
        if(key.rfind("WAN_FIXTURE_",0)==0 || key.rfind("LTX_FIXTURE_",0)==0) keys.push_back(key); }
    for(const auto &key:keys)unsetenv(key.c_str());
    for(const auto &[key,value]:controls.items())setenv(key.c_str(),value.get<std::string>().c_str(),1);
}
namespace {
void le(std::ofstream &out, uint32_t n, int bytes) { for (int i = 0; i < bytes; ++i) out.put(static_cast<char>(n >> (8 * i))); }
}
int generation(int argc, char **argv) {
    refresh_controls();
    using Json = nlohmann::json;
    std::map<std::string,std::string> args;
    for (int i=1;i+1<argc;++i) args[argv[i]]=argv[i+1];
    Json capture = {{"args", std::vector<std::string>(argv + 1, argv + argc)},
        {"pageable", std::getenv("GGML_CUDA_NO_PINNED") ? std::getenv("GGML_CUDA_NO_PINNED") : ""}};
    if (args.count("--ltx-conditions")) {
        capture["conditions"]=Json::parse(args["--ltx-conditions"]);
        for (auto &c:capture["conditions"]) c["exists"]=std::filesystem::is_regular_file(c["image"].get<std::string>());
    }
    int attempt=0;
    const bool preload=ezlocalai_sd_worker::preload;
    capture["pid"]=getpid();
    if (const char *file=preload ? "preload.jsonl" : std::getenv("LTX_FIXTURE_CAPTURE")) {
        std::ifstream in(file); for(std::string line;std::getline(in,line);) ++attempt;
        std::ofstream out(file,std::ios::app);out<<capture.dump()<<'\n';
    }
    if (const char *fail=std::getenv("LTX_FIXTURE_OOM_COUNT");!preload && fail && attempt<std::stoi(fail)) {std::cerr<<"CUDA out of memory\n";return 7;}
    if (const char *fail=std::getenv("LTX_FIXTURE_FAIL_AT");!preload && fail && attempt==std::stoi(fail)) {std::cerr<<"Fixture scene failure\n";return 7;}
    if (std::getenv("LTX_FIXTURE_NF4_FAIL") && args.count("--ltx-nf4-sm-count")) {
        std::cerr << "LTX NF4 conversion failed for fixture: allocation failure\n";
        return 7;
    }
    if (preload && std::getenv("LTX_FIXTURE_PRELOAD_FAIL")) { std::cerr << "Fixture preload failure\n"; return 7; }
    if (!preload && std::getenv("LTX_FIXTURE_MALFORMED_REPLY")) { ezlocalai::media_wire::send(3, "{"); return 0; }
    if (preload && std::getenv("LTX_FIXTURE_PLACEMENT_FAIL")) {
        const std::string mode=std::getenv("LTX_FIXTURE_PLACEMENT_FAIL");
        const bool offload=std::find(argv,argv+argc,std::string("--offload-to-cpu"))!=argv+argc;
        if ((!offload && mode.find("full")!=std::string::npos) || (offload && args["--ltx-offload-mode"]=="1" && mode.find("model_offload")!=std::string::npos)) {
            std::cerr << "CUDA out of memory during placement\n"; return 7;
        }
    }
    auto context=ezlocalai_sd_worker::acquire({false,args.count("--ltx-offload-mode")?std::stoi(args["--ltx-offload-mode"]):0});
    if (preload) {
        if (const char* after=std::getenv("LTX_FIXTURE_AFTER_MEMORY")) {
            const char* inventory=std::getenv("EZLOCALAI_TEST_CUDA_DEVICES");
            if (!inventory) throw std::runtime_error("missing CUDA fixture inventory");
            std::ofstream(inventory) << after;
        }
        return 0;
    }
    const int code=wan_fixture_main(argc,argv);if(code)return code;
    if (std::getenv("LTX_FIXTURE_NO_AUDIO")) return 0;
    auto path=std::filesystem::path(args["-o"]);path.replace_extension(".wav");
    const int rate=48000,channels=2,samples=15840,bytes=samples*channels*2;
    std::ofstream out(path,std::ios::binary);out<<"RIFF";le(out,36+bytes,4);out<<"WAVEfmt ";le(out,16,4);le(out,1,2);le(out,channels,2);le(out,rate,4);le(out,rate*channels*2,4);le(out,channels*2,2);le(out,16,2);out<<"data";le(out,bytes,4);
    for(int i=0;i<samples*channels;++i)le(out,(i%200)*20-2000,2);
    return out ? 0:81;
}

int main(int argc,char **argv) {
    if(argc==5 && std::string(argv[1])=="--ezlocalai-worker-fd")
        return ezlocalai_sd_worker::serve(std::stoi(argv[2]),std::stoi(argv[4]),[](int n,const char **v){
            std::vector<char *> args;for(int i=0;i<n;++i)args.push_back(const_cast<char *>(v[i]));args.push_back(nullptr);return generation(n,args.data());
        });
    return generation(argc,argv);
}
