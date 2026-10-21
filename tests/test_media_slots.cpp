#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/wan_video.hpp"
#include <nlohmann/json.hpp>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <unistd.h>
namespace {
using namespace ezlocalai;
using Json = nlohmann::json;
using namespace std::chrono_literals;
namespace fs = std::filesystem;
void check(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
struct Gate {
    std::mutex mutex; std::condition_variable cv; bool entered = false, open = false;
    void wait() { std::unique_lock<std::mutex> lock(mutex); entered = true; cv.notify_all(); check(cv.wait_for(lock, 15s, [&]{ return open; }), "gate timeout"); }
    void ready() { std::unique_lock<std::mutex> lock(mutex); check(cv.wait_for(lock, 5s, [&]{ return entered; }), "gate not entered"); }
    void release() { std::lock_guard<std::mutex> lock(mutex); open = true; cv.notify_all(); }
};
struct Backend : InferenceBackend {
    std::atomic<bool> loaded{true}; Gate *llm = nullptr, *unload = nullptr, *restore = nullptr; bool fail_restore = false, replicated = false;
    std::optional<std::vector<LlmReplica>> llm_replica_state() const override {
        if (!replicated) return std::nullopt;
        return std::vector<LlmReplica>{{"fixture-text", "fixture-text", 1, 65536, {}, false, false, loaded}, {"fixture-text#2", "fixture-text", 1, 65536, {}, false, false, loaded}};
    }
    std::string chat(const std::string &) override { if (llm) llm->wait(); return "{}"; }
    std::string completion(const std::string &s) override { return chat(s); }
    std::string embedding(const std::string &) override { return "{}"; }
    std::string loaded_models_json() const override { return loaded ? R"({"llm":{"name":"fixture-text","device":"cpu"}})" : "{}"; }
    std::function<void(bool)> unload_llms_for_video() override {
        if (unload) unload->wait(); loaded = false;
        return [this](bool) { if (restore) restore->wait(); if (fail_restore) throw std::runtime_error("fixture restore failure"); loaded = true; };
    }
};
void environment(const Json &row) {
    for (const auto &entry : environment_defaults) unsetenv(std::string(entry.first).c_str());
    for (const auto &[key,value] : row.at("environment").items()) setenv(key.c_str(),value.get<std::string>().c_str(),1);
    setenv("ROUTER_URL","",1); setenv("EZLOCALAI_API_KEY","",1); setenv("LAZY_LOAD_VOICE","true",1);
    setenv("ACE_STEP_SERVER_URL","http://127.0.0.1:1",1);
    setenv("SDCPP_BIN",TEST_WAN_FIXTURE_BIN,1); setenv("HF_HUB_OFFLINE","1",1);
    setenv("HF_HUB_CACHE",(fs::current_path()/"models").c_str(),1);
}
std::shared_ptr<ResourceManager> manager(double free) {
    auto probes = native_resource_probes(); probes.gpu_count=[] { return 1; };
    probes.total_per_gpu=[] { return std::vector<double>{24}; }; probes.free_per_gpu=[free] { return std::vector<double>{free}; };
    probes.free_vram=[free](auto) { return free; }; probes.allocator_stats=[] { return std::vector<CudaAllocatorStats>{}; };
    probes.system_ram=probes.available_ram=[] { return 16.; }; return std::make_shared<ResourceManager>(probes);
}
Json slots(NativeServer &server) {
    auto response=server.handle_request("GET","/v1/resources","",""); check(response.status==200,"resources status: "+response.body);
    return Json::parse(response.body).at("slots");
}
void compare(NativeServer &server,const Json &row) { auto actual=slots(server); check(actual==row.at("slots"),row.at("name").get<std::string>()+": actual="+actual.dump()+" expected="+row.at("slots").dump()); }
void static_cases(const Json &rows) {
    for (const auto &row:rows) {
        environment(row); ezl_env_config config{}; ezl_load_config(&config);
        auto resource=manager(row.at("free_gb")); auto backend=std::make_unique<Backend>(); backend->loaded=row.at("loaded").get<bool>();
        Gate llm; backend->llm=&llm;
        NativeServer server(config,std::move(backend),resource);
        if(row.at("image_loaded").get<bool>()) resource->register_model(ModelType::Image,"fixture-image","cpu",0);
        auto counts=row.at("counts");
        for(int i=0;i<counts[0].get<int>();++i) resource->mark_model_in_use(ModelType::Image,true);
        std::future<HttpResponse> request;
        struct Release { Gate &gate; ~Release(){gate.release();} } release{llm};
        if(row.at("llm_active").get<int>()) { request=std::async(std::launch::async,[&]{ return server.handle_request("POST","/v1/chat/completions","",R"({"model":"fixture-text","messages":[]})"); }); llm.ready(); }
        compare(server,row); llm.release(); if(request.valid()) check(request.get().status==200,"chat completes");
    }
}
const Json &named(const Json &rows,const std::string &name) { for(const auto &row:rows) if(row.at("name")==name) return row; throw std::runtime_error("missing "+name); }
void wait_file(const fs::path &path) { auto deadline=std::chrono::steady_clock::now()+5s; while(!fs::exists(path)&&std::chrono::steady_clock::now()<deadline) std::this_thread::sleep_for(5ms); check(fs::exists(path),"fixture process not entered"); }
void session(const Json &rows,bool keep,bool handoff,bool failed=false,bool fail_generation=false,bool replicated=false) {
    const std::string prefix=std::string(replicated?"replicas-":"")+(keep?"true":"false")+"-"+(handoff?"true":"false")+"-";
    environment(named(rows,prefix+"idle"));
    fs::remove("wan-entered"); fs::remove("wan-release");
    setenv("WAN_FIXTURE_GATE","wan",1);
    if(fail_generation) setenv("WAN_FIXTURE_EXIT","1",1); else unsetenv("WAN_FIXTURE_EXIT");
    ezl_env_config config{}; ezl_load_config(&config);
    Gate llm,unload,restore; auto backend=std::make_unique<Backend>(); backend->llm=&llm; backend->unload=&unload; backend->restore=&restore; backend->fail_restore=failed; backend->replicated=replicated;
    NativeServer server(config,std::move(backend),manager(0));
    compare(server,named(rows,prefix+"idle"));
    auto chat=std::async(std::launch::async,[&]{return server.handle_request("POST","/v1/chat/completions","",R"({"model":"fixture-text","messages":[]})");});
    llm.ready();
    auto request=std::async(std::launch::async,[&]{return server.handle_request("POST","/v1/videos/generations","",R"({"prompt":"test","size":"64x64","num_frames":5})");});
    struct Release { Gate &a,&b,&c; ~Release(){ a.release(); b.release(); c.release(); std::ofstream("wan-release")<<"release"; } } release{llm,unload,restore};
    if(handoff) {
        auto deadline=std::chrono::steady_clock::now()+5s;
        while(slots(server).at("cap_slots").at("text").at("capacity")!=0 && std::chrono::steady_clock::now()<deadline) std::this_thread::sleep_for(5ms);
    } else wait_file("wan-entered");
    compare(server,named(rows,prefix+"waiting"));
    llm.release(); check(chat.get().status==200,"concurrent chat completed");
    if(handoff) { unload.ready(); compare(server,named(rows,prefix+"unloading")); unload.release(); }
    wait_file("wan-entered"); compare(server,named(rows,prefix+"generation"));
    const auto busy_resources=Json::parse(server.handle_request("GET","/v1/resources","","").body);
    const auto &busy_video=busy_resources.at("loaded_models").at("video");
    check(busy_video.at("vram_gb")==0 && busy_video.at("in_use")==true && busy_video.at("active_requests")==1,"Wan wrapper registration tracks the active generation");
    std::ofstream("wan-release")<<"release";
    if(handoff) { restore.ready(); compare(server,named(rows,prefix+"restoring")); restore.release(); }
    check(request.get().status==((failed||fail_generation)?500:200),"video response");
    const auto idle_models=Json::parse(server.handle_request("GET","/v1/resources","","").body).at("loaded_models");
    if(handoff) check(!idle_models.contains("video"),"Wan handoff removes wrapper registration");
    else check(idle_models.at("video").at("active_requests")==0 && idle_models.at("video").at("in_use")==false,"retained Wan wrapper has no active request after generation");
    compare(server,named(rows,prefix+"idle")); unsetenv("WAN_FIXTURE_GATE");
}
}
int main(int argc,char **argv) { try {
    check(argc==2,"fixture required"); std::ifstream input(argv[1]); Json fixture; input>>fixture;
    setenv("CUDA_VISIBLE_DEVICES","",1);
    char pattern[]="/tmp/ezlocalai-media-slots-XXXXXX"; check(mkdtemp(pattern),"temp directory");
    auto old=fs::current_path(); fs::current_path(pattern);
    struct Cleanup { fs::path old,path; ~Cleanup(){fs::current_path(old); std::error_code ec; fs::remove_all(path,ec);} } cleanup{old,pattern};
    static_cases(fixture.at("cases"));
    for(const auto &asset:wan_model_assets(false,"Q4_K_M")) {
        auto repo=asset.repository; size_t p=0; while((p=repo.find('/',p))!=std::string::npos){repo.replace(p,1,"--");p+=2;}
        auto path=fs::path("models")/("models--"+repo)/"snapshots"/asset.revision/asset.filename;
        fs::create_directories(path.parent_path());std::ofstream(path)<<"fixture model";
    }
    for(bool keep:{false,true}) for(bool handoff:{false,true}) session(fixture.at("transitions"),keep,handoff);
    session(fixture.at("transitions"),false,true,true);
    session(fixture.at("transitions"),true,true,false,true);
    for(bool keep:{false,true}) for(bool handoff:{false,true}) session(fixture.at("transitions"),keep,handoff,false,false,true);
    std::cout<<fixture.at("cases").size()<<" source snapshots; concurrent video generation/restoration and restoration failure passed\n";
} catch(const std::exception &e){ std::cerr<<e.what()<<'\n'; return 1; } }
