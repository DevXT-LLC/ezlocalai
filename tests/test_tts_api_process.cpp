#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "whisper_transcription_test.hpp"
#include <future>
#include "tts_ws_client.hpp"
#include "fixture_ports.hpp"
#include <sys/socket.h>
#include <signal.h>
using namespace transcription_test;
int main(int argc,char **argv){try{
    expect(argc==2,"native TTS worker fixture");const auto worker=fs::absolute(argv[1]);size_t cases=0;
    for(const std::string mode:{"transient","preload","dedicated","handoff"})for(const std::string transport:{"http","ws","ws-stop"}){
        Temporary temporary;struct Restore{fs::path path=fs::current_path();~Restore(){fs::current_path(path);}}restore;fs::current_path(temporary.path);fs::create_directories("voices");std::ofstream("voices/default.wav")<<"reference";for(const auto file:{"model","mmproj"})std::ofstream(file).close();const auto trace=temporary.path/"trace",settings=temporary.path/"settings";std::ofstream(settings)<<Json{{"trace",trace.string()},{"actions",Json::array({Json::object(),Json{{"hang",true}}})}};
        for(const auto &entry:environment_defaults)setenv(std::string(entry.first).c_str(),std::string(entry.second).c_str(),1);
        for(const auto &[key,value]:std::map<std::string,std::string>{{"EZLOCALAI_API_KEY",""},{"DEFAULT_MODEL","fixture-text"},{"TTS_N_PARALLEL","1"},{"TTS_UNLOAD_LLM_DURING_GENERATION",mode=="handoff"?"true":"false"},{"VOICE_SERVER",mode=="dedicated"?"true":""},{"LAZY_LOAD_VOICE",mode=="preload"?"false":"true"},{"QWEN_TTS_STREAM_FRAME_DRAIN_SECONDS","0"}})setenv(key.c_str(),value.c_str(),1);
        auto platform=native_llama_tts_platform();platform.environment["EZLOCALAI_TTS_FIXTURE"]=settings.string();platform.environment["QWEN_TTS_BIN"]=worker.string();platform.environment["QWEN_TTS_MODEL_FILE"]=(temporary.path/"model").string();platform.environment["QWEN_TTS_MMPROJ_FILE"]=(temporary.path/"mmproj").string();platform.environment["QWEN_TTS_TIMEOUT"]="5";
        CttsServiceOperations service;service.runtime.load=[platform](const auto &device,const auto &kwargs){auto owner=std::make_shared<LlamaTts>(llama_tts_launch(Json(device).dump(),kwargs,platform));auto backend=std::make_shared<CttsBackend>();backend->generate=[owner](const auto &request){auto audio=owner->generate(request);const auto rate=audio.rate;return CttsGenerated{{std::move(audio)},rate};};backend->stream=[owner](const auto &request){return owner->stream_iterator(request);};backend->close=[owner]{owner->close();};return backend;};
        auto probes=native_resource_probes();probes.gpu_count=[]{return 0;};probes.total_per_gpu=[]{return std::vector<double>{};};probes.system_ram=[]{return 16.;};auto resources=std::make_shared<ResourceManager>(probes);ezl_env_config config{};ezl_load_config(&config);
        TtsPoolFactory factory=[&](auto resources){auto pool=tts_pool_resources(tts_pool_ctts_operations(service),resources);pool.can_load=[]{return ModelAdmission{true,"cpu","fixture"};};return pool;};
        NativeServer server(config,std::make_unique<EchoInferenceBackend>("fixture-text"),resources,{},factory,{},true);
        std::future<void> writer;int peer=-1;std::unique_ptr<tts_ws_test::Client> ws;
        if(transport=="http"){
            auto response=server.handle_request("POST","/v1/audio/speech/stream","",R"({"input":"Blocked."})","application/json","","",{},true);expect(response.status==200&&bool(response.stream_writer),mode+" acquires before response");expect(server.tts_permits()->available==0&&server.tts_pool_snapshot()->leases.size()==1,mode+" pre-header lease");
            int sockets[2];expect(socketpair(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0,sockets)==0,"stream socket pair");peer=sockets[1];writer=std::async(std::launch::async,[response=std::move(response),fd=sockets[0]]{response.stream_writer(fd);close(fd);});
        }else{
            const int port=fixture_ports();std::string error;expect(server.start("127.0.0.1",port,&error),error);ws=std::make_unique<tts_ws_test::Client>(port);ws->message(R"({"text":"Blocked."})");
            std::this_thread::sleep_for(std::chrono::milliseconds(30));expect(server.tts_permits()->available==1&&server.tts_pool_snapshot()->leases.empty(),mode+" composing text does not reserve model");
            ws->message(R"({"flush":true})");
        }
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);while(!fs::exists(trace)||records(trace).size()<3){expect(std::chrono::steady_clock::now()<deadline,mode+" worker enters blocked generation");std::this_thread::sleep_for(std::chrono::milliseconds(2));}
        auto following=std::async(std::launch::async,[&]{return server.handle_request("POST","/v1/audio/speech/stream","",R"({"input":""})","application/json");});
        if(mode!="handoff")while(server.tts_permits()->waiters.empty()){expect(std::chrono::steady_clock::now()<deadline,"following request queues");std::this_thread::sleep_for(std::chrono::milliseconds(1));}
        if(transport=="http"){shutdown(peer,SHUT_RDWR);close(peer);}
        else if(transport=="ws-stop")writer=std::async(std::launch::async,[&]{server.stop();});
        else ws->disconnect();
        if(writer.valid()){expect(writer.wait_for(std::chrono::seconds(3))==std::future_status::ready,mode+" "+transport+" cancels worker");writer.get();}
        expect(following.wait_for(std::chrono::seconds(3))==std::future_status::ready,mode+" following request admitted");const auto next=following.get();expect(next.status==200&&!next.transport_error&&next.body.size()==12,mode+" following empty stream completes");expect(server.tts_permits()->available==1&&server.tts_pool_snapshot()->leases.empty(),mode+" all leases and permits released");expect(resources->get_model_active_count(ModelType::Tts)==0,mode+" active usage released");
        // The cancelled worker must be gone; a preloaded replacement may remain until server destruction.
        const auto first=records(trace).front().at("pid").get<int>();expect(kill(first,0)==-1&&errno==ESRCH,mode+" cancelled native worker reaped");server.stop();++cases;
    }
    std::cout<<cases<<" native HTTP/WebSocket disconnect and shutdown modes cancel blocked workers and admit queued speech\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
