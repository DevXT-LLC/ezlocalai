#include "ezlocalai/tts_service.hpp"
#include "whisper_transcription_test.hpp"
#include <atomic>
#include <future>
#include <thread>
#include <signal.h>
using namespace transcription_test;
void lifetime_and_admission(){
    struct Empty:CttsByteStream{std::optional<std::string> next()override{return {};}void close()override{}void request_cancel()override{}};
    std::atomic<int> closed{0};TtsPoolOperations pool;pool.load=[&](const auto &){auto model=std::make_shared<TtsPoolModel>();model->close=[&]{++closed;};model->stream=[](const auto &){return std::make_unique<Empty>();};return model;};
    TtsRequestOperations request;request.mark_in_use=[](bool){};request.fallback_generate=[](const auto &){return "";};auto shared=std::make_shared<VoicePermitQueue>(1,true);
    auto owner=std::make_unique<TtsService>(1,pool,VoiceSlotOperations{},shared,nullptr,request);auto stream=owner->stream("{}",1);
    auto waiter=std::async(std::launch::async,[&]{try{owner->generate("{}",2);return false;}catch(const VoiceCancelled &){return true;}});
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);while(owner->permits().waiters.empty()){expect(std::chrono::steady_clock::now()<deadline,"service admission waiter enters");std::this_thread::yield();}
    owner->cancel();expect(waiter.get()&&owner->permits().available==0,"service shutdown cancellation removes waiting request only");
    owner.reset();expect(closed==0,"active stream retains service and model");stream->consumer_key(3);expect(!stream->next()&&closed==1,"stream completes cleanup after service handle is dropped");stream->close();expect(closed==1,"finished stream close is idempotent");
    struct Packet:Empty{bool pending=true;std::optional<std::string> next()override{if(!pending)return {};pending=false;return "packet";}};
    pool.load=[](const auto &){auto model=std::make_shared<TtsPoolModel>();model->stream=[](const auto &){return std::make_unique<Packet>();};return model;};
    TtsService automatic(1,pool,{},shared,nullptr,request);auto auto_stream=automatic.stream("{}",999);auto_stream->consumer_key(0);expect(auto_stream->next()==std::optional<std::string>("packet"),"automatic consumer begins");const auto key=automatic.snapshot().leases.at(0).first;
    auto_stream->consumer_key(0);expect(automatic.snapshot().leases.at(0).first==key&&!auto_stream->next()&&automatic.snapshot().leases.empty()&&automatic.permits().available==1,"repeated automatic identity preserves lease cleanup");
    bool rejected=false;try{TtsService invalid(1,pool,{},shared,nullptr,{});}catch(const std::invalid_argument &){rejected=true;}expect(rejected,"missing fallback callback rejects construction without invalid destruction");
}
int main(int argc,char **argv){try{
    expect(argc==2,"native TTS worker fixture");size_t cases=0;
    for(const auto mode:{"preload","dedicated","transient","handoff"})for(const auto operation:{"buffered","stream","cancel"}){
        const std::string name=mode,action=operation;const bool cancel=action=="cancel"; Temporary temporary; const auto original=fs::current_path(); fs::current_path(temporary.path);
        struct Restore { fs::path path; ~Restore(){fs::current_path(path);} } restore{original};
        fs::create_directories("voices"); std::ofstream("voices/default.wav")<<"reference";
        for(const auto file:{"model","mmproj"})std::ofstream(file).close();
        const auto trace=temporary.path/"trace",config=temporary.path/"config";
        std::ofstream(config)<<Json{{"trace",trace.string()},{"actions",Json::array({Json::object(),cancel?Json{{"hang",true}}:Json::object()})}}.dump();
        auto platform=native_llama_tts_platform(); auto &env=platform.environment;
        env["EZLOCALAI_TTS_FIXTURE"]=config.string(); env["QWEN_TTS_BIN"]=argv[1]; env["QWEN_TTS_MODEL_FILE"]=(temporary.path/"model").string(); env["QWEN_TTS_MMPROJ_FILE"]=(temporary.path/"mmproj").string(); env["QWEN_TTS_TIMEOUT"]="5";
        CttsServiceOperations service; service.runtime.environment=[](const std::string &key)->std::optional<std::string>{if(key=="QWEN_TTS_STREAM_FRAME_DRAIN_SECONDS")return "0"; return std::nullopt;};
        service.runtime.load=[&](const auto &device,const auto &kwargs){
            auto owner=std::make_shared<LlamaTts>(llama_tts_launch(Json(device).dump(),kwargs,platform)); auto backend=std::make_shared<CttsBackend>();
            backend->generate=[owner](const auto &request){auto audio=owner->generate(request);const auto rate=audio.rate;return CttsGenerated{{std::move(audio)},rate};};
            backend->stream=[owner](const auto &request){return owner->stream_iterator(request);};backend->close=[owner]{owner->close();};return backend;
        };
        auto probes=native_resource_probes();probes.gpu_count=[]{return 0;};probes.total_per_gpu=[]{return std::vector<double>{};};probes.system_ram=[]{return 16.;};probes.fallback_server=[]{return std::string();};
        auto resources=std::make_shared<ResourceManager>(probes); auto operations=tts_pool_resources(tts_pool_ctts_operations(service),resources);

        operations.should_preload=[&]{return name=="preload"||name=="dedicated";};std::vector<std::function<void()>> pending;operations.schedule_cleanup=[&](auto work){pending.push_back(std::move(work));};
        auto shared=std::make_shared<VoicePermitQueue>(1,true);std::atomic<bool> handoff{false};size_t restores=0;VoiceSlotOperations voice;
        voice.should_handoff=[&](const auto &){return name=="handoff";};voice.mark_handoff=[&](const auto &,bool active){handoff=active;};voice.unload=[](const auto &,bool){return "{\"model\":\"fixture\"}";};voice.restore=[&](const auto &,const auto &,const auto &){++restores;};
        TtsRequestOperations requests;requests.voice_server_mode=[&]{return name=="dedicated";};requests.fallback_generate=[](const auto &)->std::string{throw std::runtime_error("unexpected fallback");};
        TtsService owner(1,std::move(operations),std::move(voice),shared,resources,std::move(requests));
        if(action=="buffered"){
            const auto output=owner.generate("{\"text\":\"Buffered.\",\"voice\":\"default\",\"language\":\"auto\"}",1);expect(output.substr(0,4)=="RIFF","service decodes buffered WAV");
            expect(resources->get_model_active_count(ModelType::Tts)==0,"buffered usage unwinds both marks");
            expect(owner.snapshot().leases.size()==(name=="dedicated"?1u:0u),"dedicated buffered model lease retained");
            expect(pending.size()==(name=="transient"?1u:0u),"ordinary transient buffer cleanup queued");
            for(auto &work:pending)work();pending.clear();
        }else{
            auto stream=owner.stream("{\"text\":\"Stream.\",\"voice\":\"default\",\"language\":\"auto\"}",1);
            expect(owner.snapshot().leases.at(0).first==1&&resources->get_model_active_count(ModelType::Tts)==1,"stream reserves model before iteration");
            auto waiter=std::async(std::launch::async,[&]{return owner.stream("{\"text\":\"\",\"voice\":\"default\",\"language\":\"auto\"}",3);});
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);while((name=="handoff"?shared->snapshot().waiters:owner.permits().waiters).empty()){expect(std::chrono::steady_clock::now()<deadline,"following speech request queues");std::this_thread::sleep_for(std::chrono::milliseconds(1));}
            auto consumer=std::async(std::launch::async,[&]{stream->consumer_key(2);size_t packets=0;std::string error;try{while(stream->next())++packets;}catch(const TtsError &e){error=e.category;}return std::make_pair(packets,error);});
            if(cancel){const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(2);while(records(trace).size()<3){expect(std::chrono::steady_clock::now()<until,"stream enters native worker");std::this_thread::sleep_for(std::chrono::milliseconds(2));}stream->request_cancel();}
            expect(consumer.wait_for(std::chrono::seconds(3))==std::future_status::ready,"service consumer cleanup completes");const auto result=consumer.get();expect(cancel?result.second=="CancelledError":result.first==6&&result.second.empty(),"service stream bytes or cancellation");
            expect(waiter.wait_for(std::chrono::seconds(3))==std::future_status::ready,"waiting speech request admitted");auto following=waiter.get();following->consumer_key(4);size_t packets=0;while(following->next())++packets;expect(packets==2,"following empty stream header and terminator");
            expect(owner.snapshot().leases.empty(),"stream lease released even in dedicated mode");
            expect(restores==(name=="handoff"?2u:0u),"stream handoff restored after each request");
        }
        expect(owner.permits().available==1&&shared->snapshot().available==1&&!handoff,"service releases voice admission");owner.unload();expect(!resources->is_model_loaded(ModelType::Tts),"service unload clears resource");
        for(const auto &row:records(trace))if(row.contains("pid")){expect(kill(row.at("pid").get<int>(),0)==-1&&errno==ESRCH,"service native worker reaped");}++cases;
    }
    lifetime_and_admission();std::cout<<cases<<" TTS service native lifecycle cases plus admission cancellation and retained ownership pass\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
