#include "ezlocalai/tts_pool.hpp"
#include "ezlocalai/voice_slots.hpp"
#include "whisper_transcription_test.hpp"
#include <atomic>
#include <future>
#include <thread>
#include <signal.h>
using namespace transcription_test;
int main(int argc,char **argv) { try {
    expect(argc==2,"native TTS worker fixture"); size_t cases=0;
    for(const auto mode:{"preload","transient","handoff"}) for(bool cancel:{false,true}) {
        const std::string name=mode; Temporary temporary; const auto original=fs::current_path(); fs::current_path(temporary.path);
        struct Restore { fs::path path; ~Restore(){fs::current_path(path);} } restore{original};
        fs::create_directories("voices"); std::ofstream("voices/default.wav")<<"reference";
        for(const auto file:{"model","mmproj"})std::ofstream(file).close();
        const auto trace=temporary.path/"trace",config=temporary.path/"config";
        std::ofstream(config)<<Json{{"trace",trace.string()},{"actions",Json::array({Json::object(),Json::object(),cancel?Json{{"hang",true}}:Json::object()})}}.dump();
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
        std::atomic<bool> handoff{false}; operations.handoff_active=[&]{return handoff.load();};operations.should_preload=[&]{return name=="preload";};
        TtsPool pool(1,std::move(operations));auto shared=std::make_shared<VoicePermitQueue>(1,true); VoiceSlotOperations slots;
        slots.should_handoff=[&](const auto &service){expect(service=="tts","TTS slot provider");return name=="handoff";};
        slots.mark_handoff=[&](const auto &,bool active){handoff=active;};
        slots.unload=[](const auto &,bool force){expect(force,"handoff force unload");return "{\"model\":\"fixture\"}";};
        size_t restores=0;slots.destroy=[&](const auto &provider,bool async,bool force){expect(provider=="tts"&&!async&&force,"TTS guard destroys synchronously");pool.destroy(999,async,force);};
        slots.restore=[&](const auto &provider,const auto &state,const auto &variable){expect(provider=="tts"&&state=="{\"model\":\"fixture\"}"&&variable=="TTS_RELOAD_LLM_AFTER_GENERATION","TTS handoff restore inputs");++restores;};
        VoiceSlotGuard guard("tts",1,shared,std::move(slots)); auto slot=guard.acquire(); if(name=="preload")pool.warm(); auto model=pool.get(1);
        expect(model&&model->device()=="cpu"&&!model->closed()&&model->voices()==std::vector<std::string>{"default"}&&model->sample_rate()=="24000","native CTTS factory metadata");
        expect(resources->get_model_active_count(ModelType::Tts)==1,"pool resource active");
        const auto output=model->generate("{\"text\":\"Buffered.\",\"local_uri\":\"http://fixture\",\"output_file_name\":\"result\",\"use_cache\":false}");
        expect(output=="http://fixture/outputs/result.wav"&&read("outputs/result.wav").substr(0,4)=="RIFF","pooled native buffered synthesis");
        const auto pid=records(trace).at(0).at("pid").get<int>(); auto stream=model->stream("{\"text\":\"Stream.\"}");
        auto waiter=std::async(std::launch::async,[&]{return guard.acquire();});
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
        while((name=="handoff"?shared->snapshot().waiters:guard.permits().waiters).empty()){expect(std::chrono::steady_clock::now()<deadline,"second request queued behind owner");std::this_thread::sleep_for(std::chrono::milliseconds(1));}
        auto consumer=std::async(std::launch::async,[&]{
            pool.transfer(1,2);expect(pool.snapshot().leases.at(0).first==2,"consumer owns transferred model lease");
            size_t packets=0;std::string category;try{while(stream->next())++packets;}catch(const TtsError &e){category=e.category;}
            stream->close();pool.destroy(2,false);guard.release(slot);return std::make_pair(packets,category);
        });
        if(cancel){const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(2);while(records(trace).size()<4){expect(std::chrono::steady_clock::now()<until,"pooled stream enters worker");std::this_thread::sleep_for(std::chrono::milliseconds(2));}stream->request_cancel();}
        expect(consumer.wait_for(std::chrono::seconds(3))==std::future_status::ready,"consumer releases model and slot promptly");const auto result=consumer.get();
        expect(cancel?result.second=="CancelledError":result.first==6&&result.second.empty(),"pooled stream completion/cancellation");
        expect(waiter.wait_for(std::chrono::seconds(2))==std::future_status::ready,"waiting request admitted after consumer release");guard.release(waiter.get());
        expect(guard.permits().available==1&&shared->snapshot().available==1&&!handoff,"slot and handoff released across threads");
        expect(restores==(name=="handoff"?2u:0u),"each handoff restored once");
        expect(pool.snapshot().leases.empty(),"consumer lease removed");
        if(name=="preload"&&!cancel){expect(!model->closed()&&pool.snapshot().available.size()==1&&resources->get_model_active_count(ModelType::Tts)==0,"completed resident returned idle");expect(pool.get(3)==model,"resident reused");pool.destroy(3,false,true);}
        expect(model->closed()&&!resources->is_model_loaded(ModelType::Tts),"native model closed and resource unregistered");
        expect(kill(pid,0)==-1&&errno==ESRCH,"pooled native worker reaped");++cases;
    }
    std::cout<<cases<<" native TTS pool/CTTS/voice-slot transfer, streaming and process lifetime cases pass\n";return 0;
} catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;} }
