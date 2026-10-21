#include "tts_service_test.hpp"
#include "ezlocalai/wakeword_tts.hpp"
#include "ezlocalai/ctts_audio.hpp"
#include <openssl/evp.h>
#include <future>
#include <thread>

namespace {
std::string encoded(const std::string &bytes) {
    std::string out(4*((bytes.size()+2)/3)+1,'\0');
    out.resize(EVP_EncodeBlock(reinterpret_cast<unsigned char *>(out.data()),
        reinterpret_cast<const unsigned char *>(bytes.data()),bytes.size()));return out;
}
std::string wav() {
    CttsAudioArray audio;audio.shape={2400};audio.samples.resize(2400);
    for(size_t i=0;i<audio.samples.size();++i)audio.samples[i]=float(int(i%100)-50)/100;
    return ctts_wav_bytes(audio,24000);
}
void released(Environment &env,TtsService &service) {
    const auto state=env.state(service);
    expect(state.at("leases").empty()&&state.at("stacks").empty(),"Qwen leases released");
    expect(env.permit==1&&env.shared==1&&env.handoff==0,"voice admission restored");
    expect(env.counts["fallback"]==0&&env.counts["fallback.env"]==0,"no provider fallback consulted");
    if(!state.at("resource").is_null())expect(state.at("resource").at("active")==0,"resource no longer active");
}
void service_case(bool handoff,const std::string &failure,bool dedicated=false) {
    Json config{{"capacity",1},{"handoff",handoff},{"voice_mode",dedicated},{"preload",dedicated},{"audio",encoded(wav())}};
    if(!failure.empty())config["failures"]=Json::array({Json{{"event",failure},{"message","GPU generation failed"}}});
    Environment env(config);
    TtsService service(1,env.pool(),env.voice(),std::make_shared<VoicePermitQueue>(1,true),env.resources,env.request());
    Temporary dir;
    WakewordSampleGenerator generator(dir.path.string(),wakeword_qwen_tts_operations(service,{"default","second"}));
    std::vector<std::pair<size_t,size_t>> progress;
    const auto samples=generator.generate("hello ezlocalai",3,[&](size_t current,size_t total,const auto &){progress.emplace_back(current,total);});
    expect(samples.size()==(failure.empty()?3:2),"failed sample omitted; remaining samples generated");
    expect(progress==std::vector<std::pair<size_t,size_t>>{{1,3},{2,3},{3,3}},"completed attempt progress");
    const auto normalized=wakeword_prepare_tts_wav(wav());
    for(size_t i=0;i<samples.size();++i){const auto &sample=samples[i];
        expect(sample.audio==normalized&&sample.engine=="qwen"&&sample.sample_rate==16000,"Qwen PCM normalization and metadata");
        expect(sample.voice==((i+(failure.empty()?0:1))%2?"second":"default"),"voice cycle and failed-sample ordering");
    }
    size_t generated=0;
    for(const auto &event:env.events)if(event.at("event")=="generate"){
        const auto &request=event.at("request");
        expect(request.at("use_cache")==false&&request.at("local_uri").is_null(),"fresh WAV rather than cached URL");
        expect(request.at("text")=="hello ezlocalai"&&request.at("language")=="auto","word and language");++generated;
        expect(request.at("voice")==((generated-1+(failure=="load"?1:0))%2?"second":"default"),"selected Qwen voice reaches synthesis");
    }
    expect(generated==(failure=="load"?2:3),"fresh Qwen invocation per attempt");released(env,service);
}
void in_flight_cancellation(bool during_load) {
    Environment env(Json{{"capacity",1},{"audio",encoded(wav())}});VoiceCancellation cancellation;
    auto pool=env.pool();const auto load=pool.load;
    pool.load=[&,load](const auto &device){auto model=load(device);
        if(during_load)cancellation.cancel();else{const auto generate=model->generate;model->generate=[&,generate](const auto &request){auto bytes=generate(request);cancellation.cancel();return bytes;};}
        return model;
    };
    TtsService service(1,std::move(pool),env.voice(),std::make_shared<VoicePermitQueue>(1,true),env.resources,env.request());
    bool cancelled=false;try{service.generate_local("{}",cancellation);}catch(const VoiceCancelled &){cancelled=true;}
    expect(cancelled&&env.counts["generate"]==(during_load?0:1),"in-flight cancellation discards audio and prevents the next stage");released(env,service);
}
void cancellation_case(bool during) {
    Environment env(Json{{"capacity",1},{"audio",encoded(wav())}});
    TtsService service(1,env.pool(),env.voice(),std::make_shared<VoicePermitQueue>(1,true),env.resources,env.request());
    VoiceCancellation cancellation;auto ops=wakeword_qwen_tts_operations(service,{"default"},cancellation);
    if(during){auto call=ops.qwen_wav;ops.qwen_wav=[&,call](const auto &word,const auto &voice){auto bytes=call(word,voice);cancellation.cancel();return bytes;};}
    else cancellation.cancel();
    Temporary dir;WakewordSampleGenerator generator(dir.path.string(),std::move(ops));bool cancelled=false;
    try{wakeword_similar_samples("hello",generator,3);}catch(const VoiceCancelled &){cancelled=true;}
    expect(cancelled,"similar-word generation propagates cancellation");
    expect(env.counts["generate"]==(during?1:0),"no additional synthesis after cancellation");released(env,service);
}
void waiting_cancellation(bool service_cancel) {
    Environment env(Json{{"capacity",1}});auto permit=std::make_shared<VoicePermitQueue>(1);
    permit->acquire();auto voice=env.voice();std::promise<void> waiting;
    voice.acquire_permit=[&](const auto &token){waiting.set_value();permit->acquire(token);};
    voice.release_permit=[&]{permit->release();};
    TtsService service(1,env.pool(),std::move(voice),std::make_shared<VoicePermitQueue>(1,true),env.resources,env.request());
    VoiceCancellation cancellation;
    auto worker=std::async(std::launch::async,[&]{try{service.generate_local("{}",cancellation);return false;}catch(const VoiceCancelled &){return true;}});
    waiting.get_future().wait();if(service_cancel)service.cancel();else cancellation.cancel();
    const auto ready=worker.wait_for(std::chrono::seconds(2))==std::future_status::ready;
    if(!ready)permit->release();
    expect(worker.get()&&ready,"job/service cancellation wakes blocked admission");
    expect(permit->snapshot().waiters.empty()&&env.counts["load"]==0,"cancelled waiter never loads Qwen");
    if(ready)permit->release();
}
}
int main(int argc,char **argv){try{
    if(argc==5){
        const auto worker=fs::absolute(argv[1]),models=fs::absolute(argv[2]),speaker=fs::absolute(argv[3]),evidence=fs::absolute(argv[4]);
        fs::create_directories(evidence);
        setenv("QWEN_TTS_BIN",worker.c_str(),1);
        setenv("QWEN_TTS_MODEL_FILE",(models/"qwen-talker-0.6b-base-Q8_0.gguf").c_str(),1);
        setenv("QWEN_TTS_MMPROJ_FILE",(models/"qwen-tokenizer-12hz-f16.gguf").c_str(),1);
        setenv("QWEN_TTS_THREADS","8",1);setenv("QWEN_TTS_ALLOW_CPU_FALLBACK","true",1);
        Temporary dir;const auto original=fs::current_path();fs::current_path(dir.path);
        struct Restore{fs::path path;~Restore(){fs::current_path(path);}} restore{original};
        fs::create_directories("voices");fs::copy_file(speaker,"voices/default.wav");
        Environment env(Json{{"capacity",1},{"preload",true}});env.native=tts_pool_ctts_operations();
        TtsService service(1,env.pool(),env.voice(),std::make_shared<VoicePermitQueue>(1,true),env.resources,env.request());
        auto operations=wakeword_qwen_tts_operations(service);auto invoke=operations.qwen_wav;size_t index=0;
        Json report{{"device","cpu"},{"samples",Json::array()}};
        operations.qwen_wav=[&](const auto &word,const auto &voice){
            auto bytes=invoke(word,voice);std::ofstream(evidence/ ("qwen-"+std::to_string(index++)+".wav"),std::ios::binary).write(bytes.data(),bytes.size());return bytes;
        };
        WakewordSampleGenerator generator("samples",std::move(operations));const auto samples=generator.generate("Hello ezlocalai",2);
        expect(samples.size()==2,"real Qwen returned two samples");
        for(size_t i=0;i<samples.size();++i){const auto &sample=samples[i];
            expect(sample.audio.size()>44&&sample.sample_rate==16000&&sample.engine=="qwen","real normalized PCM");
            std::ofstream(evidence/("normalized-"+std::to_string(i)+".wav"),std::ios::binary).write(sample.audio.data(),sample.audio.size());
            report["samples"].push_back(Json{{"bytes",sample.audio.size()},{"sha256",hash(sample.audio)}});
        }
        released(env,service);service.unload();
        expect(env.state(service).at("resource").is_null(),"real model resource unregistered");
        report["calls"]=env.counts["generate"];report["fallback_calls"]=env.counts["fallback"];
        std::ofstream(evidence/"validation.json")<<report.dump(2)<<'\n';std::cout<<report.dump()<<'\n';return 0;
    }
    expect(argc==1,"optional arguments: Qwen worker, model directory, speaker WAV, evidence directory");
    for(bool handoff:{false,true})for(const auto &failure:{"","load","generate"})service_case(handoff,failure);
    service_case(false,"",true);service_case(false,"generate",true);
    in_flight_cancellation(false);in_flight_cancellation(true);
    cancellation_case(false);cancellation_case(true);waiting_cancellation(false);waiting_cancellation(true);
    Temporary dir;auto bytes=wav();int calls=0;
    WakewordTtsOperations ops;ops.qwen_wav=[&](const auto &,const auto &){++calls;return bytes;};
    ops.gtts_mp3=ops.edge_mp3=[](const auto &,const auto &)->std::string{throw std::runtime_error("legacy engine invoked");};
    WakewordSampleGenerator generator(dir.path.string(),ops);
    expect(generator.generate("hello",0).empty()&&generator.generate("hello",-1).empty()&&calls==0,"nonpositive count makes no calls");
    expect(generator.generate("hello",7).size()==7&&calls==7,"Qwen count is not capped by historical provider lists");
    ops.qwen_voices.clear();bool rejected=false;
    try{WakewordSampleGenerator(dir.path.string(),ops).generate("hello",1);}catch(const std::runtime_error &){rejected=true;}
    expect(rejected,"empty voice configuration rejected");
    std::cout<<"Qwen wake-word samples: normalization, uncached synthesis, provider isolation, admission, failures and cancellation passed\n";
    return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
