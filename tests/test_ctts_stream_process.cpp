#include "ezlocalai/ctts.hpp"
#include "whisper_transcription_test.hpp"
#include <atomic>
#include <future>
#include <thread>
#include <signal.h>
using namespace transcription_test;
int main(int argc,char **argv){try{
 expect(argc==2,"native worker fixture");size_t cases=0;
 for(const auto mode:{"complete","cancel_wait","cancel_drain","cancel_large_drain","cancel_infinite_drain","abandon_payload","unstarted_owner","header_owner","cancel_unstarted"}){
  const std::string name=mode;const bool draining=name=="cancel_drain"||name=="cancel_large_drain"||name=="cancel_infinite_drain";Temporary temp;const auto original=fs::current_path();fs::current_path(temp.path);struct Restore{fs::path path;~Restore(){fs::current_path(path);}}restore{original};fs::create_directories("voices");std::ofstream("voices/default.wav")<<"reference";for(const auto file:{"model","mmproj"})std::ofstream(file).close();const auto trace=temp.path/"trace",config=temp.path/"config";const Json action=name=="cancel_wait"?Json{{"hang",true}}:Json::object();std::ofstream(config)<<Json{{"trace",trace.string()},{"actions",Json::array({Json::object(),action})}}.dump();auto platform=native_llama_tts_platform();auto &env=platform.environment;env["EZLOCALAI_TTS_FIXTURE"]=config.string();env["QWEN_TTS_BIN"]=argv[1];env["QWEN_TTS_MODEL_FILE"]=(temp.path/"model").string();env["QWEN_TTS_MMPROJ_FILE"]=(temp.path/"mmproj").string();env["QWEN_TTS_TIMEOUT"]="5";std::atomic<size_t> closes{0};std::weak_ptr<LlamaTts> weak;CttsServiceOperations ops;
  ops.runtime.environment=[&](const std::string &key)->std::optional<std::string>{if(key=="QWEN_TTS_STREAM_FRAME_DRAIN_SECONDS")return name=="cancel_large_drain"?"1e300":name=="cancel_infinite_drain"?"inf":draining?"3600":"0";return std::nullopt;};ops.runtime.load=[&](const auto &device,const auto &kwargs){auto model=std::make_shared<LlamaTts>(llama_tts_launch(Json(device).dump(),kwargs,platform));weak=model;auto backend=std::make_shared<CttsBackend>();backend->generate=[model](const auto &request){auto audio=model->generate(request);const auto rate=audio.rate;return CttsGenerated{{std::move(audio)},rate};};backend->stream=[model](const auto &request){return model->stream_iterator(request);};backend->close=[model,&closes]{++closes;model->close();};return backend;};auto owner=std::make_unique<Ctts>("{\"enabled\":false}","\"cpu\"",std::move(ops));auto stream=owner->generate_stream("{\"text\":\"Hello.\"}");const auto pid=records(trace).at(0).at("pid").get<int>();
  if(name=="unstarted_owner"||name=="header_owner"){if(name=="header_owner")expect(stream->next()->size()==8,"stream header");owner.reset();expect(!weak.expired()&&kill(pid,0)==0,"stream retains CTTS owner");stream->close();expect(weak.expired()&&closes==0,"closing outside chunk releases owner without CTTS.close");}
  else if(name=="complete"){size_t count=0;while(stream->next())++count;expect(count==6&&!owner->runtime().closed()&&!weak.expired()&&closes==0,"complete stream retains live model");owner->close();}
  else if(name=="abandon_payload"){for(int i=0;i<3;++i)expect(bool(stream->next()),"payload yield");stream->close();expect(owner->runtime().closed()&&closes==2,"payload close tears down bridge then owner");}
  else if(name=="cancel_unstarted"){stream->request_cancel();std::string category;try{stream->next();}catch(const TtsError &e){category=e.category;}expect(category=="CancelledError"&&!owner->runtime().closed()&&closes==0,"unstarted cancellation leaves owner open");owner->close();}
  else{
   const int count=draining?5:1;for(int i=0;i<count;++i)expect(bool(stream->next()),"pre-cancel yield");auto pending=std::async(std::launch::async,[&]{try{stream->next();return std::string("returned");}catch(const TtsError &e){return e.category;}});
   if(name=="cancel_wait"){const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);while(records(trace).size()<3){expect(std::chrono::steady_clock::now()<deadline,"worker entered stream");std::this_thread::sleep_for(std::chrono::milliseconds(5));}}
   else expect(pending.wait_for(std::chrono::milliseconds(50))==std::future_status::timeout,"long drain still pending");
   const auto start=std::chrono::steady_clock::now();for(int i=0;i<20;++i)stream->request_cancel();expect(pending.wait_for(std::chrono::seconds(2))==std::future_status::ready,"cancel completes native teardown");expect(pending.get()=="CancelledError"&&owner->runtime().closed(),"cancellation closes CTTS");expect(std::chrono::steady_clock::now()-start<std::chrono::seconds(2),"wait/drain interrupted promptly");if(name=="cancel_wait")expect(closes==2,"blocked bridge then owner close");else expect(closes>=1&&closes<=2,"drain cancellation closes owner");
  }
  stream->close();expect(kill(pid,0)==-1&&errno==ESRCH,"worker reaped before release");++cases;
 }
 std::cout<<cases<<" CTTS stream process cancellation/ownership cases pass\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
