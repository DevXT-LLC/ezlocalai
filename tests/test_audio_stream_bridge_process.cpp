#include "ezlocalai/ctts_stream_bridge.hpp"
#include "whisper_transcription_test.hpp"
#include <future>
#include <thread>
#include <signal.h>
using namespace transcription_test;
int main(int argc,char **argv){try{
 expect(argc==2,"native process fixture");size_t cases=0;
 for(const auto mode:{"complete","unstarted","abandon","cancel","consumer_error","worker_error","reference_error","iterator_unstarted","iterator_owner","iterator_owner_unstarted","iterator_owner_abandon"}){
  Temporary temp;const auto trace=temp.path/"trace",configuration=temp.path/"config";Json action=Json::object();if(std::string(mode)=="cancel")action["hang"]=true;if(std::string(mode)=="worker_error")action["responses"]=Json::array({Json{{"error","fixture failed"}}});std::ofstream(configuration)<<Json{{"trace",trace.string()},{"actions",Json::array({action})}}.dump();for(const auto name:{"model","mmproj","speaker"})std::ofstream(temp.path/name).close();auto platform=native_llama_tts_platform();auto &env=platform.environment;env["EZLOCALAI_TTS_FIXTURE"]=configuration.string();env["QWEN_TTS_BIN"]=argv[1];env["QWEN_TTS_MODEL_FILE"]=(temp.path/"model").string();env["QWEN_TTS_MMPROJ_FILE"]=(temp.path/"mmproj").string();env["QWEN_TTS_TIMEOUT"]="5";
  auto model=std::make_shared<LlamaTts>(llama_tts_launch("\"cpu\"","{}",std::move(platform)));auto backend=std::make_shared<CttsBackend>();backend->stream=[model](const auto &request){return model->stream_iterator(request);};backend->close=[model]{model->close();};Json request={{"text","Hello."},{"language","English"},{"ref_audio",std::string(mode)=="reference_error"?Json():Json((temp.path/"speaker").string())}};if(std::string(mode).rfind("iterator_owner",0)==0){auto iterator=model->stream_iterator(request.dump());const int pid=records(trace).at(0).at("pid");backend.reset();model.reset();expect(kill(pid,0)==0,"lazy iterator retains native owner");if(std::string(mode)=="iterator_owner"){size_t count=0;while(iterator->next())++count;expect(count==2,"retained iterator completes");}else if(std::string(mode)=="iterator_owner_abandon")expect(bool(iterator->next()),"retained iterator yields");iterator->close();expect(kill(pid,0)==-1&&errno==ESRCH,"last iterator reference reaps worker");++cases;continue;}
  CttsStreamBridge bridge(backend,request.dump());size_t frames=0;std::string error;std::future<void> cancellation;const auto started=std::chrono::steady_clock::now();
  try{
   if(std::string(mode)=="unstarted")bridge.close();else if(std::string(mode)=="iterator_unstarted"){auto iterator=model->stream_iterator(request.dump());iterator->close();expect(!iterator->next(),"closed lazy iterator exhausted");}else if(std::string(mode)=="cancel"){
    cancellation=std::async(std::launch::async,[&]{const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);while(records(trace).size()<2){expect(std::chrono::steady_clock::now()<deadline,"native worker received stream request");std::this_thread::sleep_for(std::chrono::milliseconds(5));}bridge.request_cancel();});bridge.next();
   }else if(std::string(mode)=="abandon"||std::string(mode)=="consumer_error"){expect(bool(bridge.next()),"first native frame");++frames;if(std::string(mode)=="consumer_error")throw TtsError("RuntimeError","consumer failed");bridge.close();}else while(const auto frame=bridge.next()){expect(frame->rate=="24000"&&frame->samples.size()==3,"native PCM frame");++frames;}
  }catch(const TtsError &e){error=e.category;bridge.close();}
  if(cancellation.valid())cancellation.get();
  if(std::string(mode)=="complete"){expect(frames==2&&error.empty()&&model->alive(),"completed bridge retains worker");expect(model->generate(request.dump()).samples.size()==6,"buffered recovery after complete stream");}
  else if(std::string(mode)=="unstarted"||std::string(mode)=="iterator_unstarted")expect(records(trace).size()==1&&model->alive(),"lazy close sends no native request");
  else{expect(!model->alive(),"incomplete bridge kills worker");if(std::string(mode)=="cancel")expect(error=="CancelledError"&&std::chrono::steady_clock::now()-started<std::chrono::seconds(2),"waiting stream cancelled and joined promptly");else if(std::string(mode)=="reference_error")expect(error=="TypeError","reference failure preserved");else if(std::string(mode)!="abandon")expect(error=="RuntimeError","generation/consumer failure preserved");}
  bridge.close();model->close();const auto pid=records(trace).at(0).at("pid").get<int>();expect(kill(pid,0)==-1&&errno==ESRCH,"native worker reaped before release");++cases;
 }
 std::cout<<cases<<" real process bridge completion/cancellation/lazy-iterator cases pass\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
