#include "ezlocalai/ctts_stream_bridge.hpp"
#include "whisper_transcription_test.hpp"
#include <condition_variable>
#include <future>
#include <thread>
using namespace transcription_test;
struct Event {std::mutex mutex;std::condition_variable condition;bool ready=false;void set(){std::lock_guard<std::mutex> lock(mutex);ready=true;condition.notify_all();}void wait(){std::unique_lock<std::mutex> lock(mutex);expect(condition.wait_for(lock,std::chrono::seconds(3),[&]{return ready;}),"bridge gate timeout");}bool is_set(){std::lock_guard<std::mutex> lock(mutex);return ready;}};
void fail(const Json &value){if(!value.is_null())throw TtsError(value.at("category"),value.at("message"));}
struct Iterator final:TtsStreamIterator {std::function<std::optional<TtsAudio>()> advance;std::function<void()> finish;std::optional<TtsAudio> next()override{return advance();}void close()override{finish();}};
int main(int argc,char **argv){try{
 expect(argc==3,"source and bridge fixture");const auto rows=records(argv[2]);expect(hash(read(fs::path(argv[1])/"ezlocalai/AudioStreaming.py"))==rows[0].at("source_sha256"),"pinned native audio bridge source");
 for(size_t n=1;n<rows.size();++n){const auto &row=rows[n];const auto mode=row.at("mode").get<std::string>();const auto main=std::this_thread::get_id();Event gate,entered,finished,prefetched;std::mutex traces_mutex;Json trace=Json::array(),closes=Json::array();size_t next_count=0;auto model=std::make_shared<CttsBackend>();
 const auto event=[&](const std::string &name,Json args=Json::array()){std::lock_guard<std::mutex> lock(traces_mutex);Json entry=Json::array({name,std::this_thread::get_id()==main});for(const auto &arg:args)entry.push_back(arg);trace.push_back(entry);if(name=="next")++next_count;};
 model->stream=[&](const std::string &request)->std::unique_ptr<TtsStreamIterator>{event("open",Json::array({Json::parse(request)}));if(row.contains("factory_error")){entered.set();finished.set();fail(row.at("factory_error"));}auto iterator=std::make_unique<Iterator>();iterator->advance=[&,index=size_t{0}]()mutable->std::optional<TtsAudio>{const auto i=index++;event("next",Json::array({i}));if(i==3)prefetched.set();if(row.contains("gate")&&row.at("gate")==i){entered.set();gate.wait();}if(row.contains("next_error")&&row.at("next_error").at("index")==i){entered.set();fail(row.at("next_error"));}if(i>=row.at("frames").get<size_t>())return std::nullopt;return TtsAudio{{float(i),.25f,-.5f},std::to_string(24000+i),1};};iterator->finish=[&]{event("iterator_close");finished.set();fail(row.value("iterator_error",Json()));};return iterator;};
 model->close=[&]{{std::lock_guard<std::mutex> lock(traces_mutex);closes.push_back(std::this_thread::get_id()==main);}if(!row.value("defer_model_release",false))gate.set();fail(row.value("model_error",Json()));};
 CttsStreamBridge stream(model,R"({"text":"Hello.","language":"English","ref_audio":"voice.wav","max_new_tokens":8})");Json actual={{"chunks",Json::array()}};std::future<void> cancellation;
 const auto take=[&]{const auto value=stream.next();expect(bool(value),"expected stream audio");actual["chunks"].push_back(Json::array({value->samples,value->rate}));};
 try{
 if(mode=="unstarted")stream.close();else if(mode=="cancel"){cancellation=std::async(std::launch::async,[&]{entered.wait();std::this_thread::sleep_for(std::chrono::milliseconds(10));stream.request_cancel();});stream.next();}else if(mode=="abandon"||mode=="finished"||mode=="bounded"){for(size_t i=0;i<row.value("take",size_t{1});++i)take();if(mode=="bounded"){prefetched.wait();std::this_thread::sleep_for(std::chrono::milliseconds(20));std::lock_guard<std::mutex> lock(traces_mutex);actual["prefetched"]=next_count;}else if(mode=="finished")finished.wait();else entered.wait();stream.close();}else{while(const auto value=stream.next())actual["chunks"].push_back(Json::array({value->samples,value->rate}));}
 }catch(const TtsError &e){actual["error"]={e.category,e.what()};if(row.value("defer_model_release",false))actual["producer_pending"]=!finished.is_set();}
 gate.set();if(cancellation.valid())cancellation.get();if(mode!="unstarted")finished.wait();stream.close();actual["trace"]=trace;actual["model_close_threads"]=closes;actual["producer_finished"]=finished.is_set();compare_json(actual,row.at("expected"),std::to_string(n)+" "+row.at("name").get<std::string>());
 }
 expect(rows.size()-1==rows[0].at("cases"),"complete bridge comparisons");std::cout<<rows.size()-1<<" source audio bridge cases match, including bounded prefetch and cancellation\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
