#include "ezlocalai/tts_websocket.hpp"
#include "whisper_transcription_test.hpp"
#include <condition_variable>
#include <future>
#include <mutex>
using namespace transcription_test;
std::string hex(const std::string &value){std::string out;for(unsigned char c:value){out+="0123456789abcdef"[c>>4];out+="0123456789abcdef"[c&15];}return out;}
namespace {
using namespace std::chrono_literals;
struct LifecycleState {
    std::mutex mutex;
    std::condition_variable changed;
    bool receiver_waiting=false,release_receiver=false,interrupted=false,active=false,idle_notice=false;
    int closes=0;
    std::vector<bool> observations;
    std::string audio;
};
struct EmptyPcmStream final:CttsByteStream {
    bool emitted=false;
    std::optional<std::string> next()override{if(emitted)return {};emitted=true;return tts_pcm_header()+std::string(4,'\0');}
    void close()override{}
    void request_cancel()override{}
};
struct ObservedLease final:TtsWebSocketLease {
    LifecycleState &state;bool fail_close;
    ObservedLease(LifecycleState &state,bool fail_close):state(state),fail_close(fail_close){}
    std::unique_ptr<CttsByteStream> stream(const std::string &)override{return std::make_unique<EmptyPcmStream>();}
    void close()override{
        {std::lock_guard<std::mutex> lock(state.mutex);state.active=false;++state.closes;}
        if(fail_close)throw TtsError("RuntimeError","lease close failure");
    }
};
void lifecycle_case(bool observer_failure,bool close_failure,bool acquire_failure){
    LifecycleState state;
    TtsWebSocketOperations ops;
    size_t received=0;
    ops.receive=[&]{
        if(received++==0)return std::string(R"({"text":"Hello.","flush":true})");
        std::unique_lock<std::mutex> lock(state.mutex);state.receiver_waiting=true;state.changed.notify_all();
        state.changed.wait(lock,[&]{return state.release_receiver;});
        if(state.interrupted)throw TtsError("WebSocketDisconnect","");
        return std::string(R"({"done":true})");
    };
    ops.send=[&](const std::string &bytes){std::lock_guard<std::mutex> lock(state.mutex);state.audio+=bytes;};
    ops.interrupt_receive=[&]{std::lock_guard<std::mutex> lock(state.mutex);state.interrupted=true;state.release_receiver=true;state.changed.notify_all();};
    ops.acquire=[&](const VoiceCancellation &)->std::unique_ptr<TtsWebSocketLease>{
        {std::lock_guard<std::mutex> lock(state.mutex);state.active=true;}
        if(acquire_failure){std::lock_guard<std::mutex> lock(state.mutex);state.active=false;throw TtsError("RuntimeError","lease acquire failure");}
        return std::make_unique<ObservedLease>(state,close_failure);
    };
    ops.state_changed=[&]{
        {std::lock_guard<std::mutex> lock(state.mutex);state.observations.push_back(state.active);if(!state.active)state.idle_notice=true;state.changed.notify_all();}
        if(observer_failure)throw std::runtime_error("observer failure");
    };
    auto run=std::async(std::launch::async,[&]{run_tts_websocket(ops);});
    struct ReleaseReceiver{LifecycleState &state;~ReleaseReceiver(){std::lock_guard<std::mutex> lock(state.mutex);state.release_receiver=true;state.changed.notify_all();}} cleanup{state};
    {
        std::unique_lock<std::mutex> lock(state.mutex);
        expect(state.changed.wait_for(lock,2s,[&]{return state.receiver_waiting&&state.idle_notice;}),"lease release is published while websocket receiver remains open");
        expect(!state.active,"observer sees released capacity, not the pre-close state");
        if(!close_failure&&!acquire_failure)expect(run.wait_for(0ms)==std::future_status::timeout,"the socket session stays open after its model lease ends");
        state.release_receiver=true;state.changed.notify_all();
    }
    expect(run.wait_for(2s)==std::future_status::ready,"websocket lifecycle finishes without a stranded receiver");
    std::string error;
    try{run.get();}catch(const TtsError &e){error=e.what();}
    expect(error==(close_failure?"lease close failure":acquire_failure?"lease acquire failure":""),"observer errors preserve the synthesis/cleanup outcome");
    expect(state.observations==(acquire_failure?std::vector<bool>{false}:std::vector<bool>{true,false}),"capacity notifications follow actual acquire/release transitions");
    expect(state.closes==(acquire_failure?0:1),"model lease is closed exactly once");
    if(!close_failure&&!acquire_failure)expect(state.audio==tts_pcm_header()+std::string(4,'\0'),"notification observers preserve the emitted PCM bytes");
}
void lifecycle_contracts(){lifecycle_case(false,false,false);lifecycle_case(true,false,false);lifecycle_case(true,true,false);lifecycle_case(true,false,true);}
}
int main(int argc,char **argv){try{
    lifecycle_contracts();
    if(argc==2&&std::string(argv[1])=="--state-notifications"){std::cout<<"WebSocket model lease state notifications and failure isolation passed\n";return 0;}
    expect(argc==3,"source and WebSocket fixture paths");gzFile file=gzopen(argv[2],"rb");expect(file,"open WebSocket fixture");std::string raw;char buffer[65536];int count;while((count=gzread(file,buffer,sizeof(buffer)))>0)raw.append(buffer,count);expect(count==0&&gzclose(file)==Z_OK,"complete WebSocket fixture");std::vector<Json> cases;std::istringstream lines(raw);for(std::string line;std::getline(lines,line);)if(!line.empty())cases.push_back(Json::parse(line));expect(hash(read(fs::path(argv[1])/"ezlocalai/TTSWebSocket.py"))==cases[0].at("source_sha256"),"unchanged WebSocket source");expect(cases.size()-1==cases[0].at("cases"),"complete WebSocket fixture");
    size_t pcm=0,input=0;
    for(size_t index=1;index<cases.size();++index){
        const auto &row=cases[index];Json error=nullptr,events=Json::array();size_t consumed=0;
        try{
            if(row.at("kind")=="pcm"){
                ++pcm;TtsPcmSegment segment;
                for(const auto &packet:row.at("packets"))segment.feed(unhex(packet),[&](bool header,const auto &bytes){events.push_back(Json::array({header?"header":"audio",hex(bytes)}));});
                segment.finish();
            }else{
                ++input;TtsWebSocketInput parser;
                for(const auto &message:row.at("messages")){
                    ++consumed;const auto result=parser.receive(message);
                    if(result.request){const auto request=Json::parse(*result.request);events.push_back(Json::array({request.at("text"),request.at("voice"),request.at("language")}));}
                    if(result.done){events.push_back(nullptr);break;}
                }
            }
        }catch(const TtsError &e){error={{"category",e.category},{"message",e.what()}};}
        const auto label="WebSocket "+std::to_string(index);
        compare_json(error,row.at("error"),label+" exception");
        compare_json(events,row.at(row.at("kind")=="pcm"?"events":"requests"),label+" emitted events");
        if(row.at("kind")=="input")compare_json(consumed,row.at("consumed"),label+" consumed messages");
    }
    std::cout<<pcm<<" source PCM cases and "<<input<<" receiver scenarios match\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
