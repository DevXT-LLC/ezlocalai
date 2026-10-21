#include "ezlocalai/tts_websocket.hpp"
#include "whisper_transcription_test.hpp"
#include <condition_variable>
#include <future>
using namespace transcription_test;
struct Session {
    std::string mode;std::mutex mutex;std::condition_variable changed;
    bool gate=false,interrupted=false,blocked=false;
    Json counts={{"received",0},{"acquire",0},{"get",0},{"stream",0},{"close",0},{"destroy",0},{"release",0}},frames=Json::array();
    void increment(const char *name){counts[name]=counts[name].get<int>()+1;changed.notify_all();}
    template<class Predicate>void wait(std::unique_lock<std::mutex> &lock,Predicate predicate){expect(changed.wait_for(lock,std::chrono::seconds(5),predicate),mode+" barrier timeout");}
    struct Audio: CttsByteStream {
        Session &s;size_t at=0;explicit Audio(Session &value):s(value){}
        std::optional<std::string> next()override{
            std::unique_lock<std::mutex> lock(s.mutex);
            if(s.mode=="queue-error")throw TtsError("RuntimeError","next failure");
            if(s.mode.rfind("queue",0)!=0)s.wait(lock,[&]{return s.counts["received"].get<int>()>=2;});
            if(at==1&&(s.mode=="next-error"||s.mode=="close-error"))throw TtsError("RuntimeError","next failure");
            if(at==1&&s.mode=="cancel-model"){s.blocked=true;s.changed.notify_all();s.wait(lock,[&]{return s.interrupted;});throw VoiceCancelled("");}
            const std::vector<std::string> packets={tts_pcm_header(),unhex("020000004100"),std::string(4,'\0')};
            if(at==packets.size())return {};return packets[at++];
        }
        void request_cancel()override{std::lock_guard<std::mutex> lock(s.mutex);s.interrupted=true;s.changed.notify_all();}
        void close()override{std::lock_guard<std::mutex> lock(s.mutex);s.increment("close");if(s.mode=="close-error")throw TtsError("RuntimeError","close failure");}
    };
    struct Lease: TtsWebSocketLease {
        Session &s;explicit Lease(Session &value):s(value){}
        std::unique_ptr<CttsByteStream> stream(const std::string &)override{std::lock_guard<std::mutex> lock(s.mutex);s.increment("stream");return std::make_unique<Audio>(s);}
        void close()override{std::lock_guard<std::mutex> lock(s.mutex);s.increment("destroy");s.increment("release");}
    };
    TtsWebSocketOperations operations(){return {
        [this]{std::unique_lock<std::mutex> lock(mutex);increment("received");const auto i=counts["received"].get<int>();
            if(mode.rfind("queue",0)==0)return i<=9?Json{{"text","part "+std::to_string(i)},{"flush",true}}.dump():Json{{"done",true}}.dump();
            if(i==1)return Json{{"text","one"},{"flush",true}}.dump();
            if(mode=="cancel-acquire"){wait(lock,[&]{return counts["acquire"]!=0;});return std::string("{\"text\":null}");}
            if(mode=="cancel-model"){wait(lock,[&]{return blocked;});return std::string("{\"text\":null}");}
            wait(lock,[&]{return interrupted;});throw VoiceCancelled("");},
        [this](const std::string &bytes){std::lock_guard<std::mutex> lock(mutex);if(mode=="send-timeout")throw TtsError("TimeoutError","");std::string hex;for(unsigned char c:bytes){hex+="0123456789abcdef"[c>>4];hex+="0123456789abcdef"[c&15];}frames.push_back(hex);},
        [this]{std::lock_guard<std::mutex> lock(mutex);interrupted=true;changed.notify_all();},
        [this](const VoiceCancellation &cancellation)->std::unique_ptr<TtsWebSocketLease>{std::unique_lock<std::mutex> lock(mutex);increment("acquire");if(mode.rfind("queue",0)==0||mode=="cancel-acquire")wait(lock,[&]{return gate||interrupted;});if(cancellation.requested())throw VoiceCancelled("");increment("get");return std::make_unique<Lease>(*this);}
    };}
};
int main(int argc,char **argv){try{
    expect(argc==3,"source and session fixture");const auto cases=records(argv[2]);expect(hash(read(fs::path(argv[1])/"ezlocalai/TTSWebSocket.py"))==cases[0].at("source_sha256"),"unchanged session source");expect(cases.size()-1==cases[0].at("cases"),"complete session fixture");
    for(size_t i=1;i<cases.size();++i){const auto &row=cases[i];Session s;s.mode=row.at("mode");Json error=nullptr;
        auto run=std::async(std::launch::async,[&]{try{run_tts_websocket(s.operations());}catch(const TtsError &e){error={{"category",e.category},{"message",e.what()}};}});
        if(s.mode.rfind("queue",0)==0){
            std::unique_lock<std::mutex> lock(s.mutex);s.wait(lock,[&]{return s.counts["received"].get<int>()>=6;});
            // One item is held by synthesis, four await synthesis, and the
            // sixth receive is blocked on the bounded queue's next insertion.
            expect(!s.changed.wait_for(lock,std::chrono::milliseconds(25),[&]{return s.counts["received"].get<int>()>6;}),"bounded queue stops receiver");
            compare_json(Json{{"received",s.counts["received"]},{"pending",4},{"models",s.counts["get"]}},row.at("before"),s.mode+" backpressure");s.gate=true;s.changed.notify_all();
        }
        expect(run.wait_for(std::chrono::seconds(10))==std::future_status::ready,s.mode+" session completion");run.get();
        compare_json(error,row.at("error"),s.mode+" error precedence");compare_json(s.frames,row.at("frames"),s.mode+" wire audio");
        if(s.mode=="queue"){
            // Thread scheduling changes where the ready queue becomes empty;
            // each resulting batch must still own exactly one complete lease.
            const int leases=s.counts["acquire"];expect(leases>=1&&leases<=9,"batch lease count");for(const auto *key:{"get","destroy","release"})expect(s.counts[key]==leases,"balanced batch ownership");
            for(const auto *key:{"received","stream","close"})compare_json(s.counts[key],row.at("counts").at(key),std::string("queue ")+key);
        }else compare_json(s.counts,row.at("counts"),s.mode+" lifecycle");
    }
    std::cout<<cases.size()-1<<" complete source sessions match backpressure, cancellation, audio and cleanup\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
