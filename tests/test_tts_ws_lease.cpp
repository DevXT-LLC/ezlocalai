#include "ezlocalai/tts_websocket.hpp"
#include "tts_service_test.hpp"
template<class Body,class Cleanup>void finally(Body body,Cleanup cleanup){std::exception_ptr error;try{body();}catch(...){error=std::current_exception();}cleanup();if(error)std::rethrow_exception(error);}
int main(int argc,char **argv){try{
    expect(argc==3,"source and lease fixture paths");gzFile file=gzopen(argv[2],"rb");expect(file,"open WebSocket lease fixture");std::string raw;char buffer[65536];int count;while((count=gzread(file,buffer,sizeof(buffer)))>0)raw.append(buffer,count);expect(count==0&&gzclose(file)==Z_OK,"complete lease fixture");std::vector<Json> cases;std::istringstream lines(raw);for(std::string line;std::getline(lines,line);)if(!line.empty())cases.push_back(Json::parse(line));
    expect(hash(read(fs::path(argv[1])/"ezlocalai/TTSWebSocket.py"))==cases[0].at("source_sha256")&&hash(read(fs::path(argv[1])/"Pipes.py"))==cases[0].at("pool_sha256"),"unchanged WebSocket synthesis, pool and guard");expect(cases.size()-1==cases[0].at("cases"),"all lease scenarios");size_t operations=0;
    for(size_t index=1;index<cases.size();++index){
        const auto &row=cases[index];Environment env(row.at("config"));TtsService service(env.config.value("capacity",2),env.pool(),env.voice(),std::make_shared<VoicePermitQueue>(1,true),env.resources,env.request());size_t step=0;uint64_t key=0;
        for(const auto &action:row.at("actions")){
            Json actual=Json::object();key=action.value("key",key);
            try{
                const auto op=action.at("op");
                if(op=="batch"){
                    bool header_sent=false;auto lease=service.websocket_lease({},key);
                    finally([&]{for(auto request:action.at("requests")){
                        if(!request.contains("voice"))request["voice"]="default";if(!request.contains("language"))request["language"]="en";
                        auto stream=lease->stream(request.dump());
                        finally([&]{TtsPcmSegment segment;while(auto packet=stream->next())segment.feed(*packet,[&](bool header,const auto &bytes){if(!header||!header_sent){env.event("send",{{"hex",hex(bytes)}});if(header)header_sent=true;}});segment.finish();},[&]{stream->close();});
                    }},[&]{lease->close();});
                    if(!header_sent)env.event("send",{{"hex",hex(tts_pcm_header())}});env.event("send",{{"hex","00000000"}});env.event("send",{{"hex",""}});
                }else if(op=="drain"){while(!env.pending.empty()){auto work=std::move(env.pending.front());env.pending.erase(env.pending.begin());work();}}
                else if(op=="destroy")service.unload(key);
                else throw std::runtime_error("unknown lease action");
            }catch(const TtsError &e){actual["error"]={{"category",e.category},{"message",e.what()}};}
            actual["state"]=env.state(service);compare_json(actual,row.at("outcomes")[step],"lease "+std::to_string(index)+" operation "+std::to_string(step));++step;++operations;
        }
        compare_json(env.events,row.at("events"),"lease "+std::to_string(index)+" events");
    }
    std::cout<<cases.size()-1<<" unchanged-source WebSocket lease scenarios, "<<operations<<" operations\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
