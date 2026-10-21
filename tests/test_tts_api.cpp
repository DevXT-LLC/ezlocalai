#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "whisper_transcription_test.hpp"
#include <sys/socket.h>
#include <future>
using namespace transcription_test;
std::string bytes_hex(const std::string &value){std::ostringstream out;for(unsigned char c:value)out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(c);return out.str();}
struct Audio:CttsByteStream {
    Json config;Json &calls;size_t at=0;
    Audio(Json config,Json &calls):config(std::move(config)),calls(calls){}
    std::optional<std::string> next()override{calls.push_back({{"op","next"}});if(int(at)==config.value("stream_error_at",-1))throw TtsError("RuntimeError","stream failure");const auto packets=config.value("packets",Json::array({"00000000","410042","00000000"}));if(at==packets.size())return {};return unhex(packets[at++]);}
    void close()override{calls.push_back({{"op","close"}});}
    void request_cancel()override{}
};
int main(int argc,char **argv){try{
    expect(argc==3,"source and API fixture paths");const auto cases=records(argv[2]);expect(hash(read(fs::path(argv[1])/"app.py"))==cases[0].at("source_sha256"),"unchanged full speech endpoints");expect(cases.size()-1==cases[0].at("cases"),"all source HTTP cases");
    Temporary temporary;struct Restore{fs::path path=fs::current_path();~Restore(){fs::current_path(path);}}restore;fs::current_path(temporary.path);
    size_t compared=0;
    for(size_t index=1;index<cases.size();++index)for(bool deferred:{false,true}){
        const auto &row=cases[index];const auto config=row.at("config");Json calls=Json::array();std::vector<std::function<void()>> pending;
        for(const auto &entry:environment_defaults)setenv(std::string(entry.first).c_str(),std::string(entry.second).c_str(),1);
        for(const auto &[key,value]:std::map<std::string,std::string>{{"EZLOCALAI_API_KEY",""},{"DEFAULT_MODEL","fixture-text"},{"TTS_UNLOAD_LLM_DURING_GENERATION","false"},{"TTS_N_PARALLEL","1"},{"LAZY_LOAD_VOICE","true"},{"VOICE_SERVER",""},{"ROUTER_URL",""},{"EZLOCALAI_FALLBACK_URL",""}})setenv(key.c_str(),value.c_str(),1);
        const auto configured_environment=config.value("environment",Json::object());for(const auto &[key,value]:configured_environment.items())setenv(key.c_str(),value.get<std::string>().c_str(),1);
        auto probes=native_resource_probes();probes.gpu_count=[]{return 0;};probes.total_per_gpu=[]{return std::vector<double>{};};probes.system_ram=[]{return 16.;};auto resources=std::make_shared<ResourceManager>(probes);
        TtsPoolFactory factory=[&](auto manager){auto ops=tts_pool_resources({},manager);ops.can_load=[]{return ModelAdmission{true,"cpu","fixture"};};ops.schedule_cleanup=[&](auto work){pending.push_back(std::move(work));};ops.load=[&](const auto &){if(config.contains("get_error"))throw TtsError("RuntimeError",config.at("get_error"));auto model=std::make_shared<TtsPoolModel>();model->device=[]{return "cpu";};model->close=[]{};model->generate=[&](const auto &request){calls.push_back({{"op","generate"},{"request",Json::parse(request)}});if(config.contains("generate_error"))throw TtsError("RuntimeError",config.at("generate_error"));return config.value("audio",std::string("UklGRgAB"));};model->stream=[&](const auto &request){calls.push_back({{"op","stream"},{"request",Json::parse(request)}});return std::make_unique<Audio>(config,calls);};return model;};return ops;};
        TtsRequestOperations request;request.fallback_generate=[&](const auto &request){const auto data=Json::parse(request);calls.push_back({{"op","fallback"},{"request",{{"text",data.at("text")},{"language",data.at("language")}}}});return "fallback WAV";};
        ezl_env_config environment{};ezl_load_config(&environment);NativeServer server(environment,std::make_unique<EchoInferenceBackend>("fixture-text"),resources,{},factory,request,true);
        auto result=server.handle_request("POST",row.at("path"),"",row.at("body"),"application/json","","",{},deferred);bool aborted=result.transport_error;
        if(result.stream_writer){int sockets[2];expect(socketpair(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0,sockets)==0,"stream socket pair");result.stream_writer(sockets[0]);shutdown(sockets[0],SHUT_WR);std::string wire;char buffer[4096];for(ssize_t n;(n=read(sockets[1],buffer,sizeof(buffer)))>0;)wire.append(buffer,n);close(sockets[0]);close(sockets[1]);size_t at=0;result.body.clear();aborted=true;while(at<wire.size()){const auto end=wire.find("\r\n",at);expect(end!=std::string::npos,"HTTP chunk length");const auto size=std::stoul(wire.substr(at,end-at),nullptr,16);at=end+2;if(!size){aborted=false;break;}expect(at+size+2<=wire.size(),"complete HTTP chunk");result.body+=wire.substr(at,size);at+=size+2;}}
        const auto label="API "+std::to_string(index)+(deferred?" wire":" buffered");expect(result.status==row.at("status"),label+" status "+std::to_string(result.status));Json headers={{"content-type",result.content_type}};for(auto [key,value]:result.headers){for(auto &c:key)c=std::tolower(static_cast<unsigned char>(c));headers[key]=value;}if(!result.chunked)headers["content-length"]=std::to_string(result.body.size());
        if(result.content_type=="application/json"){compare_json(Json::parse(result.body),Json::parse(unhex(row.at("hex"))),label+" JSON response");headers.erase("content-length");auto expected=row.at("headers");expected.erase("content-length");compare_json(headers,expected,label+" headers");}else{compare_json(bytes_hex(result.body),row.at("hex"),label+" bytes");compare_json(headers,row.at("headers"),label+" headers");}
        compare_json(aborted,row.at("aborted"),label+" transfer completion");compare_json(calls,row.at("calls"),label+" model calls");expect(server.tts_permits()->available==1&&server.tts_pool_snapshot()->leases.empty(),label+" permit and leases released");expect(resources->get_model_active_count(ModelType::Tts)==0,label+" usage released");for(auto &work:pending)work();pending.clear();++compared;
    }
    std::cout<<cases.size()-1<<" full source HTTP scenarios and "<<compared<<" native buffered/deferred comparisons match\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
