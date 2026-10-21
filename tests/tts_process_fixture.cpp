#include <nlohmann/json.hpp>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <signal.h>
#include <unistd.h>
using Json=nlohmann::ordered_json;
void word(std::ofstream &out,uint32_t value,int bytes) { for(int i=0;i<bytes;++i) out.put(static_cast<char>(value>>(8*i))); }
void wav(const std::string &path,int channels,bool pcm16) {
    const float samples[]={0.f,0.25f,-0.5f,0.75f,-1.f,0.125f}; const int width=pcm16?2:4; std::ofstream out(path,std::ios::binary); out.write("RIFF",4); word(out,36+6*width,4); out.write("WAVEfmt ",8); word(out,16,4); word(out,pcm16?1:3,2); word(out,channels,2); word(out,24000,4); word(out,24000*channels*width,4); word(out,channels*width,2); word(out,width*8,2); out.write("data",4); word(out,6*width,4);
    for(float sample:samples) { uint32_t bits; if(pcm16) bits=static_cast<uint16_t>(static_cast<int16_t>(sample*32768)); else std::memcpy(&bits,&sample,4); word(out,bits,width); }
}
int main(int argc,char **argv) {
    const auto *file=std::getenv("EZLOCALAI_TTS_FIXTURE"); if(!file) return 2; Json config; std::ifstream(file)>>config; const std::string trace=config.at("trace");
    auto record=[&](const Json &item) { std::ofstream(trace,std::ios::app)<<item.dump()<<'\n'; };
    Json environment=Json::object(); for(const auto *name:{"CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","GGML_CUDA_DISABLE_GRAPHS"}) if(const auto *value=std::getenv(name)) environment[name]=value;
    struct sigaction action{}; sigaction(SIGPIPE,nullptr,&action);
    record(Json{{"pid",getpid()},{"arguments",std::vector<std::string>(argv,argv+argc)},{"environment",environment},{"sigpipe_default",action.sa_handler==SIG_DFL}});
    auto emit=[](const Json &lines) { for(const auto &line:lines) std::cout<<(line.is_string()?line.get<std::string>():line.dump())<<std::endl; };
    const auto hang=[] { for(;;) std::this_thread::sleep_for(std::chrono::seconds(1)); };
    if(config.value("startup_hang",false)) hang();
    emit(config.value("ready",Json::array({Json{{"ready",true}}}))); size_t index=0;
    for(std::string line;std::getline(std::cin,line);) {
        const auto request=Json::parse(line); auto recorded=request; if(recorded.contains("output")) recorded["output"]="<OUTPUT>"; recorded["speaker"]="<SPEAKER>"; record(recorded);
        const auto actions=config.value("actions",Json::array()); const auto action=index<actions.size()?actions[index]:Json::object(); ++index;
        if(action.value("exit",false)) return 0; if(action.value("hang",false)) hang();
        if(action.contains("wait_file")) while(access(action.at("wait_file").get<std::string>().c_str(),F_OK)) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if(request.contains("output") && action.value("write",true)) wav(request.at("output"),action.value("channels",1),action.value("pcm16",false));
        emit(action.value("responses",request.value("stream",false)?Json::array({Json{{"pcm_f32","AAAAAAAAgD4AAAC/"},{"sample_rate",24000}},Json{{"pcm_f32","AABAPwAAgL8AAAA+"},{"sample_rate",24000}},Json{{"ok",true}}}):Json::array({Json{{"ok",true}}})));
    }
    return 0;
}
