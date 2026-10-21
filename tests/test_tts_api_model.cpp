#include "whisper_transcription_test.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/process.hpp"
#include "fixture_ports.hpp"
#include "tts_ws_client.hpp"
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <openssl/evp.h>
#include "ezlocalai/stt_input.hpp"
#include "ezlocalai/stt_runtime.hpp"
using namespace transcription_test;
struct Reply {int status=0;std::map<std::string,std::string> headers;std::string body;std::vector<std::string> chunks;};
Reply http(int port,const std::string &path,const std::string &body=""){
    int fd=socket(AF_INET,SOCK_STREAM|SOCK_CLOEXEC,0);expect(fd>=0,"HTTP socket");struct Close{int fd;~Close(){close(fd);}}closer{fd};sockaddr_in address{};address.sin_family=AF_INET;address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);address.sin_port=htons(port);if(connect(fd,reinterpret_cast<sockaddr *>(&address),sizeof(address)))return {};
    timeval timeout{300,0};setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));const auto request=(body.empty()?"GET ":"POST ")+path+" HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: "+std::to_string(body.size())+"\r\n\r\n"+body;
    for(size_t sent=0;sent<request.size();){const auto n=send(fd,request.data()+sent,request.size()-sent,MSG_NOSIGNAL);expect(n>0,"send HTTP request");sent+=n;}
    std::string raw;char buffer[65536];for(;;){const auto n=recv(fd,buffer,sizeof(buffer),0);if(n<0&&errno==EINTR)continue;expect(n>=0,"read HTTP response");if(!n)break;raw.append(buffer,n);}const auto boundary=raw.find("\r\n\r\n");expect(boundary!=std::string::npos,"HTTP headers");Reply reply;std::istringstream input(raw.substr(0,boundary));std::string version,line;input>>version>>reply.status;std::getline(input,line);while(std::getline(input,line)){if(!line.empty()&&line.back()=='\r')line.pop_back();const auto split=line.find(':');if(split==std::string::npos)continue;auto key=line.substr(0,split),value=line.substr(split+1);for(auto &c:key)c=std::tolower(static_cast<unsigned char>(c));while(!value.empty()&&value[0]==' ')value.erase(0,1);reply.headers[key]=value;}
    if(reply.headers["transfer-encoding"]=="chunked"){size_t at=boundary+4;for(;;){const auto end=raw.find("\r\n",at);expect(end!=std::string::npos,"HTTP stream terminator");const auto n=std::stoul(raw.substr(at,end-at),nullptr,16);at=end+2;if(!n)break;expect(at+n+2<=raw.size(),"HTTP stream packet");reply.chunks.push_back(raw.substr(at,n));reply.body+=reply.chunks.back();at+=n+2;}}
    else reply.body=raw.substr(boundary+4);return reply;
}
int main(int argc,char **argv){try{
    expect(argc==6||argc==8,"server, source model capture, worker, models, speaker and optional STT model/VAD");const auto server=fs::absolute(argv[1]),worker=fs::absolute(argv[3]),models=fs::absolute(argv[4]),speaker=fs::absolute(argv[5]);const auto cases=records(argv[2]);const auto &header=cases[0],&scenario=cases[1];
    expect(hash(read(speaker))==header.at("speaker_sha256"),"pinned speaker");
    Temporary temporary;struct Restore{fs::path path=fs::current_path();~Restore(){fs::current_path(path);}}restore;fs::current_path(temporary.path);fs::create_directories("voices");fs::copy_file(speaker,"voices/default.wav");
    for(const auto &entry:environment_defaults)setenv(std::string(entry.first).c_str(),std::string(entry.second).c_str(),1);
    for(const auto &[key,value]:header.at("environment").items())setenv(key.c_str(),value.get<std::string>().c_str(),1);
    for(const auto &[key,value]:std::map<std::string,std::string>{{"EZLOCALAI_NATIVE_BACKEND","echo"},{"EZLOCALAI_API_KEY",""},{"DEFAULT_MODEL","none"},{"CUDA_VISIBLE_DEVICES",""},{"LAZY_LOAD_VOICE","false"},{"VOICE_SERVER",""},{"STT_ENABLED","false"},{"TTS_ENABLED","true"},{"TTS_N_PARALLEL","1"},{"TTS_UNLOAD_LLM_DURING_GENERATION","false"},{"QWEN_TTS_ENABLE_GTTS_FALLBACK","false"},{"QWEN_TTS_ALLOW_CPU_FALLBACK","true"},{"QWEN_TTS_BIN",worker.string()},{"QWEN_TTS_MODEL_FILE",(models/"qwen-talker-0.6b-base-Q8_0.gguf").string()},{"QWEN_TTS_MMPROJ_FILE",(models/"qwen-tokenizer-12hz-f16.gguf").string()}})setenv(key.c_str(),value.c_str(),1);
    if(argc==8){setenv("STT_ENABLED","true",1);setenv("STT_UNLOAD_LLM_DURING_GENERATION","false",1);setenv("WHISPER_MODEL",fs::absolute(argv[6]).c_str(),1);setenv("WHISPER_VAD_DIRECTORY",fs::absolute(argv[7]).c_str(),1);setenv("OPENBLAS_NUM_THREADS","1",1);setenv("EZLOCALAI_URL","",1);}
    const int port=fixture_ports();const auto port_text=std::to_string(port);const auto log=temporary.path/"server.log";const pid_t pid=fork();expect(pid>=0,"fork native server");if(!pid){const int fd=open(log.c_str(),O_CREAT|O_WRONLY|O_TRUNC,0600);dup2(fd,1);dup2(fd,2);close(fd);execl(server.c_str(),server.c_str(),"--host","127.0.0.1","--port",port_text.c_str(),nullptr);_exit(127);}
    struct Child{pid_t pid;~Child(){if(pid>0){kill(pid,SIGKILL);while(waitpid(pid,nullptr,0)<0&&errno==EINTR){}}}}child{pid};
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(180);bool ready=false;while(std::chrono::steady_clock::now()<deadline){if(http(port,"/health").status==200){ready=true;break;}int status;if(waitpid(pid,&status,WNOHANG)==pid){child.pid=-1;throw std::runtime_error("server exited: "+read(log));}std::this_thread::sleep_for(std::chrono::milliseconds(25));}expect(ready,"TTS startup: "+read(log));
    for(size_t i=0;i<2;++i){const auto reply=http(port,"/v1/audio/speech",R"({"input":"Hello."})");expect(reply.status==200&&reply.headers.at("content-type")=="audio/wav","buffered speech status: "+reply.body+" "+read(log));compare_json(Json{{"size",reply.body.size()},{"sha256",hash(reply.body)}},scenario.at("outcomes")[i].at("value"),"real HTTP buffered audio and cache hit");}
    const auto reply=http(port,"/v1/audio/speech/stream",R"({"input":"Hello. Good day."})");expect(reply.status==200&&reply.headers.at("content-type")=="application/octet-stream","stream HTTP status");expect(reply.headers.at("x-audio-format")=="pcm"&&reply.headers.at("x-sample-rate")=="24000"&&reply.headers.at("x-bits-per-sample")=="16"&&reply.headers.at("x-channels")=="1","source stream metadata");
    std::vector<Json> expected;for(size_t i=3;i<scenario.at("actions").size();++i){if(scenario.at("actions")[i].at("op")!="next")break;const auto &value=scenario.at("outcomes")[i].at("value");if(value.is_null())break;expected.push_back(value);}expect(reply.chunks.size()==expected.size(),"source stream packet count");for(size_t i=0;i<expected.size();++i)compare_json(Json{{"size",reply.chunks[i].size()},{"sha256",hash(reply.chunks[i])}},expected[i],"HTTP stream packet "+std::to_string(i));
    const auto ws_cases=records(fs::path(argv[2]).parent_path()/"tts_ws_model_cases.jsonl");expect(ws_cases[0].at("model_hashes")==header.at("model_hashes")&&ws_cases[0].at("environment")==header.at("environment"),"WebSocket model capture uses identical pins");
    tts_ws_test::Client ws(port);ws.message(R"({"text":"Hello. Good day.","voice":"default","language":"auto","done":true})");Json ws_frames=Json::array();for(;;){const auto frame=ws.frame();if(frame.first==8){expect(frame.second==unhex("03e8"),"normal WebSocket completion");break;}expect(frame.first==2,"binary speech frame");ws_frames.push_back(Json{{"size",frame.second.size()},{"sha256",hash(frame.second)}});}compare_json(ws_frames,ws_cases[1].at("frames"),"real WebSocket frame hashes");ws.disconnect();
    size_t pipeline_cases=0;
    if(argc==8){
        const auto captures=records(fs::path(argv[2]).parent_path()/"tts_audio_model_cases.jsonl");expect(captures[0].at("model_hashes")==header.at("model_hashes"),"same Qwen pipeline pins");for(const auto &[file,digest]:captures[0].at("stt_pins").at("models").at("en").items())expect(hash(read(fs::path(argv[6])/file))==digest,"pinned STT pipeline model");
        expect(captures[0].at("stt_pins").at("version")=="1.2.0","matching native faster-whisper profile");expect(stt_cpu_threads([]{return std::thread::hardware_concurrency();})==captures[0].at("stt_pins").at("cpu_threads"),"matching native STT thread count");
        for(const auto *file:{"silero_encoder_v5.onnx","silero_decoder_v5.onnx"})expect(hash(read(fs::path(argv[7])/file))==captures[0].at("faster_whisper_files").at(std::string("assets/")+file),"matching source/native Silero VAD assets");
        const auto encode=[](const std::string &bytes){std::string result(4*((bytes.size()+2)/3)+1,'\0');const auto n=EVP_EncodeBlock(reinterpret_cast<unsigned char *>(result.data()),reinterpret_cast<const unsigned char *>(bytes.data()),bytes.size());expect(n>=0,"encode pipeline audio");result.resize(n);return result;};
        for(size_t i=1;i<captures.size();++i){const auto &row=captures[i];const auto data=read(speaker);const auto payload="data:audio/wav;base64,"+encode(row.at("double").get<bool>()?encode(data):data);const auto response=http(port,"/v1/audio/speech",Json{{"input",payload}}.dump());
            if(row.contains("error"))expect(response.status==500&&response.body=="Internal Server Error","source double-decode failure: "+response.body);
            else{expect(response.status==200,"local audio pipeline status: "+response.body+" "+read(log));const auto value=Json::parse(response.body).get<std::string>();const auto wav=stt_base64_decode(Json(value).dump());compare_json(Json{{"value_size",value.size()},{"audio_size",wav.size()},{"audio_sha256",hash(wav)}},Json{{"value_size",row.at("value_size")},{"audio_size",row.at("audio_size")},{"audio_sha256",row.at("audio_sha256")}},"real STT-to-TTS pipeline WAV");}++pipeline_cases;
        }
    }
    kill(pid,SIGTERM);int status=0;expect(waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"graceful native server shutdown");child.pid=-1;
    std::cout<<"Real native executable matches two source WAV hashes, "<<expected.size()<<" HTTP packets and "<<ws_frames.size()<<" WebSocket frame hashes and "<<pipeline_cases<<" real data-URI pipeline cases over TCP\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
