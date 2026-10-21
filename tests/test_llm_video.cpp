#include "ezlocalai/llm_input.hpp"
#include "ezlocalai/llm_response.hpp"
#include "ezlocalai/stt_input.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "fixture_ports.hpp"
#include <arpa/inet.h>
#include <csignal>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>
#include <sstream>
#include <set>
#include <thread>
#include <unistd.h>
namespace fs = std::filesystem;
namespace {
void expect(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::string read(const fs::path &path) { std::ifstream in(path, std::ios::binary); return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()}; }
std::string hex(const std::string &value) { std::ostringstream out; for (unsigned char c : value) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c); return out.str(); }
std::string hash(const std::string &value) { unsigned char bytes[32]; SHA256(reinterpret_cast<const unsigned char *>(value.data()), value.size(), bytes); return hex({reinterpret_cast<const char *>(bytes),32}); }
std::string unhex(const std::string &value) { std::string out; for (size_t i = 0; i < value.size(); i += 2) out += static_cast<char>(std::stoi(value.substr(i,2), nullptr,16)); return out; }
std::string quote(const std::string &value) { std::string out = "\""; for (unsigned char c : value) {
    if (c == '"' || c == '\\') { out += '\\'; out += c; } else if (c == '\n') out += "\\n"; else if (c == '\r') out += "\\r"; else if (c == '\t') out += "\\t";
    else if (c < 32) out += "\\u00" + hex({static_cast<char>(c)}); else out += c;
} return out + '"'; }
std::string number(double value) { if (std::isnan(value)) return "NaN"; if (std::isinf(value)) return value < 0 ? "-Infinity" : "Infinity"; std::ostringstream out; out << std::setprecision(17) << value; auto text = out.str(); if (text.find_first_of(".eE") == std::string::npos) text += ".0"; return text; }
std::string array(const std::vector<std::string> &values) { std::string out = "["; for (const auto &value : values) { if (out.size() > 1) out += ','; out += value; } return out + ']'; }
std::vector<double> numbers(std::string value) { for (auto &c : value) if (c == '[' || c == ']' || c == ',') c = ' '; std::istringstream in(value); std::vector<double> out; for (double n; in >> n;) out.push_back(n); return out; }
std::vector<std::string> columns(const std::string &line) { std::vector<std::string> out; size_t at=0; for (;;) { const auto end=line.find('\t',at); const auto value=line.substr(at,end==std::string::npos?end:end-at); out.push_back(out.empty()?value:unhex(value)); if(end==std::string::npos)return out;at=end+1; } }
std::vector<std::vector<std::string>> rows(const fs::path &reference, const fs::path &fixture) {
    std::istringstream input(read(fixture)); std::vector<std::vector<std::string>> result;
    for (std::string line; std::getline(input,line);) {
        if(line.empty())continue;
        if(line.front()=='#') { const auto at=line.find(" sha256="); if(at!=std::string::npos)expect(hash(read(reference/line.substr(2,at-2)))==line.substr(at+8),"video reference changed"); continue; }
        result.push_back(columns(line));
    }
    return result;
}
std::string encode(const std::string &bytes) {
    static constexpr char abc[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"; std::string out;
    for(size_t i=0;i<bytes.size();i+=3) { uint32_t n=static_cast<unsigned char>(bytes[i])<<16; if(i+1<bytes.size())n|=static_cast<unsigned char>(bytes[i+1])<<8; if(i+2<bytes.size())n|=static_cast<unsigned char>(bytes[i+2]);
        out+=abc[n>>18];out+=abc[(n>>12)&63];out+=i+1<bytes.size()?abc[(n>>6)&63]:'=';out+=i+2<bytes.size()?abc[n&63]:'='; }
    return out;
}
void policies(const fs::path &reference, const fs::path &fixture) {
    size_t pipeline=0,format=0,selection=0;
    for(const auto &c:rows(reference,fixture)) {
        std::string category="OK",message,result="null";
        if(c[0]=="P") {
            ++pipeline;expect(c.size()==13,"video prefix columns");std::vector<std::string> events;
            ezlocalai::LlmInputOperations ops;ops.local_uri="http://local";ops.video_fps=c[4];ops.video_max_frames=c[5];
            ops.load=[&]{events.push_back("[\"load\"]");return ezlocalai::LlmInputModel{"source",c[3]=="true"};};
            ops.extract_video=[&](const std::string &url,double fps,int64_t maximum){
                events.push_back("[\"extract\","+url+','+number(fps)+','+std::to_string(maximum)+']');
                if(c[8]=="extract")throw ezlocalai::LlmResponseError{"RuntimeError","extract failed"};
                ezlocalai::LlmInputVideo video;video.duration=2.65;video.timestamps={0,1.25};
                if(c[6].rfind("empty",0)!=0)video.frames={"data:image/jpeg;base64,YQ==","data:image/jpeg;base64,Yg=="};
                if(c[6].find("audio")!=std::string::npos)video.audio_path="fixture-audio.wav";
                return video;
            };
            ops.read_audio=[&](const std::string &path){events.push_back("[\"read\","+quote(path)+']');if(c[8]=="read")throw ezlocalai::LlmResponseError{"RuntimeError","read failed"};return "audio bytes";};
            ops.remove_audio=[&](const std::string &path){events.push_back("[\"remove\","+quote(path)+']');if(c[8]=="remove")throw ezlocalai::LlmResponseError{"RuntimeError","remove failed"};};
            ops.transcribe_segments=[&](const std::string &audio,const std::string &kind){events.push_back("[\"transcribe\","+audio+','+kind+",true]");if(c[8]=="stt")throw ezlocalai::LlmResponseError{"RuntimeError","stt failed"};return c[7];};
            ops.describe=[&](const auto &images,const auto &text){events.push_back("[\"describe\","+array(images)+','+quote(text)+']');return "null";};
            try{result=ezlocalai::llm_prepare_text_image_input(c[1],c[2]=="true",ops);}catch(const ezlocalai::LlmResponseError &error){category=error.category;message=error.message;}
            const auto label="video pipeline "+std::to_string(pipeline)+' '+c[1];
            expect(quote(category)==c[9]&&quote(message)==c[10],label+" error "+category+' '+message+" expected "+c[9]+' '+c[10]);
            expect(result==c[11],label+" result "+result+" expected "+c[11]);
            expect(array(events)==c[12],label+" events "+array(events)+" expected "+c[12]);
        } else if(c[0]=="F") {
            ++format;expect(c.size()==5,"transcript columns");
            try{result=quote(ezlocalai::llm_video_transcript(c[1]));}catch(const ezlocalai::LlmResponseError &error){category=error.category;message=error.message;}
            expect(quote(category)==c[2]&&quote(message)==c[3]&&result==c[4],"video transcript "+std::to_string(format)+' '+c[1]+" got "+category+' '+message+' '+result+" expected "+c[2]+' '+c[3]+' '+c[4]);
        } else {
            ++selection;expect(c[0]=="S"&&c.size()==9,"selection columns");std::vector<double> actual;
            try{const auto plan=ezlocalai::llm_video_frame_plan(std::stod(c[1]),std::stoll(c[2]),std::stoll(c[3]),numbers(c[4]),c[5]=="true");for(const auto &[frame,time]:plan){actual.push_back(frame);actual.push_back(time);}}
            catch(const std::runtime_error &error){category="ZeroDivisionError";message=error.what();}
            expect(quote(category)==c[6]&&quote(message)==c[7]&&(category!="OK"||actual==numbers(c[8])),"video frame plan "+std::to_string(selection));
        }
    }
    expect(pipeline==776&&format==492&&selection==1440,"all video policy fixtures");
    std::cout<<pipeline<<" pipeline, "<<format<<" transcript and "<<selection<<" frame-plan cases passed\n";
}
void media(const fs::path &reference,const fs::path &fixture) {
    size_t count=0;
    for(const auto &c:rows(reference,fixture)) {
        expect(c[0]=="M"&&c.size()==9,"media columns");++count;
        const auto source="data:video/mp4;base64,"+encode(read(fixture.parent_path()/"llm_video_assets"/c[1]));
        auto video=ezlocalai::extract_llm_video(source,std::stod(c[2]),std::stoll(c[3]),{},c[4]=="true");
        std::vector<std::string> hashes;for(const auto &frame:video.frames)hashes.push_back(quote(hash(ezlocalai::stt_audio_input(quote(frame.substr(frame.find(',')+1)),"\"wav\"").bytes)));
        const auto audio=video.audio_path.empty()?"":hash(read(video.audio_path));if(!video.audio_path.empty())fs::remove(video.audio_path);
        const auto label="real video "+c[1]+" budget "+c[3]+" scene "+c[4];
        expect(video.duration==std::stod(c[5])&&video.timestamps==numbers(c[6]),label+" decoded duration/timestamps");
        expect(array(hashes)==c[7],label+" JPEG bytes "+array(hashes)+" expected "+c[7]);
        expect(audio==c[8],label+" audio WAV bytes "+audio+" expected "+c[8]);
    }
    expect(count==25,"all real video cases");
    for(const auto &source:{"local.mp4","file:///tmp/private.mp4","data:bad","data:video/mp4;base64,a","data:video/mp4;base64,é"}) {
        auto result=ezlocalai::extract_llm_video(source,1,16,[](auto){throw std::runtime_error("not a remote URL");return "";});
        expect(result.frames.empty()&&result.audio_path.empty()&&result.duration==0,"invalid video is omitted");
    }
    std::cout<<count<<" real video JPEG, timestamp, duration and audio-byte cases passed\n";
}
int port() { return fixture_ports(); }
class RemoteVideo {
    int listener_;
    std::atomic<bool> running_{true};
    std::thread thread_;
public:
    std::string url;
    std::atomic<int> downloads{0};
    std::atomic<bool> user_agent{false};
    RemoteVideo(std::string bytes,bool fail) {
        listener_=socket(AF_INET,SOCK_STREAM,0);sockaddr_in address{};address.sin_family=AF_INET;address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
        expect(!bind(listener_,reinterpret_cast<sockaddr *>(&address),sizeof(address))&&!listen(listener_,8),"video HTTP bind");socklen_t size=sizeof(address);getsockname(listener_,reinterpret_cast<sockaddr *>(&address),&size);
        url="http://127.0.0.1:"+std::to_string(ntohs(address.sin_port))+"/redirect";
        thread_=std::thread([this,bytes=std::move(bytes),fail]{
            while(running_) {
                const auto peer=accept(listener_,nullptr,nullptr);if(peer<0)break;std::string request;char buffer[2048];
                while(request.find("\r\n\r\n")==std::string::npos){const auto n=recv(peer,buffer,sizeof(buffer),0);if(n<=0)break;request.append(buffer,n);}
                user_agent=request.find("User-Agent: Mozilla/5.0\r\n")!=std::string::npos;
                const bool redirect=request.rfind("GET /redirect ",0)==0;if(!redirect)++downloads;
                const auto status=redirect?"302 Found":fail?"404 Missing":"200 OK";
                const auto body=redirect?std::string():bytes;
                const auto response="HTTP/1.1 "+std::string(status)+"\r\n"+(redirect?"Location: /video\r\n":"")+"Content-Type: video/mp4\r\nContent-Length: "+std::to_string(body.size())+"\r\nConnection: close\r\n\r\n"+body;
                for(size_t at=0;at<response.size();){const auto n=send(peer,response.data()+at,response.size()-at,MSG_NOSIGNAL);if(n<=0)break;at+=n;}close(peer);
            }
        });
    }
    ~RemoteVideo(){running_=false;shutdown(listener_,SHUT_RDWR);close(listener_);thread_.join();}
};
void managed(const fs::path &assets) {
    char pattern[]="/tmp/ezlocalai-video-managed-XXXXXX";const auto path=mkdtemp(pattern);expect(path,"temporary video directory");const fs::path root(path);
    for(int mode=0;mode<7;++mode) {
        for(const auto &entry:ezlocalai::environment_defaults)unsetenv(entry.first.data());
        for(const auto *key:{"LLAMA_CPP_SERVER_URL","LLAMA_CPP_MODEL_PATHS","LLAMA_CPP_CONTEXT","LLM_AUDIO_FIXTURE_EXIT","LLM_AUDIO_FIXTURE_BLOCK"})unsetenv(key);
        setenv("EZLOCALAI_DOTENV_PATH","",1);setenv("EZLOCALAI_API_KEY","none",1);setenv("DEFAULT_MODEL","ordinary",1);setenv("GPU_LAYERS","0",1);setenv("LLM_MAX_TOKENS","512",1);setenv("SPECULATIVE_TYPE","none",1);
        const auto folder=root/std::to_string(mode);fs::create_directories(folder);std::ofstream(folder/"model.gguf")<<"fixture";std::ofstream(folder/"whisper.bin")<<"fixture";
        const auto suffix='-'+std::to_string(port());
        setenv("LLAMA_CPP_SERVER_PORT",suffix.substr(1).c_str(),1);setenv("LLAMA_CPP_SERVER_BIN",TEST_EMBEDDING_FIXTURE_BIN,1);setenv("LLAMA_CPP_START_TIMEOUT","2",1);
        setenv("LLAMA_CPP_MODEL_PATH",(folder/"model.gguf").c_str(),1);setenv("LLAMA_CPP_MMPROJ_PATH","projector.gguf",1);setenv("EMBEDDING_FIXTURE_DIR",folder.c_str(),1);
        setenv("WHISPER_CPP_BIN",TEST_LLM_AUDIO_FIXTURE_BIN,1);setenv("WHISPER_MODEL_PATH",(folder/"whisper.bin").c_str(),1);setenv("LLM_AUDIO_FIXTURE_DIR",folder.c_str(),1);setenv("ROUTER_OUTPUTS_DIR",(folder/"outputs").c_str(),1);
        setenv("LLM_AUDIO_FIXTURE_RESULT",R"({"segments":[{"text":" first ","start":0.25,"end":1.5,"speaker":"S"},{"text":" second\n","start":1.5,"end":2.25}],"language":"en"})",1);
        setenv("STT_UNLOAD_LLM_DURING_GENERATION",mode==2?"true":"false",1);setenv("STT_WAIT_FOR_LLM_IDLE_TIMEOUT","0",1);setenv("VIDEO_MAX_FRAMES",mode==3?"1":"4",1);
        if(mode==1)setenv("LLM_AUDIO_FIXTURE_EXIT","7",1);
        std::unique_ptr<RemoteVideo> remote;if(mode>=5)remote=std::make_unique<RemoteVideo>(read(assets/"audio.mkv"),mode==6);
        const auto url=remote?remote->url:"data:video/mp4;base64,"+encode(read(assets/"audio.mkv"));
        const auto body=mode==4?R"({"messages":[{"content":[{"video_url":1}]}]})":"{\"messages\":[{\"content\":[\"before\",{\"video_url\":"+quote(url)+"},\"after\"]}]}";
        {
            ezl_env_config config{};ezl_load_config(&config);config.request_timeout_ms=5000;ezlocalai::NativeServer server(config,std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model));
            const auto response=server.handle_request("POST","/v1/chat/completions","",body);
            expect(response.status==(mode==4?500:200),"managed video status "+response.body);
            const auto seen=read(folder/("seen"+suffix));
            expect(fs::exists(folder/("launch"+suffix))==(mode!=4),"video type error precedes model load");
            if(mode<3)expect(seen.find("[Video: 4.0s duration,")!=std::string::npos&&seen.find("data:image/jpeg;base64,")!=std::string::npos&&seen.find("beforeafter")!=std::string::npos,"video metadata and frames reach LLM");
            if(mode==0)expect(seen.find(R"([0.2s - 1.5s] S: first\n[1.5s - 2.2s] second)")!=std::string::npos,"timestamped STT reaches LLM "+seen);
            if(mode==1||mode==2)expect(seen.find("Video Audio Transcript")==std::string::npos,"STT failure leaves video metadata and frames");
            if(mode==3)expect(seen.find("[Video:")==std::string::npos&&seen.find("beforeafter")!=std::string::npos&&!fs::exists(folder/"audio-launches"),"zero scene budget skips frames and STT");
            if(remote)expect(remote->downloads==1&&remote->user_agent&&((seen.find("[Video:")!=std::string::npos)==(mode==5)),"remote video follows redirect, sends source user agent and omits HTTP errors: mode="+std::to_string(mode)+" downloads="+std::to_string(remote->downloads.load())+" user_agent="+std::to_string(remote->user_agent.load())+" has_video="+std::to_string(seen.find("[Video:")!=std::string::npos)+" url="+remote->url);
            if(mode==0||mode==1||mode==5){std::istringstream launches(read(folder/"audio-launches"));int pid=0;launches>>pid;expect(pid&&read(folder/("audio-llm-state-"+std::to_string(pid)))=="0","video STT starts before LLM load");}
            expect(server.handle_request("POST","/v1/chat/completions","",R"({"messages":[{"content":"recovered"}]})").status==200,"video error releases handoff and inference slots");
        }
        for(const auto &file:{folder/("launch"+suffix),folder/"audio-launches"}) {std::istringstream in(read(file));for(int pid;in>>pid;)expect(kill(pid,0)!=0,"video child reaped");}
        expect(!fs::exists(folder/"outputs")||fs::is_empty(folder/"outputs"),"video transcription input cleaned");
    }
    fs::remove_all(root);std::cout<<"Managed video frame/audio enrichment, nonfatal STT failures, handoff and recovery passed\n";
}
}
int main(int argc,char **argv) {try{
    expect(argc==4,"reference/policy/media arguments");
    const auto temporary_files=[] {std::set<fs::path> files;for(const auto &file:fs::directory_iterator(fs::temp_directory_path()))if(file.is_regular_file()&&file.path().filename().string().rfind("ezlocalai-video-",0)==0)files.insert(file.path());return files;};
    const auto before=temporary_files();
    policies(argv[1],argv[2]);media(argv[1],argv[3]);managed(fs::path(argv[3]).parent_path()/"llm_video_assets");
    expect(temporary_files()==before,"video/audio temporary files removed on success, no-frame results and STT failures");return 0;
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
