#include "ezlocalai/gtts.hpp"
#include "whisper_transcription_test.hpp"
#include <arpa/inet.h>
#include <atomic>
#include <poll.h>
#include <sys/socket.h>
#include <thread>
#include <csignal>
using namespace transcription_test;
namespace {
std::string hex(const std::string &s){std::ostringstream out;for(unsigned char c:s)out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(c);return out.str();}
std::string lower(std::string s){for(auto &c:s)if(c>='A'&&c<='Z')c+=32;return s;}
struct Server {
    int fd=-1;unsigned port=0;std::atomic<bool> stop{false};std::thread worker;Json plans,events=Json::array();std::exception_ptr failure;
    explicit Server(Json plans):plans(std::move(plans)){
        fd=socket(AF_INET,SOCK_STREAM|SOCK_CLOEXEC,0);expect(fd>=0,"loopback socket");sockaddr_in address{};address.sin_family=AF_INET;address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
        if(bind(fd,reinterpret_cast<sockaddr *>(&address),sizeof(address))||listen(fd,8)){close(fd);throw std::runtime_error("loopback bind/listen");}socklen_t length=sizeof(address);getsockname(fd,reinterpret_cast<sockaddr *>(&address),&length);port=ntohs(address.sin_port);
        worker=std::thread([this]{try{run();}catch(...){failure=std::current_exception();}});
    }
    ~Server(){finish();close(fd);}
    void finish(){stop=true;if(worker.joinable())worker.join();}
    void run(){while(!stop){pollfd ready{fd,POLLIN,0};const auto result=poll(&ready,1,50);if(result<0&&errno==EINTR)continue;expect(result>=0,"loopback accept poll");if(!result)continue;const int client=accept4(fd,nullptr,nullptr,SOCK_CLOEXEC);expect(client>=0,"loopback accept");struct Close{int fd;~Close(){close(fd);}}closer{client};timeval timeout{3,0};setsockopt(client,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));setsockopt(client,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));std::string wire;char buffer[8192];size_t end;
        while((end=wire.find("\r\n\r\n"))==std::string::npos){const auto n=recv(client,buffer,sizeof(buffer),0);expect(n>0,"complete request headers");wire.append(buffer,n);}
        std::istringstream lines(wire.substr(0,end));std::string method,path,version;lines>>method>>path>>version;std::string line;std::getline(lines,line);Json headers=Json::object();while(std::getline(lines,line)){if(!line.empty()&&line.back()=='\r')line.pop_back();const auto colon=line.find(':');expect(colon!=std::string::npos,"header colon");auto value=line.substr(colon+1);while(!value.empty()&&(value[0]==' '||value[0]=='\t'))value.erase(0,1);headers[lower(line.substr(0,colon))]=value;}
        const auto size=headers.contains("content-length")?std::stoul(headers.at("content-length").get<std::string>()):0;while(wire.size()-end-4<size){const auto n=recv(client,buffer,sizeof(buffer),0);expect(n>0,"complete request body");wire.append(buffer,n);}headers["host"]="<host>";events.push_back({{"method",method},{"path",path},{"headers",headers},{"body",wire.substr(end+4,size)}});
        expect(events.size()<=plans.size(),"no extra network requests");const auto &plan=plans[events.size()-1];const int status=plan.at("status");const auto body=unhex(plan.at("hex"));const std::map<int,std::string> reasons={{200,"OK"},{301,"Moved Permanently"},{302,"Found"},{303,"See Other"},{307,"Temporary Redirect"},{308,"Permanent Redirect"},{403,"Forbidden"}};std::string response="HTTP/1.1 "+std::to_string(status)+" "+reasons.at(status)+"\r\nContent-Length: "+std::to_string(body.size())+"\r\nConnection: close\r\n";for(const auto &[key,value]:plan.at("headers").items())response+=key+": "+value.get<std::string>()+"\r\n";response+="\r\n"+body;
        for(size_t sent=0;sent<response.size();){const auto n=send(client,response.data()+sent,response.size()-sent,MSG_NOSIGNAL);expect(n>0,"send loopback response");sent+=n;}
    }}
};
}
int main(int argc,char **argv){try{
    if(const auto *action=getenv("EZLOCALAI_GTTS_TEST_CHILD")){const int value=std::stoi(action);if(value<0){raise(-value);return 99;}return value;}
    expect(argc==3,"HTTP and process fixture paths");const auto cases=records(argv[1]);size_t requests=0;
    for(const auto &row:cases){Server server(row.at("steps"));auto native=gtts_native_operations();GttsOperations ops;ops.post=[&](auto request){request.url="http://127.0.0.1:"+std::to_string(server.port)+"/_/TranslateWebserverUi/data/batchexecute";return native.post(request);};ops.to_wav=[](const auto &bytes){return bytes;};Json actual;
        try{actual["result"]=hex(gtts_generate_wav(row.at("text"),"\"en\"",ops));}catch(const TtsError &e){actual["error"]={{"category",e.category},{"message",e.what()}};}
        server.finish();if(server.failure)std::rethrow_exception(server.failure);const auto name=row.at("name").get<std::string>();compare_json(server.events,row.at("events"),name+" actual HTTP requests");requests+=server.events.size();compare_json(actual,row.contains("error")?Json{{"error",row.at("error")}}:Json{{"result",row.at("result")}},name+" result");
    }
    // No listener: connection failure must be mapped before conversion.
    auto ops=gtts_native_operations();auto request=gtts_prepare("test")[0];unsigned unused=0;{Server server(Json::array());unused=server.port;}request.url="http://127.0.0.1:"+std::to_string(unused)+"/";try{ops.post(request);expect(false,"connection error required");}catch(const TtsError &e){expect(e.category=="gTTSError"&&std::string(e.what())=="Failed to connect. Probable cause: Unknown","connection error mapping");}
    const auto saved=std::string(getenv("PATH")?getenv("PATH"):"");setenv("PATH","/nonexistent-ezlocalai-programs",1);Json missing;try{ops.to_wav("invalid");}catch(const TtsError &e){missing={{"category",e.category},{"message",e.what()}};}setenv("PATH",saved.c_str(),1);compare_json(missing,Json{{"category","FileNotFoundError"},{"message","[Errno 2] No such file or directory: 'ffmpeg'"}},"missing native converter");
    std::string pattern=(fs::temp_directory_path()/"ezlocalai-gtts-child-XXXXXX").string();expect(mkdtemp(pattern.data()),"child fixture directory");struct Remove{fs::path path;~Remove(){std::error_code error;fs::remove_all(path,error);}}remove{pattern};fs::create_symlink(fs::canonical(argv[0]),fs::path(pattern)/"ffmpeg");
    const auto previous_pipe=std::signal(SIGPIPE,SIG_IGN);
    for(const auto &row:records(argv[2])){setenv("PATH",pattern.c_str(),1);setenv("EZLOCALAI_GTTS_TEST_CHILD",row.at("status").dump().c_str(),1);Json actual;try{ops.to_wav("invalid");}catch(const TtsError &e){actual={{"category",e.category},{"message",e.what()}};}unsetenv("EZLOCALAI_GTTS_TEST_CHILD");setenv("PATH",saved.c_str(),1);compare_json(actual,row.at("error"),"native process exit or signal");}
    std::signal(SIGPIPE,previous_pipe);
    std::cout<<cases.size()<<" source HTTP scenarios and "<<requests<<" wire requests match; connection and converter errors pass\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
