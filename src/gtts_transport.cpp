#include "ezlocalai/gtts.hpp"
#include "ezlocalai/process.hpp"
#include "ezlocalai/http_url.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
namespace ezlocalai {
namespace {
namespace fs=std::filesystem;
struct Workspace {
    fs::path path;
    Workspace(){auto pattern=(fs::temp_directory_path()/"ezlocalai-gtts-XXXXXX").string();if(!mkdtemp(pattern.data()))throw std::system_error(errno,std::generic_category(),"gTTS temporary directory");path=pattern;}
    ~Workspace(){std::error_code ignored;fs::remove_all(path,ignored);}
    std::string file(const char *name)const{return (path/name).string();}
};
std::string read(const std::string &path){std::ifstream input(path,std::ios::binary);if(!input)throw std::runtime_error("Cannot read gTTS subprocess output");return std::string(std::istreambuf_iterator<char>(input),{});}
void write(const std::string &path,const std::string &data){std::ofstream output(path,std::ios::binary);output.write(data.data(),data.size());if(!output)throw std::runtime_error("Cannot write gTTS subprocess input");}
[[noreturn]] void connection_error(){throw TtsError("gTTSError","Failed to connect. Probable cause: Unknown");}
std::string signal_description(int number){
    // CPython's signal.Signals representation on the Linux deployment target.
    static const std::map<int,std::string> names={{SIGHUP,"SIGHUP"},{SIGINT,"SIGINT"},{SIGQUIT,"SIGQUIT"},{SIGILL,"SIGILL"},{SIGTRAP,"SIGTRAP"},{SIGABRT,"SIGABRT"},{SIGBUS,"SIGBUS"},{SIGFPE,"SIGFPE"},{SIGKILL,"SIGKILL"},{SIGUSR1,"SIGUSR1"},{SIGSEGV,"SIGSEGV"},{SIGUSR2,"SIGUSR2"},{SIGPIPE,"SIGPIPE"},{SIGALRM,"SIGALRM"},{SIGTERM,"SIGTERM"},{SIGSTKFLT,"SIGSTKFLT"},{SIGCHLD,"SIGCHLD"},{SIGCONT,"SIGCONT"},{SIGSTOP,"SIGSTOP"},{SIGTSTP,"SIGTSTP"},{SIGTTIN,"SIGTTIN"},{SIGTTOU,"SIGTTOU"},{SIGURG,"SIGURG"},{SIGXCPU,"SIGXCPU"},{SIGXFSZ,"SIGXFSZ"},{SIGVTALRM,"SIGVTALRM"},{SIGPROF,"SIGPROF"},{SIGWINCH,"SIGWINCH"},{SIGIO,"SIGIO"},{SIGPWR,"SIGPWR"},{SIGSYS,"SIGSYS"},{SIGRTMIN,"SIGRTMIN"},{SIGRTMAX,"SIGRTMAX"}};
    const auto found=names.find(number);return found==names.end()?"unknown signal "+std::to_string(number):"<Signals."+found->second+": "+std::to_string(number)+">";
}
GttsResponse post(const GttsRequest &request){
    Workspace work;write(work.file("request"),request.body);
    auto url=request.url;auto request_headers=request.headers;bool send_body=true;write(work.file("cookies"),"");
    for(size_t hop=0;;++hop){
        std::vector<std::string> args={"curl","--disable","--silent","--show-error","--http1.1","--insecure","--proto","=http,https","--compressed","--cookie",work.file("cookies"),"--cookie-jar",work.file("cookies"),"--header","Accept:","--header","Accept-Encoding: identity","--header","Expect:","--dump-header",work.file("headers"),"--output",work.file("response")};
        if(send_body){args.push_back("--data-binary");args.push_back("@"+work.file("request"));}
        for(const auto &[key,value]:request_headers){args.push_back("--header");args.push_back(key+": "+value);}
        args.push_back("--url");args.push_back(url);
        FileProcessResult process;try{process=run_process_to_files(args,work.file("stdout"),work.file("stderr"),-1,false,"/dev/null");}catch(const std::system_error &){connection_error();}
        if(process.status)connection_error();GttsResponse response;response.status=0;std::string location;std::istringstream headers(read(work.file("headers")));
        for(std::string line;std::getline(headers,line);){if(!line.empty()&&line.back()=='\r')line.pop_back();if(line.rfind("HTTP/",0)==0){const auto first=line.find(' ');if(first==std::string::npos)continue;std::istringstream status(line.substr(first+1));status>>response.status;std::getline(status,response.reason);if(!response.reason.empty()&&response.reason[0]==' ')response.reason.erase(0,1);location.clear();}
            else{const auto colon=line.find(':');if(colon==std::string::npos)continue;auto key=line.substr(0,colon);for(auto &c:key)if(c>='A'&&c<='Z')c+=32;if(key=="location"){const auto start=line.find_first_not_of(" \t",colon+1);location=start==std::string::npos?"":line.substr(start);}}
        }
        if(!response.status)connection_error();response.body=read(work.file("response"));
        if(location.empty()||(response.status!=301&&response.status!=302&&response.status!=303&&response.status!=307&&response.status!=308))return response;
        if(hop==30)connection_error();
        try{auto next=HttpUrl::parse(url).resolve(location);next.require_http();url=next.str();}catch(...){connection_error();}
        if(response.status!=307&&response.status!=308){send_body=false;request_headers.erase("Content-Length");request_headers.erase("Content-Type");request_headers.erase("Transfer-Encoding");}
    }
}
std::string to_wav(const std::string &mp3){
    Workspace work;write(work.file("input"),mp3);
    const std::vector<std::string> args={"ffmpeg","-hide_banner","-loglevel","error","-i","pipe:0","-ar","24000","-ac","1","-f","wav","pipe:1"};
    FileProcessResult result;try{result=run_process_to_files(args,work.file("wav"),work.file("stderr"),-1,false,work.file("input"),true);}
    catch(const std::system_error &error){if(error.code().value()==ENOENT)throw TtsError("FileNotFoundError","[Errno 2] No such file or directory: 'ffmpeg'");if(error.code().value()==EACCES)throw TtsError("PermissionError","[Errno 13] Permission denied: 'ffmpeg'");throw;}
    if(result.status)throw TtsError("CalledProcessError","Command '"+tts_python_string(nlohmann::json(args).dump())+"' "+(result.status<0?"died with "+signal_description(-result.status):"returned non-zero exit status "+std::to_string(result.status))+".");
    return read(work.file("wav"));
}
}
GttsOperations gtts_native_operations(GttsOperations operations){if(!operations.post)operations.post=post;if(!operations.to_wav)operations.to_wav=to_wav;return operations;}
}
