// Non-strict POSIX realpath follows CPython 3.12; see licenses/CPython.txt.
#include "ezlocalai/ctts_paths.hpp"
#include "ezlocalai/tts_text.hpp"
#include "ezlocalai/llama_tts.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/llm_startup.hpp"
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iomanip>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
namespace ezlocalai {
namespace {
std::string join(const std::string &a,const std::string &b) { if(a.empty() || (!b.empty() && b.front()=='/')) return b; return a+(a.back()=='/'?"":"/")+b; }
[[noreturn]] void os_error(const std::string &path) { const auto code=errno; throw TtsError(code==ENOENT?"FileNotFoundError":code==EACCES?"PermissionError":"OSError","[Errno "+std::to_string(code)+"] "+std::strerror(code)+": "+python_string_repr(path)); }
std::pair<std::string,bool> resolve(std::string path,std::string rest,std::map<std::string,std::optional<std::string>> &seen) {
    if(!rest.empty() && rest.front()=='/') { path="/"; rest.erase(0,1); }
    while(!rest.empty()) { const auto slash=rest.find('/'); const auto name=rest.substr(0,slash); rest=slash==std::string::npos?"":rest.substr(slash+1); if(name.empty() || name==".") continue;
        if(name=="..") { if(path!="/") { const auto split=path.rfind('/'); path=split==0?"/":path.substr(0,split); } continue; }
        const auto next=join(path,name); struct stat info{}; if(lstat(next.c_str(),&info)<0 || !S_ISLNK(info.st_mode)) { path=next; continue; }
        const auto found=seen.find(next); if(found!=seen.end()) { if(found->second) { path=*found->second; continue; } return {join(next,rest),false}; }
        seen[next]=std::nullopt; std::string target(256,'\0'); for(;;) { const auto n=readlink(next.c_str(),target.data(),target.size()); if(n<0) os_error(next); if(size_t(n)<target.size()) { target.resize(n); break; } target.resize(target.size()*2); }
        auto resolved=resolve(path,target,seen); path=std::move(resolved.first); if(!resolved.second) return {join(path,rest),false}; seen[next]=path;
    } return {path,true};
}
std::string realpath(const std::string &value) {
    if(value.find('\0')!=std::string::npos) throw TtsError("ValueError","embedded null byte"); std::map<std::string,std::optional<std::string>> seen;
    const auto absolute=!value.empty() && value.front()=='/'?value:join(std::filesystem::current_path().string(),value); auto result=std::filesystem::path(resolve("",absolute,seen).first).lexically_normal().string(); if(result.size()>1 && result.back()=='/') result.pop_back(); return result;
}
bool exists(const std::string &path) { struct stat info{}; return path.find('\0')==std::string::npos && stat(path.c_str(),&info)==0; }
[[noreturn]] void utf8_error(const std::string &input,size_t start,size_t end,const char *reason) {
    std::ostringstream message; message<<"'utf-8' codec can't decode "; if(end==start+1) message<<"byte 0x"<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(static_cast<unsigned char>(input[start]))<<std::dec<<" in position "<<start; else message<<"bytes in position "<<start<<'-'<<end-1; message<<": "<<reason; throw TtsError("UnicodeDecodeError",message.str());
}
void validate_utf8(const std::string &input) {
    for(size_t i=0;i<input.size();) { const auto c=static_cast<unsigned char>(input[i]); if(c<128) { ++i; continue; } const int n=c>=0xc2 && c<=0xdf?2:c>=0xe0 && c<=0xef?3:c>=0xf0 && c<=0xf4?4:0; if(!n) utf8_error(input,i,i+1,"invalid start byte"); for(int j=1;j<n;++j) { if(i+j==input.size()) utf8_error(input,i,i+j,"unexpected end of data"); const auto d=static_cast<unsigned char>(input[i+j]); if(d<0x80 || d>0xbf || (j==1 && ((c==0xe0 && d<0xa0)||(c==0xed && d>=0xa0)||(c==0xf0 && d<0x90)||(c==0xf4 && d>0x8f)))) utf8_error(input,i,i+j,"invalid continuation byte"); } i+=n; }
}
}
std::string ctts_real_path(const std::string &path) { return realpath(path); }
std::string ctts_contained_file_path(const std::string &base,const std::optional<std::string> &stem,const std::string &extension) {
    if(extension!=".wav" && extension!=".txt") throw TtsError("ValueError","Invalid file extension"); const auto parent=realpath(base),path=realpath(join(parent,tts_safe_file_stem(stem)+extension)); if(path!=parent && path.rfind(parent+"/",0)==0) return path; throw TtsError("ValueError","Invalid file path");
}
std::optional<std::string> ctts_voice_audio_path(const std::string &voices,const std::optional<std::string> &voice) {
    const auto path=ctts_contained_file_path(voices,tts_safe_file_stem(voice),".wav"); if(exists(path)) return path; const auto fallback=ctts_contained_file_path(voices,"default",".wav"); if(exists(fallback)) return fallback; return {};
}
std::optional<std::string> ctts_voice_ref_text(const std::string &voices,const std::optional<std::string> &audio_path) {
    if(!audio_path || audio_path->empty()) return {}; const auto slash=audio_path->rfind('/'); const auto stem=tts_safe_file_stem(audio_path->substr(slash==std::string::npos?0:slash+1)); const auto sidecar=ctts_contained_file_path(voices,stem,".txt"); if(!exists(sidecar)) return {}; const int fd=open(sidecar.c_str(),O_RDONLY|O_CLOEXEC); if(fd<0) return {}; struct Close { int fd; ~Close() { close(fd); } } closer{fd}; std::string text; char buffer[8192]; for(;;) { const auto n=read(fd,buffer,sizeof(buffer)); if(n<0) { if(errno==EINTR) continue; return {}; } if(!n) break; text.append(buffer,n); }
    validate_utf8(text); std::string normalized; normalized.reserve(text.size()); for(size_t i=0;i<text.size();++i) { if(text[i]=='\r') { normalized+='\n'; if(i+1<text.size() && text[i+1]=='\n') ++i; } else normalized+=text[i]; } normalized=strip_config_value(normalized); return normalized.empty()?std::nullopt:std::optional<std::string>(normalized);
}
}
