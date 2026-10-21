#include "ezlocalai/llama_tts.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/hub_snapshot.hpp"
#include "ezlocalai/llm_parameters.hpp"
#include "ezlocalai/llm_slots.hpp"
#include "ezlocalai/llm_startup.hpp"
#include <nlohmann/json.hpp>
#include <cstdio>
#include <unicode/utf8.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <dlfcn.h>
#include <filesystem>
#include <fcntl.h>
#include <mutex>
#include <pwd.h>
#include <signal.h>
#include <spawn.h>
#include <sstream>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#ifndef EZLOCALAI_SNDFILE_LIBRARY
#define EZLOCALAI_SNDFILE_LIBRARY "libsndfile.so.1"
#endif
extern char **environ;
namespace ezlocalai {
namespace {
using Json = nlohmann::ordered_json;
const std::string repo_default = "Mouserat/qwen3-tts-0.6b-base-gguf",revision_default = "30954b7a9d99fad81064bf1dc35f881b029f9032";
std::string type(const Json &v) { return v.is_null() ? "NoneType" : v.is_boolean() ? "bool" : v.is_string() ? "str" : v.is_array() ? "list" : v.is_object() ? "dict" : v.is_number_float() ? "float" : "int"; }
bool truth(const Json &v) { return !v.is_null() && !(v.is_boolean() && !v.get<bool>()) && !(v.is_number() && v == 0) && !((v.is_string() || v.is_array() || v.is_object()) && v.empty()) && !(v.is_string() && v.get<std::string>().empty()); }
std::string repr(const Json &v,bool nested=false) {
    if (v.is_string()) return nested ? python_string_repr(v.get<std::string>()) : v.get<std::string>();
    if (v.is_null()) return "None"; if (v.is_boolean()) return v.get<bool>() ? "True" : "False";
    if (v.is_array() || v.is_object()) { std::string out = v.is_array() ? "[" : "{"; bool first = true; for (const auto &item : v.items()) { if (!first) out += ", "; first = false; if (v.is_object()) out += python_string_repr(item.key()) + ": "; out += repr(item.value(),true); } return out + (v.is_array() ? "]" : "}"); }
    return v.dump();
}
std::string integer(const Json &v) {
    if (v.is_boolean()) return v.get<bool>() ? "1" : "0";
    if (v.is_number_integer() || v.is_number_unsigned()) return v.dump();
    if (v.is_number_float()) { const auto d = v.get<double>(); if (!std::isfinite(d)) throw TtsError("ValueError","cannot convert float NaN to integer"); char decimal[512]; std::snprintf(decimal,sizeof(decimal),"%.0f",std::trunc(d)); return d == 0 ? "0" : std::string(decimal); }
    if (v.is_string()) { try { return llm_parameter_integer(v.get<std::string>()); } catch (const LlmParameterError &e) { throw TtsError(e.category,e.what()); } }
    throw TtsError("TypeError","int() argument must be a string, a bytes-like object or a real number, not '" + type(v) + "'");
}
std::string get(const std::map<std::string,std::string> &env,const std::string &key,const std::string &fallback) { const auto found = env.find(key); return found == env.end() ? fallback : found->second; }
std::string absolute(const std::string &path) { return std::filesystem::weakly_canonical(std::filesystem::absolute(path)).string(); }
std::string path(const Json &v) { if (!v.is_string()) throw TtsError("TypeError","argument should be a str or an os.PathLike object where __fspath__ returns a str, not '"+type(v)+"'"); return absolute(v.get<std::string>()); }
struct Temporary { std::string directory; Temporary() { std::string pattern = (std::filesystem::temp_directory_path()/"ezlocalai-tts-XXXXXX").string(); if (!mkdtemp(pattern.data())) throw std::runtime_error("Cannot create TTS temporary directory"); directory = pattern; } ~Temporary() { std::error_code ignored; std::filesystem::remove_all(directory,ignored); } };
TtsAudio read_wav(const std::string &name) {
    struct Info { int64_t frames; int samplerate,channels,format,sections,seekable; };
    struct Sndfile {
        void *library = dlopen(EZLOCALAI_SNDFILE_LIBRARY,RTLD_NOW|RTLD_LOCAL);
        using Open = void *(*)(const char *,int,Info *); using Read = int64_t (*)(void *,float *,int64_t); using Close = int (*)(void *); using Error = const char *(*)(void *);
        using ErrorCode = int (*)(void *); using ErrorNumber = const char *(*)(int);
        Open open; Read read; Close close; Error error; ErrorCode error_code; ErrorNumber error_number;
        Sndfile() { if (!library) throw std::runtime_error("Cannot load libsndfile.so.1"); open = reinterpret_cast<Open>(dlsym(library,"sf_open")); read = reinterpret_cast<Read>(dlsym(library,"sf_readf_float")); close = reinterpret_cast<Close>(dlsym(library,"sf_close")); error = reinterpret_cast<Error>(dlsym(library,"sf_strerror")); error_code = reinterpret_cast<ErrorCode>(dlsym(library,"sf_error")); error_number = reinterpret_cast<ErrorNumber>(dlsym(library,"sf_error_number")); if (!open || !read || !close || !error || !error_code || !error_number) throw std::runtime_error("Incomplete libsndfile ABI"); }
    };
    static Sndfile sf; Info info{}; auto *file = sf.open(name.c_str(),0x10,&info); if (!file) throw TtsError("LibsndfileError","Error opening '"+name+"': "+sf.error_number(sf.error_code(nullptr)));
    struct Close { Sndfile &sf; void *file; ~Close() { sf.close(file); } } close{sf,file};
    TtsAudio out; out.rate = std::to_string(info.samplerate); out.channels = info.channels;
    if (info.frames < 0 || info.channels <= 0 || static_cast<uint64_t>(info.frames) > SIZE_MAX/sizeof(float)/info.channels) throw std::runtime_error("Invalid WAV dimensions");
    out.samples.resize(info.frames*info.channels); const auto frames = sf.read(file,out.samples.data(),info.frames); if (frames < 0) throw TtsError("LibsndfileError",sf.error(file)); out.samples.resize(frames*info.channels); return out;
}
std::vector<float> decode_pcm(const Json &value) {
    if (!value.is_string()) throw TtsError("TypeError","argument should be a bytes-like object or ASCII string, not '"+type(value)+"'");
    const auto input = value.get<std::string>(); std::string bytes; int bits = 0, count = 0, quad = 0, pads = 0; bool padded = false; size_t digits = 0;
    for (unsigned char c : input) if (c >= 128) throw TtsError("ValueError","string argument should contain only ASCII characters");
    if (!input.empty() && input[0] == '=') throw TtsError("Error","Leading padding not allowed");
    for (size_t i = 0; i < input.size(); ++i) {
        const unsigned char c = input[i];
        if (c == '=') { padded = true; if (quad >= 2 && quad + (++pads) >= 4) { if (i+1 < input.size()) throw TtsError("Error","Excess data after padding"); quad = 0; break; } continue; }
        const int digit = c >= 'A' && c <= 'Z' ? c-'A' : c >= 'a' && c <= 'z' ? c-'a'+26 : c >= '0' && c <= '9' ? c-'0'+52 : c == '+' ? 62 : c == '/' ? 63 : -1;
        if (digit < 0) throw TtsError("Error","Only base64 data is allowed"); if (padded) throw TtsError("Error","Discontinuous padding not allowed");
        ++digits; quad = (quad+1)%4; pads = 0; bits = ((bits << 6) | digit) & 0xffffff; count += 6; if (count >= 8) { count -= 8; bytes += static_cast<char>((bits >> count)&255); }
    }
    if (quad == 1) throw TtsError("Error","Invalid base64-encoded string: number of data characters ("+std::to_string(digits)+") cannot be 1 more than a multiple of 4");
    if (quad) throw TtsError("Error","Incorrect padding");
    if (bytes.size()%4) throw TtsError("ValueError","buffer size must be a multiple of element size");
    std::vector<float> out(bytes.size()/4); for (size_t i=0;i<out.size();++i) { const auto *b=reinterpret_cast<const unsigned char *>(bytes.data()+4*i); uint32_t word=uint32_t(b[0])|(uint32_t(b[1])<<8)|(uint32_t(b[2])<<16)|(uint32_t(b[3])<<24); std::memcpy(&out[i],&word,4); } return out;
}
}
std::string tts_python_integer(const std::string &json) {
    const auto raw=strip_config_value(json); size_t at=!raw.empty() && raw[0]=='-'?1:0; const auto start=at;
    while(at<raw.size() && raw[at]>='0' && raw[at]<='9') ++at;
    if(at==raw.size() && at>start && (raw[start]!='0' || at-start==1)) { const auto value=parse_configuration_integer(raw); if(value) return *value; }
    return integer(Json::parse(json));
}
std::string tts_python_string(const std::string &json) { return repr(Json::parse(json)); }
LlamaTtsPlatform native_llama_tts_platform() {
    LlamaTtsPlatform p; for (auto **entry=environ;entry && *entry;++entry) { const std::string value=*entry; const auto split=value.find('='); if(split!=std::string::npos) p.environment[value.substr(0,split)]=value.substr(split+1); }
    p.is_file=[](const auto &name) { std::error_code ec; return std::filesystem::is_regular_file(name,ec); }; p.absolute=absolute;
    p.home=[] { if (const auto *home=std::getenv("HOME")) return std::string(home); if (const auto *pw=getpwuid(getuid())) return std::string(pw->pw_dir); throw std::runtime_error("Could not determine home directory."); };
    p.executable=[](const auto &name) { if(name.empty()) return false; const auto executable=[](const auto &name) { std::error_code ec; return access(name.c_str(),X_OK)==0 && !std::filesystem::is_directory(name,ec); }; if(name.find('/')!=std::string::npos) return executable(name); std::istringstream paths(std::getenv("PATH") ? std::getenv("PATH") : "/bin:/usr/bin"); for(std::string dir;std::getline(paths,dir,':');) if(executable((std::filesystem::path(dir)/name).string())) return true; return false; };
    p.download=[](const auto &repo,const auto &file,const auto &revision) { HubSnapshotOptions options; options.revision=revision; return hub_snapshot_file(repo,file,options); }; return p;
}
std::string resolve_tts_model_id(const std::string &configured) {
    auto value=strip_config_value(configured);
    return value.empty() || value=="Qwen/Qwen3-TTS-12Hz-0.6B-Base" ? repo_default : value;
}
LlamaTtsLaunch llama_tts_launch(const std::string &encoded_device,const std::string &encoded_kwargs,LlamaTtsPlatform p) {
    LlamaTtsLaunch out; out.environment=p.environment; const auto timeout=get(p.environment,"QWEN_TTS_TIMEOUT","300");
    if (!parse_configuration_float(timeout,&out.timeout)) throw TtsError("ValueError","could not convert string to float: "+python_string_repr(timeout));
    if (!std::isfinite(out.timeout) || out.timeout<=0) throw TtsError("ValueError","QWEN_TTS_TIMEOUT must be positive");
    const auto device=Json::parse(encoded_device); auto kwargs=Json::parse(encoded_kwargs); if(!truth(kwargs)) kwargs=Json::object();
    const auto repo=resolve_tts_model_id(get(p.environment,"QWEN_TTS_MODEL",repo_default));
    auto revision=get(p.environment,"QWEN_TTS_REVISION",""); if(revision.empty()) revision=repo==repo_default ? revision_default : "main";
    std::vector<std::string> models; for(const auto &item:std::vector<std::pair<std::string,std::string>>{{"QWEN_TTS_MODEL_FILE","qwen-talker-0.6b-base-Q8_0.gguf"},{"QWEN_TTS_MMPROJ_FILE","qwen-tokenizer-12hz-f16.gguf"}}) { auto file=strip_config_value(get(p.environment,item.first,item.second)); if(file.empty()) file=item.second; models.push_back(p.is_file(file) ? p.absolute(file) : p.download(repo,file,revision)); }
    auto binary=get(p.environment,"QWEN_TTS_BIN","/opt/ezlocalai-tts/build/bin/ezlocalai-tts"); if(!p.environment.count("QWEN_TTS_BIN") && !p.is_file(binary)) binary=(std::filesystem::path(p.home())/".ezlocalai/tts-build/bin/ezlocalai-tts").string();
    if(!p.executable(binary)) throw TtsError("RuntimeError","Native TTS worker not found: "+binary+". Rebuild the image or run python scripts/build_tts.py and set QWEN_TTS_BIN to its output.");
    std::vector<Json> keys,unsupported;
    if(kwargs.is_object()) { for(const auto &item:kwargs.items()) keys.push_back(item.key()); }
    else if(kwargs.is_array()) { for(const auto &item:kwargs) { if(item.is_array() || item.is_object()) throw TtsError("TypeError","unhashable type: '"+type(item)+"'"); if(std::find(keys.begin(),keys.end(),item)==keys.end()) keys.push_back(item); } }
    else if(kwargs.is_string()) { const auto value=kwargs.get<std::string>(); for(int32_t i=0;i<static_cast<int32_t>(value.size());) { const auto start=i; UChar32 c; U8_NEXT(value.data(),i,static_cast<int32_t>(value.size()),c); Json item=value.substr(start,i-start); if(std::find(keys.begin(),keys.end(),item)==keys.end()) keys.push_back(item); } }
    else throw TtsError("TypeError","'"+type(kwargs)+"' object is not iterable");
    for(const auto &key:keys) if(!key.is_string() || (key!="max_new_tokens" && key!="temperature" && key!="top_k" && key!="top_p" && key!="repetition_penalty" && key!="seed")) unsupported.push_back(key);
    if(!unsupported.empty()) { std::sort(unsupported.begin(),unsupported.end(),[](const auto &a,const auto &b) { if(a.is_string()!=b.is_string()) throw TtsError("TypeError","'<' not supported between instances of '"+type(a)+"' and '"+type(b)+"'"); return a<b; }); throw TtsError("ValueError","Unsupported llama.cpp TTS generation settings: "+repr(Json(unsupported))); }
    if(!kwargs.is_object()) throw TtsError("AttributeError","'"+type(kwargs)+"' object has no attribute 'get'");
    out.arguments={binary,"-m",models[0],"-mm",models[1],"-c",get(p.environment,"QWEN_TTS_CONTEXT_SIZE","4096"),"-b","512","-ub","512","-t",get(p.environment,"QWEN_TTS_THREADS","20"),"--temp",repr(kwargs.value("temperature",Json(0.9))),"--top-k",repr(kwargs.value("top_k",Json(50))),"--top-p",repr(kwargs.value("top_p",Json(1.0))),"--repeat-penalty",repr(kwargs.value("repetition_penalty",Json(1.05))),"--flash-attn","on"};
    if(kwargs.contains("seed")) { out.arguments.push_back("--seed"); out.arguments.push_back(integer(kwargs["seed"])); }
    if(!device.is_string()) throw TtsError("AttributeError","'"+type(device)+"' object has no attribute 'startswith'"); out.device=device.get<std::string>();
    if(out.device.rfind("cuda",0)==0) {
        const auto graphs=llm_unicode_lower(get(p.environment,"QWEN_TTS_DISABLE_CUDA_GRAPHS","false")); if(graphs=="1" || graphs=="true" || graphs=="yes" || graphs=="on") out.environment["GGML_CUDA_DISABLE_GRAPHS"]="1";
        const auto colon=out.device.find(':'); const auto index=integer(colon==std::string::npos ? Json(0) : Json(out.device.substr(colon+1))); std::vector<std::string> visible; const auto text=get(p.environment,"CUDA_VISIBLE_DEVICES",""); size_t start=0;
        for(;;) { const auto end=text.find(',',start); visible.push_back(text.substr(start,end==std::string::npos ? end : end-start)); if(end==std::string::npos) break; start=end+1; }
        std::string selected=index; if(visible!=std::vector<std::string>{""} && configuration_integer_less(index,std::to_string(visible.size()))) { int64_t n; try { n=std::stoll(index); } catch (...) { throw TtsError("IndexError","cannot fit 'int' into an index-sized integer"); } if(n<0) n+=visible.size(); if(n<0) throw TtsError("IndexError","list index out of range"); selected=visible.at(n); }
        out.environment["CUDA_VISIBLE_DEVICES"]=selected; out.environment["HIP_VISIBLE_DEVICES"]=selected; out.arguments.insert(out.arguments.end(),{"-ngl","999"});
    } else out.arguments.insert(out.arguments.end(),{"-ngl","0","--no-mmproj-offload","--device","none"});
    return out;
}
std::string llama_tts_language(const std::string &encoded_language,const std::string &encoded_text) {
    const auto language=Json::parse(encoded_language); if(!language.is_string()) throw TtsError("AttributeError","'"+type(language)+"' object has no attribute 'lower'"); auto value=llm_unicode_lower(language.get<std::string>()); if(value!="auto") return value;
    const auto text=Json::parse(encoded_text); if(!text.is_string()) throw TtsError("TypeError","expected string or bytes-like object, got '"+type(text)+"'"); const auto input=text.get<std::string>();
    for(const auto &range:std::vector<std::tuple<int,int,std::string>>{{0x3040,0x30ff,"japanese"},{0xac00,0xd7af,"korean"},{0x4e00,0x9fff,"chinese"},{0x0400,0x04ff,"russian"}}) for(int32_t i=0;i<static_cast<int32_t>(input.size());) { UChar32 c; U8_NEXT(input.data(),i,static_cast<int32_t>(input.size()),c); if(c>=std::get<0>(range) && c<=std::get<1>(range)) return std::get<2>(range); }
    return "english";
}
struct LlamaTts::Impl {
    LlamaTtsLaunch launch; std::recursive_mutex inference_mutex; mutable std::mutex close_mutex; std::mutex queue_mutex; std::condition_variable available; std::deque<Json> responses;
    pid_t pid=-1; int input=-1,output=-1; std::thread reader;
    explicit Impl(LlamaTtsLaunch value):launch(std::move(value)) {}
    ~Impl() { stop(); }
    void enqueue(Json response) { { std::lock_guard<std::mutex> lock(queue_mutex); responses.push_back(std::move(response)); } available.notify_all(); }
    void read() {
        std::string pending; char buffer[8192];
        const auto consume=[&](const std::string &line) { try { auto value=Json::parse(line); if(value.is_object()) enqueue(std::move(value)); } catch(const Json::exception &) {} };
        for(;;) { const auto n=::read(output,buffer,sizeof(buffer)); if(n>0) { pending.append(buffer,n); size_t end; while((end=pending.find('\n'))!=std::string::npos) { consume(pending.substr(0,end)); pending.erase(0,end+1); } } else if(n<0 && errno==EINTR) continue; else break; }
        if(!pending.empty()) consume(pending); enqueue(Json{{"error","Native TTS worker exited"}});
    }
    bool alive() { std::lock_guard<std::mutex> lock(close_mutex); if(pid<=0) return false; int status; const auto result=waitpid(pid,&status,WNOHANG); if(result==pid || (result<0 && errno==ECHILD)) { pid=-1; return false; } return true; }
    void stop() {
        std::lock_guard<std::mutex> lock(close_mutex);
        if(pid>0) { int status; auto result=waitpid(pid,&status,WNOHANG); if(!result) { kill(pid,SIGTERM); const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10); do { result=waitpid(pid,&status,WNOHANG); if(result==pid || (result<0 && errno!=EINTR)) break; std::this_thread::sleep_for(std::chrono::milliseconds(10)); } while(std::chrono::steady_clock::now()<deadline); if(result==0 || (result<0 && errno==EINTR)) { kill(pid,SIGKILL); while(waitpid(pid,&status,0)<0 && errno==EINTR) {} } } pid=-1; }
        if(input>=0) { ::close(input); input=-1; }
        // The child owns the only writer; terminating/reaping it releases read.
        if(reader.joinable()) reader.join(); if(output>=0) { ::close(output); output=-1; }
    }
    void start() {
        if(launch.arguments.empty()) throw std::invalid_argument("TTS launch requires an executable");
        int in[2],out[2]; if(pipe2(in,O_CLOEXEC)) throw std::system_error(errno,std::generic_category(),"TTS pipe"); if(pipe2(out,O_CLOEXEC)) { ::close(in[0]); ::close(in[1]); throw std::system_error(errno,std::generic_category(),"TTS pipe"); }
        posix_spawn_file_actions_t actions; int error=posix_spawn_file_actions_init(&actions); const bool actions_ready=!error;
        if(!error) error=posix_spawn_file_actions_adddup2(&actions,in[0],STDIN_FILENO); if(!error) error=posix_spawn_file_actions_adddup2(&actions,out[1],STDOUT_FILENO);
#if defined(__GLIBC__) && __GLIBC_PREREQ(2,34)
        if(!error) error=posix_spawn_file_actions_addclosefrom_np(&actions,3);
#else
        for(int fd:{in[0],in[1],out[0],out[1]}) if(!error) error=posix_spawn_file_actions_addclose(&actions,fd);
#endif
        std::vector<char *> argv,envp; std::vector<std::string> env;
        for(auto &arg:launch.arguments) argv.push_back(arg.data()); argv.push_back(nullptr); for(const auto &item:launch.environment) env.push_back(item.first+'='+item.second); for(auto &item:env) envp.push_back(item.data()); envp.push_back(nullptr);
        // subprocess.Popen restores these signals in the child, even when the
        // HTTP host ignores SIGPIPE. The parent's disposition stays untouched.
        posix_spawnattr_t attributes; const int attr_error=posix_spawnattr_init(&attributes); const bool attr_ready=!attr_error; if(!error) error=attr_error;
        sigset_t defaults; sigemptyset(&defaults); sigaddset(&defaults,SIGPIPE);
#ifdef SIGXFZ
        sigaddset(&defaults,SIGXFZ);
#endif
#ifdef SIGXFSZ
        sigaddset(&defaults,SIGXFSZ);
#endif
        if(!error) error=posix_spawnattr_setsigdefault(&attributes,&defaults); if(!error) error=posix_spawnattr_setflags(&attributes,POSIX_SPAWN_SETSIGDEF);
        if(!error) error=posix_spawnp(&pid,argv.at(0),&actions,&attributes,argv.data(),envp.data()); if(attr_ready) posix_spawnattr_destroy(&attributes); if(actions_ready) posix_spawn_file_actions_destroy(&actions); ::close(in[0]); ::close(out[1]);
        if(error) { ::close(in[1]); ::close(out[0]); pid=-1; throw std::system_error(error,std::generic_category(),"TTS spawn"); }
        input=in[1]; output=out[0]; try { reader=std::thread([this] { read(); }); } catch(...) { kill(pid,SIGKILL); while(waitpid(pid,nullptr,0)<0 && errno==EINTR) {} pid=-1; ::close(input); ::close(output); input=output=-1; throw; }
    }
    void send(const Json &value) {
        int fd; { std::lock_guard<std::mutex> lock(close_mutex); fd=input<0 ? -1 : fcntl(input,F_DUPFD_CLOEXEC,3); } if(fd<0) throw TtsError("RuntimeError","Native TTS worker is closed");
        struct Descriptor { int fd; ~Descriptor() { ::close(fd); } } descriptor{fd};
        // Match Python's BrokenPipeError without changing process-wide signal
        // disposition (other inference workers may be active concurrently).
        sigset_t block,previous,pending; sigemptyset(&block); sigaddset(&block,SIGPIPE); pthread_sigmask(SIG_BLOCK,&block,&previous); sigpending(&pending); const bool was_pending=sigismember(&pending,SIGPIPE);
        struct Signals { sigset_t previous,block; bool pending; ~Signals() { if(!pending) { timespec zero{}; while(sigtimedwait(&block,nullptr,&zero)<0 && errno==EINTR) {} } pthread_sigmask(SIG_SETMASK,&previous,nullptr); } } signals{previous,block,was_pending};
        const auto bytes=value.dump(-1,' ',true)+'\n'; size_t offset=0; while(offset<bytes.size()) { const auto n=::write(fd,bytes.data()+offset,bytes.size()-offset); if(n>0) offset+=n; else if(n<0 && errno==EINTR) continue; else throw TtsError("BrokenPipeError","[Errno 32] Broken pipe"); }
    }
    Json receive() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if(!available.wait_for(lock,std::chrono::duration<double>(launch.timeout),[&] { return !responses.empty(); })) { lock.unlock(); stop(); throw TtsError("RuntimeError","Native TTS worker on "+launch.device+" timed out and was stopped"); }
        auto result=std::move(responses.front()); responses.pop_front(); lock.unlock();
        if(result.contains("error")) throw TtsError("RuntimeError","Native TTS worker on "+launch.device+": "+repr(result["error"])); return result;
    }
    Json request(const std::string &encoded,bool streaming,const std::string &output_file={}) {
        const auto data=Json::parse(encoded); Json out{{"text",data.at("text")},{"speaker",path(data.at("ref_audio"))}};
        if(!streaming) out["output"]=output_file; out["language"]=llama_tts_language(data.at("language").dump(),data.at("text").dump()); if(streaming) out["stream"]=true;
        out["max_new_tokens"]=Json::parse(integer(data.value("max_new_tokens",Json(320)))); return out;
    }
};
LlamaTts::LlamaTts(LlamaTtsLaunch launch):impl_(std::make_shared<Impl>(std::move(launch))) {
    try { impl_->start(); const auto ready=impl_->receive(); if(!ready.contains("ready") || !ready["ready"].is_boolean() || !ready["ready"].get<bool>()) throw TtsError("RuntimeError","Native TTS worker did not become ready"); } catch(...) { impl_->stop(); throw; }
}
LlamaTts::~LlamaTts()=default;
void LlamaTts::close() { if(impl_) impl_->stop(); }
bool LlamaTts::alive() const { return impl_ && impl_->alive(); }
TtsAudio LlamaTts::generate(const std::string &request) {
    std::lock_guard<std::recursive_mutex> lock(impl_->inference_mutex); if(!impl_->alive()) throw TtsError("RuntimeError","Native TTS worker is closed"); Temporary directory; const auto output=directory.directory+"/speech.wav"; const auto payload=impl_->request(request,false,output); impl_->send(payload); const auto response=impl_->receive();
    if(!response.contains("ok") || !response["ok"].is_boolean() || !response["ok"].get<bool>()) throw TtsError("RuntimeError","Invalid native TTS response"); return read_wav(output);
}
std::unique_ptr<TtsStreamIterator> LlamaTts::stream_iterator(const std::string &request) {
    struct Iterator final : TtsStreamIterator {
        std::shared_ptr<Impl> model;
        std::string request;
        std::unique_lock<std::recursive_mutex> lock;
        bool started=false,complete=false,closed=false;
        Iterator(std::shared_ptr<Impl> owner,std::string encoded):model(std::move(owner)),request(std::move(encoded)) {}
        ~Iterator() override { try { close(); } catch(...) {} }
        void close() override {
            if(closed) return;
            closed=true;
            if(started && !complete) model->stop();
            if(lock.owns_lock()) lock.unlock();
            model.reset();
        }
        std::optional<TtsAudio> next() override {
            if(closed) return std::nullopt;
            try {
                if(!started) {
                    lock=std::unique_lock<std::recursive_mutex>(model->inference_mutex);
                    if(!model->alive()) throw TtsError("RuntimeError","Native TTS worker is closed");
                    const auto payload=model->request(request,true);
                    started=true;
                    model->send(payload);
                }
                const auto response=model->receive();
                if(response.contains("pcm_f32")) {
                    TtsAudio audio; audio.samples=decode_pcm(response["pcm_f32"]);
                    if(!response.contains("sample_rate")) throw TtsError("KeyError","'sample_rate'");
                    audio.rate=integer(response["sample_rate"]); return audio;
                }
                if(response.contains("ok") && response["ok"].is_boolean() && response["ok"].get<bool>()) {
                    complete=true; close(); return std::nullopt;
                }
                throw TtsError("RuntimeError","Invalid native TTS streaming response");
            } catch(...) { close(); throw; }
        }
    };
    return std::make_unique<Iterator>(impl_,request);
}
void LlamaTts::stream(const std::string &request,const std::function<bool(const TtsAudio &)> &consume) {
    auto iterator=stream_iterator(request);
    try { while(const auto audio=iterator->next()) if(!consume(*audio)) break; }
    catch(...) { iterator->close(); throw; }
    iterator->close();
}
}
