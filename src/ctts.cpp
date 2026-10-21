#include "ezlocalai/ctts.hpp"
#include "ezlocalai/ctts_stream_bridge.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/ctts_audio.hpp"
#include "ezlocalai/ctts_paths.hpp"
#include "ezlocalai/tts_text.hpp"
#include "ezlocalai/llm_startup.hpp"
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <thread>
#include <cerrno>
#include <charconv>
#include <climits>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <filesystem>
#include <future>
#include <limits>
#include <mutex>
#include <sys/stat.h>
#include <unistd.h>
namespace ezlocalai {
namespace {
using Json=nlohmann::ordered_json;
namespace fs=std::filesystem;
bool truth(const Json &v) { if(v.is_null()) return false; if(v.is_boolean()) return v.get<bool>(); if(v.is_number()) return v.get<double>()!=0; if(v.is_string()) return !v.get_ref<const std::string &>().empty(); return !v.empty(); }
std::string type(const Json &v) { return v.is_null()?"NoneType":v.is_boolean()?"bool":v.is_string()?"str":v.is_array()?"list":v.is_object()?"dict":v.is_number_float()?"float":"int"; }
std::optional<std::string> string_or_default(const Json &v) { return v.is_string()?std::optional<std::string>(v.get<std::string>()):std::nullopt; }
[[noreturn]] void os_error(const fs::path &path,int error=errno) { throw TtsError(error==ENOENT?"FileNotFoundError":error==EACCES?"PermissionError":error==EEXIST?"FileExistsError":error==ENOTDIR?"NotADirectoryError":error==EISDIR?"IsADirectoryError":"OSError","[Errno "+std::to_string(error)+"] "+std::strerror(error)+": "+python_string_repr(path.string())); }
bool path_exists(const fs::path &path) { struct stat st{}; return stat(path.c_str(),&st)==0; }
void mkdirs(const fs::path &path) { if(::mkdir(path.c_str(),0777)==0) return; auto error=errno; if(error==ENOENT && path.parent_path()!=path && !path.parent_path().empty()) { mkdirs(path.parent_path()); if(::mkdir(path.c_str(),0777)==0) return; error=errno; } struct stat st{}; if(stat(path.c_str(),&st)==0 && S_ISDIR(st.st_mode)) return; os_error(path,error); }
void write_file(const fs::path &path,const std::string &bytes) { const int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC,0666); if(fd<0) os_error(path); struct Close {int fd;~Close(){::close(fd);}} closer{fd}; size_t written=0; while(written<bytes.size()) { const auto n=write(fd,bytes.data()+written,bytes.size()-written); if(n<0) { if(errno==EINTR) continue; os_error(path); } if(!n) os_error(path,EIO); written+=n; } }
std::vector<std::string> list_voices(const std::string &path) { auto *directory=opendir(path.c_str()); if(!directory) os_error(path); struct Close {DIR *dir;~Close(){closedir(dir);}} closer{directory}; std::vector<std::string> result; while(const auto *entry=readdir(directory)) { const std::string name=entry->d_name; if(name.size()<4) continue; auto suffix=name.substr(name.size()-4); for(auto &c:suffix) if(c>='A' && c<='Z') c+=32; if(suffix==".wav") result.push_back(fs::path(name).stem().string()); } std::sort(result.begin(),result.end()); return result; }
std::string uuid() { unsigned char bytes[16]; if(RAND_bytes(bytes,16)!=1) throw TtsError("RuntimeError","UUID entropy unavailable"); bytes[6]=(bytes[6]&15)|64; bytes[8]=(bytes[8]&63)|128; const char *hex="0123456789abcdef"; std::string out; for(auto byte:bytes) { out+=hex[byte>>4]; out+=hex[byte&15]; } return out; }
std::string base64(const std::string &bytes) { if(bytes.size()>INT_MAX) throw TtsError("OverflowError","Native audio base64 size exceeded"); std::string out(4*((bytes.size()+2)/3)+1,'\0'); const auto size=EVP_EncodeBlock(reinterpret_cast<unsigned char *>(out.data()),reinterpret_cast<const unsigned char *>(bytes.data()),bytes.size()); out.resize(size); return out; }
CttsAudioArray audio_array(const TtsAudio &audio) { if(audio.channels<=0 || audio.samples.size()%audio.channels) throw TtsError("ValueError","Invalid native audio shape"); return {audio.samples,audio.channels==1?std::vector<size_t>{audio.samples.size()}:std::vector<size_t>{audio.samples.size()/size_t(audio.channels),size_t(audio.channels)}}; }
}
struct Ctts::Impl {
    struct Stream;
    CttsServiceOperations ops; std::unique_ptr<CttsRuntime> model; std::unique_ptr<CttsAudioCache> cache; std::string output,voices_path,rate="24000"; std::vector<std::string> voices; Json use_cache; mutable std::mutex mutex;
};
Ctts::Ctts(const std::string &configuration,const std::string &device,CttsServiceOperations operations):impl_(std::make_shared<Impl>()) {
    impl_->ops=std::move(operations); impl_->ops.runtime=ctts_llama_operations(std::move(impl_->ops.runtime)); if(!impl_->ops.uuid_hex) impl_->ops.uuid_hex=uuid;
    impl_->model=std::make_unique<CttsRuntime>(device,impl_->ops.runtime);
    impl_->output=(fs::current_path()/"outputs").string(); mkdirs(impl_->output); impl_->voices_path=(fs::current_path()/"voices").string(); mkdirs(impl_->voices_path); impl_->voices=list_voices(impl_->voices_path);
    impl_->cache=std::make_unique<CttsAudioCache>(configuration,impl_->ops.cache); impl_->use_cache=Json::parse(impl_->cache->configuration()).at("enabled");
    if(path_exists(fs::path(impl_->voices_path)/"default.wav")) try { const auto path=ctts_voice_audio_path(impl_->voices_path,"default"); const auto kwargs=impl_->model->warmup_generation_kwargs(); impl_->model->generate_sample(Json{{"text","Hello."},{"audio_path",path?Json(*path):Json()},{"qwen_language","English"},{"generation_kwargs",Json::parse(kwargs)}}.dump()); } catch(const std::exception &) {}
}
Ctts::~Ctts()=default;
void Ctts::close() { impl_->model->close(); }
std::string Ctts::sample_rate() const { std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->rate; }
const std::vector<std::string> &Ctts::voices() const { return impl_->voices; }
const std::string &Ctts::output_folder() const { return impl_->output; }
const std::string &Ctts::voices_path() const { return impl_->voices_path; }
std::string Ctts::use_cache_json() const { return impl_->use_cache.dump(); }
CttsRuntime &Ctts::runtime() { return *impl_->model; }
CttsAudioCache &Ctts::cache() { return *impl_->cache; }
std::string Ctts::generate(const std::string &encoded) {
    const auto request=Json::parse(encoded); for(const auto &entry:request.items()) if(entry.key()!="text" && entry.key()!="voice" && entry.key()!="language" && entry.key()!="local_uri" && entry.key()!="output_file_name" && entry.key()!="use_cache") throw TtsError("TypeError","CTTS.generate() got an unexpected keyword argument '"+entry.key()+"'");
    if(!request.contains("text")) throw TtsError("TypeError","CTTS.generate() missing 1 required positional argument: 'text'");
    auto use_cache=request.value("use_cache",Json()); if(use_cache.is_null()) use_cache=impl_->use_cache;
    const auto voice=request.value("voice",Json("default")),raw_language=request.value("language",Json("auto")),raw_text=request.at("text"); const auto voice_name=tts_safe_file_stem(string_or_default(voice));
    const auto language_string=truth(raw_language)?tts_python_string(raw_language.dump()):"";
    if(!raw_text.is_string()) { if(tts_resolve_language(language_string,"")=="English" && !raw_text.is_structured()) throw TtsError("TypeError","'"+type(raw_text)+"' object is not iterable"); throw TtsError("TypeError","expected string or bytes-like object, got '"+type(raw_text)+"'"); }
    const auto language=tts_resolve_language(language_string,raw_text.get<std::string>()); const auto text=tts_prepare_text(raw_text.get<std::string>(),language); if(text.empty()) return "";
    const auto audio_path=ctts_voice_audio_path(impl_->voices_path,string_or_default(voice)); const auto &settings=impl_->model->settings();
    Json extra{{"engine","llama.cpp-qwen-tts-v1"},{"model",settings.model_id},{"x_vector_only",true},{"ref_text_sha256",""},{"generation",Json::parse(settings.generation_kwargs)}};
    for(const auto &field:std::vector<std::pair<std::string,std::string>>{{"model_file","QWEN_TTS_MODEL_FILE"},{"mmproj_file","QWEN_TTS_MMPROJ_FILE"},{"revision","QWEN_TTS_REVISION"}}) extra[field.first]=impl_->ops.runtime.environment(field.second).value_or("default");
    const auto uri=request.value("local_uri",Json());
    if(truth(use_cache)) { const auto key=ctts_cache_key(text,voice_name,language,extra.dump()); if(path_exists(fs::path(impl_->output)/"cache/audio"/(key+".wav"))) { const auto cached=impl_->cache->get(key); if(cached && !cached->empty()) return truth(uri)?tts_python_string(uri.dump())+"/outputs/cache/audio/"+key+".wav":base64(*cached); } }
    double maximum; try { maximum=std::stod(settings.max_chunk_chars); } catch(const std::out_of_range &) { maximum=std::numeric_limits<double>::infinity(); } const auto chunks=tts_split_text(text,maximum);
    std::vector<CttsAudioArray> generated; auto rate=sample_rate();
    for(const auto &chunk:chunks) { const auto call=Json{{"text",chunk},{"audio_path",audio_path?Json(*audio_path):Json()},{"qwen_language",language},{"ref_text",nullptr},{"x_vector_only",true}}.dump(); const auto audio=std::async(std::launch::async,[&]{return impl_->model->generate_sample(call);}).get(); rate=audio.rate; if(!audio.samples.empty()) generated.push_back(ctts_mono_float32(audio_array(audio))); }
    if(generated.empty()) return "";
    CttsAudioArray joined; joined.shape={0}; for(auto &audio:generated) { joined.samples.insert(joined.samples.end(),audio.samples.begin(),audio.samples.end()); joined.shape[0]+=audio.samples.size(); }
    { std::lock_guard<std::mutex> lock(impl_->mutex); impl_->rate=rate; }
    std::string bytes; try { bytes=ctts_wav_bytes(joined,std::stoll(rate)); } catch(const std::out_of_range &) {} if(bytes.empty()) return "";
    if(truth(use_cache)) { const auto key=ctts_cache_key(text,voice_name,language,extra.dump()); auto metadata=Json{{"text",text},{"voice",voice_name},{"language",language},{"generation_method","llama.cpp-qwen-tts"},{"model",settings.model_id},{"chunks",chunks.size()}}.dump(); impl_->cache->store(key,bytes,metadata); }
    if(truth(uri)) { const auto fallback=impl_->ops.uuid_hex(); const auto stem=tts_safe_file_stem(string_or_default(request.value("output_file_name",Json())),fallback); const auto path=ctts_contained_file_path(impl_->output,stem,".wav"); write_file(path,bytes); return tts_python_string(uri.dump())+"/outputs/"+stem+".wav"; }
    return base64(bytes);
}
namespace {
std::string packed_u32(const std::string &number) {
    if(configuration_integer_less(number,"0") || configuration_integer_less("4294967295",number)) throw TtsError("error","'I' format requires 0 <= number <= 4294967295");
    const auto value=std::stoull(number); std::string out; for(int shift=0;shift<32;shift+=8) out+=static_cast<char>(value>>shift); return out;
}
double stream_limit(const std::string &number) {try{return std::stod(number);}catch(const std::out_of_range &){return std::numeric_limits<double>::infinity();}}
size_t stream_write_limit(const std::string &number) {try{return std::stoull(number);}catch(const std::out_of_range &){return std::numeric_limits<size_t>::max();}}
std::string stream_silence(const std::string &rate,const std::string &milliseconds) {
    // int * int / 1000 is one correctly rounded Python floating division.
    // Decimal multiplication followed by a decimal point avoids double rounding.
    const auto multiplier=std::stoull(rate); uint64_t carry=0; std::string product;
    for(auto i=milliseconds.rbegin();i!=milliseconds.rend();++i){carry+=uint64_t(*i-'0')*multiplier;product+=char('0'+carry%10);carry/=10;}
    while(carry){product+=char('0'+carry%10);carry/=10;}std::reverse(product.begin(),product.end());
    if(product.size()<4)product.insert(0,4-product.size(),'0');product.insert(product.size()-3,1,'.');
    double value=0;const auto parsed=std::from_chars(product.data(),product.data()+product.size(),value);
    if(parsed.ec==std::errc::result_out_of_range)throw TtsError("OverflowError","integer division result too large for a float");
    const double count=std::max(1.0,std::trunc(value));
    if(static_cast<long double>(count)>std::numeric_limits<ssize_t>::max())throw TtsError("OverflowError","cannot fit 'int' into an index-sized integer");
    if(static_cast<long double>(count)>std::numeric_limits<ssize_t>::max()/2)throw TtsError("OverflowError","repeated bytes are too long");
    try{return std::string(size_t(count)*2,'\0');}catch(const std::bad_alloc &){throw TtsError("MemoryError","");}
}
}
struct Ctts::Impl::Stream final:CttsByteStream {
    enum Phase {prepare,start_chunk,get_audio,payload,after_payload,silence,drain,finished};
    std::shared_ptr<Impl> owner;
    Json request;
    std::string language,pcm;
    std::optional<std::string> audio_path;
    std::vector<std::string> chunks;
    size_t index=0,offset=0;
    Phase phase=prepare;
    bool active=false,closed=false;
    std::atomic<bool> cancelled{false};
    std::shared_ptr<CttsStreamBridge> bridge;
    std::mutex wait_mutex;
    std::condition_variable wake;
    Stream(std::shared_ptr<Impl> state,Json value):owner(std::move(state)),request(std::move(value)){}
    ~Stream() override {try{close();}catch(...) {}}
    std::string rate() const {std::lock_guard<std::mutex> lock(owner->mutex);return owner->rate;}
    void request_cancel() override {cancelled.store(true);wake.notify_all();if(const auto current=std::atomic_load(&bridge))current->request_cancel();}
    void checkpoint() {if(cancelled.load())throw TtsError("CancelledError","");}
    void sleep(double seconds) {
        checkpoint();
        if(owner->ops.stream_sleep)owner->ops.stream_sleep(seconds);
        else if(seconds==0)std::this_thread::yield();
        else {
            std::unique_lock<std::mutex> lock(wait_mutex);
            if(std::isinf(seconds))wake.wait(lock,[&]{return cancelled.load();});
            else while(seconds>0&&!cancelled.load()){
                const auto begin=std::chrono::steady_clock::now();
                wake.wait_for(lock,std::chrono::duration<double>(std::min(seconds,3600.0)),[&]{return cancelled.load();});
                seconds-=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            }
        }
        checkpoint();
    }
    void terminate(std::exception_ptr failure={}) {
        if(closed){if(failure)std::rethrow_exception(failure);return;}
        closed=true;
        if(active){if(const auto current=std::atomic_load(&bridge))try{current->close();}catch(...){failure=std::current_exception();}try{owner->model->close();}catch(...){failure=std::current_exception();}}
        std::atomic_store(&bridge,std::shared_ptr<CttsStreamBridge>());owner.reset();release_buffers();
        if(failure)std::rethrow_exception(failure);
    }
    void close() override {terminate();}
    void release_buffers() {std::string().swap(pcm);std::string().swap(language);std::vector<std::string>().swap(chunks);audio_path.reset();request=nullptr;}
    std::optional<std::string> next() override {
        if(closed)return std::nullopt;
        try {
            checkpoint();
            for(;;){
                const auto &settings=owner->model->settings();
                switch(phase){
                case prepare:{
                    const auto raw_language=request.value("language",Json("auto")),raw_text=request.at("text");const auto name=truth(raw_language)?tts_python_string(raw_language.dump()):"";
                    if(!raw_text.is_string()){if(tts_resolve_language(name,"")=="English"&&!raw_text.is_structured())throw TtsError("TypeError","'"+type(raw_text)+"' object is not iterable");throw TtsError("TypeError","expected string or bytes-like object, got '"+type(raw_text)+"'");}
                    language=tts_resolve_language(name,raw_text.get<std::string>());const auto text=tts_prepare_text(raw_text.get<std::string>(),language);
                    audio_path=ctts_voice_audio_path(owner->voices_path,string_or_default(request.value("voice",Json("default"))));
                    chunks=tts_split_stream_text(text,stream_limit(settings.stream_chunk_target_chars),stream_limit(settings.stream_first_chunk_target_chars),stream_limit(settings.stream_min_first_chunk_chars));
                    auto header=packed_u32(rate());header.append("\x10\0\x01\0",4);phase=start_chunk;return header;
                }
                case start_chunk:{
                    if(index==chunks.size()){phase=finished;return std::string(4,'\0');}
                    active=true;Json kwargs{{"text",chunks[index]},{"language",language},{"ref_audio",audio_path?Json(*audio_path):Json()}};
                    const auto generation=Json::parse(settings.generation_kwargs);
                    for(const auto &entry:generation.items()){if(kwargs.contains(entry.key()))throw TtsError("TypeError","ezlocalai.AudioStreaming.stream_native_audio() got multiple values for keyword argument '"+entry.key()+"'");kwargs[entry.key()]=entry.value();}
                    if(kwargs.contains("model"))throw TtsError("TypeError","stream_native_audio() got multiple values for argument 'model'");
                    auto current=std::make_shared<CttsStreamBridge>(owner->model->backend(),kwargs.dump());std::atomic_store(&bridge,current);if(cancelled.load())current->request_cancel();phase=get_audio;break;
                }
                case get_audio:{
                    const auto current=std::atomic_load(&bridge);const auto audio=current->next();
                    if(!audio){std::atomic_store(&bridge,std::shared_ptr<CttsStreamBridge>());phase=silence;break;}
                    if(tts_python_integer(Json(audio->rate).dump())!=rate())throw TtsError("RuntimeError","TTS sample rate changed after stream header");
                    pcm=ctts_pcm16_bytes(audio_array(*audio));if(pcm.empty())break;
                    const auto length=packed_u32(std::to_string(pcm.size()));offset=0;phase=payload;return length;
                }
                case payload:{const auto count=std::min(pcm.size()-offset,stream_write_limit(settings.stream_write_bytes));auto bytes=pcm.substr(offset,count);offset+=count;phase=after_payload;return bytes;}
                case after_payload:sleep(0);phase=offset<pcm.size()?payload:get_audio;break;
                case silence:phase=drain;if(index+1<chunks.size()&&settings.stream_flush_silence_ms!="0"){auto bytes=stream_silence(rate(),settings.stream_flush_silence_ms);return packed_u32(std::to_string(bytes.size()))+bytes;}break;
                case drain:if(settings.stream_frame_drain_seconds>0)sleep(settings.stream_frame_drain_seconds);active=false;++index;phase=start_chunk;break;
                case finished:closed=true;owner.reset();release_buffers();return std::nullopt;
                }
            }
        }catch(...){terminate(std::current_exception());throw;}
    }
};
std::unique_ptr<CttsByteStream> Ctts::generate_stream(const std::string &encoded) {
    const auto request=Json::parse(encoded);for(const auto &entry:request.items())if(entry.key()!="text"&&entry.key()!="voice"&&entry.key()!="language")throw TtsError("TypeError","CTTS.generate_stream() got an unexpected keyword argument '"+entry.key()+"'");
    if(!request.contains("text"))throw TtsError("TypeError","CTTS.generate_stream() missing 1 required positional argument: 'text'");
    return std::make_unique<Impl::Stream>(impl_,request);
}

}
