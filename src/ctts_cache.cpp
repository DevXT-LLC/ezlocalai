#include "ezlocalai/ctts_cache.hpp"
#include "ezlocalai/tts_text.hpp"
#include "ezlocalai/llama_tts.hpp"
#include "ezlocalai/llm_startup.hpp"
#include "ezlocalai/process.hpp"
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <ctime>
#include <cstring>
#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <regex>
#include <sstream>
#include <sys/stat.h>
namespace ezlocalai {
std::string python_float_string(double);
namespace {
using Json=nlohmann::ordered_json;
namespace fs=std::filesystem;
bool truth(const Json &v) { if(v.is_null()) return false; if(v.is_boolean()) return v.get<bool>(); if(v.is_number()) return v.get<double>()!=0; if(v.is_string()) return !v.get_ref<const std::string &>().empty(); return !v.empty(); }
std::string type(const Json &v) { return v.is_null()?"NoneType":v.is_boolean()?"bool":v.is_string()?"str":v.is_object()?"dict":v.is_array()?"list":v.is_number_float()?"float":"int"; }
const Json &field(const Json &v,const std::string &key) { if(!v.is_object()) throw TtsError("TypeError",type(v)=="list"?"list indices must be integers or slices, not str":"'"+type(v)+"' object is not subscriptable"); if(!v.contains(key)) throw TtsError("KeyError",python_string_repr(key)); return v.at(key); }
double number(const Json &v) { if(v.is_boolean()) return v.get<bool>()?1:0; if(!v.is_number()) throw TtsError("TypeError","unsupported operand type(s) for numeric operation: '"+type(v)+"'"); return v.get<double>(); }
Json add(const Json &a,const Json &b) { if(a.is_string()) { if(b.is_string()) return a.get<std::string>()+b.get<std::string>(); throw TtsError("TypeError","can only concatenate str (not \""+type(b)+"\") to str"); } if((a.is_number()||a.is_boolean())&&(b.is_number()||b.is_boolean())) { if(a.is_number_float()||b.is_number_float()) return number(a)+number(b); const auto ai=a.is_boolean()?int64_t(a.get<bool>()):a.get<int64_t>(),bi=b.is_boolean()?int64_t(b.get<bool>()):b.get<int64_t>(); if((bi>0&&ai>INT64_MAX-bi)||(bi<0&&ai<INT64_MIN-bi)) throw TtsError("OverflowError","native cache integer range exceeded"); return ai+bi; } throw TtsError("TypeError","unsupported operand type(s) for +: '"+type(a)+"' and '"+type(b)+"'"); }
std::string ascii_string(const std::string &v) { const auto encoded=Json(v).dump(-1,' ',true); std::string out; for(unsigned char c:encoded) { if(c==127) out+="\\u007f"; else out+=static_cast<char>(c); } return out; }
std::string dump(const Json &v,int indent=-1,int depth=0,bool sorted=false) {
    if(v.is_string()) return ascii_string(v.get<std::string>()); if(v.is_number_float()) return python_float_string(v.get<double>()); if(!v.is_structured()) return v.dump(); if(v.empty()) return v.is_object()?"{}":"[]";
    std::vector<std::pair<std::string,Json>> items; for(const auto &item:v.items()) items.push_back({item.key(),item.value()}); if(sorted && v.is_object()) std::sort(items.begin(),items.end(),[](const auto &a,const auto &b){return a.first<b.first;}); std::string out=v.is_object()?"{":"[";
    for(size_t i=0;i<items.size();++i) { if(i) out+=indent<0?", ":","; if(indent>=0) out+='\n'+std::string((depth+1)*indent,' '); if(v.is_object()) out+=ascii_string(items[i].first)+": "; out+=dump(items[i].second,indent,depth+1,sorted); }
    if(indent>=0) out+='\n'+std::string(depth*indent,' '); return out+(v.is_object()?"}":"]");
}
[[noreturn]] void os_error(const fs::path &p,int error=errno) { const auto category=error==ENOENT?"FileNotFoundError":error==EACCES?"PermissionError":error==EEXIST?"FileExistsError":error==ENOTDIR?"NotADirectoryError":error==EISDIR?"IsADirectoryError":"OSError"; throw TtsError(category,"[Errno "+std::to_string(error)+"] "+std::strerror(error)+": "+python_string_repr(p.string())); }
void valid_path(const fs::path &p) { if(p.string().find('\0')!=std::string::npos) throw TtsError("ValueError","embedded null byte"); }
bool path_exists(const fs::path &p) { struct stat st{}; return p.string().find('\0')==std::string::npos && stat(p.c_str(),&st)==0; }
void mkdirs(const fs::path &p) { valid_path(p); if(::mkdir(p.c_str(),0777)==0) return; auto error=errno; if(error==ENOENT && !p.parent_path().empty() && p.parent_path()!=p) { mkdirs(p.parent_path()); if(::mkdir(p.c_str(),0777)==0) return; error=errno; } struct stat st{}; if(stat(p.c_str(),&st)==0 && S_ISDIR(st.st_mode)) return; os_error(p,error); }
std::string read_file(const fs::path &p) { valid_path(p); const int fd=open(p.c_str(),O_RDONLY|O_CLOEXEC); if(fd<0) os_error(p); struct Close{int fd;~Close(){close(fd);}} close_file{fd}; std::string out; char buffer[16384]; for(;;) { const auto n=read(fd,buffer,sizeof(buffer)); if(n<0) { if(errno==EINTR) continue; os_error(p); } if(!n) break; out.append(buffer,n); } return out; }
void write_file(const fs::path &p,const std::string &bytes) { valid_path(p); const int fd=open(p.c_str(),O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC,0666); if(fd<0) os_error(p); struct Close{int fd;~Close(){close(fd);}} close_file{fd}; size_t offset=0; while(offset<bytes.size()) { const auto n=write(fd,bytes.data()+offset,bytes.size()-offset); if(n<0) { if(errno==EINTR) continue; os_error(p); } if(!n) os_error(p,EIO); offset+=n; } }
std::vector<fs::path> glob(const fs::path &base,const std::string &suffix) { std::vector<fs::path> out; auto *dir=opendir(base.c_str()); if(!dir) return out; struct Close{DIR *dir;~Close(){closedir(dir);}} close_dir{dir}; while(const auto *entry=readdir(dir)) { const std::string name=entry->d_name; if(name.size()>=suffix.size() && name.compare(name.size()-suffix.size(),suffix.size(),suffix)==0) out.push_back(base/name); } return out; }
void unlink_if_exists(const fs::path &p) { if(path_exists(p) && unlink(p.c_str())<0) os_error(p); }
void remove_tree(const fs::path &p) { struct stat st{}; if(lstat(p.c_str(),&st)<0 || S_ISLNK(st.st_mode) || !S_ISDIR(st.st_mode)) return; auto *dir=opendir(p.c_str()); if(!dir) return; while(const auto *e=readdir(dir)) { const std::string name=e->d_name; if(name=="."||name=="..") continue; const auto child=p/name; if(lstat(child.c_str(),&st)<0) continue; if(S_ISDIR(st.st_mode)) remove_tree(child); else unlink(child.c_str()); } closedir(dir); rmdir(p.c_str()); }
std::string local_now() { using namespace std::chrono; const auto us=duration_cast<microseconds>(system_clock::now().time_since_epoch()).count(); const time_t sec=us/1000000; tm date{}; localtime_r(&sec,&date); char buffer[32]; std::strftime(buffer,sizeof(buffer),"%Y-%m-%dT%H:%M:%S",&date); std::ostringstream out; out<<buffer; if(us%1000000) out<<'.'<<std::setw(6)<<std::setfill('0')<<us%1000000; return out.str(); }
int64_t timestamp(const std::string &original,bool *aware=nullptr) {
    std::string value=original; if(aware) { *aware=false; if(value.size()>10 && value.back()=='Z') { *aware=true; value.pop_back(); } else { const auto sign=value.find_first_of("+-",10); if(sign!=std::string::npos) { const auto zone=value.substr(sign); if(!std::regex_match(zone,std::regex(R"([+-](?:[01][0-9]|2[0-3])(?::?[0-5][0-9])?(?::?[0-5][0-9](?:[.,][0-9]+)?)?)"))) throw TtsError("ValueError","Invalid timezone"); *aware=true; value.resize(sign); } } }
    // Cache datetime.now() is naive; aware metadata cannot be subtracted from it.
    static const std::regex pattern(R"(^(\d{4})-?(\d{2})-?(\d{2})(?:.(\d{2})(?::?(\d{2}))?(?::?(\d{2}))?(?:[.,](\d+))?)?$)"); std::smatch match; if(!std::regex_match(value,match,pattern)) throw TtsError("ValueError","Invalid isoformat string: "+python_string_repr(value));
    tm date{}; const int year=std::stoi(match[1]),month=std::stoi(match[2]),day=std::stoi(match[3]); date.tm_year=year-1900; date.tm_mon=month-1; date.tm_mday=day; date.tm_hour=match[4].matched?std::stoi(match[4]):0; date.tm_min=match[5].matched?std::stoi(match[5]):0; date.tm_sec=match[6].matched?std::stoi(match[6]):0; if(year<1 || year>9999 || month<1 || month>12 || day<1 || day>31 || date.tm_hour>23 || date.tm_min>59 || date.tm_sec>59) throw TtsError("ValueError","Invalid date"); const auto seconds=timegm(&date); if(date.tm_mon!=month-1 || date.tm_mday!=day) throw TtsError("ValueError","Invalid date"); std::string fraction=match[7]; fraction.resize(6,'0'); return int64_t(seconds)*1000000+std::stoll(fraction);
}
long double ttl_us(const Json &v) { const auto days=number(v); if(!std::isfinite(days) || std::abs(days)>999999999) throw TtsError("OverflowError","timedelta too large"); return std::nearbyint(static_cast<long double>(days)*86400000000.0L); }
Json new_stats(const std::string &now) { return { {"hits",0},{"misses",0},{"stores",0},{"errors",0},{"expired",0},{"last_reset",now} }; }
bool less(const Json &a,const Json &b) { if(a.is_string()&&b.is_string()) return a.get<std::string>()<b.get<std::string>(); if((a.is_number()||a.is_boolean())&&(b.is_number()||b.is_boolean())) return number(a)<number(b); throw TtsError("TypeError","'<' not supported between instances of '"+type(a)+"' and '"+type(b)+"'"); }
// NumPy float64 pairwise sum order, as in loops_utils.h.src; licenses/NumPy.txt.
double sum(const std::vector<double> &v,size_t begin,size_t n) { if(n<8) { double out=-0.0; for(size_t i=0;i<n;++i) out+=v[begin+i]; return out; } if(n<=128) { double a[8]; std::copy_n(v.begin()+begin,8,a); size_t i=8; for(;i<n-n%8;i+=8) for(size_t j=0;j<8;++j) a[j]+=v[begin+i+j]; auto out=((a[0]+a[1])+(a[2]+a[3]))+((a[4]+a[5])+(a[6]+a[7])); for(;i<n;++i) out+=v[begin+i]; return out; } auto half=n/2; half-=half%8; return sum(v,begin,half)+sum(v,begin+half,n-half); }
}
std::string ctts_cache_key(const std::string &text,const std::string &voice,const std::string &language,const std::string &encoded) { Json input{{"text",tts_cache_normalize_text(text)},{"voice",voice},{"language",language}}; const auto extra=Json::parse(encoded); if(truth(extra)) { if(!extra.is_object()) throw TtsError("TypeError","cache extra parameters must be a mapping"); for(const auto &entry:extra.items()) input[entry.key()]=entry.value(); } const auto bytes=dump(input,-1,0,true); unsigned char hash[32]; SHA256(reinterpret_cast<const unsigned char *>(bytes.data()),bytes.size(),hash); std::ostringstream out; for(auto c:hash) out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(c); return out.str(); }
struct CttsAudioCache::Impl {
    Json config,stats; fs::path base,audio,metadata,stats_path; CttsCacheOperations ops;
    void save() { try { write_file(stats_path,dump(stats,2)); } catch(const std::exception &) {} }
    void record(const std::string &event) { if(!truth(field(config,"enable_stats"))) return; const auto key=event=="hit"?"hits":event=="miss"?"misses":event=="store"?"stores":event=="error"?"errors":event=="expired"?"expired":""; if(*key) stats[key]=add(field(stats,key),1); Json total=0; for(const auto *name:{"hits","misses","stores","errors"}) total=add(total,field(stats,name)); if(std::fmod(number(total),10)==0) save(); }
};
CttsAudioCache::CttsAudioCache(const std::string &encoded,CttsCacheOperations ops):impl_(std::make_unique<Impl>()) {
    impl_->ops=std::move(ops); if(!impl_->ops.now) impl_->ops.now=local_now; if(!impl_->ops.duration_ms) impl_->ops.duration_ms=ctts_cache_audio_duration;
    impl_->config={{"enabled",true},{"cache_dir",(fs::current_path()/"outputs/cache").string()},{"max_size_mb",500},{"ttl_days",30},{"generation_count",2},{"selection_method","shortest"},{"warm_cache_on_init",false},{"common_phrases",Json::array({"Hello","Goodbye","Thank you","Yes","No"})},{"enable_stats",true}};
    const auto input=Json::parse(encoded); if(truth(input)) { if(!input.is_object()) throw TtsError("TypeError","'"+type(input)+"' object is not a mapping"); for(const auto &item:input.items()) impl_->config[item.key()]=item.value(); }
    const auto &path=field(impl_->config,"cache_dir"); if(!path.is_string()) throw TtsError("TypeError","argument should be a str or an os.PathLike object where __fspath__ returns a str, not '"+type(path)+"'"); impl_->base=fs::path(path.get<std::string>()); if(impl_->base.empty()) impl_->base="."; impl_->audio=impl_->base/"audio"; impl_->metadata=impl_->base/"metadata"; impl_->stats_path=impl_->base/"stats.json";
    mkdirs(impl_->audio); mkdirs(impl_->metadata); bool loaded=false; if(path_exists(impl_->stats_path)) try { impl_->stats=Json::parse(read_file(impl_->stats_path)); loaded=true; } catch(const std::exception &) {} if(!loaded) impl_->stats=new_stats(impl_->ops.now()); if(truth(field(impl_->config,"enabled"))) { cleanup_expired(); check_size(); }
}
CttsAudioCache::~CttsAudioCache()=default;
std::string CttsAudioCache::configuration() const { return impl_->config.dump(); }
std::string CttsAudioCache::raw_statistics() const { return impl_->stats.dump(); }
std::optional<std::string> CttsAudioCache::get(const std::string &key) {
    if(!truth(field(impl_->config,"enabled"))) return {}; const auto audio=impl_->audio/(key+".wav"),metadata=impl_->metadata/(key+".json"); if(!path_exists(audio)||!path_exists(metadata)) { impl_->record("miss"); return {}; }
    try { auto record=Json::parse(read_file(metadata)); bool aware=false; const auto created=timestamp(field(record,"created_at").get<std::string>(),&aware); const auto now=timestamp(impl_->ops.now()); if(aware) throw TtsError("TypeError","can't subtract offset-naive and offset-aware datetimes"); if(now-created>ttl_us(field(impl_->config,"ttl_days"))) { remove(key); impl_->record("expired"); return {}; } record["last_accessed"]=impl_->ops.now(); record["access_count"]=add(record.value("access_count",Json(0)),1); write_file(metadata,dump(record,2)); auto bytes=read_file(audio); impl_->record("hit"); return bytes; } catch(const std::exception &) { impl_->record("error"); return {}; }
}
bool CttsAudioCache::store(const std::string &key,const std::string &bytes,std::string &encoded) {
    if(!truth(field(impl_->config,"enabled"))) return false;
    try { auto record=Json::parse(encoded); const auto assign=[&](const std::string &name,const Json &value) { if(!record.is_object()) throw TtsError("TypeError","metadata is not a mapping"); record[name]=value; encoded=record.dump(); }; assign("created_at",impl_->ops.now()); assign("last_accessed",impl_->ops.now()); assign("access_count",0); assign("cache_key",key); write_file(impl_->audio/(key+".wav"),bytes); write_file(impl_->metadata/(key+".json"),dump(record,2)); impl_->record("store"); check_size(); return true; } catch(const std::exception &) { return false; }
}
void CttsAudioCache::remove(const std::string &key) { unlink_if_exists(impl_->audio/(key+".wav")); unlink_if_exists(impl_->metadata/(key+".json")); }
void CttsAudioCache::cleanup_expired() { if(!truth(field(impl_->config,"enabled"))) return; const auto now=timestamp(impl_->ops.now()); for(const auto &path:glob(impl_->metadata,".json")) try { const auto record=Json::parse(read_file(path)); bool aware=false; const auto created=timestamp(field(record,"created_at").get<std::string>(),&aware); if(aware) throw TtsError("TypeError","can't subtract offset-naive and offset-aware datetimes"); if(now-created>ttl_us(field(impl_->config,"ttl_days"))) remove(path.stem().string()); } catch(const std::exception &) {} }
void CttsAudioCache::check_size() {
    if(!truth(field(impl_->config,"enabled"))) return; const auto limit=number(field(impl_->config,"max_size_mb"))*1024*1024; uint64_t total=0; struct Entry {std::string key;int64_t size;Json accessed;}; std::vector<Entry> entries;
    for(const auto &path:glob(impl_->audio,".wav")) { struct stat st{}; if(stat(path.c_str(),&st)<0) os_error(path); total+=st.st_size; const auto metadata=impl_->metadata/(path.stem().string()+".json"); if(path_exists(metadata)) try { const auto record=Json::parse(read_file(metadata)); const auto created=field(record,"created_at"); entries.push_back({path.stem().string(),st.st_size,record.value("last_accessed",created)}); } catch(const std::exception &) {} }
    if(total>limit) { std::stable_sort(entries.begin(),entries.end(),[](const auto &a,const auto &b){return less(a.accessed,b.accessed);}); for(const auto &entry:entries) { if(!(total>limit)) break; remove(entry.key); total-=entry.size; } }
}
std::string CttsAudioCache::statistics() const { const auto hits=field(impl_->stats,"hits"); const auto misses=field(impl_->stats,"misses"); const auto requests=add(hits,misses); const auto count=number(requests),rate=count>0?number(field(impl_->stats,"hits"))/count*100:0; std::ostringstream formatted; formatted<<std::fixed<<std::setprecision(2)<<rate<<'%'; auto result=impl_->stats; result["hit_rate"]=formatted.str(); result["total_requests"]=requests; return result.dump(); }
void CttsAudioCache::clear(const std::optional<std::string> &voice) { if(voice && !voice->empty()) { for(const auto &path:glob(impl_->metadata,".json")) try { const auto record=Json::parse(read_file(path)); if(record.value("voice",Json())==*voice) remove(path.stem().string()); } catch(const std::exception &) {} } else { remove_tree(impl_->audio); remove_tree(impl_->metadata); mkdirs(impl_->audio); mkdirs(impl_->metadata); impl_->stats=new_stats(impl_->ops.now()); impl_->save(); } }
CttsCacheSelection CttsAudioCache::select_best_audio(const CttsCacheGenerator &generate,const std::string &parameters) {
    const auto &raw=field(impl_->config,"generation_count"); if(!raw.is_number_integer() && !raw.is_boolean()) throw TtsError("TypeError","'"+type(raw)+"' object cannot be interpreted as an integer"); const auto count=raw.is_boolean()?int64_t(raw.get<bool>()):raw.get<int64_t>(); const auto method=field(impl_->config,"selection_method"); struct Sample {std::string data;int64_t duration,index;}; std::vector<Sample> samples; std::vector<double> durations; Json all=Json::array();
    for(int64_t i=0;i<count;++i) try { const auto bytes=generate(parameters); const auto path=impl_->base/("temp_"+std::to_string(i)+".wav"); write_file(path,bytes); const auto duration=impl_->ops.duration_ms(path.string()); samples.push_back({bytes,duration,i}); durations.push_back(duration); all.push_back(duration); if(unlink(path.c_str())<0) os_error(path); } catch(const std::exception &) {}
    if(samples.empty()) throw TtsError("ValueError","Failed to generate any audio samples"); auto sorted=durations; std::sort(sorted.begin(),sorted.end()); const auto median=sorted.size()%2?sorted[sorted.size()/2]:(sorted[sorted.size()/2-1]+sorted[sorted.size()/2])/2; size_t chosen=0; for(size_t i=1;i<samples.size();++i) if(method=="statistical"?std::abs(durations[i]-median)<std::abs(durations[chosen]-median):durations[i]<durations[chosen]) chosen=i; const auto mean=sum(durations,0,durations.size())/durations.size(); std::vector<double> squares; for(auto value:durations) squares.push_back((value-mean)*(value-mean)); const auto deviation=std::sqrt(sum(squares,0,squares.size())/squares.size()); const Json metadata{{"selection_method",method},{"generation_count",samples.size()},{"selected_index",samples[chosen].index},{"selected_duration_ms",samples[chosen].duration},{"all_durations_ms",all},{"mean_duration_ms",mean},{"std_duration_ms",deviation},{"min_duration_ms",static_cast<int64_t>(sorted.front())},{"max_duration_ms",static_cast<int64_t>(sorted.back())}}; return {samples[chosen].data,metadata.dump()};
}
void CttsAudioCache::warm(const std::vector<std::string> &phrases,const std::string &voice,const std::string &language,const CttsCacheGenerator &generate,const std::string &base) { if(!truth(field(impl_->config,"enabled"))||!truth(field(impl_->config,"warm_cache_on_init"))) return; for(const auto &phrase:phrases) { const auto key=ctts_cache_key(phrase,voice,language); if(path_exists(impl_->audio/(key+".wav"))) continue; try { auto parameters=Json::parse(base); if(!parameters.is_object()) throw TtsError("TypeError","base parameters must be a mapping"); parameters["text"]=phrase; parameters["voice"]=voice; parameters["language"]=language; auto chosen=select_best_audio(generate,parameters.dump()); auto metadata=Json::parse(chosen.metadata); metadata["text"]=phrase; metadata["voice"]=voice; metadata["language"]=language; metadata["warmed"]=true; auto encoded=metadata.dump(); store(key,chosen.audio,encoded); } catch(const std::exception &) {} } }
}
namespace ezlocalai {
namespace {
int64_t wav_duration(const std::string &bytes) {
    const auto word=[&](size_t offset,int size)->uint32_t { if(offset+size>bytes.size()) throw TtsError("CouldntDecodeError","Truncated WAV header"); uint32_t out=0; for(int i=0;i<size;++i) out|=uint32_t(static_cast<unsigned char>(bytes[offset+i]))<<(i*8); return out; };
    size_t offset=12,fmt=0,data=0,data_size=0; int headers=0;
    while(offset+8<=bytes.size() && headers++<10) { const auto size=word(offset+4,4); const auto id=bytes.substr(offset,4); if(id=="fmt " && !fmt) { if(size<16) throw TtsError("CouldntDecodeError","Short WAV format"); fmt=offset+8; } if(id=="data") { data=offset+8; data_size=std::min<size_t>(size,bytes.size()-data); break; } offset+=uint64_t(size)+8; }
    if(!fmt || !data) throw TtsError("CouldntDecodeError","Missing WAV header"); const auto format=word(fmt,2),channels=word(fmt+2,2),rate=word(fmt+4,4),width=word(fmt+14,2)/8;
    if((format!=1 && format!=0xfffe) || !channels || !rate || !width || (width==3 && data_size%3)) throw TtsError("CouldntDecodeError","Unsupported WAV format");
    const auto frames=std::floor(double(data_size)/(channels*width)); return static_cast<int64_t>(std::nearbyint(1000*(frames/rate)));
}
struct DurationTemporary { fs::path path; DurationTemporary() { auto name=(fs::temp_directory_path()/"ezlocalai-cache-duration-XXXXXX").string(); if(!mkdtemp(name.data())) os_error(name); path=name; } ~DurationTemporary() { std::error_code error; fs::remove_all(path,error); } };
}
int64_t ctts_cache_audio_duration(const std::string &path) {
    try { return wav_duration(read_file(path)); } catch(const std::exception &) {}
    DurationTemporary temp; const auto output=temp.path/"output",error=temp.path/"error";
    run_process_to_files({"ffprobe","-of","json","-v","info","-show_format","-show_streams",path},output.string(),error.string()); const auto info=Json::parse(read_file(output)); std::vector<std::string> command{"ffmpeg","-y","-i",path};
    if(!info.empty()) { Json stream; for(const auto &item:info.at("streams")) if(item.value("codec_type","")=="audio") { stream=item; break; } if(stream.is_null()) throw TtsError("CouldntDecodeError","No audio stream"); const auto format=stream.value("sample_fmt",""),codec=stream.value("codec_name",""); int bits=stream.value("bits_per_sample",0); if(!bits) { std::smatch match; if(std::regex_match(format,match,std::regex("[su]([0-9]{1,2})p?"))) bits=std::stoi(match[1]); else if(format=="flt"||format=="fltp") bits=32; else if(format=="dbl"||format=="dblp") bits=64; }
        if(format=="fltp" && (codec=="mp3"||codec=="mp4"||codec=="aac"||codec=="webm"||codec=="ogg")) bits=16; command.insert(command.end(),{"-acodec",bits==8?"pcm_u8":"pcm_s"+std::to_string(bits)+"le"});
    }
    command.insert(command.end(),{"-vn","-f","wav","-"}); const auto result=run_process_to_files(command,output.string(),error.string()); if(result.status) throw TtsError("CouldntDecodeError","Audio duration decoding failed"); return wav_duration(read_file(output));
}
}
