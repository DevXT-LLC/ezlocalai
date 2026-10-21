#include "ezlocalai/ctts.hpp"
#include "whisper_transcription_test.hpp"
#include <openssl/evp.h>
#include <thread>
#ifdef EZLOCALAI_CTTS_TORCH
#include "ezlocalai/ctts_torch.hpp"
#include <c10/core/InferenceMode.h>
#endif
using namespace transcription_test;
std::string substitute(std::string value,const std::string &from,const std::string &to) {size_t at=0;while((at=value.find(from,at))!=std::string::npos){value.replace(at,from.size(),to);at+=to.size();}return value;}
std::string file_hash(const fs::path &path) {std::ifstream input(path,std::ios::binary);expect(bool(input),"read "+path.string());const std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(),EVP_MD_CTX_free);expect(ctx&&EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr)==1,"digest init");char buffer[65536];while(input){input.read(buffer,sizeof(buffer));expect(EVP_DigestUpdate(ctx.get(),buffer,input.gcount())==1,"digest update");}expect(input.eof(),"complete file read");unsigned char digest[32];unsigned size=0;expect(EVP_DigestFinal_ex(ctx.get(),digest,&size)==1&&size==32,"digest finish");std::ostringstream out;for(auto c:digest)out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(c);return out.str();}
int main(int argc,char **argv){try{
 expect(argc==7,"source, fixture, worker, models, speaker and libsndfile");const auto rows=records(argv[2]);const auto &header=rows[0];for(const auto &item:std::vector<std::pair<std::string,std::string>>{{"ezlocalai/CTTS.py","source_sha256"},{"ezlocalai/AudioCache.py","cache_sha256"},{"ezlocalai/LlamaTTS.py","llama_sha256"},{"tts/worker.cpp","worker_sha256"}})expect(file_hash(fs::path(argv[1])/item.first)==header.at(item.second),"pinned "+item.first);expect(file_hash(argv[5])==header.at("speaker_sha256"),"pinned speaker");expect(file_hash(argv[6])==header.at("libsndfile_sha256"),"pinned WAV encoder");for(const auto &item:header.at("model_hashes").items())expect(file_hash(fs::path(argv[4])/item.key())==item.value(),"pinned GGUF");
 for(const auto &entry:header.at("environment").items())setenv(entry.key().c_str(),entry.value().get<std::string>().c_str(),1);setenv("QWEN_TTS_BIN",argv[3],1);setenv("QWEN_TTS_MODEL_FILE",(fs::path(argv[4])/"qwen-talker-0.6b-base-Q8_0.gguf").c_str(),1);setenv("QWEN_TTS_MMPROJ_FILE",(fs::path(argv[4])/"qwen-tokenizer-12hz-f16.gguf").c_str(),1);
 Temporary temporary;const auto original=fs::current_path();fs::current_path(temporary.path);struct Restore{fs::path path;~Restore(){fs::current_path(path);}}restore{original};fs::create_directories("voices");fs::copy_file(argv[5],"voices/default.wav");size_t clock=0,uuid=0,calls=0;const auto main=std::this_thread::get_id();CttsServiceOperations ops;
#ifdef EZLOCALAI_CTTS_TORCH
 ops.runtime=ctts_torch_operations();expect(!c10::InferenceMode::is_enabled(),"initial inference mode");
#endif
 // Cache identity includes these source environment strings; the worker itself
 // uses the real model paths supplied above, so fixtures remain relocatable.
 const auto runtime_environment=ctts_llama_operations().environment;
 ops.runtime.environment=[&,runtime_environment](const std::string &key)->std::optional<std::string>{if(header.at("environment").contains(key))return header.at("environment").at(key).get<std::string>();return runtime_environment(key);};
 const auto load=ctts_llama_operations().load;ops.runtime.load=[&](const std::string &device,const std::string &kwargs){auto backend=load(device,kwargs);const auto generate=backend->generate;backend->generate=[&,generate](const std::string &request){expect((std::this_thread::get_id()==main)==(calls++==0),"warmup caller thread, request worker thread");
#ifdef EZLOCALAI_CTTS_TORCH
 expect(c10::InferenceMode::is_enabled(),"worker inference context");
#endif
 return generate(request);};return backend;};
 ops.cache.now=[&]{std::ostringstream out;out<<"2026-10-17T12:00:00";if(clock)out<<'.'<<std::setw(6)<<std::setfill('0')<<clock;++clock;return out.str();};ops.uuid_hex=[&]{std::ostringstream out;out<<std::hex<<std::setfill('0')<<std::setw(32)<<++uuid;return out.str();};Ctts owner("null","\"cpu\"",std::move(ops));
 const auto snapshot=[&]{Json files=Json::object();for(const auto &entry:fs::recursive_directory_iterator(temporary.path)){const auto key=entry.path().lexically_relative(temporary.path).string();if(entry.is_directory())files[key]={{"kind","dir"}};else files[key]={{"kind","file"},{"size",entry.file_size()},{"sha256",file_hash(entry.path())}};}return Json{{"files",files},{"now_calls",clock},{"uuid_calls",uuid},{"voices",owner.voices()},{"sample_rate",owner.sample_rate()},{"device",owner.runtime().device()},{"closed",owner.runtime().closed()},{"model",bool(owner.runtime().backend())},{"cache_config",Json::parse(owner.cache().configuration())},{"cache_stats",Json::parse(owner.cache().raw_statistics())}};};
 const auto normalized=[&]{return Json::parse(substitute(snapshot().dump(),temporary.path.string(),"<ROOT>"));};
 compare_json(normalized(),header.at("startup"),"real constructor");
 for(size_t i=1;i<rows.size();++i){const auto &action=rows[i].at("action");Json result={{"value",nullptr}};try{if(action.at("op")=="generate"){const auto value=owner.generate(action.at("request").dump());result["value"]={{"size",value.size()},{"sha256",hash(value)}};}else if(action.at("op")=="remove")fs::remove("voices/default.wav");else owner.close();}catch(const TtsError &e){result={{"error",{e.category,e.what()}}};}result["state"]=normalized();compare_json(result,rows[i].at("result"),"real buffered generation "+std::to_string(i));
#ifdef EZLOCALAI_CTTS_TORCH
 expect(!c10::InferenceMode::is_enabled(),"caller inference mode restored");
#endif
 }
 expect(calls==4,"constructor warmup and three real synthesis chunks");expect(rows.size()-1==header.at("cases"),"all actual buffered generation cases");std::cout<<rows.size()-1<<" real CTTS buffered operations match exact results, WAV files and metadata\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
