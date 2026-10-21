#include "ezlocalai/ctts.hpp"
#include "whisper_transcription_test.hpp"
#include <openssl/evp.h>
#include <thread>
#ifdef EZLOCALAI_CTTS_TORCH
#include "ezlocalai/ctts_torch.hpp"
#include <c10/core/InferenceMode.h>
#endif
using namespace transcription_test;
std::string file_hash(const fs::path &path) {std::ifstream input(path,std::ios::binary);expect(bool(input),"read "+path.string());const std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(),EVP_MD_CTX_free);expect(ctx&&EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr)==1,"digest init");char buffer[65536];while(input){input.read(buffer,sizeof(buffer));expect(EVP_DigestUpdate(ctx.get(),buffer,input.gcount())==1,"digest update");}expect(input.eof(),"complete file read");unsigned char digest[32];unsigned size=0;expect(EVP_DigestFinal_ex(ctx.get(),digest,&size)==1&&size==32,"digest finish");std::ostringstream out;for(auto c:digest)out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(c);return out.str();}
int main(int argc,char **argv){try{
 expect(argc==6,"source, fixture, worker, models and speaker");const auto rows=records(argv[2]);const auto &header=rows[0];for(const auto &item:std::vector<std::pair<std::string,std::string>>{{"ezlocalai/CTTS.py","source_sha256"},{"ezlocalai/LlamaTTS.py","llama_sha256"},{"ezlocalai/AudioStreaming.py","bridge_sha256"},{"tts/worker.cpp","worker_sha256"}})expect(file_hash(fs::path(argv[1])/item.first)==header.at(item.second),"pinned "+item.first);expect(file_hash(argv[5])==header.at("speaker_sha256"),"pinned speaker");for(const auto &item:header.at("model_hashes").items())expect(file_hash(fs::path(argv[4])/item.key())==item.value(),"pinned GGUF");for(const auto &entry:header.at("environment").items())setenv(entry.key().c_str(),entry.value().get<std::string>().c_str(),1);setenv("QWEN_TTS_BIN",argv[3],1);setenv("QWEN_TTS_MODEL_FILE",(fs::path(argv[4])/"qwen-talker-0.6b-base-Q8_0.gguf").c_str(),1);setenv("QWEN_TTS_MMPROJ_FILE",(fs::path(argv[4])/"qwen-tokenizer-12hz-f16.gguf").c_str(),1);
 Temporary temporary;const auto original=fs::current_path();fs::current_path(temporary.path);struct Restore{fs::path path;~Restore(){fs::current_path(path);}}restore{original};fs::create_directories("voices");fs::copy_file(argv[5],"voices/default.wav");const auto main=std::this_thread::get_id();size_t streams=0;CttsServiceOperations ops;
#ifdef EZLOCALAI_CTTS_TORCH
 ops.runtime=ctts_torch_operations();expect(!c10::InferenceMode::is_enabled(),"initial inference mode");
#endif
 const auto load=ctts_llama_operations().load;ops.runtime.load=[&](const auto &device,const auto &kwargs){auto backend=load(device,kwargs);const auto factory=backend->stream;backend->stream=[&,factory](const auto &request){expect(std::this_thread::get_id()!=main,"stream iterator created on producer");++streams;
#ifdef EZLOCALAI_CTTS_TORCH
 expect(!c10::InferenceMode::is_enabled(),"streaming does not enter buffered inference context");
#endif
 return factory(request);};return backend;};Ctts owner("{\"enabled\":false}","\"cpu\"",std::move(ops));const auto state=[&]{return Json{{"voices",owner.voices()},{"sample_rate",owner.sample_rate()},{"closed",owner.runtime().closed()},{"model",bool(owner.runtime().backend())}};};compare_json(state(),header.at("startup"),"real stream startup");size_t packets=0;
 for(size_t i=1;i<rows.size();++i){const auto &action=rows[i].at("action");Json actual=Json::object();std::unique_ptr<CttsByteStream> stream;try{if(action.at("op")=="stream"){actual["packets"]=Json::array();stream=owner.generate_stream(action.at("request").dump());while(const auto packet=stream->next()){actual["packets"].push_back({{"size",packet->size()},{"sha256",hash(*packet)}});++packets;if(action.contains("stop")&&action.at("stop")==actual["packets"].size()){stream->close();break;}}}else actual["value"]=owner.generate(action.at("request").dump());}catch(const TtsError &e){actual["error"]={e.category,e.what()};}if(stream)stream->close();actual["state"]=state();compare_json(actual,rows[i].at("result"),"real CTTS stream "+std::to_string(i));}
 expect(streams==4,"three complete chunks and one abandoned chunk");expect(rows.size()-1==header.at("cases"),"all real stream cases");owner.close();std::cout<<rows.size()-1<<" real CTTS stream operations and "<<packets<<" exact write hashes match source\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
