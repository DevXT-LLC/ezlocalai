#include "tts_service_test.hpp"
#include <openssl/evp.h>
#ifdef EZLOCALAI_CTTS_TORCH
#include "ezlocalai/ctts_torch.hpp"
#endif
std::string file_hash(const fs::path &path){
    std::ifstream input(path,std::ios::binary);expect(bool(input),"read "+path.string());auto *ctx=EVP_MD_CTX_new();expect(ctx&&EVP_DigestInit_ex(ctx,EVP_sha256(),nullptr)==1,"digest init");char buffer[65536];while(input){input.read(buffer,sizeof(buffer));expect(EVP_DigestUpdate(ctx,buffer,input.gcount())==1,"digest update");}expect(input.eof(),"complete read");unsigned char digest[32];unsigned size=0;expect(EVP_DigestFinal_ex(ctx,digest,&size)==1&&size==32,"digest finish");EVP_MD_CTX_free(ctx);return hex(std::string(reinterpret_cast<char *>(digest),32));
}
int main(int argc,char **argv){try{
    expect(argc==6,"source, fixture, worker, models and speaker");const auto cases=records(argv[2]);const auto &header=cases[0];
    for(const auto &[path,key]:std::vector<std::pair<std::string,std::string>>{{"app.py","source_sha256"},{"Pipes.py","pool_sha256"},{"ezlocalai/CTTS.py","ctts_sha256"},{"ezlocalai/LlamaTTS.py","llama_sha256"},{"ezlocalai/AudioStreaming.py","bridge_sha256"},{"ezlocalai/AudioCache.py","cache_sha256"},{"tts/worker.cpp","worker_sha256"}})expect(file_hash(fs::path(argv[1])/path)==header.at(key),"pinned "+path);
    expect(file_hash(argv[5])==header.at("speaker_sha256"),"pinned speaker");for(const auto &entry:header.at("model_hashes").items())expect(file_hash(fs::path(argv[4])/entry.key())==entry.value(),"pinned model");
    for(const auto &entry:header.at("environment").items())setenv(entry.key().c_str(),entry.value().get<std::string>().c_str(),1);setenv("QWEN_TTS_BIN",argv[3],1);setenv("QWEN_TTS_MODEL_FILE",(fs::path(argv[4])/"qwen-talker-0.6b-base-Q8_0.gguf").c_str(),1);setenv("QWEN_TTS_MMPROJ_FILE",(fs::path(argv[4])/"qwen-tokenizer-12hz-f16.gguf").c_str(),1);
    Temporary temporary;const auto original=fs::current_path();fs::current_path(temporary.path);struct Restore{fs::path path;~Restore(){fs::current_path(path);}}restore{original};fs::create_directories("voices");fs::copy_file(argv[5],"voices/default.wav");
    Environment env(cases[1].at("config"));CttsServiceOperations operations;
#ifdef EZLOCALAI_CTTS_TORCH
    operations.runtime=ctts_torch_operations();
#endif
    env.native=tts_pool_ctts_operations(std::move(operations));const auto actions=run_service_scenario(env,cases[1],"actual TTS service");expect(cases.size()==2,"complete model fixture");
    std::cout<<actions<<" real-model TTS service operations match output hashes, pool/slot state and event order\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
