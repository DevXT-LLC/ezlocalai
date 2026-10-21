#include "ezlocalai/ctts_runtime.hpp"
#include "ezlocalai/ctts_audio.hpp"
#include "whisper_transcription_test.hpp"
#include <openssl/evp.h>
#ifdef EZLOCALAI_CTTS_TORCH
#include "ezlocalai/ctts_torch.hpp"
#include <c10/core/InferenceMode.h>
#endif
using namespace transcription_test;
std::string file_hash(const fs::path &path) { std::ifstream input(path,std::ios::binary); expect(bool(input),"read "+path.string()); const std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(),EVP_MD_CTX_free); expect(ctx && EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr)==1,"model digest"); char buffer[65536]; while(input) { input.read(buffer,sizeof(buffer)); expect(EVP_DigestUpdate(ctx.get(),buffer,input.gcount())==1,"model digest update"); } expect(input.eof(),"model read complete"); unsigned char digest[32]; unsigned size=0; expect(EVP_DigestFinal_ex(ctx.get(),digest,&size)==1 && size==32,"model digest finish"); std::ostringstream out; for(auto c:digest) out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(c); return out.str(); }
int main(int argc,char **argv) { try {
    expect(argc==6,"source, fixture, worker, model directory, speaker"); const auto rows=records(argv[2]); const auto &header=rows[0]; expect(hash(read(fs::path(argv[1])/"ezlocalai/CTTS.py"))==header.at("source_sha256"),"pinned CTTS source"); expect(hash(read(fs::path(argv[1])/"ezlocalai/LlamaTTS.py"))==header.at("llama_sha256"),"pinned model wrapper"); expect(hash(read(fs::path(argv[1])/"tts/worker.cpp"))==header.at("worker_sha256"),"pinned native worker"); expect(hash(read(argv[5]))==header.at("speaker_sha256"),"pinned speaker"); for(const auto &item:header.at("model_hashes").items()) expect(file_hash(fs::path(argv[4])/item.key())==item.value(),"pinned GGUF");
    for(const auto &entry:std::map<std::string,std::string>{{"QWEN_TTS_BIN",argv[3]},{"QWEN_TTS_MODEL_FILE",(fs::path(argv[4])/"qwen-talker-0.6b-base-Q8_0.gguf").string()},{"QWEN_TTS_MMPROJ_FILE",(fs::path(argv[4])/"qwen-tokenizer-12hz-f16.gguf").string()},{"QWEN_TTS_THREADS","4"},{"QWEN_TTS_CONTEXT_SIZE","4096"},{"QWEN_TTS_TIMEOUT","300"},{"QWEN_TTS_GENERATE_KWARGS","{\"seed\":23,\"max_new_tokens\":8}"},{"QWEN_TTS_WARMUP_MAX_NEW_TOKENS","4"}}) setenv(entry.first.c_str(),entry.second.c_str(),1);
    CttsOperations ops;
#ifdef EZLOCALAI_CTTS_TORCH
    ops=ctts_torch_operations(); expect(!c10::InferenceMode::is_enabled(),"initial inference mode");
    if(ops.cuda_available()) { const auto memory=ops.memory_info("0"); expect(memory.first>=0 && memory.first<=memory.second && memory.second>0,"real CUDA free memory"); expect(ops.total_bytes("0")>=memory.first,"device total memory"); expect(ops.allocated_bytes("0")>=0,"allocator memory"); }
    const auto load=ctts_llama_operations().load;
    ops.load=[load](const std::string &device,const std::string &kwargs) { auto backend=load(device,kwargs); const auto generate=backend->generate; backend->generate=[generate](const std::string &request) { expect(c10::InferenceMode::is_enabled(),"Torch inference mode during synthesis"); return generate(request); }; return backend; };
#endif
    CttsRuntime model("\"cpu\"",std::move(ops)); expect(model.device()=="cpu" && !model.closed(),"explicit CPU model"); compare_json(Json::parse(model.warmup_generation_kwargs()),rows[1].at("request").at("generation_kwargs"),"actual warmup kwargs");
    for(size_t i=1;i<rows.size();++i) { auto request=rows[i].at("request"); request["audio_path"]=argv[5]; Json actual;
        try { const auto audio=model.generate_sample(request.dump()); std::string bytes; for(float sample:audio.samples) { uint32_t word; std::memcpy(&word,&sample,4); for(int shift=0;shift<32;shift+=8) bytes+=static_cast<char>(word>>shift); } CttsAudioArray conversion{audio.samples,audio.channels==1?std::vector<size_t>{audio.samples.size()}:std::vector<size_t>{audio.samples.size()/size_t(audio.channels),size_t(audio.channels)}}; actual["value"]={{"wav_hash",hash(ctts_wav_bytes(conversion,std::stoll(audio.rate)))},{"pcm_hash",hash(ctts_pcm16_bytes(conversion))},{"hash",hash(bytes)},{"samples",audio.samples.size()},{"channels",audio.channels},{"rate",audio.rate}}; }
        catch(const TtsError &e) { actual["error"]={{"category",e.category},{"message",e.what()}}; }
        compare_json(actual,rows[i].contains("value")?Json{{"value",rows[i].at("value")}}:Json{{"error",rows[i].at("error")}},"real CTTS request "+std::to_string(i)); expect(model.backend()!=nullptr && !model.closed(),"model retained after generation/error");
#ifdef EZLOCALAI_CTTS_TORCH
        expect(!c10::InferenceMode::is_enabled(),"inference mode restored after synthesis/error");
#endif
    }
    model.close(); expect(model.closed() && !model.backend(),"model detached on close"); model.close(); expect(rows.size()-1==header.at("cases"),"all real CTTS cases"); std::cout<<rows.size()-1<<" real CTTS sample/error requests match source, including exact float32 SHA256 and recovery\n"; return 0;
} catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; } }
