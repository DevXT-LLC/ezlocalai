#include "ezlocalai/llama_tts.hpp"
#include "ezlocalai/ctts_stream_bridge.hpp"
#include "whisper_transcription_test.hpp"
#include <openssl/evp.h>
using namespace transcription_test;
std::string file_hash(const fs::path &path) {
    std::ifstream input(path,std::ios::binary); expect(bool(input),"read "+path.string());
    const std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(),EVP_MD_CTX_free); expect(context && EVP_DigestInit_ex(context.get(),EVP_sha256(),nullptr)==1,"initialize model digest"); char buffer[65536]; while(input) { input.read(buffer,sizeof(buffer)); expect(EVP_DigestUpdate(context.get(),buffer,input.gcount())==1,"update model digest"); } expect(input.eof(),"read complete model "+path.string()); unsigned char digest[32]; unsigned size=0; expect(EVP_DigestFinal_ex(context.get(),digest,&size)==1 && size==32,"finish model digest"); std::ostringstream out; for(auto byte:digest) out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(byte); return out.str();
}
int main(int argc,char **argv) { try {
    expect(argc==6,"source, fixtures, worker, model directory, speaker"); const auto cases=records(argv[2]); const auto &header=cases[0]; expect(hash(read(fs::path(argv[1])/"ezlocalai/LlamaTTS.py"))==header.at("source_sha256"),"pinned wrapper"); expect(hash(read(fs::path(argv[1])/"tts/worker.cpp"))==header.at("worker_sha256"),"pinned source worker"); expect(hash(read(fs::path(__FILE__).parent_path().parent_path()/"tts/worker.cpp"))==header.value("native_worker_sha256",header.at("worker_sha256")),"unchanged native worker"); expect(hash(read(argv[5]))==header.at("speaker_sha256"),"pinned reference speaker");
    for(const auto &item:header.at("model_hashes").items()) expect(file_hash(fs::path(argv[4])/item.key())==item.value(),"pinned model "+item.key());
    auto platform=native_llama_tts_platform(); auto &env=platform.environment; env["QWEN_TTS_BIN"]=argv[3]; env["QWEN_TTS_MODEL_FILE"]=(fs::path(argv[4])/"qwen-talker-0.6b-base-Q8_0.gguf").string(); env["QWEN_TTS_MMPROJ_FILE"]=(fs::path(argv[4])/"qwen-tokenizer-12hz-f16.gguf").string(); env["QWEN_TTS_THREADS"]=header.at("threads").dump(); env["QWEN_TTS_CONTEXT_SIZE"]="4096"; env["QWEN_TTS_TIMEOUT"]="300";
    LlamaTts model(llama_tts_launch(header.value("device",Json("cpu")).dump(),Json{{"seed",header.at("seed")}}.dump(),std::move(platform))); size_t chunks=0;
    for(size_t i=1;i<cases.size();++i) { const auto &row=cases[i]; auto request=row.at("request"); request["ref_audio"]=argv[5]; Json actual=Json::array(); const auto append=[&](const TtsAudio &audio) { std::string bytes; for(float sample:audio.samples) { uint32_t word; std::memcpy(&word,&sample,4); for(int shift=0;shift<32;shift+=8) bytes.push_back(static_cast<char>(word>>shift)); } actual.push_back({{"hash",hash(bytes)},{"samples",audio.samples.size()},{"channels",audio.channels},{"rate",Json::parse(audio.rate)}}); ++chunks; return true; };
        if(row.at("stream").get<bool>()) { auto backend=std::make_shared<CttsBackend>(); backend->stream=[&model](const auto &request) {return model.stream_iterator(request);}; backend->close=[&model] {model.close();}; CttsStreamBridge stream(backend,request.dump()); while(const auto audio=stream.next()) append(*audio); } else append(model.generate(request.dump())); compare_json(actual,row.at("chunks"),"real Qwen TTS "+std::to_string(i)); expect(model.alive(),"persistent worker after completed request");
    }
    model.close(); model.close(); expect(!model.alive(),"real worker close"); expect(cases.size()-1==header.at("cases"),"all model cases"); std::cout<<cases.size()-1<<" real Qwen TTS requests / "<<chunks<<" float32 audio buffers match source SHA256 exactly\n"; return 0;
} catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; } }
