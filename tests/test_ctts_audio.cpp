#include "ezlocalai/ctts_audio.hpp"
#include "whisper_transcription_test.hpp"
#include <zlib.h>
#ifdef EZLOCALAI_CTTS_TORCH
#include "ezlocalai/ctts_torch.hpp"
#include <ATen/Parallel.h>
#endif
using namespace transcription_test;
std::string audio_unhex(const std::string &input) { expect(input.size()%2==0,"even hex length"); std::string out; out.reserve(input.size()/2); for(size_t i=0;i<input.size();i+=2) out+=static_cast<char>(std::stoi(input.substr(i,2),nullptr,16)); return out; }
CttsAudioArray array(const Json &input) { CttsAudioArray out; out.shape=input.at("shape").get<std::vector<size_t>>(); if(input.contains("strides")) out.strides=input.at("strides").get<std::vector<ptrdiff_t>>(); out.offset=input.value("offset",ptrdiff_t{0}); const auto bytes=audio_unhex(input.at("data")); expect(bytes.size()%4==0,"float32 storage"); out.samples.resize(bytes.size()/4); for(size_t i=0;i<out.samples.size();++i) { uint32_t word=0; for(int j=0;j<4;++j) word|=uint32_t(static_cast<unsigned char>(bytes[4*i+j]))<<(j*8); std::memcpy(&out.samples[i],&word,4); } return out; }
void compare_array(const CttsAudioArray &actual,const Json &expected,const std::string &where) { expect(actual.shape==expected.at("shape").get<std::vector<size_t>>(),where+" shape"); const auto want=array(expected); expect(actual.samples.size()==want.samples.size(),where+" samples"); for(size_t i=0;i<want.samples.size();++i) { uint32_t a,b; std::memcpy(&a,&actual.samples[i],4); std::memcpy(&b,&want.samples[i],4); if(a!=b && !(std::isnan(actual.samples[i]) && std::isnan(want.samples[i]))) throw std::runtime_error(where+" sample "+std::to_string(i)+" bits "+std::to_string(a)+" != "+std::to_string(b)); } }
int main(int argc,char **argv) { try {
    expect(argc==4,"source, fixture and optional libsndfile arguments"); gzFile file=gzopen(argv[2],"rb"); expect(file!=nullptr,"open compressed fixture"); std::string content; char buffer[65536]; int n; while((n=gzread(file,buffer,sizeof(buffer)))>0) content.append(buffer,n); const auto closed=gzclose(file); expect(n==0 && closed==Z_OK,"read complete gzip"); std::istringstream lines(content); std::string line; std::getline(lines,line); const auto header=Json::parse(line); expect(hash(read(fs::path(argv[1])/"ezlocalai/CTTS.py"))==header.at("source_sha256"),"pinned CTTS source"); if(*argv[3]) expect(hash(read(argv[3]))==header.at("libsndfile_sha256"),"pinned libsndfile codec build"); CttsResampler resample;
#ifdef EZLOCALAI_CTTS_TORCH
    at::set_num_threads(1); resample=ctts_torch_resample;
#endif
    size_t cases=0,skipped=0; std::map<std::string,size_t> methods;
    while(std::getline(lines,line)) { if(line.empty()) continue; const auto row=Json::parse(line); const auto method=row.at("method").get<std::string>(); const auto where="case "+std::to_string(++cases)+" "+method+" "+row.value("view",""); ++methods[method]; expect(!row.contains("error"),where+" typed input source success"); const auto sample_rate=row.at("rate").get<int64_t>();
        if(method=="decode") {
            if(!resample && row.at("source_rate")!=row.at("rate")) { ++skipped; continue; }
            const auto actual=ctts_bytes_to_tensor(audio_unhex(row.at("input")),sample_rate,resample); expect(bool(actual)!=row.at("value").is_null(),where+" optional result"); if(actual) compare_array(*actual,row.at("value"),where);
        } else { const auto input=array(row.at("input")); if(method=="mono") compare_array(ctts_mono_float32(input),row.at("value"),where);
            else if(method=="tensor") { const auto actual=ctts_tensor_to_bytes(input,sample_rate); expect(bool(actual)!=row.at("value").is_null(),where+" optional bytes"); if(actual) expect(*actual==audio_unhex(row.at("value")),where+" complete WAV bytes"); }
            else { const auto actual=method=="wav"?ctts_wav_bytes(input,sample_rate):ctts_pcm16_bytes(input); expect(actual==audio_unhex(row.at("value")),where+" complete "+method+" bytes"); }
        }
    }
    expect(cases==header.at("cases"),"all fixture rows"); compare_json(Json(methods),header.at("methods"),"method counts"); std::cout<<cases-skipped<<" unchanged CTTS audio conversion cases match, "<<skipped<<" resampling cases unavailable without LibTorch\n"; return 0;
} catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; } }
