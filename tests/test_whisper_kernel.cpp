#include "ezlocalai/whisper_kernel.h"
#include "ezlocalai/whisper_features.hpp"
#include "whisper_test_data.hpp"
#include <cmath>
#include <iostream>
using namespace whisper_test;
struct Result {std::vector<uint64_t> tokens;float score=0,no_speech=0;size_t calls=0;};
void result(const uint64_t *tokens,size_t count,float score,float no_speech,void *user){auto &out=*static_cast<Result *>(user);out.tokens.assign(tokens,tokens+count);out.score=score;out.no_speech=no_speech;++out.calls;}
int main(int argc,char **argv){try{
    expect(argc==3,"model and kernel fixture directory");const fs::path directory(argv[2]);char error[4096]{};
    const auto cases=rows(directory/"cases.tsv");expect(!cases.empty()&&cases.front().size()==4&&cases.front()[0]=="H","kernel fixture metadata");
    expect(hash(read(fs::path(argv[1])/"model.bin"))==cases.front()[2],"pinned CTranslate2 model bytes");
    const auto audio=packed(directory/"audio.f32z",std::stoul(cases.front()[1]));
    struct Handle {void *value;~Handle(){ezl_whisper_kernel_destroy(value);}} model{ezl_whisper_kernel_create(argv[1],"cpu","int8",4,error,sizeof(error))};
    expect(model.value,std::string("create native CTranslate2 Whisper: ")+error);
    const auto source=packed(directory/"features.f32z",80*3000);
    const auto generated=ezlocalai::whisper_log_mel(audio);
    const auto native=ezlocalai::whisper_feature_window(generated,0,generated.frames);
    const auto legacy_generated=ezlocalai::whisper_log_mel_legacy(audio);
    const auto legacy=ezlocalai::whisper_feature_window(legacy_generated,0,legacy_generated.frames);
    const auto legacy_source=packed(directory/"legacy.f32z",80*3000);
    size_t count=0;
    for(const auto &row:cases){
        if(row[0]=="H")continue;
        expect(row[0]=="K"&&row.size()==9,"kernel fixture columns");++count;ezl_whisper_options options;options.beam_size=std::stoul(row[1]);
        const auto prompt=integers<uint64_t>(row[2]),expected=integers<uint64_t>(row[4]);const auto suppressed=integers<int>(row[3]);options.suppress_tokens=suppressed.data();options.suppress_count=suppressed.size();
        for(bool own_features:{false,true}){
            const auto &features=row[8]=="legacy"?(own_features?legacy.values:legacy_source):(own_features?native.values:source);Result actual;
            expect(!ezl_whisper_kernel_generate(model.value,features.data(),80,3000,prompt.data(),prompt.size(),&options,result,&actual,error,sizeof(error)),std::string("generate ")+error);
            const auto label=std::string(own_features?"native":"source")+" features, case "+std::to_string(count);
            if(actual.calls!=1||actual.tokens!=expected){
                std::ofstream dump(fs::temp_directory_path()/("ezlocalai-whisper-"+row[8]+"-actual.f32"),std::ios::binary);dump.write(reinterpret_cast<const char *>(features.data()),features.size()*sizeof(float));
                std::cerr<<label<<" actual tokens:";for(auto token:actual.tokens)std::cerr<<' '<<token;std::cerr<<"\nexpected:";for(auto token:expected)std::cerr<<' '<<token;std::cerr<<'\n';
                throw std::runtime_error(label+" exact decoder token IDs");
            }
            const float tolerance=own_features?0.0001f:0.000001f;
            expect(std::abs(actual.score-std::stof(row[5]))<=tolerance&&std::abs(actual.no_speech-std::stof(row[6]))<=tolerance,label+" decoder score/no-speech probability "+std::to_string(actual.score)+" expected "+row[5]);
        }
    }
    expect(count==8,"complete real Whisper kernel cases");Result invalid;const uint64_t prompt=50257;ezl_whisper_options options;
    expect(ezl_whisper_kernel_generate(model.value,source.data(),128,3000,&prompt,1,&options,result,&invalid,error,sizeof(error))!=0&&invalid.calls==0,"invalid mel shape reports error without callback");
    std::cout<<count<<" real CPU int8 CTranslate2 cases match source tokens with source and native log-Mel features\n";return 0;
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
