#include "ezlocalai/whisper_features.hpp"
#include "whisper_test_data.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
using namespace whisper_test;
int main(int argc,char **argv){try{
    expect(argc==2,"feature fixture argument");const fs::path fixture(argv[1]);const auto assets=fixture.parent_path()/"whisper_feature_assets";
    size_t features=0,legacy=0,decoded=0;float maximum=0;
    for(const auto &row:rows(fixture)){
        if(row[0]=="F"||row[0]=="L"){
            expect(row.size()==6,"feature columns");row[0]=="F"?++features:++legacy;const auto size=std::stoul(row[1]),bands=std::stoul(row[2]),frames=std::stoul(row[3]);
            const auto audio=packed(assets/row[4],size),expected=packed(assets/row[5],bands*frames);
            const auto actual=row[0]=="F"?ezlocalai::whisper_log_mel(audio,bands):ezlocalai::whisper_log_mel_legacy(audio,bands);expect(actual.bands==bands&&actual.frames==frames&&actual.values.size()==expected.size(),"log-Mel shape");
            float difference=0;for(size_t i=0;i<expected.size();++i){expect(std::isfinite(actual.values[i]),"finite log-Mel");difference=std::max(difference,std::abs(actual.values[i]-expected[i]));}
            maximum=std::max(maximum,difference);expect(difference<=0.0000002f,"log-Mel "+row[4]+" bands "+row[2]+" maximum error "+std::to_string(difference));
            const auto start=frames/2,count=frames-start,padded=std::max<size_t>(1,count+7);const auto window=ezlocalai::whisper_feature_window(actual,start,count,padded);
            for(size_t band=0;band<bands;++band)for(size_t i=0;i<padded;++i)expect(window.values[band*padded+i]==(i<count?actual.values[band*frames+start+i]:0),"frame-major padding/window layout");
        }else{
            expect(row[0]=="D"&&row.size()==4,"decode columns");++decoded;
            const auto actual=ezlocalai::whisper_decode_audio((assets/row[1]).string()),expected=packed(assets/row[3],std::stoul(row[2]));
            expect(actual==expected,"PyAV decoded PCM sample equality "+row[1]+" actual "+std::to_string(actual.size())+" expected "+std::to_string(expected.size()));
        }
    }
    expect(features==16&&legacy==16&&decoded==5,"complete Whisper feature fixtures");
    bool failed=false;try{ezlocalai::whisper_decode_audio((assets/"missing.wav").string());}catch(const std::exception &){failed=true;}expect(failed,"decode errors propagate");
    std::cout<<features<<" modern and "<<legacy<<" legacy log-Mel cases, "<<decoded<<" exact PCM decode cases; maximum feature error "<<maximum<<'\n';return 0;
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
