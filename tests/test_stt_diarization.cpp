#include "ezlocalai/stt_diarization.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cmath>
#include <thread>
using Json=nlohmann::json;
static void check(bool value,const std::string &message){if(!value)throw std::runtime_error(message);}
int main(int argc,char **argv){try{
    check(argc==3,"fixture and assets required");std::ifstream input(argv[1]);const auto fixture=Json::parse(input);const std::string assets=argv[2];double maximum_error=0;
    for (const auto &row : fixture.at("assets")) check(ezlocalai::stt_diarization_audio(assets+"/"+row.at("file").get<std::string>()).size()==row.at("frames_16k").get<size_t>(),"resampled frame count");
    for(const auto &row:fixture.at("features")){
        const auto audio=ezlocalai::stt_diarization_audio(assets+"/"+row.at("file").get<std::string>());
        const size_t start=row.at("start").get<double>()*16000,end=row.at("end").get<double>()*16000;
        const auto feature=ezlocalai::stt_diarization_feature({audio.begin()+start,audio.begin()+end});
        for(size_t i=0;i<feature.size();++i){const auto error=std::abs(feature[i]-row.at("feature")[i].get<double>());maximum_error=std::max(maximum_error,error);check(error<.003,"MFCC "+row.at("file").get<std::string>()+" index="+std::to_string(i)+" error="+std::to_string(error));}
    }
    std::cout<<fixture.at("features").size()<<" source feature vectors passed; maximum absolute error="<<maximum_error<<'\n';
    size_t index=0;
    for(const auto &row:fixture.at("clusters")){
        std::optional<int> speakers;if(!row.at("speakers").is_null())speakers=row.at("speakers").get<int>();
        const auto labels=ezlocalai::stt_diarization_labels(row.at("features").get<std::vector<ezlocalai::SttSpeakerFeature>>(),speakers);
        check(Json(labels)==row.at("labels"),"cluster case "+std::to_string(index)+": "+Json(labels).dump()+" expected "+row.at("labels").dump());++index;
    }
    double time=100;unsigned counter=0;ezlocalai::SttDiarizer diarizer([&]{return time;},[&]{std::ostringstream out;out<<std::hex<<std::setw(12)<<std::setfill('0')<<++counter;return out.str();});
    for(const auto &row:fixture.at("diarization")){
        time=row.at("time");const auto result=Json::parse(diarizer.diarize(assets+"/"+row.at("file").get<std::string>(),row.at("segments").dump(),row.at("speakers").dump(),row.at("session").dump()));
        check(result==row.at("result"),"diarization "+row.at("name").get<std::string>()+": "+result.dump()+" expected "+row.at("result").dump());
    }
    std::vector<std::thread> threads;std::vector<std::string> concurrent(16);
    for (size_t i=0;i<concurrent.size();++i) threads.emplace_back([&,i]{concurrent[i]=diarizer.diarize(assets+"/signal-16000-1.wav",R"([{"start":0,"end":0.5,"text":"parallel"}])","null",R"("parallel-session")");});
    for (auto &thread:threads) thread.join();
    for (const auto &value:concurrent) check(value==concurrent.front(),"concurrent session IDs are consistent");
    std::cout<<index<<" source cluster cases and "<<fixture.at("diarization").size()<<" source diarization/session cases passed\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
