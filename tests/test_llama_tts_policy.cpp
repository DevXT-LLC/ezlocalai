#include "ezlocalai/llama_tts.hpp"
#include "whisper_transcription_test.hpp"
using namespace transcription_test;
int main(int argc,char **argv) { try {
    expect(argc==3,"source and TTS policy fixtures"); const auto cases=records(argv[2]); expect(hash(read(fs::path(argv[1])/"ezlocalai/LlamaTTS.py"))==cases[0].at("source_sha256"),"unchanged TTS wrapper"); size_t launches=0,languages=0;
    for(size_t i=1;i<cases.size();++i) {
        const auto &row=cases[i]; Json actual=Json::object(),events=Json::array();
        try {
            if(row.at("kind")=="language") { ++languages; actual["value"]=llama_tts_language(row.at("language").dump(),row.at("text").dump()); }
            else {
                ++launches; LlamaTtsPlatform p; p.environment=row.at("env").get<std::map<std::string,std::string>>();
                p.is_file=[&](const auto &name) { events.push_back(Json::array({"is_file",name})); return row.at("local").get<bool>() && name.rfind("/opt/",0)!=0; };
                p.absolute=[&](const auto &name) { events.push_back(Json::array({"absolute",name})); return "/absolute/"+name.substr(name.find_first_not_of('/')); };
                p.home=[&] { events.push_back(Json::array({"home"})); return "/fixture/home"; };
                p.executable=[&](const auto &name) { events.push_back(Json::array({"which",name})); return !row.at("missing").get<bool>() && !name.empty(); };
                p.download=[&](const auto &repo,const auto &file,const auto &revision) { events.push_back(Json::array({"download",repo,file,revision})); return "/download/"+file; };
                const auto launch=llama_tts_launch(row.at("device").dump(),row.at("kwargs").dump(),std::move(p)); actual["value"]={{"timeout",launch.timeout},{"device",launch.device},{"arguments",launch.arguments},{"environment",launch.environment}};
            }
        } catch(const TtsError &e) { actual["error"]={{"category",e.category},{"message",e.what()}}; }
        const auto label="TTS policy "+std::to_string(i); compare_json(actual,row.contains("error") ? Json{{"error",row.at("error")}} : Json{{"value",row.at("value")}},label); if(row.contains("events")) compare_json(events,row.at("events"),label+" boundary order");
    }
    expect(launches==cases[0].at("launch_cases") && languages==cases[0].at("language_cases"),"complete TTS policy cases"); std::cout<<launches<<" source constructor and "<<languages<<" language cases match\n"; return 0;
} catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; } }
