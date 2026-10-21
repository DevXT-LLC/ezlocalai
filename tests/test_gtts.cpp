#include "ezlocalai/gtts.hpp"
#include "whisper_transcription_test.hpp"
using namespace transcription_test;
std::string hex(const std::string &value){std::ostringstream out;for(unsigned char c:value)out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(c);return out.str();}
int main(int argc,char **argv){try{
    expect(argc==3,"source and fixture paths");gzFile file=gzopen(argv[2],"rb");expect(file,"open gTTS fixture");std::string raw;char buffer[65536];int size;while((size=gzread(file,buffer,sizeof(buffer)))>0)raw.append(buffer,size);expect(size==0&&gzclose(file)==Z_OK,"complete gTTS fixture");std::vector<Json> cases;std::istringstream input(raw);for(std::string line;std::getline(input,line);)if(!line.empty())cases.push_back(Json::parse(line));
    const auto &header=cases[0];expect(hash(read(fs::path(argv[1])/"app.py"))==header.at("source_sha256"),"unchanged gTTS source helpers");expect(hash(read(fs::path(__FILE__).parent_path().parent_path()/"include/ezlocalai/tts_unicode.hpp"))==header.at("unicode_sha256"),"pinned Python Unicode data");expect(cases.size()-1==header.at("cases"),"all gTTS cases");size_t failures=0;std::map<std::string,size_t> counts;
    for(size_t i=1;i<cases.size();++i){const auto &row=cases[i];const auto kind=row.at("kind").get<std::string>();++counts[kind];Json actual;
        try{if(kind=="language")actual["result"]=gtts_language(row.at("language").dump(),row.at("text"));
            else if(kind=="prepare"){Json result;result["tokens"]=gtts_tokenize(row.at("text"));result["requests"]=Json::array();for(const auto &r:gtts_prepare(row.at("text"),row.at("language").dump()))result["requests"].push_back({{"url",r.url},{"body",r.body},{"headers",r.headers}});actual["result"]=result;}
            else if(kind=="response")actual["result"]=hex(gtts_response_audio({row.at("status"),row.at("reason"),unhex(row.at("hex"))}));
            else throw std::runtime_error("unknown gTTS case");
        }catch(const TtsError &e){actual["error"]={{"category",e.category},{"message",e.what()}};}
        const auto expected=row.contains("error")?Json{{"error",row.at("error")}}:Json{{"result",row.at("result")}};
        try{compare_json(actual,expected,kind+" "+std::to_string(i));}catch(const std::exception &e){if(failures++<12)std::cerr<<e.what()<<" input="<<row.dump()<<'\n';}
    }
    compare_json(counts,header.at("counts"),"gTTS case counts");expect(!failures,std::to_string(failures)+" gTTS cases differ");
    const auto mp3=read(fs::path(argv[2]).parent_path()/"gtts_audio.mp3");expect(hash(mp3)==header.at("mp3_sha256"),"source conversion input");
    for(const auto &row:header.at("conversions")){Json actual;GttsOperations ops;ops.post=[&](const auto &){return GttsResponse{};};const auto native=gtts_native_operations();ops.to_wav=[&](const auto &){std::string bytes;for(int i=0;i<row.at("copies").get<int>();++i)bytes+=mp3;return native.to_wav(bytes);};
        try{const auto wav=gtts_generate_wav("test","\"en\"",ops);actual={{"copies",row.at("copies")},{"size",wav.size()},{"sha256",hash(wav)}};}catch(const TtsError &e){actual={{"copies",row.at("copies")},{"error",{{"category",e.category},{"message",e.what()}}}};}compare_json(actual,row,"real native ffmpeg conversion");
    }
    std::vector<std::string> events;GttsOperations controlled;controlled.post=[&](const auto &r){events.push_back(r.body);return GttsResponse{200,"OK","jQ1olc\",\"[\\\"YQ==\\\"]"};};controlled.to_wav=[&](const auto &bytes){events.push_back("convert:"+bytes);return "wav:"+bytes;};
    const std::string long_text(201,'x');const auto prepared=gtts_prepare(long_text);expect(gtts_generate_wav(long_text,"null",controlled)=="wav:aaa","concatenated chunk audio before conversion");expect(events.size()==4&&events[0]==prepared[0].body&&events[1]==prepared[1].body&&events[2]==prepared[2].body&&events[3]=="convert:aaa","sequential source fallback operations");
    events.clear();try{gtts_generate_wav("!!!","null",controlled);expect(false,"empty text raises");}catch(const TtsError &e){expect(e.category=="AssertionError"&&events.empty(),"prepare all requests before transport");}
    controlled.post=[&](const auto &){events.push_back("post");return GttsResponse{events.size()==2?403:200,"Forbidden","jQ1olc\",\"[\\\"YQ==\\\"]"};};
    try{gtts_generate_wav(long_text,"null",controlled);expect(false,"HTTP failure raises");}catch(const TtsError &e){expect(e.category=="gTTSError"&&events==std::vector<std::string>({"post","post"}),"error stops later requests and conversion");}
    events.clear();controlled.post=[](const auto &){return GttsResponse{200,"OK","jQ1olc\",\"[\\\"YQ==\\\"]"};};auto service=tts_gtts_operations({},controlled);expect(service.fallback_generate(R"({"text":"test","language":"en","voice":"ignored"})")=="wav:a","TTS fallback factory");
    TtsRequestOperations custom;custom.fallback_generate=[](const auto &){return "custom";};expect(tts_gtts_operations(custom,controlled).fallback_generate("{}")=="custom","preserve explicit fallback adapter");
    std::cout<<cases.size()-1<<" source gTTS cases, "<<header.at("conversions").size()<<" real ffmpeg comparisons, and fallback orchestration checks match\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
