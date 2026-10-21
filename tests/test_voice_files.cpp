#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "whisper_transcription_test.hpp"
using namespace transcription_test;
std::string hex(const std::string &value){std::string result;for(unsigned char c:value){result+="0123456789abcdef"[c>>4];result+="0123456789abcdef"[c&15];}return result;}
Json snapshot(){
    std::map<std::string,Json> entries;
    for(const auto &entry:fs::recursive_directory_iterator(".")) {
        const auto path=entry.path();const auto name=path.lexically_relative(".").string();
        Json value={{"path",name},{"kind",entry.is_symlink()?"symlink":entry.is_directory()?"dir":"file"}};
        if(entry.is_symlink())value["target"]=fs::read_symlink(path).string();
        else if(entry.is_regular_file())value["hex"]=hex(read(path));
        entries[name]=value;
    }
    Json result=Json::array();for(const auto &[name,value]:entries)result.push_back(value);return result;
}
int main(int argc,char **argv){try{
    expect(argc==3,"source and voice filesystem fixture");const auto cases=records(argv[2]);expect(hash(read(fs::path(argv[1])/"app.py"))==cases[0].at("source_sha256"),"unchanged voice endpoints");expect(cases.size()-1==cases[0].at("cases"),"complete voice fixture");
    size_t compared=0;
    for(size_t i=1;i<cases.size();++i){const auto &row=cases[i];Temporary temporary;struct Restore{fs::path path=fs::current_path();~Restore(){fs::current_path(path);}}restore;fs::current_path(temporary.path);
        for(const auto &entry:environment_defaults)setenv(std::string(entry.first).c_str(),std::string(entry.second).c_str(),1);
        for(const auto &[key,value]:std::map<std::string,std::string>{{"EZLOCALAI_API_KEY",""},{"DEFAULT_MODEL","fixture-text"},{"VOICE_SERVER",""},{"LAZY_LOAD_VOICE","true"},{"ROUTER_URL",""},{"FALLBACK_SERVER",""},{"TTS_ENABLED","true"}})setenv(key.c_str(),value.c_str(),1);
        for(const auto &[key,value]:row.at("environment").items())setenv(key.c_str(),value.get<std::string>().c_str(),1);
        ezl_env_config config{};ezl_load_config(&config);NativeServer server(config,std::make_unique<EchoInferenceBackend>("fixture-text"));
        for(const auto &entry:row.at("entries")){
            const fs::path path=entry.at("path").get<std::string>();if(!path.parent_path().empty())fs::create_directories(path.parent_path());
            if(entry.at("kind")=="dir")fs::create_directories(path);
            else if(entry.at("kind")=="symlink")fs::create_symlink(entry.at("target").get<std::string>(),path);
            else {std::ofstream out(path,std::ios::binary);const auto bytes=unhex(entry.value("hex",std::string()));out.write(bytes.data(),bytes.size());}
        }
        for(size_t j=0;j<row.at("actions").size();++j){const auto &action=row.at("actions")[j],&expected=row.at("results")[j];const auto label="voice filesystem "+std::to_string(i)+" action "+std::to_string(j);
            const auto result=server.handle_request(action.at("method"),"/v1/audio/voices","",unhex(action.value("hex",std::string())),action.value("content_type","multipart/form-data; boundary=voice-fixture"));
            compare_json(result.status,expected.at("status"),label+" status");compare_json(result.content_type,expected.at("content_type"),label+" type");
            if(result.content_type=="application/json")compare_json(Json::parse(result.body),Json::parse(unhex(expected.at("hex"))),label+" response");else compare_json(result.body,unhex(expected.at("hex")),label+" response");
            compare_json(snapshot(),expected.at("files"),label+" filesystem");++compared;
        }
    }
    std::cout<<cases.size()-1<<" source voice filesystem scenarios / "<<compared<<" endpoint calls match\n";return 0;
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
