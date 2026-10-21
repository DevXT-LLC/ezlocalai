#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "tts_ws_client.hpp"
#include "fixture_ports.hpp"
using namespace transcription_test;
int main(int argc,char **argv){try{
    expect(argc==3,"source and WebSocket transport capture");const auto cases=records(argv[2]);expect(hash(read(fs::path(argv[1])/"ezlocalai/TTSWebSocket.py"))==cases[0].at("source_sha256")&&hash(read(fs::path(argv[1])/"app.py"))==cases[0].at("app_sha256"),"unchanged source endpoint and session");expect(cases.size()-1==cases[0].at("cases"),"all transport cases");
    Temporary temporary;struct Restore{fs::path path=fs::current_path();~Restore(){fs::current_path(path);}}restore;fs::current_path(temporary.path);
    for(const auto &entry:environment_defaults)setenv(std::string(entry.first).c_str(),std::string(entry.second).c_str(),1);setenv("EZLOCALAI_API_KEY","",1);setenv("TTS_ENABLED","true",1);setenv("DEFAULT_MODEL","fixture-text",1);
    ezl_env_config config{};ezl_load_config(&config);NativeServer server(config,std::make_unique<EchoInferenceBackend>("fixture-text"));const int port=fixture_ports();std::string error;expect(server.start("127.0.0.1",port,&error),error);
    for(size_t i=1;i<cases.size();++i){const auto &row=cases[i];tts_ws_test::Client client(port);timeval timeout{3,0};setsockopt(client.fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));client.send(unhex(row.at("wire")));Json frames=Json::array();for(;;){const auto frame=client.frame();std::string hex;for(unsigned char c:frame.second){hex+="0123456789abcdef"[c>>4];hex+="0123456789abcdef"[c&15];}frames.push_back({{"opcode",frame.first},{"hex",hex}});if(frame.first==8)break;}compare_json(frames,row.at("frames"),row.at("name").get<std::string>()+" exact TCP frames");}
    server.stop();std::cout<<cases.size()-1<<" source Uvicorn/WebSocket transport captures match over TCP\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
