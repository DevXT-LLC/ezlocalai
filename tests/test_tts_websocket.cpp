#include "ezlocalai/tts_websocket.hpp"
#include "whisper_transcription_test.hpp"
using namespace transcription_test;
std::string hex(const std::string &value){std::string out;for(unsigned char c:value){out+="0123456789abcdef"[c>>4];out+="0123456789abcdef"[c&15];}return out;}
int main(int argc,char **argv){try{
    expect(argc==3,"source and WebSocket fixture paths");gzFile file=gzopen(argv[2],"rb");expect(file,"open WebSocket fixture");std::string raw;char buffer[65536];int count;while((count=gzread(file,buffer,sizeof(buffer)))>0)raw.append(buffer,count);expect(count==0&&gzclose(file)==Z_OK,"complete WebSocket fixture");std::vector<Json> cases;std::istringstream lines(raw);for(std::string line;std::getline(lines,line);)if(!line.empty())cases.push_back(Json::parse(line));expect(hash(read(fs::path(argv[1])/"ezlocalai/TTSWebSocket.py"))==cases[0].at("source_sha256"),"unchanged WebSocket source");expect(cases.size()-1==cases[0].at("cases"),"complete WebSocket fixture");
    size_t pcm=0,input=0;
    for(size_t index=1;index<cases.size();++index){
        const auto &row=cases[index];Json error=nullptr,events=Json::array();size_t consumed=0;
        try{
            if(row.at("kind")=="pcm"){
                ++pcm;TtsPcmSegment segment;
                for(const auto &packet:row.at("packets"))segment.feed(unhex(packet),[&](bool header,const auto &bytes){events.push_back(Json::array({header?"header":"audio",hex(bytes)}));});
                segment.finish();
            }else{
                ++input;TtsWebSocketInput parser;
                for(const auto &message:row.at("messages")){
                    ++consumed;const auto result=parser.receive(message);
                    if(result.request){const auto request=Json::parse(*result.request);events.push_back(Json::array({request.at("text"),request.at("voice"),request.at("language")}));}
                    if(result.done){events.push_back(nullptr);break;}
                }
            }
        }catch(const TtsError &e){error={{"category",e.category},{"message",e.what()}};}
        const auto label="WebSocket "+std::to_string(index);
        compare_json(error,row.at("error"),label+" exception");
        compare_json(events,row.at(row.at("kind")=="pcm"?"events":"requests"),label+" emitted events");
        if(row.at("kind")=="input")compare_json(consumed,row.at("consumed"),label+" consumed messages");
    }
    std::cout<<pcm<<" source PCM cases and "<<input<<" receiver scenarios match\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
