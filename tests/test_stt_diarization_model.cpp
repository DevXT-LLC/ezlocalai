#include "ezlocalai/stt_whisper.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "whisper_transcription_test.hpp"
#include <openssl/evp.h>
#include <regex>
using namespace transcription_test;
static void normalize(Json &value,std::map<std::string,std::string> &ids){
    if(value.is_string()){
        const auto text=value.get<std::string>();const std::regex pattern("SPK_[0-9a-f]{12}");std::string out;size_t at=0;
        for(std::sregex_iterator i(text.begin(),text.end(),pattern),end;i!=end;++i){const auto id=i->str();auto found=ids.find(id);if(found==ids.end())found=ids.emplace(id,"SPK_UUID_"+std::to_string(ids.size()+1)).first;out+=text.substr(at,i->position()-at)+found->second;at=i->position()+i->length();}
        value=out+text.substr(at);
    }else if(value.is_structured())for(auto &item:value)normalize(item,ids);
}
int main(int argc,char **argv){try{
    expect(argc==6,"reference fixture model vad assets");const auto fixture=Json::parse(read(argv[2]));
    expect(hash(read(fs::path(argv[1])/"ezlocalai/STT.py"))==fixture.at("source_sha256"),"unchanged source identity");
    for(const auto &entry:fixture.at("model_hashes").items())expect(hash(read(fs::path(argv[3])/entry.key()))==entry.value(),"pinned Whisper model "+entry.key());
    Temporary temporary;fs::create_directories(temporary.path/"outputs");
    SttWhisperConfiguration configuration;configuration.vad_directory=argv[4];SttRuntimeOperations platform;
    platform.cuda_available=[]{return false;};platform.cpu_count=[]{return 16;};platform.working_directory=[&]{return temporary.path.string();};
    auto create=[&]{return std::make_unique<SttRuntime>(argv[3],stt_whisper_operations(configuration,platform));};
    auto runtime=create();std::map<std::string,std::string> reference_ids,native_ids;size_t count=0;
    for(const auto &row:fixture.at("cases")){
        if(row.at("reload").get<bool>()){runtime->close();runtime=create();}
        const auto bytes=read(fs::path(argv[5])/row.at("file").get<std::string>());expect(hash(bytes)==row.at("audio_sha256"),"audio identity");
        std::string encoded(4*((bytes.size()+2)/3),'\0');EVP_EncodeBlock(reinterpret_cast<unsigned char *>(encoded.data()),reinterpret_cast<const unsigned char *>(bytes.data()),bytes.size());
        auto request=row.at("request");request["base64_audio"]=encoded;auto actual=Json::parse(runtime->transcribe(request.dump()));auto expected=row.at("output");
        normalize(actual,native_ids);normalize(expected,reference_ids);compare_json(actual,expected,"real diarization case "+std::to_string(count));
        expect(runtime->device()=="cpu"&&runtime->compute_type()=="int8","CPU inference remains on selected device");
        expect(fs::is_empty(temporary.path/"outputs"),"temporary audio released");std::cout<<"real diarization case "<<count++<<" passed\n"<<std::flush;
    }
    runtime->close();expect(!runtime->has_model()&&!runtime->has_audio(),"model/audio handles closed");
    expect(native_ids.size()==2&&reference_ids.size()==2,"same-session identity survives model reload; other session is distinct");
    for (const auto &entry:environment_defaults) unsetenv(entry.first.data());
    setenv("EZLOCALAI_DOTENV_PATH","",1);setenv("EZLOCALAI_API_KEY","none",1);setenv("API_KEY","",1);setenv("ROUTER_URL","",1);setenv("DEFAULT_MODEL","none",1);
    for (auto key:{"TTS_ENABLED","EMBEDDING_ENABLED","IMAGE_ENABLED","VIDEO_ENABLED","MUSIC_ENABLED"}) setenv(key,"false",1);
    setenv("STT_ENABLED","true",1);setenv("LAZY_LOAD_VOICE","true",1);setenv("STT_UNLOAD_LLM_DURING_GENERATION","false",1);setenv("WHISPER_MODEL",argv[3],1);
    ezl_env_config config{};ezl_load_config(&config);
    auto probes=native_resource_probes();probes.gpu_count=[]{return 0;};probes.total_per_gpu=[]{return std::vector<double>{};};probes.free_per_gpu=[]{return std::vector<double>{};};
    auto resources=std::make_shared<ResourceManager>(probes);
    NativeServer server(config,std::make_unique<EchoInferenceBackend>("none"),resources,[&](auto manager){return stt_whisper_pool_operations(configuration,manager,platform);});
    for (const auto format:{"verbose_json","json"}) {
        const auto &row=fixture.at("cases")[2];const auto audio=read(fs::path(argv[5])/row.at("file").get<std::string>());
        std::string body="--diarization-boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"speech.wav\"\r\nContent-Type: audio/wav\r\n\r\n"+audio+"\r\n";
        for(const auto &field:std::map<std::string,std::string>{{"language","en"},{"enable_diarization","true"},{"session_id","a"},{"response_format",format}})
            body+="--diarization-boundary\r\nContent-Disposition: form-data; name=\""+field.first+"\"\r\n\r\n"+field.second+"\r\n";
        body+="--diarization-boundary--\r\n";
        const auto response=server.handle_request("POST","/v1/audio/transcriptions","",body,"multipart/form-data; boundary=diarization-boundary");
        expect(response.status==200,"public diarization API status: "+response.body);
        auto actual=Json::parse(response.body);auto expected=row.at("output");if(std::string(format)=="json")expected=Json{{"text",expected.at("text")}};
        normalize(actual,native_ids);normalize(expected,reference_ids);compare_json(actual,expected,"public API "+std::string(format));
        std::cout<<"public diarization API "<<format<<" passed\n"<<std::flush;
    }
    std::cout<<count<<" source model-backed diarization calls and two public API calls passed\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
