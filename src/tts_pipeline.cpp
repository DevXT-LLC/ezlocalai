#include "ezlocalai/tts_pipeline.hpp"
#include "ezlocalai/stt_input.hpp"
#include "ezlocalai/llm_response.hpp"
#include "ezlocalai/llama_tts.hpp"
#include <nlohmann/json.hpp>
namespace ezlocalai {
namespace {
std::string part(const std::string &value,char separator,size_t index){
    size_t at=0;for(size_t i=0;i<index;++i){const auto next=value.find(separator,at);if(next==std::string::npos)throw TtsError("IndexError","list index out of range");at=next+1;}
    return value.substr(at,value.find(separator,at)-at);
}
}
TtsAudioPipelineInput tts_audio_pipeline_input(const std::string &encoded){
    const auto value=nlohmann::json::parse(encoded);
    if(!value.is_string()){
        const auto type=value.is_null()?"NoneType":value.is_boolean()?"bool":value.is_number_integer()?"int":value.is_number()?"float":value.is_array()?"list":"dict";
        throw TtsError("AttributeError",std::string("'")+type+"' object has no attribute 'split'");
    }
    const auto audio=value.get<std::string>();TtsAudioPipelineInput result;
    result.format=part(part(part(part(audio,',',0),':',1),';',0),'/',1);
    try{result.bytes=stt_base64_decode(nlohmann::json(part(audio,',',1)).dump());}catch(const LlmResponseError &error){throw TtsError(error.category,error.message);}
    return result;
}
std::string tts_audio_to_audio(const std::string &encoded,const std::string &uri,const TtsAudioPipelineOperations &ops){
    const auto request=nlohmann::json::parse(encoded);const auto input=tts_audio_pipeline_input(request.at("audio").dump());
    const auto text=nlohmann::json::parse(ops.transcribe(nlohmann::json{{"audio_format",input.format}}.dump(),input.bytes));
    return ops.generate(nlohmann::json{{"text",text},{"voice",request.at("voice")},{"local_uri",nlohmann::json::parse(uri)}}.dump());
}
}
