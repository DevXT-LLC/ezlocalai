#include "ezlocalai/tts_websocket.hpp"
#include "ezlocalai/json_validation.hpp"
#include "ezlocalai/tts_unicode.hpp"
#include <nlohmann/json.hpp>
#include <unicode/utf8.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <thread>

namespace ezlocalai {
namespace {
[[noreturn]] void invalid(const std::string &message){throw TtsError("ValueError",message);}
uint32_t little32(const std::string &bytes){uint32_t value=0;for(unsigned i=0;i<4;++i)value|=uint32_t(static_cast<unsigned char>(bytes[i]))<<(8*i);return value;}
bool space(char32_t c){for(const auto &range:tts_unicode::spaces)if(c>=range.first&&c<=range.last)return true;return false;}
std::u32string decode(const std::string &value){std::u32string result;for(int32_t at=0;at<static_cast<int32_t>(value.size());){UChar32 c;U8_NEXT(value.data(),at,static_cast<int32_t>(value.size()),c);if(c<0)invalid("Invalid UTF-8 text");result.push_back(c);}return result;}
std::string encode(const std::u32string &value){std::string result;for(auto c:value){char bytes[4];int32_t at=0;U8_APPEND_UNSAFE(bytes,at,c);result.append(bytes,at);}return result;}
std::u32string strip(const std::u32string &value){size_t begin=0,end=value.size();while(begin<end&&space(value[begin]))++begin;while(end>begin&&space(value[end-1]))--end;return value.substr(begin,end-begin);}
bool truthy(const nlohmann::json &value){if(value.is_null())return false;if(value.is_boolean())return value.get<bool>();if(value.is_number())return value!=0;if(value.is_string())return !value.get_ref<const std::string &>().empty();return !value.empty();}
// Numeric fields only participate in type validation or Python truth testing.
// Preserve that information without narrowing Python integers or rejecting its
// JSON NaN/Infinity constants in the native JSON library.
std::string input_json(const std::string &raw){
    std::string out;
    for(size_t at=0;at<raw.size();){
        const auto start=at;
        if(raw[at]=='"'){++at;while(at<raw.size()){if(raw[at++]=='"')break;if(raw[at-1]=='\\'&&at<raw.size())++at;}out+=raw.substr(start,at-start);}
        else if(raw[at]=='-'||(raw[at]>='0'&&raw[at]<='9')||raw[at]=='N'||raw[at]=='I'){
            while(at<raw.size()&&std::string(" \t\r\n,]}").find(raw[at])==std::string::npos)++at;
            const auto token=raw.substr(start,at-start);
            if(token.find_first_of(".eENI")==std::string::npos){
                const auto digits=token.size()-(token.front()=='-');
                if(digits>4300)invalid("Exceeds the limit (4300 digits) for integer string conversion: value has "+std::to_string(digits)+" digits; use sys.set_int_max_str_digits() to increase the limit");
                out+=token.find_first_not_of("-0")==std::string::npos?'0':'1';
            }else out+=std::strtod(token.c_str(),nullptr)==0?'0':'1';
        }else out+=raw[at++];
    }
    return out;
}
template<class Body,class Cleanup>void finally(Body body,Cleanup cleanup){std::exception_ptr failure;try{body();}catch(...){failure=std::current_exception();}cleanup();if(failure)std::rethrow_exception(failure);}
}

std::string tts_pcm_header(){return std::string("\xc0\x5d\0\0\x10\0\x01\0",8);}
void TtsPcmSegment::feed(const std::string &bytes,const Emit &emit){
    if(ended_&&!bytes.empty())invalid("Unexpected bytes after TTS segment end");
    buffer_+=bytes;
    if(!header_){
        if(buffer_.size()<8)return;
        const auto header=buffer_.substr(0,8);buffer_.erase(0,8);
        if(header!=tts_pcm_header())invalid("Unsupported TTS audio format");
        emit(true,header);header_=true;
    }
    while(!buffer_.empty()&&!ended_){
        if(!remaining_){
            if(buffer_.size()<4)break;
            const auto length=buffer_.substr(0,4);remaining_=little32(length);buffer_.erase(0,4);
            if(!*remaining_){ended_=true;if(!buffer_.empty())invalid("Unexpected bytes after TTS segment end");break;}
            if(*remaining_>16*1024*1024||*remaining_%2)invalid("Invalid TTS PCM frame size");
            emit(false,length);
        }
        const auto take=std::min<size_t>(*remaining_,buffer_.size());
        if(take){emit(false,buffer_.substr(0,take));buffer_.erase(0,take);*remaining_-=take;}
        if(!*remaining_)remaining_.reset();
    }
}
void TtsPcmSegment::finish()const{if(!header_||!ended_||!buffer_.empty())invalid("Truncated TTS segment");}

TtsWebSocketInputResult TtsWebSocketInput::receive(const std::string &json){
    if(const auto error=json_decode_error(json)){
        const auto chars=decode(json);size_t line=1,column=1;
        for(size_t i=0;i<error->position&&i<chars.size();++i){if(chars[i]=='\n'){++line;column=1;}else ++column;}
        throw TtsError("JSONDecodeError",error->message+": line "+std::to_string(line)+" column "+std::to_string(column)+" (char "+std::to_string(error->position)+")");
    }
    const auto data=nlohmann::json::parse(input_json(json));
    if(!data.is_object())invalid("Expected a JSON object");
    const auto text=data.value("text",nlohmann::json(""));
    if(!text.is_string())invalid("Text buffer exceeds 4096 characters");
    const auto added=decode(text.get<std::string>());
    if(buffer_.size()+added.size()>4096)invalid("Text buffer exceeds 4096 characters");
    const auto voice=data.value("voice",nlohmann::json(voice_)),language=data.value("language",nlohmann::json(language_));
    if(!voice.is_string()||!language.is_string())invalid("Voice and language must be strings");
    if(!strip(buffer_).empty()&&(voice!=voice_||language!=language_))invalid("Flush buffered text before changing voice or language");
    voice_=voice.get<std::string>();language_=language.get<std::string>();buffer_+=added;
    TtsWebSocketInputResult result;result.done=truthy(data.value("done",nlohmann::json()));
    size_t boundary=0;
    if(result.done||truthy(data.value("flush",nlohmann::json())))boundary=buffer_.size();
    else if(buffer_.size()>=50){
        for(size_t at=0;at<buffer_.size();++at){
            if(std::u32string(U".!?。！？").find(buffer_[at])==std::u32string::npos)continue;
            if(at+1==buffer_.size()){boundary=at+1;continue;}
            if(!space(buffer_[at+1]))continue;
            size_t end=at+1;while(end<buffer_.size()&&space(buffer_[end]))++end;boundary=end;at=end-1;
        }
        if(!boundary&&buffer_.size()>=350){
            const auto found=buffer_.rfind(U' ',350);boundary=found==std::u32string::npos||found==0?350:found;
        }
    }
    if(boundary){
        const auto ready=strip(buffer_.substr(0,boundary));buffer_.erase(0,boundary);
        if(!ready.empty())result.request=nlohmann::json{{"text",encode(ready)},{"voice",voice_},{"language",language_}}.dump();
    }
    return result;
}

void run_tts_websocket(const TtsWebSocketOperations &ops){
    std::mutex mutex;std::condition_variable ready;std::deque<std::optional<std::string>> pending;
    VoiceCancellation cancellation;std::exception_ptr failure;std::shared_ptr<CttsByteStream> active;
    const auto fail=[&](std::exception_ptr error){
        {std::lock_guard<std::mutex> lock(mutex);if(!failure)failure=error;}
        cancellation.cancel();ready.notify_all();
        try{if(const auto stream=std::atomic_load(&active))stream->request_cancel();}catch(...){}
        try{ops.interrupt_receive();}catch(...){}
    };
    const auto push=[&](std::optional<std::string> request){std::unique_lock<std::mutex> lock(mutex);ready.wait(lock,[&]{return pending.size()<4||cancellation.requested();});if(cancellation.requested())throw VoiceCancelled("");pending.push_back(std::move(request));lock.unlock();ready.notify_all();};
    std::thread receiver([&]{try{TtsWebSocketInput input;for(;;){const auto result=input.receive(ops.receive());if(result.request)push(result.request);if(result.done){push({});return;}}}catch(...){fail(std::current_exception());}});
    try{
        bool header_sent=false,done=false;
        while(!done){
            std::optional<std::string> request;
            {std::unique_lock<std::mutex> lock(mutex);ready.wait(lock,[&]{return !pending.empty()||cancellation.requested();});if(cancellation.requested())throw VoiceCancelled("");request=std::move(pending.front());pending.pop_front();}ready.notify_all();
            if(!request)break;
            auto lease=ops.acquire(cancellation);
            finally([&]{
                for(;;){
                    auto stream=std::shared_ptr<CttsByteStream>(lease->stream(*request));std::atomic_store(&active,stream);
                    finally([&]{
                        TtsPcmSegment segment;
                        for(;;){if(cancellation.requested())throw VoiceCancelled("");const auto packet=stream->next();if(!packet)break;segment.feed(*packet,[&](bool header,const auto &bytes){if(!header||!header_sent){ops.send(bytes);if(header)header_sent=true;}});}
                        segment.finish();
                    },[&]{std::atomic_store(&active,std::shared_ptr<CttsByteStream>{});stream->close();});
                    {std::lock_guard<std::mutex> lock(mutex);if(cancellation.requested())throw VoiceCancelled("");if(pending.empty())break;request=std::move(pending.front());pending.pop_front();}ready.notify_all();
                    if(!request){done=true;break;}
                }
            },[&]{lease->close();});
        }
        if(cancellation.requested())throw VoiceCancelled("");
        if(!header_sent)ops.send(tts_pcm_header());
        ops.send(std::string(4,'\0'));ops.send("");
    }catch(...){fail(std::current_exception());}
    receiver.join();
    if(failure)std::rethrow_exception(failure);
}
}
