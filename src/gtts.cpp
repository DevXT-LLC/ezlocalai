#include "ezlocalai/gtts.hpp"
#include "ezlocalai/tts_unicode.hpp"
#include "ezlocalai/tts_text.hpp"
#include "ezlocalai/llm_slots.hpp"
#include "ezlocalai/stt_input.hpp"
#include "ezlocalai/llm_response.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <iomanip>
#include <sstream>
namespace ezlocalai {
namespace {
using Json=nlohmann::json;
using Text=std::u32string;
const Text all_punctuation=U"?!？！.,¡()[]¿…‥،;:—。，、：\n";
bool tone(char32_t c){return c==U'?'||c==U'!'||c==U'？'||c==U'！';}
template<size_t N> bool in(char32_t c,const tts_unicode::Range (&ranges)[N]){
    const auto it=std::upper_bound(std::begin(ranges),std::end(ranges),c,[](auto c,const auto &r){return c<r.first;});
    return it!=std::begin(ranges)&&c<=(it-1)->last;
}
[[noreturn]] void decode_error(const std::string &s,size_t start,size_t end,const char *reason){
    std::ostringstream out;out<<"'utf-8' codec can't decode ";
    if(end==start+1)out<<"byte 0x"<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(static_cast<unsigned char>(s[start]))<<std::dec<<" in position "<<start;
    else out<<"bytes in position "<<start<<'-'<<end-1;
    out<<": "<<reason;throw TtsError("UnicodeDecodeError",out.str());
}
Text decode(const std::string &s){
    Text out;for(size_t i=0;i<s.size();){const auto c=static_cast<unsigned char>(s[i]);if(c<128){out+=c;++i;continue;}
        const int n=c>=0xc2&&c<=0xdf?2:c>=0xe0&&c<=0xef?3:c>=0xf0&&c<=0xf4?4:0;
        if(!n)decode_error(s,i,i+1,"invalid start byte");uint32_t value=c&((1<<(7-n))-1);
        for(int j=1;j<n;++j){if(i+j==s.size())decode_error(s,i,i+j,"unexpected end of data");const auto d=static_cast<unsigned char>(s[i+j]);
            if(d<0x80||d>0xbf||(j==1&&((c==0xe0&&d<0xa0)||(c==0xed&&d>=0xa0)||(c==0xf0&&d<0x90)||(c==0xf4&&d>0x8f))))decode_error(s,i,i+j,"invalid continuation byte");value=(value<<6)|(d&63);}
        out+=value;i+=n;
    }return out;
}
std::string encode(const Text &s){std::string out;for(uint32_t c:s){if(c<128)out+=char(c);else if(c<0x800){out+=char(0xc0|(c>>6));out+=char(0x80|(c&63));}else if(c<0x10000){out+=char(0xe0|(c>>12));out+=char(0x80|((c>>6)&63));out+=char(0x80|(c&63));}else{out+=char(0xf0|(c>>18));out+=char(0x80|((c>>12)&63));out+=char(0x80|((c>>6)&63));out+=char(0x80|(c&63));}}return out;}
Text strip(const Text &s){size_t a=0,b=s.size();while(a<b&&in(s[a],tts_unicode::spaces))++a;while(b>a&&in(s[b-1],tts_unicode::spaces))--b;return s.substr(a,b-a);}
char32_t fold(char32_t c){if(c>=U'A'&&c<=U'Z')return c+32;if(c==0x130||c==0x131)return U'i';if(c==0x17f)return U's';if(c==0x212a)return U'k';return c;}
bool equal_at(const Text &s,size_t at,const Text &word){if(at+word.size()>s.size())return false;for(size_t i=0;i<word.size();++i)if(fold(s[at+i])!=word[i])return false;return true;}
bool clean_keep(const Text &s){for(auto c:s)if(all_punctuation.find(c)==Text::npos&&Text(U" \t\n\r\v\f").find(c)==Text::npos)return true;return false;}
std::vector<Text> clean(const std::vector<Text> &parts){std::vector<Text> out;for(const auto &s:parts)if(clean_keep(s))out.push_back(strip(s));return out;}
std::string quote(const std::string &s){const char *hex="0123456789ABCDEF";std::string out;for(unsigned char c:s){if((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-'||c=='.'||c=='~'||c=='/')out+=char(c);else{out+='%';out+=hex[c>>4];out+=hex[c&15];}}return out;}
[[noreturn]] void response_error(const GttsResponse &r){throw TtsError("gTTSError",std::to_string(r.status)+" ("+r.reason+") from TTS API. Probable cause: "+(r.status==403?"Bad token or upstream API changes":r.status>=500?"Upstream API error. Try again later.":"Unknown"));}
void check_ascii(const Text &s){for(size_t i=0;i<s.size();++i)if(s[i]>127){size_t end=i+1;while(end<s.size()&&s[end]>127)++end;std::ostringstream out;out<<"'ascii' codec can't encode character";
        if(end==i+1){out<<" '\\"<<(s[i]<=255?"x":s[i]<=65535?"u":"U")<<std::hex<<std::setw(s[i]<=255?2:s[i]<=65535?4:8)<<std::setfill('0')<<uint32_t(s[i])<<std::dec<<"' in position "<<i;}
        else out<<"s in position "<<i<<'-'<<end-1;
        out<<": ordinal not in range(128)";throw TtsError("UnicodeEncodeError",out.str());}}
}
std::string gtts_language(const std::string &language_json,const std::string &text){
    const auto value=Json::parse(language_json);const bool truth=!value.is_null()&&!(value.is_boolean()&&!value.get<bool>())&&!(value.is_number()&&value==0)&&!((value.is_string()||value.is_array()||value.is_object())&&value.empty());
    const auto requested=llm_unicode_lower(encode(strip(decode(truth?tts_python_string(language_json):""))));
    if(requested=="ru"||requested=="rus"||requested=="russian")return "ru";if(requested=="en"||requested=="eng"||requested=="english")return "en";return tts_contains_cyrillic(text)?"ru":"en";
}
std::vector<std::string> gtts_tokenize(const std::string &text){
    Text s;for(auto c:strip(decode(text))){s+=c;if(tone(c))s+=U' ';}
    for(size_t at=0;(at=s.find(U"-\n",at))!=Text::npos;)s.erase(at,2);
    for(const Text abbreviation:{U"dr",U"jr",U"mr",U"mrs",U"ms",U"msgr",U"prof",U"sr",U"st"}){
        Text out;for(size_t i=0;i<s.size();++i)if(s[i]!=U'.'||i<abbreviation.size()||!equal_at(s,i-abbreviation.size(),abbreviation))out+=s[i];s=std::move(out);
    }
    {Text out;for(size_t i=0;i<s.size();)if(equal_at(s,i,U"esq.")){out+=U"Esquire";i+=4;}else out+=s[i++];s=std::move(out);}
    std::vector<Text> parts;
    if(s.size()<=100)parts=clean({s});
    else{
        size_t start=0;for(size_t i=0;i<s.size();){size_t width=0;const auto c=s[i];
            if(i&&tone(s[i-1])&&c!=U'\n')width=1;
            else if((c==U'.'||c==U',')&&i+1<s.size()&&s[i+1]==U' '&&!(i>=2&&s[i-2]==U'.'&&fold(s[i-1])>=U'a'&&fold(s[i-1])<=U'z'))width=2;
            else if(c==U':'&&(!i||!in(s[i-1],tts_unicode::decimals)))width=1;
            else if(all_punctuation.find(c)!=Text::npos&&!tone(c)&&c!=U'.'&&c!=U','&&c!=U':')width=1;
            if(width){parts.push_back(s.substr(start,i-start));i+=width;start=i;}else ++i;
        }parts.push_back(s.substr(start));parts=clean(parts);std::vector<Text> minimized;
        for(auto part:parts){for(;;){if(!part.empty()&&part[0]==U' ')part.erase(0,1);if(part.size()<=100){if(!part.empty())minimized.push_back(part);break;}const auto found=part.rfind(U' ',99);const auto cut=found==Text::npos?100:found;if(cut)minimized.push_back(part.substr(0,cut));part.erase(0,cut);}}
        parts=std::move(minimized);
    }
    std::vector<std::string> out;for(const auto &s:parts)out.push_back(encode(s));return out;
}
std::vector<GttsRequest> gtts_prepare(const std::string &text,const std::string &language_json){
    if(text.empty())throw TtsError("AssertionError","No text to speak");const auto language=gtts_language(language_json,text);const auto parts=gtts_tokenize(text);if(parts.empty())throw TtsError("AssertionError","No text to send to TTS API");
    std::vector<GttsRequest> out;for(const auto &part:parts){const auto parameter=Json::array({part,language,nullptr,"null"}).dump(-1,' ',true);const auto rpc=Json::array({Json::array({Json::array({"jQ1olc",parameter,nullptr,"generic"})})}).dump(-1,' ',true);
        GttsRequest request;request.url="https://translate.google.com/_/TranslateWebserverUi/data/batchexecute";request.body="f.req="+quote(rpc)+"&";
        request.headers={{"Referer","http://translate.google.com/"},{"User-Agent","Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36"},{"Content-Type","application/x-www-form-urlencoded;charset=utf-8"},{"Content-Length",std::to_string(request.body.size())}};out.push_back(std::move(request));
    }return out;
}
std::string gtts_response_audio(const GttsResponse &r){
    if(r.status>=400&&r.status<600)response_error(r);std::string out;
    // requests.Response.iter_lines() splits bytes at CR and LF before UTF-8 decoding.
    for(size_t start=0;start<r.body.size();){auto end=r.body.find_first_of("\r\n",start);if(end==std::string::npos)end=r.body.size();const auto line=r.body.substr(start,end-start);decode(line);start=end==r.body.size()?end:end+1;if(end<r.body.size()&&r.body[end]=='\r'&&start<r.body.size()&&r.body[start]=='\n')++start;
        if(line.find("jQ1olc")==std::string::npos)continue;const std::string prefix="jQ1olc\",\"[\\\"",suffix="\\\"]";const auto begin=line.find(prefix),last=line.rfind(suffix);
        if(begin==std::string::npos||last==std::string::npos||last<begin+prefix.size())response_error(r);const auto encoded=line.substr(begin+prefix.size(),last-begin-prefix.size());check_ascii(decode(encoded));
        try{out+=stt_base64_decode(Json(encoded).dump());}catch(const LlmResponseError &e){throw TtsError(e.category,e.message);}
    }return out;
}
std::string gtts_generate_wav(const std::string &text,const std::string &language_json,GttsOperations operations){
    const auto requests=gtts_prepare(text,language_json);operations=gtts_native_operations(std::move(operations));std::string mp3;for(const auto &request:requests)mp3+=gtts_response_audio(operations.post(request));return operations.to_wav(mp3);
}
TtsRequestOperations tts_gtts_operations(TtsRequestOperations operations,GttsOperations fallback){
    if(!operations.fallback_generate)operations.fallback_generate=[fallback=std::move(fallback)](const std::string &request){const auto data=Json::parse(request);return gtts_generate_wav(data.at("text").get<std::string>(),data.value("language",Json()).dump(),fallback);};return operations;
}
}
