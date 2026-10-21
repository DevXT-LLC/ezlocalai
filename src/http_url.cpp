#include "ezlocalai/http_url.hpp"
#include "ezlocalai/configuration.hpp"
#include <algorithm>
#include <arpa/inet.h>
#include <array>
#include <cctype>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unicode/uidna.h>
#include <unicode/unorm2.h>
#include <unicode/ustring.h>
#include <vector>

namespace ezlocalai {
namespace {
[[noreturn]] void invalid_url() { throw std::invalid_argument("Invalid HTTP URL"); }
std::string lower(std::string value) { for(auto &c:value)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));return value; }
int hex(char c) { return c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:c>='A'&&c<='F'?c-'A'+10:-1; }
std::u16string unicode(const std::string &value,bool replace=false) {
    UErrorCode error=U_ZERO_ERROR;int32_t size=0;
    if(replace)u_strFromUTF8WithSub(nullptr,0,&size,value.data(),value.size(),0xfffd,nullptr,&error);
    else u_strFromUTF8(nullptr,0,&size,value.data(),value.size(),&error);
    if(error!=U_BUFFER_OVERFLOW_ERROR&&U_FAILURE(error))invalid_url();
    std::u16string out(size,0);error=U_ZERO_ERROR;
    if(replace)u_strFromUTF8WithSub(reinterpret_cast<UChar *>(out.data()),size,nullptr,value.data(),value.size(),0xfffd,nullptr,&error);
    else u_strFromUTF8(reinterpret_cast<UChar *>(out.data()),size,nullptr,value.data(),value.size(),&error);
    if(U_FAILURE(error))invalid_url();return out;
}
void check_netloc(const std::string &value) {
    if(std::all_of(value.begin(),value.end(),[](unsigned char c){return c<128;}))return;
    std::string cleaned;for(char c:value)if(std::string("@:#?").find(c)==std::string::npos)cleaned+=c;
    // Surrogateescaped header bytes are inert for this delimiter check. Their
    // acceptance is decided later by host encoding or userinfo requoting.
    const auto raw=unicode(cleaned,true);UErrorCode error=U_ZERO_ERROR;const auto *normalizer=unorm2_getNFKCInstance(&error);
    const auto size=unorm2_normalize(normalizer,reinterpret_cast<const UChar *>(raw.data()),raw.size(),nullptr,0,&error);
    if(error!=U_BUFFER_OVERFLOW_ERROR&&U_FAILURE(error))invalid_url();
    std::u16string out(size,0);error=U_ZERO_ERROR;unorm2_normalize(normalizer,reinterpret_cast<const UChar *>(raw.data()),raw.size(),reinterpret_cast<UChar *>(out.data()),size,&error);
    if(U_FAILURE(error))invalid_url();if(out!=raw&&out.find_first_of(u"/?#@:")!=std::u16string::npos)invalid_url();
}
std::string idna_host(const std::string &host) {
    UErrorCode error=U_ZERO_ERROR;auto *idna=uidna_openUTS46(UIDNA_NONTRANSITIONAL_TO_ASCII|UIDNA_CHECK_BIDI|UIDNA_CHECK_CONTEXTJ|UIDNA_CHECK_CONTEXTO,&error);
    if(U_FAILURE(error))invalid_url();UIDNAInfo info=UIDNA_INFO_INITIALIZER;std::vector<char> out(host.size()*8+256);
    const int32_t size=uidna_nameToASCII_UTF8(idna,host.data(),host.size(),out.data(),out.size(),&info,&error);uidna_close(idna);
    if(U_SUCCESS(error)&&!info.errors)return std::string(out.data(),size);
    // yarl falls back to Python's IDNA 2003 codec when IDNA 2008 rejects a name.
    const auto raw=unicode(host);std::vector<UChar> legacy(host.size()*8+256);error=U_ZERO_ERROR;
    const int32_t legacy_size=uidna_IDNToASCII(reinterpret_cast<const UChar *>(raw.data()),raw.size(),legacy.data(),legacy.size(),UIDNA_ALLOW_UNASSIGNED,nullptr,&error);
    if(U_FAILURE(error))invalid_url();std::string result;for(int32_t i=0;i<legacy_size;++i){if(legacy[i]>127)invalid_url();result+=static_cast<char>(legacy[i]);}return result;
}
std::string encoded_host(std::string host) {
    const auto percent=host.find('%');const auto ip=host.substr(0,percent);const auto scope=percent==std::string::npos?"":host.substr(percent);in6_addr v6{};in_addr v4{};
    if(inet_pton(AF_INET6,ip.c_str(),&v6)==1){
        std::array<unsigned,8> words{};for(size_t i=0;i<8;++i)words[i]=(unsigned(v6.s6_addr[i*2])<<8)|v6.s6_addr[i*2+1];
        size_t best=8,length=0;for(size_t i=0;i<8;){if(words[i]){++i;continue;}const auto begin=i;while(i<8&&!words[i])++i;if(i-begin>length){best=begin;length=i-begin;}}
        auto join=[&](size_t a,size_t b){std::ostringstream text;for(size_t i=a;i<b;++i){if(i>a)text<<':';text<<std::hex<<words[i];}return text.str();};
        host=length>=2?join(0,best)+"::"+join(best+length,8):join(0,8);
        return host+scope;
    }
    if(inet_pton(AF_INET,ip.c_str(),&v4)==1){char buffer[INET_ADDRSTRLEN];inet_ntop(AF_INET,&v4,buffer,sizeof(buffer));return std::string(buffer)+(percent==std::string::npos?"":host.substr(percent));}
    if(std::all_of(host.begin(),host.end(),[](unsigned char c){return c<128;}))return lower(host);
    return idna_host(host);
}
size_t utf8_width(const std::string &value,size_t offset) {
    const auto first=static_cast<unsigned char>(value[offset]);
    const size_t width=first>=0xc2&&first<=0xdf?2:first>=0xe0&&first<=0xef?3:first>=0xf0&&first<=0xf4?4:0;
    if(!width||offset+width>value.size())return 0;
    unsigned code=first&((1<<(7-width))-1);
    for(size_t i=1;i<width;++i){const auto next=static_cast<unsigned char>(value[offset+i]);if(next<0x80||next>0xbf)return 0;code=(code<<6)|(next&63);}
    return code>=(width==2?0x80U:width==3?0x800U:0x10000U)&&code<=0x10ffffU&&(code<0xd800||code>0xdfff)?width:0;
}
std::string utf8_ignore_errors(const std::string &input) {
    std::string value;
    for(size_t i=0;i<input.size();){const auto c=static_cast<unsigned char>(input[i]);const auto width=c<128?1:utf8_width(input,i);if(width)value.append(input,i,width);i+=width?width:1;}
    return value;
}
std::string quote(const std::string &input,const std::string &extra,const std::string &protected_chars,bool query=false) {
    const auto value=utf8_ignore_errors(input);
    std::string safe="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~!$'()*,"+extra+protected_chars;
    if(!query)safe+="+&=;";static const char digits[]="0123456789ABCDEF";std::string out;
    auto escaped=[&](unsigned char c){out+='%';out+=digits[c>>4];out+=digits[c&15];};
    for(size_t i=0;i<value.size();++i){const auto c=static_cast<unsigned char>(value[i]);
        // HTTP header text uses surrogateescape. Yarl drops undecodable bytes
        // when quoting changes the text, but returns the original string when
        // it can skip allocation. The HTTP writer later drops retained surrogates.
        if(c>=128){const auto width=utf8_width(value,i);if(!width)continue;for(size_t j=0;j<width;++j)escaped(static_cast<unsigned char>(value[i+j]));i+=width-1;}
        else if(c=='%'&&i+2<value.size()&&hex(value[i+1])>=0&&hex(value[i+2])>=0){const auto byte=static_cast<unsigned char>(hex(value[i+1])*16+hex(value[i+2]));if(safe.find(byte)!=std::string::npos&&protected_chars.find(byte)==std::string::npos)out+=byte;else escaped(byte);i+=2;}
        else if(query&&c==' ')out+='+';else if(safe.find(c)!=std::string::npos)out+=c;else escaped(c);
    }return out==value?input:out;
}
std::string normalize_path(const std::string &value) {
    const bool absolute=!value.empty()&&value.front()=='/';std::vector<std::string> parts;size_t pos=absolute?1:0;std::string last;
    for(;;){const auto end=value.find('/',pos);last=value.substr(pos,end==std::string::npos?end:end-pos);if(last==".."){if(!parts.empty())parts.pop_back();}else if(last!=".")parts.push_back(last);if(end==std::string::npos)break;pos=end+1;}
    if(last=="."||last=="..")parts.emplace_back();std::string out=absolute?"/":"";for(size_t i=0;i<parts.size();++i){if(i)out+='/';out+=parts[i];}return out;
}
std::string unquote(const std::string &value) {
    // Yarl preserves percent escapes that do not form valid UTF-8.
    auto byte_at=[&](size_t i)->int{return i+2<value.size()&&value[i]=='%'&&hex(value[i+1])>=0&&hex(value[i+2])>=0?hex(value[i+1])*16+hex(value[i+2]):-1;};
    std::string out;
    for(size_t i=0;i<value.size();){
        const int first=byte_at(i);if(first<0){out+=value[i++];continue;}
        if(first<128){out+=static_cast<char>(first);i+=3;continue;}
        const int length=first>=0xc2&&first<=0xdf?2:first>=0xe0&&first<=0xef?3:first>=0xf0&&first<=0xf4?4:0;
        unsigned code=length?first&((1<<(7-length))-1):0;bool valid=length!=0;
        for(int j=1;j<length;++j){const int next=byte_at(i+3*j);if(next<0x80||next>0xbf){valid=false;break;}code=(code<<6)|(next&63);}
        valid=valid&&code>=(length==2?0x80U:length==3?0x800U:0x10000U)&&code<=0x10ffffU&&(code<0xd800||code>0xdfff);
        if(!valid){out+=value.substr(i,3);i+=3;continue;}
        for(int j=0;j<length;++j)out+=static_cast<char>(byte_at(i+3*j));i+=3*length;
    }
    return out;
}
std::string unquote_latin1(const std::string &value) {
    std::string result;for(const auto c:unicode(unquote(value))){if(c>255)invalid_url();result+=static_cast<char>(c);}return result;
}
std::string host_part(const std::string &host) { return host.find(':')==std::string::npos?host:"["+host+"]"; }
}

HttpUrl HttpUrl::parse(const std::string &input) {
    HttpUrl out;size_t begin=0;while(begin<input.size()&&static_cast<unsigned char>(input[begin])<=32)++begin;std::string rest;
    for(size_t i=begin;i<input.size();++i)if(input[i]!='\r'&&input[i]!='\n'&&input[i]!='\t')rest+=input[i];
    const auto colon=rest.find(':');const std::string scheme_chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.";
    if(colon!=std::string::npos&&colon>0&&rest.substr(0,colon).find_first_not_of(scheme_chars)==std::string::npos){out.scheme=lower(rest.substr(0,colon));rest.erase(0,colon+1);}
    if(rest.compare(0,2,"//")==0){
        const auto end=rest.find_first_of("/?#",2);std::string netloc=rest.substr(2,end==std::string::npos?end:end-2);rest=end==std::string::npos?"":rest.substr(end);out.authority=!netloc.empty();check_netloc(netloc);
        const auto at=netloc.rfind('@');if(at!=std::string::npos){const auto userinfo=netloc.substr(0,at);const auto split=userinfo.find(':');auto user=userinfo.substr(0,split);if(!user.empty())out.raw_user=quote(user,"","");if(split!=std::string::npos)out.raw_password=quote(userinfo.substr(split+1),"","");netloc.erase(0,at+1);}
        std::string host,port;
        if(netloc.find('[')!=std::string::npos){const auto open=netloc.find('['),close=netloc.find(']',open);if(close==std::string::npos)invalid_url();host=netloc.substr(open+1,close-open-1);if(host.find(':')==std::string::npos&&!std::regex_match(host,std::regex("v[0-9a-fA-F]+\\..+")))invalid_url();if(host.find(':')!=std::string::npos){in6_addr address{};if(inet_pton(AF_INET6,host.substr(0,host.find('%')).c_str(),&address)!=1)invalid_url();}const auto p=netloc.find(':',close+1);if(p!=std::string::npos)port=netloc.substr(p+1);}
        else{if(netloc.find(']')!=std::string::npos)invalid_url();const auto split=netloc.find(':');host=netloc.substr(0,split);if(split!=std::string::npos)port=netloc.substr(split+1);}
        if(!port.empty()){
            for(unsigned char c:port)if(c>=0x1c&&c<=0x1f)invalid_url();port=normalize_decimal_digits(strip_config_value(port));if(!std::regex_match(port,std::regex("[+-]?[0-9]+(_[0-9]+)*")))invalid_url();port.erase(std::remove(port.begin(),port.end(),'_'),port.end());
            const auto number=std::stoll(port);if(number<0||number>65535)invalid_url();out.explicit_port=static_cast<int>(number);
        }
        out.host=encoded_host(host);
        if(out.authority&&out.host.empty()&&(out.scheme=="http"||out.scheme=="https"))invalid_url();
    }
    const auto hash=rest.find('#');if(hash!=std::string::npos){out.fragment=quote(rest.substr(hash+1),"?/:@","");rest.resize(hash);}
    const auto query=rest.find('?');if(query!=std::string::npos){out.query=quote(rest.substr(query+1),"?/:@","=+&;",true);rest.resize(query);}
    out.path=quote(rest,"@:","/+");if(out.authority&&out.path.find('.')!=std::string::npos)out.path=normalize_path(out.path);return out;
}
HttpUrl HttpUrl::resolve(const std::string &location) const {
    auto other=parse(location);if(!other.scheme.empty())return other;
    if(other.authority){other.scheme=scheme;return other;}
    auto result=*this;
    if(!other.path.empty()){
        if(other.path.front()=='/')result.path=other.path;
        else if(path.empty())result.path="/"+other.path;
        else{const auto slash=path.rfind('/');result.path=(slash==std::string::npos?"":path.substr(0,slash+1))+other.path;}
        if(result.path.find('.')!=std::string::npos)result.path=normalize_path(result.path);
    }
    if(!other.path.empty()||!other.query.empty())result.query=other.query;
    if(!other.path.empty()||!other.fragment.empty())result.fragment=other.fragment;
    return result;
}
void HttpUrl::require_http() const {if((scheme!="http"&&scheme!="https")||host.empty())invalid_url();}
int HttpUrl::port() const {return explicit_port.value_or(scheme=="https"?443:80);}
std::string HttpUrl::str() const {
    std::string out=scheme.empty()?"":scheme+":";
    if(authority){out+="//";if(raw_user||raw_password){out+=raw_user.value_or("");if(raw_password)out+=":"+*raw_password;out+='@';}out+=host_part(host);if(explicit_port&&*explicit_port!=(scheme=="https"?443:scheme=="http"?80:-1))out+=":"+std::to_string(*explicit_port);}
    out+=path;if(authority&&path.empty()&&(!query.empty()||!fragment.empty()))out+='/';if(!query.empty())out+="?"+query;if(!fragment.empty())out+="#"+fragment;return out;
}
std::string HttpUrl::target() const {return (path.empty()?"/":path)+(query.empty()?"":"?"+query);}
std::string HttpUrl::request_target() const {return utf8_ignore_errors(target());}
std::string HttpUrl::decoded_path() const {return unquote(path.empty()?"/":path);}
std::string HttpUrl::host_header() const {auto value=host;while(!value.empty()&&value.back()=='.')value.pop_back();auto out=host_part(value);if(explicit_port&&*explicit_port!=(scheme=="https"?443:80))out+=":"+std::to_string(*explicit_port);return out;}
bool HttpUrl::same_origin(const HttpUrl &other) const {return scheme==other.scheme&&host==other.host&&port()==other.port();}
std::optional<std::pair<std::string,std::string>> HttpUrl::credentials() const {if(!raw_user&&!raw_password)return {};const auto user=unquote_latin1(raw_user.value_or(""));if(user.find(':')!=std::string::npos)invalid_url();return std::make_pair(user,unquote_latin1(raw_password.value_or("")));}
void HttpUrl::remove_credentials(){raw_user.reset();raw_password.reset();}
}
