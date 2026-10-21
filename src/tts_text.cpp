#include "ezlocalai/tts_text.hpp"
#include "ezlocalai/tts_unicode.hpp"
#include <algorithm>
#include <map>
#include <stdexcept>
#include <iterator>
namespace ezlocalai {
namespace {
using Text=std::u32string;
template<size_t N> const tts_unicode::Range *range(char32_t c,const tts_unicode::Range (&items)[N]) {
    const auto item=std::upper_bound(std::begin(items),std::end(items),c,[](auto c,const auto &r) { return c<r.first; });
    return item!=std::begin(items) && c<=(item-1)->last ? item-1 : nullptr;
}
bool space(char32_t c) { return range(c,tts_unicode::spaces); }
bool word(char32_t c) { return range(c,tts_unicode::words); }
bool cased(char32_t c) { return range(c,tts_unicode::cased); }
bool ignorable(char32_t c) { return range(c,tts_unicode::case_ignorable); }
int digit(char32_t c) { const auto *r=range(c,tts_unicode::decimals); return r ? static_cast<int>((c-r->first)%10) : -1; }
Text decode(const std::string &input) {
    Text out; out.reserve(input.size());
    for(size_t i=0;i<input.size();) {
        const auto first=static_cast<unsigned char>(input[i++]); uint32_t c=first; int remaining=0;
        if(first<0x80) {} else if(first>=0xc2 && first<=0xdf) { c=first&31; remaining=1; } else if(first>=0xe0 && first<=0xef) { c=first&15; remaining=2; } else if(first>=0xf0 && first<=0xf4) { c=first&7; remaining=3; } else throw std::invalid_argument("Invalid UTF-8 text");
        const auto width=remaining; while(remaining--) { if(i==input.size()) throw std::invalid_argument("Incomplete UTF-8 text"); const auto next=static_cast<unsigned char>(input[i++]); if((next&0xc0)!=0x80) throw std::invalid_argument("Invalid UTF-8 text"); c=(c<<6)|(next&63); }
        if((width==1 && c<0x80) || (width==2 && c<0x800) || (width==3 && c<0x10000) || c>0x10ffff || (c>=0xd800 && c<=0xdfff)) throw std::invalid_argument("Invalid UTF-8 text"); out.push_back(c);
    }
    return out;
}
std::string encode(const Text &input) {
    std::string out; for(uint32_t c:input) { if(c<0x80) out+=static_cast<char>(c); else if(c<0x800) { out+=static_cast<char>(0xc0|(c>>6)); out+=static_cast<char>(0x80|(c&63)); } else if(c<0x10000) { out+=static_cast<char>(0xe0|(c>>12)); out+=static_cast<char>(0x80|((c>>6)&63)); out+=static_cast<char>(0x80|(c&63)); } else { out+=static_cast<char>(0xf0|(c>>18)); out+=static_cast<char>(0x80|((c>>12)&63)); out+=static_cast<char>(0x80|((c>>6)&63)); out+=static_cast<char>(0x80|(c&63)); } } return out;
}
Text ascii(const std::string &value) { return Text(value.begin(),value.end()); }
Text strip(const Text &input) { size_t begin=0,end=input.size(); while(begin<end && space(input[begin])) ++begin; while(end>begin && space(input[end-1])) --end; return input.substr(begin,end-begin); }
bool final_sigma(const Text &text,size_t at) {
    size_t before=at,after=at+1; while(before && ignorable(text[before-1])) --before; if(!before || !cased(text[before-1])) return false; while(after<text.size() && ignorable(text[after])) ++after; return after==text.size() || !cased(text[after]);
}
Text change_case(const Text &input,bool title) {
    Text out; bool previous=false;
    for(size_t i=0;i<input.size();++i) { const auto c=input[i]; const bool upper=title && !previous; const auto mapping=std::lower_bound(std::begin(tts_unicode::mappings),std::end(tts_unicode::mappings),c,[](const auto &m,auto c) { return m.code<c; });
        if(!upper && c==0x03a3 && final_sigma(input,i)) out+=0x03c2;
        else if(mapping!=std::end(tts_unicode::mappings) && mapping->code==c) { const auto *chars=upper?mapping->title:mapping->lower; const auto count=upper?mapping->title_size:mapping->lower_size; out.append(chars,chars+count); }
        else out+=c;
        previous=cased(c);
    }
    return out;
}
std::string words(unsigned n) {
    static const std::string ones[]={"zero","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen"};
    static const std::string tens[]={"","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"};
    if(n<20) return ones[n]; if(n<100) return tens[n/10]+(n%10?" "+ones[n%10]:""); if(n<1000) return ones[n/100]+" hundred"+(n%100?" "+words(n%100):""); if(n<1000000) return words(n/1000)+" thousand"+(n%1000?" "+words(n%1000):""); return std::to_string(n);
}
std::pair<unsigned,size_t> digits(const Text &text,size_t start,size_t maximum) { unsigned value=0; size_t count=0; while(count<maximum && start+count<text.size()) { const auto d=digit(text[start+count]); if(d<0) break; value=value*10+d; ++count; } return {value,count}; }
Text normalize_times(const Text &text) {
    Text out;
    for(size_t i=0;i<text.size();) { auto [hour,n]=digits(text,i,2); size_t next=i+n;
        if(n && next<text.size() && text[next]==U':') { const auto [minute,m]=digits(text,next+1,2); if(m==2) { next+=3; while(next<text.size() && space(text[next])) ++next; Text period=text.substr(next,2); bool am=period==U"AM" || period==U"am",pm=period==U"PM" || period==U"pm";
            std::string replacement=words(hour)+(minute==0?" o'clock":minute<10?" oh "+words(minute):" "+words(minute)); if(am || pm) { replacement+=am?" A M":" P M"; next+=2; } out+=ascii(replacement); i=next; continue;
        } } out+=text[i++];
    } return out;
}
Text normalize_dates(const Text &text) {
    static const std::string months[]={"","January","February","March","April","May","June","July","August","September","October","November","December"}; Text out;
    for(size_t i=0;i<text.size();) { auto [month,n]=digits(text,i,2); size_t next=i+n;
        if(n && next<text.size() && text[next]==U'/') { auto [day,d]=digits(text,next+1,2); if(d) { next+=1+d; unsigned year=0; bool has_year=false;
            if(next<text.size() && text[next]==U'/') { const auto [y,count]=digits(text,next+1,4); if(count==4) { has_year=true; year=y; next+=5; } }
            if(month>=1 && month<=12) { const auto suffix=day%100>=10 && day%100<=20?"th":day%10==1?"st":day%10==2?"nd":day%10==3?"rd":"th"; auto replacement=months[month]+" "+words(day)+suffix; if(has_year) replacement+=" "+(year>=2000?words(year):words(year/100)+" "+words(year%100)); out+=ascii(replacement); }
            else out+=text.substr(i,next-i); i=next; continue;
        } } out+=text[i++];
    } return out;
}
Text normalize_numbers(const Text &text) { Text out; for(size_t i=0;i<text.size();) { const auto [value,n]=digits(text,i,6); if(n && (!i || !word(text[i-1])) && (i+n==text.size() || !word(text[i+n]))) { out+=ascii(words(value)); i+=n; } else out+=text[i++]; } return out; }
bool sentence_end(char32_t c) { return c==U'.' || c==U'!' || c==U'?' || c==U'。' || c==U'！' || c==U'？'; }
bool punctuation(char32_t c) { return c==U',' || c==U';' || c==U':' || c==U'，' || c==U'；' || c==U'：'; }
std::vector<Text> sentences(const Text &text) { std::vector<Text> out; size_t start=0; for(size_t i=1;i<text.size();) { if(sentence_end(text[i-1]) && space(text[i])) { out.push_back(text.substr(start,i-start)); while(i<text.size() && space(text[i])) ++i; start=i; } else ++i; } out.push_back(text.substr(start)); return out; }
std::vector<Text> force_words(const Text &text,double maximum) { std::vector<Text> out; Text current; for(size_t i=0;i<text.size();) { while(i<text.size() && space(text[i])) ++i; const auto start=i; while(i<text.size() && !space(text[i])) ++i; if(i==start) break; auto word=text.substr(start,i-start); if(!current.empty() && current.size()+word.size()+1>maximum) { out.push_back(strip(current)); current=std::move(word); } else current=strip(current+U" "+word); } if(!current.empty()) out.push_back(strip(current)); return out; }
std::vector<Text> split(const Text &text,double maximum,bool keep_short) {
    maximum=std::max(1.0,maximum); if(keep_short && text.size()<=maximum) return text.empty()?std::vector<Text>{}:std::vector<Text>{text};
    std::vector<Text> chunks,out; Text current;
    for(auto sentence:sentences(text)) { sentence=strip(sentence); if(sentence.empty()) continue; if(!current.empty() && current.size()+sentence.size()+1>maximum) { chunks.push_back(strip(current)); current=std::move(sentence); } else current=strip(current+U" "+sentence); }
    if(!current.empty()) chunks.push_back(strip(current));
    for(const auto &chunk:chunks) { if(chunk.size()<=maximum) { out.push_back(chunk); continue; } std::vector<Text> parts; size_t start=0; for(size_t i=0;i<chunk.size();++i) if(punctuation(chunk[i])) { parts.push_back(chunk.substr(start,i-start)); parts.push_back(chunk.substr(i,1)); start=i+1; } parts.push_back(chunk.substr(start)); Text sub;
        for(auto part:parts) { part=strip(part); if(part.empty()) continue; if(!sub.empty() && sub.size()+part.size()+1>maximum) { out.push_back(strip(sub)); sub=std::move(part); } else sub=strip(sub+U" "+part); }
        if(!sub.empty()) { const auto final=force_words(sub,maximum); out.insert(out.end(),final.begin(),final.end()); }
    }
    if(out.empty() && !text.empty()) out.push_back(text); return out;
}
std::vector<std::string> encoded(const std::vector<Text> &chunks) { std::vector<std::string> out; for(const auto &chunk:chunks) out.push_back(encode(chunk)); return out; }
}
std::string tts_number_to_words(const std::string &decimal) {
    if(decimal.empty()) throw std::invalid_argument("Empty integer"); const bool negative=decimal[0]=='-'; const auto first=negative || decimal[0]=='+'?1u:0u; if(first==decimal.size()) throw std::invalid_argument("Invalid integer"); for(size_t i=first;i<decimal.size();++i) if(decimal[i]<'0' || decimal[i]>'9') throw std::invalid_argument("Invalid integer"); auto start=decimal.find_first_not_of('0',first); if(start==std::string::npos) return "zero"; const auto value=decimal.substr(start); return (negative?"negative ":"")+(value.size()>6?value:words(static_cast<unsigned>(std::stoul(value))));
}
std::string tts_cache_normalize_text(const std::string &value) {
    const auto text=change_case(decode(value),false); Text out; bool pending=false;
    for(auto c:text) { if(space(c)) { pending=!out.empty(); continue; } if(pending) out+=U' '; pending=false; out+=c; } return encode(out);
}
std::string tts_normalize_english(const std::string &text) { return encode(normalize_numbers(normalize_dates(normalize_times(decode(text))))); }
std::string tts_clean_text(const std::string &text) {
    const auto input=decode(text); Text collapsed,controls,out;
    for(auto c:input) if(!sentence_end(c) || collapsed.empty() || collapsed.back()!=c) collapsed+=c;
    for(auto c:collapsed) if(!(c<=8 || c==11 || c==12 || (c>=14 && c<=31) || c==127)) controls+=c;
    for(auto c:controls) { if(space(c)) { if(out.empty() || out.back()!=U' ') out+=U' '; } else out+=c; }
    out.erase(std::remove(out.begin(),out.end(),U'#'),out.end()); return encode(strip(out));
}
bool tts_contains_cyrillic(const std::string &text) { for(auto c:decode(text)) if(c>=0x0400 && c<=0x04ff) return true; return false; }
std::string tts_resolve_language(const std::string &language,const std::string &text) {
    static const std::map<std::string,std::string> aliases={{"","Auto"},{"auto","Auto"},{"automatic","Auto"},{"multilingual","Auto"},{"multi","Auto"},{"en","English"},{"eng","English"},{"english","English"},{"ru","Russian"},{"rus","Russian"},{"russian","Russian"},{"zh","Chinese"},{"cn","Chinese"},{"chinese","Chinese"},{"ja","Japanese"},{"japanese","Japanese"},{"ko","Korean"},{"korean","Korean"},{"de","German"},{"german","German"},{"fr","French"},{"french","French"},{"pt","Portuguese"},{"portuguese","Portuguese"},{"es","Spanish"},{"spanish","Spanish"},{"it","Italian"},{"italian","Italian"}};
    const auto requested=strip(decode(language)); const auto found=aliases.find(encode(change_case(requested,false))); auto result=found==aliases.end()?encode(change_case(requested,true)):found->second; if(result=="English" && tts_contains_cyrillic(text)) result="Auto"; return result.empty()?"Auto":result;
}
std::string tts_prepare_text(const std::string &text,const std::string &language) { return tts_clean_text(language=="English" && !tts_contains_cyrillic(text)?tts_normalize_english(text):text); }
std::string tts_safe_file_stem(const std::optional<std::string> &value,const std::string &fallback) {
    auto name=strip(decode(value && !value->empty()?*value:fallback)); std::replace(name.begin(),name.end(),U'\\',U'/'); const auto slash=name.rfind(U'/'); if(slash!=Text::npos) name=name.substr(slash+1); if(name.size()>=4 && change_case(name.substr(name.size()-4),false)==U".wav") name.resize(name.size()-4);
    Text out; for(auto c:name) if((c>=U'a' && c<=U'z') || (c>=U'A' && c<=U'Z') || (c>=U'0' && c<=U'9') || c==U'_' || c==U'-') out+=c; if(out.empty()) out=decode(fallback); return encode(out.substr(0,100));
}
std::vector<std::string> tts_split_text(const std::string &text,double maximum,bool keep_short) { return encoded(split(decode(text),maximum,keep_short)); }
std::vector<std::string> tts_split_stream_text(const std::string &value,double maximum,std::optional<double> first,std::optional<double> minimum) {
    maximum=std::max(1.0,maximum); const auto first_target=std::min(maximum,std::max(1.0,first.value_or(120))); const auto min_first=std::min(first_target,std::max(1.0,minimum.value_or(50))); const auto text=decode(value); std::vector<Text> chunks; Text current; size_t count=0;
    auto flush=[&] { if(!current.empty()) chunks.push_back(strip(current)); current.clear(); count=0; };
    for(auto sentence:sentences(text)) { sentence=strip(sentence); if(sentence.empty()) continue; if(chunks.empty() && !current.empty() && current.size()>=first_target) flush();
        if(sentence.size()>maximum) { flush(); const auto sub=split(sentence,maximum,true); chunks.insert(chunks.end(),sub.begin(),sub.end()); continue; }
        const auto active=chunks.empty()?first_target:maximum; if(!current.empty() && current.size()+sentence.size()+1>active) flush(); current=strip(current+U" "+sentence); ++count;
        if(chunks.empty() && current.size()>=min_first && (count>1 || current.size()>=first_target)) flush();
    }
    flush(); if(chunks.empty() && !text.empty()) chunks.push_back(text); return encoded(chunks);
}
}
