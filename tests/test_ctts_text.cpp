#include "ezlocalai/tts_text.hpp"
#include "ezlocalai/ezlocalai_c.h"
#include "whisper_transcription_test.hpp"
#include <charconv>
#include <climits>
using namespace transcription_test;
std::vector<Json> compressed_records(const char *path) {
    const auto file=gzopen(path,"rb"); expect(file,"open compressed text fixtures"); struct Close { gzFile file; ~Close() { gzclose(file); } } close{file}; std::vector<Json> out; std::string pending; char buffer[65536];
    for(;;) { const int count=gzread(file,buffer,sizeof(buffer)); expect(count>=0,"read compressed fixture"); if(!count) break; pending.append(buffer,count); size_t begin=0,end=0; while((end=pending.find('\n',begin))!=std::string::npos) { if(end>begin) out.push_back(Json::parse(pending.substr(begin,end-begin))); begin=end+1; } pending.erase(0,begin); }
    expect(pending.empty(),"complete fixture lines"); return out;
}
int main(int argc,char **argv) { try {
    expect(argc==3,"source and CTTS text fixtures"); const auto cases=compressed_records(argv[2]); const auto &header=cases[0]; expect(hash(read(fs::path(argv[1])/"ezlocalai/CTTS.py"))==header.at("source_sha256"),"unchanged CTTS source"); expect(hash(read(fs::path(__FILE__).parent_path().parent_path()/"include/ezlocalai/tts_unicode.hpp"))==header.at("unicode_sha256"),"pinned CPython Unicode data"); size_t errors=0,c_abi=0; std::map<std::string,size_t> counts;
    for(size_t i=1;i<cases.size();++i) { const auto &row=cases[i]; const auto kind=row.at("kind").get<std::string>(); ++counts[kind]; Json actual; bool abi_string=false; std::string text,decimal; long n=0;
        try {
            if(kind=="number") { decimal=row.at("decimal"); actual=tts_number_to_words(decimal); const auto result=std::from_chars(decimal.data(),decimal.data()+decimal.size(),n); abi_string=result.ec==std::errc{} && result.ptr==decimal.data()+decimal.size(); }
            else if(kind=="normalize") { text=row.at("text"); actual=tts_normalize_english(text); abi_string=text.find('\0')==std::string::npos; }
            else if(kind=="clean") actual=tts_clean_text(row.at("text"));
            else if(kind=="chunks") actual=tts_split_text(row.at("text"),row.at("maximum"),row.at("keep_short"));
            else if(kind=="stream") { std::optional<double> first,minimum; if(!row.at("first").is_null()) first=row.at("first"); if(!row.at("minimum").is_null()) minimum=row.at("minimum"); actual=tts_split_stream_text(row.at("text"),row.at("maximum"),first,minimum); }
            else if(kind=="language") actual=tts_resolve_language(row.at("language"),row.at("text"));
            else if(kind=="prepare") actual=tts_prepare_text(row.at("text"),row.at("language"));
            else if(kind=="stem") { std::optional<std::string> value; if(!row.at("value").is_null()) value=row.at("value"); actual=tts_safe_file_stem(value,row.at("default")); }
            else throw std::runtime_error("unknown text operation");
            compare_json(actual,row.at("result"),kind+" "+std::to_string(i));
            if(abi_string) { std::vector<char> output(actual.get<std::string>().size()+1); const int rc=kind=="number"?ezl_number_to_words(n,output.data(),output.size()):ezl_normalize_text_for_tts(text.c_str(),output.data(),output.size()); expect(rc==0,"C ABI text conversion"); compare_json(std::string(output.data()),row.at("result"),"C ABI "+kind); ++c_abi; }
            if(kind=="chunks" && row.at("keep_short")==true && row.at("maximum").get<double>()>=0 && std::floor(row.at("maximum").get<double>())==row.at("maximum").get<double>()) { text=row.at("text"); if(text.find('\0')==std::string::npos) { ezl_string_list list{}; expect(ezl_split_text_into_chunks(text.c_str(),row.at("maximum").get<size_t>(),&list)==0,"C ABI chunks"); Json chunks=Json::array(); for(size_t j=0;j<list.count;++j) chunks.push_back(list.items[j]); ezl_string_list_free(&list); compare_json(chunks,row.at("result"),"C ABI chunks"); ++c_abi; } }
        } catch(const std::exception &e) { if(errors++<30) std::cerr<<e.what()<<" input="<<row.dump()<<'\n'; }
    }
    compare_json(counts,header.at("counts"),"all text case groups"); expect(cases.size()-1==header.at("cases"),"complete source fixture"); expect(!errors,std::to_string(errors)+" text cases differ"); std::cout<<cases.size()-1<<" unchanged-source CTTS text cases and "<<c_abi<<" C ABI comparisons match\n"; return 0;
} catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; } }
