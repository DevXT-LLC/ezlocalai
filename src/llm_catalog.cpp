#include "ezlocalai/llm_catalog.hpp"
#include "ezlocalai/configuration.hpp"
#include <algorithm>
#include <charconv>

namespace ezlocalai {
namespace {
std::vector<std::string> csv(const std::string &value) {
    std::vector<std::string> out; size_t begin = 0;
    for (;;) { const auto end = value.find(',',begin); out.push_back(strip_config_value(value.substr(begin,end == std::string::npos ? end : end-begin))); if(end == std::string::npos)return out; begin=end+1; }
}
int64_t integer(const std::string &raw) {
    const auto value=normalize_decimal_digits(strip_config_value(raw));
    std::string normalized; bool digit=false; size_t begin=0;
    if(!value.empty() && (value[0]=='+' || value[0]=='-')) { if(value[0]=='-')normalized+='-'; begin=1; }
    for(size_t i=begin;i<value.size();++i) {
        const char c=value[i];
        if(c>='0' && c<='9') { normalized+=c; digit=true; }
        else if(c=='_' && digit)digit=false;
        else throw LlmCatalogError("ValueError","invalid integer configuration: "+raw);
    }
    if(!digit)throw LlmCatalogError("ValueError","invalid integer configuration: "+raw);
    int64_t result=0; const auto parsed=std::from_chars(normalized.data(),normalized.data()+normalized.size(),result);
    if(parsed.ec!=std::errc{} || parsed.ptr!=normalized.data()+normalized.size())throw LlmCatalogError("OverflowError","integer configuration exceeds native range");
    return result;
}
std::string short_name(const std::string &value) {const auto slash=value.rfind('/');return slash==std::string::npos ? value : value.substr(slash+1);}
void add_group(LlmCatalog &catalog,const std::string &source,const std::string &id) {
    for(auto &group:catalog.groups)if(group.first==source){group.second.push_back(id);return;}
    catalog.groups.push_back({source,{id}});
}
}
std::string llm_speculative_backend(const std::string &model,const std::string &setting) {
    const auto name=llm_unicode_lower(model);bool qwen=false;size_t pos=0;
    while((pos=name.find("qwen3.8-27b",pos))!=std::string::npos) {pos+=11;if(pos==name.size() || (pos+1==name.size() && name[pos]=='\n') || std::string("-_./").find(name[pos])!=std::string::npos){qwen=true;break;}}
    const bool mtp=qwen || name.find("-mtp")!=std::string::npos;
    auto backend=llm_unicode_lower(strip_config_value(setting));
    if(backend=="draft-dflash")backend="dflash2";else if(backend=="draft-mtp")backend="mtp";else if(backend=="off")backend="none";
    if(backend.empty() || backend=="auto")return mtp ? "mtp" : "none";
    if(backend!="dflash2" && backend!="mtp" && backend!="none")throw LlmCatalogError("ValueError","LLM_SPECULATIVE_TYPE must be auto, dflash2, mtp or none");
    if(backend=="dflash2" && !qwen)return mtp ? "mtp" : "none";
    if(backend=="mtp" && !mtp)return "none";
    return backend;
}
LlmCatalog parse_llm_catalog(const LlmCatalogSettings &settings) {
    LlmCatalog catalog;
    if(llm_unicode_lower(settings.models)=="none")return catalog;
    std::map<std::string,size_t> counts;
    for(auto name:csv(settings.models)) {
        const auto at=name.rfind('@');if(at!=std::string::npos)name.resize(at);
        if(name.empty())continue;
        const auto count=++counts[name];const auto id=count==1 ? name : name+"#"+std::to_string(count);
        catalog.available.push_back(id);catalog.sources[id]=name;add_group(catalog,name,id);
    }
    const auto gpus=csv(settings.main_gpus),contexts=csv(settings.contexts),parallels=csv(settings.parallels),quants=csv(settings.quants);
    const auto originals=catalog.available;
    for(size_t i=0;i<originals.size();++i) {
        const auto &id=originals[i];LlmModelConfiguration config;
        config.main_gpu=integer(gpus[std::min(i,gpus.size()-1)]);
        config.context=integer(contexts[std::min(i,contexts.size()-1)]);
        config.parallel=integer(parallels[std::min(i,parallels.size()-1)]);
        const auto quant=quants[std::min(i,quants.size()-1)];if(!quant.empty())config.quant=quant;
        const auto source=catalog.source(id);config.speculative_type=llm_speculative_backend(source,settings.speculative_type);config.original_index=i;
        int64_t instances=1;if(config.speculative_type!="none"){instances=std::max<int64_t>(1,config.parallel);config.parallel=1;}
        if(settings.gpu_count>1 && (catalog.available.size()>1 || instances>1)) {
            if(config.main_gpu < -128 || config.main_gpu >= 128)throw LlmCatalogError("IndexError","list assignment index out of range");
            config.tensor_split=std::array<double,128>{};(*config.tensor_split)[config.main_gpu<0 ? config.main_gpu+128 : config.main_gpu]=1;
        }
        catalog.configs[id]=config;
        for(int64_t k=1;k<instances;++k) {
            size_t suffix=2;std::string replica;
            do{replica=source+"#"+std::to_string(suffix++);}while(catalog.sources.count(replica));
            catalog.automatic.insert(replica);catalog.available.push_back(replica);catalog.sources[replica]=source;add_group(catalog,source,replica);catalog.configs[replica]=config;
        }
    }
    return catalog;
}
std::string LlmCatalog::source(const std::string &id) const {const auto found=sources.find(id);return found==sources.end() ? id : found->second;}
std::optional<std::string> LlmCatalog::resolve(const std::string &request) const {
    if(request.empty())return std::nullopt;const auto full=llm_unicode_lower(request),shortened=llm_unicode_lower(short_name(request));
    for(const auto &group:groups)if(llm_unicode_lower(group.first)==full || llm_unicode_lower(short_name(group.first))==shortened)return group.second.empty() ? group.first : group.second.front();
    for(const auto &id:available)if(llm_unicode_lower(id)==full || llm_unicode_lower(short_name(id))==shortened)return id;
    return std::nullopt;
}
std::optional<std::string> LlmCatalog::slot_model(const std::string &request) const {const auto id=resolve(request);return id ? source(*id) : available.empty() ? std::nullopt : std::optional<std::string>(source(available.front()));}
std::vector<std::string> LlmCatalog::public_models() const {std::vector<std::string> out;std::set<std::string> seen;for(const auto &id:available){const auto name=source(id);if(seen.insert(name).second)out.push_back(name);}return out;}
std::vector<LlmReplica> LlmCatalog::replicas() const {std::vector<LlmReplica> out;for(const auto &id:available){const auto &c=configs.at(id);out.push_back({id,source(id),c.parallel,c.context,std::nullopt,c.speculative_type!="none",false});}return out;}
}
