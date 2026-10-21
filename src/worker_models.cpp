#include "ezlocalai/worker_models.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/ezlocalai_c.h"
#include <unicode/unistr.h>
#include <unicode/locid.h>
#include <filesystem>
#include <regex>
#include <sstream>
namespace ezlocalai {
namespace {
std::string lower(std::string value) { for(auto &c:value)c=std::tolower(static_cast<unsigned char>(c));return value; }
std::string setting(const char *name) { return strip_config_value(ezl_getenv_default(name)); }
std::vector<std::string> list(const std::string &raw) {
    std::vector<std::string> out;std::istringstream in(raw);std::string item;
    while(std::getline(in,item,',')){item=strip_config_value(item);if(!item.empty())out.push_back(item);}return out;
}
}
std::string worker_model_file_quant(const std::string &path) {
    auto name=icu::UnicodeString::fromUTF8(std::filesystem::path(path).filename().string());name.toUpper(icu::Locale::getRoot());
    std::string upper;name.toUTF8String(upper);
    static const std::regex pattern(R"((I?Q[0-9][A-Z0-9_]*?)(?=\.GGUF|$|-))");
    std::smatch match;return std::regex_search(upper,match,pattern)?match[1].str():"";
}
std::map<std::string,std::string> worker_model_quants(std::map<std::string,std::string> result) {
    const auto server=lower(setting("EMBEDDING_SERVER"));const std::string raw_enabled=ezl_getenv_default("EMBEDDING_ENABLED");
    const auto enabled=strip_config_value(raw_enabled.empty()?"true":raw_enabled);
    const auto model=setting("EMBEDDING_MODEL");
    if((lower(enabled)=="true")&&!model.empty()&&server.rfind("http://",0)!=0&&server.rfind("https://",0)!=0){
        // Python parses embedding context before adding its quant metadata.
        if(parse_configuration_integer(ezl_getenv_default("EMBEDDING_CONTEXT_LENGTH"))){
            const std::string quant=ezl_getenv_default("EMBEDDING_QUANT_TYPE");
            result.emplace(model,quant);const auto alias=setting("EMBEDDING_MODEL_ALIAS");if(!alias.empty())result.emplace(alias,quant);
        }
    }
    const auto models=list(ezl_getenv_default("DEFAULT_MODEL")),quants=list(ezl_getenv_default("QUANT_TYPE"));
    if(!quants.empty())for(size_t i=0;i<models.size();++i)result.emplace(models[i],i<quants.size()?quants[i]:quants.front());
    return result;
}
}
