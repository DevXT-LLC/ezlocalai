#include "ezlocalai/llm_catalog.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>
#include <sstream>
namespace fs=std::filesystem;
namespace {
void expect(bool ok,const std::string &message){if(!ok)throw std::runtime_error(message);}
std::string read(const fs::path &path){std::ifstream in(path);return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};}
std::vector<std::string> split(const std::string &s,char delimiter='\t'){std::vector<std::string> out;size_t begin=0;for(;;){const auto end=s.find(delimiter,begin);out.push_back(s.substr(begin,end==std::string::npos?end:end-begin));if(end==std::string::npos)return out;begin=end+1;}}
std::string hex(const std::string &s){std::ostringstream out;for(unsigned char c:s)out<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(c);return out.str();}
std::string unhex(const std::string &s){std::string out;for(size_t i=0;i<s.size();i+=2)out+=static_cast<char>(std::stoi(s.substr(i,2),nullptr,16));return out;}
std::string sha256(const std::string &s){unsigned char hash[32];SHA256(reinterpret_cast<const unsigned char *>(s.data()),s.size(),hash);return hex({reinterpret_cast<char *>(hash),32});}
std::string optional(const std::optional<std::string>&s){return s?"S"+hex(*s):"N";}
std::string snapshot(const ezlocalai::LlmCatalog &catalog,const std::string &expected){
    std::ostringstream out;const auto line=[&](const std::string &s){if(out.tellp()>0)out<<'\n';out<<s;};
    for(const auto &id:catalog.available)line("A\t"+hex(id));
    for(const auto &p:catalog.sources)line("S\t"+hex(p.first)+"\t"+hex(p.second));
    for(const auto &p:catalog.groups){std::string ids;for(const auto &id:p.second){if(!ids.empty())ids+=',';ids+=hex(id);}line("G\t"+hex(p.first)+"\t"+ids);}
    for(const auto &p:catalog.configs){const auto &c=p.second;std::string tensor="N";if(c.tensor_split){for(size_t i=0;i<128;++i)if((*c.tensor_split)[i]==1)tensor=std::to_string(i);}line("C\t"+hex(p.first)+"\t"+std::to_string(c.main_gpu)+"\t"+std::to_string(c.context)+"\t"+std::to_string(c.parallel)+"\t"+optional(c.quant)+"\t"+tensor+"\t"+c.speculative_type);}
    for(const auto &id:catalog.automatic)line("H\t"+hex(id));
    for(const auto &name:catalog.public_models())line("M\t"+hex(name));
    for(const auto &entry:split(expected,'\n'))if(entry.compare(0,2,"Q\t")==0){const auto fields=split(entry);const auto request=unhex(fields[1]);line("Q\t"+fields[1]+"\t"+optional(catalog.resolve(request))+"\t"+optional(catalog.slot_model(request)));}
    return out.str();
}
size_t run(const fs::path &root,const fs::path &fixture){size_t count=0;std::istringstream in(read(fixture));for(std::string line;std::getline(in,line);){if(line.empty())continue;if(line[0]=='#'){const auto at=line.find(" sha256=");if(at!=std::string::npos)expect(sha256(read(root/line.substr(2,at-2)))==line.substr(at+8),"catalog source hash");continue;}++count;const auto c=split(line);ezlocalai::LlmCatalogSettings settings{unhex(c[1]),unhex(c[2]),unhex(c[3]),unhex(c[4]),unhex(c[5]),unhex(c[6]),std::stoi(c[7])};const auto expected=unhex(c[8]);std::string actual;try{actual=snapshot(ezlocalai::parse_llm_catalog(settings),expected);}catch(const ezlocalai::LlmCatalogError &e){actual="E\t"+e.type;}expect(actual==expected,"catalog case "+c[0]+" got "+actual+" expected "+expected);}expect(count==2216,"all source catalog cases");return count;}
}
int main(int argc,char **argv){try{expect(argc==3,"source root and fixture required");std::cout<<"Verified "<<run(argv[1],argv[2])<<" source model catalog cases\n";return 0;}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
