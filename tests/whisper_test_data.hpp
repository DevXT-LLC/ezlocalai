#pragma once
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
#include <iomanip>
#include <openssl/sha.h>
#include <zlib.h>
namespace whisper_test {
namespace fs=std::filesystem;
inline void expect(bool value,const std::string &message){if(!value)throw std::runtime_error(message);}
inline std::string read(const fs::path &path){std::ifstream in(path,std::ios::binary);expect(bool(in),"read "+path.string());return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};}
inline std::string hash(const std::string &value){unsigned char digest[32];SHA256(reinterpret_cast<const unsigned char *>(value.data()),value.size(),digest);std::ostringstream out;for(auto byte:digest)out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(byte);return out.str();}
inline std::string unhex(const std::string &value){std::string out;for(size_t i=0;i<value.size();i+=2)out+=static_cast<char>(std::stoi(value.substr(i,2),nullptr,16));return out;}
inline std::vector<std::vector<std::string>> rows(const fs::path &path){std::istringstream input(read(path));std::vector<std::vector<std::string>> out;for(std::string line;std::getline(input,line);){if(line.empty()||line[0]=='#')continue;std::vector<std::string> fields;for(size_t at=0;;){const auto end=line.find('\t',at);const auto value=line.substr(at,end==std::string::npos?end:end-at);fields.push_back(fields.empty()?value:unhex(value));if(end==std::string::npos)break;at=end+1;}out.push_back(std::move(fields));}return out;}
inline std::vector<float> floats(const fs::path &path){const auto raw=read(path);expect(raw.size()%4==0,"float data shape");std::vector<float> out(raw.size()/4);std::memcpy(out.data(),raw.data(),raw.size());return out;}
inline std::vector<float> packed(const fs::path &path,size_t size){const auto raw=read(path);std::vector<float> out(size+1);uLongf length=out.size()*4;expect(uncompress(reinterpret_cast<Bytef *>(out.data()),&length,reinterpret_cast<const Bytef *>(raw.data()),raw.size())==Z_OK&&length==size*4,"packed float data "+path.string());out.resize(size);return out;}
template<class Number>std::vector<Number> integers(std::string value){for(auto &c:value)if(c=='['||c==']'||c==',')c=' ';std::istringstream input(value);std::vector<Number> out;for(Number n;input>>n;)out.push_back(n);return out;}
}
