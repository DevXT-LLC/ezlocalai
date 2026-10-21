#pragma once
#include <map>
#include <string>
namespace ezlocalai {
std::string worker_model_file_quant(const std::string &path);
std::map<std::string,std::string> worker_model_quants(std::map<std::string,std::string> loaded);
}
