#include "pipeline/ltx_audio.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("fixture required");std::ifstream in(argv[1]);nlohmann::json data;in>>data;
 for(const auto &row:data.at("cases"))if(sd::ltx_audio_latents(row.at("frames"),row.at("fps"))!=row.at("audio_latents"))throw std::runtime_error("audio latent count mismatch: "+row.dump());
 if(data.at("cases").size()!=196)throw std::runtime_error("all audio cases required");
 std::cout<<"Verified 196 installed-source audio counts, including halfway rounding\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
