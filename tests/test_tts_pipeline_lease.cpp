#include "tts_service_test.hpp"
int main(int argc,char **argv){try{
    expect(argc==3,"reference and fixture paths");gzFile file=gzopen(argv[2],"rb");expect(file,"open service fixtures");std::string raw;char buffer[65536];int size;while((size=gzread(file,buffer,sizeof(buffer)))>0)raw.append(buffer,size);expect(size==0&&gzclose(file)==Z_OK,"complete service fixture");std::vector<Json> cases;std::istringstream input(raw);for(std::string line;std::getline(input,line);)if(!line.empty())cases.push_back(Json::parse(line));
    expect(hash(read(fs::path(argv[1])/"Pipes.py"))==cases[0].at("source_sha256"),"unchanged source endpoints, pool and guard");expect(cases.size()-1==cases[0].at("cases"),"all service cases");size_t actions=0;
    for(size_t row=1;row<cases.size();++row){Environment env(cases[row].at("config"));actions+=run_service_scenario(env,cases[row],"service "+std::to_string(row));}
    std::cout<<cases.size()-1<<" TTS pipeline service source scenarios and "<<actions<<" operations match\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
