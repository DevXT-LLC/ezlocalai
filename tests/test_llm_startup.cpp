#include "ezlocalai/llm_startup.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include <arpa/inet.h>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>
namespace fs=std::filesystem;
namespace {
void expect(bool ok,const std::string &message){if(!ok)throw std::runtime_error(message);}
std::string read(const fs::path &path){std::ifstream in(path);return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};}
std::vector<std::string> split(const std::string &s,char delimiter='\t'){std::vector<std::string> out;size_t begin=0;for(;;){const auto end=s.find(delimiter,begin);out.push_back(s.substr(begin,end==std::string::npos?end:end-begin));if(end==std::string::npos)return out;begin=end+1;}}
std::string hex(const std::string &s){std::ostringstream out;for(unsigned char c:s)out<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(c);return out.str();}
std::string unhex(const std::string &s){std::string out;for(size_t i=0;i<s.size();i+=2)out+=static_cast<char>(std::stoi(s.substr(i,2),nullptr,16));return out;}
std::string sha256(const std::string &s){unsigned char hash[32];SHA256(reinterpret_cast<const unsigned char *>(s.data()),s.size(),hash);return hex({reinterpret_cast<char *>(hash),32});}
std::optional<std::string> optional(const std::string&s){return s=="N"?std::nullopt:std::optional<std::string>(unhex(s.substr(1)));}
size_t reference_cases(const fs::path &root,const fs::path &fixture){size_t count=0;std::istringstream in(read(fixture));for(std::string line;std::getline(in,line);){if(line.empty())continue;if(line[0]=='#'){const auto at=line.find(" sha256=");if(at!=std::string::npos)expect(sha256(read(root/line.substr(2,at-2)))==line.substr(at+8),"startup source hash");continue;}++count;const auto c=split(line);const auto actual=ezlocalai::llm_startup_failure(unhex(c[1]),unhex(c[2]),optional(c[3])),expected=optional(c[4]);expect(actual==expected,"startup case "+c[0]+" got "+actual.value_or("success")+" expected "+expected.value_or("success"));}expect(count==741,"all startup source cases");return count;}
size_t occurrences(const std::string &s,const std::string &needle){size_t count=0,pos=0;while((pos=s.find(needle,pos))!=std::string::npos){++count;pos+=needle.size();}return count;}
int available_port(){const int fd=socket(AF_INET,SOCK_STREAM,0);sockaddr_in a{};a.sin_family=AF_INET;a.sin_addr.s_addr=htonl(INADDR_LOOPBACK);expect(!bind(fd,reinterpret_cast<sockaddr *>(&a),sizeof(a)),"fixture port");socklen_t n=sizeof(a);getsockname(fd,reinterpret_cast<sockaddr *>(&a),&n);close(fd);return ntohs(a.sin_port);}
void environment(){for(const auto &setting:ezlocalai::environment_defaults)unsetenv(setting.first.data());for(const auto key:{"LLM_FIXTURE_PROBE_RESULT","LLM_FIXTURE_PROBE_STATUS","LLM_FIXTURE_PROBE_BLOCK","LLM_FIXTURE_PROBE_DROP","LLM_FIXTURE_PROBE_MAX_LAYERS"})unsetenv(key);setenv("EZLOCALAI_DOTENV_PATH","",1);setenv("GPU_LAYERS","0",1);setenv("LLAMA_CPP_CONTEXT","128",1);setenv("LLAMA_CPP_SERVER_BIN",TEST_EMBEDDING_FIXTURE_BIN,1);setenv("LLAMA_CPP_START_TIMEOUT","2",1);}
struct Fixture {
    fs::path dir;int port;
    Fixture(const std::string &name):dir(fs::temp_directory_path()/("ezl-startup-"+std::to_string(getpid())+"-"+name)),port(available_port()) {fs::create_directories(dir);const auto model=dir/"model.gguf";std::ofstream(model)<<"GGUF fixture";setenv("LLAMA_CPP_MODEL_PATH",model.c_str(),1);setenv("EMBEDDING_FIXTURE_DIR",dir.c_str(),1);setenv("LLAMA_CPP_SERVER_PORT",std::to_string(port).c_str(),1);}
    fs::path path(const std::string &prefix)const{return dir/(prefix+"-"+std::to_string(port));}
    void clean(){std::istringstream pids(read(path("launch")));for(int pid;pids>>pid;)expect(kill(pid,0)!=0,"startup child reaped");fs::remove_all(dir);}
};
void lifecycle(bool vision){
    environment();Fixture fixture(vision?"vision":"text");if(vision){const auto projector=fixture.dir/"mmproj.gguf";std::ofstream(projector)<<"projector";setenv("LLAMA_CPP_MMPROJ_PATH",projector.c_str(),1);}
    {
        ezlocalai::LlamaCppInferenceBackend backend("repo/model");setenv("LLM_FIXTURE_PROBE_RESULT",R"({"error":null})",1);setenv("LLM_FIXTURE_PROBE_BLOCK","1",1);
        auto request=std::async(std::launch::async,[&]{return backend.chat(R"({"messages":[]})");});
        struct Release{fs::path p;~Release(){std::ofstream(p)<<"release";}} release{fixture.path("release-probe")};
        for(int i=0;i<300&&!fs::exists(fixture.path("probe"));++i)std::this_thread::sleep_for(std::chrono::milliseconds(10));
        expect(fs::exists(fixture.path("probe"))&&!fs::exists(fixture.path("seen")),"startup inference precedes user generation");expect(request.wait_for(std::chrono::milliseconds(40))==std::future_status::timeout,"startup result gates user generation");
        std::ofstream(fixture.path("release-probe"))<<"release";
        const auto failure=request.get();expect(failure.find("LLM server test failed: {'error': None}")!=std::string::npos,"falsey startup error fails load");expect(backend.loaded_models_json()=="{}"&&!fs::exists(fixture.path("seen")),"failed probe never publishes residency or invokes user generation");expect(backend.model_lifecycle_json().find("\"count\":1,\"failures\":1")!=std::string::npos,"probe failure recorded as load failure");
        unsetenv("LLM_FIXTURE_PROBE_RESULT");unsetenv("LLM_FIXTURE_PROBE_BLOCK");
        expect(backend.chat(R"({"messages":[]})").find("choices")!=std::string::npos,"retry after failed probe succeeds");expect(backend.loaded_models_json().find(vision?"\"vision_llm\":" : "\"llm\":")!=std::string::npos,"verified engine publishes correct residency");const auto initial_probes=occurrences(read(fixture.path("probe")),"POST /v1/completions");expect(initial_probes>2,"environment CPU setting still allows isolated fit probes");
        backend.completion(R"({"prompt":"ordinary"})");expect(occurrences(read(fixture.path("probe")),"POST /v1/completions")==initial_probes,"resident reuse skips startup probe");
        auto restore=backend.unload_llms_for_media();setenv("LLM_FIXTURE_PROBE_RESULT",R"({"error":false})",1);restore(true);expect(backend.loaded_models_json()=="{}","failed restore probe leaves model absent");unsetenv("LLM_FIXTURE_PROBE_RESULT");restore(true);expect(backend.model_lifecycle_json().find("\"count\":4,\"failures\":2")!=std::string::npos,"restore probe failure and recovery recorded");expect(occurrences(read(fixture.path("probe")),"POST /v1/completions")==initial_probes+2,"known CPU placement fails directly and each restoration verifies native inference");
    }
    fixture.clean();
}
void gpu_probe_recovery(){
    environment();Fixture fixture("gpu-retry");setenv("GPU_LAYERS","12",1);setenv("LLAMA_CPP_TOTAL_LAYERS","12",1);setenv("LLM_FIXTURE_PROBE_MAX_LAYERS","6",1);
    {
        ezlocalai::LlamaCppInferenceBackend backend("model");expect(backend.chat(R"({"messages":[]})").find("choices")!=std::string::npos,"startup inference memory error reaches retry and recovers");
        expect(occurrences(read(fixture.path("probe")),"POST /v1/completions")>=2,"each retry verifies native inference");
        expect(backend.model_lifecycle_json().find("\"count\":1,\"failures\":0")!=std::string::npos,"complete recovered startup records one successful load");
        std::istringstream pids(read(fixture.path("launch")));std::vector<int> launched;for(int pid;pids>>pid;)launched.push_back(pid);expect(launched.size()>=2,"failed probe launches replaced");for(size_t i=0;i+1<launched.size();++i)expect(kill(launched[i],0)!=0,"failed startup probe child reaped before recovery");
    }
    fixture.clean();
}

void result_shapes(){
    size_t index=0;
    for(const auto &scenario:std::vector<std::pair<std::string,bool>>{{"null",true},{"[]",true},{"false",true},{R"({"code":400,"message":"top-level status"})",true},{R"({"error":[]})",false},{R"({"error":0})",false},{"malformed",false}}){
        environment();Fixture fixture("shape"+std::to_string(index++));setenv("LLM_FIXTURE_PROBE_RESULT",scenario.first.c_str(),1);setenv("LLM_FIXTURE_PROBE_STATUS","400",1);
        {ezlocalai::LlamaCppInferenceBackend backend("model");const auto result=backend.chat(R"({"messages":[]})");expect((result.find("choices")!=std::string::npos)==scenario.second,"native startup result shape acceptance");expect((backend.loaded_models_json()!="{}")==scenario.second,"native startup result shape residency");}fixture.clean();
    }
    environment();Fixture fixture("drop");setenv("LLM_FIXTURE_PROBE_DROP","1",1);{ezlocalai::LlamaCppInferenceBackend backend("model");expect(backend.chat(R"({"messages":[]})").find("Failed to initialize LLM server for model")!=std::string::npos&&backend.loaded_models_json()=="{}","startup transport failure remains unloaded");}fixture.clean();
}
}
int main(int argc,char **argv){try{expect(argc==3,"reference root and fixture required");const auto count=reference_cases(argv[1],argv[2]);lifecycle(false);lifecycle(true);result_shapes();gpu_probe_recovery();std::cout<<"Verified "<<count<<" source startup cases and managed probe lifetimes\n";return 0;}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
