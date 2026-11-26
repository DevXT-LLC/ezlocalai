#include "ezlocalai/llm_request.hpp"
#include "ezlocalai/llm_parameters.hpp"
#include "ezlocalai/engine_revision.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include <arpa/inet.h>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <sstream>
#include <unistd.h>
namespace fs=std::filesystem;
namespace {
void expect(bool value,const std::string &message){if(!value)throw std::runtime_error(message);}
std::string read(const fs::path &path){std::ifstream in(path);return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};}
std::string hash(const std::string &value){unsigned char digest[32];SHA256(reinterpret_cast<const unsigned char *>(value.data()),value.size(),digest);std::ostringstream out;for(auto c:digest)out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(c);return out.str();}
std::vector<std::string> split(const std::string &value){std::vector<std::string> out;for(size_t at=0;;){const auto end=value.find('\t',at);out.push_back(value.substr(at,end==std::string::npos?end:end-at));if(end==std::string::npos)return out;at=end+1;}}
std::string unhex(const std::string &value){std::string out;for(size_t i=0;i<value.size();i+=2)out+=static_cast<char>(std::stoi(value.substr(i,2),nullptr,16));return out;}
size_t policy(const fs::path &reference,const fs::path &fixture){
    std::istringstream in(read(fixture));size_t count=0;
    for(std::string line;std::getline(in,line);){if(line.empty())continue;
        if(line.front()=='#'){if(line.rfind("# binding",0)==0){expect(line.find(ezlocalai::xllamacpp_reference_revision)!=std::string::npos,"binding revision changed");continue;}const auto at=line.find(" sha256=");if(at!=std::string::npos)expect(hash(read(reference/line.substr(2,at-2)))==line.substr(at+8),"request reference changed: "+line);continue;}
        auto c=split(line);for(size_t i=1;i<c.size();++i)c[i]=unhex(c[i]);++count;
        if(c[0]=="K")expect(ezlocalai::llm_qwen38_model(c[1])==(c[2]=="true")&&ezlocalai::llm_minicpm5_model(c[1])==(c[3]=="true"),"model recognition "+std::to_string(count)+": "+c[1]);
        else if(c[0]=="S"){const auto actual=ezlocalai::llm_model_request_settings(c[2],c[1]);expect(actual==c[3],"model settings "+std::to_string(count)+" "+c[1]+" input "+c[2]+" actual "+actual+" expected "+c[3]);}
        else{
            expect(c[0]=="P"&&c.size()==9,"request fixture fields");std::string category="OK",error,payload="null",guard;
            try{payload=ezlocalai::llm_engine_request(c[2],c[1]=="chat",c[3],c[4]);}catch(const ezlocalai::LlmParameterError &e){category=e.category;error=e.message;}
            std::string expected_payload = c[7];
            if (c[1] == "chat" && expected_payload != "null" && c[2].find("\"tools\"") != std::string::npos) {
                // The Python binding drops OpenAI tool options. The worker
                // forwards them while keeping the rest of the recorded shape.
                const auto request = nlohmann::ordered_json::parse(c[2]);
                auto expected = nlohmann::ordered_json::parse(expected_payload);
                for (const auto *key : {"tools", "tool_choice", "parallel_tool_calls"}) {
                    if (request.contains(key)) expected[key] = request.at(key);
                }
                expected_payload = expected.dump();
            }
            expect(category==c[5]&&error==c[6]&&payload==expected_payload,"engine payload "+std::to_string(count)+": "+category+" "+error+" "+payload+" expected "+c[5]+" "+c[6]+" "+expected_payload);
            if(payload!="null"&&c[1]=="completion")try{ezlocalai::llm_completion_callback_guard(payload);}catch(const ezlocalai::LlmParameterError &e){guard=e.category+':'+e.message;}
            expect(guard==c[8],"compiled binding guard "+std::to_string(count));
        }
    }expect(count==6183,"all source request cases");return count;
}
void environment(){for(const auto &item:ezlocalai::environment_defaults)unsetenv(item.first.data());for(const auto *key:{"LLAMA_CPP_CONTEXT","EZLOCALAI_COMPOSE_DEFAULT_LLM_MAX_TOKENS","LLAMA_CPP_SERVER_URL","LLAMA_CPP_MODEL_PATHS","LLAMA_CPP_EXTRA_ARGS","LLM_FIXTURE_INFERENCE_RESULT","LLM_FIXTURE_INFERENCE_FILE","LLM_MAX_OUTPUT_TOKENS"})unsetenv(key);setenv("EZLOCALAI_DOTENV_PATH","",1);setenv("EZLOCALAI_API_KEY","none",1);setenv("GPU_LAYERS","0",1);setenv("LLM_MAX_TOKENS","512",1);setenv("LLM_PROMPT_CACHE_MIB","0",1);setenv("SPECULATIVE_TYPE","none",1);setenv("LLAMA_CPP_SERVER_BIN",TEST_EMBEDDING_FIXTURE_BIN,1);setenv("LLAMA_CPP_START_TIMEOUT","2",1);}
int available_port(){int fd=socket(AF_INET,SOCK_STREAM,0);sockaddr_in a{};a.sin_family=AF_INET;a.sin_addr.s_addr=htonl(INADDR_LOOPBACK);expect(!bind(fd,reinterpret_cast<sockaddr *>(&a),sizeof(a)),"bind");socklen_t size=sizeof(a);getsockname(fd,reinterpret_cast<sockaddr *>(&a),&size);close(fd);return ntohs(a.sin_port);}
struct Fixture{
    fs::path root;int port;
    Fixture(const fs::path &parent,const std::string &name):root(parent/name),port(available_port()){fs::create_directories(root);std::ofstream(root/"model.gguf")<<"fixture";setenv("LLAMA_CPP_MODEL_PATH",(root/"model.gguf").c_str(),1);setenv("EMBEDDING_FIXTURE_DIR",root.c_str(),1);setenv("LLAMA_CPP_SERVER_PORT",std::to_string(port).c_str(),1);}
    fs::path path(const std::string &name)const{return root/(name+'-'+std::to_string(port));}
    std::string last_payload()const{const auto seen=read(path("seen"));const auto at=seen.rfind("\r\n\r\n");expect(at!=std::string::npos,"managed request recorded");auto value=seen.substr(at+4);if(!value.empty()&&value.back()=='\n')value.pop_back();return value;}
    void clear()const{std::ofstream(path("seen"));}
    void reaped()const{std::istringstream in(read(path("launch")));for(int pid;in>>pid;)expect(kill(pid,0)!=0,"request child reaped");}
};
void managed(const fs::path &root){
    const std::vector<std::pair<std::string,std::string>> cases{
        {"unsloth/Inkling-2-Mini-GGUF",R"({"messages":[{"role":"user","content":""}],"max_tokens":8192,"temperature":0.12,"top_p":0.13,"stream":false,"cache_prompt":true,"top_k":3,"min_p":0.0,"repeat_penalty":2,"chat_template_kwargs":{"reasoning_effort":"high"}})"},
        {"ordinary",R"({"messages":[{"role":"user","content":""}],"max_tokens":8192,"temperature":0.12,"top_p":0.13,"stream":false,"cache_prompt":true,"top_k":3,"repeat_penalty":2})"},
        {"other/MiniCPM5-2B-Q8_0.gguf",R"({"messages":[{"role":"user","content":""}],"max_tokens":8192,"temperature":1.0,"top_p":0.95,"stream":false,"cache_prompt":true,"top_k":3,"min_p":0.0,"repeat_penalty":2,"chat_template_kwargs":{"enable_thinking":false}})"},
        {"other/Qwen3.8-27B-TEE",R"({"messages":[{"role":"user","content":""}],"max_tokens":8192,"temperature":1.0,"top_p":0.95,"stream":false,"cache_prompt":true,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repeat_penalty":2,"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"xhigh"}})"},
        {"unsloth/Qwen3.5-2B-MTP-GGUF",R"({"messages":[{"role":"user","content":""}],"max_tokens":8192,"temperature":0.7,"top_p":0.8,"stream":false,"cache_prompt":true,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repeat_penalty":2,"chat_template_kwargs":{"enable_thinking":false}})"},
    };
    size_t index=0;
    for(const auto &item:cases){environment();Fixture run(root,std::to_string(index++));setenv("DEFAULT_MODEL",item.first.c_str(),1);setenv("LLM_MAX_OUTPUT_TOKENS","999",1);
        {ezl_env_config config{};ezl_load_config(&config);config.request_timeout_ms=5000;ezlocalai::NativeServer server(config,std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model));
         const auto result=server.handle_request("POST","/v1/chat/completions","",R"({"messages":[{"role":"user","content":""}],"temperature":0.12,"top_p":0.13,"top_k":3,"repeat_penalty":2,"repetition_penalty":3,"model":"unknown-alias","stop":["custom"],"seed":7,"n":3,"user":"private-user","cache_prompt":false,"disable_fallback":true})");
         expect(result.status==200&&result.body.find("fixture")!=std::string::npos,"managed API response");expect(run.last_payload()==item.second,"managed profile payload "+item.first+": "+run.last_payload());
         if(item.first=="other/Qwen3.8-27B-TEE"){const auto instruct=server.handle_request("POST","/v1/chat/completions","",R"({"messages":[{"role":"user","content":""}],"reasoning":{"effort":"none"},"disable_fallback":true})");expect(instruct.status==200&&run.last_payload()==R"({"messages":[{"role":"user","content":""}],"max_tokens":8192,"temperature":0.7,"top_p":0.8,"stream":false,"cache_prompt":true,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repeat_penalty":1.0,"chat_template_kwargs":{"enable_thinking":false}})","effective reasoning changes profile and filters provider-only fields");}
        }run.reaped();
    }
    environment();
    Fixture inkling(root, "inkling-defaults");
    unsetenv("LLM_MAX_TOKENS");
    setenv("DEFAULT_MODEL", "unsloth/Inkling-2-Mini-GGUF", 1);
    {
        ezlocalai::LlamaCppInferenceBackend backend("unsloth/Inkling-2-Mini-GGUF");
        expect(!backend.text_result(R"({"messages":[{"role":"user","content":""}],"reasoning_effort":"low"})", "", true).failure,
               "Inkling request with defaults");
        expect(inkling.last_payload() == R"({"messages":[{"role":"user","content":""}],"max_tokens":8192,"temperature":1.0,"top_p":1.0,"stream":false,"cache_prompt":true,"top_k":0,"min_p":0.0,"repeat_penalty":1.0,"chat_template_kwargs":{"reasoning_effort":"low"}})",
               "Inkling sampling defaults and request reasoning effort");
        expect(read(inkling.path("args")).find("262144") != std::string::npos, "Inkling native context default");
    }
    inkling.reaped();
    const auto check_context = [&](const std::string &name, const char *model, const char *configured,
                                   const char *compose_marker, const char *native_override, const char *expected) {
        environment();
        Fixture run(root, name);
        setenv("LLM_MAX_TOKENS", configured, 1);
        if (compose_marker) setenv("EZLOCALAI_COMPOSE_DEFAULT_LLM_MAX_TOKENS", compose_marker, 1);
        if (native_override) setenv("LLAMA_CPP_CONTEXT", native_override, 1);
        {
            ezlocalai::LlamaCppInferenceBackend backend(model);
            expect(!backend.text_result(R"({"messages":[]})", "", true).failure, name + " startup");
            const auto args = read(run.path("args"));
            expect(args.find(std::string("-c\n") + expected + '\n') != std::string::npos, name + " context: " + args);
        }
        run.reaped();
    };
    check_context("inkling-compose", "unsloth/Inkling-2-Mini-GGUF", "65536", "65536", nullptr, "262144");
    check_context("inkling-small-profile", "unsloth/Inkling-2-Mini-GGUF", "8192", "8192", nullptr, "262144");
    check_context("inkling-set-same-as-compose", "unsloth/Inkling-2-Mini-GGUF", "65536", "explicit:65536", nullptr, "65536");
    check_context("inkling-context-from-env", "unsloth/Inkling-2-Mini-GGUF", "1048576", nullptr, nullptr, "1048576");
    check_context("inkling-context-override", "unsloth/Inkling-2-Mini-GGUF", "65536", "65536", "524288", "524288");
    check_context("other-model-in-compose", "ordinary", "65536", "65536", nullptr, "65536");
    environment();Fixture run(root,"captured");setenv("LLM_MAX_OUTPUT_TOKENS","123",1);
    {ezlocalai::LlamaCppInferenceBackend backend("ordinary");expect(!backend.text_result(R"({"prompt":"one"})","",false).failure,"first captured-default request");expect(run.last_payload()==R"({"prompt":"one","max_tokens":123,"temperature":1.31,"top_p":0.95,"stream":false})","constructor sampling defaults");
     setenv("LLM_MAX_OUTPUT_TOKENS","456",1);backend.completion(R"({"prompt":"two"})");expect(run.last_payload().find("\"max_tokens\":123")!=std::string::npos,"loaded engine keeps constructor default");
     auto restore=backend.unload_llms_for_media();restore(true);backend.completion(R"({"prompt":"three"})");expect(run.last_payload().find("\"max_tokens\":456")!=std::string::npos,"restoration re-reads output limit");
     run.clear();const auto stream=backend.text_result(R"({"prompt":"four","stream":true})","",false);expect(stream.failure&&stream.failure->category=="ValueError"&&!stream.lazy_chat_stream&&read(run.path("seen")).empty(),"completion callback guard runs before engine inference");
     backend.chat(R"({"messages":[],"stream":"false"})");expect(run.last_payload().find("\"stream\":true")!=std::string::npos,"chat generator forces stream true at the engine boundary");
    }run.reaped();
    environment();Fixture invalid(root,"invalid");setenv("LLM_MAX_OUTPUT_TOKENS","bad",1);
    {ezlocalai::LlamaCppInferenceBackend backend("ordinary");const auto result=backend.text_result(R"({"prompt":"hi"})","",false);expect(!result.from_engine&&result.body.find("invalid literal for int()")!=std::string::npos&&!fs::exists(invalid.path("launch")),"invalid output limit fails before process launch");}
}
}
int main(int argc,char **argv){try{expect(argc==3,"reference and fixture arguments");const auto count=policy(argv[1],argv[2]);char path[]="/tmp/ezlocalai-request-XXXXXX";const auto directory=mkdtemp(path);expect(directory,"temporary directory");managed(directory);fs::remove_all(directory);std::cout<<count<<" source request/settings/binding cases and managed payload/default checks passed\n";return 0;}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
