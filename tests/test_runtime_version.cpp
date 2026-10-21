#include "ezlocalai/runtime_version.hpp"
#include "ezlocalai/process.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
namespace fs=std::filesystem;
using Json=nlohmann::json;
static void check(bool value,const std::string &message){if(!value)throw std::runtime_error(message);}
static void clear(){for(auto key:{"EZLOCALAI_VERSION","EZLOCALAI_COMMIT","GIT_COMMIT"})unsetenv(key);}
int main(int argc,char **argv){try{
    check(argc==2,"version fixture required");std::ifstream file(argv[1]);const auto fixture=Json::parse(file);
    std::string pattern=(fs::temp_directory_path()/"ezlocalai-version-test-XXXXXX").string();check(mkdtemp(pattern.data()),"temporary directory");
    struct Cleanup{fs::path path;~Cleanup(){std::error_code ec;fs::remove_all(path,ec);}}cleanup{pattern};
    for(const auto &row:fixture.at("clean"))check(ezlocalai::clean_runtime_version(row.at("input"))==row.at("expected"),"version cleaning");
    size_t index=0;
    for(const auto &row:fixture.at("cases")){
        clear();const auto root=cleanup.path/std::to_string(index++);fs::create_directories(root);
        for(const auto &entry:row.at("environment").items())setenv(entry.key().c_str(),entry.value().get<std::string>().c_str(),1);
        for(const auto &entry:row.at("files").items()){
            const auto path=root/entry.key();fs::create_directories(path.parent_path());std::string bytes;
            if(entry.value().is_string())bytes=entry.value();
            else{const auto hex=entry.value().at("hex").get<std::string>();for(size_t i=0;i<hex.size();i+=2)bytes+=char(std::stoi(hex.substr(i,2),nullptr,16));}
            std::ofstream(path,std::ios::binary)<<bytes;
        }
        size_t calls=0;auto git=[&](const std::string &directory){check(directory==root.string(),"Git repository path");++calls;const auto value=row.at("git").get<std::string>();if(value=="THROW")throw std::runtime_error("git missing");return value;};
        const auto package=row.at("package").get<std::string>();
        const auto actual=ezlocalai::runtime_version_at(root.string(),package=="THROW"?"":package,git);
        check(actual==row.at("expected"),row.at("name").get<std::string>()+": "+actual);
        const auto expected_calls=row.at("calls").get<std::vector<std::string>>();
        check(calls==size_t(std::count(expected_calls.begin(),expected_calls.end(),"git")),"precedence avoids unnecessary Git process");
    }
    clear();const auto root=cleanup.path/"git-repository";fs::create_directories(root);
    check(ezlocalai::run_process({"git","init","--quiet",root.string()},true,true).status==0,"real git init");
    check(ezlocalai::run_process({"git","-C",root.string(),"-c","user.name=Version Test","-c","user.email=version@example.invalid","-c","commit.gpgsign=false","commit","--quiet","--allow-empty","-m","test"},true,true).status==0,"isolated Git commit");
    const auto expected=ezlocalai::clean_runtime_version(ezlocalai::run_process({"git","-C",root.string(),"rev-parse","--short","HEAD"},true,true).output);
    check(ezlocalai::runtime_version_at(root.string(),"fallback")==expected,"actual Git subprocess resolution");
    check(!ezlocalai::native_build_version().empty(),"packaged version fallback");
    setenv("EZLOCALAI_VERSION","cached-worker-build",1);check(ezlocalai::runtime_version()=="cached-worker-build","default version override");
    setenv("EZLOCALAI_VERSION","changed-after-first-heartbeat",1);check(ezlocalai::runtime_version()=="cached-worker-build","process version cache");clear();
    std::cout<<fixture.at("clean").size()<<" cleaning and "<<index<<" source resolution cases, real Git and cached worker version passed\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
