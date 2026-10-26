#include "ezlocalai/media_engine_process.hpp"
#include "ezlocalai/media_worker_protocol.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <sys/prctl.h>
using Json=nlohmann::json;
namespace fs=std::filesystem;
void check(bool value,const std::string &message){if(!value)throw std::runtime_error(message);}
void record(const std::string &line){std::ofstream("context.log",std::ios::app)<<line<<'\n';}
struct sd_ctx_params_t {bool eager_load=false; int ltx_offload_mode=0;};
#define EZLOCALAI_LTX_OFFLOAD_POLICY 1
struct sd_ctx_t {int id; std::vector<char> weights;};
int sequence=0, current_mode=0;
bool sd_set_ltx_offload_mode(sd_ctx_t*,int mode){if(mode<0||mode>2)return false;current_mode=mode;return true;}
sd_ctx_t *new_sd_ctx(const sd_ctx_params_t *p){check(p->eager_load,"persistent worker must request eager loading");record("load "+std::to_string(getpid()));return new sd_ctx_t{++sequence,std::vector<char>(1024*1024,'w')};}
void free_sd_ctx(sd_ctx_t *p){if(p){record("free "+std::to_string(getpid()));delete p;}}
void sd_set_log_callback(void*,void*){} void sd_set_progress_callback(void*,void*){}
enum preview_t{PREVIEW_NONE}; void sd_set_preview_callback(void*,preview_t,int,bool,bool,void*){}
#include "sd_cli_worker.hpp"
std::string read(const fs::path &path){std::ifstream file(path);return {std::istreambuf_iterator<char>(file),{}};}
int engine(int argc,const char **argv){
 std::map<std::string,std::string> args;for(int i=1;i+1<argc;i+=2)args[argv[i]]=argv[i+1];
 check(fcntl(57,F_GETFD)==-1 && errno==EBADF,"unrelated parent descriptor inherited");
 auto context=ezlocalai_sd_worker::acquire({false,args.count("--ltx-offload-mode")?std::stoi(args["--ltx-offload-mode"]):0});
 if(ezlocalai_sd_worker::preload)return 0;
 if(args.count("--parent-death-ready")){
  {std::ofstream ready(args.at("--parent-death-ready"));ready<<getpid();check(ready.good(),"parent death readiness marker");}
  // Stay inside generation so closing the IPC socket cannot race the death
  // signal by letting an idle worker observe EOF and exit successfully.
  for(;;)pause();
 }
 if(args.count("--sleep"))std::this_thread::sleep_for(std::chrono::milliseconds(std::stoi(args["--sleep"])));
 if(args.count("--crash"))raise(SIGKILL);
 if(args.count("--fail")){std::cerr<<"fixture model failure\n";return 7;}
 std::ofstream(args.at("-o"))<<Json{{"pid",getpid()},{"context",context->id},{"mode",current_mode},{"prompt",args["-p"]},{"pageable",std::getenv("GGML_CUDA_NO_PINNED")?std::getenv("GGML_CUDA_NO_PINNED"):""}};
 return 0;
}
void wire(){
 int fd[2];check(!socketpair(AF_UNIX,SOCK_STREAM,0,fd),"socketpair");
 struct Close{int *f;~Close(){close(f[0]);close(f[1]);}}close{fd};
 auto writer=std::async(std::launch::async,[&]{const std::string payload="fragmented";unsigned char header[]{0,0,0,10};for(auto byte:header){::send(fd[0],&byte,1,MSG_NOSIGNAL);std::this_thread::sleep_for(std::chrono::milliseconds(1));}for(char c:payload)::send(fd[0],&c,1,MSG_NOSIGNAL);});
 std::string out;check(ezlocalai::media_wire::receive(fd[1],out,ezlocalai::media_wire::deadline(2))&&out=="fragmented","fragmented frame");writer.get();
 unsigned char oversized[]{1,0,0,1};::send(fd[0],oversized,4,MSG_NOSIGNAL);bool rejected=false;try{ezlocalai::media_wire::receive(fd[1],out);}catch(const std::length_error&){rejected=true;}check(rejected,"oversize frame");
 char partial=0;::send(fd[0],&partial,1,MSG_NOSIGNAL);bool timed=false;try{ezlocalai::media_wire::receive(fd[1],out,ezlocalai::media_wire::deadline(.02));}catch(const ezlocalai::media_wire::Timeout&){timed=true;}check(timed,"partial frame respects one absolute deadline");
}
int main(int argc,char **argv){
 if(argc==5&&std::string(argv[1])=="--ezlocalai-worker-fd")return ezlocalai_sd_worker::serve(std::stoi(argv[2]),std::stoi(argv[4]),engine);
 if(argc==2&&std::string(argv[1])=="--abandon"){
  ezlocalai::MediaEngineProcess worker({fs::canonical("/proc/self/exe").string()});
  worker.request({"fixture","--model","one"},true,"preload.log",2);
  const auto child=worker.pid();
  fs::remove("abandoned.ready");
  std::thread([&]{try{worker.request({"fixture","--model","one","--parent-death-ready","abandoned.ready"},false,"abandoned.log",10);}catch(...){}}).detach();
  bool ready=false;
  for(int i=0;i<200;++i){if(read("abandoned.ready")==std::to_string(child)){ready=true;break;}std::this_thread::sleep_for(std::chrono::milliseconds(10));}
  if(!ready){kill(child,SIGKILL);_exit(1);}
  std::ofstream("abandoned.pid")<<child;_exit(0);
 }
 if(argc==3&&std::string(argv[1])=="--model-case"){
  try{
   auto config=Json::parse(read(argv[2]));ezlocalai::MediaEngineProcess worker(config.at("command").get<std::vector<std::string>>());Json report;
   const int idle_ms=config.value("idle_observation_ms",0);check(idle_ms>=0&&idle_ms<=10000,"invalid observation delay");
   auto observe=[&](const std::string& phase,bool pause=true){std::ofstream("observation.json.tmp")<<Json{{"phase",phase},{"pid",worker.pid()}}.dump()<<'\n';fs::rename("observation.json.tmp","observation.json");if(pause&&idle_ms)std::this_thread::sleep_for(std::chrono::milliseconds(idle_ms));};
   auto measure=[&](const Json &args,bool preload,const std::string &log){const auto start=std::chrono::steady_clock::now();auto result=worker.request(args.get<std::vector<std::string>>(),preload,log,600);check(!result.timed_out&&result.status==0,"real model request failed: "+log);return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};
   report["preload_seconds"]=measure(config.at("preload"),true,"preload.log");const auto pid=worker.pid();report["pid"]=pid;report["preload_status"]=read("/proc/"+std::to_string(pid)+"/status");check(read("preload.log").find("[media-worker] context ready")!=std::string::npos,"real context was not initialized");
   observe("preloaded");report["requests"]=Json::array();int count=0;for(const auto &request:config.at("requests")){const auto log="request-"+std::to_string(count++)+".log";observe("request-"+std::to_string(count)+"-running",false);const auto seconds=measure(request,false,log);check(worker.pid()==pid&&read(log).find("[media-worker] context ready")==std::string::npos,"generation reloaded the context");report["requests"].push_back({{"seconds",seconds},{"pid",worker.pid()}});observe("request-"+std::to_string(count)+"-complete");}
   report["final_status"]=read("/proc/"+std::to_string(pid)+"/status");observe("stopping",false);worker.stop();check(kill(pid,0)<0,"real engine was not reaped");report["reaped"]=true;std::ofstream("result.json")<<report.dump(2)<<'\n';std::cout<<report.dump()<<'\n';return 0;
  }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
 }
 const auto old=fs::current_path();fs::path directory;
 try{
  wire();directory=fs::temp_directory_path()/("ezlocalai-media-engine-"+std::to_string(getpid()));fs::create_directories(directory);fs::current_path(directory);
  const auto self=fs::canonical("/proc/self/exe").string();
  int opened=open("inherited",O_CREAT|O_RDWR,0600);check(opened>=0,"descriptor fixture");check(dup2(opened,57)==57,"descriptor duplicate");close(opened);
  pid_t child;
  {ezlocalai::MediaEngineProcess worker({"/usr/bin/env","GGML_CUDA_NO_PINNED=1",self});
   check(worker.request({"fixture","--model","one"},true,"preload.log",2).status==0,"preload");child=worker.pid();
   check(!fs::exists("output.json")&&read("context.log")=="load "+std::to_string(child)+"\n","preload loads without generating");
   for(const auto &prompt:{std::string("first"),std::string(2*1024*1024,'x')}){
    check(worker.request({"fixture","--model","one","-p",prompt,"-o","output.json"},false,"generation.log",2).status==0,"generate");auto output=Json::parse(read("output.json"));check(output["pid"]==child&&output["context"]==1&&output["prompt"]==prompt&&output["pageable"]=="1","context reuse / large literal prompt / child-only environment");
   }
   for(int mode:{2,1,0,2}){
    check(worker.request({"fixture","--model","one","--ltx-offload-mode",std::to_string(mode),"-p","policy","-o","output.json"},false,"mode.log",2).status==0,"placement update");
    const auto output=Json::parse(read("output.json"));check(output["pid"]==child&&output["context"]==1&&output["mode"]==mode,"placement updates without replacing parameters");
   }
   check(read("context.log")=="load "+std::to_string(child)+"\n","one load across preload, generation and placement changes");
   check(worker.request({"fixture","--model","two"},true,"changed.log",2).status==0,"changed model");
   check(read("context.log")=="load "+std::to_string(child)+"\nfree "+std::to_string(child)+"\nload "+std::to_string(child)+"\n","changed context released before replacement");
   check(worker.request({"fixture","--model","two","--fail","yes"},false,"failure.log",2).status==7,"model error status");check(worker.pid()==-1&&kill(child,0)<0&&read("failure.log").find("fixture model failure")!=std::string::npos,"failed model worker reaped and diagnostics retained");
   check(worker.request({"fixture","--model","two"},true,"retry.log",2).status==0,"restart after failure");child=worker.pid();
   check(worker.request({"fixture","--model","two","--sleep","5000"},false,"timeout.log",.03).timed_out,"generation timeout");check(worker.pid()==-1&&kill(child,0)<0,"timed-out worker reaped");
   bool crashed=false;try{worker.request({"fixture","--crash","yes"},false,"crash.log",2);}catch(const std::runtime_error&){crashed=true;}check(crashed&&worker.pid()==-1,"crash detected and reaped");
   check(worker.request({"fixture","--model","one"},true,"last.log",2).status==0,"restart after crash");child=worker.pid();
  }
  check(kill(child,0)<0,"destructor reaps worker");close(57);
  check(prctl(PR_SET_CHILD_SUBREAPER,1)==0,"subreaper fixture");pid_t parent=fork();check(parent>=0,"fork parent");if(!parent){execl(self.c_str(),self.c_str(),"--abandon",nullptr);_exit(127);}int status;waitpid(parent,&status,0);check(WIFEXITED(status)&&WEXITSTATUS(status)==0,"abandoned parent exit");child=std::stoi(read("abandoned.pid"));bool reaped=false;for(int i=0;i<200;++i){if(waitpid(child,&status,WNOHANG)==child){reaped=true;break;}std::this_thread::sleep_for(std::chrono::milliseconds(10));}if(!reaped){kill(child,SIGKILL);waitpid(child,&status,0);}check(reaped&&WIFSIGNALED(status)&&WTERMSIG(status)==SIGKILL,"parent death releases abandoned context");
  fs::current_path(old);fs::remove_all(directory);std::cout<<"Persistent media protocol, context reuse/reload, preload, failure, timeout, crash, parent death and descriptor isolation passed\n";return 0;
 }catch(const std::exception &e){fs::current_path(old);std::cerr<<e.what()<<"\nArtifacts: "<<directory<<'\n';return 1;}
}
