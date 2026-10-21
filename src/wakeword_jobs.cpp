#include "ezlocalai/wakeword_jobs.hpp"
#include "ezlocalai/wakeword_paths.hpp"
#include "ezlocalai/llm_slots.hpp"
#include <nlohmann/json.hpp>
#include <openssl/rand.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

namespace ezlocalai {
namespace {
using Json=nlohmann::json;namespace fs=std::filesystem;
bool terminal(const Json &job){const auto status=job.value("status","");return status=="completed"||status=="failed"||status=="cancelled";}
std::string now(int minutes=0){
    const auto time=std::chrono::system_clock::now()+std::chrono::minutes(minutes);
    const auto seconds=std::chrono::system_clock::to_time_t(time);std::tm local{};localtime_r(&seconds,&local);
    const auto micros=std::chrono::duration_cast<std::chrono::microseconds>(time.time_since_epoch()).count()%1000000;
    std::ostringstream out;out<<std::put_time(&local,"%Y-%m-%dT%H:%M:%S")<<'.'<<std::setw(6)<<std::setfill('0')<<micros;return out.str();
}
std::string uuid(){unsigned char bytes[16];if(RAND_bytes(bytes,sizeof(bytes))!=1)throw std::runtime_error("Cannot create wake-word job ID");bytes[6]=(bytes[6]&15)|64;bytes[8]=(bytes[8]&63)|128;std::ostringstream out;for(size_t i=0;i<16;++i){if(i==4||i==6||i==8||i==10)out<<'-';out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(bytes[i]);}return out.str();}
void check(const VoiceCancellation &token){if(token.requested())throw VoiceCancelled("");}
struct ExportDirectory {
    fs::path path;
    explicit ExportDirectory(fs::path path):path(std::move(path)){fs::create_directories(this->path);}
    ~ExportDirectory(){std::error_code ignored;fs::remove_all(path,ignored);}
};
}
struct WakewordJobs::Impl {
    struct Task {std::string id,word;int64_t samples,epochs,batch;VoiceCancellation cancellation;};
    fs::path root,models,samples,jobs;WakewordJobOperations operations;
    mutable std::mutex mutex;std::condition_variable changed;bool stopping=false;std::thread worker;
    std::map<std::string,Json> records;std::vector<std::string> order;
    std::map<std::string,std::string> words;std::map<std::string,VoiceCancellation> tokens;std::deque<Task> pending;
    WakewordRandom random;
    Impl(fs::path path,WakewordJobOperations ops):root(fs::absolute(path)),models(root/"wakeword_models"),samples(root/"wakeword_samples"),jobs(root/"wakeword_jobs"),operations(std::move(ops)){
        for(const auto &dir:{models,samples,jobs})fs::create_directories(dir);
        std::vector<std::string> loaded;
        if(operations.load)loaded=operations.load();else for(const auto &entry:fs::directory_iterator(jobs))if(entry.path().extension()==".json"&&entry.is_regular_file()){
            try{const auto path=wakeword_contained_path(jobs,entry.path().filename());std::ifstream input(path);loaded.emplace_back(std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>());}catch(const std::exception &){}
        }
        for(const auto &raw:loaded)try{
            auto job=Json::parse(raw);const auto id=job.at("job_id").get<std::string>(),word=job.at("word").get<std::string>();
            const auto status=job.at("status").get<std::string>();const std::vector<std::string> allowed{"pending","generating_samples","augmenting","training","exporting","completed","failed","cancelled"};
            if(id.empty()||std::find(allowed.begin(),allowed.end(),status)==allowed.end()||!job.at("created_at").is_string()||!job.at("updated_at").is_string())continue;
            wakeword_contained_path(jobs,id+".json");
            if(!records.count(id))order.push_back(id);records[id]=job;if(!terminal(job))words[llm_unicode_lower(word)]=id;
        }catch(const std::exception &){}
        worker=std::thread([this]{loop();});
    }
    void save(const Json &job){
        const auto target=wakeword_contained_path(jobs,job.at("job_id").get<std::string>()+".json");
        const auto temporary=wakeword_contained_path(jobs,"."+uuid()+".tmp");
        try{std::ofstream file(temporary,std::ios::binary);file<<job.dump(2)<<'\n';file.close();if(!file)throw std::runtime_error("Cannot persist wake-word job");fs::rename(temporary,target);}
        catch(...){std::error_code ignored;fs::remove(temporary,ignored);throw;}
    }
    void unmap(const Json &job){const auto word=llm_unicode_lower(job.at("word").get<std::string>());const auto found=words.find(word);if(found!=words.end()&&found->second==job.at("job_id"))words.erase(found);}
    void update(const Task &task,Json fields){
        check(task.cancellation);std::lock_guard<std::mutex> lock(mutex);check(task.cancellation);
        update_locked(task,std::move(fields));
    }
    // Caller holds mutex, including when publishing a completed export.
    void update_locked(const Task &task,Json fields){
        auto &job=records.at(task.id);if(terminal(job))throw VoiceCancelled("");
        auto next=job;for(const auto &field:fields.items())if(field.key()=="metrics")next["metrics"].update(field.value());else next[field.key()]=field.value();
        next["updated_at"]=now();save(next);job=std::move(next);if(terminal(job))unmap(job);
    }
    void run(const Task &task){
        check(task.cancellation);const auto model_dir=wakeword_contained_path(models,wakeword_name(task.word));
        const auto sample_dir=wakeword_contained_path(samples,wakeword_name(task.word));
        update(task,{{"status","generating_samples"},{"progress",0},{"current_stage","Generating TTS samples"}});
        if(!operations.samples||!operations.trainer)throw std::runtime_error("Native wake-word training is unavailable in this build");
        auto adapters=operations.samples(task.cancellation);adapters.cancellation=task.cancellation;
        WakewordSampleGenerator generator(sample_dir.string(),std::move(adapters));
        auto generated=generator.generate(task.word,task.samples,[&](size_t current,size_t total,const auto &){update(task,{{"progress",25.0*current/total}});});
        if(generated.size()<5)throw std::runtime_error("Only generated "+std::to_string(generated.size())+" samples, need at least 5");
        update(task,{{"status","augmenting"},{"progress",25},{"current_stage","Augmenting samples"}});
        const auto check_cancel=[&]{check(task.cancellation);};
        auto positive=wakeword_augment_dataset(generated,random,5,[&](size_t current,size_t total,const auto &){update(task,{{"progress",25+15.0*current/total}});},check_cancel);
        generated.clear();update(task,{{"progress",40},{"current_stage","Generating negative samples"}});
        auto negative=wakeword_noise_samples(positive.size()/4,random,16000,1.0,check_cancel);check(task.cancellation);
        auto similar=wakeword_similar_samples(task.word,generator,positive.size()/4);
        negative.insert(negative.end(),std::make_move_iterator(similar.begin()),std::make_move_iterator(similar.end()));
        update(task,{{"status","training"},{"progress",50},{"current_stage","Training model"}});
        auto trainer=operations.trainer(task.word);if(!trainer)throw std::runtime_error("Native wake-word trainer unavailable");check(task.cancellation);
        std::vector<std::string> yes,no;for(auto &sample:positive)yes.push_back(std::move(sample.audio));for(auto &sample:negative)no.push_back(std::move(sample.audio));
        WakewordTrainingOptions options;options.epochs=task.epochs;options.batch_size=task.batch;options.check_cancel=check_cancel;
        auto history=trainer->train(yes,no,options,[&](int64_t epoch,int64_t total,const auto &metrics){update(task,{{"progress",50+40.0*epoch/total},{"metrics",metrics}});});
        update(task,{{"status","exporting"},{"progress",90},{"current_stage","Exporting model"}});
        for(const auto &name:{"model.pt","config.json","model.onnx","model.onnx.data","model.espdl"})wakeword_contained_path(model_dir,name);
        // Readers recognize a model only when model.pt exists. Keep export
        // artifacts one level deeper until cancellation and metrics validation
        // have succeeded; publish the checkpoint last.
        ExportDirectory staging(wakeword_contained_path(model_dir,".export-"+task.id));
        const auto saved=trainer->save(staging.path.string());check(task.cancellation);
        Json metrics{{"positive_samples",yes.size()},{"negative_samples",no.size()},{"exported_formats",Json::array()}};
        for(const auto &name:{"train_loss","val_loss","train_acc","val_acc"}){if(history[name].empty())throw std::runtime_error("list index out of range");metrics[std::string("final_")+name]=history[name].back();}
        for(const auto &file:saved)metrics["exported_formats"].push_back(file.first);
        std::lock_guard<std::mutex> lock(mutex);
        check(task.cancellation);
        if(terminal(records.at(task.id)))throw VoiceCancelled("");
        if(fs::exists(wakeword_contained_path(model_dir,"model.pt")))
            throw std::runtime_error("Wake-word model already exists");
        if(!fs::is_regular_file(wakeword_contained_path(staging.path,"model.pt")))
            throw std::runtime_error("Wake-word export did not produce a checkpoint");
        for(const auto &name:{"config.json","model.onnx.data","model.onnx","model.espdl","model.pt"}){
            const auto source=wakeword_contained_path(staging.path,name);
            if(fs::exists(source))fs::rename(source,wakeword_contained_path(model_dir,name));
        }
        try {
            update_locked(task,{{"status","completed"},{"progress",100},{"current_stage","Complete"},{"model_path",model_dir.string()},{"metrics",metrics}});
        } catch (...) {
            // If status persistence fails, do not leave a failed job's model
            // discoverable. This checkpoint was created by this task only.
            std::error_code ignored;fs::remove(model_dir/"model.pt",ignored);throw;
        }
    }
    void loop(){for(;;){Task task;{std::unique_lock<std::mutex> lock(mutex);changed.wait(lock,[&]{return stopping||!pending.empty();});if(stopping)return;task=std::move(pending.front());pending.pop_front();}
        try{run(task);}catch(const VoiceCancelled &){}catch(const std::exception &error){try{update(task,{{"status","failed"},{"error_message",error.what()}});}catch(const std::exception &persist){std::cerr<<"Wake-word job update failed: "<<persist.what()<<'\n';}}
        std::lock_guard<std::mutex> lock(mutex);tokens.erase(task.id);
    }}
};
WakewordJobs::WakewordJobs(fs::path root,WakewordJobOperations operations):impl_(std::make_unique<Impl>(std::move(root),std::move(operations))){}
WakewordJobs::~WakewordJobs(){stop();}
std::pair<std::string,bool> WakewordJobs::create(const std::string &word,int64_t samples,int64_t epochs,int64_t batch){
    auto &s=*impl_;wakeword_contained_path(s.models,wakeword_name(word));wakeword_contained_path(s.samples,wakeword_name(word));
    std::lock_guard<std::mutex> lock(s.mutex);if(s.stopping)throw std::runtime_error("Wake-word training is stopping");
    const auto key=llm_unicode_lower(word);const auto found=s.words.find(key);if(found!=s.words.end())return {s.records.at(found->second).dump(),false};
    Impl::Task task{uuid(),word,samples,epochs,batch,{}};
    Json job{{"job_id",task.id},{"word",word},{"status","pending"},{"created_at",now()},{"updated_at",now()},{"estimated_completion",now(15)},{"progress",0},{"current_stage",""},{"error_message",nullptr},{"model_path",nullptr},{"metrics",Json::object()}};
    s.save(job);s.records[task.id]=job;s.order.push_back(task.id);s.words[key]=task.id;s.tokens[task.id]=task.cancellation;s.pending.push_back(task);s.changed.notify_one();return {job.dump(),true};
}
std::optional<std::string> WakewordJobs::get(const std::string &id) const{std::lock_guard<std::mutex> lock(impl_->mutex);const auto found=impl_->records.find(id);return found==impl_->records.end()?std::nullopt:std::optional<std::string>(found->second.dump());}
std::optional<std::string> WakewordJobs::active(const std::string &word) const{std::lock_guard<std::mutex> lock(impl_->mutex);const auto found=impl_->words.find(llm_unicode_lower(word));return found==impl_->words.end()?std::nullopt:std::optional<std::string>(impl_->records.at(found->second).dump());}
std::vector<std::string> WakewordJobs::list(const std::string &status,int64_t limit) const{
    std::lock_guard<std::mutex> lock(impl_->mutex);std::vector<Json> jobs;for(const auto &id:impl_->order){const auto &job=impl_->records.at(id);if(status.empty()||job.at("status")==status)jobs.push_back(job);}
    std::stable_sort(jobs.begin(),jobs.end(),[](const auto &a,const auto &b){return a.at("created_at")>b.at("created_at");});
    const auto count=limit<0?(uint64_t(-(limit+1))>=jobs.size()?0:jobs.size()-uint64_t(-(limit+1))-1):std::min<uint64_t>(limit,jobs.size());
    std::vector<std::string> out;for(size_t i=0;i<count;++i)out.push_back(jobs[i].dump());return out;
}
bool WakewordJobs::cancel(const std::string &id){
    VoiceCancellation token;{auto &s=*impl_;std::lock_guard<std::mutex> lock(s.mutex);const auto found=s.records.find(id);if(found==s.records.end()||terminal(found->second))return false;
        auto job=found->second;job["status"]="cancelled";job["updated_at"]=now();s.save(job);found->second=job;s.unmap(job);const auto running=s.tokens.find(id);if(running!=s.tokens.end())token=running->second;
    }token.cancel();return true;
}
void WakewordJobs::stop(){if(!impl_)return;std::vector<VoiceCancellation> tokens;{std::lock_guard<std::mutex> lock(impl_->mutex);impl_->stopping=true;for(const auto &entry:impl_->tokens)tokens.push_back(entry.second);}for(const auto &token:tokens)token.cancel();impl_->changed.notify_all();if(impl_->worker.joinable())impl_->worker.join();}
}
