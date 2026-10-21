#include "ezlocalai/wakeword_jobs.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/wakeword_paths.hpp"
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <future>
#include <iostream>
#include <thread>
#include <unistd.h>
namespace {
using namespace ezlocalai;using Json=nlohmann::json;namespace fs=std::filesystem;
void check(bool value,const std::string &message){if(!value)throw std::runtime_error(message);}
struct Temporary {fs::path path;Temporary(){char pattern[]="/tmp/ezlocalai-jobs-XXXXXX";const auto p=mkdtemp(pattern);check(p,"temporary directory");path=p;}~Temporary(){std::error_code ec;fs::remove_all(path,ec);}};
std::string audio(){CttsAudioArray data;data.shape={1600};data.samples.resize(1600,.1f);return ctts_wav_bytes(data,16000);}
std::string encode(const std::string &bytes){std::string out(4*((bytes.size()+2)/3)+1,'\0');out.resize(EVP_EncodeBlock(reinterpret_cast<unsigned char *>(out.data()),reinterpret_cast<const unsigned char *>(bytes.data()),bytes.size()));return out;}
Json read(const fs::path &path){std::ifstream file(path);Json value;file>>value;return value;}
template<class Get> Json finished(Get get){for(size_t i=0;i<1000;++i){auto job=get();if(job.at("status")=="completed"||job.at("status")=="failed"||job.at("status")=="cancelled")return job;std::this_thread::sleep_for(std::chrono::milliseconds(5));}throw std::runtime_error("job did not finish");}
struct Trainer:WakewordTrainableModel {
    std::string word;explicit Trainer(std::string word):word(std::move(word)){}
    std::pair<bool,double> predict(const std::string &)override{return {true,.8};}
    WakewordHistory train(const std::vector<std::string> &yes,const std::vector<std::string> &no,const WakewordTrainingOptions &options,const WakewordProgress &progress)override{
        check(yes.size()==30&&no.size()==21,"complete augmented/negative dataset reaches trainer");check(options.batch_size==4,"requested batch size reaches trainer");
        if(word=="badtrain")throw std::runtime_error("training failed intentionally");
        WakewordHistory history;for(int64_t epoch=1;epoch<=options.epochs;++epoch){WakewordMetrics metrics{{"train_loss",1.0/epoch},{"val_loss",2.0/epoch},{"train_acc",.75},{"val_acc",.5}};for(const auto &item:metrics)history[item.first].push_back(item.second);progress(epoch,options.epochs,metrics);}return history;
    }
    void save_checkpoint(const std::string &directory)override{fs::create_directories(directory);std::ofstream(fs::path(directory)/"model.pt")<<"checkpoint";std::ofstream(fs::path(directory)/"config.json")<<Json{{"word",word}};}
    void export_onnx(const std::string &)override{}
    WakewordSavedFiles save(const std::string &directory)override{save_checkpoint(directory);if(word=="badsave")throw std::runtime_error("export failed intentionally");return {{"pytorch",(fs::path(directory)/"model.pt").string()}};}
};
struct Gate {std::mutex mutex;std::condition_variable changed;bool open=false,entered=false;
    void wait(const VoiceCancellation &token){std::unique_lock<std::mutex> lock(mutex);entered=true;changed.notify_all();while(!open){if(token.requested())throw VoiceCancelled("");changed.wait_for(lock,std::chrono::milliseconds(5));}}
    void entered_wait(){std::unique_lock<std::mutex> lock(mutex);check(changed.wait_for(lock,std::chrono::seconds(2),[&]{return entered;}),"synthesis entered");}
    void release(){std::lock_guard<std::mutex> lock(mutex);open=true;changed.notify_all();}
};
struct ReleaseGate {Gate &gate;~ReleaseGate(){gate.release();}};
struct PausedTrainer:Trainer {
    Gate &gate;bool exporting;
    PausedTrainer(const std::string &word,Gate &gate,bool exporting):Trainer(word),gate(gate),exporting(exporting){}
    WakewordHistory train(const std::vector<std::string> &yes,const std::vector<std::string> &no,const WakewordTrainingOptions &options,const WakewordProgress &progress)override{
        if(!exporting){
            check(bool(options.check_cancel),"background job passes batch cancellation to trainer");
            progress(1,3,{{"train_loss",.25}});
            std::unique_lock<std::mutex> lock(gate.mutex);gate.entered=true;gate.changed.notify_all();
            while(!gate.open){options.check_cancel();gate.changed.wait_for(lock,std::chrono::milliseconds(5));}
            options.check_cancel();
        }
        return Trainer::train(yes,no,options,progress);
    }
    WakewordSavedFiles save(const std::string &directory)override{
        auto result=Trainer::save(directory);
        if(exporting)gate.wait(VoiceCancellation{});
        return result;
    }
};
WakewordJobOperations operations(Gate *gate=nullptr){WakewordJobOperations ops;ops.samples=[gate](const auto &token){WakewordTtsOperations tts;tts.cancellation=token;tts.qwen_wav=[gate,token](const auto &word,const auto &){if(gate&&word=="Hello")gate->wait(token);return audio();};return tts;};ops.trainer=[](const auto &word){return std::make_shared<Trainer>(word);};return ops;}
void manager(){
    Temporary directory;Gate gate;WakewordJobs jobs(directory.path,operations(&gate));
    const auto created=jobs.create("Hello",5,2,4);const auto initial=Json::parse(created.first);const std::string id=initial.at("job_id");
    check(created.second&&initial.at("status")=="pending","create returns pending snapshot");gate.entered_wait();
    std::vector<std::future<std::pair<std::string,bool>>> concurrent;for(int i=0;i<12;++i)concurrent.push_back(std::async(std::launch::async,[&]{return jobs.create("hELLo",5,2,4);}));
    for(auto &request:concurrent){auto result=request.get();check(!result.second&&Json::parse(result.first).at("job_id")==id,"atomic case-insensitive deduplication");}
    const auto cancelled=Json::parse(jobs.create("queued",5,2,4).first).at("job_id").get<std::string>();check(jobs.cancel(cancelled)&&!jobs.cancel(cancelled),"cancel queued job once");
    gate.release();auto complete=finished([&]{return Json::parse(*jobs.get(id));});
    check(complete.at("status")=="completed",complete.dump());check(complete.at("progress")==100&&complete.at("current_stage")=="Complete","completion status");
    const auto &metrics=complete.at("metrics");check(metrics.at("positive_samples")==30&&metrics.at("negative_samples")==21&&metrics.at("final_train_loss")==.5,"final counts/history");
    check(metrics.at("exported_formats")==Json::array({"pytorch"}),"ordered exports");
    check(!jobs.active("hello")&&!jobs.cancel(id),"completed word released");
    check(read(directory.path/"wakeword_jobs"/(id+".json"))==complete,"atomic persisted snapshot");
    check(jobs.list("cancelled").size()==1&&jobs.list("",0).empty()&&jobs.list("",-1).size()==1,"filters and Python limit slicing");
    for(const auto &word:{"badtrain","badsave","emptyepochs","few"}){
        auto row=Json::parse(jobs.create(word,std::string(word)=="few"?2:5,std::string(word)=="emptyepochs"?0:2,4).first);
        const auto job_id=row.at("job_id").get<std::string>();auto failed=finished([&]{return Json::parse(*jobs.get(job_id));});
        check(failed.at("status")=="failed"&&!failed.at("error_message").get<std::string>().empty(),"failure is visible and persisted");check(!jobs.active(word),"failed word released");
        check(!wakeword_model_directory(directory.path/"wakeword_models",word),"failed training/export publishes no checkpoint");
    }
    check(!jobs.get("missing")&&!jobs.cancel("missing"),"unknown job");bool invalid=false;try{jobs.create("../escape");}catch(const std::invalid_argument &){invalid=true;}check(invalid,"path validation before enqueue");
    jobs.stop();WakewordJobs reloaded(directory.path,operations());check(Json::parse(*reloaded.get(id))==complete,"reload completed job");
}
void cancellation(){Temporary directory;Gate gate;WakewordJobs jobs(directory.path,operations(&gate));const auto id=Json::parse(jobs.create("Hello",5,2,4).first).at("job_id").get<std::string>();gate.entered_wait();check(jobs.cancel(id),"cancel active synthesis");jobs.stop();const auto job=Json::parse(*jobs.get(id));check(job.at("status")=="cancelled"&&job.at("progress")==0&&!jobs.active("hello"),"worker cannot overwrite cancellation");check(!fs::exists(directory.path/"wakeword_models"/"hello"/"model.pt"),"cancelled job does not export");}
void sample_cancellation(){
    WakewordRandom random(42);int calls=0;bool cancelled=false;
    try{wakeword_noise_samples(100,random,16000,1,[&]{if(++calls==3)throw VoiceCancelled("");});}
    catch(const VoiceCancelled &){cancelled=true;}
    check(cancelled&&calls==3,"noise generation cancels between samples");
    std::vector<WakewordTtsSample> samples(3,{audio(),"qwen","default","hello"});
    calls=0;cancelled=false;
    try{wakeword_augment_dataset(samples,random,5,{},[&]{if(++calls==5)throw VoiceCancelled("");});}
    catch(const VoiceCancelled &){cancelled=true;}
    check(cancelled&&calls==5,"augmentation cancels before ten-sample progress boundary");
}
void stage_cancellation(bool exporting){
    Temporary directory;Gate gate;auto ops=operations();
    ops.trainer=[&](const auto &word){return std::make_shared<PausedTrainer>(word,gate,exporting);};
    WakewordJobs jobs(directory.path,std::move(ops));ReleaseGate release{gate};
    const auto id=Json::parse(jobs.create("stage",5,3,4).first).at("job_id").get<std::string>();
    gate.entered_wait();const auto snapshot=Json::parse(*jobs.get(id));
    check(snapshot.at("status")==(exporting?"exporting":"training"),"live stage is visible");
    check(snapshot.at("progress").get<double>()==(exporting?90.0:50+40.0/3),"live stage progress");
    check(!wakeword_model_directory(directory.path/"wakeword_models","stage"),"incomplete export is not downloadable");
    check(jobs.cancel(id),"cancel in-flight stage");gate.release();jobs.stop();
    const auto cancelled=Json::parse(*jobs.get(id));check(cancelled.at("status")=="cancelled","cancel remains terminal after stage returns");
    check(read(directory.path/"wakeword_jobs"/(id+".json"))==cancelled,"cancel persists with latest progress");
    check(!wakeword_model_directory(directory.path/"wakeword_models","stage"),"cancelled export publishes no model");
    const auto model_dir=directory.path/"wakeword_models/stage";
    check(!fs::exists(model_dir)||fs::is_empty(model_dir),"cancelled export removes its staging artifacts");
    WakewordJobs restarted(directory.path,operations());
    const auto retry=Json::parse(restarted.create("stage",5,2,4).first).at("job_id").get<std::string>();
    check(finished([&]{return Json::parse(*restarted.get(retry));}).at("status")=="completed","cancelled word can be trained again");
}
void loaded_active(){
    Temporary directory;fs::create_directories(directory.path/"wakeword_jobs");
    Json original{{"job_id","old-job"},{"word","Old"},{"status","training"},{"created_at","2026-10-19T10:00:00"},{"updated_at","2026-10-19T10:00:00"},{"progress",60},{"current_stage","Training model"},{"metrics",Json::object()}};
    const auto path=directory.path/"wakeword_jobs/old-job.json";std::ofstream(path)<<original;
    std::ofstream(directory.path/"wakeword_jobs/broken.json")<<"{";
    std::atomic<int> calls{0};auto ops=operations();ops.samples=[&](const auto &){++calls;return WakewordTtsOperations{};};
    WakewordJobs jobs(directory.path,std::move(ops));check(jobs.active("OLD").has_value()&&jobs.list().size()==1,"load active job and skip malformed record");
    auto changed=original;changed["status"]="completed";std::ofstream(path)<<changed;
    check(Json::parse(*jobs.get("old-job"))==original,"jobs cached like Python manager");
    check(!jobs.create("old").second&&calls==0,"loaded active job prevents duplicate without restarting");
    check(jobs.cancel("old-job")&&read(path).at("status")=="cancelled","loaded job cancellation persists");
}
void http(){
    Temporary directory;const auto previous=fs::current_path();fs::current_path(directory.path);struct Restore{fs::path path;~Restore(){fs::current_path(path);}}restore{previous};
    for(const auto &entry:environment_defaults)unsetenv(std::string(entry.first).c_str());
    setenv("EZLOCALAI_API_KEY","",1);setenv("VOICE_SERVER","false",1);setenv("DEFAULT_MODEL","model",1);
    for(const auto *key:{"TTS_ENABLED","STT_ENABLED","EMBEDDING_ENABLED","IMAGE_ENABLED","VIDEO_ENABLED","MUSIC_ENABLED","EZLOCALAI_PRELOAD_MODEL_ON_START","EZLOCALAI_MODEL_STATUS_ON_START","TTS_UNLOAD_LLM_DURING_GENERATION"})setenv(key,"false",1);
    ezl_env_config config{};ezl_load_config(&config);
    TtsPoolFactory pool=[](auto resources){TtsPoolOperations ops;ops.can_load=[] {return ModelAdmission{true,"cpu","test"};};ops.load=[](const auto &){auto model=std::make_shared<TtsPoolModel>();model->closed=[] {return false;};model->device=[] {return "cpu";};model->close=[] {};model->generate=[](const auto &request){const auto data=Json::parse(request);check(data.at("use_cache")==false,"HTTP training uses uncached Qwen");return encode(audio());};return model;};return tts_pool_resources(std::move(ops),resources);};
    Gate gate;NativeServer server(config,std::make_unique<EchoInferenceBackend>("model"),{},{},pool);ReleaseGate release{gate};
    server.set_wakeword_trainer([&](const auto &word)->std::shared_ptr<WakewordTrainableModel>{
        if(word=="cancelhttp")return std::make_shared<PausedTrainer>(word,gate,false);
        return std::make_shared<Trainer>(word);
    });setenv("STT_ENABLED","true",1);
    const auto response=server.handle_request("POST","/v1/wakeword/train","",R"({"word":"http","sample_count":5,"epochs":2,"batch_size":4})","application/json");check(response.status==200,"HTTP job accepted");const auto id=Json::parse(response.body).at("job_id").get<std::string>();
    const auto job=finished([&]{const auto poll=server.handle_request("GET","/v1/wakeword/jobs/"+id,"","");check(poll.status==200,"HTTP status available");return Json::parse(poll.body);});check(job.at("status")=="completed",job.dump());
    const auto model=server.handle_request("GET","/v1/wakeword/models/http?format=pytorch","","");check(model.status==200&&model.body=="checkpoint","HTTP trained checkpoint download");
    const auto repeated=server.handle_request("POST","/v1/wakeword/train","",R"({"word":"http"})","application/json");check(Json::parse(repeated.body).at("status")=="completed","existing model skips training");
    Temporary outside;fs::create_directory_symlink(outside.path,directory.path/"wakeword_samples/badpath");
    const auto invalid=server.handle_request("POST","/v1/wakeword/train","",R"({"word":"badpath"})","application/json");check(invalid.status==409,"source training-conflict response for escaping sample directory");
    const auto accepted=server.handle_request("POST","/v1/wakeword/train","",R"({"word":"cancelhttp","sample_count":5,"epochs":3,"batch_size":4})","application/json");
    check(accepted.status==200,"HTTP cancellation job accepted");
    const auto cancel_id=Json::parse(accepted.body).at("job_id").get<std::string>();
    gate.entered_wait();
    const auto poll=server.handle_request("GET","/v1/wakeword/jobs/"+cancel_id,"","");
    const auto live=Json::parse(poll.body);
    check(poll.status==200&&live.at("status")=="training"&&live.at("metrics").at("train_loss")==.25,"HTTP live training metrics");
    check(std::abs(live.at("progress").get<double>()-(50+40.0/3))<1e-12,"HTTP preserves full progress precision");
    const auto listed=server.handle_request("GET","/v1/wakeword/jobs?status=training","","");
    check(Json::parse(listed.body)==Json::array({live}),"list and poll have identical live snapshots");
    const auto cancelled=server.handle_request("DELETE","/v1/wakeword/jobs/"+cancel_id,"","");
    check(cancelled.status==200&&Json::parse(cancelled.body)==Json{{"message","Job cancelled successfully"}},"HTTP cancellation response");
    const auto final=Json::parse(server.handle_request("GET","/v1/wakeword/jobs/"+cancel_id,"","").body);
    check(final.at("status")=="cancelled"&&final.at("progress")==live.at("progress")&&final.at("metrics")==live.at("metrics"),"HTTP cancelled snapshot retains progress/metrics");
    check(server.handle_request("DELETE","/v1/wakeword/jobs/"+cancel_id,"","").status==400,"HTTP repeat cancellation error");
    check(server.handle_request("GET","/v1/wakeword/models/cancelhttp?format=pytorch","","").status==404,"HTTP cancelled job has no checkpoint");
}
}
int main(){try{manager();cancellation();sample_cancellation();stage_cancellation(false);stage_cancellation(true);loaded_active();http();std::cout<<"Wake-word background jobs: pipeline, persistence, failures, duplicate requests, synthesis/training/export cancellation and live HTTP status passed\n";return 0;}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
