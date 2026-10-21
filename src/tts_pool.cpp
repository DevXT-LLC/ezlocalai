#include "ezlocalai/tts_pool.hpp"
#include "ezlocalai/stt_pool.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/llm_slots.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
namespace ezlocalai {
namespace {
bool base_exception(const TtsError &error) {return error.category=="CancelledError"||error.category=="GeneratorExit"||error.category=="KeyboardInterrupt"||error.category=="SystemExit";}
std::string device(const TtsPoolHandle &model,const std::string &fallback="cpu") {try{return model&&model->device?model->device():fallback;}catch(const TtsError &error){if(error.category=="AttributeError")return fallback;throw;}}
bool closed(const TtsPoolHandle &model) {try{return model&&model->closed?model->closed():false;}catch(const TtsError &error){if(error.category=="AttributeError")return false;throw;}}
void erase_first(std::vector<TtsPoolHandle> &models,const TtsPoolHandle &model){const auto at=std::find(models.begin(),models.end(),model);if(at!=models.end())models.erase(at);}
}
TtsPoolOperations tts_pool_resources(TtsPoolOperations ops,std::shared_ptr<ResourceManager> resources){
    ops.can_load=[resources]{return resources->can_load_model(ModelType::Tts,4.0);};
    ops.register_model=[resources](const auto &name,const auto &device,double vram){resources->register_model(ModelType::Tts,name,device,vram);};
    ops.unregister_model=[resources]{resources->unregister_model(ModelType::Tts);};
    ops.mark_in_use=[resources](bool active){resources->mark_model_in_use(ModelType::Tts,active);};return ops;
}
TtsPoolOperations tts_pool_ctts_operations(CttsServiceOperations service){
    service.runtime=ctts_llama_operations(std::move(service.runtime));TtsPoolOperations ops;
    ops.environment=service.runtime.environment;ops.cuda_available=service.runtime.cuda_available;ops.collect=service.runtime.collect;ops.synchronize=service.runtime.synchronize;ops.empty_cache=service.runtime.empty_cache;ops.ipc_collect=service.runtime.ipc_collect;
    ops.load=[service](const std::optional<std::string> &device){auto owner=std::make_shared<Ctts>("null",device?nlohmann::json(*device).dump():"null",service);auto model=std::make_shared<TtsPoolModel>();model->device=[owner]{return owner->runtime().device();};model->closed=[owner]{return owner->runtime().closed();};model->close=[owner]{owner->close();};model->generate=[owner](const auto &request){return owner->generate(request);};model->stream=[owner](const auto &request){return owner->generate_stream(request);};model->voices=[owner]{return owner->voices();};model->sample_rate=[owner]{return owner->sample_rate();};return model;};return ops;
}
TtsPool::TtsPool(int64_t capacity,TtsPoolOperations ops):capacity_(capacity),operations_(std::move(ops)){
    if(!operations_.environment)operations_.environment=[](const auto &key)->std::optional<std::string>{const auto value=std::getenv(key.c_str());return value?std::optional<std::string>(value):std::nullopt;};
    if(!operations_.load){CttsServiceOperations service;service.runtime.environment=operations_.environment;operations_.load=tts_pool_ctts_operations(std::move(service)).load;}
    if(!operations_.available)operations_.available=[]{return true;};
    for(auto *predicate:{&operations_.should_preload,&operations_.handoff_active,&operations_.cuda_available})if(!*predicate)*predicate=[]{return false;};
    if(!operations_.can_load)operations_.can_load=[]{return ModelAdmission{true,"cpu",""};};
    for(auto *action:{&operations_.unregister_model,&operations_.collect,&operations_.synchronize,&operations_.empty_cache,&operations_.ipc_collect})if(!*action)*action=[]{};
    if(!operations_.register_model)operations_.register_model=[](const auto &,const auto &,double){};
    if(!operations_.mark_in_use)operations_.mark_in_use=[](bool){};
    if(!operations_.now)operations_.now=[]{return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();};
    if(!operations_.schedule_cleanup)operations_.schedule_cleanup=schedule_stt_cleanup;
}
std::string TtsPool::name() const{return "Qwen-TTS ("+resolve_tts_model_id(operations_.environment("QWEN_TTS_MODEL").value_or(""))+")";}
TtsPoolHandle TtsPool::create(const std::optional<std::string> &device,bool force_cpu){if(!operations_.available())return {};return operations_.load(force_cpu?std::optional<std::string>("cpu"):device);}
TtsPoolHandle TtsPool::load_instance(bool force_cpu){
    const auto provider=name();auto admission=operations_.can_load();if(force_cpu)admission.device="cpu";
    const auto fallback=llm_unicode_lower(strip_config_value(operations_.environment("QWEN_TTS_ALLOW_CPU_FALLBACK").value_or("true")));const bool allow=fallback=="1"||fallback=="true"||fallback=="yes"||fallback=="on";
    if(admission.device=="cpu"&&!force_cpu&&!allow)throw TtsError("RuntimeError",provider+" needs CUDA VRAM for reliable generation; "+admission.reason+". Reduce TTS_N_PARALLEL, free VRAM, or set QWEN_TTS_ALLOW_CPU_FALLBACK=true to try CPU generation.");
    operations_.now();auto model=create(admission.device);device(model,admission.device.empty()?"cpu":admission.device);operations_.now();return model;
}
void TtsPool::register_locked(){
    auto loaded=resident_;loaded.insert(loaded.end(),transient_.begin(),transient_.end());if(loaded.empty()){operations_.unregister_model();return;}
    bool any_cuda=false;for(const auto &model:loaded)if(device(model,"").find("cuda")!=std::string::npos){any_cuda=true;break;}
    double vram=0;for(const auto &model:loaded)if(device(model,"").find("cuda")!=std::string::npos)vram+=4;
    const auto provider=name();const auto selected=any_cuda?"cuda":device(loaded.front());operations_.register_model(provider,selected,vram);
}
void TtsPool::register_resources(){std::lock_guard<std::recursive_mutex> lock(mutex_);register_locked();}
void TtsPool::warm_cache(){
    // Pipes.__init__ warms one CTTS instance directly, without a pool lease or
    // the resident-pool CPU fallback gate, then schedules its normal cleanup.
    name();operations_.now();const auto admission=operations_.can_load();
    primary_=create(admission.device);operations_.now();destroy(0);
}
void TtsPool::warm(){
    const auto target=std::max<int64_t>(1,capacity_);name();size_t initial;{std::lock_guard<std::recursive_mutex> lock(mutex_);initial=resident_.size();}
    // Source warm-load does not hold the pool lock while constructing models.
    for(int64_t slot=initial+1;slot<=target;++slot)try{auto model=load_instance();std::lock_guard<std::recursive_mutex> lock(mutex_);resident_.push_back(model);primary_=resident_.front();available_.push_back(model);register_locked();}catch(const TtsError &error){if(base_exception(error))throw;break;}catch(const std::exception &){break;}
}
TtsPoolHandle TtsPool::get(uint64_t key,bool force_cpu){
    TtsPoolHandle model;{
        std::lock_guard<std::recursive_mutex> lock(mutex_);auto lease=std::find_if(leases_.begin(),leases_.end(),[&](const auto &item){return item.first==key;});if(lease!=leases_.end()&&lease->second){model=lease->second;operations_.mark_in_use(true);return model;}
        if(!available_.empty()){model=available_.front();available_.pop_front();}
        else if(operations_.should_preload()){
            if(static_cast<int64_t>(resident_.size())<capacity_){model=load_instance(force_cpu);resident_.push_back(model);primary_=resident_.front();register_locked();}
            else if(!resident_.empty())model=resident_.front();
            else{model=load_instance(force_cpu);resident_.push_back(model);primary_=model;register_locked();}
        }else{model=load_instance(force_cpu);transient_.push_back(model);if(!primary_)primary_=model;register_locked();}
        lease=std::find_if(leases_.begin(),leases_.end(),[&](const auto &item){return item.first==key;});if(lease==leases_.end())leases_.emplace_back(key,model);else lease->second=model;
    }operations_.mark_in_use(true);return model;
}
void TtsPool::transfer(uint64_t owner,uint64_t consumer){
    if(owner==consumer)return;std::lock_guard<std::recursive_mutex> lock(mutex_);auto from=std::find_if(leases_.begin(),leases_.end(),[&](const auto &item){return item.first==owner;});if(from==leases_.end())throw TtsError("KeyError",std::to_string(owner));auto model=from->second;leases_.erase(from);
    const auto to=std::find_if(leases_.begin(),leases_.end(),[&](const auto &item){return item.first==consumer;});if(to==leases_.end())leases_.emplace_back(consumer,model);else to->second=model;
}
bool TtsPool::unload_idle() {
    std::vector<TtsPoolHandle> refs;
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        if (!leases_.empty() || (resident_.empty() && transient_.empty())) return false;
        refs = resident_; refs.insert(refs.end(), transient_.begin(), transient_.end());
        resident_.clear(); transient_.clear(); available_.clear(); primary_.reset(); register_locked();
    }
    for (const auto &model : refs) cleanup_sync(model, operations_);
    return true;
}
void TtsPool::destroy(uint64_t key,bool async_cleanup,bool force){
    if(operations_.handoff_active()){async_cleanup=false;force=true;}operations_.mark_in_use(false);std::vector<TtsPoolHandle> refs;{
        std::lock_guard<std::recursive_mutex> lock(mutex_);const auto lease=std::find_if(leases_.begin(),leases_.end(),[&](const auto &item){return item.first==key;});TtsPoolHandle model;if(lease!=leases_.end()){model=lease->second;leases_.erase(lease);}
        if(model&&operations_.should_preload()&&!force&&!closed(model)){available_.push_back(model);return;}
        if(force&&!model){refs=resident_;refs.insert(refs.end(),transient_.begin(),transient_.end());resident_.clear();transient_.clear();available_.clear();primary_.reset();register_locked();}
        else if(model){erase_first(resident_,model);erase_first(transient_,model);primary_=resident_.empty()?TtsPoolHandle{}:resident_.front();register_locked();refs.push_back(model);}
        else if(primary_){model=primary_;erase_first(resident_,model);primary_=resident_.empty()?TtsPoolHandle{}:resident_.front();register_locked();refs.push_back(model);}
    }
    for(const auto &model:refs){if(async_cleanup)operations_.schedule_cleanup([model,ops=operations_]{cleanup_sync(model,ops);});else cleanup_sync(model,operations_);}
}
void TtsPool::cleanup_sync(TtsPoolHandle model,const TtsPoolOperations &ops){
    try{ops.now();if(model&&model->close)model->close();model.reset();ops.collect();if(ops.cuda_available()){try{ops.synchronize();}catch(const TtsError &error){if(base_exception(error))throw;}catch(const std::exception &){}ops.empty_cache();try{ops.ipc_collect();}catch(const TtsError &error){if(base_exception(error))throw;}catch(const std::exception &){}}ops.now();}catch(const TtsError &error){if(base_exception(error))throw;}catch(const std::exception &){}
}
TtsPoolSnapshot TtsPool::snapshot() const{std::lock_guard<std::recursive_mutex> lock(mutex_);return {capacity_,primary_,resident_,transient_,{available_.begin(),available_.end()},leases_};}
}
