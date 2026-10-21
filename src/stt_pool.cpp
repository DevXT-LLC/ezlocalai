#include "ezlocalai/stt_pool.hpp"
#include "ezlocalai/resource_manager.hpp"
#include <algorithm>
#include <condition_variable>
#include <iostream>
#include <thread>

namespace ezlocalai {
namespace {
struct CleanupQueue {
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::function<void()>> work;
    CleanupQueue() {
        std::thread([this] {
            for (;;) {
                std::function<void()> job;
                { std::unique_lock<std::mutex> lock(mutex); changed.wait(lock,[&] { return !work.empty(); }); job = std::move(work.front()); work.pop_front(); }
                try { job(); } catch (const std::exception &error) { std::cerr << "[Cleanup] Background cleanup error: " << error.what() << '\n'; } catch (...) {}
            }
        }).detach();
    }
};
std::string device(const SttPoolHandle &model) { return model && model->device ? model->device() : "cpu"; }
void erase_first(std::vector<SttPoolHandle> &models,const SttPoolHandle &model) { const auto at = std::find(models.begin(),models.end(),model); if (at != models.end()) models.erase(at); }
}
void schedule_stt_cleanup(std::function<void()> cleanup) {
    // The reference worker is a daemon. Its queue and synchronization objects
    // remain alive until process exit rather than racing static destruction.
    static auto *queue = new CleanupQueue;
    { std::lock_guard<std::mutex> lock(queue->mutex); queue->work.push_back(std::move(cleanup)); }
    queue->changed.notify_one();
}
SttPoolOperations stt_pool_resources(SttPoolOperations operations,std::shared_ptr<ResourceManager> resources) {
    operations.can_load = [resources] { resources->can_load_model(ModelType::Stt); };
    operations.register_model = [resources](const auto &name,const auto &device,double vram) { resources->register_model(ModelType::Stt,name,device,vram); };
    operations.unregister_model = [resources] { resources->unregister_model(ModelType::Stt); };
    operations.mark_in_use = [resources](bool active) { resources->mark_model_in_use(ModelType::Stt,active); };
    return operations;
}
SttPool::SttPool(std::string model,int64_t capacity,SttPoolOperations operations)
    : model_(std::move(model)),capacity_(capacity),operations_(std::move(operations)) {
    if (!operations_.should_preload) operations_.should_preload = [] { return false; };
    if (!operations_.handoff_active) operations_.handoff_active = [] { return false; };
    if (!operations_.cuda_available) operations_.cuda_available = [] { return false; };
    for (auto *operation : {&operations_.can_load,&operations_.unregister_model,&operations_.collect,&operations_.synchronize,&operations_.empty_cache,&operations_.ipc_collect})
        if (!*operation) *operation = [] {};
    if (!operations_.register_model) operations_.register_model = [](const auto &,const auto &,double) {};
    if (!operations_.mark_in_use) operations_.mark_in_use = [](bool) {};
    if (!operations_.schedule_cleanup) operations_.schedule_cleanup = schedule_stt_cleanup;
}
SttPoolHandle SttPool::load_instance() {
    // can_load_model and force_cpu affect the source's logging only. STT itself
    // chooses its device; neither the admission result nor a CPU hint overrides it.
    operations_.can_load(); return operations_.load(model_);
}
void SttPool::register_locked() {
    auto loaded = resident_; loaded.insert(loaded.end(),transient_.begin(),transient_.end());
    if (loaded.empty()) { operations_.unregister_model(); return; }
    bool any_cuda = false;
    for (const auto &model : loaded) if (device(model).find("cuda") != std::string::npos) { any_cuda = true; break; }
    double vram = 0; for (const auto &model : loaded) if (device(model).find("cuda") != std::string::npos) vram += 2;
    operations_.register_model(model_,any_cuda ? "cuda" : device(loaded.front()),vram);
}
void SttPool::register_resources() { std::lock_guard<std::recursive_mutex> lock(mutex_); register_locked(); }
void SttPool::warm() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    const auto target = std::max<int64_t>(1,capacity_);
    const auto initial = resident_.size();
    for (int64_t slot = initial + 1; slot <= target; ++slot) {
        try { auto model = load_instance(); resident_.push_back(model); primary_ = resident_.front(); available_.push_back(model); register_locked(); }
        catch (const std::exception &error) { std::cerr << "[STT] Failed to warm-load instance " << slot << '/' << target << ": " << error.what() << '\n'; break; }
    }
}
SttPoolHandle SttPool::get(uint64_t key,bool force_cpu) {
    (void)force_cpu; SttPoolHandle model;
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        auto lease = std::find_if(leases_.begin(),leases_.end(),[&](const auto &item) { return item.first == key; });
        if (lease != leases_.end() && lease->second) { operations_.mark_in_use(true); return lease->second; }
        if (!available_.empty()) { model = available_.front(); available_.pop_front(); }
        else if (operations_.should_preload()) {
            if (static_cast<int64_t>(resident_.size()) < capacity_) { model = load_instance(); resident_.push_back(model); primary_ = resident_.front(); register_locked(); }
            else if (!resident_.empty()) model = resident_.front();
            else { model = load_instance(); resident_.push_back(model); primary_ = model; register_locked(); }
        } else { model = load_instance(); transient_.push_back(model); if (!primary_) primary_ = model; register_locked(); }
        if (lease == leases_.end()) leases_.emplace_back(key,model); else lease->second = model;
    }
    operations_.mark_in_use(true); return model;
}
bool SttPool::unload_idle() {
    std::vector<SttPoolHandle> refs;
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        if (!leases_.empty() || (resident_.empty() && transient_.empty())) return false;
        refs = resident_; refs.insert(refs.end(), transient_.begin(), transient_.end());
        resident_.clear(); transient_.clear(); available_.clear(); primary_.reset(); register_locked();
    }
    for (const auto &model : refs) cleanup_sync(model, operations_);
    return true;
}
void SttPool::destroy(uint64_t key,bool async_cleanup,bool force) {
    if (operations_.handoff_active()) { async_cleanup = false; force = true; }
    operations_.mark_in_use(false); std::vector<SttPoolHandle> refs;
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        const auto lease = std::find_if(leases_.begin(),leases_.end(),[&](const auto &item) { return item.first == key; });
        SttPoolHandle model; if (lease != leases_.end()) { model = lease->second; leases_.erase(lease); }
        if (model && operations_.should_preload() && !force) { available_.push_back(model); return; }
        if (force && !model) { refs = resident_; refs.insert(refs.end(),transient_.begin(),transient_.end()); resident_.clear(); transient_.clear(); available_.clear(); primary_.reset(); register_locked(); }
        else if (model) { erase_first(resident_,model); erase_first(transient_,model); primary_ = resident_.empty() ? SttPoolHandle{} : resident_.front(); register_locked(); refs.push_back(model); }
        else if (primary_) { model = primary_; erase_first(resident_,model); primary_ = resident_.empty() ? SttPoolHandle{} : resident_.front(); register_locked(); refs.push_back(model); }
    }
    for (const auto &model : refs) {
        if (async_cleanup) operations_.schedule_cleanup([model,operations = operations_] { cleanup_sync(model,operations); });
        else cleanup_sync(model,operations_);
    }
}
void SttPool::cleanup_sync(SttPoolHandle model,const SttPoolOperations &operations) noexcept {
    try {
        if (model && model->close) model->close(); model.reset(); operations.collect();
        if (operations.cuda_available()) {
            try { operations.synchronize(); } catch (...) {}
            operations.empty_cache(); try { operations.ipc_collect(); } catch (...) {}
        }
    } catch (const std::exception &error) { std::cerr << "[STT] Error during cleanup: " << error.what() << '\n'; } catch (...) {}
}
SttPoolSnapshot SttPool::snapshot() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return {capacity_,primary_,resident_,transient_,{available_.begin(),available_.end()},leases_};
}
}
