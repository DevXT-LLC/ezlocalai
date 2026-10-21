#pragma once
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace ezlocalai {
class ResourceManager;
struct SttPoolModel {
    std::function<std::string()> device;
    std::function<void()> close;
    std::function<std::string(const std::string &)> transcribe;
    // Source audio-to-audio passes bytes, rather than a Unicode base64 string.
    std::function<std::string(const std::string &,const std::string &)> transcribe_bytes;
};
using SttPoolHandle = std::shared_ptr<SttPoolModel>;
struct SttPoolOperations {
    std::function<void()> can_load;
    std::function<SttPoolHandle(const std::string &)> load;
    std::function<bool()> should_preload, handoff_active, cuda_available;
    std::function<void(const std::string &,const std::string &,double)> register_model;
    std::function<void()> unregister_model, collect, synchronize, empty_cache, ipc_collect;
    std::function<void(bool)> mark_in_use;
    std::function<void(std::function<void()>)> schedule_cleanup;
};
struct SttPoolSnapshot {
    int64_t capacity;
    SttPoolHandle primary;
    std::vector<SttPoolHandle> resident, transient, available;
    std::vector<std::pair<uint64_t,SttPoolHandle>> leases;
};
// One process-wide FIFO cleanup worker, matching Pipes._schedule_cleanup.
void schedule_stt_cleanup(std::function<void()> cleanup);
SttPoolOperations stt_pool_resources(SttPoolOperations operations,std::shared_ptr<ResourceManager> resources);
class SttPool {
public:
    SttPool(std::string model,int64_t capacity,SttPoolOperations operations);
    void warm();
    SttPoolHandle get(uint64_t lease_key,bool force_cpu = false);
    void destroy(uint64_t lease_key,bool async_cleanup = true,bool force = false);
    SttPoolSnapshot snapshot() const;
    bool unload_idle();
    void register_resources();
    static void cleanup_sync(SttPoolHandle model,const SttPoolOperations &operations) noexcept;
private:
    SttPoolHandle load_instance();
    void register_locked();
    std::string model_;
    int64_t capacity_;
    SttPoolOperations operations_;
    mutable std::recursive_mutex mutex_;
    SttPoolHandle primary_;
    std::vector<SttPoolHandle> resident_, transient_;
    std::deque<SttPoolHandle> available_;
    std::vector<std::pair<uint64_t,SttPoolHandle>> leases_;
};
}
