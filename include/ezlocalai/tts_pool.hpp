#pragma once
#include "ezlocalai/ctts.hpp"
#include "ezlocalai/resource_manager.hpp"
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <vector>
#include <mutex>
namespace ezlocalai {
struct TtsPoolModel {
    std::function<std::string()> device;
    std::function<bool()> closed;
    std::function<void()> close;
    std::function<std::string(const std::string &)> generate;
    std::function<std::unique_ptr<CttsByteStream>(const std::string &)> stream;
    std::function<std::vector<std::string>()> voices;
    std::function<std::string()> sample_rate;
};
using TtsPoolHandle=std::shared_ptr<TtsPoolModel>;
struct TtsPoolOperations {
    std::function<std::optional<std::string>(const std::string &)> environment;
    std::function<bool()> available,should_preload,handoff_active,cuda_available;
    std::function<ModelAdmission()> can_load;
    std::function<TtsPoolHandle(const std::optional<std::string> &)> load;
    std::function<void(const std::string &,const std::string &,double)> register_model;
    std::function<void()> unregister_model,collect,synchronize,empty_cache,ipc_collect;
    std::function<void(bool)> mark_in_use;
    std::function<double()> now;
    std::function<void(std::function<void()>)> schedule_cleanup;
};
struct TtsPoolSnapshot {
    int64_t capacity;
    TtsPoolHandle primary;
    std::vector<TtsPoolHandle> resident,transient,available;
    std::vector<std::pair<uint64_t,TtsPoolHandle>> leases;
};
using TtsPoolFactory=std::function<TtsPoolOperations(std::shared_ptr<ResourceManager>)>;
TtsPoolOperations tts_pool_resources(TtsPoolOperations operations,std::shared_ptr<ResourceManager> resources);
TtsPoolOperations tts_pool_ctts_operations(CttsServiceOperations service={});
// Explicit request identities allow a streaming consumer to take over a lease.
// Acquiring a pool lease does not acquire a voice permit; compose VoiceSlotGuard.
// Model callbacks retain CTTS state; stream handles also retain their owner.
class TtsPool {
public:
    TtsPool(int64_t capacity,TtsPoolOperations operations={});
    std::string name() const;
    TtsPoolHandle create(const std::optional<std::string> &device={},bool force_cpu=false);
    TtsPoolHandle load_instance(bool force_cpu=false);
    void warm();
    void warm_cache();
    TtsPoolHandle get(uint64_t lease_key,bool force_cpu=false);
    void transfer(uint64_t owner,uint64_t consumer);
    void destroy(uint64_t lease_key,bool async_cleanup=true,bool force=false);
    void register_resources();
    TtsPoolSnapshot snapshot() const;
    bool unload_idle();
    static void cleanup_sync(TtsPoolHandle model,const TtsPoolOperations &operations);
private:
    void register_locked();
    int64_t capacity_;
    TtsPoolOperations operations_;
    mutable std::recursive_mutex mutex_;
    TtsPoolHandle primary_;
    std::vector<TtsPoolHandle> resident_,transient_;
    std::deque<TtsPoolHandle> available_;
    std::vector<std::pair<uint64_t,TtsPoolHandle>> leases_;
};
}
