#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ezlocalai {
struct VoiceCancelled : std::runtime_error { using std::runtime_error::runtime_error; };
// Copies refer to the same cancellation request. A blocked permit acquisition
// registers a wakeup and removes it before returning or throwing.
class VoiceCancellation {
public:
    VoiceCancellation();
    void cancel() const;
    bool requested() const;
private:
    struct State;
    std::shared_ptr<State> state_;
    uint64_t subscribe(std::function<void()> wake) const;
    void unsubscribe(uint64_t id) const;
    friend class VoicePermitQueue;
    friend class LinkedVoiceCancellation;
};
// Scoped subscription to two independent owners, such as a service and a job.
class LinkedVoiceCancellation {
public:
    LinkedVoiceCancellation(const VoiceCancellation &first,const VoiceCancellation &second);
    ~LinkedVoiceCancellation();
    LinkedVoiceCancellation(const LinkedVoiceCancellation &)=delete;
    LinkedVoiceCancellation &operator=(const LinkedVoiceCancellation &)=delete;
    const VoiceCancellation &token() const { return combined_; }
private:
    VoiceCancellation first_,second_,combined_;
    uint64_t a_=0,b_=0;
};
struct VoicePermitSnapshot { int64_t available; std::vector<uint64_t> waiters; };
// FIFO admission without thread ownership: a different request/thread may
// release an acquired lease, as with asyncio.Lock and asyncio.Semaphore.
class VoicePermitQueue {
public:
    explicit VoicePermitQueue(int64_t permits,bool binary_lock = false);
    void acquire(const VoiceCancellation &cancellation = {});
    void release();
    VoicePermitSnapshot snapshot() const;
private:
    struct State;
    std::shared_ptr<State> state_;
};
struct VoiceSlotState {
    bool handoff = false, shared_lock = false, semaphore = false;
    std::string handoff_state = "null";
    std::optional<bool> released;
};
class VoiceSlotLease {
public:
    VoiceSlotState snapshot() const;
private:
    mutable std::mutex mutex_;
    VoiceSlotState state_;
    friend class VoiceSlotGuard;
};
using VoiceSlotHandle = std::shared_ptr<VoiceSlotLease>;
struct VoiceSlotOperations {
    std::function<bool(const std::string &)> should_handoff;
    std::function<void(const VoiceCancellation &)> acquire_shared, acquire_permit;
    std::function<void()> release_shared, release_permit;
    std::function<void(const std::string &,bool)> mark_handoff;
    std::function<void(bool)> set_unavailable;
    std::function<void(const std::string &,const VoiceCancellation &)> wait_for_idle;
    // Handoff data is opaque source-shaped JSON, passed unchanged to restore.
    std::function<std::string(const std::string &,bool)> unload;
    std::function<void(const std::string &,bool,bool)> destroy;
    std::function<void(const std::string &,const std::string &,const std::string &)> restore;
};
class VoiceSlotGuard {
public:
    VoiceSlotGuard(std::string service,int64_t capacity,
        std::shared_ptr<VoicePermitQueue> shared_handoff,VoiceSlotOperations operations);
    VoiceSlotHandle acquire(const VoiceCancellation &cancellation = {});
    void release(const VoiceSlotHandle &lease);
    VoiceSlotGuard &enter(uint64_t key,const VoiceCancellation &cancellation = {});
    bool exit(uint64_t key);
    std::vector<std::pair<uint64_t,std::vector<VoiceSlotHandle>>> stacks() const;
    VoicePermitSnapshot permits() const;
private:
    void destroy_and_restore(const VoiceSlotState &state);
    void finish_handoff();
    std::string service_,reload_env_;
    VoiceSlotOperations operations_;
    std::shared_ptr<VoicePermitQueue> permits_;
    mutable std::mutex mutex_;
    std::vector<std::pair<uint64_t,std::vector<VoiceSlotHandle>>> stacks_;
};
}
