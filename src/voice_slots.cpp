#include "ezlocalai/voice_slots.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <exception>
#include <map>

namespace ezlocalai {
struct VoiceCancellation::State {
    std::mutex mutex;
    bool cancelled = false;
    uint64_t next = 1;
    std::map<uint64_t,std::function<void()>> wakeups;
};
VoiceCancellation::VoiceCancellation() : state_(std::make_shared<State>()) {}
bool VoiceCancellation::requested() const { std::lock_guard<std::mutex> lock(state_->mutex); return state_->cancelled; }
uint64_t VoiceCancellation::subscribe(std::function<void()> wake) const {
    std::lock_guard<std::mutex> lock(state_->mutex); if (state_->cancelled) return 0;
    const auto id = state_->next++; state_->wakeups.emplace(id,std::move(wake)); return id;
}
void VoiceCancellation::unsubscribe(uint64_t id) const { std::lock_guard<std::mutex> lock(state_->mutex); state_->wakeups.erase(id); }
void VoiceCancellation::cancel() const {
    std::vector<std::function<void()>> wakeups;
    { std::lock_guard<std::mutex> lock(state_->mutex); if (state_->cancelled) return; state_->cancelled = true; for (const auto &entry : state_->wakeups) wakeups.push_back(entry.second); }
    for (const auto &wake : wakeups) wake();
}
LinkedVoiceCancellation::LinkedVoiceCancellation(const VoiceCancellation &first,const VoiceCancellation &second):first_(first),second_(second){
    a_=first_.subscribe([token=combined_]{token.cancel();});
    try{b_=second_.subscribe([token=combined_]{token.cancel();});}catch(...){first_.unsubscribe(a_);throw;}
    if(!a_||!b_)combined_.cancel();
}
LinkedVoiceCancellation::~LinkedVoiceCancellation(){first_.unsubscribe(a_);second_.unsubscribe(b_);}
struct VoicePermitQueue::State {
    mutable std::mutex mutex;
    std::condition_variable changed;
    int64_t available;
    bool binary;
    uint64_t next = 1;
    std::deque<uint64_t> waiters;
    State(int64_t value,bool lock) : available(value),binary(lock) {}
};
VoicePermitQueue::VoicePermitQueue(int64_t permits,bool binary_lock) : state_(std::make_shared<State>(permits,binary_lock)) {
    if (permits < 0 || (binary_lock && permits != 1)) throw std::invalid_argument("Invalid voice admission capacity");
}
void VoicePermitQueue::acquire(const VoiceCancellation &cancellation) {
    const auto state = state_; std::unique_lock<std::mutex> lock(state->mutex);
    if (cancellation.requested()) throw VoiceCancelled("");
    if (state->available && state->waiters.empty()) { --state->available; return; }
    const auto ticket = state->next++; state->waiters.push_back(ticket);
    const auto subscription = cancellation.subscribe([weak = std::weak_ptr<State>(state)] {
        if (const auto state = weak.lock()) { std::lock_guard<std::mutex> lock(state->mutex); state->changed.notify_all(); }
    });
    try {
        state->changed.wait(lock,[&] { return cancellation.requested() || (state->available && !state->waiters.empty() && state->waiters.front() == ticket); });
        if (cancellation.requested()) throw VoiceCancelled("");
        --state->available; state->waiters.pop_front();
    } catch (...) {
        state->waiters.erase(std::find(state->waiters.begin(),state->waiters.end(),ticket));
        cancellation.unsubscribe(subscription); state->changed.notify_all(); throw;
    }
    cancellation.unsubscribe(subscription); state->changed.notify_all();
}
void VoicePermitQueue::release() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->binary && state_->available) throw std::runtime_error("Lock is not acquired.");
    ++state_->available; state_->changed.notify_all();
}
VoicePermitSnapshot VoicePermitQueue::snapshot() const {
    std::lock_guard<std::mutex> lock(state_->mutex); return {state_->available,{state_->waiters.begin(),state_->waiters.end()}};
}
VoiceSlotState VoiceSlotLease::snapshot() const { std::lock_guard<std::mutex> lock(mutex_); return state_; }
namespace {
bool truth(const std::string &encoded) {
    const auto value = nlohmann::json::parse(encoded);
    if (value.is_null()) return false; if (value.is_boolean()) return value.get<bool>();
    if (value.is_number()) return value.get<double>() != 0; if (value.is_string()) return !value.get_ref<const std::string &>().empty(); return !value.empty();
}
}
VoiceSlotGuard::VoiceSlotGuard(std::string service,int64_t capacity,std::shared_ptr<VoicePermitQueue> shared,VoiceSlotOperations ops)
    : service_(std::move(service)),reload_env_(service_),operations_(std::move(ops)),permits_(std::make_shared<VoicePermitQueue>(std::max<int64_t>(1,capacity))) {
    for (auto &c : reload_env_) if (c >= 'a' && c <= 'z') c -= 'a' - 'A'; reload_env_ += "_RELOAD_LLM_AFTER_GENERATION";
    if (!shared) throw std::invalid_argument("Voice handoff admission is not configured");
    if (bool(operations_.acquire_shared) != bool(operations_.release_shared) || bool(operations_.acquire_permit) != bool(operations_.release_permit)) throw std::invalid_argument("Voice admission overrides require acquire and release callbacks");
    if (!operations_.acquire_shared) { operations_.acquire_shared = [shared](const auto &cancel) { shared->acquire(cancel); }; operations_.release_shared = [shared] { shared->release(); }; }
    if (!operations_.acquire_permit) { operations_.acquire_permit = [queue = permits_](const auto &cancel) { queue->acquire(cancel); }; operations_.release_permit = [queue = permits_] { queue->release(); }; }
    if (!operations_.should_handoff) operations_.should_handoff = [](const auto &) { return false; };
    if (!operations_.mark_handoff) operations_.mark_handoff = [](const auto &,bool) {};
    if (!operations_.set_unavailable) operations_.set_unavailable = [](bool) {};
    if (!operations_.wait_for_idle) operations_.wait_for_idle = [](const auto &,const auto &) {};
    if (!operations_.unload) operations_.unload = [](const auto &,bool) { return "null"; };
    if (!operations_.destroy) operations_.destroy = [](const auto &,bool,bool) {};
    if (!operations_.restore) operations_.restore = [](const auto &,const auto &,const auto &) {};
}
void VoiceSlotGuard::destroy_and_restore(const VoiceSlotState &state) {
    operations_.destroy(service_ == "tts" ? "tts" : "stt",false,true);
    operations_.restore(service_,state.handoff_state,reload_env_);
}
void VoiceSlotGuard::finish_handoff() { operations_.mark_handoff(service_,false); operations_.set_unavailable(false); }
VoiceSlotHandle VoiceSlotGuard::acquire(const VoiceCancellation &cancellation) {
    const bool handoff = operations_.should_handoff(service_);
    auto lease = std::make_shared<VoiceSlotLease>(); auto &state = lease->state_; state.handoff = handoff;
    try {
        if (handoff) {
            operations_.acquire_shared(cancellation); state.shared_lock = true;
            operations_.mark_handoff(service_,true); operations_.set_unavailable(true);
            operations_.wait_for_idle(service_,cancellation); state.handoff_state = operations_.unload(service_,true);
        }
        operations_.acquire_permit(cancellation); state.semaphore = true; return lease;
    } catch (...) {
        auto failure = std::current_exception();
        try { if (handoff && truth(state.handoff_state)) destroy_and_restore(state); } catch (...) { failure = std::current_exception(); }
        if (state.shared_lock) finish_handoff();
        if (state.shared_lock) operations_.release_shared();
        std::rethrow_exception(failure);
    }
}
void VoiceSlotGuard::release(const VoiceSlotHandle &lease) {
    if (!lease) return; VoiceSlotState state;
    { std::lock_guard<std::mutex> lock(lease->mutex_); if (lease->state_.released.value_or(false)) return; lease->state_.released = true; state = lease->state_; }
    std::exception_ptr failure;
    try { if (state.handoff) destroy_and_restore(state); } catch (...) { failure = std::current_exception(); }
    if (state.handoff) finish_handoff();
    if (state.semaphore) operations_.release_permit();
    if (state.shared_lock) operations_.release_shared();
    if (failure) std::rethrow_exception(failure);
}
VoiceSlotGuard &VoiceSlotGuard::enter(uint64_t key,const VoiceCancellation &cancellation) {
    auto lease = acquire(cancellation); std::lock_guard<std::mutex> lock(mutex_);
    auto at = std::find_if(stacks_.begin(),stacks_.end(),[&](const auto &item) { return item.first == key; });
    if (at == stacks_.end()) stacks_.emplace_back(key,std::vector<VoiceSlotHandle>{std::move(lease)}); else at->second.push_back(std::move(lease)); return *this;
}
bool VoiceSlotGuard::exit(uint64_t key) {
    VoiceSlotHandle lease;
    { std::lock_guard<std::mutex> lock(mutex_); const auto at = std::find_if(stacks_.begin(),stacks_.end(),[&](const auto &item) { return item.first == key; });
        if (at != stacks_.end()) { if (!at->second.empty()) { lease = at->second.back(); at->second.pop_back(); } if (at->second.empty()) stacks_.erase(at); } }
    release(lease); return false;
}
std::vector<std::pair<uint64_t,std::vector<VoiceSlotHandle>>> VoiceSlotGuard::stacks() const { std::lock_guard<std::mutex> lock(mutex_); return stacks_; }
VoicePermitSnapshot VoiceSlotGuard::permits() const { return permits_->snapshot(); }
}
