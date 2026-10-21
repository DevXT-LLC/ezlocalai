#pragma once
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <vector>

namespace ezlocalai {
bool embedding_keep_loaded();
bool embedding_should_preload();
int embedding_parallel_slots();

class EmbeddingPool {
public:
    using Handle = uint64_t;
    using Load = std::function<Handle(size_t)>;
    using Destroy = std::function<void(Handle)>;
    struct Snapshot {
        size_t capacity, acquired, waiting;
        std::vector<Handle> resident, transient, available;
    };
    using Changed = std::function<void(const Snapshot &)>;
    EmbeddingPool(size_t capacity, Load load, Destroy destroy,
                  std::function<bool()> keep_loaded = embedding_keep_loaded, Changed changed = {});
    Handle acquire();
    void release(Handle handle);
    void warm();
    bool unload_idle();
    Snapshot snapshot() const;
private:
    const size_t capacity_;
    Load load_;
    Destroy destroy_;
    std::function<bool()> keep_loaded_;
    Changed residency_changed_;
    mutable std::mutex mutex_;
    std::mutex load_mutex_;
    std::condition_variable changed_;
    size_t acquired_ = 0;
    uint64_t next_ticket_ = 0;
    std::deque<uint64_t> waiters_;
    std::deque<Handle> available_;
    std::vector<Handle> resident_, transient_;
    void return_permit();
    void destroy(Handle handle) noexcept;
    void notify_residency_locked();
};
}
