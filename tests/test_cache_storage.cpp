#include "server-task.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <stdexcept>

namespace allocation_probe {
bool reject_large = false;
size_t threshold = 64 * 1024;
size_t large_calls = 0;
}

void * operator new(std::size_t count) {
    if (allocation_probe::reject_large && count >= allocation_probe::threshold) {
        ++allocation_probe::large_calls;
        throw std::bad_alloc();
    }
    if (void * result = std::malloc(count ? count : 1)) return result;
    throw std::bad_alloc();
}
void * operator new[](std::size_t count) { return ::operator new(count); }
void operator delete(void * ptr) noexcept { std::free(ptr); }
void operator delete[](void * ptr) noexcept { std::free(ptr); }
void operator delete(void * ptr, std::size_t) noexcept { std::free(ptr); }
void operator delete[](void * ptr, std::size_t) noexcept { std::free(ptr); }

namespace {
void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}
struct allocation_failure {
    size_t saved_threshold;
    explicit allocation_failure(size_t threshold = allocation_probe::threshold) : saved_threshold(allocation_probe::threshold) {
        allocation_probe::threshold = threshold;
        allocation_probe::large_calls = 0;
        allocation_probe::reject_large = true;
    }
    ~allocation_failure() { allocation_probe::reject_large = false; allocation_probe::threshold = saved_threshold; }
};
struct context_tag {} target_tag, draft_tag;
llama_context * target() { return reinterpret_cast<llama_context *>(&target_tag); }
llama_context * draft() { return reinterpret_cast<llama_context *>(&draft_tag); }

struct backend_plan {
    size_t main_size = 128 * 1024;
    size_t draft_size = 64 * 1024;
    uint8_t main_byte = 0xa1;
    uint8_t draft_byte = 0xd2;
    bool fail_restore = false;
    bool short_save = false;
    int restores = 0;
} plan;

server_prompt prompt(int branch, size_t checkpoint_bytes = 0) {
    llama_tokens tokens;
    for (int i = 0; i < 128; ++i) tokens.push_back(i < 64 ? i : branch * 1000 + i);
    server_prompt result{server_tokens(tokens, false), {}};
    if (checkpoint_bytes) {
        auto & checkpoint = result.checkpoints.emplace_back();
        checkpoint.n_tokens = 32;
        checkpoint.pos_min = 0;
        checkpoint.pos_max = 31;
        checkpoint.data_tgt.resize(checkpoint_bytes);
        std::memset(checkpoint.data_tgt.data(), 0x4c, checkpoint_bytes);
        checkpoint.data_dft = {4, 5, 6};
        checkpoint.data_spec = {7, 8, 9};
    }
    return result;
}

const uint8_t * checkpoint_data(const server_prompt & value) {
    return value.checkpoints.front().data_tgt.data();
}

server_prompt_cache_state * save(server_prompt_cache & cache, const server_prompt & value, bool with_draft = true) {
    check(cache.save(value, target(), with_draft ? draft() : nullptr, 7), "valid snapshot save failed");
    check(!cache.states.empty(), "successful snapshot save published no cached state");
    return &cache.states.back();
}

void checkpoint_copy_and_write() {
    auto original = prompt(1, 128 * 1024);
    auto copy = original.clone();
    check(checkpoint_data(original) == checkpoint_data(copy), "checkpoint clone must share immutable bytes");
    auto & changed = copy.checkpoints.front();
    changed.data_tgt.data()[17] = 0x9d;
    check(checkpoint_data(original) != checkpoint_data(copy), "checkpoint writer must detach shared bytes");
    check(checkpoint_data(original)[17] == 0x4c && checkpoint_data(copy)[17] == 0x9d,
        "checkpoint mutation damaged its other owner");
    changed.data_dft.clear();
    check(!original.checkpoints.front().data_dft.empty() && changed.data_dft.empty(), "checkpoint clear mutated a shared owner");
    changed.data_tgt.resize(256 * 1024);
    check(changed.data_tgt.size() == 256 * 1024 && original.checkpoints.front().data_tgt.size() == 128 * 1024,
        "checkpoint resize changed another owner");
}

void actual_checkpoint_update_and_restore() {
    common_prompt_checkpoint original{};
    original.update_tgt(target(), 7, LLAMA_STATE_SEQ_FLAGS_NONE);
    auto copy = original;
    check(static_cast<const common_prompt_checkpoint &>(copy).data_tgt.data() ==
        static_cast<const common_prompt_checkpoint &>(original).data_tgt.data(), "production checkpoint copy deep-copied bytes");
    plan.main_byte = 0xb3;
    copy.update_tgt(target(), 7, LLAMA_STATE_SEQ_FLAGS_NONE);
    check(static_cast<const common_prompt_checkpoint &>(original).data_tgt.data()[0] == 0xa1 &&
        static_cast<const common_prompt_checkpoint &>(copy).data_tgt.data()[0] == 0xb3,
        "production checkpoint update overwrote shared bytes");
    check(copy.load_tgt(target(), 7, LLAMA_STATE_SEQ_FLAGS_NONE), "production checkpoint restore changed bytes");
    plan.main_byte = 0xa1;
    check(original.load_tgt(target(), 7, LLAMA_STATE_SEQ_FLAGS_NONE), "original checkpoint bytes were not preserved");
}

void checkpoint_copy_without_large_allocation() {
    auto value = prompt(1, 256 * 1024);
    server_prompt_cache cache(2, 4096);
    server_prompt_cache_state * entry = nullptr;
    {
        allocation_failure fail;
        entry = cache.alloc(value, 32, 16);
    }
    check(entry && allocation_probe::large_calls == 0, "cache cloning checkpoint bytes still needs a large allocation");
    check(checkpoint_data(entry->prompt) == checkpoint_data(value), "cached checkpoint payload is not shared");
}

void shared_checkpoint_branches_fit_budget() {
    auto a = prompt(1, 1024 * 1024), b = prompt(2);
    a.checkpoints.front().data_spec.resize(16 * 1024);
    b.checkpoints = a.checkpoints;
    server_prompt_cache cache(2, 4096);
    std::vector<uint8_t> spec_a(32 * 1024), spec_b(48 * 1024);
    check(cache.save(a, target(), draft(), 7, nullptr, std::move(spec_a)), "first shared branch save failed");
    check(cache.states.size() == 1, "first shared branch save published no cached state");
    auto * first = &cache.states.front();
    check(cache.save(b, target(), draft(), 7, nullptr, std::move(spec_b)), "second shared branch save failed");
    check(cache.states.size() == 2 && &cache.states.front() == first,
        "shared checkpoint backing needlessly evicted an existing branch");
    const auto & second = cache.states.back();
    const auto & checkpoint = first->prompt.checkpoints.front();
    const size_t expected = first->data.size() + second.data.size() + checkpoint.data_tgt.capacity() +
        checkpoint.data_dft.capacity() + checkpoint.data_spec.capacity() +
        second.prompt.checkpoints.front().data_spec.capacity();
    size_t accounted = 0;
    {
        allocation_failure fail(1);
        accounted = cache.size();
    }
    check(accounted == expected && cache.size() <= cache.limit_size &&
        first->size() + second.size() > cache.limit_size,
        "shared backing union lost full-state or CPU speculative capacity charges");
}

void protected_shared_checkpoint_save_fits() {
    auto a = prompt(1, 1024 * 1024), b = prompt(2), resident = prompt(3);
    b.checkpoints = a.checkpoints;
    server_prompt_cache cache(2, 4096);
    auto * protected_state = save(cache, a);
    check(cache.save(b, target(), draft(), 7, &a.tokens),
        "shared backing caused a false protected-state admission failure");
    check(cache.states.size() == 2 && &cache.states.front() == protected_state,
        "saving a shared branch displaced the pending restore");
    check(cache.load(resident, a.tokens, target(), draft(), 7), "protected shared branch restore failed");
    check(cache.states.size() == 1, "protected shared branch restore removed the wrong cached states");
    const auto & survivor = cache.states.front();
    check(cache.size() == survivor.size() + cache.spare.size() &&
        cache.size() <= cache.limit_size,
        "consumed shared snapshot spare escaped the byte budget");
}

void independent_checkpoints_do_not_bypass_protection() {
    auto a = prompt(1, 1024 * 1024), b = prompt(2, 1024 * 1024);
    check(checkpoint_data(a) != checkpoint_data(b), "independent checkpoint fixture unexpectedly shares backing");
    server_prompt_cache cache(2, 4096);
    auto * protected_state = save(cache, a);
    const size_t before = cache.size();
    check(!cache.save(b, target(), draft(), 7, &a.tokens),
        "equal bytes in independent allocations bypassed the protected-state limit");
    check(cache.states.size() == 1 && &cache.states.front() == protected_state && cache.size() == before,
        "denied independent checkpoint save disturbed the protected state");
}

void independent_checkpoint_union_evicts_oldest() {
    auto a = prompt(1, 1024 * 1024), b = prompt(2, 1024 * 1024), c = prompt(3, 1024 * 1024);
    server_prompt_cache cache(3, 4096);
    save(cache, a);
    auto * survivor = save(cache, b);
    check(cache.states.size() == 2, "independent checkpoint pair did not fit its byte budget");
    auto * newest = save(cache, c);
    check(cache.states.size() == 2 && &cache.states.front() == survivor && &cache.states.back() == newest,
        "independent checkpoint pressure failed to evict the actual oldest state");
    check(cache.size() == survivor->size() + newest->size() && cache.size() <= cache.limit_size,
        "independent checkpoint union lost distinct allocation charges");
}

void shared_identity_across_target_and_draft() {
    auto a = prompt(1, 1024 * 1024), b = prompt(2);
    a.checkpoints.front().data_dft = a.checkpoints.front().data_tgt;
    b.checkpoints = a.checkpoints;
    server_prompt_cache cache(2, 4096);
    auto * first = save(cache, a);
    auto * second = save(cache, b);
    check(cache.states.size() == 2 && &cache.states.front() == first && &cache.states.back() == second,
        "shared target/draft branches did not both survive admission");
    const auto & checkpoint = first->prompt.checkpoints.front();
    check(cache.size() == first->data.size() + second->data.size() +
        checkpoint.data_tgt.capacity() + checkpoint.data_spec.capacity() +
        second->prompt.checkpoints.front().data_spec.capacity(),
        "shared target/draft backing was charged more than once");
}

void shared_checkpoint_growth_and_release() {
    auto a = prompt(1, 1024 * 1024), b = prompt(2);
    b.checkpoints = a.checkpoints;
    server_prompt_cache cache(2, 4096);
    auto * first = save(cache, a);
    save(cache, b);
    check(cache.states.size() == 2 && &cache.states.front() == first,
        "shared branches did not survive before release accounting");
    const size_t both = cache.size();
    const size_t first_private = first->data.size() + first->prompt.checkpoints.front().data_spec.capacity();
    cache.states.pop_front();
    check(cache.size() == both - first_private,
        "removing one shared owner released backing still retained by another cached branch");
    auto * changed = save(cache, a);
    check(cache.states.size() == 2 && &cache.states.back() == changed,
        "shared branches did not survive before detached-growth accounting");
    const size_t before = cache.size();
    auto & checkpoint = changed->prompt.checkpoints.front();
    const size_t detached_capacity = checkpoint.data_tgt.capacity();
    checkpoint.data_tgt.data()[17] = 0x9d;
    check(cache.size() == before + detached_capacity &&
        checkpoint_data(cache.states.front().prompt)[17] == 0x4c,
        "detached checkpoint growth was uncharged or corrupted its shared owner");
    cache.update();
    check(cache.states.size() == 1 && &cache.states.front() == changed &&
        cache.size() == changed->size() && cache.size() <= cache.limit_size,
        "COW growth failed to enforce the byte limit");
    cache.states.clear();
    check(cache.size() == 0 && !a.checkpoints.empty() && !b.checkpoints.empty(),
        "last cached owner release charged checkpoints retained solely by live prompts");
}

void shared_checkpoint_snapshot_failures_preserve_budget() {
    auto a = prompt(1, 1024 * 1024), b = prompt(2), resident = prompt(3);
    b.checkpoints = a.checkpoints;
    server_prompt_cache cache(2, 4096);
    auto * protected_state = save(cache, a);
    const size_t before = cache.size();
    plan.short_save = true;
    check(!cache.save(b, target(), draft(), 7, &a.tokens), "partial shared snapshot was published");
    plan.short_save = false;
    check(cache.states.size() == 1 && &cache.states.front() == protected_state && cache.size() == before,
        "failed shared snapshot damaged the protected backing or its charge");
    auto * survivor = save(cache, b);
    check(cache.states.size() == 2 && &cache.states.front() == protected_state && &cache.states.back() == survivor,
        "shared branches did not survive before restore fault injection");
    plan.fail_restore = true;
    check(!cache.load(resident, a.tokens, target(), draft(), 7), "failed shared restore was accepted");
    plan.fail_restore = false;
    check(cache.states.size() == 1 && &cache.states.front() == survivor && cache.size() == survivor->size() &&
        cache.spare.size() == 0, "failed restore released surviving shared backing or recycled invalid buffers");
    check(cache.load(resident, b.tokens, target(), draft(), 7), "surviving shared branch restore failed");
    check(cache.states.empty() && cache.size() == cache.spare.size() && cache.size() > 0,
        "consumed shared snapshot buffers were not charged");
    plan.short_save = true;
    check(!cache.save(a, target(), draft(), 7), "partial recycled shared snapshot was published");
    plan.short_save = false;
    check(cache.states.empty() && cache.size() == 0, "failed recycled save retained snapshot or checkpoint charges");
}

void disabled_storage_shared_sources_still_need_independent_space() {
    auto a = prompt(1, 1024 * 1024), b = prompt(2);
    b.checkpoints = a.checkpoints;
    setenv("EZLOCALAI_LLAMA_CACHE_STORAGE", "0", 1);
    server_prompt_cache cache(2, 4096);
    auto * protected_state = save(cache, a);
    check(checkpoint_data(protected_state->prompt) != checkpoint_data(a),
        "storage opt-out reused source checkpoint backing");
    const size_t before = cache.size();
    check(!cache.save(b, target(), draft(), 7, &a.tokens),
        "storage opt-out treated independent checkpoint clones as shared");
    check(cache.states.size() == 1 && &cache.states.front() == protected_state && cache.size() == before,
        "storage opt-out admission disturbed protected state accounting");
    setenv("EZLOCALAI_LLAMA_CACHE_STORAGE", "1", 1);
}

void snapshot_reuse_without_large_allocation() {
    auto a = prompt(1), b = prompt(2), resident = prompt(3);
    server_prompt_cache cache(2, 4096);
    auto * entry = save(cache, a);
    const auto * main_ptr = entry->data.main.data();
    const auto * draft_ptr = entry->data.drft.data();
    const auto charged = entry->data.main.capacity() + entry->data.drft.capacity();
    check(cache.load(resident, a.tokens, target(), draft(), 7), "initial snapshot restore failed");
    check(cache.states.empty() && cache.size() == charged, "consumed snapshot storage is not charged to the cache budget");
    server_prompt_cache_state * reused = nullptr;
    {
        allocation_failure fail;
        reused = cache.alloc(b, plan.main_size, plan.draft_size);
    }
    check(reused && allocation_probe::large_calls == 0, "same-size cache save allocated fresh snapshot storage");
    check(reused->data.main.data() == main_ptr && reused->data.drft.data() == draft_ptr, "snapshot buffers were not reused");
    check(reused->data.main.front() == plan.main_byte && reused->data.drft.front() == plan.draft_byte,
        "reused snapshot storage was unnecessarily zeroed");
    check(cache.size() == charged, "reuse changed the hard-budget charge");
}

void smaller_snapshot_charges_capacity() {
    auto a = prompt(1), b = prompt(2), resident = prompt(3);
    server_prompt_cache cache(2, 4096);
    auto * entry = save(cache, a);
    const size_t charged = entry->data.main.capacity() + entry->data.drft.capacity();
    check(cache.load(resident, a.tokens, target(), draft(), 7), "restore before smaller snapshot failed");
    auto * smaller = cache.alloc(b, plan.main_size / 2, plan.draft_size / 2);
    check(smaller && smaller->data.main.size() == plan.main_size / 2 && smaller->data.drft.size() == plan.draft_size / 2,
        "smaller snapshot has incorrect logical lengths");
    check(cache.size() == charged && cache.size() > smaller->data.main.size() + smaller->data.drft.size(),
        "retained capacity bypassed the hard cache budget");
}

void oversized_spare_does_not_evict_useful_state() {
    auto a = prompt(1), b = prompt(2), c = prompt(3, 600 * 1024), resident = prompt(4);
    server_prompt_cache cache(2, 4096);
    plan.main_size = 1024 * 1024;
    plan.draft_size = 0;
    save(cache, a, false);
    plan.main_size = 600 * 1024;
    auto * survivor = save(cache, b, false);
    plan.main_size = 1024 * 1024;
    check(cache.load(resident, a.tokens, target(), nullptr, 7), "restore before oversized spare test failed");
    auto * fresh = cache.alloc(c, 256 * 1024, 0, &b.tokens);
    check(fresh && fresh->data.main.capacity() < 1024 * 1024, "oversized spare displaced a useful cached state");
    check(cache.states.size() == 2 && cache.find_best(resident, b.tokens) != cache.states.end() &&
        &*cache.find_best(resident, b.tokens) == survivor && cache.size() <= cache.limit_size,
        "spare pressure changed protected eviction or exceeded the hard budget");
}

void update_discards_spare_before_cached_state() {
    auto a = prompt(1), b = prompt(2), resident = prompt(3);
    server_prompt_cache cache(2, 4096);
    plan.main_size = 256 * 1024;
    plan.draft_size = 0;
    save(cache, a, false);
    plan.main_size = 128 * 1024;
    auto * survivor = save(cache, b, false);
    plan.main_size = 256 * 1024;
    check(cache.load(resident, a.tokens, target(), nullptr, 7), "restore before reduced budget failed");
    cache.limit_size = 160 * 1024;
    cache.update();
    check(cache.states.size() == 1 && &cache.states.front() == survivor && cache.size() <= cache.limit_size,
        "reduced byte budget evicted a useful state before idle spare storage");
}

void failure_does_not_recycle_partial_state() {
    auto a = prompt(1), b = prompt(2), resident = prompt(3);
    server_prompt_cache cache(2, 4096);
    save(cache, a);
    plan.fail_restore = true;
    check(!cache.load(resident, a.tokens, target(), draft(), 7), "injected restore failure was accepted");
    plan.fail_restore = false;
    check(cache.states.empty() && cache.size() == 0, "failed restore retained invalid storage");
    server_prompt_cache_state * result = nullptr;
    {
        allocation_failure fail;
        result = cache.alloc(b, plan.main_size, plan.draft_size);
    }
    check(!result && allocation_probe::large_calls > 0 && cache.states.empty(), "fresh allocation failure was not contained");
}

void failed_reused_save_is_not_published() {
    auto a = prompt(1), b = prompt(2), resident = prompt(3);
    server_prompt_cache cache(2, 4096);
    save(cache, a);
    check(cache.load(resident, a.tokens, target(), draft(), 7), "restore before reused failed save failed");
    plan.short_save = true;
    check(!cache.save(b, target(), draft(), 7), "short write into recycled storage was published");
    plan.short_save = false;
    check(cache.states.empty() && cache.find_best(resident, b.tokens) == cache.states.end(),
        "partially overwritten recycled bytes remain selectable");
}

void disabled_storage_uses_legacy_ownership() {
    setenv("EZLOCALAI_LLAMA_CACHE_STORAGE", "0", 1);
    auto a = prompt(1, 128 * 1024), resident = prompt(3);
    auto copy = a.clone();
    check(checkpoint_data(a) != checkpoint_data(copy), "storage opt-out did not deep-copy checkpoint bytes");
    server_prompt_cache cache(2, 4096);
    save(cache, a);
    check(cache.load(resident, a.tokens, target(), draft(), 7), "storage opt-out changed restore bytes");
    check(cache.states.empty() && cache.size() == 0, "storage opt-out retained consumed snapshot buffers");
    setenv("EZLOCALAI_LLAMA_CACHE_STORAGE", "1", 1);
}
}

extern "C" size_t llama_state_seq_get_size_ext(llama_context * context, llama_seq_id sequence, llama_state_seq_flags) {
    check(sequence == 7 && (context == target() || context == draft()), "unexpected snapshot size context");
    return context == target() ? plan.main_size : plan.draft_size;
}

extern "C" size_t llama_state_seq_get_data_ext(llama_context * context, uint8_t * bytes,
        size_t size, llama_seq_id sequence, llama_state_seq_flags) {
    check(sequence == 7 && (context == target() || context == draft()), "unexpected snapshot write context");
    const auto expected = context == target() ? plan.main_size : plan.draft_size;
    check(size == expected, "wrong snapshot write length");
    std::memset(bytes, context == target() ? plan.main_byte : plan.draft_byte, plan.short_save ? size / 2 : size);
    return plan.short_save ? size / 2 : size;
}

extern "C" size_t llama_state_seq_set_data_ext(llama_context * context, const uint8_t * bytes,
        size_t size, llama_seq_id sequence, llama_state_seq_flags) {
    check(sequence == 7 && (context == target() || context == draft()), "unexpected snapshot restore context");
    ++plan.restores;
    if (plan.fail_restore) return 0;
    const auto expected_size = context == target() ? plan.main_size : plan.draft_size;
    const auto expected_byte = context == target() ? plan.main_byte : plan.draft_byte;
    check(size == expected_size && std::all_of(bytes, bytes + size, [=](uint8_t value) { return value == expected_byte; }),
        "snapshot roundtrip changed serialized bytes");
    return size;
}

int main(int argc, char ** argv) {
    try {
        setenv("EZLOCALAI_LLAMA_CACHE_STORAGE", "1", 1);
        if (argc == 2 && std::strcmp(argv[1], "--pool-only") == 0) {
            snapshot_reuse_without_large_allocation();
            std::cout << "Initialized snapshot reuse passed with large allocations disabled\n";
            return 0;
        }
        checkpoint_copy_and_write();
        plan = {}; actual_checkpoint_update_and_restore();
        plan = {}; checkpoint_copy_without_large_allocation();
        plan = {}; shared_checkpoint_branches_fit_budget();
        plan = {}; protected_shared_checkpoint_save_fits();
        plan = {}; independent_checkpoints_do_not_bypass_protection();
        plan = {}; independent_checkpoint_union_evicts_oldest();
        plan = {}; shared_identity_across_target_and_draft();
        plan = {}; shared_checkpoint_growth_and_release();
        plan = {}; shared_checkpoint_snapshot_failures_preserve_budget();
        plan = {}; disabled_storage_shared_sources_still_need_independent_space();
        plan = {}; snapshot_reuse_without_large_allocation();
        plan = {}; smaller_snapshot_charges_capacity();
        plan = {}; oversized_spare_does_not_evict_useful_state();
        plan = {}; update_discards_spare_before_cached_state();
        plan = {}; failure_does_not_recycle_partial_state();
        plan = {}; failed_reused_save_is_not_published();
        plan = {}; disabled_storage_uses_legacy_ownership();
        std::cout << "18 cache storage scenarios passed: shared backing union, protected admission, detached mutation, initialized buffer reuse, capacity budget, allocation/save failures and opt-out\n";
        return 0;
    } catch (const std::exception & error) {
        allocation_probe::reject_large = false;
        std::cerr << error.what() << '\n';
        return 1;
    }
}
