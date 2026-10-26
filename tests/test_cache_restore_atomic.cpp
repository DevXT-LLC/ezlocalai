#include "server-task.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// This test compiles the managed engine's actual server-task.cpp. Only the
// external state-deserializer boundary is replaced, so failures exercise the
// production cache selection, ownership, return value and invalidation code.
namespace {
void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

struct context_tag {} target_tag, draft_tag;
llama_context * target_context() { return reinterpret_cast<llama_context *>(&target_tag); }
llama_context * draft_context() { return reinterpret_cast<llama_context *>(&draft_tag); }

enum class result { complete, rejected, partial, oversized, exception };
struct injected_state_error : std::runtime_error {
    injected_state_error() : std::runtime_error("injected serializer exception") {}
};
struct restore_call { bool draft; size_t size; };
struct restore_plan {
    server_prompt_cache * cache = nullptr;
    const server_prompt_cache_state * selected = nullptr;
    std::vector<uint8_t> main;
    std::vector<uint8_t> drft;
    size_t cache_size = 0;
    result target_result = result::complete;
    result draft_result = result::complete;
    std::vector<restore_call> calls;
} plan;

#ifndef CACHE_RESTORE_ONLY
struct snapshot_plan {
    server_prompt_cache * cache = nullptr;
    const server_prompt * active = nullptr;
    size_t target_size = 32;
    size_t draft_size = 16;
    result target_result = result::complete;
    result draft_result = result::complete;
    std::vector<restore_call> size_calls;
    std::vector<restore_call> data_calls;
} snapshot;
#endif

server_prompt prompt(int branch) {
    llama_tokens tokens;
    for (int i = 0; i < 128; ++i) tokens.push_back(i < 64 ? i : branch * 1000 + i);
    common_prompt_checkpoint checkpoint{};
    checkpoint.n_tokens = 32;
    checkpoint.pos_min = 0;
    checkpoint.pos_max = 31;
    checkpoint.data_tgt = {1, 2, 3};
    checkpoint.data_dft = {4, 5};
    checkpoint.data_spec = {6, 7, 8, 9};
    return {server_tokens(tokens, false), {std::move(checkpoint)}};
}

bool same_prompt(const server_prompt & a, const server_prompt & b) {
    return a.n_tokens() == b.n_tokens() && a.tokens.get_common_prefix(b.tokens) == a.n_tokens() &&
        a.checkpoints.size() == b.checkpoints.size() &&
        a.checkpoints.front().data_tgt == b.checkpoints.front().data_tgt &&
        a.checkpoints.front().data_dft == b.checkpoints.front().data_dft &&
        a.checkpoints.front().data_spec == b.checkpoints.front().data_spec;
}

server_prompt_cache_state * save(server_prompt_cache & cache, const server_prompt & value,
        size_t main_size = 32, size_t draft_size = 16) {
    auto * entry = cache.alloc(value, main_size, draft_size);
    check(entry != nullptr, "cache allocation failed");
    std::fill(entry->data.main.begin(), entry->data.main.end(), uint8_t(0xa1));
    std::fill(entry->data.drft.begin(), entry->data.drft.end(), uint8_t(0xd2));
    return entry;
}

void prepare(server_prompt_cache & cache, const server_prompt_cache_state * selected,
        result target_result = result::complete, result draft_result = result::complete) {
    plan = {};
    plan.cache = &cache;
    plan.selected = selected;
    if (selected) {
        plan.main = selected->data.main;
        plan.drft = selected->data.drft;
    }
    plan.cache_size = cache.size();
    plan.target_result = target_result;
    plan.draft_result = draft_result;
}

void complete_restore(bool with_draft) {
    auto a = prompt(1), b = prompt(2), resident = prompt(3);
    server_prompt_cache cache(1, 4096);
    auto * saved = save(cache, a, 32, with_draft ? 16 : 0);
    save(cache, b);
    prepare(cache, saved);
    check(cache.load(resident, a.tokens, target_context(), with_draft ? draft_context() : nullptr, 7),
        "complete restore should succeed");
    check(plan.calls.size() == (with_draft ? 2u : 1u), "restore called the wrong contexts");
    check(!plan.calls[0].draft && (!with_draft || plan.calls[1].draft), "target must restore before draft");
    check(same_prompt(resident, a), "prompt and checkpoints commit only after a complete restore");
    check(cache.states.size() == 1 && same_prompt(cache.states.front().prompt, b),
        "successful restore consumes only its selected entry");
}

void failed_restore(result target_result, result draft_result, bool expect_draft) {
    auto a = prompt(1), b = prompt(2), resident = prompt(3);
    const auto before = resident.clone();
    server_prompt_cache cache(1, 4096);
    auto * saved = save(cache, a);
    auto * survivor = save(cache, b);
    const auto survivor_main = survivor->data.main, survivor_draft = survivor->data.drft;
    prepare(cache, saved, target_result, draft_result);
    check(!cache.load(resident, a.tokens, target_context(), draft_context(), 7), "invalid restore should fail");
    check(plan.calls.size() == (expect_draft ? 2u : 1u), "failed target must prevent draft restore");
    check(same_prompt(resident, before), "failed restore must not commit cached prompt metadata");
    check(cache.states.size() == 1 && &cache.states.front() == survivor &&
        survivor->data.main == survivor_main && survivor->data.drft == survivor_draft,
        "failed restore must evict its entry without damaging other entries");
    check(cache.find_best(resident, a.tokens) == cache.states.end(), "damaged entry remains selectable");
    const auto calls = plan.calls.size();
    check(cache.load(resident, a.tokens, target_context(), draft_context(), 7) && plan.calls.size() == calls,
        "a later miss must not retry the damaged entry");

    // The caller can clear/recompute and cache a valid replacement. This also
    // catches a stale empty-main entry shadowing the new complete state.
    prepare(cache, save(cache, a));
    check(cache.load(resident, a.tokens, target_context(), draft_context(), 7) && same_prompt(resident, a),
        "valid replacement should restore successfully after a failure");
}

void incomplete_entry(size_t main_size, size_t draft_size, llama_context * target, llama_context * draft) {
    auto a = prompt(1), resident = prompt(3);
    const auto before = resident.clone();
    server_prompt_cache cache(1, 4096);
    auto * saved = save(cache, a, main_size, draft_size);
    prepare(cache, saved);
    check(!cache.load(resident, a.tokens, target, draft, 7), "missing payload/context must fail");
    check(plan.calls.empty(), "unrestorable entry must be rejected before mutating either context");
    check(cache.states.empty() && same_prompt(resident, before), "unrestorable entry must be evicted without committing prompt");
}

void protected_eviction_after_failure() {
    auto a = prompt(1), b = prompt(2), outgoing = prompt(3), resident = prompt(4);
    server_prompt_cache cache(1, 4096);
    auto * bad = save(cache, a, 400 * 1024, 200 * 1024);
    auto * protected_b = save(cache, b, 128 * 1024, 64 * 1024);
    prepare(cache, bad, result::complete, result::rejected);
    check(!cache.load(resident, a.tokens, target_context(), draft_context(), 7), "injected draft failure");
    check(cache.alloc(outgoing, 768 * 1024, 0, &b.tokens) != nullptr, "failed entry should release its full cache budget");
    const auto best = cache.find_best(resident, b.tokens);
    check(best != cache.states.end() && &*best == protected_b && cache.size() <= 1024 * 1024,
        "allocation after failure must retain the protected requested state and hard budget");
    prepare(cache, protected_b);
    check(cache.load(resident, b.tokens, target_context(), draft_context(), 7) && same_prompt(resident, b),
        "unrelated protected entry must remain restorable");
}

void cache_miss() {
    auto a = prompt(1), resident = prompt(3);
    server_prompt_cache cache(1, 4096);
    prepare(cache, nullptr);
    check(cache.load(resident, a.tokens, target_context(), draft_context(), 7) && plan.calls.empty(),
        "cache miss must retain the existing success-without-restore behavior");
}

#ifndef CACHE_RESTORE_ONLY
void prepare_snapshot(server_prompt_cache & cache, const server_prompt & active,
        result target_result = result::complete, result draft_result = result::complete) {
    snapshot = {};
    snapshot.cache = &cache;
    snapshot.active = &active;
    snapshot.target_result = target_result;
    snapshot.draft_result = draft_result;
}

void complete_snapshot(bool with_draft) {
    auto active = prompt(1), resident = prompt(3);
    const auto before = active.clone();
    server_prompt_cache cache(1, 4096);
    prepare_snapshot(cache, active);
    check(cache.save(active, target_context(), with_draft ? draft_context() : nullptr, 7), "complete snapshot should succeed");
    check(snapshot.size_calls.size() == (with_draft ? 2u : 1u) && snapshot.data_calls.size() == (with_draft ? 2u : 1u),
        "snapshot should size/write exactly its existing contexts");
    check(same_prompt(active, before), "saving must preserve the active prompt/checkpoints");
    check(cache.states.size() == 1 && same_prompt(cache.states.front().prompt, active), "complete outgoing entry should be cached");
    const auto & saved = cache.states.front();
    check(saved.data.main == std::vector<uint8_t>(32, 0x5a) &&
        saved.data.drft == std::vector<uint8_t>(with_draft ? 16 : 0, 0xa5), "only complete serializer bytes should be cached");
    prepare(cache, &saved);
    check(cache.load(resident, active.tokens, target_context(), with_draft ? draft_context() : nullptr, 7) && same_prompt(resident, active),
        "a complete snapshot should roundtrip through the production loader");
}

void failed_snapshot(result target_result, result draft_result, bool expect_draft) {
    auto active = prompt(1), requested = prompt(2);
    const auto before = active.clone();
    server_prompt_cache cache(1, 4096);
    auto * protected_entry = save(cache, requested);
    const auto original_main = protected_entry->data.main, original_draft = protected_entry->data.drft;
    prepare_snapshot(cache, active, target_result, draft_result);
    check(!cache.save(active, target_context(), draft_context(), 7, &requested.tokens), "incomplete snapshot should fail");
    check(snapshot.data_calls.size() == (expect_draft ? 2u : 1u), "failed target serialization must prevent draft serialization");
    check(same_prompt(active, before), "failed save must preserve active prompt/checkpoint ownership");
    check(cache.states.size() == 1 && &cache.states.front() == protected_entry &&
        protected_entry->data.main == original_main && protected_entry->data.drft == original_draft,
        "failed save must remove only its new entry and retain the protected requested state");
    check(cache.find_best(requested, active.tokens) == cache.states.end(), "partial outgoing entry remains selectable");
    prepare_snapshot(cache, active);
    check(cache.save(active, target_context(), draft_context(), 7, &requested.tokens), "valid save retry should succeed after a partial write");
    check(cache.states.size() == 2 && same_prompt(active, before), "save retry must retain the unrelated protected entry and active prompt");
}

void throwing_snapshot(bool draft) {
    auto active = prompt(1), requested = prompt(2);
    const auto before = active.clone();
    server_prompt_cache cache(1, 4096);
    auto * protected_entry = save(cache, requested);
    const auto original_main = protected_entry->data.main, original_draft = protected_entry->data.drft;
    prepare_snapshot(cache, active, draft ? result::complete : result::exception, draft ? result::exception : result::complete);
    bool caught = false;
    try {
        cache.save(active, target_context(), draft_context(), 7, &requested.tokens);
    } catch (const injected_state_error &) {
        caught = true;
    }
    check(caught, "save must preserve an escaping serializer exception");
    check(snapshot.data_calls.size() == (draft ? 2u : 1u), "exception should stop serialization at the failing context");
    check(cache.states.size() == 1 && &cache.states.front() == protected_entry &&
        protected_entry->data.main == original_main && protected_entry->data.drft == original_draft && same_prompt(active, before),
        "serializer exception must discard only its new entry and preserve active/protected prompt ownership");
    check(cache.find_best(requested, active.tokens) == cache.states.end(), "throwing serializer left a selectable partial entry");
}

void unavailable_snapshot_size(bool draft) {
    auto active = prompt(1), requested = prompt(2);
    server_prompt_cache cache(1, 4096);
    auto * protected_entry = save(cache, requested);
    prepare_snapshot(cache, active);
    (draft ? snapshot.draft_size : snapshot.target_size) = 0;
    check(!cache.save(active, target_context(), draft_context(), 7, &requested.tokens), "unavailable state size must fail");
    check(snapshot.data_calls.empty() && cache.states.size() == 1 && &cache.states.front() == protected_entry,
        "unavailable size must not allocate/publish an outgoing entry or call the serializer");
}

void invalid_snapshot_input(bool empty_prompt) {
    auto active = empty_prompt ? server_prompt{server_tokens(llama_tokens{}, false), {}} : prompt(1);
    server_prompt_cache cache(1, 4096);
    prepare_snapshot(cache, active);
    check(!cache.save(active, empty_prompt ? target_context() : nullptr, draft_context(), 7), "missing active prompt/target context must fail");
    check(snapshot.size_calls.empty() && snapshot.data_calls.empty() && cache.states.empty(), "invalid input must not reach the state API or cache");
}

void refused_snapshot_budget(bool oversized) {
    auto active = prompt(1), requested = prompt(2);
    server_prompt_cache cache(1, 4096);
    auto * protected_entry = save(cache, requested, 400 * 1024, 200 * 1024);
    prepare_snapshot(cache, active);
    snapshot.target_size = oversized ? 2 * 1024 * 1024 : 300 * 1024;
    snapshot.draft_size = 200 * 1024;
    check(!cache.save(active, target_context(), draft_context(), 7, &requested.tokens), "over-budget save must be refused");
    check(snapshot.data_calls.empty() && cache.states.size() == 1 && &cache.states.front() == protected_entry && cache.size() <= 1024 * 1024,
        "refused save must not serialize or evict its protected requested entry");
}

void duplicate_snapshot() {
    auto active = prompt(1);
    server_prompt_cache cache(1, 4096);
    auto * previous = save(cache, active);
    const auto original = previous->data.main;
    prepare_snapshot(cache, active);
    check(!cache.save(active, target_context(), draft_context(), 7), "already cached prompt should retain skip behavior");
    check(snapshot.data_calls.empty() && cache.states.size() == 1 && &cache.states.front() == previous && previous->data.main == original,
        "duplicate skip must not overwrite existing cached state");
}
#endif
}

#ifndef CACHE_RESTORE_ONLY
extern "C" size_t llama_state_seq_get_size_ext(llama_context * context, llama_seq_id sequence, llama_state_seq_flags flags) {
    check(snapshot.cache && snapshot.active, "unexpected snapshot-size call");
    check(context == target_context() || context == draft_context(), "unexpected snapshot-size context");
    check(sequence == 7 && flags == LLAMA_STATE_SEQ_FLAGS_NONE, "snapshot size sequence/flags were not preserved");
    const bool draft = context == draft_context();
    const size_t size = draft ? snapshot.draft_size : snapshot.target_size;
    snapshot.size_calls.push_back({draft, size});
    return size;
}

extern "C" size_t llama_state_seq_get_data_ext(llama_context * context, uint8_t * bytes, size_t size,
        llama_seq_id sequence, llama_state_seq_flags flags) {
    check(snapshot.cache && snapshot.active, "unexpected snapshot-data call");
    check(context == target_context() || context == draft_context(), "unexpected snapshot-data context");
    check(sequence == 7 && flags == LLAMA_STATE_SEQ_FLAGS_NONE, "snapshot data sequence/flags were not preserved");
    const bool draft = context == draft_context();
    check(size == (draft ? snapshot.draft_size : snapshot.target_size), "snapshot did not use the reported size");
    const auto entry = std::find_if(snapshot.cache->states.begin(), snapshot.cache->states.end(), [&](const auto & candidate) {
        return candidate.prompt.n_tokens() == snapshot.active->n_tokens() &&
            candidate.prompt.tokens.get_common_prefix(snapshot.active->tokens) == snapshot.active->n_tokens();
    });
    check(entry != snapshot.cache->states.end() &&
        bytes == (draft ? entry->data.drft.data() : entry->data.main.data()), "serializer was not passed its allocated cache buffer");
    check(snapshot.cache->size() <= snapshot.cache->limit_size, "snapshot allocation exceeded the hard cache budget");
    snapshot.data_calls.push_back({draft, size});
    size_t written = size;
    switch (draft ? snapshot.draft_result : snapshot.target_result) {
        case result::complete: break;
        case result::rejected: written = 0; break;
        case result::partial: written = size ? size - 1 : 0; break;
        case result::oversized: written = size + 1; break;
        case result::exception: throw injected_state_error();
    }
    std::fill(bytes, bytes + std::min(size, written), draft ? uint8_t(0xa5) : uint8_t(0x5a));
    return written;
}
#endif

extern "C" size_t llama_state_seq_set_data_ext(llama_context * context, const uint8_t * bytes,
        size_t size, llama_seq_id sequence, llama_state_seq_flags flags) {
    check(plan.cache && plan.selected, "unexpected deserializer call");
    check(context == target_context() || context == draft_context(), "unexpected restore context");
    check(sequence == 7 && flags == 0, "sequence/flags were not preserved");
    // Check this inside BOTH deserializer calls. The old implementation cleared
    // main after target success, so the draft call observes the half-consumed
    // entry and this assertion fails even before injecting a draft error.
    check(plan.selected->data.main == plan.main && plan.selected->data.drft == plan.drft &&
        plan.cache->size() == plan.cache_size, "cache payload was consumed before the whole restore completed");
    const bool draft = context == draft_context();
    const auto & expected = draft ? plan.drft : plan.main;
    check(size == expected.size() && std::equal(expected.begin(), expected.end(), bytes), "wrong cached restore bytes");
    plan.calls.push_back({draft, size});
    switch (draft ? plan.draft_result : plan.target_result) {
        case result::complete: return size;
        case result::rejected: return 0;
        case result::partial: return size ? size - 1 : 0;
        case result::oversized: return size + 1;
        case result::exception: throw injected_state_error();
    }
    throw std::runtime_error("unreachable restore result");
}

int main() {
    try {
        complete_restore(true);
        complete_restore(false);
        for (auto failure : {result::rejected, result::partial, result::oversized}) {
            failed_restore(failure, result::complete, false);
            failed_restore(result::complete, failure, true);
        }
        incomplete_entry(0, 16, target_context(), draft_context());
        incomplete_entry(0, 0, target_context(), nullptr);
        incomplete_entry(32, 16, target_context(), nullptr);
        incomplete_entry(32, 16, nullptr, draft_context());
        protected_eviction_after_failure();
        cache_miss();
#ifndef CACHE_RESTORE_ONLY
        complete_snapshot(true);
        complete_snapshot(false);
        for (auto failure : {result::rejected, result::partial, result::oversized}) {
            failed_snapshot(failure, result::complete, false);
            failed_snapshot(result::complete, failure, true);
        }
        throwing_snapshot(false);
        throwing_snapshot(true);
        unavailable_snapshot_size(false);
        unavailable_snapshot_size(true);
        invalid_snapshot_input(true);
        invalid_snapshot_input(false);
        refused_snapshot_budget(true);
        refused_snapshot_budget(false);
        duplicate_snapshot();
        std::cout << "31 atomic cache save/restore scenarios passed: exact byte counts, exception cleanup, payload ownership and recovery\n";
#else
        std::cout << "14 atomic cache restore scenarios passed: payload ownership, failure eviction, prompt commit and recovery\n";
#endif
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
