#pragma once
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ezlocalai {
std::string llm_unicode_lower(const std::string &value);
struct LlmReplica {
    std::string id, source;
    int64_t parallel = 1, context = 16384;
    std::optional<int64_t> actual_parallel;
    bool speculative = false, failed = false;
    bool resident = false;
};
std::optional<std::string> resolve_llm_replica(const std::vector<LlmReplica> &replicas, const std::string &requested);
std::optional<std::string> resolve_llm_slot_model(const std::vector<LlmReplica> &replicas, const std::string &requested);
int64_t llm_replica_capacity(const LlmReplica &replica);
int64_t llm_model_capacity(const std::vector<LlmReplica> &replicas, const std::optional<std::string> &model, bool swap);
struct LlmSlotState {
    int64_t total = 0;
    std::map<std::string, int64_t> models, replicas;
};
struct LlmSlotAcquisition { bool acquired = false; std::optional<std::string> replica; };
// The caller owns synchronization. A failed acquisition leaves every counter
// unchanged; waiters retry after the reference's 50 ms scheduling interval.
LlmSlotAcquisition try_acquire_llm_slot(const std::vector<LlmReplica> &replicas, LlmSlotState &state,
    const std::optional<std::string> &model, const std::optional<std::string> &preferred,
    bool handoff = false, bool swap = false);
void release_llm_slot(LlmSlotState &state, const std::optional<std::string> &model, const std::optional<std::string> &replica);
}
