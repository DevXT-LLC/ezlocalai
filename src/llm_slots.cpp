#include "ezlocalai/llm_slots.hpp"
#include <algorithm>
#include <unicode/ustring.h>

namespace ezlocalai {
std::string llm_unicode_lower(const std::string &value) {
    if (value.empty()) return value;
    UErrorCode error = U_ZERO_ERROR; int32_t size = 0;
    u_strFromUTF8(nullptr, 0, &size, value.data(), value.size(), &error);
    if (error != U_BUFFER_OVERFLOW_ERROR) return value;
    error = U_ZERO_ERROR; std::vector<UChar> input(size + 1);
    u_strFromUTF8(input.data(), input.size(), nullptr, value.data(), value.size(), &error);
    if (U_FAILURE(error)) return value;
    error = U_ZERO_ERROR; const int32_t lowered_size = u_strToLower(nullptr, 0, input.data(), size, "root", &error);
    if (error != U_BUFFER_OVERFLOW_ERROR) return value;
    error = U_ZERO_ERROR; std::vector<UChar> lowered(lowered_size + 1);
    u_strToLower(lowered.data(), lowered.size(), input.data(), size, "root", &error);
    if (U_FAILURE(error)) return value;
    error = U_ZERO_ERROR; int32_t bytes = 0;
    u_strToUTF8(nullptr, 0, &bytes, lowered.data(), lowered_size, &error);
    if (error != U_BUFFER_OVERFLOW_ERROR) return value;
    error = U_ZERO_ERROR; std::string result(bytes, '\0');
    u_strToUTF8(result.data(), result.size(), nullptr, lowered.data(), lowered_size, &error);
    return U_FAILURE(error) ? value : result;
}
namespace {
std::string short_name(const std::string &value) { const auto pos = value.rfind('/'); return pos == std::string::npos ? value : value.substr(pos + 1); }
int64_t count(const std::map<std::string, int64_t> &counts, const std::string &key) { const auto found = counts.find(key); return found == counts.end() ? 0 : found->second; }
}
std::optional<std::string> resolve_llm_replica(const std::vector<LlmReplica> &replicas, const std::string &requested) {
    if (requested.empty()) return std::nullopt;
    const auto full = llm_unicode_lower(requested), name = llm_unicode_lower(short_name(requested));
    // Public source names take precedence, including ambiguous short names.
    for (const auto &replica : replicas) if (llm_unicode_lower(replica.source) == full || llm_unicode_lower(short_name(replica.source)) == name) return replica.id;
    for (const auto &replica : replicas) if (llm_unicode_lower(replica.id) == full || llm_unicode_lower(short_name(replica.id)) == name) return replica.id;
    return std::nullopt;
}
std::optional<std::string> resolve_llm_slot_model(const std::vector<LlmReplica> &replicas, const std::string &requested) {
    if (replicas.empty()) return std::nullopt;
    const auto id = resolve_llm_replica(replicas, requested);
    if (id) for (const auto &replica : replicas) if (replica.id == *id) return replica.source;
    return replicas.front().source;
}
int64_t llm_replica_capacity(const LlmReplica &replica) {
    if (replica.speculative) return 1;
    if (replica.actual_parallel && *replica.actual_parallel) return std::max<int64_t>(1, *replica.actual_parallel);
    if (!replica.parallel) return std::min<int64_t>(16, std::max<int64_t>(1, replica.context / 32768));
    return std::max<int64_t>(1, replica.parallel);
}
int64_t llm_model_capacity(const std::vector<LlmReplica> &replicas, const std::optional<std::string> &model, bool swap) {
    if (!model || model->empty()) return 1;
    int64_t capacity = 0;
    for (const auto &replica : replicas) if (replica.source == *model && !replica.failed) capacity += llm_replica_capacity(replica);
    return swap ? 1 : std::max<int64_t>(1, capacity);
}
LlmSlotAcquisition try_acquire_llm_slot(const std::vector<LlmReplica> &replicas, LlmSlotState &state,
    const std::optional<std::string> &model, const std::optional<std::string> &preferred, bool handoff, bool swap) {
    const auto capacity = llm_model_capacity(replicas, model, swap);
    bool available = (!model ? 0 : count(state.models, *model)) < capacity;
    bool has_replicas = false; std::optional<std::string> selected;
    if (model) for (const auto &replica : replicas) {
        if (replica.source != *model || replica.failed) continue;
        has_replicas = true;
        if (preferred && *preferred != replica.id) continue;
        if (count(state.replicas, replica.id) >= llm_replica_capacity(replica)) continue;
        if (!selected || count(state.replicas, replica.id) < count(state.replicas, *selected)) selected = replica.id;
    }
    if (has_replicas && !selected) available = false;
    bool conflict = false;
    for (const auto &entry : state.models) if (entry.second > 0 && (!model || entry.first != *model)) conflict = true;
    if (handoff || !available || (swap && conflict)) return {};
    if (selected) ++state.replicas[*selected];
    ++state.total;
    if (model && !model->empty()) ++state.models[*model];
    return {true, selected};
}
void release_llm_slot(LlmSlotState &state, const std::optional<std::string> &model, const std::optional<std::string> &replica) {
    if (replica) { auto &value = state.replicas[*replica]; value = std::max<int64_t>(0, value - 1); if (!value) state.replicas.erase(*replica); }
    state.total = std::max<int64_t>(0, state.total - 1);
    if (model && !model->empty()) { const auto found = state.models.find(*model); if (found != state.models.end()) { found->second = std::max<int64_t>(0, found->second - 1); if (!found->second) state.models.erase(found); } }
}
}
