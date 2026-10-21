#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace ezlocalai {
int64_t hybrid_llm_kv_bytes_per_token(const std::string &path, const std::string &cache);
// Reproduces Pipes' estimate against the installed xllamacpp memory API. Its
// result has no `memory` field, so successful parsing selects the empirical
// estimate; an estimator exception selects the file-size exception estimate.
double estimate_llm_memory_gb(const std::string &path, int64_t context,
    const std::vector<std::string> &projectors, const std::string &cache);
}
