// Optional integration test against the actual pinned llama.cpp C++ parser.
#include "ezlocalai/engine_revision.hpp"
#include "arg.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>
#include <set>
#include <sstream>

extern char **environ;
namespace {
void expect(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
std::string read(const std::filesystem::path &path) { std::ifstream in(path); return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()}; }
std::string sha256(const std::string &s) { unsigned char hash[32]; SHA256(reinterpret_cast<const unsigned char *>(s.data()), s.size(), hash); std::ostringstream out; for (auto c : hash) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c); return out.str(); }
void verify_sources(const std::filesystem::path & fixture) {
#if defined(TEST_LLAMA_MANAGED_ENGINE) && TEST_LLAMA_MANAGED_ENGINE
    const std::set<std::string> expected{
        "common/common.h", "common/arg.cpp", "common/speculative.cpp", "common/speculative.h",
        "common/speculative-mtp-state.h", "common/state-storage.h", "tools/server/server-task.cpp",
        "tools/server/server-task.h", "tools/server/server-context.cpp"};
#else
    const std::set<std::string> expected{"common/common.h", "common/arg.cpp", "common/speculative.cpp"};
#endif
    expect(std::filesystem::is_regular_file(fixture), "engine source hash fixture is missing");
    std::istringstream hashes(read(fixture));
    std::set<std::string> seen;
    for (std::string line; std::getline(hashes, line);) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            expect(line.find(ezlocalai::llama_cpp_reference_revision) != std::string::npos, "engine revision hash fixture");
            continue;
        }
        const auto tab = line.find('\t');
        expect(tab != std::string::npos && line.find('\t', tab + 1) == std::string::npos, "malformed engine source hash row");
        const auto name = line.substr(0, tab), hash = line.substr(tab + 1);
        expect(expected.count(name) == 1 && seen.insert(name).second, "unexpected or duplicate engine source hash path");
        expect(hash.size() == 64 && std::all_of(hash.begin(), hash.end(), [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        }), "malformed engine source hash");
        const auto source = std::filesystem::path(TEST_LLAMA_SOURCE_DIRECTORY) / name;
        expect(std::filesystem::is_regular_file(source), "expected engine source file is missing");
        expect(sha256(read(source)) == hash, "compiled engine source differs from selected pin");
    }
    expect(seen == expected, "all engine source hashes are required");
}
common_params parse(std::vector<std::string> options) {
    std::vector<std::string> args{"llama-server", "-m", "fixture.gguf"}; args.insert(args.end(), options.begin(), options.end());
    std::vector<char *> argv; for (auto &s : args) argv.push_back(s.data());
    common_params params;
    expect(common_params_parse(static_cast<int>(argv.size()), argv.data(), params, LLAMA_EXAMPLE_SERVER), "pinned engine rejected launch options");
    return params;
}
}
int main(int argc, char **argv) {
    try {
        expect(argc == 2, "source hash fixture required");
        verify_sources(argv[1]);
        std::vector<std::string> environment;
        for (char **entry = environ; *entry; ++entry) { std::string s(*entry); if (s.rfind("LLAMA_ARG_", 0) == 0) environment.push_back(s.substr(0, s.find('='))); }
        for (const auto &key : environment) unsetenv(key.c_str());
        const common_params binding_defaults{};
        expect(binding_defaults.n_ctx_checkpoints == ezlocalai::llm_default_context_checkpoints && binding_defaults.checkpoint_min_step == ezlocalai::llm_default_checkpoint_min_step, "reference checkpoint defaults");
        expect(binding_defaults.sampling.reasoning_budget_tokens == -1 && binding_defaults.image_min_tokens == -1 && binding_defaults.warmup && binding_defaults.cont_batching, "reference constructor defaults");
        const std::vector<std::string> base{"-np", "1", "-c", "65536", "-b", "1024", "-ub", "256", "--cache-ram", "8192", "-ctk", "q4_0", "-ctv", "q4_0", "--ctx-checkpoints", "32", "--checkpoint-min-step", "8192", "--warmup", "--cont-batching"};
        for (const auto type : {"none", "draft-mtp", "draft-dflash"}) {
            auto args = base; args.insert(args.end(), {"--spec-type", type});
            if (std::string(type) != "none") args.insert(args.end(), {"--spec-draft-n-max", "3", "--spec-draft-p-min", "0.1"});
            if (std::string(type) == "draft-dflash") args.insert(args.end(), {"--spec-draft-model", "draft.gguf", "--spec-draft-ngl", "0", "--fit", "off"});
            const auto p = parse(args);
            expect(p.n_parallel == 1 && p.n_ctx == 65536 && p.n_batch == 1024 && p.n_ubatch == 256 && p.cache_ram_mib == 8192, "actual launch dimensions/cache");
            expect(p.n_ctx_checkpoints == 32 && p.checkpoint_min_step == 8192 && p.warmup && p.cont_batching, "actual launch checkpoint/warmup/batching settings");
            expect(p.cache_type_k == GGML_TYPE_Q4_0 && p.cache_type_v == GGML_TYPE_Q4_0, "actual launch KV precision");
            const auto expected = std::string(type) == "none" ? COMMON_SPECULATIVE_TYPE_NONE : std::string(type) == "draft-mtp" ? COMMON_SPECULATIVE_TYPE_DRAFT_MTP : COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
            for (auto item : p.speculative.types) expect(item == expected || item == COMMON_SPECULATIVE_TYPE_NONE, "unexpected active speculative backend");
            expect(std::find(p.speculative.types.begin(), p.speculative.types.end(), expected) != p.speculative.types.end(), "selected speculative backend is active");
            if (std::string(type) != "none") expect(p.speculative.draft.n_max == 3 && p.speculative.draft.p_min == static_cast<float>(.1), "actual draft length/probability");
            if (std::string(type) == "draft-dflash") expect(!p.fit_params && p.speculative.draft.mparams.path == "draft.gguf" && p.speculative.draft.n_gpu_layers == 0, "actual DFlash model/fitting/offload settings");
        }
        const auto placed = parse({"-np", "1", "--main-gpu", "1", "--split-mode", "none", "--fit", "off", "--mmproj", "projector.gguf", "--image-min-tokens", "1024"});
        expect(placed.main_gpu == 1 && placed.split_mode == LLAMA_SPLIT_MODE_NONE && !placed.fit_params && placed.image_min_tokens == 1024 && placed.mmproj.path == "projector.gguf", "actual single GPU vision parameters");
        for (float ratio : placed.tensor_split) expect(ratio == 0, "source split mode leaves native tensor ratios at zero");
        const auto custom = parse({"-ctxcp", "7", "-cms", "2048", "--no-warmup", "--no-cont-batching"});
        expect(custom.n_ctx_checkpoints == 7 && custom.checkpoint_min_step == 2048 && !custom.warmup && !custom.cont_batching, "native extension overrides parse correctly");
        for (const auto text : {"0x1.16c2p-133", "0", "-0", "nan"}) {
            const auto p = parse({"--spec-type", "draft-mtp", "--spec-draft-p-min", text}); const float expected = std::strtof(text, nullptr);
            expect((std::isnan(expected) && std::isnan(p.speculative.draft.p_min)) || (expected == p.speculative.draft.p_min && std::signbit(expected) == std::signbit(p.speculative.draft.p_min)), "actual engine float conversion");
        }
        std::cout << "Verified pinned engine parser against native launch contract and source hashes\n"; return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
