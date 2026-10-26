#include "ezlocalai/llm_cache.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/engine_revision.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/process.hpp"
#include <arpa/inet.h>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <openssl/sha.h>
#include <sstream>
#include <unistd.h>

namespace fs = std::filesystem;
namespace {
void expect(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::string read(const fs::path &path) { std::ifstream in(path); return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()}; }
std::vector<std::string> split(const std::string &s, char delimiter = '\t') {
    std::vector<std::string> out;
    for (size_t begin = 0;;) { const auto end = s.find(delimiter, begin); out.push_back(s.substr(begin, end == std::string::npos ? end : end - begin)); if (end == std::string::npos) return out; begin = end + 1; }
}
std::string hex(const std::string &s) { std::ostringstream out; for (unsigned char c : s) out << std::hex << std::setfill('0') << std::setw(2) << unsigned(c); return out.str(); }
std::string unhex(const std::string &s) { std::string out; for (size_t i = 0; i < s.size(); i += 2) out += static_cast<char>(std::stoi(s.substr(i, 2), nullptr, 16)); return out; }
std::string sha256(const std::string &s) { unsigned char hash[32]; SHA256(reinterpret_cast<const unsigned char *>(s.data()), s.size(), hash); return hex({reinterpret_cast<char *>(hash), 32}); }
std::optional<std::string> optional(const std::string &s) { return s == "N" ? std::nullopt : std::optional<std::string>(unhex(s.substr(1))); }
std::optional<int64_t> integer(const std::string &s) { return s == "N" ? std::nullopt : std::optional<int64_t>(std::stoll(s)); }
void event(std::string &trace, const std::string &name) { if (!trace.empty()) trace += '\n'; trace += name; }
template<class Callback> size_t cases(const fs::path &reference, const fs::path &path, Callback callback) {
    std::istringstream input(read(path)); size_t count = 0;
    for (std::string line; std::getline(input, line);) {
        if (line.empty()) continue;
        if (line[0] == '#') { const auto at = line.find(" sha256="); if (at != std::string::npos) expect(sha256(read(reference / line.substr(2, at - 2))) == line.substr(at + 8), "cache source hash"); continue; }
        callback(split(line)); ++count;
    }
    return count;
}
size_t reference_cases(const fs::path &reference, const fs::path &fixtures) {
    const auto resolved = cases(reference, fixtures / "llm_cache_resolve_cases.tsv", [](const auto &c) {
        const auto result = ezlocalai::resolve_llm_prompt_cache(optional(c[1]), std::stoll(c[2]));
        expect(result.mib == c[3] && result.reason == unhex(c[4]), "resolve cache case " + c[0] + ": " + result.reason);
    });
    const auto capped = cases(reference, fixtures / "llm_cache_cap_cases.tsv", [](const auto &c) {
        std::string trace, category = "S", value, reason;
        const auto environment = [&](const std::string &key, const std::string &fallback) {
            event(trace, "G:" + key);
            return optional(c[key == "LLM_PROMPT_CACHE_ALLOW_UNSAFE" ? 3 : key == "LLM_PROMPT_CACHE_RAM_MARGIN_MIB" ? 4 : 5]).value_or(fallback);
        };
        const auto memory = [&] {
            event(trace, "M"); if (!c[10].empty()) throw ezlocalai::LlmCacheError(c[10], "fixture memory unavailable");
            return ezlocalai::PromptCacheRam{std::stoll(c[8]), std::stoll(c[9])};
        };
        try { const auto result = ezlocalai::cap_llm_prompt_cache({c[1], unhex(c[2])}, environment, memory, integer(c[6]), integer(c[7])); value = result.mib; reason = result.reason; }
        catch (const ezlocalai::LlmCacheError &error) { category = "E"; value = error.category; reason = error.what(); }
        expect(category == c[11] && value == c[12] && reason == unhex(c[13]), "cap cache case " + c[0] + ": " + category + ' ' + value + ' ' + reason + " expected " + c[11] + ' ' + c[12] + ' ' + unhex(c[13]));
        expect(trace == unhex(c[14]), "cap cache probe order " + c[0]);
    });
    const auto kv = cases(reference, fixtures / "llm_cache_kv_cases.tsv", [](const auto &c) {
        std::string trace, category = "S", value;
        const auto environment = [&](const std::string &key, const std::string &fallback) {
            event(trace, "G:" + key); return optional(c[key == "KV_CACHE_TYPE" ? 2 : 3]).value_or(fallback);
        };
        try { value = ezlocalai::configured_llm_kv_cache_type(unhex(c[1]), environment); }
        catch (const ezlocalai::LlmCacheError &error) { category = error.category; value = error.what(); }
        expect(category == c[4] && value == unhex(c[5]) && trace == unhex(c[6]), "KV cache case " + c[0]);
    });
    const auto ram = cases(reference, fixtures / "llm_cache_ram_cases.tsv", [](const auto &c) {
        std::string trace;
        const auto result = ezlocalai::prompt_cache_ram_from_proc(unhex(c[1]), [&] { event(trace, "Z"); return optional(c[2]); }, std::stoll(c[3]));
        expect(result.available_mib == std::stoll(c[4]) && result.total_mib == std::stoll(c[5]) && trace == unhex(c[6]), "RAM cache case " + c[0]);
    });
    expect(resolved == 549 && capped == 7052 && kv == 550 && ram == 960, "all cache source cases");
    return resolved + capped + kv + ram;
}
void environment() {
    for (const auto &setting : ezlocalai::environment_defaults) unsetenv(setting.first.data());
    for (const auto key : {"KV_CACHE_TYPE_5090", "KV_CACHE_TYPE_T4", "KV_CACHE_TYPE_A100"}) unsetenv(key);
    setenv("EZLOCALAI_DOTENV_PATH", "", 1); setenv("GPU_LAYERS", "0", 1);
    setenv("LLAMA_CPP_SERVER_BIN", TEST_EMBEDDING_FIXTURE_BIN, 1); setenv("LLAMA_CPP_START_TIMEOUT", "2", 1);
    setenv("LLM_PROMPT_CACHE_ALLOW_UNSAFE", "true", 1);
}
int available_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0); sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    expect(!bind(fd, reinterpret_cast<sockaddr *>(&a), sizeof(a)), "fixture port"); socklen_t n = sizeof(a); getsockname(fd, reinterpret_cast<sockaddr *>(&a), &n); close(fd); return ntohs(a.sin_port);
}
struct Fixture {
    fs::path directory; std::string port;
    Fixture(const fs::path &root, const std::string &name, int context = 65536) : directory(root / name), port(std::to_string(available_port())) {
        fs::create_directories(directory); const auto model = directory / "model.gguf"; std::ofstream(model) << "fixture GGUF";
        setenv("LLAMA_CPP_MODEL_PATH", model.c_str(), 1); setenv("EMBEDDING_FIXTURE_DIR", directory.c_str(), 1); setenv("LLAMA_CPP_SERVER_PORT", port.c_str(), 1);
        setenv("LLAMA_CPP_CONTEXT", std::to_string(context).c_str(), 1);
    }
    fs::path path(const std::string &name) const { return directory / (name + '-' + port); }
    void reaped() const { std::istringstream pids(read(path("launch"))); for (int pid; pids >> pid;) expect(kill(pid, 0) != 0, "cache fixture child reaped"); }
};
void managed_cases(const fs::path &root) {
    unsetenv("CUDA_VISIBLE_DEVICES"); unsetenv("EZLOCALAI_TEST_CUDA_FAIL");
    environment(); Fixture fixture(root, "cache");
    setenv("LLM_PROMPT_CACHE_MIB", "8192", 1);
    setenv("KV_CACHE_TYPE", "f16", 1); setenv("KV_CACHE_TYPE_5090", "Q8_0", 1);
    {
        ezlocalai::LlamaCppInferenceBackend backend("ordinary");
        expect(backend.chat(R"({"messages":[],"cache_prompt":false})").find("choices") != std::string::npos, "cache-configured inference");
        const auto args = read(fixture.path("args"));
        for (const auto flag : {"--cache-ram\n8192\n", "-ctk\nq8_0\n", "-ctv\nq8_0\n"}) expect(args.find(flag) != std::string::npos, "cache engine argument");
        auto runtime = backend.model_lifecycle_json();
        for (const auto field : {"\"prompt_cache_mib\":8192", "\"prompt_cache_requested_mib\":8192", "\"kv_cache_type\":\"q8_0\""}) expect(runtime.find(field) != std::string::npos, "cache runtime metadata");
        expect(read(fixture.path("seen")).find("\"cache_prompt\":true") != std::string::npos, "chat forces prefix caching");
        backend.completion(R"({"prompt":"ordinary"})"); const auto requests = read(fixture.path("seen"));
        const auto at = requests.rfind("POST /v1/completions "); expect(at != std::string::npos && requests.substr(at).find("cache_prompt") == std::string::npos, "completion retains distinct request settings");
        auto restore = backend.unload_llms_for_media();
        setenv("LLM_PROMPT_CACHE_ALLOW_UNSAFE", "false", 1); setenv("LLM_PROMPT_CACHE_RAM_MARGIN_MIB", "99999999999999999999999999", 1);
        setenv("KV_CACHE_TYPE_5090", " ", 1); restore(true);
        runtime = backend.model_lifecycle_json();
        expect(runtime.find("\"prompt_cache_mib\":0") != std::string::npos && runtime.find("\"prompt_cache_requested_mib\":8192") != std::string::npos, "restoration refreshes RAM cap while preserving requested size");
        const auto restored = read(fixture.path("args")); expect(restored.find("--cache-ram\n0\n") != std::string::npos && restored.find("-ctk\nq4_0\n") != std::string::npos, "whitespace card override selects default before global fallback");
        expect(runtime.find("\"count\":2,\"failures\":0") != std::string::npos, "cache restoration lifecycle");
    }
    fixture.reaped();
    for (const auto &scenario : std::vector<std::pair<std::string, std::string>>{{"auto", "32768"}, {"off", "0"}, {"bad", "8192"}, {"２_５６", "256"}}) {
        environment(); Fixture run(root, "size-" + scenario.second, 500000); setenv("LLM_PROMPT_CACHE_MIB", scenario.first.c_str(), 1);
        { ezlocalai::LlamaCppInferenceBackend backend("ordinary"); expect(backend.chat(R"({"messages":[]})").find("choices") != std::string::npos, "cache size startup"); expect(read(run.path("args")).find("--cache-ram\n" + scenario.second + '\n') != std::string::npos, "resolved cache size argument"); }
        run.reaped();
    }
    for (bool kv : {false, true}) {
        environment(); Fixture run(root, kv ? "invalid-kv" : "nan-fraction");
        if (kv) setenv("KV_CACHE_TYPE_5090", "none", 1);
        else { setenv("LLM_PROMPT_CACHE_ALLOW_UNSAFE", "false", 1); setenv("LLM_PROMPT_CACHE_MAX_RAM_FRACTION", "NaN", 1); }
        { ezlocalai::LlamaCppInferenceBackend backend("ordinary"); const auto response = backend.chat(R"({"messages":[]})"); expect(response.find(kv ? "KV_CACHE_TYPE must be auto" : "cannot convert float NaN to integer") != std::string::npos, "cache validation failure"); expect(!fs::exists(run.path("launch")) && backend.loaded_models_json() == "{}", "invalid cache configuration stops before process launch"); expect(backend.model_lifecycle_json().find("\"count\":1,\"failures\":1") != std::string::npos, "invalid cache load lifecycle"); }
    }
    const auto actual_ram = ezlocalai::native_prompt_cache_ram(); expect(actual_ram.total_mib > 0 && actual_ram.available_mib >= 0, "native host RAM probe");
}
void native_policy_cases(const fs::path &root) {
    std::map<std::string, std::string> settings;
    const auto env = [&](const std::string &key, const std::string &fallback) {
        const auto found = settings.find(key); return found == settings.end() ? fallback : found->second;
    };
    ezlocalai::PromptCacheRam ram{100000, 131072, 8192};
    const auto budget = [&](const std::string &value = "auto") {
        return ezlocalai::native_llm_prompt_cache_budget(value, 230000, env, [&] { return ram; });
    };
    auto b = budget();
    expect(b.requested.mib == "32768" && b.effective.mib == "32768" && b.swap_allowance_mib == 0, "128 GiB auto uses bounded 32 GiB cache without swap");
    ram.total_mib = 122880; b = budget();
    expect(b.effective.mib == "30720", "120 GiB auto remains within the RAM fraction");
    ram = {24000, 32768, 8192}; b = budget();
    expect(b.effective.mib == "8192", "32 GiB workers retain their quarter-RAM ceiling");
    settings["LLM_PROMPT_CACHE_SWAP_MIB"] = "16384"; b = budget();
    expect(b.requested.mib == "22528" && b.effective.mib == "15360" && b.swap_allowance_mib == 7168, "opt-in swap uses only free capacity after its reserve");
    b = budget("4096");
    expect(b.effective.mib == "4096" && b.swap_allowance_mib == 0, "explicit cache ceiling remains an upper bound");
    settings["LLM_PROMPT_CACHE_SWAP_MIB"] = "bad"; b = budget();
    expect(b.effective.mib == "8192" && b.swap_allowance_mib == 0, "invalid swap setting disables swap allowance");
    settings["LLM_PROMPT_CACHE_SWAP_MIB"] = "4096";
    ram.swap_available_mib = 512; b = budget();
    expect(b.effective.mib == "8192" && b.swap_allowance_mib == 0, "already occupied swap is unavailable");
    ram = {1000, 32768, 8192}; b = budget();
    expect(b.effective.mib == "0" && b.swap_allowance_mib == 0, "swap cannot rescue an exhausted RAM budget");
    const auto disabled = ezlocalai::native_llm_prompt_cache_budget("off", 230000, env,
        []() -> ezlocalai::PromptCacheRam { throw std::runtime_error("disabled cache must not probe RAM"); });
    expect(disabled.effective.mib == "0", "disabled cache remains disabled");
    const auto unavailable = ezlocalai::native_llm_prompt_cache_budget("auto", 230000, env,
        []() -> ezlocalai::PromptCacheRam { throw ezlocalai::LlmCacheError("ValueError", "unavailable"); });
    expect(unavailable.effective.mib == "4096" && unavailable.swap_allowance_mib == 0, "unavailable probe retains conservative source fallback");

    const auto cg = root / "cgroup"; fs::create_directories(cg / "parent/leaf");
    const auto write = [&](const fs::path &path, int64_t mib) { std::ofstream(cg / path) << mib * 1048576; };
    std::ofstream(cg / "memory.max") << "max";
    write("parent/memory.max", 65536); write("parent/memory.current", 32768);
    std::ofstream(cg / "parent/memory.stat") << "inactive_file " << 8192LL * 1048576;
    write("parent/memory.swap.max", 4096); write("parent/memory.swap.current", 1024);
    write("parent/leaf/memory.max", 32768); write("parent/leaf/memory.current", 8192);
    std::ofstream(cg / "parent/leaf/memory.stat") << "inactive_file " << 1024LL * 1048576;
    write("parent/leaf/memory.swap.max", 8192); write("parent/leaf/memory.swap.current", 0);
    auto constrained = ezlocalai::prompt_cache_ram_with_cgroup({100000, 131072, 8192}, cg.string(), "/parent/leaf");
    expect(constrained.total_mib == 32768 && constrained.available_mib == 25600 && constrained.swap_available_mib == 3072,
        "nested cgroups constrain RAM and swap by every ancestor, allowing reclaimable inactive file pages");
    write("parent/memory.swap.max", 0);
    constrained = ezlocalai::prompt_cache_ram_with_cgroup({100000, 131072, 8192}, cg.string(), "/parent/leaf");
    expect(constrained.swap_available_mib == 0, "parent swap prohibition overrides child allowance");
    fs::remove(cg / "parent/leaf/memory.current");
    constrained = ezlocalai::prompt_cache_ram_with_cgroup({100000, 131072, 8192}, cg.string(), "/parent/leaf");
    expect(constrained.available_mib == 0, "missing usage data cannot authorize unobserved RAM headroom");
    std::ofstream(cg / "parent/leaf/memory.max") << "invalid";
    constrained = ezlocalai::prompt_cache_ram_with_cgroup({100000, 131072, 8192}, cg.string(), "/parent/leaf");
    expect(constrained.total_mib == 0 && constrained.available_mib == 0, "malformed readable limits fail closed");
    write("memory.max", 8192); write("memory.current", 4096);
    constrained = ezlocalai::prompt_cache_ram_with_cgroup({100000, 131072, 8192}, cg.string(), "/../../hidden");
    expect(constrained.total_mib == 8192 && constrained.available_mib == 4096, "hidden namespace path cannot escape controller root");
}
void revision(const fs::path &reference) {
    const std::string pin = ezlocalai::llama_cpp_reference_revision;
    // The native engine runs llama.cpp v0.7.0, ahead of the reference's
    // xllamacpp pin. Its Docker and Compose defaults must still agree.
    expect(read(reference / "scripts/build_xllamacpp.py").find("LLAMA_REVISION = \"") != std::string::npos, "reference still pins an engine revision");
    const fs::path native(TEST_NATIVE_SOURCE_DIRECTORY);
    expect(read(native / "cuda.Dockerfile").find("ARG EZLOCALAI_LLAMA_CPP_REF=" + pin) != std::string::npos, "Docker engine default matches pin");
    expect(read(native / "docker-compose-cuda.yml").find("${EZLOCALAI_LLAMA_CPP_REF:-" + pin + '}') != std::string::npos, "Compose engine default matches pin");
}
}
int main(int argc, char **argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--cuda") { managed_cases(argv[2]); return 0; }
        expect(argc == 3, "reference root and fixtures required");
        const auto count = reference_cases(argv[1], argv[2]); revision(argv[1]);
        const auto directory = fs::temp_directory_path() / ("ezl-cache-" + std::to_string(getpid())); fs::create_directories(directory);
        struct Cleanup { fs::path path; ~Cleanup() { fs::remove_all(path); } } cleanup{directory};
        native_policy_cases(directory);
        std::ofstream(directory / "devices.tsv") << "NVIDIA GeForce RTX 5090\t12\t0\t34359738368\t32212254720\n";
        setenv("EZLOCALAI_TEST_CUDA_DEVICES", (directory / "devices.tsv").c_str(), 1); setenv("LD_LIBRARY_PATH", TEST_CUDA_DRIVER_DIRECTORY, 1);
        const auto result = ezlocalai::run_process_to_files({fs::absolute(argv[0]).string(), "--cuda", directory.string()}, (directory / "child.log").string(), "", 30, true);
        expect(!result.timed_out && result.status == 0, "managed cache integration: " + read(directory / "child.log"));
        std::cout << "Verified " << count << " source cache/RAM cases, managed startup/restoration and engine revision defaults\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
