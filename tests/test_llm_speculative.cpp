#include "ezlocalai/llm_speculative.hpp"
#include "ezlocalai/llm_cache.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/process.hpp"
#include <arpa/inet.h>
#include <atomic>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <openssl/sha.h>
#include <poll.h>
#include <sstream>
#include <unistd.h>

namespace fs = std::filesystem;
namespace {
#ifdef EZLOCALAI_USE_HIP
constexpr const char *draft_device_zero = "ROCm0", *draft_device_one = "ROCm1", *hidden_devices = "-1";
#else
constexpr const char *draft_device_zero = "CUDA0", *draft_device_one = "CUDA1", *hidden_devices = "";
#endif
void expect(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
std::string read(const fs::path &path) { std::ifstream in(path, std::ios::binary); return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()}; }
std::vector<std::string> split(const std::string &s, char separator = '\t') {
    std::vector<std::string> result;
    for (size_t start = 0;;) { const auto end = s.find(separator, start); result.push_back(s.substr(start, end == std::string::npos ? end : end - start)); if (end == std::string::npos) return result; start = end + 1; }
}
std::string hex(const std::string &s) { std::ostringstream out; for (unsigned char c : s) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c); return out.str(); }
std::string unhex(const std::string &s) { std::string result; for (size_t i = 0; i < s.size(); i += 2) result += static_cast<char>(std::stoi(s.substr(i, 2), nullptr, 16)); return result; }
std::string sha256(const std::string &s) { unsigned char digest[32]; SHA256(reinterpret_cast<const unsigned char *>(s.data()), s.size(), digest); return hex({reinterpret_cast<char *>(digest), 32}); }
std::optional<std::string> optional(const std::string &s) { return s == "N" ? std::nullopt : std::optional<std::string>(unhex(s.substr(1))); }
void event(std::string &trace, const std::string &value) { if (!trace.empty()) trace += '\n'; trace += value; }
bool same(double value, const std::string &expected) {
    const auto target = std::strtod(expected.c_str(), nullptr);
    return (std::isnan(value) && std::isnan(target)) || (value == target && (value != 0 || std::signbit(value) == std::signbit(target)));
}
template<class F> size_t cases(const fs::path &reference, const fs::path &fixtures, const std::string &name, F callback) {
    std::istringstream input(read(fixtures / ("llm_speculative_" + name + "_cases.tsv"))); size_t count = 0;
    for (std::string line; std::getline(input, line);) {
        if (line.empty()) continue;
        if (line[0] == '#') { const auto at = line.find(" sha256="); expect(sha256(read(reference / line.substr(2, at - 2))) == line.substr(at + 8), "speculative reference hash"); continue; }
        callback(split(line)); ++count;
    }
    return count;
}
ezlocalai::LlmBatchProbes probes(const std::string &encoded, bool available, const std::string &failure, std::string &trace) {
    std::vector<ezlocalai::CudaDevice> devices;
    if (!encoded.empty()) for (const auto &line : split(unhex(encoded), '\n')) { const auto c = split(line, ','); devices.push_back({static_cast<int>(devices.size()), unhex(c[0]), 0, 0, std::stoull(c[1])}); }
    auto record = [&, failure](const std::string &name) { event(trace, name); if (name.substr(0, 1) == failure) throw std::runtime_error("fixture probe failed"); };
    return {[=] { record("A"); return available; }, [=] { record("C"); return static_cast<int>(devices.size()); }, {}, {},
        [=](int64_t index) { record("P:" + std::to_string(index)); if (index < 0 || static_cast<size_t>(index) >= devices.size()) throw std::runtime_error("fixture properties failed"); return devices[index]; }};
}
size_t reference_cases(const fs::path &reference, const fs::path &fixtures) {
    const auto sizes = cases(reference, fixtures, "size", [](const auto &c) { expect(same(ezlocalai::llm_model_size_billions(unhex(c[1])), c[2]), "model size case " + c[0]); });
    const auto lengths = cases(reference, fixtures, "length", [](const auto &c) {
        std::string trace, category = "S", message; std::pair<int, double> value;
        const auto environment = [&](const std::string &key, const std::string &fallback) { event(trace, "G:" + key); return optional(c[key == "MTP_SPEC_DRAFT_N_MAX" ? 6 : 7]).value_or(fallback); };
        try { value = ezlocalai::llm_mtp_draft_length(std::stoll(c[2]), unhex(c[1]), environment, probes(c[3], c[4] == "1", c[5], trace)); }
        catch (const std::exception &error) { category = "RuntimeError"; message = error.what(); }
        expect(category == c[8] && (category == "S" ? std::to_string(value.first) == c[9] && same(value.second, c[10]) : message == unhex(c[9])), "MTP length case " + c[0]);
        expect(trace == unhex(c[11]), "MTP probe order " + c[0] + "\n" + trace + "\nexpected:\n" + unhex(c[11]));
    });
    const auto probabilities = cases(reference, fixtures, "probability", [](const auto &c) {
        std::string trace; const auto environment = [&](const std::string &key, const std::string &fallback) { event(trace, "G:" + key); return optional(c[2]).value_or(fallback); };
        expect(same(ezlocalai::llm_mtp_draft_probability(unhex(c[1]), environment), c[3]) && trace == unhex(c[4]), "MTP probability case " + c[0]);
    });
    const auto dflash = cases(reference, fixtures, "dflash", [](const auto &c) {
        std::string trace, category = "S", message; ezlocalai::LlmDraftSettings value;
        const auto environment = [&](const std::string &key, const std::string &fallback) { event(trace, "G:" + key); return optional(c[key == "DFLASH_SPEC_DRAFT_N_MAX" ? 5 : key == "DFLASH_SPEC_DRAFT_P_MIN" ? 7 : 6]).value_or(fallback); };
        try { value = ezlocalai::llm_dflash_settings(std::stoll(c[1]), environment, probes(c[2], c[3] == "1", c[4], trace)); }
        catch (const ezlocalai::LlmSpeculativeError &error) { category = error.category; message = error.message; }
        expect(category == c[8] && (category == "S" ? std::to_string(value.n_max) == c[9] && same(value.p_min, c[10]) : message == unhex(c[9])), "DFlash case " + c[0] + ": " + message + " expected " + (c[8] == "S" ? c[9] : unhex(c[9])));
        expect(trace == unhex(c[11]), "DFlash probe order " + c[0]);
    });
    const auto downloads = cases(reference, fixtures, "download", [](const auto &c) {
        std::string trace, category = "S", value;
        const auto environment = [&](const std::string &key, const std::string &fallback) { event(trace, "G:" + key); return optional(c[key == "DFLASH_MODEL_PATH" ? 1 : 2]).value_or(fallback); };
        ezlocalai::DflashDownloadHooks hooks{[&](const auto &path) { event(trace, "F:" + path); return c[3] == "1"; }, [&](const auto &asset) { event(trace, "D:" + asset.repository + '\t' + asset.filename + '\t' + asset.revision); return "/cache/" + asset.filename; }};
        try { value = ezlocalai::download_dflash_model(environment, hooks); } catch (const ezlocalai::LlmSpeculativeError &error) { category = error.category; value = error.message; }
        expect(category == c[4] && value == unhex(c[5]) && trace == unhex(c[7]), "DFlash download selection " + c[0]);
    });
    expect(sizes == 210 && lengths == 1100 && probabilities == 168 && dflash == 932 && downloads == 90, "all speculative oracle cases");
    return sizes + lengths + probabilities + dflash + downloads;
}
void environment() {
    for (const auto &entry : ezlocalai::environment_defaults) unsetenv(entry.first.data());
    for (const auto prefix : {"MTP_SPEC_DRAFT_N_MAX_", "DFLASH_SPEC_DRAFT_N_MAX_"}) for (const auto family : {"T4", "3090", "4090", "5090", "A100", "H100"}) unsetenv((std::string(prefix) + family).c_str());
    for (const auto key : {"HF_HUB_OFFLINE", "HF_HUB_DISABLE_IMPLICIT_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_ENDPOINT", "LLAMA_CPP_SERVER_URL", "LLAMA_CPP_EXTRA_ARGS", "LLM_FIXTURE_MAX_LAYERS", "LLM_FIXTURE_MAX_UBATCH", "LLM_FIXTURE_FAIL_LAUNCH", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "EZLOCALAI_TEST_CUDA_FAIL"}) unsetenv(key);
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
    Fixture(const fs::path &root, const std::string &name) : directory(root / name), port(std::to_string(available_port())) {
        fs::create_directories(directory); std::ofstream(directory / "model.gguf") << "fixture GGUF";
        setenv("LLAMA_CPP_MODEL_PATH", (directory / "model.gguf").c_str(), 1); setenv("EMBEDDING_FIXTURE_DIR", directory.c_str(), 1); setenv("LLAMA_CPP_SERVER_PORT", port.c_str(), 1);
    }
    fs::path path(const std::string &name) const { return directory / (name + '-' + port); }
    void reaped() const { std::istringstream pids(read(path("launch"))); for (int pid; pids >> pid;) expect(kill(pid, 0) != 0, "speculative fixture child reaped"); }
};
std::string argument(const std::string &args, const std::string &flag) { const auto at = args.find(flag + '\n'); if (at == std::string::npos) return {}; const auto begin = at + flag.size() + 1; return args.substr(begin, args.find('\n', begin) - begin); }
void managed_cases(const fs::path &root) {
    const auto draft = root / "draft.gguf"; std::ofstream(draft) << "draft fixture";
    for (const auto &scenario : std::vector<std::tuple<std::string, std::string, int, double>>{{"Qwen3.8-27B", "auto", 3, .1}, {"3B-MTP", "auto", 4, .25}, {"35B-MTP", "auto", 2, .25}, {"ordinary", "dflash2", 0, 0}, {"Qwen3.8-27B", "none", 0, 0}, {"Qwen3.8-27B", "dflash2", 4, 0}}) {
        environment(); const auto model = std::get<0>(scenario), backend_name = std::get<1>(scenario); Fixture run(root, model + '-' + backend_name);
        setenv("LLM_SPECULATIVE_TYPE", backend_name.c_str(), 1); setenv("DFLASH_MODEL_PATH", draft.c_str(), 1);
        { ezlocalai::LlamaCppInferenceBackend backend(model); const auto response = backend.chat(R"({"messages":[]})"); expect(response.find("choices") != std::string::npos, "speculative inference " + response);
          const auto args = read(run.path("args")); const auto expected = std::get<2>(scenario) == 0 ? "none" : backend_name == "dflash2" ? "draft-dflash" : "draft-mtp";
          expect(argument(args, "--spec-type") == expected, "speculative engine type");
          if (std::get<2>(scenario)) { expect(argument(args, "--spec-draft-n-max") == std::to_string(std::get<2>(scenario)), "speculative draft length"); expect(std::stod(argument(args, "--spec-draft-p-min")) == std::get<3>(scenario), "speculative draft probability"); expect(argument(args, "-np") == "1", "speculative single slot"); }
          if (backend_name == "dflash2" && std::get<2>(scenario)) { expect(argument(args, "--spec-draft-model") == draft && argument(args, "--spec-draft-device") == draft_device_zero && argument(args, "--spec-draft-ngl") == "0" && argument(args, "--fit") == "off", "DFlash placement and memory fitting"); }
          expect(backend.model_lifecycle_json().find("\"speculative_type\":") != std::string::npos, "speculative runtime metadata");
        } run.reaped();
    }
    environment(); Fixture restored(root, "restore"); setenv("DFLASH_MODEL_PATH", draft.c_str(), 1);
    { ezlocalai::LlamaCppInferenceBackend backend("Qwen3.8-27B"); backend.chat(R"({"messages":[]})"); auto restore = backend.unload_llms_for_media();
      setenv("LLM_SPECULATIVE_TYPE", "dflash2", 1); setenv("DFLASH_SPEC_DRAFT_N_MAX_5090", "7", 1); setenv("DFLASH_SPEC_DRAFT_P_MIN", ".5", 1);
      restore(true);
      expect(backend.chat(R"({"messages":[]})").find("choices") != std::string::npos, "speculative restore");
      const auto args = read(restored.path("args")); expect(argument(args, "--spec-type") == "draft-dflash" && argument(args, "--spec-draft-n-max") == "7" && argument(args, "--spec-draft-p-min") == "0.5", "restore re-reads speculative settings");
    } restored.reaped();
    for (const auto value : {"1e-40", "1e-200", "-0.0", "NaN", "inf", "-inf"}) {
        environment(); Fixture run(root, "probability-" + std::string(value)); setenv("MTP_SPEC_DRAFT_P_MIN", value, 1); setenv("MTP_SPEC_DRAFT_N_MAX_5090", "100", 1);
        { ezlocalai::LlamaCppInferenceBackend backend("Qwen3.8-27B"); expect(backend.chat(R"({"messages":[]})").find("choices") != std::string::npos, "probability boundary startup");
          const auto args = read(run.path("args")); const auto actual = std::stof(argument(args, "--spec-draft-p-min"));
          const auto expected = static_cast<float>(ezlocalai::llm_mtp_draft_probability("Qwen3.8-27B", ezlocalai::native_llm_cache_environment()));
          expect((std::isnan(actual) && std::isnan(expected)) || (actual == expected && std::signbit(actual) == std::signbit(expected)), "engine float conversion matches direct assignment");
          expect(argument(args, "--spec-draft-n-max") == "16", "MTP card override clamps to sixteen");
        } run.reaped();
    }
    environment(); Fixture retry(root, "dflash-retry"); setenv("LLM_SPECULATIVE_TYPE", "dflash2", 1); setenv("DFLASH_MODEL_PATH", draft.c_str(), 1);
    setenv("GPU_LAYERS", "12", 1); setenv("LLAMA_CPP_TOTAL_LAYERS", "12", 1); setenv("EMBEDDING_FIXTURE_MAX_LAYERS", "5", 1); setenv("LLM_UBATCH_SIZE", "128", 1);
    { ezlocalai::LlamaCppInferenceBackend backend("Qwen3.8-27B"); expect(backend.chat(R"({"messages":[]})").find("choices") != std::string::npos, "DFlash isolated fit retry");
      const auto args = read(retry.path("args")); expect(argument(args, "-ngl") == "5" && argument(args, "--spec-draft-ngl") == "5", "draft offload tracks final target residency");
    } retry.reaped(); unsetenv("EMBEDDING_FIXTURE_MAX_LAYERS");
    environment(); Fixture worker(root, "dflash-worker"); setenv("LLM_SPECULATIVE_TYPE", "dflash2", 1); setenv("DFLASH_MODEL_PATH", draft.c_str(), 1); setenv("MAIN_GPU", "1", 1);
    { ezlocalai::LlamaCppInferenceBackend backend("Qwen3.8-27B"); expect(backend.chat(R"({"messages":[]})").find("choices") != std::string::npos, "DFlash worker GPU"); expect(argument(read(worker.path("args")), "--spec-draft-device") == draft_device_one, "draft uses selected visible GPU ordinal"); } worker.reaped();
    environment(); Fixture cpu(root, "dflash-cpu"); setenv("LLM_SPECULATIVE_TYPE", "dflash2", 1); setenv("DFLASH_MODEL_PATH", draft.c_str(), 1); setenv("CUDA_VISIBLE_DEVICES", hidden_devices, 1);
    { ezlocalai::LlamaCppInferenceBackend backend("Qwen3.8-27B"); expect(backend.chat(R"({"messages":[]})").find("choices") != std::string::npos, "DFlash CPU startup"); expect(argument(read(cpu.path("args")), "--spec-draft-device").empty(), "CPU draft leaves devices unset"); } cpu.reaped();
    for (const auto key : {"DFLASH_SPEC_DRAFT_N_MAX", "DFLASH_SPEC_DRAFT_P_MIN", "DFLASH_MODEL_PATH"}) {
        environment(); Fixture run(root, key); setenv("LLM_SPECULATIVE_TYPE", "dflash2", 1); setenv("DFLASH_MODEL_PATH", draft.c_str(), 1); setenv(key, "invalid", 1);
        { ezlocalai::LlamaCppInferenceBackend backend("Qwen3.8-27B"); const auto response = backend.chat(R"({"messages":[]})"); expect(response.find("error") != std::string::npos && !fs::exists(run.path("launch")) && backend.loaded_models_json() == "{}", "invalid DFlash settings fail before launch"); }
    }
}
}
int main(int argc, char **argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--cuda") { managed_cases(argv[2]); return 0; }
        expect(argc == 3, "reference root and fixtures required"); const auto count = reference_cases(argv[1], argv[2]);
        const auto directory = fs::temp_directory_path() / ("ezl-speculative-" + std::to_string(getpid())); fs::create_directories(directory);
        struct Cleanup { fs::path path; ~Cleanup() { fs::remove_all(path); } } cleanup{directory};
        std::ofstream(directory / "devices.tsv") << "NVIDIA GeForce RTX 5090\t12\t0\t34359738368\t32212254720\nNVIDIA GeForce RTX 3090\t8\t6\t25769803776\t21474836480\n";
        setenv("EZLOCALAI_TEST_CUDA_DEVICES", (directory / "devices.tsv").c_str(), 1); setenv("LD_LIBRARY_PATH", TEST_CUDA_DRIVER_DIRECTORY, 1);
        const auto result = ezlocalai::run_process_to_files({fs::absolute(argv[0]).string(), "--cuda", directory.string()}, (directory / "child.log").string(), "", 40, true);
        expect(!result.timed_out && result.status == 0, "managed speculative integration: " + read(directory / "child.log"));
        std::cout << "Verified " << count << " source speculative cases and managed engine startup/restoration\n"; return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
