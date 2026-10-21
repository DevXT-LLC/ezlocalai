#include "ezlocalai/model_download.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/hub_assets.hpp"
#include "ezlocalai/llm_speculative.hpp"
#include "ezlocalai/llm_cache.hpp"
#include "ezlocalai/embedding_placement.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/process.hpp"
#include "fixture_ports.hpp"
#include <arpa/inet.h>
#include <atomic>
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
#include <set>
#include <sstream>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace {
void expect(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
std::string read(const fs::path &path) { std::ifstream in(path, std::ios::binary); return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()}; }
void write(const fs::path &path, const std::string &value) { if (!path.parent_path().empty()) fs::create_directories(path.parent_path()); std::ofstream out(path, std::ios::binary); out << value; }
std::string sha256(const std::string &value) {
    unsigned char digest[SHA256_DIGEST_LENGTH]; SHA256(reinterpret_cast<const unsigned char *>(value.data()), value.size(), digest);
    std::ostringstream out; for (auto c : digest) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c); return out.str();
}
std::string hex(const std::string &value) { std::ostringstream out; for (const unsigned char c : value) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c); return out.str(); }
std::string unhex(const std::string &value) { std::string out; for (size_t i = 0; i < value.size(); i += 2) out += static_cast<char>(std::stoi(value.substr(i, 2), nullptr, 16)); return out; }
std::vector<std::string> split(const std::string &value, char delimiter = '\t') {
    std::vector<std::string> out; size_t start = 0;
    do { const auto end = value.find(delimiter, start); out.push_back(value.substr(start, end == std::string::npos ? end : end - start)); if (end == std::string::npos) break; start = end + 1; } while (true);
    return out;
}
std::vector<std::string> vector(const std::string &value) { std::vector<std::string> out; if (value != "-") for (const auto &item : split(value, ',')) out.push_back(unhex(item)); return out; }
std::string join(const std::vector<std::string> &values) { std::string out; for (const auto &value : values) { if (!out.empty()) out += '\n'; out += value; } return out; }
void environment() {
    for (const auto &entry : ezlocalai::environment_defaults) unsetenv(entry.first.data());
    for (const auto *name : {"HF_HUB_OFFLINE", "HF_HUB_DISABLE_IMPLICIT_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_ENDPOINT", "LLAMA_CPP_SERVER_URL", "LLAMA_CPP_EMBEDDING_SERVER_URL", "LLAMA_CPP_EMBEDDING_MODEL_PATH", "LLAMA_CPP_EMBEDDING_MODEL", "LLAMA_CPP_EMBEDDING_ALIAS", "LLAMA_CPP_CONTEXT", "LLAMA_CPP_EMBEDDING_CONTEXT", "LLAMA_CPP_EXTRA_ARGS", "LLAMA_CPP_EMBEDDING_EXTRA_ARGS", "EMBEDDING_FIXTURE_MAX_LAYERS", "EMBEDDING_FIXTURE_WARMUP_ERROR", "EMBEDDING_FIXTURE_TIMEOUT", "EMBEDDING_FIXTURE_FAIL_PORT"}) unsetenv(name);
    setenv("EZLOCALAI_DOTENV_PATH", "", 1);
}
size_t policy_cases(const fs::path &source, const fs::path &fixture) {
    std::istringstream input(read(fixture)); size_t count = 0;
    for (std::string line; std::getline(input, line);) {
        if (line.empty()) continue;
        if (line[0] == '#') { const auto marker = line.find(" sha256="); expect(sha256(read(source / line.substr(2, marker - 2))) == line.substr(marker + 8), "LLM.download_model reference changed"); continue; }
        const auto c = split(line); ++count; environment();
        setenv("DEFAULT_MODEL", unhex(c[1]).c_str(), 1); setenv("QUANT_TYPE", unhex(c[4]).c_str(), 1);
        std::optional<std::string> quant; if (c[3] != "N") quant = unhex(c[3].substr(1));
        auto local = vector(c[6]); const auto remote = vector(c[7]), available = vector(c[8]);
        const std::set<std::string> allowed(available.begin(), available.end()); std::set<std::string> directories;
        std::vector<std::string> events;
        ezlocalai::ModelDownloadHooks hooks;
        hooks.mkdir = [&](const auto &path) { events.push_back("mkdir\t" + path); if (c[10] == "1") throw std::runtime_error("mkdir denied"); directories.insert(path); };
        hooks.list_local = [&](const auto &path) { events.push_back("local\t" + path); return local; };
        hooks.exists = [&](const auto &path) { events.push_back("exists\t" + path); return directories.count(path) || std::find(local.begin(), local.end(), fs::path(path).filename().string()) != local.end(); };
        hooks.list_remote = [&](const auto &repo) { events.push_back("list\t" + repo); if (c[9] == "1") throw std::runtime_error("listing failed"); return remote; };
        hooks.download = [&](const auto &repo, const auto &file, const auto &dir) {
            events.push_back("download\t" + repo + '\t' + file + '\t' + dir);
            if (!allowed.count(file)) throw std::runtime_error("download failed");
            if (file.find('/') == std::string::npos && std::find(local.begin(), local.end(), file) == local.end()) local.push_back(file);
        };
        std::string result;
        try { const auto files = ezlocalai::download_gguf_model(unhex(c[0]), unhex(c[2]), quant, c[5] == "1", hooks); result = "ok\t" + files.model + '\t' + files.projector; }
        catch (const std::exception &error) { result = "error\t" + std::string(error.what()); }
        expect(result == unhex(c[11]), "download policy case " + std::to_string(count) + ": " + result + " != " + unhex(c[11]));
        expect(join(events) == unhex(c[12]), "download call order case " + std::to_string(count) + ":\n" + join(events) + "\nexpected:\n" + unhex(c[12]));
    }
    expect(count == 1068, "all source download cases"); return count;
}
struct Reply { int status = 200; std::map<std::string, std::string> headers; std::string body; };
class HttpServer {
public:
    using Handler = std::function<Reply(const std::string &, const std::string &, const std::string &)>;
    explicit HttpServer(Handler handler) : handler_(std::move(handler)) {
        fd_ = socket(AF_INET, SOCK_STREAM, 0); expect(fd_ >= 0, "Hub socket");
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        expect(!bind(fd_, reinterpret_cast<sockaddr *>(&a), sizeof(a)), "Hub bind"); socklen_t size = sizeof(a); getsockname(fd_, reinterpret_cast<sockaddr *>(&a), &size);
        port = ntohs(a.sin_port); url = "http://127.0.0.1:" + std::to_string(port); expect(!listen(fd_, 32), "Hub listen");
        thread_ = std::thread([this] { serve(); });
    }
    ~HttpServer() { stop_ = true; shutdown(fd_, SHUT_RDWR); thread_.join(); close(fd_); }
    std::string url; int port = 0;
private:
    int fd_ = -1; std::atomic<bool> stop_{false}; std::thread thread_; Handler handler_;
    void serve() {
        while (!stop_) {
            pollfd ready{fd_, POLLIN, 0}; if (poll(&ready, 1, 100) <= 0) continue;
            const int peer = accept(fd_, nullptr, nullptr); if (peer < 0) continue;
            timeval timeout{2, 0}; setsockopt(peer, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            std::string request; char buffer[4096];
            while (request.find("\r\n\r\n") == std::string::npos) { const auto size = recv(peer, buffer, sizeof(buffer), 0); if (size <= 0) break; request.append(buffer, size); }
            std::istringstream start(request); std::string method, path; start >> method >> path;
            const auto reply = handler_(method, path, request);
            std::ostringstream response; response << "HTTP/1.1 " << reply.status << " Response\r\nConnection: close\r\nContent-Length: " << reply.body.size() << "\r\n";
            for (const auto &header : reply.headers) response << header.first << ": " << header.second << "\r\n";
            response << "\r\n"; if (method != "HEAD") response << reply.body;
            const auto wire = response.str(); size_t sent = 0;
            while (sent < wire.size()) { const auto size = send(peer, wire.data() + sent, wire.size() - sent, MSG_NOSIGNAL); if (size <= 0) break; sent += size; }
            close(peer);
        }
    }
};
size_t transport_cases(const fs::path &fixture) {
    std::istringstream input(read(fixture)); size_t count = 0;
    const auto initial_payload = std::string("GGUF") + '\0' + "fixture data\n";
    for (std::string line; std::getline(input, line);) {
        if (line.empty() || line[0] == '#') continue;
        ++count; const auto c = split(line); const auto mode = c[0]; environment();
        const auto base = fs::absolute(mode), local = base / "local", cache = base / "cache", file = local / "model.gguf", metadata = local / ".cache/huggingface/download/model.gguf.metadata";
        fs::create_directories(local); setenv("HF_HUB_CACHE", cache.c_str(), 1); setenv("HF_TOKEN", "test-readonly", 1);
        std::string payload = initial_payload, commit(40, 'a'); std::vector<std::string> trace; std::mutex mutex;
        const bool git = mode == "preplaced_git" || mode == "corrupt_metadata" || mode == "download_failure";
        auto record = [&](const auto &host, const auto &method, const auto &path, const auto &raw) { trace.push_back(host + '\t' + method + '\t' + path + '\t' + (raw.find("Authorization:") == std::string::npos ? "0" : "1")); };
        HttpServer cdn([&](const auto &method, const auto &path, const auto &raw) { std::lock_guard<std::mutex> lock(mutex); record(std::string("cdn"), method, path, raw); return Reply{200, {}, payload}; });
        std::string base_url;
        HttpServer server([&](const auto &method, const auto &path, const auto &raw) {
            std::lock_guard<std::mutex> lock(mutex); record(std::string("hub"), method, path, raw);
            if (path.rfind("/api/", 0) == 0) {
                if (path.find("page=2") != std::string::npos) return Reply{200, {}, R"([{"type":"file","path":"nested/quoted\"file.gguf","oid":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","size":12}])"};
                return Reply{200, {{"Link", '<' + base_url + "/api/models/Owner/Model-GGUF/tree/main?page=2>; rel=\"next\""}}, R"([{"type":"directory","path":"nested","oid":"dddddddddddddddddddddddddddddddddddddddd","size":0},{"type":"file","path":"\u6a21\u578b-Q8_0.gguf","oid":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","size":12}])"};
            }
            if (method == "HEAD") {
                if (mode == "head_forbidden_local") return Reply{403, {}, ""};
                if (mode == "head_missing_local") return Reply{404, {{"X-Error-Code", "EntryNotFound"}}, ""};
                if (mode == "relative" && path != "/canonical") return Reply{307, {{"Location", "/canonical"}}, ""};
                std::map<std::string, std::string> headers{{"X-Repo-Commit", commit}, {"ETag", '"' + (git ? std::string(40, 'b') : sha256(payload)) + '"'}, {"X-Linked-Size", std::to_string(payload.size() + (mode == "wrong_size"))}};
                if (mode == "redirect") { headers["Location"] = cdn.url + "/blob"; return Reply{302, headers, ""}; }
                return Reply{200, headers, payload};
            }
            return mode == "download_failure" ? Reply{400, {}, ""} : Reply{200, {}, payload};
        });
        { std::lock_guard<std::mutex> lock(mutex); base_url = server.url; }
        setenv("HF_ENDPOINT", server.url.c_str(), 1);
        if (mode == "preplaced_hash" || mode == "preplaced_git" || mode == "corrupt_metadata" || mode == "head_forbidden_local" || mode == "head_missing_local" || mode == "download_failure") write(file, payload);
        if (mode == "corrupt_metadata") write(metadata, "bad");
        if (mode == "cache") write(cache / "models--Owner--Model-GGUF/snapshots" / commit / "model.gguf", payload);
        auto download = [&](const std::string &revision = "main") { return ezlocalai::download_hub_file("Owner/Model-GGUF", "model.gguf", local.string(), revision); };
        std::string status = "ok"; std::vector<std::string> listing;
        try {
            if (mode == "list") listing = ezlocalai::list_hub_files("Owner/Model-GGUF");
            else {
                download();
                if (mode == "reuse") download();
                if (mode == "commit_reuse") download(commit);
                if (mode == "stale_metadata") { write(file, "changed"); fs::last_write_time(file, fs::file_time_type::clock::now() + std::chrono::seconds(3)); download(); }
                if (mode == "update") { { std::lock_guard<std::mutex> lock(mutex); payload = "updated payload"; commit = std::string(40, 'c'); } download(); }
            }
        } catch (const std::exception &) { status = "error"; }
        expect(status == c[1], mode + " result " + status + " != " + c[1]);
        expect((fs::is_regular_file(file) ? hex(read(file)) : "-") == c[2], mode + " downloaded bytes/file cleanup");
        auto lines = split(read(metadata), '\n'); const auto saved = lines.size() >= 2 ? lines[0] + '\n' + lines[1] : "";
        expect(hex(saved) == c[3], mode + " local commit/ETag metadata");
        { std::lock_guard<std::mutex> lock(mutex); expect(join(trace) == unhex(c[4]), mode + " HTTP trace:\n" + join(trace) + "\nexpected:\n" + unhex(c[4])); }
        expect(join(listing) == unhex(c[5]), mode + " listed file order/Unicode");
        for (const auto &entry : fs::recursive_directory_iterator(local)) expect(entry.path().filename().string().rfind(".hub-", 0) != 0, mode + " download temporaries cleaned");
    }
    expect(count == 16, "all Hub oracle traces"); return count;
}
void dflash_cache_cases() {
    environment(); setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1", 1);
    const auto cache = fs::absolute("draft-cache"); const auto payload = std::string("GGUF draft fixture");
    setenv("HF_HUB_CACHE", cache.c_str(), 1);
    std::atomic<int> heads{0}, gets{0}; std::atomic<bool> wrong_path{false};
    const auto prefix = std::string("/") + ezlocalai::dflash_repository + "/resolve/" + ezlocalai::dflash_revision + '/';
    HttpServer server([&](const auto &method, const auto &path, const auto &) {
        if (path != prefix + ezlocalai::dflash_filename && path != prefix + "custom.gguf") wrong_path = true;
        if (method == "HEAD") ++heads; else ++gets;
        return Reply{200, {{"ETag", '"' + sha256(payload) + '"'}, {"X-Repo-Commit", ezlocalai::dflash_revision}}, payload};
    });
    setenv("HF_ENDPOINT", server.url.c_str(), 1);
    const auto download = [] { return ezlocalai::download_dflash_model(ezlocalai::native_llm_cache_environment()); };
    std::vector<std::future<std::string>> workers;
    for (int i = 0; i < 6; ++i) workers.push_back(std::async(std::launch::async, download));
    const auto expected = cache / "models--incoai--Qwen3.8-27B-DFlash2-GGUF/snapshots" / ezlocalai::dflash_revision / ezlocalai::dflash_filename;
    for (auto &worker : workers) expect(worker.get() == expected && read(expected) == payload, "DFlash shared immutable cache path");
    expect(heads == 1 && gets == 1 && !wrong_path, "concurrent DFlash resolution transfers draft once at pinned revision");
    setenv("HF_HUB_OFFLINE", "1", 1); expect(download() == expected && heads == 1 && gets == 1, "DFlash shared-cache offline reuse");
    setenv("DFLASH_MODEL_FILE", "custom.gguf", 1); bool failed = false;
    try { download(); } catch (const std::exception &) { failed = true; }
    expect(failed && heads == 1 && gets == 1, "missing offline draft fails without traffic");
    unsetenv("HF_HUB_OFFLINE"); const auto custom = download();
    expect(fs::path(custom).filename() == "custom.gguf" && read(custom) == payload && heads == 2 && gets == 1, "custom draft filename reuses matching blob");
    setenv("DFLASH_MODEL_PATH", expected.c_str(), 1); setenv("HF_HUB_OFFLINE", "1", 1);
    expect(download() == expected && heads == 2, "explicit draft file bypasses Hub");
    setenv("DFLASH_MODEL_PATH", cache.c_str(), 1); failed = false;
    try { download(); } catch (const ezlocalai::LlmSpeculativeError &error) { failed = error.category == "FileNotFoundError"; }
    expect(failed && heads == 2, "directory cannot substitute for explicit draft file");
    environment(); setenv("HF_HOME", "/fixture/home", 1); expect(ezlocalai::default_hub_cache_directory() == "/fixture/home/hub", "HF home cache default");
    setenv("HUGGINGFACE_HUB_CACHE", "/fixture/legacy", 1); expect(ezlocalai::default_hub_cache_directory() == "/fixture/legacy", "legacy cache override");
    setenv("HF_HUB_CACHE", "/fixture/current", 1); expect(ezlocalai::default_hub_cache_directory() == "/fixture/current", "current cache takes precedence");
    environment();
}
int free_port() { return fixture_ports(3); }
void engine_child(const std::string &endpoint, const fs::path &model_root, const std::string &mode) {
    environment(); fs::current_path(model_root);
    setenv("HF_ENDPOINT", endpoint.c_str(), 1); setenv("HF_TOKEN", "test-readonly", 1); setenv("HF_HUB_CACHE", (model_root / "cache").c_str(), 1);
    setenv("DEFAULT_MODEL", "none", 1); setenv("EMBEDDING_MODEL", "Owner/Model-GGUF", 1); setenv("EMBEDDING_QUANT_TYPE", "Q8_0", 1);
    setenv("EMBEDDING_VRAM_SAFETY_MARGIN", "0.1", 1); setenv("EMBEDDING_MAIN_GPU", "0", 1); setenv("CUDA_VISIBLE_DEVICES", "0", 1);
    const auto port = free_port(); setenv("LLAMA_CPP_EMBEDDING_SERVER_PORT", std::to_string(port).c_str(), 1);
    setenv("LLAMA_CPP_SERVER_BIN", TEST_EMBEDDING_FIXTURE_BIN, 1); setenv("LLAMA_CPP_START_TIMEOUT", "2", 1); setenv("EMBEDDING_FIXTURE_DIR", model_root.c_str(), 1);
    {
        ezlocalai::LlamaCppInferenceBackend backend("none"); backend.warm_aux_model("embedding");
        if (mode == "failure") expect(backend.loaded_models_json() == "{}" && !fs::exists("launch-" + std::to_string(port)), "failed acquisition publishes no engine");
        else {
            expect(backend.loaded_models_json().find("\"device\":\"cuda\"") != std::string::npos, "downloaded embedding published with GPU placement");
            const auto args = read("args-" + std::to_string(port));
            expect(args.find("-m\n") != std::string::npos && args.find("model-Q8_0.gguf") != std::string::npos && args.find("-hf\n") == std::string::npos, "engine receives an acquired local GGUF");
            expect(args.find("-ngl\n4\n") != std::string::npos, "downloaded GGUF metadata and size determine partial offload");
            expect(read("seen-" + std::to_string(port)).find("Warm up the embedding model.") != std::string::npos, "downloaded engine performs inference warmup");
            if (mode == "restore") {
                expect(backend.unload_aux_model("embedding"), "downloaded engine unloads");
                backend.warm_aux_model("embedding");
                expect(backend.loaded_models_json().find("\"device\":\"cuda\"") != std::string::npos, "reacquired engine restores");
            }
        }
    }
    std::istringstream pids(read("launch-" + std::to_string(port))); for (int pid; pids >> pid;) expect(kill(pid, 0) != 0, "downloaded engine is reaped");
}
void acquisition_cases(const std::string &self, const fs::path &fixtures) {
    environment(); const auto weights = read(fixtures / "embedding_gguf/normal_le3.gguf");
    std::atomic<int> lists{0}, heads{0}, gets{0}; std::atomic<bool> reject{false}, wrong_auth{false};
    HttpServer hub([&](const auto &method, const auto &path, const auto &raw) {
        if (raw.find("Authorization: Bearer test-readonly\r\n") == std::string::npos) wrong_auth = true;
        if (path.rfind("/api/", 0) == 0) { ++lists; return reject ? Reply{403, {}, ""} : Reply{200, {}, R"([{"type":"file","path":"mmproj-F16.gguf"},{"type":"file","path":"model-Q4_K_M.gguf"},{"type":"file","path":"nested/model-Q8_0.gguf"}])"}; }
        if (method == "HEAD") { ++heads; return Reply{200, {{"X-Repo-Commit", std::string(40, 'a')}, {"ETag", '"' + sha256(weights) + '"'}}, weights}; }
        ++gets; return Reply{200, {}, weights};
    });
    const auto devices = fs::absolute("devices.tsv"); write(devices, "Fixture\t8\t6\t8589934592\t805306368\n");
    setenv("EZLOCALAI_TEST_CUDA_DEVICES", devices.c_str(), 1);
    const char *previous = std::getenv("LD_LIBRARY_PATH"); const std::string saved = previous ? previous : "";
    setenv("LD_LIBRARY_PATH", (std::string(TEST_CUDA_DRIVER_DIRECTORY) + ':' + saved).c_str(), 1);
    const auto root = fs::absolute("managed"); fs::create_directories(root);
    auto child = [&](const std::string &mode) {
        const auto result = ezlocalai::run_process_to_files({self, "--engine", hub.url, root.string(), mode}, (root / "child.log").string(), "", 15, true);
        expect(!result.timed_out && result.status == 0, "acquisition/placement child: " + read(root / "child.log"));
    };
    reject = true; child("failure"); expect(lists == 1 && heads == 0 && gets == 0, "listing failure stops before launch/download");
    reject = false; child("success"); expect(lists == 2 && heads == 1 && gets == 1 && !wrong_auth, "selected GGUF acquired once without projector requests");
    child("restore"); expect(lists == 4 && heads == 3 && gets == 1, "new and restored instances recheck nested cached files without downloading weights");
    if (previous) setenv("LD_LIBRARY_PATH", saved.c_str(), 1); else unsetenv("LD_LIBRARY_PATH");
    environment(); setenv("HF_HUB_OFFLINE", "1", 1); setenv("HF_ENDPOINT", hub.url.c_str(), 1);
    const auto local = fs::absolute("offline/models/Model"); write(local / "model-Q8_0.gguf", weights);
    const auto files = ezlocalai::download_gguf_model("Owner/Model-GGUF", local.parent_path().string(), std::string("Q8_0"), true);
    expect(files.model == (local / "model-Q8_0.gguf").string() && lists == 4, "matching local model needs no Hub access even offline");
    bool failed = false; try { ezlocalai::download_gguf_model("Owner/Model-GGUF", local.parent_path().string(), std::string("Q5_K_M"), true); } catch (const std::exception &error) { failed = std::string(error.what()) == "No suitable model file found for Owner/Model-GGUF"; }
    expect(failed && lists == 4, "offline unmatched quant does not silently substitute a cached file");
}
void text_engine_child(const std::string &endpoint, const fs::path &root, const std::string &mode) {
    environment(); fs::current_path(root);
    setenv("HF_ENDPOINT", endpoint.c_str(), 1); setenv("HF_TOKEN", "text-fixture-readonly", 1); setenv("HF_HUB_CACHE", (root / "cache").c_str(), 1);
    setenv("LLAMA_CPP_SERVER_BIN", TEST_EMBEDDING_FIXTURE_BIN, 1); setenv("LLAMA_CPP_START_TIMEOUT", "2", 1); setenv("EMBEDDING_FIXTURE_DIR", root.c_str(), 1);
    setenv("GPU_LAYERS", "0", 1); setenv("LLM_PROMPT_CACHE_ALLOW_UNSAFE", "true", 1);
    const int port = free_port(); setenv("LLAMA_CPP_SERVER_PORT", std::to_string(port).c_str(), 1);
    std::string models = "Owner/Model-GGUF", request = R"({"messages":[]})", base = "Model"; int selected = port;
    setenv("QUANT_TYPE", "Q8_0", 1);
    if (mode == "selected") { models = "Owner/Other-GGUF,Owner/Model-GGUF"; setenv("QUANT_TYPE", "Q4_K_M,Q8_0", 1); request = R"({"model":"Owner/Model-GGUF","messages":[]})"; selected = port + 1; }
    if (mode == "replica") { models = "Owner/Qwen3.8-27B-GGUF"; base = "Qwen3.8-27B"; setenv("N_PARALLEL", "2", 1); request = R"({"model":"Owner/Qwen3.8-27B-GGUF#2","messages":[]})"; selected = port + 1; }
    if (mode == "override") { setenv("LLAMA_CPP_MODEL_PATH", (root / "override.gguf").c_str(), 1); setenv("LLAMA_CPP_MMPROJ_PATH", "explicit-projector.gguf", 1); }
    if (mode == "offline") setenv("HF_HUB_OFFLINE", "1", 1);
    if (mode == "retry") { setenv("GPU_LAYERS", "1000", 1); setenv("LLM_UBATCH_SIZE", "128", 1); setenv("EMBEDDING_FIXTURE_MAX_LAYERS", "5", 1); }
    if (mode == "custom-root") setenv("MODELS_PATH", "custom-models", 1);
    const auto suffix = std::to_string(selected), folder = (mode == "custom-root" ? "custom-models/" : "models/") + base;
    {
        ezlocalai::LlamaCppInferenceBackend backend(models); const auto response = backend.chat(request);
        if (mode == "failure") {
            expect(response.find("No suitable model file found for Owner/Model-GGUF") != std::string::npos && !fs::exists("launch-" + suffix), "failed text acquisition stops before child startup");
            expect(backend.loaded_models_json() == "{}" && backend.model_lifecycle_json().find("\"count\":1,\"failures\":1") != std::string::npos, "failed text acquisition records one lifecycle failure");
        } else {
            expect(response.find("choices") != std::string::npos, "acquired text inference: " + response);
            const auto args = read("args-" + suffix);
            expect(args.find("-hf\n") == std::string::npos && args.find("-m\n") != std::string::npos, "owned text engine receives acquired local path");
            if (mode == "override") expect(args.find("override.gguf") != std::string::npos && args.find("explicit-projector.gguf") != std::string::npos, "native explicit model/projector overrides");
            else {
                expect(args.find(folder + (mode == "offline" ? "/model-Q8_0.gguf" : "/nested/model-Q8_0.gguf")) != std::string::npos, "requested quant selected within source model folder");
                expect((args.find("--mmproj\n") != std::string::npos) == (mode != "no-projector"), "optional discovered projector reaches engine");
                if (mode != "no-projector") expect(args.find(folder + "/mmproj-BF16.gguf") != std::string::npos, "BF16 projector discovery");
            }
            expect(!read("probe-" + suffix).empty(), "acquired text engine performs one-token verification");
            if (mode == "replica") expect(args.find("--image-min-tokens\n1024\n") != std::string::npos && args.find("--spec-type\ndraft-mtp\n") != std::string::npos, "acquired vision projector and inherited speculative replica settings");
            if (mode == "retry") {
                const auto layers = ezlocalai::embedding_model_metadata(folder + "/nested/model-Q8_0.gguf").layers;
                std::istringstream attempts(read("attempts-" + suffix)); int initial, first_probe; attempts >> initial >> first_probe;
                // Text fit searches include the output layer; embedding
                // metadata reports only transformer blocks.
                expect(initial == 1000 && first_probe == layers / 2, "acquired GGUF layer count bounds isolated fit search");
                expect(args.find("-ngl\n5\n") != std::string::npos, "acquired model retry reaches maximum fitting layers");
            }
            if (mode == "restore" || mode == "missing-restore") {
                auto restore = backend.unload_llms_for_media(); setenv("QUANT_TYPE", "Q4_K_M", 1);
                if (mode == "missing-restore") { fs::remove(folder + "/nested/model-Q8_0.gguf"); write("reject", "1"); }
                restore(true);
                if (mode == "missing-restore") expect(backend.loaded_models_json() == "{}", "failed restoration cannot use stale selected model path");
                else expect(backend.loaded_models_json() != "{}" && read("args-" + suffix).find("nested/model-Q8_0.gguf") != std::string::npos, "restoration repeats acquisition with captured per-model quant");
            }
        }
    }
    std::istringstream pids(read("launch-" + suffix)); int launches = 0;
    for (int pid; pids >> pid;) { expect(kill(pid, 0) != 0, "acquired text children reaped"); ++launches; }
    write("launch-count", std::to_string(launches));
}
void text_acquisition_cases(const std::string &self, const fs::path &fixtures) {
    const auto weights = read(fixtures / "embedding_gguf/normal_le3.gguf");
    for (const auto mode : {"selected", "restore", "replica", "wrong-local", "preflight-recovery", "failure", "no-projector", "retry", "offline", "override", "custom-root", "missing-restore"}) {
        environment(); const auto root = fs::absolute(std::string("text-") + mode); fs::create_directories(root);
        const std::string expected_repo = std::string(mode) == "replica" ? "Owner/Qwen3.8-27B-GGUF" : "Owner/Model-GGUF";
        std::atomic<int> lists{0}, model_heads{0}, model_gets{0}, projector_heads{0}, projector_gets{0}; std::atomic<bool> wrong_request{false};
        HttpServer hub([&](const auto &method, const auto &path, const auto &raw) {
            if (raw.find("Authorization: Bearer text-fixture-readonly\r\n") == std::string::npos || path.find(expected_repo + '/') == std::string::npos) wrong_request = true;
            if (path.rfind("/api/", 0) == 0) {
                const int count = ++lists;
                if (std::string(mode) == "failure" || fs::exists(root / "reject") || (std::string(mode) == "preflight-recovery" && count == 1)) return Reply{403, {}, ""};
                return Reply{200, {}, R"([{"type":"file","path":"mmproj-BF16.gguf"},{"type":"file","path":"model-Q4_K_M.gguf"},{"type":"file","path":"nested/model-Q8_0.gguf"}])"};
            }
            const bool projector = path.find("mmproj") != std::string::npos;
            if (projector) {
                if (method == "HEAD") ++projector_heads; else ++projector_gets;
                if (std::string(mode) == "no-projector" || path.find("mmproj-BF16.gguf") == std::string::npos) return Reply{404, {{"X-Error-Code", "EntryNotFound"}}, ""};
            } else {
                if (path.find("nested/model-Q8_0.gguf") == std::string::npos) wrong_request = true;
                if (method == "HEAD") ++model_heads; else ++model_gets;
            }
            const auto payload = projector ? std::string("projector fixture") : weights;
            return Reply{200, {{"X-Repo-Commit", std::string(40, 'a')}, {"ETag", '"' + sha256(payload) + '"'}}, payload};
        });
        if (std::string(mode) == "wrong-local") { write(root / "models/Model/wrong-Q4_K_M.gguf", "wrong quant"); write(root / "models/other/unrelated-Q8_0.gguf", "unrelated model"); }
        if (std::string(mode) == "offline") { write(root / "models/Model/model-Q8_0.gguf", weights); write(root / "models/Model/mmproj-BF16.gguf", "projector fixture"); }
        if (std::string(mode) == "override") write(root / "override.gguf", weights);
        const auto result = ezlocalai::run_process_to_files({self, "--text-engine", hub.url, root.string(), mode}, (root / "child.log").string(), "", 35, true);
        expect(!result.timed_out && result.status == 0, std::string(mode) + " text acquisition: " + read(root / "child.log"));
        expect(!wrong_request, std::string(mode) + " model source, quant and authorization");
        if (std::string(mode) == "offline" || std::string(mode) == "override") expect(lists == 0 && projector_heads == 0 && model_heads == 0, "local text files require no Hub traffic");
        else {
            const int expected_lists = std::string(mode) == "restore" || std::string(mode) == "missing-restore" ? 4 : std::string(mode) == "retry" ? std::stoi(read(root / "launch-count")) + 1 : 2;
            expect(lists == expected_lists, std::string(mode) + " repeats reference preliminary and constructor acquisition: " + std::to_string(lists));
            const int expected_heads = std::string(mode) == "failure" ? 0 : std::string(mode) == "preflight-recovery" ? 1 : std::string(mode) == "missing-restore" ? 2 : expected_lists;
            expect(model_heads == expected_heads && model_gets == (std::string(mode) == "failure" ? 0 : 1), "model HEAD metadata rechecked while unchanged bytes download once");
            expect(projector_heads == (std::string(mode) == "no-projector" ? 16 : 2) && projector_gets == (std::string(mode) == "no-projector" ? 0 : 1), "source projector preference and local reuse");
        }
    }
    environment();
}
void real_engine_smoke(const fs::path &engine, const fs::path &model) {
    // Optional, supplied dependency: ggml-org/models at
    // 499bc8821c6b12b4e53c5bffcb21ec206f212d81, tinyllamas/stories260K.gguf.
    // The normal test suite never downloads weights or invokes a real engine.
    const auto weights = read(model), digest = sha256(weights);
    expect(digest == "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d", "real smoke model checksum differs from pinned asset");
    const auto initial = fs::current_path(), root = fs::temp_directory_path() / ("ezlocalai-real-engine-" + std::to_string(getpid()));
    struct Cleanup { fs::path initial, root; ~Cleanup() { fs::current_path(initial); fs::remove_all(root); } } cleanup{initial, root};
    fs::create_directories(root); fs::current_path(root); environment();
    for (const auto *key : {"LLAMA_CPP_MODEL_PATH", "LLAMA_CPP_MODEL_PATHS", "LLAMA_CPP_MMPROJ_PATH", "MODELS_PATH", "TENSOR_SPLIT"}) unsetenv(key);
    std::atomic<int> lists{0}, heads{0}, gets{0}, projector_heads{0}; std::atomic<bool> wrong_request{false};
    HttpServer hub([&](const auto &method, const auto &path, const auto &raw) {
        if (path.find("/Smoke/Stories-GGUF/") == std::string::npos || raw.find("Authorization:") != std::string::npos) wrong_request = true;
        if (path.rfind("/api/", 0) == 0) { ++lists; return Reply{200, {}, R"([{"type":"file","path":"nested/stories-F32.gguf"}])"}; }
        if (path.find("mmproj") != std::string::npos) { ++projector_heads; return Reply{404, {{"X-Error-Code", "EntryNotFound"}}, ""}; }
        if (path.find("/nested/stories-F32.gguf") == std::string::npos) wrong_request = true;
        if (method == "HEAD") ++heads; else ++gets;
        return Reply{200, {{"X-Repo-Commit", std::string(40, 'a')}, {"ETag", '"' + digest + '"'}}, weights};
    });
    setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1", 1); setenv("HF_ENDPOINT", hub.url.c_str(), 1); setenv("HF_HUB_CACHE", (root / "cache").c_str(), 1);
    const int engine_port = free_port();
    setenv("LLAMA_CPP_SERVER_BIN", engine.c_str(), 1); setenv("LLAMA_CPP_SERVER_PORT", std::to_string(engine_port).c_str(), 1); setenv("LLAMA_CPP_START_TIMEOUT", "30", 1);
    setenv("DEFAULT_MODEL", "Smoke/Stories-GGUF", 1); setenv("QUANT_TYPE", "F32", 1); setenv("GPU_LAYERS", "0", 1); setenv("LLM_MAX_TOKENS", "512", 1);
    setenv("LLM_BATCH_SIZE", "64", 1); setenv("LLM_UBATCH_SIZE", "64", 1); setenv("KV_CACHE_TYPE", "f16", 1); setenv("LLM_PROMPT_CACHE_MIB", "0", 1);
    setenv("LLAMA_CPP_EXTRA_ARGS", "--threads 2 --threads-batch 2 --chat-template chatml", 1);
    const std::string prompt = R"({"model":"Smoke/Stories-GGUF","prompt":"Once upon a time","max_tokens":16,"temperature":0,"seed":1234})";
    std::string generated;
    {
        ezlocalai::LlamaCppInferenceBackend backend("Smoke/Stories-GGUF");
        const auto first = backend.completion(prompt);
        expect(first.find("\"choices\"") != std::string::npos && first.find("\"error\"") == std::string::npos, "real CPU completion: " + first);
        generated = ezlocalai::extract_json_string(first, "text", "");
        expect(!generated.empty() && ezlocalai::extract_json_string(first, "finish_reason", "") == "length", "real CPU engine generates requested tokens: " + first);
        expect(backend.llm_context_size() == 512 && backend.loaded_models_json() != "{}", "real acquired engine records runtime context and residency");
        expect(sha256(read("models/Stories/nested/stories-F32.gguf")) == digest, "native downloader preserves real GGUF bytes");
        const auto repeated = backend.completion(prompt);
        expect(ezlocalai::extract_json_string(repeated, "text", "") == generated, "real repeated greedy completion is stable");
        const auto rejected = backend.text_result(prompt.substr(0, prompt.size() - 1) + ",\"stream\":true}", "", false);
        expect(rejected.failure && rejected.failure->category == "ValueError", "completion streaming preserves pinned binding callback requirement");
        const std::string chat_prompt = R"({"messages":[{"role":"user","content":"Once upon a time"}],"max_tokens":16,"temperature":0})";
        const auto chat = backend.chat(chat_prompt);
        const auto chat_text = ezlocalai::extract_json_string(chat, "content", "");
        expect(!chat_text.empty(), "real buffered chat generates text: " + chat);
        const auto streamed = backend.chat(chat_prompt.substr(0, chat_prompt.size() - 1) + ",\"stream\":true}");
        expect(streamed.find("data: [DONE]") != std::string::npos, "real streaming chat terminates");
        std::istringstream events(streamed); std::string streamed_text;
        for (std::string line; std::getline(events, line);) if (line.rfind("data: {", 0) == 0)
            streamed_text += ezlocalai::extract_json_string(line.substr(6), "content", "");
        // LLM.completion strips buffered text; the streaming wrapper preserves
        // raw token whitespace. This particular pinned sample has no stop or
        // placeholder markers (those branches have source-executed fixtures).
        expect(streamed_text.find('<') == std::string::npos && streamed_text.find("[Insert ") == std::string::npos,
            "real sample remains within the independently checked whitespace case");
        expect(ezlocalai::strip_config_value(streamed_text) == chat_text, "real buffered cleaning preserves greedy streaming chat text");
        auto restore = backend.unload_llms_for_media();
        expect(backend.loaded_models_json() == "{}", "real engine unloads for media handoff");
        restore(true);
        const auto restored = backend.completion(prompt);
        expect(ezlocalai::extract_json_string(restored, "text", "") == generated, "real restored engine preserves greedy completion");
    }
    expect(!wrong_request && lists == 4 && heads == 4 && gets == 1 && projector_heads == 32, "real acquisition and reload follow reference selection while reusing weights");
    const int probe = socket(AF_INET, SOCK_STREAM, 0); expect(probe >= 0, "real engine cleanup probe socket");
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK); address.sin_port = htons(engine_port);
    const auto connected = connect(probe, reinterpret_cast<sockaddr *>(&address), sizeof(address)); close(probe);
    expect(connected != 0, "real engine no longer listens after backend destruction");
    std::cout << "Verified pinned real CPU engine acquisition, startup, repeated/streamed generation and unload/reload; generated text: " << generated << '\n';
}
}
int main(int argc, char **argv) {
    try {
        if (argc == 5 && std::string(argv[1]) == "--engine") { engine_child(argv[2], argv[3], argv[4]); return 0; }
        if (argc == 5 && std::string(argv[1]) == "--text-engine") { text_engine_child(argv[2], argv[3], argv[4]); return 0; }
        if (argc == 4 && std::string(argv[1]) == "--real-engine") { real_engine_smoke(fs::absolute(argv[2]), fs::absolute(argv[3])); return 0; }
        expect(argc == 4, "reference and two fixtures required");
        const auto root = fs::absolute(argv[1]), policy = fs::absolute(argv[2]), transport = fs::absolute(argv[3]), self = fs::absolute(argv[0]);
        const auto initial = fs::current_path(), temp = fs::temp_directory_path() / ("ezlocalai-model-download-" + std::to_string(getpid()));
        struct Cleanup { fs::path initial, temp; ~Cleanup() { fs::current_path(initial); fs::remove_all(temp); } } cleanup{initial, temp};
        fs::create_directories(temp); fs::current_path(temp);
        const auto policies = policy_cases(root, policy), traces = transport_cases(transport);
        acquisition_cases(self.string(), policy.parent_path());
        dflash_cache_cases();
        text_acquisition_cases(self.string(), policy.parent_path());
        std::cout << "Verified " << policies << " reference model-download cases, " << traces << " Hub HTTP traces, offline reuse and managed acquisition/placement/warmup\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
