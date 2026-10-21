#include "ezlocalai/embedding_runtime.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/native_server.hpp"
#include <arpa/inet.h>
#include <atomic>
#include <csignal>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <openssl/sha.h>
#include <sstream>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
using Pool = ezlocalai::EmbeddingPool;
namespace {
void expect(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
std::string read(const fs::path &path) { std::ifstream in(path, std::ios::binary); return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()}; }
std::vector<std::string> split(const std::string &value, char delimiter = '\t') {
    std::vector<std::string> out; size_t offset = 0;
    do { const auto end = value.find(delimiter, offset); out.push_back(value.substr(offset, end == std::string::npos ? end : end - offset)); if (end == std::string::npos) break; offset = end + 1; } while (true);
    return out;
}
std::string unhex(const std::string &value) { std::string out; for (size_t i = 0; i < value.size(); i += 2) out += static_cast<char>(std::stoi(value.substr(i, 2), nullptr, 16)); return out; }
std::string sha256(const std::string &value) {
    unsigned char hash[SHA256_DIGEST_LENGTH]; SHA256(reinterpret_cast<const unsigned char *>(value.data()), value.size(), hash);
    std::ostringstream out; for (const auto byte : hash) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte); return out.str();
}
template<class Predicate> void wait_until(Predicate predicate, const std::string &message) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!predicate()) { expect(std::chrono::steady_clock::now() < deadline, message); std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
}
void environment() {
    for (const auto &entry : ezlocalai::environment_defaults) unsetenv(entry.first.data());
    for (const char *key : {"LLAMA_CPP_SERVER_URL", "LLAMA_CPP_EMBEDDING_SERVER_URL", "LLAMA_CPP_SERVER_BIN", "LLAMA_CPP_CONTEXT", "LLAMA_CPP_MODEL_PATH", "LLAMA_CPP_MODEL_PATHS", "LLAMA_CPP_EMBEDDING_MODEL_PATH", "EZLOCALAI_NATIVE_BACKEND", "EMBEDDING_FIXTURE_FAIL_PORT"}) unsetenv(key);
    setenv("EZLOCALAI_DOTENV_PATH", "", 1); setenv("ROUTER_DRAIN_FILE", "", 1); setenv("USAGE_FILE", "usage.json", 1); setenv("DEFAULT_MODEL", "none", 1);
    // This fixture verifies embedding startup. Keep speech cache warmup from
    // downloading real TTS weights before the executable opens its listener.
    setenv("TTS_ENABLED", "false", 1); setenv("STT_ENABLED", "false", 1);
}
std::string ids(const std::vector<Pool::Handle> &values) { std::string out; for (auto v : values) { if (!out.empty()) out += ','; out += std::to_string(v); } return out.empty() ? "-" : out; }
size_t reference_cases(const fs::path &root, const fs::path &fixture) {
    std::istringstream lines(read(fixture)); size_t count = 0, steps = 0;
    for (std::string line; std::getline(lines, line);) {
        if (line.empty()) continue;
        if (line[0] == '#') { const auto marker = line.find(" sha256="); expect(sha256(read(root / line.substr(2, marker - 2))) == line.substr(marker + 8), "embedding lifecycle reference changed"); continue; }
        ++count; const auto c = split(line); environment();
        if (c[0] == "slots") {
            setenv("EMBEDDING_N_PARALLEL", unhex(c[1]).c_str(), 1);
            expect(ezlocalai::embedding_parallel_slots() == std::stoi(c[2]), "parallel slots: " + unhex(c[1]));
        } else if (c[0] == "config") {
            setenv("EMBEDDING_KEEP_LOADED", unhex(c[1]).c_str(), 1); setenv("EMBEDDING_ENABLED", unhex(c[2]).c_str(), 1); setenv("EMBEDDING_SERVER", unhex(c[3]).c_str(), 1);
            expect(ezlocalai::embedding_keep_loaded() == (c[4] == "1") && ezlocalai::embedding_should_preload() == (c[5] == "1"), "embedding retention/preload predicate");
        } else {
            expect(c[0] == "pool", "known pool case");
            bool keep = c[2] == "1"; int attempt = 0, fail_at = 0; Pool::Handle next = 0; std::string events;
            auto event = [&](const std::string &value) { if (!events.empty()) events += ','; events += value; };
            Pool pool(std::stoul(c[1]), [&](size_t slot) {
                if (++attempt == fail_at) { event("fail:" + std::to_string(slot)); throw std::runtime_error("fixture load failed"); }
                event("load:" + std::to_string(++next) + ':' + std::to_string(slot)); return next;
            }, [&](auto id) { event("destroy:" + std::to_string(id)); }, [&] { return keep; });
            std::map<std::string, Pool::Handle> held;
            const auto commands = split(unhex(c[3]), ';'), expected = split(unhex(c[4]), '\n');
            expect(commands.size() == expected.size(), "oracle trace length");
            for (size_t i = 0; i < commands.size(); ++i) {
                const auto action = split(commands[i], ':'); std::string result = "-";
                if (action[0] == "W") pool.warm();
                else if (action[0] == "D") pool.unload_idle();
                else if (action[0] == "K") keep = action[1] == "1";
                else if (action[0] == "F") fail_at = attempt + std::stoi(action[1]);
                else if (action[0] == "A") { try { held[action[1]] = pool.acquire(); result = std::to_string(held[action[1]]); } catch (const std::runtime_error &) { result = "error"; } }
                else if (action[0] == "R") { pool.release(held.at(action[1])); held.erase(action[1]); }
                else throw std::runtime_error("Unknown pool action");
                const auto state = pool.snapshot();
                const auto actual = result + '|' + std::to_string(state.acquired) + '|' + ids(state.resident) + '|' + ids(state.transient) + '|' + ids(state.available) + '|' + (events.empty() ? "-" : events);
                expect(actual == expected[i], "pool case " + std::to_string(count) + " step " + std::to_string(i) + " " + commands[i] + ": " + actual + " != " + expected[i]);
                events.clear(); ++steps;
            }
        }
    }
    expect(count == 569 && steps == 4167, "complete pool oracle coverage"); return count;
}
void waiting_cases() {
    std::atomic<unsigned> loads{0}, destroyed{0};
    Pool pool(1, [&](size_t) { return ++loads; }, [&](auto) { ++destroyed; }, [] { return true; });
    const auto first = pool.acquire();
    auto second = std::async(std::launch::async, [&] { return pool.acquire(); });
    wait_until([&] { return pool.snapshot().waiting == 1; }, "second request waits for a permit");
    auto third = std::async(std::launch::async, [&] { return pool.acquire(); });
    wait_until([&] { return pool.snapshot().waiting == 2; }, "third request queues behind second");
    expect(!pool.unload_idle(), "active pool cannot be unloaded");
    pool.release(first);
    expect(second.wait_for(std::chrono::seconds(2)) == std::future_status::ready, "oldest waiter wakes");
    const auto next = second.get();
    expect(third.wait_for(std::chrono::milliseconds(25)) == std::future_status::timeout && next == first && loads == 1, "FIFO waiter receives existing resident");
    pool.release(next);
    expect(third.wait_for(std::chrono::seconds(2)) == std::future_status::ready, "next waiter wakes");
    pool.release(third.get()); expect(pool.unload_idle() && destroyed == 1, "one resident destroyed after all leases return");
}
void residency_notification_case() {
    bool keep = true; Pool::Handle next = 0; std::vector<std::string> events;
    Pool pool(2, [&](size_t) { return ++next; }, [&](auto handle) { events.push_back("destroy:" + std::to_string(handle)); }, [&] { return keep; },
        [&](const Pool::Snapshot &state) { events.push_back("register:" + ids(state.resident) + '/' + ids(state.transient)); });
    pool.warm(); const auto resident = pool.acquire(); pool.release(resident);
    keep = false; const auto transient = pool.acquire(); pool.release(transient);
    expect(pool.unload_idle(), "mixed resident/transient pool can unload");
    expect(events == std::vector<std::string>({"register:1/-", "register:1,2/-", "register:1,2/3", "register:1,2/-", "destroy:3", "register:-/-", "destroy:1", "destroy:2"}), "pool publishes aggregate residency in source order, before destruction");
}
int free_ports(int count) {
    for (int attempt = 0; attempt < 100; ++attempt) {
        std::vector<int> sockets; int base = 0; bool ok = true;
        for (int i = 0; i < count; ++i) {
            const int fd = socket(AF_INET, SOCK_STREAM, 0); sockets.push_back(fd);
            sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = htons(i ? base + i : 0);
            if (fd < 0 || bind(fd, reinterpret_cast<sockaddr *>(&a), sizeof(a))) { ok = false; break; }
            socklen_t size = sizeof(a); getsockname(fd, reinterpret_cast<sockaddr *>(&a), &size); if (!i) base = ntohs(a.sin_port);
        }
        for (const auto fd : sockets) if (fd >= 0) close(fd);
        if (ok && base + count <= 65535) return base;
    }
    throw std::runtime_error("No contiguous test ports");
}
fs::path configure(const std::string &name, int port, int capacity = 2) {
    environment(); const auto path = fs::absolute(name); fs::create_directories(path);
    setenv("EMBEDDING_FIXTURE_DIR", path.c_str(), 1); setenv("LLAMA_CPP_SERVER_BIN", TEST_EMBEDDING_FIXTURE_BIN, 1);
    std::ofstream(path / "model.gguf") << "GGUF";
    setenv("LLAMA_CPP_EMBEDDING_MODEL_PATH", (path / "model.gguf").c_str(), 1);
    setenv("LLAMA_CPP_EMBEDDING_SERVER_PORT", std::to_string(port).c_str(), 1); setenv("EMBEDDING_N_PARALLEL", std::to_string(capacity).c_str(), 1);
    setenv("EMBEDDING_GPU_LAYERS", "0", 1); setenv("LLAMA_CPP_START_TIMEOUT", "1", 1); return path;
}
int latest_pid(const fs::path &directory, int port) { std::istringstream lines(read(directory / ("launch-" + std::to_string(port)))); int value = 0, next; while (lines >> next) value = next; return value; }
void managed_cases() {
    const int port = free_ports(3); const auto directory = configure("resident", port);
    {
        auto backend = std::make_unique<ezlocalai::LlamaCppInferenceBackend>("none"); auto *tracked = backend.get();
        backend->warm_aux_model("embedding");
        expect(backend->chat("{\"messages\":[]}").find("No llama.cpp server is configured") != std::string::npos, "embedding replicas cannot serve text requests");
        const int first_pid = latest_pid(directory, port), second_pid = latest_pid(directory, port + 1);
        expect(first_pid > 0 && second_pid > 0 && first_pid != second_pid, "two independent full-context processes warm-load");
        for (int i = 0; i < 2; ++i) {
            const auto args = read(directory / ("args-" + std::to_string(port + i)));
            expect(args.find("-np\n1\n") != std::string::npos && args.find("-c\n10000\n") != std::string::npos, "each replica retains its full context and one slot");
        }
        ezl_env_config config{}; ezl_load_config(&config); ezlocalai::NativeServer server(config, std::move(backend));
        auto request = [&](const char *input) { return server.handle_request("POST", "/v1/embeddings", "", std::string("{\"input\":\"") + input + "\"}"); };
        std::future<ezlocalai::HttpResponse> a, b, c;
        struct ReleaseBlocked {
            fs::path directory; int port;
            ~ReleaseBlocked() { for (int i = 0; i < 2; ++i) std::ofstream(directory / ("release-" + std::to_string(port + i))) << "release"; }
        } release_blocked{directory, port};
        a = std::async(std::launch::async, [&] { return request("block-a"); });
        wait_until([&] { return fs::exists(directory / ("seen-" + std::to_string(port))); }, "first replica receives first request");
        b = std::async(std::launch::async, [&] { return request("block-b"); });
        wait_until([&] { return fs::exists(directory / ("seen-" + std::to_string(port + 1))); }, "second replica runs concurrently");
        c = std::async(std::launch::async, [&] { return request("queued"); });
        if (c.wait_for(std::chrono::milliseconds(40)) != std::future_status::timeout) {
            const auto early = c.get(); throw std::runtime_error("third request returned before an instance was released: " + std::to_string(early.status) + " " + early.body);
        }
        const auto resources = server.handle_request("GET", "/v1/resources", "", "");
        expect(resources.body.find("\"embedding\":{\"configured\":2,\"devices\":[\"cpu\",\"cpu\"],\"resident\":2}") != std::string::npos, "resource pool exposes both resident embedding devices");
        expect(resources.body.find("\"embedding\":{\"capacity\":2,\"in_flight\":2,\"queued\":0,\"available\":0") != std::string::npos, "resource slots match active engines, not waiters");
        expect(tracked->loaded_models_json().find("\"active_requests\":2") != std::string::npos && !tracked->unload_aux_model("embedding"), "aggregate busy metadata protects the entire pool");
        expect(tracked->resource_manager()->get_model_active_count(ezlocalai::ModelType::Embedding) == 2, "shared manager counts both in-flight replicas");
        std::ofstream(directory / ("release-" + std::to_string(port))) << "release";
        expect(a.get().status == 200, "first embedding finishes");
        expect(c.wait_for(std::chrono::seconds(3)) == std::future_status::ready, "queued embedding uses released instance");
        expect(c.get().body.find(std::to_string(port)) != std::string::npos, "released FIFO instance reused");
        std::ofstream(directory / ("release-" + std::to_string(port + 1))) << "release";
        expect(b.get().status == 200, "second embedding finishes");
        expect(tracked->resource_manager()->get_model_active_count(ezlocalai::ModelType::Embedding) == 0, "shared activity returns to idle after all requests");
        expect(tracked->unload_aux_model("embedding") && tracked->loaded_models_json() == "{}", "all resident instances detach");
        expect(kill(first_pid, 0) != 0 && kill(second_pid, 0) != 0, "all owned pool processes terminate");
        expect(!tracked->resource_manager()->is_model_loaded(ezlocalai::ModelType::Embedding), "full-pool destruction unregisters aggregate");
        tracked->warm_aux_model("embedding");
        expect(latest_pid(directory, port) != first_pid && latest_pid(directory, port + 1) != second_pid, "video restoration refills the pool");
    }
    const int transient_port = free_ports(2); const auto transient = configure("transient", transient_port);
    setenv("EMBEDDING_KEEP_LOADED", "false", 1);
    {
        ezlocalai::LlamaCppInferenceBackend backend("none");
        backend.embedding("{\"input\":\"one\"}"); const int first = latest_pid(transient, transient_port);
        expect(first > 0 && kill(first, 0) != 0 && backend.loaded_models_json() == "{}", "transient engine destroyed after success");
        expect(backend.embedding("{\"input\":\"fixture-error\"}").find("fixture inference failed") != std::string::npos, "inference error retained");
        const int second = latest_pid(transient, transient_port);
        expect(second != first && kill(second, 0) != 0 && backend.loaded_models_json() == "{}", "new transient engine destroyed after error");
    }
    const int failed_port = free_ports(2); const auto failed = configure("failed-warm", failed_port);
    setenv("EMBEDDING_FIXTURE_FAIL_PORT", std::to_string(failed_port + 1).c_str(), 1);
    {
        ezlocalai::LlamaCppInferenceBackend backend("none"); backend.warm_aux_model("embedding");
        const int first = latest_pid(failed, failed_port);
        expect(first > 0 && kill(first, 0) == 0, "partial warm failure retains successful instances");
        unsetenv("EMBEDDING_FIXTURE_FAIL_PORT"); backend.warm_aux_model("embedding");
        expect(latest_pid(failed, failed_port) == first && kill(latest_pid(failed, failed_port + 1), 0) == 0, "warm retry loads only missing instances");
    }
}
bool healthy(int port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0); if (fd < 0) return false;
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = htons(port);
    if (connect(fd, reinterpret_cast<sockaddr *>(&a), sizeof(a))) { close(fd); return false; }
    timeval timeout{1, 0}; setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    const std::string request = "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    send(fd, request.data(), request.size(), MSG_NOSIGNAL); char bytes[1024]; const auto n = recv(fd, bytes, sizeof(bytes), 0); close(fd);
    return n > 0 && std::string(bytes, n).find("200 OK") != std::string::npos;
}
void startup_cases() {
    for (const std::string mode : {"resident", "transient", "delegated", "disabled"}) {
        const int embedding_port = free_ports(2), api_port = free_ports(1); const auto directory = configure("startup-" + mode, embedding_port);
        if (mode == "transient") setenv("EMBEDDING_KEEP_LOADED", "false", 1);
        if (mode == "delegated") setenv("EMBEDDING_SERVER", "http://worker", 1);
        if (mode == "disabled") setenv("EMBEDDING_ENABLED", "false", 1);
        const auto port_text = std::to_string(api_port); const pid_t pid = fork(); expect(pid >= 0, "fork native service");
        if (!pid) {
            const int log = open((directory / "server.log").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600); dup2(log, STDOUT_FILENO); dup2(log, STDERR_FILENO); close(log);
            execl(TEST_NATIVE_SERVER_BIN, TEST_NATIVE_SERVER_BIN, "--host", "127.0.0.1", "--port", port_text.c_str(), nullptr); _exit(127);
        }
        struct Cleanup { pid_t pid; ~Cleanup() { kill(pid, SIGTERM); int status; for (int i = 0; i < 100; ++i) { if (waitpid(pid, &status, WNOHANG) == pid) return; std::this_thread::sleep_for(std::chrono::milliseconds(20)); } kill(pid, SIGKILL); waitpid(pid, &status, 0); } } cleanup{pid};
        wait_until([&] { return healthy(api_port); }, "native startup health: " + mode);
        expect((latest_pid(directory, embedding_port) > 0 && latest_pid(directory, embedding_port + 1) > 0) == (mode == "resident"), "startup honors embedding preload policy: " + mode);
    }
}
}
int main(int argc, char **argv) {
    try {
        expect(argc == 3, "reference and fixture required"); const auto root = fs::absolute(argv[1]), fixture = fs::absolute(argv[2]);
        const auto previous = fs::current_path(); char directory[] = "/tmp/ezlocalai-embedding-runtime-XXXXXX"; expect(mkdtemp(directory), "temporary directory"); fs::current_path(directory);
        const auto count = reference_cases(root, fixture); waiting_cases(); residency_notification_case(); managed_cases(); startup_cases();
        fs::current_path(previous); fs::remove_all(directory);
        std::cout << "Verified " << count << " embedding runtime cases (4167 lifecycle steps), FIFO waiting, managed pools and startup\n"; return 0;
    } catch (const std::exception &error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
