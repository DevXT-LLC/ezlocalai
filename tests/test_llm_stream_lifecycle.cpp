#include "ezlocalai/llm_stream.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include <arpa/inet.h>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>
#include <sstream>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace std::chrono_literals;
namespace {
void expect(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::string read(const fs::path &path) { std::ifstream input(path); return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()}; }
std::string hash(const std::string &value) { unsigned char d[32]; SHA256(reinterpret_cast<const unsigned char *>(value.data()), value.size(), d); std::ostringstream out; for (auto c : d) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c); return out.str(); }
template<class Predicate> void wait(Predicate predicate, const std::string &message, int milliseconds = 2500) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(milliseconds);
    while (!predicate()) { expect(std::chrono::steady_clock::now() < until, message); std::this_thread::sleep_for(5ms); }
}
struct Gate {
    std::mutex mutex; std::condition_variable cv; bool open = false;
    void release() { std::lock_guard<std::mutex> lock(mutex); open = true; cv.notify_all(); }
    void await() { std::unique_lock<std::mutex> lock(mutex); cv.wait(lock, [&] { return open; }); }
};
struct Counters { Gate gate; std::atomic<int> started{0}, released{0}; std::atomic<bool> stopped{false}, done{false}; };

void reference_cases(const fs::path &root, const fs::path &fixture) {
    std::istringstream input(read(fixture)); size_t cases = 0;
    for (std::string line; std::getline(input, line);) {
        if (line.empty()) continue;
        if (line.front() == '#') { const auto at = line.find(" sha256="); if (at != std::string::npos) expect(hash(read(root / line.substr(2, at - 2))) == line.substr(at + 8), "stream lifecycle source changed"); continue; }
        const auto tab = line.find('\t'); const auto kind = line.substr(0, tab), expected = line.substr(tab + 1); ++cases;
        auto counters = std::make_shared<Counters>();
        ezlocalai::LlmInferenceStream stream("ordinary", "none", "2026.10.11549", kind != "cancel_many",
            [counters, kind](const auto &emit, const auto &interrupt) {
                interrupt([counters] { counters->gate.release(); }); ++counters->started;
                if (kind == "keepalive") counters->gate.await();
                const bool first = emit(R"({"content":"first"})");
                if (kind.rfind("cancel", 0) == 0) { counters->gate.await(); counters->stopped = emit(R"({"content":"after"})"); }
                else if (!first) emit(R"({"choices":[{"delta":{},"finish_reason":"stop"}]})");
                counters->done = true;
            }, [counters] { ++counters->released; });
        struct Unblock { std::shared_ptr<Counters> c; ~Unblock() { c->gate.release(); } } unblock{counters};
        std::string trace = "created:" + std::to_string(counters->started) + ':' + std::to_string(counters->released), chunk;
        if (kind == "unstarted") {
            stream.close(); stream.close(); trace += ",closed:" + std::to_string(counters->started) + ':' + std::to_string(counters->released);
        } else if (kind == "complete") {
            expect(stream.next(&chunk) && chunk.find("first") != std::string::npos, "first generated value"); wait([&] { return counters->done.load(); }, "native completed");
            trace += ",first:" + std::to_string(counters->released);
            expect(stream.next(&chunk) && chunk.find("\"finish_reason\": \"stop\"") != std::string::npos, "held final value");
            trace += ",final:" + std::to_string(counters->released) + ":stop";
            expect(!stream.next(&chunk), "generator exhausted"); stream.close(); trace += ",exhausted:" + std::to_string(counters->released);
        } else if (kind.rfind("cancel", 0) == 0) {
            expect(stream.next(&chunk), "generation started before cancellation"); const auto start = std::chrono::steady_clock::now(); stream.close();
            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            expect(elapsed >= 4.8 && elapsed < 6.5, "source five-second close bound");
            trace += ",bounded_close:" + std::to_string(counters->released); counters->gate.release();
            wait([&] { return counters->released == 1 && counters->stopped; }, "native cancellation callback and release");
            trace += ",native_done:" + std::to_string(counters->released);
        } else {
            const auto start = std::chrono::steady_clock::now(); expect(stream.next(&chunk), "keepalive value");
            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            expect(elapsed >= 4.8 && elapsed < 6.5 && chunk.find("\"delta\": {}") != std::string::npos && chunk.find("\"finish_reason\": null") != std::string::npos, "source five-second keepalive");
            trace += ",keepalive:" + std::to_string(counters->released); counters->gate.release(); int remaining = 0;
            while (stream.next(&chunk)) ++remaining;
            trace += ",completed:" + std::to_string(counters->released) + ':' + std::to_string(remaining);
        }
        expect(trace == expected, kind + ": " + trace + " expected " + expected);
    }
    expect(cases == 5, "all source lifecycle scenarios");
}

bool send_all(int fd, const std::string &value) { for (size_t at = 0; at < value.size();) { const auto n = send(fd, value.data() + at, value.size() - at, MSG_NOSIGNAL); if (n <= 0) return false; at += n; } return true; }
int listen_socket(int *port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0); sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    expect(!bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)), "listener bind"); socklen_t size = sizeof(address); getsockname(fd, reinterpret_cast<sockaddr *>(&address), &size); *port = ntohs(address.sin_port);
    expect(!listen(fd, 16), "listener start"); return fd;
}
struct Upstream {
    int fd, port; Gate headers, tail; bool cancellation; std::atomic<bool> stopping{false}, finished{false}, peer_closed{false}; std::atomic<int> calls{0};
    std::thread accept; std::vector<std::thread> workers; std::mutex mutex; std::set<int> peers; std::string payload;
    explicit Upstream(bool cancel = false) : cancellation(cancel) {
        fd = listen_socket(&port);
        accept = std::thread([this] {
            while (!stopping) { const int peer = ::accept(fd, nullptr, nullptr); if (peer < 0) break;
                { std::lock_guard<std::mutex> lock(mutex); peers.insert(peer); }
                workers.emplace_back([this, peer] { serve(peer); std::lock_guard<std::mutex> lock(mutex); peers.erase(peer); close(peer); });
            }
        });
    }
    ~Upstream() {
        stopping = true; headers.release(); tail.release(); shutdown(fd, SHUT_RDWR); close(fd); accept.join();
        { std::lock_guard<std::mutex> lock(mutex); for (const auto peer : peers) shutdown(peer, SHUT_RDWR); }
        for (auto &worker : workers) worker.join();
    }
    std::string url() const { return "http://127.0.0.1:" + std::to_string(port); }
    void serve(int peer) {
        std::string request; char buffer[4096]; size_t end;
        while ((end = request.find("\r\n\r\n")) == std::string::npos) { const auto n = recv(peer, buffer, sizeof(buffer), 0); if (n <= 0) return; request.append(buffer, n); }
        const auto length_at = request.find("Content-Length:"); const auto length = length_at == std::string::npos ? 0 : std::stoul(request.substr(length_at + 15));
        while (request.size() < end + 4 + length) { const auto n = recv(peer, buffer, sizeof(buffer), 0); if (n <= 0) return; request.append(buffer, n); }
        if (request.rfind("POST /v1/chat/completions", 0) != 0) { send_all(peer, "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}"); return; }
        { std::lock_guard<std::mutex> lock(mutex); payload = request.substr(end + 4, length); }
        const int index = calls++; if (!index) headers.await();
        if (!send_all(peer, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n")) return;
        const auto chunk = [&](const std::string &value) {
            std::ostringstream size; size << std::hex << value.size();
            // Deliberately fragment HTTP framing and SSE records independently.
            const auto wire = size.str() + "\r\n" + value + "\r\n";
            for (size_t at = 0; at < wire.size(); at += 3) if (!send_all(peer, wire.substr(at, 3))) return false;
            return true;
        };
        if (!chunk("data: {\"content\":\"hel") || !chunk("lo\"}\n\n")) return;
        if (!index) tail.await();
        if (cancellation && !index) {
            chunk("data: {\"content\":\"cancel callback\"}\n\n");
            timeval timeout{2, 0}; setsockopt(peer, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            peer_closed = recv(peer, buffer, sizeof(buffer), 0) == 0;
        } else chunk("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n");
        send_all(peer, "0\r\n\r\n"); finished = true;
    }
};
void environment(const Upstream &upstream) {
    for (const auto &setting : ezlocalai::environment_defaults) unsetenv(setting.first.data());
    for (const auto *key : {"LLAMA_CPP_CONTEXT", "LLAMA_CPP_MODEL_PATH", "LLAMA_CPP_MODEL_PATHS", "LLAMA_CPP_EXTRA_ARGS"}) unsetenv(key);
    setenv("EZLOCALAI_DOTENV_PATH", "", 1); setenv("EZLOCALAI_API_KEY", "none", 1); setenv("DEFAULT_MODEL", "ordinary", 1);
    setenv("LLAMA_CPP_SERVER_URL", upstream.url().c_str(), 1); setenv("N_PARALLEL", "1", 1);
}
ezlocalai::HttpResponse lazy(ezlocalai::NativeServer &server) { return server.handle_request("POST", "/v1/chat/completions", "", R"({"messages":[{"role":"user","content":""}],"stream":true,"disable_fallback":true})", "application/json", "", "", {}, true); }
bool busy(ezlocalai::NativeServer &server, int count, int capacity = 1) { return server.handle_request("GET", "/v1/resources", "", "").body.find("\"ordinary\":{\"capacity\":" + std::to_string(capacity) + ",\"in_flight\":" + std::to_string(count)) != std::string::npos; }
int connect_client(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0); sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK); address.sin_port = htons(port);
    expect(!connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)), "native API connect"); timeval timeout{8, 0}; setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    const std::string body = R"({"messages":[{"role":"user","content":""}],"stream":true,"disable_fallback":true})";
    expect(send_all(fd, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body), "send API request"); return fd;
}
void receive_until(int fd, std::string *wire, const std::string &needle) {
    char buffer[4096]; while (wire->find(needle) == std::string::npos) { const auto n = recv(fd, buffer, sizeof(buffer), 0); expect(n > 0, "waiting for API stream " + needle + ": " + *wire); wire->append(buffer, n); }
}
void network_cases(const fs::path &root, const fs::path &fixture) {
    std::map<int, bool> releases_before_callback;
    std::istringstream input(read(fixture));
    for (std::string line; std::getline(input, line);) {
        if (line.empty()) continue;
        if (line.front() == '#') { const auto at = line.find(" sha256="); if (at != std::string::npos) expect(hash(read(root / line.substr(2, at - 2))) == line.substr(at + 8), "stream ASGI source changed"); continue; }
        const auto at = line.find('\t'); releases_before_callback[std::stoi(line.substr(0, at))] = line.substr(at + 1) == "1";
    }
    expect(releases_before_callback.size() == 2, "both actual ASGI concurrency scenarios");
    for (int scenario : {0, 1, 2}) {
        const bool cancel = scenario != 0; const int capacity = scenario ? scenario : 1;
        Upstream upstream(cancel); environment(upstream); if (cancel) upstream.headers.release();
        setenv("N_PARALLEL", std::to_string(capacity).c_str(), 1);
        ezl_env_config config{}; ezl_load_config(&config); config.max_concurrent_requests = 4; config.request_timeout_ms = 15000;
        ezlocalai::NativeServer server(config, std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model));
        // A discarded lazy response never invokes native generation and still
        // releases the lease held while the generator was unstarted.
        if (capacity == 1) {
            auto held = lazy(server); expect(held.stream_writer && upstream.calls == 0 && busy(server, 1), "unstarted response holds replica without generating");
            auto waiting = std::async(std::launch::async, [&] { return lazy(server); });
            struct ReleaseHeld { ezlocalai::HttpResponse &response; ~ReleaseHeld() { response = {}; } } release_held{held};
            expect(waiting.wait_for(100ms) == std::future_status::timeout, "second request waits for unstarted stream lease");
            held = {}; auto next = waiting.get(); expect(upstream.calls == 0, "second lazy stream also unstarted"); next = {}; expect(busy(server, 0), "discarded responses release slots exactly once");
        }
        int port; const int temporary = listen_socket(&port); close(temporary); std::string error;
        expect(server.start("127.0.0.1", port, &error), error); const auto start = std::chrono::steady_clock::now();
        int client = connect_client(port); std::string wire; receive_until(client, &wire, "\r\n\r\n");
        expect(wire.rfind("HTTP/1.1 200", 0) == 0 && std::chrono::steady_clock::now() - start < 2s, "API headers precede native prompt processing");
        wait([&] { return upstream.calls == 1; }, "native generation started");
        if (!cancel) {
            receive_until(client, &wire, "\"delta\": {}"); const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            expect(elapsed >= 4.8 && elapsed < 7 && busy(server, 1), "keepalive while upstream has not sent headers"); upstream.headers.release();
        }
        receive_until(client, &wire, "hello"); expect(!upstream.finished && busy(server, 1, capacity), "first token is delivered before native completion");
        if (cancel) {
            std::this_thread::sleep_for(200ms); // Let the ASGI-style next() begin.
            shutdown(client, SHUT_RDWR); close(client); client = -1;
            std::this_thread::sleep_for(5300ms); expect(busy(server, 1, capacity) && !upstream.finished, "disconnect waits for pending next() before its bounded close");
            std::this_thread::sleep_for(5000ms);
            expect(busy(server, releases_before_callback.at(capacity) ? 0 : 1, capacity), "actual ASGI single/multiple-slot release timing");
            upstream.tail.release(); wait([&] { return upstream.peer_closed.load(); }, "cancellation closes upstream at next native callback");
        } else {
            upstream.tail.release(); receive_until(client, &wire, "data: [DONE]");
            expect(wire.find("empty_stream") == std::string::npos && wire.find("\"finish_reason\": \"stop\"") != std::string::npos, "stream final value and DONE"); close(client);
        }
        wait([&] { return busy(server, 0, capacity); }, "finished native stream releases replica");
        { std::lock_guard<std::mutex> lock(upstream.mutex); expect(upstream.payload.find("\"stream\":true") != std::string::npos, "native stream payload"); }
        server.stop();
    }
}

void owner_cases() {
    auto counters = std::make_shared<Counters>();
    auto stream = std::make_shared<ezlocalai::LlmInferenceStream>("ordinary", "none", "unknown", true,
        [counters](const auto &emit, const auto &interrupt) {
            interrupt([counters] { counters->gate.release(); }); ++counters->started;
            emit(R"({"content":"first"})"); counters->gate.await(); counters->stopped = emit(R"({"content":"cancelled"})");
        }, [counters] { ++counters->released; });
    const auto owner = stream->owner_handle(); std::string chunk;
    expect(stream->next(&chunk), "owned stream first chunk"); stream.reset();
    expect(counters->released == 0 && !owner.first.expired(), "native lifetime survives discarded generator and bounded close");
    owner.second(); expect(counters->released == 1 && counters->stopped, "owner shutdown interrupts remaining native generation and joins cleanup");
    wait([&] { return owner.first.expired(); }, "native state eventually released");

    std::atomic<int> starts{0}, releases{0};
    ezlocalai::LlmInferenceStream abandoned("ordinary", "none", "unknown", true,
        [&](const auto &, const auto &) { ++starts; }, [&] { ++releases; });
    expect(!abandoned.next(&chunk, [] { return true; }) && starts == 0 && releases == 1, "disconnect before first iteration does not start native inference");

    Upstream upstream; environment(upstream);
    auto backend = std::make_unique<ezlocalai::LlamaCppInferenceBackend>("ordinary");
    auto result = backend->lazy_text_result(R"({"messages":[],"stream":true})", "", true);
    expect(static_cast<bool>(result.stream), "deferred engine response");
    auto consumer = std::async(std::launch::async, [&] { std::string value; return result.stream->next(&value); });
    wait([&] { return upstream.calls == 1; }, "native header wait started");
    const auto start = std::chrono::steady_clock::now(); backend.reset();
    expect(std::chrono::steady_clock::now() - start < 2s && !consumer.get(), "backend destruction aborts pending transport before destroying captured state");
    result.stream.reset();

    ezl_env_config config{}; ezl_load_config(&config);
    auto server = std::make_unique<ezlocalai::NativeServer>(config, std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model));
    auto response = lazy(*server); expect(static_cast<bool>(response.stream_writer), "unstarted API response"); server.reset(); response = {};
    expect(upstream.calls == 1, "response can outlive its owner without starting inference or invoking stale cleanup");
}
}
int main(int argc, char **argv) {
    try { expect(argc == 3, "reference and lifecycle fixture"); reference_cases(argv[1], argv[2]); network_cases(argv[1], fs::path(argv[2]).parent_path() / "llm_stream_asgi_cases.tsv"); owner_cases(); std::cout << "Source lifecycle/ASGI traces, lazy queue ownership, incremental HTTP framing, keepalive, cancellation leases and owner shutdown passed\n"; return 0; }
    catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
