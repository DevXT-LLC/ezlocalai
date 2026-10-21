#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include <nlohmann/json.hpp>
#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <condition_variable>
#include <fstream>
#include <iostream>

using Json = nlohmann::json;
using namespace ezlocalai;
void check(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
struct Capture {
    int fd = -1, port = 0;
    std::atomic<bool> stopping{false};
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<std::string> wires;
    std::vector<double> times;
    std::vector<int> held;
    std::string error;
    std::thread thread;
    Json test;
    explicit Capture(Json row) : test(std::move(row)) {
        fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0); check(fd >= 0, "socket");
        sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        check(bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0 && listen(fd, 16) == 0, "listen");
        socklen_t size = sizeof(addr); getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &size); port = ntohs(addr.sin_port);
        thread = std::thread([this] {
            try {
                while (!stopping) {
                    pollfd ready{fd, POLLIN, 0}; if (poll(&ready, 1, 20) <= 0) continue;
                    const int peer = accept(fd, nullptr, nullptr); check(peer >= 0, "accept");
                    struct Close { int fd; ~Close() { if (fd >= 0) close(fd); } } close_peer{peer};
                    timeval timeout{3, 0}; setsockopt(peer, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
                    std::string wire; char buffer[4096];
                    for (;;) {
                        const auto n = recv(peer, buffer, sizeof(buffer), 0); check(n > 0, "request receive"); wire.append(buffer, n);
                        const auto end = wire.find("\r\n\r\n"); if (end == std::string::npos) continue;
                        const auto marker = wire.find("Content-Length: ");
                        const auto length = marker == std::string::npos ? 0 : std::stoul(wire.substr(marker + 16));
                        if (wire.size() >= end + 4 + length) break;
                    }
                    size_t index;
                    {
                        std::lock_guard<std::mutex> lock(mutex); index = wires.size(); wires.push_back(wire);
                        times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()); changed.notify_all();
                    }
                    std::string response = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}";
                    bool hold = false;
                    if (test.contains("shutdown")) {
                        const auto action = test.at("shutdown").get<std::string>();
                        if (wire.find(" /v1/router/" + action + " ") != std::string::npos) {
                            if (test.value("release", false))
                                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                            else {
                                response = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n";
                                hold = true;
                            }
                        }
                    } else if (test.contains("workflow")) {
                        if (index == 1 || index == 2) response = "HTTP/1.1 503 Failure\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                    } else if (index < test.at("requests").size() && wire.find(" /land ") == std::string::npos) {
                        response.clear(); const auto hex = test.at("response_hex").get<std::string>();
                        for (size_t i = 0; i < hex.size(); i += 2) response += static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16));
                        hold = test.at("hold_body");
                    }
                    size_t sent = 0;
                    while (sent < response.size()) { const auto n = send(peer, response.data() + sent, response.size() - sent, MSG_NOSIGNAL); if (n <= 0) break; sent += n; }
                    if (hold) { held.push_back(peer); close_peer.fd = -1; }
                }
            } catch (const std::exception &e) { std::lock_guard<std::mutex> lock(mutex); error = e.what(); changed.notify_all(); }
        });
    }
    ~Capture() { stopping = true; if (thread.joinable()) thread.join(); for (const int peer : held) close(peer); if (fd >= 0) close(fd); }
    std::vector<std::string> wait(size_t count) {
        std::unique_lock<std::mutex> lock(mutex);
        check(changed.wait_for(lock, std::chrono::seconds(8), [&] { return wires.size() >= count || !error.empty(); }), "heartbeat request timeout");
        check(error.empty(), error); return wires;
    }
};
void environment(const Capture &capture) {
    for (const auto &entry : environment_defaults) unsetenv(std::string(entry.first).c_str());
    for (const auto &[key, value] : std::map<std::string, std::string>{{"EZLOCALAI_DOTENV_PATH", ""}, {"DEFAULT_MODEL", "fixture"}, {"EZLOCALAI_API_KEY", "worker-key"}, {"ROUTER_API_KEY", "fixture-key"}, {"WORKER_HEARTBEAT_INTERVAL", "2"}, {"WORKER_HEARTBEAT_TIMEOUT", "1"}, {"WORKER_LABEL", "heartbeat"}, {"WORKER_TUNNEL", "false"}, {"ROUTER_MODE", "false"}, {"N_PARALLEL", "1"}, {"TTS_ENABLED", "false"}, {"STT_ENABLED", "false"}, {"EMBEDDING_ENABLED", "false"}, {"IMAGE_ENABLED", "false"}, {"VIDEO_ENABLED", "false"}, {"MUSIC_ENABLED", "false"}}) setenv(key.c_str(), value.c_str(), 1);
    setenv("ROUTER_URL", ("http://127.0.0.1:" + std::to_string(capture.port)).c_str(), 1);
}
std::shared_ptr<ResourceManager> resources() {
    auto probes = native_resource_probes(); probes.gpu_count = [] { return 0; };
    probes.total_per_gpu = probes.free_per_gpu = [] { return std::vector<double>{}; };
    probes.free_vram = [](auto) { return 0.; }; probes.allocator_stats = [] { return std::vector<CudaAllocatorStats>{}; };
    probes.system_ram = probes.available_ram = [] { return 16.; }; return std::make_shared<ResourceManager>(probes);
}
int main(int argc, char **argv) {
    try {
        check(argc == 2 || argc == 3, "reference fixture required");
        if (argc == 3) {
            std::ifstream input(argv[2]); const auto fixture = Json::parse(input);
            for (const auto &test : fixture.at("cases")) {
                Capture capture({{"shutdown",test.at("action")},{"release",test.at("release")}});
                environment(capture); setenv("WORKER_HEARTBEAT_TIMEOUT","10",1);
                ezl_env_config config{}; ezl_load_config(&config);
                NativeServer server(config,std::make_unique<EchoInferenceBackend>("fixture"),resources());
                std::string error; check(server.start("127.0.0.1",0,&error),error);
                capture.wait(test.at("action")=="register" ? 1 : 2);
                const auto started=std::chrono::steady_clock::now();
                server.stop();
                const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
                check(seconds>=test.at("minimum_seconds").get<double>() &&
                    seconds<=test.at("maximum_seconds").get<double>(),
                    test.at("name").get<std::string>()+": shutdown timing "+std::to_string(seconds));
                std::vector<std::string> actions;
                { std::lock_guard<std::mutex> lock(capture.mutex);
                  for(const auto &wire:capture.wires) {
                    const auto start=wire.find("/v1/router/")+11;
                    actions.push_back(wire.substr(start,wire.find(' ',start)-start));
                  }
                }
                check(Json(actions)==test.at("actions"),"shutdown registration/deregistration order");
                std::cout<<test.at("name")<<" passed in "<<seconds<<" seconds\n"<<std::flush;
            }
            return 0;
        }
        std::ifstream input(argv[1]); const auto fixture = Json::parse(input);
        for (const auto &test : fixture.at("cases")) {
            Capture capture(test); environment(capture); ezl_env_config config{}; ezl_load_config(&config);
            NativeServer server(config, std::make_unique<EchoInferenceBackend>("fixture"), resources());
            std::string error; check(server.start("127.0.0.1", 0, &error), error);
            const auto count = test.at("requests").size(); const auto wires = capture.wait(count + 1);
            for (size_t i = 0; i < count; ++i) check(wires[i].substr(0, wires[i].find("\r\n")) == test.at("requests").at(i), test.at("name").get<std::string>() + ": redirect/request trace");
            check(wires.front().find("Authorization: Bearer fixture-key\r\n") != std::string::npos, "router authentication");
            check(wires.front().find("User-Agent: " + test.at("user_agent").get<std::string>() + "\r\n") != std::string::npos, "source client headers");
            const auto expected = test.at("ok") == true ? "POST /v1/router/heartbeat " : "POST /v1/router/register ";
            check(wires[count].rfind(expected, 0) == 0, test.at("name").get<std::string>() + ": response acceptance differs");
            server.stop(); std::cout << test.at("name") << " passed\n" << std::flush;
        }
        Capture capture({{"workflow", true}}); environment(capture); ezl_env_config config{}; ezl_load_config(&config);
        NativeServer server(config, std::make_unique<EchoInferenceBackend>("fixture"), resources());
        std::string error; check(server.start("127.0.0.1", 0, &error), error); capture.wait(1);
        setenv("WORKER_HEARTBEAT_INTERVAL", "30", 1);
        auto wires = capture.wait(6); server.stop(); wires = capture.wait(7);
        const std::vector<std::string> sequence{"register", "heartbeat", "register", "heartbeat", "register", "heartbeat", "deregister"};
        std::string worker_id;
        for (size_t i = 0; i < sequence.size(); ++i) {
            check(wires[i].rfind("POST /v1/router/" + sequence[i] + " ", 0) == 0, "steady recovery ordering");
            const auto body = Json::parse(wires[i].substr(wires[i].find("\r\n\r\n") + 4));
            if (worker_id.empty()) worker_id = body.at("worker_id");
            check(body.at("worker_id") == worker_id, "worker ID changed during recovery");
        }
        check(capture.times[3] - capture.times[2] >= 1.7 && capture.times[5] - capture.times[4] >= 1.7, "recovery skips captured heartbeat interval");
        std::cout << "live failure recovery, captured interval, stable identity and deregistration passed\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
