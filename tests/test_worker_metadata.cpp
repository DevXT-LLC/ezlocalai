#include "ezlocalai/worker_models.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/process.hpp"
#include <nlohmann/json.hpp>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <poll.h>
#include <unistd.h>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
using namespace ezlocalai;
using Json = nlohmann::json;
using namespace std::chrono_literals;
void check(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
struct Capture {
    int fd = -1, port = 0;
    std::atomic<bool> stop{false};
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<Json> messages;
    std::string error;
    std::function<void()> received;
    std::thread thread;
    Capture() {
        fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0); check(fd >= 0, "capture socket");
        sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        check(bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0 && listen(fd, 8) == 0, "capture listen");
        socklen_t size = sizeof(address); getsockname(fd, reinterpret_cast<sockaddr *>(&address), &size); port = ntohs(address.sin_port);
        thread = std::thread([this] {
            try { while (!stop) {
                pollfd ready{fd, POLLIN, 0}; if (poll(&ready, 1, 20) <= 0) continue;
                const int client = accept(fd, nullptr, nullptr); check(client >= 0, "capture accept");
                struct Close { int fd; ~Close() { close(fd); } } close_client{client};
                timeval timeout{3, 0}; setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
                std::string wire; char buffer[4096]; size_t end = std::string::npos, length = 0;
                for (;;) {
                    const auto n = recv(client, buffer, sizeof(buffer), 0); check(n > 0, "capture receive"); wire.append(buffer, n);
                    end = wire.find("\r\n\r\n");
                    if (end == std::string::npos) continue;
                    const auto marker = wire.find("Content-Length: ");
                    check(marker != std::string::npos, "capture body length"); length = std::stoul(wire.substr(marker + 16));
                    if (wire.size() >= end + 4 + length) break;
                }
                if (wire.find("/register ") != std::string::npos || wire.find("/heartbeat ") != std::string::npos) {
                    const auto body = Json::parse(wire.substr(end + 4, length));
                    std::lock_guard<std::mutex> lock(mutex); messages.push_back(body); changed.notify_all();
                }
                std::function<void()> callback;
                { std::lock_guard<std::mutex> lock(mutex); callback = received; }
                if (callback) callback();
                const std::string response = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}";
                check(send(client, response.data(), response.size(), MSG_NOSIGNAL) == static_cast<ssize_t>(response.size()), "capture response");
            } } catch (const std::exception &e) { std::lock_guard<std::mutex> lock(mutex); error = e.what(); changed.notify_all(); }
        });
    }
    ~Capture() { stop = true; if (thread.joinable()) thread.join(); if (fd >= 0) close(fd); }
    std::string url() const { return "http://127.0.0.1:" + std::to_string(port); }
    void on_receive(std::function<void()> callback) { std::lock_guard<std::mutex> lock(mutex); received = std::move(callback); }
    void clear() { std::lock_guard<std::mutex> lock(mutex); messages.clear(); }
    std::vector<Json> wait(size_t count) {
        std::unique_lock<std::mutex> lock(mutex);
        check(changed.wait_for(lock, 6s, [&] { return !error.empty() || messages.size() >= count; }), "router update timeout");
        check(error.empty(), error); return messages;
    }
};
struct State {
    std::mutex mutex;
    Json row;
    std::atomic<bool> probe_window{false};
    explicit State(Json value) : row(std::move(value)) {}
    Json get() { std::lock_guard<std::mutex> lock(mutex); return row; }
    void set(Json value) { std::lock_guard<std::mutex> lock(mutex); row = std::move(value); }
};
struct Backend : InferenceBackend {
    std::shared_ptr<State> state;
    explicit Backend(std::shared_ptr<State> state) : state(std::move(state)) {}
    std::string chat(const std::string &) override { return "{}"; }
    std::string completion(const std::string &) override { return "{}"; }
    std::string embedding(const std::string &) override { return "{}"; }
    std::map<std::string,std::string> loaded_model_quants() const override {
        std::map<std::string,std::string> result;const auto paths=state->get().value("paths",Json::object());
        for(const auto &entry:paths.items()){const auto quant=worker_model_file_quant(entry.value());if(!quant.empty())result[entry.key()]=quant;}return result;
    }
    bool loaded_vision_model() const override {return state->get().value("vision_loaded",false);}
    std::string loaded_models_json() const override { return state->get().at("llm_loaded") == true ? R"({"llm":{"name":"org/model","device":"cuda"}})" : "{}"; }
    std::string model_lifecycle_json() const override { auto row = state->get(); state->probe_window = row.value("probe_error", false); return row.at("lifecycle").dump(); }
};
void environment(const Json &row, const Capture &capture) {
    for (const auto &entry : environment_defaults) unsetenv(std::string(entry.first).c_str());
    for (const auto &[key, value] : std::map<std::string, std::string>{
        {"EZLOCALAI_API_KEY", ""}, {"ROUTER_API_KEY", "fixture-key"}, {"DEFAULT_MODEL", "org/model,org/model"},
        {"ROUTER_URL", capture.url()}, {"WORKER_LABEL", "metadata"}, {"WORKER_HEARTBEAT_INTERVAL", "2"},
        {"EZLOCALAI_VERSION", "native-metadata-version"}, {"WORKER_TUNNEL", "false"}, {"TTS_ENABLED", "true"}, {"STT_ENABLED", "true"},
        {"EMBEDDING_ENABLED", "false"}, {"IMAGE_ENABLED", "false"}, {"VIDEO_ENABLED", "false"},
        {"MUSIC_ENABLED", "false"}, {"N_PARALLEL", "1"}, {"CUDA_VISIBLE_DEVICES", ""}})
        setenv(key.c_str(), value.c_str(), 1);
    for (const auto &[key, value] : row.at("environment").items()) setenv(key.c_str(), value.get<std::string>().c_str(), 1);
}
void reporting(const Json &expected,const Json &actual) {
    if(!expected.contains("report"))return;
    for(const auto &field:expected.at("report").items())check(actual.at(field.key())==field.value(),expected.at("name").get<std::string>()+": "+field.key()+" actual="+actual.at(field.key()).dump()+" expected="+field.value().dump());
}
Json expected_extra(const Json &row) {
    // Preserve every captured Python media-policy field exactly. Native gzip
    // request support is an additive transport capability, including when the
    // policy probe failed and the original captured extra object was empty.
    auto extra = row.at("extra");
    extra["http_request_encodings"] = Json::array({"gzip"});
    return extra;
}
void one(const Json &row, Capture &capture, const Json *detached = nullptr, const Json *restored = nullptr) {
    capture.clear(); environment(row, capture);
    auto state = std::make_shared<State>(row);
    // A transient memory failure begins after the ordinary resource snapshot.
    // Receiving the update closes that fault window before the next request.
    capture.on_receive([state] { state->probe_window = false; });
    auto probes = native_resource_probes();
    probes.gpu_count = [] { return 1; }; probes.total_per_gpu = [] { return std::vector<double>{32}; };
    probes.free_per_gpu = [state] { if (state->probe_window.exchange(false)) throw std::runtime_error("fixture memory probe failure"); return std::vector<double>{state->get().at("free_gb").get<double>()}; };
    probes.free_vram = [state](auto) { return state->get().at("free_gb").get<double>(); };
    probes.allocator_stats = [] { return std::vector<CudaAllocatorStats>{}; };
    probes.system_ram = probes.available_ram = [] { return 16.; };
    auto resources = std::make_shared<ResourceManager>(probes);
    if (row.at("image_loaded") == true) resources->register_model(ModelType::Image, "image", "cpu", 0);
    ezl_env_config config{}; ezl_load_config(&config);
    NativeServer server(config, std::make_unique<Backend>(state), resources);
    std::string error; check(server.start("127.0.0.1", 0, &error), error);
    auto messages = capture.wait(2);
    for (size_t i = 0; i < 2; ++i) check(messages[i].at("version") == "native-metadata-version", "registration and heartbeat version");
    for (size_t i = 0; i < 2; ++i) check(messages[i].at("extra") == expected_extra(row), row.at("name").get<std::string>() + ": " + messages[i].at("extra").dump());
    for(size_t i=0;i<2;++i)reporting(row,messages[i]);
    if (detached && restored) {
        // Change runtime state without changing configuration or restarting the worker.
        state->set(*detached); messages = capture.wait(3);
        check(messages.back().at("extra") == expected_extra(*detached), "heartbeat reflects detached LLMs");reporting(*detached,messages.back());
        state->set(*restored); messages = capture.wait(4);
        check(messages.back().at("extra") == expected_extra(*restored), "heartbeat reflects restored public replica identities");reporting(*restored,messages.back());
    }
    server.stop();
}
void gpu_reporting(const std::string &visibility,const Json &expected,const std::string &failure){
    Capture capture;environment(Json{{"environment",Json::object()}},capture);
#ifdef EZLOCALAI_USE_HIP
    unsetenv("HIP_VISIBLE_DEVICES");
    if(visibility=="all")unsetenv("CUDA_VISIBLE_DEVICES");else setenv("CUDA_VISIBLE_DEVICES",visibility.empty()?"-1":visibility.c_str(),1);
#else
    if(visibility=="all")unsetenv("CUDA_VISIBLE_DEVICES");else setenv("CUDA_VISIBLE_DEVICES",visibility.c_str(),1);
#endif
    for(const auto *key:{"TTS_ENABLED","STT_ENABLED","EZLOCALAI_PRELOAD_MODEL_ON_START","EZLOCALAI_MODEL_STATUS_ON_START"})setenv(key,"false",1);
    ezl_env_config config{};ezl_load_config(&config);
    NativeServer server(config,std::make_unique<EchoInferenceBackend>("org/model"));
    // Fail a live properties query after startup, as a transient driver error.
    if(!failure.empty())setenv("EZLOCALAI_TEST_CUDA_FAIL",failure.c_str(),1);
    std::string error;check(server.start("127.0.0.1",0,&error),error);
    const auto messages=capture.wait(2);
    for(const auto &message:messages){
        const auto &gpus=message.at("gpus");
        check(gpus.size()==expected.size()+1,"only CUDA-visible devices plus CPU are reported");
        int best=2;
        for(size_t i=0;i<expected.size();++i){
            if(expected[i].is_object())
                check(gpus[i]==expected[i],"source AMD GPU reporting: "+gpus.dump()+" expected "+expected.dump());
            else
                check(gpus[i].at("index")==i&&gpus[i].at("name")==expected[i][0]&&gpus[i].at("total_vram_gb")==expected[i][1]&&gpus[i].at("backend")=="cuda","visible ordinal properties: "+gpus.dump());
            best=std::max(best,gpus[i].at("tier").get<int>());
        }
        check(gpus.back().at("backend")=="cpu"&&gpus.back().at("index")==-1&&message.at("best_tier")==best,"CPU entry and best tier");
    }
    server.stop();
}
void gpu_cases(const std::string &self){
    namespace fs=std::filesystem;
#ifdef EZLOCALAI_USE_HIP
    std::ofstream("devices.tsv")<<"AMD Radeon RX 7900 XTX\t11\t0\t25232932864\nAMD Instinct MI300X\t9\t4\t34091302912\n";
#else
    std::ofstream("devices.tsv")<<"NVIDIA GeForce RTX 3090\t8\t6\t25232932864\nNVIDIA GeForce RTX 5090\t12\t0\t34091302912\n";
    const Json a=Json::array({"NVIDIA GeForce RTX 3090",23.5}),b=Json::array({"NVIDIA GeForce RTX 5090",31.75});
#endif
    // nvidia-smi can still see physical cards when CUDA visibility is empty.
    std::ofstream("nvidia-smi")<<"#!/bin/sh\nprintf '0, NVIDIA GeForce RTX 3090, 24576\\n1, NVIDIA GeForce RTX 5090, 32768\\n'\n";
    fs::permissions("nvidia-smi",fs::perms::owner_all);
    const auto path=fs::current_path().string()+":"+(std::getenv("PATH")?std::getenv("PATH"):"");
    setenv("PATH",path.c_str(),1);
    setenv("EZLOCALAI_TEST_CUDA_DEVICES",fs::absolute("devices.tsv").c_str(),1);
    setenv("LD_LIBRARY_PATH",TEST_CUDA_DRIVER_DIRECTORY,1);unsetenv("EZLOCALAI_TEST_CUDA_FAIL");
    const auto run=[&](const std::string &mask,const Json &expected,const std::string &failure){
        const auto result=run_process_to_files({self,"--gpu-report",mask,expected.dump(),failure},"gpu-report.log","",15,true);
        std::ifstream log("gpu-report.log");const std::string output{std::istreambuf_iterator<char>(log),{}};
        check(!result.timed_out&&result.status==0,"GPU registration/heartbeat case "+mask+"/"+failure+": "+output);
    };
#ifdef EZLOCALAI_USE_HIP
    std::ifstream reference(TEST_HIP_GPU_REFERENCE);
    const auto captured=Json::parse(reference);
    for(const auto &entry:captured.at("cases"))
        run(entry.at("mask"),entry.at("expected"),entry.at("failure"));
#else
    const std::vector<std::pair<std::string,Json>> cases{{"all",Json::array({a,b})},{"1,0",Json::array({b,a})},{"1",Json::array({b})},{"",Json::array()},{"-1",Json::array()}};
    for(const auto &entry:cases)run(entry.first,entry.second,"");
    for(const auto &failure:{"name","compute","memory"})
        run("1,0",Json::array({Json::array({"cuda:0",0}),Json::array({"cuda:1",0})}),failure);
#endif
    std::cout<<"8 GPU visibility/properties cases passed for registration and heartbeat\n";
}
}
int main(int argc, char **argv) { try {
    if(argc==5&&std::string(argv[1])=="--gpu-report"){gpu_reporting(argv[2],Json::parse(argv[3]),argv[4]);return 0;}
    const auto self=std::filesystem::absolute(argv[0]).string();
    check(argc == 3, "source dependency and model-reporting cases required"); std::ifstream input(argv[1]); Json fixture; input >> fixture;
    char pattern[] = "/tmp/ezlocalai-worker-metadata-XXXXXX"; check(mkdtemp(pattern), "temporary directory");
    const auto previous = std::filesystem::current_path(); std::filesystem::current_path(pattern);
    struct Cleanup { std::filesystem::path previous, path; ~Cleanup() { std::filesystem::current_path(previous); std::error_code ec; std::filesystem::remove_all(path, ec); } } cleanup{previous, pattern};
    Capture capture;
    for (const auto &row : fixture.at("cases")) one(row, capture);
    const auto &rows = fixture.at("cases"); one(rows.back(), capture, &rows.at(rows.size() - 2), &rows.back());
    std::ifstream model_input(argv[2]);const auto models=Json::parse(model_input);
    for(const auto &row:models.at("cases"))one(row,capture,row.contains("detached")?&row.at("detached"):nullptr,row.contains("detached")?&row:nullptr);
    std::cout<<models.at("cases").size()<<" model reporting snapshots and projector unload/restore passed\n";
    std::cout << rows.size() << " Python worker-extra snapshots plus native request encodings match registration and heartbeat; detach/restore updates verified\n";
    gpu_cases(self);
    return 0;
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; } }
