#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include <arpa/inet.h>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <openssl/sha.h>
#include <poll.h>
#include <sstream>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;
namespace {
void expect(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
std::string read(const fs::path &p) { std::ifstream in(p, std::ios::binary); return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()}; }
std::vector<std::string> split(const std::string &value, char delimiter = '\t') {
    std::vector<std::string> out; size_t start = 0;
    do { const auto end = value.find(delimiter, start); out.push_back(value.substr(start, end == std::string::npos ? end : end - start)); if (end == std::string::npos) break; start = end + 1; } while (true);
    return out;
}
std::string unhex(const std::string &v) { std::string out; for (size_t i = 0; i < v.size(); i += 2) out += static_cast<char>(std::stoi(v.substr(i, 2), nullptr, 16)); return out; }
std::string sha256(const std::string &value) { unsigned char hash[SHA256_DIGEST_LENGTH]; SHA256(reinterpret_cast<const unsigned char *>(value.data()), value.size(), hash); std::ostringstream out; for (auto c : hash) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c); return out.str(); }
std::string join(const std::vector<std::string> &items) { std::string out; for (const auto &v : items) { if (!out.empty()) out += '\n'; out += v; } return out; }
void set(const char *key, const std::string &encoded) { if (encoded == "N") unsetenv(key); else setenv(key, unhex(encoded.substr(1)).c_str(), 1); }
void replace(std::string &value, const std::string &from, const std::string &to) { size_t pos = 0; while ((pos = value.find(from, pos)) != std::string::npos) { value.replace(pos, from.size(), to); pos += to.size(); } }
struct Reply { int status; std::string type, body; };
class Remote {
public:
    Remote() {
        fd_ = socket(AF_INET, SOCK_STREAM, 0); expect(fd_ >= 0, "remote socket");
        sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        expect(!bind(fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)), "remote bind"); socklen_t size = sizeof(address); getsockname(fd_, reinterpret_cast<sockaddr *>(&address), &size);
        url = "http://127.0.0.1:" + std::to_string(ntohs(address.sin_port)); expect(!listen(fd_, 16), "remote listen");
        thread_ = std::thread([&] { run(); });
    }
    ~Remote() { stopped_ = true; shutdown(fd_, SHUT_RDWR); thread_.join(); close(fd_); }
    void configure(Reply resources, Reply models) { std::lock_guard<std::mutex> lock(mutex_); resources_ = std::move(resources); models_ = std::move(models); events_.clear(); error_.clear(); }
    void observe(std::function<void(const std::string &, const std::string &)> observer) { std::lock_guard<std::mutex> lock(mutex_); observer_ = std::move(observer); }
    void event(std::string value) { std::lock_guard<std::mutex> lock(mutex_); events_.push_back(std::move(value)); }
    std::string events() { std::lock_guard<std::mutex> lock(mutex_); expect(error_.empty(), error_); return join(events_); }
    std::string url;
private:
    int fd_; std::atomic<bool> stopped_{false}; std::thread thread_; std::mutex mutex_; std::string error_; std::vector<std::string> events_; Reply resources_{200,"application/json","{}"}, models_{200,"application/json","{}"};
    std::function<void(const std::string &, const std::string &)> observer_;
    void run() {
        while (!stopped_) {
            pollfd ready{fd_, POLLIN, 0}; if (poll(&ready, 1, 50) <= 0) continue;
            const auto peer = accept(fd_, nullptr, nullptr); if (peer < 0) continue;
            timeval timeout{2,0}; setsockopt(peer, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)); std::string request; char buffer[4096];
            while (request.find("\r\n\r\n") == std::string::npos) { const auto size = recv(peer, buffer, sizeof(buffer), 0); if (size <= 0) break; request.append(buffer, size); }
            std::istringstream first(request); std::string method,path; first >> method >> path; Reply reply;
            {
                std::lock_guard<std::mutex> lock(mutex_); events_.push_back(method+' '+path);
                if (method != "GET" || (path != "/v1/resources" && path != "/v1/models")) error_ = "Unexpected remote request: " + request;
                if (request.find("Content-Type: application/json\r\n") == std::string::npos || (!observer_ && request.find("Authorization:") != std::string::npos)) error_ = "Fallback probe headers differ: " + request;
                if (observer_) {
                    const auto start = request.find("\r\nAuthorization: ");
                    const auto value = start == std::string::npos ? std::string{} : request.substr(start+17, request.find("\r\n",start+17)-(start+17));
                    observer_(path,value);
                }
                reply = path == "/v1/resources" ? resources_ : models_;
            }
            std::ostringstream header; header << "HTTP/1.1 " << reply.status << " Fixture\r\nConnection: close\r\nContent-Length: " << reply.body.size() << "\r\n";
            if (!reply.type.empty()) header << "Content-Type: " << reply.type << "\r\n";
            const auto output = header.str() + "\r\n" + reply.body;
            size_t sent = 0; while (sent < output.size()) { const auto count = send(peer, output.data()+sent, output.size()-sent, MSG_NOSIGNAL); if (count <= 0) break; sent += count; }
            shutdown(peer, SHUT_RDWR); close(peer);
        }
    }
};
size_t cases(const fs::path &root, const fs::path &fixture) {
    Remote remote; std::istringstream input(read(fixture)); size_t count = 0;
    for (std::string line; std::getline(input, line);) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            const auto marker = line.find(" sha256="); if (marker != std::string::npos) expect(sha256(read(root / line.substr(2, marker-2))) == line.substr(marker+8), "Fallback reference source changed"); continue;
        }
        ++count; const auto c = split(line);
        for (const auto &entry : ezlocalai::environment_defaults) unsetenv(entry.first.data());
        setenv("EZLOCALAI_DOTENV_PATH", "", 1); setenv("DEFAULT_MODEL", "none", 1); setenv("EZLOCALAI_API_KEY", "none", 1); setenv("API_KEY", "", 1);
        auto url = unhex(c[1]); replace(url, "http://fallback", remote.url); setenv("FALLBACK_SERVER", url.c_str(), 1);
        remote.configure({std::stoi(c[5]),unhex(c[6]),unhex(c[7])},{std::stoi(c[8]),unhex(c[9]),unhex(c[10])});
        double now = 0; auto probes = ezlocalai::native_resource_probes();
        probes.gpu_count = [] { return 1; }; probes.total_per_gpu = [] { return std::vector<double>{24}; }; probes.system_ram = [] { return 128.0; };
        probes.free_per_gpu = [&] { remote.event("vram"); if (c[4] == "1") throw std::runtime_error("probe failed"); return std::vector<double>{std::stod(c[2])}; };
        probes.available_ram = [&] { remote.event("ram"); return std::stod(c[3]); };
        probes.now = [&] { remote.event("clock"); return now; };
        probes.allocator_stats = [] { return std::vector<ezlocalai::CudaAllocatorStats>{}; };
        auto manager = std::make_shared<ezlocalai::ResourceManager>(probes); ezl_env_config config{}; ezl_load_config(&config);
        ezlocalai::NativeServer server(config,std::make_unique<ezlocalai::EchoInferenceBackend>("none"),manager);
        const auto steps = split(unhex(c[11]), '\n'), results = split(unhex(c[12]), '\n'); expect(steps.size() == results.size(), "step count");
        for (size_t i = 0; i < steps.size(); ++i) {
            const auto step = split(steps[i]), expected = split(results[i]); now = std::stod(step[0]); set("FALLBACK_MEMORY_THRESHOLD",step[1]); set("FALLBACK_VRAM_THRESHOLD",step[2]);
            const auto response = server.handle_request("GET","/v1/fallback/status","",""); auto body = unhex(expected[2]); replace(body,"http://fallback",remote.url);
            const auto label = "fallback " + c[0] + " step " + std::to_string(i);
            expect(response.status == std::stoi(expected[0]),label+" status: "+std::to_string(response.status)+" != "+expected[0]+": "+response.body);
            expect(response.content_type == unhex(expected[1]),label+" media type");
            expect(response.body == body,label+" body: "+response.body+" != "+body);
        }
        expect(remote.events() == unhex(c[13]),"fallback "+c[0]+" event order: "+remote.events()+" != "+unhex(c[13]));
    }
    expect(count == 936, "complete fallback fixture coverage"); return count;
}
size_t lifecycle_cases(const fs::path &root, const fs::path &fixture) {
    Remote a, b; std::mutex mutex; std::vector<std::string> events;
    auto event = [&](std::string value) { std::lock_guard<std::mutex> lock(mutex); events.push_back(std::move(value)); };
    a.observe([&](const auto &path,const auto &key) { event("GET\ta\t"+path+'\t'+key); });
    b.observe([&](const auto &path,const auto &key) { event("GET\tb\t"+path+'\t'+key); });
    auto substitute = [&](std::string value) { replace(value,"http://a",a.url); replace(value,"http://b",b.url); return value; };
    std::istringstream input(read(fixture)); size_t count = 0;
    for (std::string line; std::getline(input,line);) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            const auto marker = line.find(" sha256="); if (marker != std::string::npos) expect(sha256(read(root / line.substr(2,marker-2))) == line.substr(marker+8),"Fallback lifecycle reference source changed"); continue;
        }
        ++count; const auto c = split(line);
        for (const auto &entry : ezlocalai::environment_defaults) unsetenv(entry.first.data());
        setenv("EZLOCALAI_DOTENV_PATH","",1); setenv("DEFAULT_MODEL","none",1); setenv("EZLOCALAI_API_KEY","none",1); setenv("API_KEY","",1);
        ezl_env_config config{}; ezl_load_config(&config);
        // API authentication is independent of the fallback client's first-use
        // environment capture. These fixture API calls use an unauthenticated app.
        unsetenv("EZLOCALAI_API_KEY");
        a.configure({200,"application/json","{\"free_vram_gb\":4.5}"},{200,"application/json","{\"data\":[{\"id\":\"a-model\"}]}"});
        b.configure({200,"application/json","{\"free_vram_gb\":5.5}"},{200,"application/json","{\"data\":[{\"id\":\"b-model\"}]}"});
        events.clear(); double now = 100; auto probes = ezlocalai::native_resource_probes();
        probes.gpu_count = [] { return 1; }; probes.total_per_gpu = [] { return std::vector<double>{24}; }; probes.system_ram = [] { return 128.0; };
        probes.free_per_gpu = [&] { event("vram"); return std::vector<double>{1.25}; };
        probes.available_ram = [&] { event("ram"); return 2.5; };
        probes.now = [&] { event("clock"); return now; };
        probes.allocator_stats = [] { return std::vector<ezlocalai::CudaAllocatorStats>{}; };
        auto manager = std::make_shared<ezlocalai::ResourceManager>(probes);
        std::unique_ptr<ezlocalai::NativeServer> server;
        const auto expected = split(unhex(c[2]),'\n'); size_t index = 0;
        for (const auto &command : split(unhex(c[1]),'\n')) {
            const auto args = split(command);
            if (args[0] == "new") server = std::make_unique<ezlocalai::NativeServer>(config,std::make_unique<ezlocalai::EchoInferenceBackend>("none"),manager);
            else if (args[0] == "set") {
                if (args[2] == "N") unsetenv(args[1].c_str());
                else setenv(args[1].c_str(),substitute(unhex(args[2].substr(1))).c_str(),1);
            } else {
                expect(args[0] == "query" && server && index < expected.size(),"lifecycle fixture operation");
                now = std::stod(args[1]); const auto want = split(expected[index++]);
                const auto response = server->handle_request("GET","/v1/fallback/status","","");
                const auto label = "fallback lifecycle "+c[0]+" query "+std::to_string(index);
                expect(response.status == std::stoi(want[0]),label+" status: "+std::to_string(response.status)+" != "+want[0]);
                expect(response.content_type == unhex(want[1]),label+" media type");
                expect(response.body == substitute(unhex(want[2])),label+" body: "+response.body+" != "+substitute(unhex(want[2])));
            }
        }
        expect(index == expected.size(),"lifecycle query count"); a.events(); b.events();
        expect(join(events) == unhex(c[3]),"fallback lifecycle "+c[0]+" events: "+join(events)+" != "+unhex(c[3]));
    }
    expect(count == 134,"complete fallback lifecycle coverage"); return count;
}
void entry_point_cases() {
    Remote remote;
    remote.configure({200,"application/json","{\"free_vram_gb\":4.5}"},{200,"application/json","{\"data\":[{\"id\":\"entry-model\"}]}"});
    struct Entry { std::string path, body, type = "application/json"; bool captures = true; };
    const std::vector<Entry> entries{
        {"/health","","application/json",false},
        {"/v1/fallback/status",""}, {"/v1/resources",""},
        {"/v1/chat/completions",R"({"messages":[],"disable_fallback":true})"},
        {"/v1/completions",R"({"prompt":"test","disable_fallback":true})"},
        {"/v1/embeddings",R"({"input":"test"})"},
        {"/v1/audio/speech",R"({"input":"test"})"},
        {"/v1/audio/transcriptions","--fixture\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF test\r\n--fixture--\r\n","multipart/form-data; boundary=fixture"},
        {"/v1/images/generations",R"({"prompt":"test"})"},
        {"/v1/images/edits",R"({"prompt":"test","image":"data:image/png;base64,dGVzdA=="})"},
        {"/v1/videos/generations",R"({"prompt":"test"})"},
        {"/v1/audio/music",R"({"prompt":"test","lyrics":"test","keyscale":"C major","duration":10})"},
        {"/v1/videos/music",R"({"prompt":"test","lyrics":"test","keyscale":"C major","duration":10})"}
    };
    for (const auto &entry : entries) {
        for (const auto &setting : ezlocalai::environment_defaults) unsetenv(setting.first.data());
        setenv("EZLOCALAI_DOTENV_PATH","",1); setenv("EZLOCALAI_API_KEY","none",1); setenv("API_KEY","",1); setenv("DEFAULT_MODEL","none",1);
        setenv("IMAGE_ENABLED","false",1); setenv("VIDEO_ENABLED","false",1); setenv("MUSIC_ENABLED","false",1); setenv("FALLBACK_SERVER","",1);
        auto probes = ezlocalai::native_resource_probes();
        probes.gpu_count = [] { return 0; }; probes.total_per_gpu = [] { return std::vector<double>{}; };
        probes.free_per_gpu = [] { return std::vector<double>{}; }; probes.allocator_stats = [] { return std::vector<ezlocalai::CudaAllocatorStats>{}; };
        probes.system_ram = [] { return 128.0; }; probes.available_ram = [] { return 2.5; }; probes.now = [] { return 100.0; };
        ezl_env_config config{}; ezl_load_config(&config);
        ezlocalai::NativeServer server(config,std::make_unique<ezlocalai::EchoInferenceBackend>("none"),std::make_shared<ezlocalai::ResourceManager>(probes));
        const auto first = server.handle_request(entry.body.empty() ? "GET" : "POST",entry.path,"",entry.body,entry.type);
        expect(first.status != 422 && first.status != 404,"valid first-use request "+entry.path+": "+first.body);
        setenv("FALLBACK_SERVER",remote.url.c_str(),1);
        const auto status = server.handle_request("GET","/v1/fallback/status","","");
        const auto configured = std::string("\"configured\":")+(entry.captures ? "false" : "true");
        expect(status.status == 200 && status.body.find(configured) != std::string::npos,"fallback capture through "+entry.path+": "+status.body);
    }
    remote.events();
}
void concurrent_initialization_case() {
    Remote remote;
    remote.configure({200,"application/json","{\"free_vram_gb\":4.5}"},{200,"application/json","{\"data\":[{\"id\":\"concurrent-model\"}]}"});
    std::atomic<int> wrong_headers{0};
    remote.observe([&](const auto &,const auto &key) { if (key != "Bearer initial-key") ++wrong_headers; });
    setenv("FALLBACK_SERVER",remote.url.c_str(),1); setenv("FALLBACK_API_KEY","initial-key",1);
    auto probes = ezlocalai::native_resource_probes();
    probes.gpu_count = [] { return 0; }; probes.total_per_gpu = [] { return std::vector<double>{}; }; probes.free_per_gpu = [] { return std::vector<double>{}; };
    probes.system_ram = [] { return 128.0; }; probes.available_ram = [] { return 2.5; }; probes.now = [] { return 100.0; };
    ezl_env_config config{}; ezl_load_config(&config);
    ezlocalai::NativeServer server(config,std::make_unique<ezlocalai::EchoInferenceBackend>("none"),std::make_shared<ezlocalai::ResourceManager>(probes));
    std::vector<std::thread> threads; std::vector<ezlocalai::HttpResponse> responses(8); std::atomic<int> ready{0}; std::atomic<bool> go{false};
    for (size_t i = 0; i < responses.size(); ++i) threads.emplace_back([&,i] { ++ready; while (!go.load()) std::this_thread::yield(); responses[i] = server.handle_request("GET","/v1/fallback/status","",""); });
    while (ready.load() != static_cast<int>(responses.size())) std::this_thread::yield(); go = true;
    for (auto &thread : threads) thread.join();
    setenv("FALLBACK_SERVER","",1); setenv("FALLBACK_API_KEY","changed-key",1);
    responses.push_back(server.handle_request("GET","/v1/fallback/status","",""));
    for (const auto &response : responses) expect(response.status == 200 && response.body.find("concurrent-model") != std::string::npos,"shared concurrent fallback initialization");
    remote.events(); expect(!wrong_headers,"concurrent captured fallback credentials");
}
void resource_threshold_cases() {
    auto probes = ezlocalai::native_resource_probes();
    probes.gpu_count = [] { return 0; }; probes.total_per_gpu = [] { return std::vector<double>{}; };
    probes.free_per_gpu = [] { return std::vector<double>{}; }; probes.allocator_stats = [] { return std::vector<ezlocalai::CudaAllocatorStats>{}; };
    probes.system_ram = [] { return 128.0; }; probes.available_ram = [] { return 64.0; };
    setenv("FALLBACK_SERVER", "", 1); ezl_env_config config{}; ezl_load_config(&config);
    ezlocalai::NativeServer server(config, std::make_unique<ezlocalai::EchoInferenceBackend>("none"), std::make_shared<ezlocalai::ResourceManager>(probes));
    for (const auto *value : {"", "bad", "1x", "nan", "inf", "-Infinity"}) {
        setenv("FALLBACK_MEMORY_THRESHOLD", value, 1);
        const auto response = server.handle_request("GET", "/v1/resources", "", "");
        expect(response.status == 500 && response.body == "Internal Server Error" && response.content_type == "text/plain; charset=utf-8", "resource threshold parsing/JSON serialization error");
    }
    setenv("FALLBACK_MEMORY_THRESHOLD", "1_0", 1);
    const auto response = server.handle_request("GET", "/v1/resources", "", "");
    expect(response.status == 200 && response.body.find("\"memory_threshold_gb\":10.0") != std::string::npos, "resource threshold uses Python float syntax");
}
}
int main(int argc, char **argv) {
    try { expect(argc == 3, "reference checkout and fixture required"); const auto count = cases(argv[1],argv[2]); const auto lifetime = lifecycle_cases(argv[1],fs::path(argv[2]).parent_path()/"fallback_lifecycle_cases.tsv"); entry_point_cases(); concurrent_initialization_case(); resource_threshold_cases(); std::cout << "Verified " << count << " source fallback API/policy/cache/JSON cases and " << lifetime << " singleton lifecycle cases over native HTTP, plus API entry points, concurrent initialization and resource threshold checks\n"; return 0; }
    catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
