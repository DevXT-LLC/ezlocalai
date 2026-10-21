#include "ezlocalai/http_deadline.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"

#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>
#include <sstream>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>

using namespace std::chrono_literals;
namespace fs = std::filesystem;
namespace {
std::atomic<bool> resolver_finished{false};
void expect(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::string read(const fs::path &p) { std::ifstream in(p); return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()}; }
std::vector<std::string> split(const std::string &s) { std::vector<std::string> out; size_t p = 0; for (;;) { const auto n = s.find('\t',p); out.push_back(s.substr(p,n == std::string::npos ? n : n-p)); if (n == std::string::npos) return out; p = n+1; } }
std::string unhex(const std::string &s) { std::string out; for (size_t i = 0; i < s.size(); i += 2) out += static_cast<char>(std::stoi(s.substr(i,2),nullptr,16)); return out; }
void env(const char *key, const std::string &value) { if (value == "N") unsetenv(key); else setenv(key,unhex(value.substr(1)).c_str(),1); }
std::string sha256(const std::string &value) { unsigned char bytes[32]; SHA256(reinterpret_cast<const unsigned char *>(value.data()),value.size(),bytes); std::ostringstream out; for (auto b : bytes) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(b); return out.str(); }
double elapsed(double start) { return ezlocalai::http_monotonic_time()-start; }
void nonblocking(int fd) { expect(fcntl(fd,F_SETFL,fcntl(fd,F_GETFL,0)|O_NONBLOCK) == 0,"nonblocking socket"); }
struct Pair {
    int fd[2];
    Pair() { expect(socketpair(AF_UNIX,SOCK_STREAM,0,fd) == 0,"socketpair"); nonblocking(fd[0]); }
    ~Pair() { close(fd[0]); close(fd[1]); }
};
size_t timer_cases(const fs::path &path) {
    std::istringstream input(read(path)); size_t count = 0;
    for (std::string line; std::getline(input,line);) {
        if (line.empty() || line.front() == '#') continue;
        ++count; const auto c = split(line); bool threw = false; std::optional<double> result;
        try { result = ezlocalai::http_total_deadline(strtod(c[1].c_str(),nullptr),strtod(c[0].c_str(),nullptr)); } catch (const std::invalid_argument &) { threw = true; }
        expect(threw == (c[2] == "E"),"reference timer exception "+line);
        if (!threw) {
            expect(result.has_value() == (c[2] != "N"),"reference timer enabled "+line);
            if (result) expect(*result == strtod(c[2].c_str(),nullptr),"reference timer deadline "+line);
        }
    }
    expect(count == 176,"complete timer fixtures"); return count;
}
void io_cases() {
    {
        Pair pair; ezlocalai::HttpDeadline deadline(.12); const auto start = ezlocalai::http_monotonic_time();
        std::thread peer([&] { for (int i = 0; i < 12; ++i) { send(pair.fd[1],"x",1,MSG_NOSIGNAL); std::this_thread::sleep_for(25ms); } });
        char byte; int received = 0; ssize_t n;
        while ((n = ezlocalai::http_read(pair.fd[0],nullptr,&byte,1,deadline)) > 0) ++received;
        const auto duration = elapsed(start); const int error = errno; peer.join();
        expect(n < 0 && error == ETIMEDOUT && received > 1 && received < 12 && duration >= .10 && duration < .35,"dribbled reads share a deadline");
    }
    {
        Pair pair; int small = 4096; setsockopt(pair.fd[0],SOL_SOCKET,SO_SNDBUF,&small,sizeof(small));
        ezlocalai::HttpDeadline deadline(.12); std::string upload(1024*1024,'x'); size_t sent = 0; const auto start = ezlocalai::http_monotonic_time();
        for (;;) { const auto n = ezlocalai::http_write(pair.fd[0],nullptr,upload.data()+sent,upload.size()-sent,deadline); if (n < 0) break; sent += n; if (sent == upload.size()) break; }
        expect(errno == ETIMEDOUT && sent > 0 && sent < upload.size() && elapsed(start) < .4,"blocked upload deadline");
    }
    {
        Pair pair; ezlocalai::HttpDeadline deadline(.18);
        std::thread peer([&] { std::this_thread::sleep_for(110ms); send(pair.fd[1],"h",1,MSG_NOSIGNAL); std::this_thread::sleep_for(110ms); send(pair.fd[1],"b",1,MSG_NOSIGNAL); });
        char byte; const bool first = ezlocalai::http_read(pair.fd[0],nullptr,&byte,1,deadline) == 1;
        const bool second = ezlocalai::http_read(pair.fd[0],nullptr,&byte,1,deadline) < 0 && errno == ETIMEDOUT;
        peer.join(); expect(first && second,"headers and body share one remaining budget");
    }
    {
        Pair pair; SSL_CTX *context = SSL_CTX_new(TLS_client_method()); SSL *ssl = SSL_new(context); SSL_set_fd(ssl,pair.fd[0]);
        ezlocalai::HttpDeadline deadline(.08); const auto start = ezlocalai::http_monotonic_time();
        const bool connected = ezlocalai::http_tls_connect(ssl,pair.fd[0],deadline); const int error = errno;
        SSL_free(ssl); SSL_CTX_free(context);
        expect(!connected && error == ETIMEDOUT && elapsed(start) < .4,"TLS handshake deadline");
    }
    {
        const auto pid = fork(); expect(pid >= 0,"TLS reset subprocess");
        if (!pid) {
            signal(SIGPIPE,SIG_DFL);
            Pair pair; close(pair.fd[1]); pair.fd[1] = -1;
            SSL_CTX *context = SSL_CTX_new(TLS_client_method()); SSL *ssl = SSL_new(context); SSL_set_fd(ssl,pair.fd[0]);
            ezlocalai::HttpDeadline deadline(.08);
            const bool connected = ezlocalai::http_tls_connect(ssl,pair.fd[0],deadline);
            SSL_free(ssl); SSL_CTX_free(context); _exit(connected ? 1 : 0);
        }
        int status = 0; waitpid(pid,&status,0);
        expect(WIFEXITED(status) && WEXITSTATUS(status) == 0,"TLS peer disconnect is a request error, not SIGPIPE");
    }
    {
        addrinfo hints{}; hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
        ezlocalai::HttpDeadline deadline(.05); const auto start = ezlocalai::http_monotonic_time();
        const auto result = ezlocalai::http_resolve("delayed.ezlocalai.test","80",hints,deadline);
        expect(result.timed_out && !result.addresses && elapsed(start) < .15 && !resolver_finished,"DNS cancellation returns before resolver completes");
        while (!resolver_finished && elapsed(start) < 1) std::this_thread::sleep_for(5ms);
        expect(resolver_finished,"detached DNS task finishes safely");
        ezlocalai::HttpDeadline unbounded(0);
        const auto failed = ezlocalai::http_resolve("missing.ezlocalai.test","80",hints,unbounded);
        expect(failed.error == EAI_NONAME && !failed.addresses && !failed.timed_out,"DNS errors preserved");
    }
}

class SlowRemote {
public:
    SlowRemote(std::string kind, std::string mode) : kind_(std::move(kind)),mode_(std::move(mode)) {
        listener_ = socket(AF_INET,SOCK_STREAM,0); expect(listener_ >= 0,"listener");
        sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        expect(bind(listener_,reinterpret_cast<sockaddr *>(&addr),sizeof(addr)) == 0,"bind");
        socklen_t size = sizeof(addr); getsockname(listener_,reinterpret_cast<sockaddr *>(&addr),&size);
        url = "http://127.0.0.1:"+std::to_string(ntohs(addr.sin_port)); expect(listen(listener_,16) == 0,"listen");
        thread_ = std::thread([&] { run(); });
    }
    ~SlowRemote() {
        stopped_ = true; shutdown(listener_,SHUT_RDWR);
        { std::lock_guard<std::mutex> lock(mutex_); if (peer_ >= 0) shutdown(peer_,SHUT_RDWR); }
        thread_.join(); close(listener_);
    }
    std::string url;
    std::atomic<int> attempts{0};
private:
    std::string kind_,mode_; int listener_ = -1, peer_ = -1; std::mutex mutex_; std::thread thread_; std::atomic<bool> stopped_{false};
    bool pause(int ms) { for (int i = 0; i < ms && !stopped_; i += 5) std::this_thread::sleep_for(5ms); return !stopped_; }
    bool send(int peer, const std::string &value) {
        for (size_t offset = 0; offset < value.size();) { const auto n = ::send(peer,value.data()+offset,value.size()-offset,MSG_NOSIGNAL); if (n <= 0) return false; offset += n; }
        return true;
    }
    void serve(int peer) {
        timeval timeout{2,0}; setsockopt(peer,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
        std::string request; char buffer[4096]; size_t length = 0, end = std::string::npos;
        for (;;) {
            const auto n = recv(peer,buffer,sizeof(buffer),0); if (n <= 0) return; request.append(buffer,n);
            if ((end = request.find("\r\n\r\n")) != std::string::npos) {
                const auto field = request.find("Content-Length: "); if (field != std::string::npos) length = std::stoul(request.substr(field+16));
                if (request.size() >= end+4+length) break;
            }
        }
        std::istringstream first(request); std::string method,path; first >> method >> path;
        const bool target = kind_ == "health" ? path == "/v1/resources" : method == "POST";
        if (!target) {
            const std::string body = path == "/v1/resources" ? "{\"free_vram_gb\":100}" : "{\"data\":[]}";
            send(peer,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "+std::to_string(body.size())+"\r\n\r\n"+body); return;
        }
        ++attempts;
        const std::string body = kind_ == "health" ? "{\"free_vram_gb\":100,\"padding\":\""+std::string(100,'.')+"\"}" : "{\"marker\":\"remote\"}";
        std::string headers = std::string("HTTP/1.1 ")+(mode_ == "error-stall" ? "503" : "200")+" Fixture\r\nContent-Type: application/json\r\n";
        if (mode_.find("chunked") == 0) headers += "Transfer-Encoding: chunked\r\n";
        else if (mode_ != "eof") headers += "Content-Length: "+std::to_string(body.size()+(mode_ == "truncated" ? 10 : 0))+"\r\n";
        headers += "\r\n";
        if (mode_ == "header-drip") { for (size_t i = 0; i < headers.size(); i += 10) if (!send(peer,headers.substr(i,10)) || !pause(40)) return; }
        else if (!send(peer,headers)) return;
        if (mode_ == "error-stall") { pause(500); return; }
        if (mode_ == "initial-delay" && !pause(180)) return;
        if (mode_ == "length" || mode_ == "chunked" || mode_ == "eof" || mode_ == "health-drip") {
            for (size_t i = 0; i < body.size(); i += 2) {
                const auto part = body.substr(i,2);
                if (!send(peer,mode_ == "chunked" ? std::to_string(part.size())+"\r\n"+part+"\r\n" : part) || !pause(mode_ == "health-drip" ? 150 : 25)) return;
            }
            if (mode_ == "chunked") send(peer,"0\r\n\r\n");
        } else if (mode_ == "chunked-missing-end" || mode_ == "chunked-incomplete-zero" || mode_ == "chunked-trailer-good" || mode_ == "chunked-trailer-stall") {
            std::ostringstream size; size << std::hex << body.size(); send(peer,size.str()+"\r\n"+body+"\r\n");
            if (mode_ == "chunked-incomplete-zero") send(peer,"0\r\n");
            if (mode_ == "chunked-trailer-good") send(peer,"0\r\nX-End: yes\r\n\r\n");
            if (mode_ == "chunked-trailer-stall") { send(peer,"0\r\nX-End: yes\r\n"); pause(500); }
        }
        else send(peer,body);
        if (mode_ == "keepalive") pause(300);
    }
    void run() {
        while (!stopped_) {
            const int peer = accept(listener_,nullptr,nullptr); if (peer < 0) continue;
            { std::lock_guard<std::mutex> lock(mutex_); peer_ = peer; }
            serve(peer);
            { std::lock_guard<std::mutex> lock(mutex_); peer_ = -1; shutdown(peer,SHUT_RDWR); close(peer); }
        }
    }
};

size_t api_cases(const fs::path &root, const fs::path &path) {
    std::istringstream input(read(path)); size_t count = 0;
    for (std::string line; std::getline(input,line);) {
        if (line.empty()) continue;
        if (line[0] == '#') { const auto marker = line.find(" sha256="); if (marker != std::string::npos) expect(sha256(read(root/line.substr(2,marker-2))) == line.substr(marker+8),"reference timeout source changed"); continue; }
        ++count; const auto c = split(line); SlowRemote remote(c[1],c[4]);
        for (const auto &setting : ezlocalai::environment_defaults) unsetenv(setting.first.data());
        setenv("EZLOCALAI_DOTENV_PATH","",1); setenv("EZLOCALAI_API_KEY","none",1); setenv("API_KEY","",1); setenv("DEFAULT_MODEL","none",1);
        setenv("MUSIC_ENABLED","false",1); setenv("VIDEO_ENABLED","false",1); setenv("FALLBACK_SERVER",remote.url.c_str(),1);
        auto probes = ezlocalai::native_resource_probes();
        probes.gpu_count = [] { return 0; }; probes.total_per_gpu = [] { return std::vector<double>{}; }; probes.free_per_gpu = [] { return std::vector<double>{}; };
        probes.system_ram = [] { return 128.0; }; probes.available_ram = [] { return 2.5; }; probes.now = [] { return 100.0; };
        ezl_env_config config{}; ezl_load_config(&config);
        ezlocalai::NativeServer server(config,std::make_unique<ezlocalai::EchoInferenceBackend>("none"),std::make_shared<ezlocalai::ResourceManager>(probes));
        env("REQUEST_TIMEOUT",c[2]); env("ACE_STEP_TIMEOUT",c[3]);
        const auto started = ezlocalai::http_monotonic_time();
        const auto response = c[1] == "health" ? server.handle_request("GET","/v1/fallback/status","","") : server.handle_request("POST",c[1] == "music" ? "/v1/audio/music" : "/v1/videos/music","",R"({"prompt":"test","lyrics":"test","keyscale":"C major","duration":10})");
        const auto duration = elapsed(started); const auto expected = unhex(c[5]);
        expect(response.status == 200,"timeout API status case "+c[0]+": "+response.body);
        if (c[1] == "health") {
            expect(response.body.find(expected) != std::string::npos,"reference availability deadline/framing "+c[0]+": "+response.body);
            if (c[4] == "health-drip") expect(duration >= 4.9 && duration < 6.3,"five-second rounded total health deadline");
            if (c[4] == "error-stall") expect(duration < .3,"failed health returns without reading body");
        } else {
            expect((response.body == "{\"marker\":\"remote\"}") == (expected == "ok"),"reference forwarding deadline/framing "+c[0]+": "+response.body);
            if (c[4] == "keepalive") expect(duration < .2,"completed payload does not wait for EOF");
        }
        expect(remote.attempts == std::stoi(c[6]),"reference HTTP attempt count "+c[0]);
    }
    expect(count == 68,"complete HTTP timeout fixtures"); return count;
}
}

// A deterministic resolver ABI double exercises cancellation of the real
// resolver task; normal hostnames still use libc and real loopback sockets.
extern "C" int getaddrinfo(const char *host, const char *service, const addrinfo *hints, addrinfo **result) {
    using Real = int (*)(const char *,const char *,const addrinfo *,addrinfo **);
    static auto real = reinterpret_cast<Real>(dlsym(RTLD_NEXT,"getaddrinfo"));
    if (host && std::string(host) == "missing.ezlocalai.test") { *result = nullptr; return EAI_NONAME; }
    if (host && std::string(host) == "delayed.ezlocalai.test") {
        std::this_thread::sleep_for(200ms);
        const auto status = real("127.0.0.1",service,hints,result); resolver_finished = true; return status;
    }
    return real(host,service,hints,result);
}

int main(int argc, char **argv) {
    try { expect(argc == 3,"reference root and fixture directory required"); const auto timers = timer_cases(fs::path(argv[2])/"http_deadline_cases.tsv"); io_cases(); const auto api = api_cases(argv[1],fs::path(argv[2])/"fallback_timeout_cases.tsv"); std::cout << "Verified " << timers << " aiohttp timer cases, " << api << " source HTTP timeout/framing cases and socket/DNS/TLS deadline checks\n"; return 0; }
    catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
