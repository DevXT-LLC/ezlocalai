#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/tunnel_connection.hpp"
#include "ezlocalai/http_deadline.hpp"
#include <nlohmann/json.hpp>
#include <arpa/inet.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
using namespace std::chrono_literals;
using Json = nlohmann::json;
static void check(bool ok, const std::string &message) { if (!ok) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); } }
static bool send_all(int fd, const std::string &value) {
    for (size_t at = 0; at < value.size();) {
        const auto count = send(fd,value.data()+at,value.size()-at,MSG_NOSIGNAL);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        at += count;
    }
    return true;
}
static std::string request(int fd) {
    std::string result; char byte;
    while (result.find("\r\n\r\n") == std::string::npos) {
        const auto count = recv(fd,&byte,1,0); if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return ""; result += byte;
    }
    auto lower = result; for (auto &c : lower) if (c >= 'A' && c <= 'Z') c += 32;
    const auto field = lower.find("content-length:"); const auto length = field == std::string::npos ? 0 : std::stoul(lower.substr(field+15));
    for (size_t i = 0; i < length;) { const auto count = recv(fd,&byte,1,0); if (count < 0 && errno == EINTR) continue; if (count <= 0) return ""; ++i; }
    return result;
}
static void run(const Json &row) {
    const auto mode = row.at("name").get<std::string>();
    int listener = socket(AF_INET,SOCK_STREAM,0); check(listener >= 0,"listener");
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    check(bind(listener,reinterpret_cast<sockaddr *>(&address),sizeof(address)) == 0,"bind");
    socklen_t length = sizeof(address); getsockname(listener,reinterpret_cast<sockaddr *>(&address),&length);
    check(listen(listener,16) == 0,"listen");
    std::atomic<bool> done{false}; std::atomic<int> tunnel{-1}; std::string upgrade;
    std::thread router([&] {
        while (!done) {
            pollfd item{listener,POLLIN,0}; if (poll(&item,1,20) <= 0) continue;
            int fd = accept(listener,nullptr,nullptr); if (fd < 0) continue;
            timeval timeout{2,0}; setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
            const auto headers = request(fd);
            if (headers.find("Upgrade: websocket") != std::string::npos) {
                if (tunnel >= 0) { close(fd); continue; }
                const auto field = headers.find("Sec-WebSocket-Key: "); check(field != std::string::npos,"upgrade key");
                const auto begin = field+19; const auto key = headers.substr(begin,headers.find("\r\n",begin)-begin);
                upgrade = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: " + ezlocalai::tunnel_websocket_accept(key) + "\r\n\r\n";
                tunnel = fd;
            } else { send_all(fd,"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}"); close(fd); }
        }
    });
    for (const auto &entry : ezlocalai::environment_defaults) unsetenv(entry.first.data());
    setenv("EZLOCALAI_DOTENV_PATH","",1); setenv("EZLOCALAI_API_KEY","none",1); setenv("API_KEY","",1); setenv("DEFAULT_MODEL","none",1);
    setenv("WORKER_TUNNEL","true",1); setenv("WORKER_HEARTBEAT_INTERVAL","2",1);
    const auto url = "http://127.0.0.1:" + std::to_string(ntohs(address.sin_port)); setenv("ROUTER_URL",url.c_str(),1);
    for (auto key : {"TTS_ENABLED","STT_ENABLED","EMBEDDING_ENABLED","IMAGE_ENABLED","VIDEO_ENABLED","MUSIC_ENABLED"}) setenv(key,"false",1);
    auto probes = ezlocalai::native_resource_probes(); probes.gpu_count = [] { return 0; };
    probes.total_per_gpu = [] { return std::vector<double>{}; }; probes.free_per_gpu = [] { return std::vector<double>{}; };
    probes.system_ram = [] { return 128.0; }; probes.available_ram = [] { return 2.5; };
    ezl_env_config config{}; ezl_load_config(&config);
    ezlocalai::NativeServer worker(config,std::make_unique<ezlocalai::EchoInferenceBackend>("none"),std::make_shared<ezlocalai::ResourceManager>(probes));
    std::string error; check(worker.start("127.0.0.1",0,&error),"start: "+error);
    auto start = ezlocalai::http_monotonic_time();
    while (tunnel < 0 && ezlocalai::http_monotonic_time()-start < 5) std::this_thread::sleep_for(5ms);
    check(tunnel >= 0,mode+": worker upgrade request");
    const int fd = tunnel;
    if (mode != "stop-upgrade") {
        start = ezlocalai::http_monotonic_time(); check(send_all(fd,upgrade),"send upgrade");
        double next_frame = 0, closed_at = -1; size_t sent = 0; std::string wire; int pongs = 0;
        while (ezlocalai::http_monotonic_time()-start < 65) {
            const auto elapsed = ezlocalai::http_monotonic_time()-start;
            if (elapsed >= next_frame) {
                const auto frame = mode == "control-ping" ? std::string("\x89\x01p",3)
                    : mode == "control-pong" ? std::string("\x8a\x01p",3)
                    : mode == "partial" ? std::string(sent ? "\x00\x01x" : "\x01\x01x",3)
                    : mode == "binary" ? std::string("\x82\x01x",3)
                    : mode == "invalid-json" ? std::string("\x81\x01{",3)
                    : std::string("\x81\x0c{\"t\":\"ping\"}",14);
                if (!send_all(fd,frame)) { closed_at = elapsed; break; }
                ++sent; next_frame = elapsed+2;
            }
            pollfd item{fd,POLLIN,0}; if (poll(&item,1,20) <= 0) continue;
            char buffer[4096]; const auto count = recv(fd,buffer,sizeof(buffer),0);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) { closed_at = elapsed; break; }
            wire.append(buffer,count);
            while (wire.size() >= 2) {
                const unsigned opcode = static_cast<unsigned char>(wire[0]) & 15, size = static_cast<unsigned char>(wire[1]) & 127;
                check(size < 126 && (wire[1] & 128),"small masked response");
                if (wire.size() < size+6) break;
                if (opcode == 8) closed_at = elapsed;
                if (opcode == 10 || opcode == 1) ++pongs;
                wire.erase(0,size+6);
            }
            if (closed_at >= 0) break;
        }
        if (row.at("timed_out").get<bool>()) check(closed_at >= 59.5 && closed_at < 62.0,mode+": actual 60-second idle expiry, observed="+std::to_string(closed_at));
        else check(closed_at < 0,mode+": complete messages keep connection alive for 65 seconds");
        if (mode == "control-ping" || mode == "application-ping") check(pongs >= 25,mode+": pings answered throughout test");
        std::cout << mode << " close_at=" << closed_at << " frames=" << sent << " pongs=" << pongs << '\n';
    } else std::this_thread::sleep_for(150ms);
    start = ezlocalai::http_monotonic_time(); worker.stop(); const auto stopping = ezlocalai::http_monotonic_time()-start;
    check(stopping < 2,mode+": worker stop interrupts connection wait");
    done = true; router.join(); close(fd); close(listener);
    std::cout << mode << " shutdown=" << stopping << '\n' << std::flush;
}
int main(int argc, char **argv) {
    check(argc == 2,"fixture argument"); std::ifstream input(argv[1]); auto cases = Json::parse(input).at("idle");
    cases.push_back(Json{{"name","stop-upgrade"}});
    std::vector<pid_t> children;
    for (const auto &row : cases) { const auto pid = fork(); check(pid >= 0,"fork"); if (!pid) { run(row); _exit(0); } children.push_back(pid); }
    bool ok = true;
    for (auto pid : children) { int status = 0; check(waitpid(pid,&status,0) == pid,"wait"); if (!WIFEXITED(status) || WEXITSTATUS(status)) ok = false; }
    check(ok,"all six full-duration worker idle cases and interrupted setup");
}
