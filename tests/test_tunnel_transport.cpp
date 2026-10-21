#include "ezlocalai/tunnel_connection.hpp"
#include "ezlocalai/http_deadline.hpp"
#include <nlohmann/json.hpp>
#include <openssl/pem.h>
#include <openssl/x509v3.h>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <thread>
#include <unistd.h>
using namespace std::chrono_literals;
using Json = nlohmann::json;
static void check(bool ok, const std::string &what) { if (!ok) { std::cerr << "FAIL: " << what << " errno=" << errno << '\n'; std::exit(1); } }
static std::atomic<bool> resolver_done{false};
extern "C" int getaddrinfo(const char *host, const char *port, const addrinfo *hints, addrinfo **result) {
    if (host && std::string(host) == "cancelled.tunnel.test") {
        std::this_thread::sleep_for(400ms); resolver_done = true; return EAI_NONAME;
    }
    using Function = int (*)(const char *, const char *, const addrinfo *, addrinfo **);
    static const auto original = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "getaddrinfo"));
    return original(host, port, hints, result);
}
struct Certificate {
    SSL_CTX *context = SSL_CTX_new(TLS_server_method());
    std::string file;
    Certificate() {
        EVP_PKEY *key = EVP_PKEY_Q_keygen(nullptr, nullptr, "EC", "P-256"); check(key, "test key");
        X509 *cert = X509_new(); X509_set_version(cert, 2); ASN1_INTEGER_set(X509_get_serialNumber(cert), 1);
        X509_gmtime_adj(X509_getm_notBefore(cert), -60); X509_gmtime_adj(X509_getm_notAfter(cert), 3600);
        X509_set_pubkey(cert, key); auto name = X509_get_subject_name(cert);
        X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC, reinterpret_cast<const unsigned char *>("localhost"), -1, -1, 0);
        X509_set_issuer_name(cert, name); X509V3_CTX extension_context{}; X509V3_set_ctx(&extension_context, cert, cert, nullptr, nullptr, 0);
        for (auto entry : {std::pair<int, const char *>{NID_subject_alt_name, "DNS:localhost,IP:127.0.0.1"}, {NID_basic_constraints, "critical,CA:TRUE"}}) {
            X509_EXTENSION *extension = X509V3_EXT_conf_nid(nullptr, &extension_context, entry.first, const_cast<char *>(entry.second));
            check(extension && X509_add_ext(cert, extension, -1), "test certificate extension"); X509_EXTENSION_free(extension);
        }
        check(X509_sign(cert, key, EVP_sha256()) > 0, "test certificate signature");
        check(SSL_CTX_use_certificate(context, cert) && SSL_CTX_use_PrivateKey(context, key), "test server certificate");
        char path[] = "/tmp/ezlocalai-tunnel-cert-XXXXXX"; int fd = mkstemp(path); check(fd >= 0, "temporary certificate");
        FILE *output = fdopen(fd, "w"); check(output && PEM_write_X509(output, cert), "write certificate"); fclose(output); file = path;
        X509_free(cert); EVP_PKEY_free(key);
    }
    ~Certificate() { SSL_CTX_free(context); unlink(file.c_str()); }
};
struct Peer {
    int listener = -1, port = 0;
    std::atomic<bool> stop{false};
    std::thread thread;
    Peer(const std::string &mode, Certificate &certificate) {
        listener = socket(AF_INET, SOCK_STREAM, 0); check(listener >= 0, "listener");
        sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_ANY);
        check(bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0, "bind");
        socklen_t length = sizeof(address); getsockname(listener, reinterpret_cast<sockaddr *>(&address), &length); port = ntohs(address.sin_port);
        check(listen(listener, 4) == 0, "listen");
        thread = std::thread([&, mode] {
            int fd = accept(listener, nullptr, nullptr); if (fd < 0) return;
            timeval timeout{20, 0}; setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            SSL *ssl = nullptr;
            if (mode == "tls-timeout" || mode == "cancel-tls") {
                while (!stop) std::this_thread::sleep_for(5ms);
                close(fd); return;
            }
            if (mode.rfind("tls-", 0) == 0) {
                if (mode == "tls-delayed") std::this_thread::sleep_for(1300ms);
                ssl = SSL_new(certificate.context); SSL_set_fd(ssl, fd);
                if (SSL_accept(ssl) != 1) { SSL_free(ssl); close(fd); return; }
            }
            const auto read = [&](void *buffer, size_t size) -> ssize_t {
                ssize_t count;
                do { count = ssl ? SSL_read(ssl, buffer, static_cast<int>(size)) : recv(fd, buffer, size, 0); } while (count < 0 && errno == EINTR);
                return count;
            };
            const auto send = [&](const std::string &bytes) {
                for (size_t at = 0; at < bytes.size();) {
                    auto count = ssl ? SSL_write(ssl, bytes.data()+at, bytes.size()-at) : ::send(fd, bytes.data()+at, bytes.size()-at, MSG_NOSIGNAL);
                    if (count < 0 && errno == EINTR) continue;
                    if (count <= 0) return false;
                    at += count;
                }
                return true;
            };
            std::string request; char byte;
            while (request.find("\r\n\r\n") == std::string::npos && read(&byte, 1) == 1) request += byte;
            const auto at = request.find("Sec-WebSocket-Key: ");
            if (at != std::string::npos) {
                const auto begin = at + 19; const auto key = request.substr(begin, request.find("\r\n", begin)-begin);
                if (mode == "cancel-upgrade") { while (!stop) std::this_thread::sleep_for(5ms); }
                else {
                    if (mode == "delayed-upgrade") std::this_thread::sleep_for(1300ms);
                    const auto response = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: " + ezlocalai::tunnel_websocket_accept(key) + "\r\n\r\n";
                    if (mode == "dripped-upgrade") { send(response.substr(0,20)); std::this_thread::sleep_for(650ms); send(response.substr(20,30)); std::this_thread::sleep_for(650ms); send(response.substr(50)); }
                    else send(response);
                    if (mode == "slow-reader" || mode == "cancel-write" || mode == "duplex" || mode == "tls-duplex") {
                        if (mode == "duplex" || mode == "tls-duplex") { std::this_thread::sleep_for(300ms); send(std::string("\x89\x01p",3)); std::this_thread::sleep_for(1000ms); }
                        else std::this_thread::sleep_for(1300ms);
                        if (mode != "cancel-write") {
                            size_t total = 0; char buffer[65536];
                            while (total < 8*1024*1024+14) { const auto count = read(buffer, sizeof(buffer)); check(count > 0, "slow reader receives complete masked frame"); total += count; }
                            check(total == 8*1024*1024+14, "masked upload size");
                        }
                    }
                    while (!stop) std::this_thread::sleep_for(5ms);
                }
            }
            if (ssl) SSL_free(ssl); close(fd);
        });
    }
    ~Peer() { stop = true; shutdown(listener, SHUT_RDWR); thread.join(); close(listener); }
};
int main(int argc, char **argv) {
    check(argc == 2, "fixture argument"); signal(SIGPIPE, SIG_IGN);
    std::ifstream input(argv[1]); const auto reference = Json::parse(input);
    Certificate certificate;
    for (const auto &row : reference.at("transport")) {
        const auto mode = row.at("name").get<std::string>();
        Peer peer(mode, certificate);
        if (mode == "tls-untrusted") unsetenv("SSL_CERT_FILE"); else setenv("SSL_CERT_FILE", certificate.file.c_str(), 1);
        const auto host = mode == "tls-dns" ? "localhost" : mode == "tls-wrong-ip" ? "127.0.0.2" : "127.0.0.1";
        const bool tls = mode.rfind("tls-", 0) == 0 || mode == "cancel-tls";
        const auto url = std::string(tls ? "wss://" : "ws://") + host + ":" + std::to_string(peer.port) + "/test";
        std::atomic<bool> cancel{false}; std::thread cancellation;
        if (mode.rfind("cancel-", 0) == 0) cancellation = std::thread([&] { std::this_thread::sleep_for(150ms); cancel = true; });
        ezlocalai::TunnelWebSocketConnection connection([&] { return cancel.load(); }); std::string buffer, error;
        const auto start = ezlocalai::http_monotonic_time(); const bool accepted = connection.connect_to(url, "worker", "", &buffer, &error);
        const double elapsed = ezlocalai::http_monotonic_time()-start;
        if (cancellation.joinable()) cancellation.join();
        check(accepted == row.at("accepted").get<bool>(), mode + " source acceptance: " + error);
        if (mode == "tls-timeout") check(elapsed >= 14.9 && elapsed < 16.5, "shared 15-second socket budget");
        if (mode.rfind("cancel-", 0) == 0) check(elapsed >= .1 && elapsed < .7, "setup cancellation");
        if (mode == "delayed-upgrade" || mode == "dripped-upgrade") check(elapsed >= 1.2, "delayed upgrade waited");
        connection.close_all();
        std::cout << mode << " accepted=" << accepted << " elapsed=" << elapsed << '\n';
    }
    setenv("SSL_CERT_FILE", certificate.file.c_str(), 1);
    for (auto mode : {"tls-delayed", "slow-reader", "cancel-write"}) {
        Peer peer(mode, certificate); std::atomic<bool> cancel{false};
        ezlocalai::TunnelWebSocketConnection connection([&] { return cancel.load(); }); std::string buffer, error;
        const auto started = ezlocalai::http_monotonic_time();
        check(connection.connect_to(std::string(mode == std::string("tls-delayed") ? "wss" : "ws") + "://127.0.0.1:" + std::to_string(peer.port), "worker", "", &buffer, &error), mode + std::string(": ") + error);
        if (mode == std::string("tls-delayed")) check(ezlocalai::http_monotonic_time()-started >= 1.2, "delayed TLS handshake succeeds");
        else {
            std::thread cancellation; if (mode == std::string("cancel-write")) cancellation = std::thread([&] { std::this_thread::sleep_for(150ms); cancel = true; });
            const auto start = ezlocalai::http_monotonic_time(); const bool sent = connection.send_text(std::string(8*1024*1024,'x')); const auto elapsed = ezlocalai::http_monotonic_time()-start;
            if (cancellation.joinable()) cancellation.join();
            check(sent == (mode == std::string("slow-reader")), "backpressure write result");
            check(sent ? elapsed >= 1.1 : elapsed < .8, "backpressure wait / cancellation time");
            std::cout << mode << " sent=" << sent << " elapsed=" << elapsed << '\n';
        }
    }
    for (auto mode : {"duplex", "tls-duplex"}) {
        Peer peer(mode, certificate); ezlocalai::TunnelWebSocketConnection connection;
        std::string buffer, error;
        check(connection.connect_to(std::string(mode == std::string("tls-duplex") ? "wss" : "ws") + "://127.0.0.1:" + std::to_string(peer.port), "worker", "", &buffer, &error), "duplex connection");
        connection.queue_frame(1, std::string(8*1024*1024,'x'));
        const auto start = ezlocalai::http_monotonic_time(); bool received = false; int pending_slices = 0;
        while (connection.writing() && ezlocalai::http_monotonic_time()-start < 5) {
            const auto status = connection.flush_frame(); check(status >= 0,"incremental write succeeds"); if (!status) ++pending_slices;
            if (connection.readable(0)) { char data[128]; const auto count = connection.read_some(data,sizeof(data)); if (count > 0) buffer.append(data,count); }
            if (!received && buffer.find(std::string("\x89\x01p",3)) != std::string::npos) {
                received = true; check(connection.writing() && ezlocalai::http_monotonic_time()-start < 1,"incoming control received during blocked write");
            }
        }
        check(!connection.writing() && pending_slices > 10 && received,"duplex write completes across retries without starving reads");
        std::cout << mode << " pending_slices=" << pending_slices << " elapsed=" << ezlocalai::http_monotonic_time()-start << '\n';
    }
    {
        std::atomic<bool> cancel{false}; ezlocalai::TunnelWebSocketConnection connection([&] { return cancel.load(); });
        std::thread cancellation([&] { std::this_thread::sleep_for(50ms); cancel = true; });
        const auto start = ezlocalai::http_monotonic_time(); std::string buffer, error;
        check(!connection.connect_to("ws://cancelled.tunnel.test", "worker", "", &buffer, &error), "DNS cancelled");
        check(ezlocalai::http_monotonic_time()-start < .3 && !resolver_done, "shutdown does not await OS resolver"); cancellation.join();
        while (!resolver_done) std::this_thread::sleep_for(5ms);
    }
    std::cout << "9 source network outcomes and delayed TLS, slow upload, write/DNS cancellation and plain/TLS duplex backpressure passed\n";
}
