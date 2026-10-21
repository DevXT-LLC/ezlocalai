#include "ezlocalai/tunnel_connection.hpp"
#include <nlohmann/json.hpp>
#include <openssl/pem.h>
#include <openssl/ssl.h>
#include <openssl/x509v3.h>
#include <arpa/inet.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <poll.h>
#include <sstream>
#include <thread>
#include <vector>
#include <unistd.h>
using Json = nlohmann::json;
using namespace std::chrono_literals;
namespace {
void check(bool ok, const std::string &message) {
    if (!ok)
        throw std::runtime_error(message);
}
struct Certificate {
    SSL_CTX *context = SSL_CTX_new(TLS_server_method());
    std::string file;
    Certificate() {
        EVP_PKEY *key = EVP_PKEY_Q_keygen(nullptr, nullptr, "EC", "P-256");
        check(context && key, "test TLS key");
        X509 *cert = X509_new();
        X509_set_version(cert, 2);
        ASN1_INTEGER_set(X509_get_serialNumber(cert), 1);
        X509_gmtime_adj(X509_getm_notBefore(cert), -60);
        X509_gmtime_adj(X509_getm_notAfter(cert), 3600);
        X509_set_pubkey(cert, key);
        auto *name = X509_get_subject_name(cert);
        X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC,
                                   reinterpret_cast<const unsigned char *>("localhost"), -1, -1, 0);
        X509_set_issuer_name(cert, name);
        X509V3_CTX extensions{};
        X509V3_set_ctx(&extensions, cert, cert, nullptr, nullptr, 0);
        for (const auto &item :
             {std::pair<int, const char *>{NID_subject_alt_name, "DNS:localhost,IP:127.0.0.1"},
              {NID_basic_constraints, "critical,CA:TRUE"}}) {
            auto *extension =
                X509V3_EXT_conf_nid(nullptr, &extensions, item.first, const_cast<char *>(item.second));
            check(extension && X509_add_ext(cert, extension, -1), "TLS extension");
            X509_EXTENSION_free(extension);
        }
        check(X509_sign(cert, key, EVP_sha256()) > 0, "TLS certificate signing");
        check(SSL_CTX_use_certificate(context, cert) && SSL_CTX_use_PrivateKey(context, key),
              "TLS certificate");
        char path[] = "/tmp/ezlocalai-redirect-cert-XXXXXX";
        const auto fd = mkstemp(path);
        check(fd >= 0, "TLS temporary certificate");
        auto *output = fdopen(fd, "w");
        check(output && PEM_write_X509(output, cert), "TLS certificate file");
        fclose(output);
        file = path;
        X509_free(cert);
        EVP_PKEY_free(key);
    }
    ~Certificate() {
        SSL_CTX_free(context);
        unlink(file.c_str());
    }
};
struct Fixture {
    const Json &row;
    Certificate &certificate;
    std::array<int, 3> listeners{}, ports{};
    std::array<std::thread, 3> threads;
    std::atomic<bool> stopped{false};
    std::mutex mutex;
    Json requests = Json::array();
    std::vector<std::string> keys;
    std::string failure;
    Fixture(const Json &scenario, Certificate &cert) : row(scenario), certificate(cert) {
        for (size_t i = 0; i < listeners.size(); ++i) {
            listeners[i] = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
            sockaddr_in address{};
            address.sin_family = AF_INET;
            address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            check(listeners[i] >= 0 &&
                      bind(listeners[i], reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0,
                  "redirect listener");
            socklen_t size = sizeof(address);
            getsockname(listeners[i], reinterpret_cast<sockaddr *>(&address), &size);
            ports[i] = ntohs(address.sin_port);
            check(listen(listeners[i], 8) == 0, "listen");
        }
        for (size_t i = 0; i < threads.size(); ++i)
            threads[i] = std::thread([this, i] { serve(i); });
    }
    ~Fixture() {
        stopped = true;
        for (auto fd : listeners)
            shutdown(fd, SHUT_RDWR);
        for (auto &thread : threads)
            thread.join();
        for (auto fd : listeners)
            close(fd);
    }
    std::string expand(std::string value) const {
        for (size_t i = 0; i < ports.size(); ++i) {
            const auto token = std::string("{") + std::array<const char *, 3>{"a", "b", "tls"}[i] + "}";
            for (size_t at; (at = value.find(token)) != std::string::npos;)
                value.replace(at, token.size(), std::to_string(ports[i]));
        }
        return value;
    }
    std::string url() const {
        const auto scheme = row.value("scheme", "ws");
        return scheme + "://" + row.value("userinfo", "") + row.value("host", "localhost") + ":" +
               std::to_string(ports[scheme == "https" || scheme == "wss" ? 2 : 0]) + "/start/entry" +
               (row.contains("query") ? "?" + row.at("query").get<std::string>() : "");
    }
    void serve(size_t index) {
        while (!stopped) {
            const auto fd = accept4(listeners[index], nullptr, nullptr, SOCK_CLOEXEC);
            if (fd < 0) {
                if (errno == EINTR)
                    continue;
                break;
            }
            timeval timeout{3, 0};
            setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
            SSL *ssl = index == 2 ? SSL_new(certificate.context) : nullptr;
            try {
                if (ssl) {
                    SSL_set_fd(ssl, fd);
                    check(SSL_accept(ssl) == 1, "fixture TLS handshake");
                }
                const auto read = [&](char *data, size_t count) -> ssize_t {
                    return ssl ? SSL_read(ssl, data, static_cast<int>(count)) : recv(fd, data, count, 0);
                };
                std::string raw;
                char buffer[4096];
                while (raw.find("\r\n\r\n") == std::string::npos) {
                    const auto n = read(buffer, sizeof(buffer));
                    if (n < 0 && errno == EINTR)
                        continue;
                    check(n > 0, "fixture request header");
                    raw.append(buffer, n);
                }
                std::istringstream input(raw);
                std::string method, target, version, line;
                input >> method >> target >> version;
                std::getline(input, line);
                std::map<std::string, std::string> headers;
                while (std::getline(input, line)) {
                    const auto colon = line.find(':');
                    if (colon == std::string::npos)
                        continue;
                    auto name = line.substr(0, colon), value = line.substr(colon + 1);
                    for (auto &c : name)
                        c = std::tolower(static_cast<unsigned char>(c));
                    while (!value.empty() && (value.back() == '\r' || value.back() == ' '))
                        value.pop_back();
                    while (!value.empty() && value.front() == ' ')
                        value.erase(0, 1);
                    headers.emplace(name, value);
                }
                size_t step;
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    requests.push_back(Json{{"target", target},
                                            {"authorization", headers["authorization"]},
                                            {"cookie", headers["cookie"]}});
                    keys.push_back(headers["sec-websocket-key"]);
                    step = requests.size();
                }
                if (row.value("stall_final", false) && step > 1) {
                    read(buffer, sizeof(buffer));
                } else {
                    std::string response;
                    if (step <= row.value("hops", 1u)) {
                        response = "HTTP/1.1 " + std::to_string(row.value("status", 302)) +
                                   " Redirect\r\nContent-Length: 0\r\nConnection: close\r\n";
                        for (const auto &field :
                             {std::pair<const char *, const char *>{"Location", "location"}, {"URI", "uri"}})
                            if (row.contains(field.second) && !row.at(field.second).is_null())
                                response +=
                                    std::string(field.first) + ": " + expand(row.at(field.second)) + "\r\n";
                        for (const auto &cookie : row.value("cookies", Json::array()))
                            response += "Set-Cookie: " + cookie.get<std::string>() + "\r\n";
                        response += "\r\n";
                    } else {
                        response = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: "
                                   "Upgrade\r\nSec-WebSocket-Accept: " +
                                   (row.value("bad_accept", false)
                                        ? "wrong"
                                        : ezlocalai::tunnel_websocket_accept(headers["sec-websocket-key"])) +
                                   "\r\n\r\n" + std::string("\x81\x05") + "hello";
                    }
                    for (size_t sent = 0; sent < response.size();) {
                        const auto n =
                            ssl ? SSL_write(ssl, response.data() + sent,
                                            static_cast<int>(response.size() - sent))
                                : send(fd, response.data() + sent, response.size() - sent, MSG_NOSIGNAL);
                        if (n < 0 && errno == EINTR)
                            continue;
                        check(n > 0, "fixture response write");
                        sent += static_cast<size_t>(n);
                    }
                }
            } catch (const std::exception &error) {
                std::lock_guard<std::mutex> lock(mutex);
                failure = error.what();
            }
            if (ssl) {
                SSL_shutdown(ssl);
                SSL_free(ssl);
            }
            close(fd);
        }
    }
};
} // namespace
int main(int argc, char **argv) {
    try {
        signal(SIGPIPE, SIG_IGN);
        check(argc == 2, "redirect fixture required");
        std::ifstream file(argv[1]);
        const auto fixture = Json::parse(file);
        Certificate certificate;
        const char *previous = getenv("SSL_CERT_FILE");
        const std::optional<std::string> prior =
            previous ? std::optional<std::string>{previous} : std::nullopt;
        struct Restore {
            std::optional<std::string> value;
            ~Restore() {
                if (value)
                    setenv("SSL_CERT_FILE", value->c_str(), 1);
                else
                    unsetenv("SSL_CERT_FILE");
            }
        } restore{prior};
        setenv("SSL_CERT_FILE", certificate.file.c_str(), 1);
        for (const auto &row : fixture.at("cases")) {
            Fixture server(row, certificate);
            ezlocalai::TunnelWebSocketConnection connection;
            std::string buffer, error;
            const bool accepted = connection.connect_to(server.url(), "worker id",
                                                        row.value("key", "router-secret"), &buffer, &error);
            check(accepted == row.at("accepted"),
                  row.at("name").get<std::string>() + ": acceptance: " + error);
            if (accepted) {
                const auto deadline = std::chrono::steady_clock::now() + 2s;
                while (buffer.size() < 7 && std::chrono::steady_clock::now() < deadline) {
                    char data[64];
                    const auto n = connection.read_some(data, sizeof(data));
                    if (n > 0)
                        buffer.append(data, n);
                    else if (!n)
                        break;
                }
                check(buffer == std::string("\x81\x05") + "hello", "upgrade preserves initial frame");
            }
            connection.close_all();
            std::lock_guard<std::mutex> lock(server.mutex);
            check(server.failure.empty(), server.failure);
            check(server.requests == row.at("requests"),
                  row.at("name").get<std::string>() + ": requests: " + server.requests.dump());
            const bool same_key = server.keys.empty() ||
                                  std::all_of(server.keys.begin(), server.keys.end(),
                                              [&](const auto &key) { return key == server.keys.front(); });
            check(same_key == row.at("same_websocket_key"), "one challenge across redirect chain");
        }
        {
            Json row{{"location", "/final"}, {"stall_final", true}};
            Fixture server(row, certificate);
            std::atomic<bool> stop{false};
            ezlocalai::TunnelWebSocketConnection connection([&] { return stop.load(); });
            std::thread cancel([&] {
                std::this_thread::sleep_for(300ms);
                stop = true;
            });
            std::string buffer, error;
            const auto start = std::chrono::steady_clock::now();
            const bool accepted =
                connection.connect_to(server.url(), "worker id", "router-secret", &buffer, &error);
            cancel.join();
            check(!accepted && std::chrono::steady_clock::now() - start < 1500ms,
                  "cancel redirected upgrade");
            std::lock_guard<std::mutex> lock(server.mutex);
            check(server.requests.size() == 2, "cancel after following redirect");
        }
        std::cout
            << fixture.at("cases").size()
            << " aiohttp redirect traces matched over TCP/TLS; redirected upgrade cancellation passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
