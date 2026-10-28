#include "ezlocalai/tunnel_connection.hpp"
#include <openssl/evp.h>
#include <openssl/sha.h>
#include "ezlocalai/http_deadline.hpp"
#include "ezlocalai/http_cookies.hpp"
#include <arpa/inet.h>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <poll.h>
#include <random>
#include <unistd.h>
#include <map>
#include <sstream>
#include <stdexcept>

namespace ezlocalai {
namespace {
std::string lower(std::string value) {
    for (auto &c : value) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    return value;
}
std::string trim(const std::string &value) {
    const auto a = value.find_first_not_of(" \t\r\n"), b = value.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : value.substr(a, b - a + 1);
}
std::string base64(const unsigned char *bytes, size_t size) {
    std::string result(4 * ((size + 2) / 3), '\0');
    EVP_EncodeBlock(reinterpret_cast<unsigned char *>(result.data()), bytes, static_cast<int>(size));
    return result;
}
std::string query_value(const std::string &value) {
    const std::string safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~!$'()*,/:?@";
    static const char hex[] = "0123456789ABCDEF";
    std::string result;
    for (unsigned char c : value) {
        if (c == ' ') result += '+';
        else if (safe.find(c) != std::string::npos) result += static_cast<char>(c);
        else { result += '%'; result += hex[c >> 4]; result += hex[c & 15]; }
    }
    return result;
}
}
TunnelConnectionUrl tunnel_connection_url(const std::string &value, const std::string &worker_id, const std::string &api_key) {
    TunnelConnectionUrl result;
    result.url = HttpUrl::parse(value);
    if (result.url.scheme == "ws") result.url.scheme = "http";
    if (result.url.scheme == "wss") result.url.scheme = "https";
    result.url.require_http();
    const bool bearer = !api_key.empty() && api_key != "none";
    if (auto credentials = result.url.credentials()) {
        if (bearer) throw std::invalid_argument("URL credentials and Authorization header cannot be combined");
        const auto plain = credentials->first + ":" + credentials->second;
        result.authorization = "Basic " + base64(reinterpret_cast<const unsigned char *>(plain.data()), plain.size());
    } else if (bearer) result.authorization = "Bearer " + api_key;
    result.url.remove_credentials();
    result.url.fragment.clear();
    if (!result.url.query.empty()) result.url.query += '&';
    result.url.query += "worker_id=" + query_value(worker_id);
    return result;
}
std::string tunnel_websocket_accept(const std::string &key) {
    const auto input = key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    unsigned char bytes[SHA_DIGEST_LENGTH];
    SHA1(reinterpret_cast<const unsigned char *>(input.data()), input.size(), bytes);
    return base64(bytes, sizeof(bytes));
}
bool tunnel_validate_upgrade(const std::string &headers, const std::string &key) {
    std::istringstream lines(headers);
    std::string line, version; int status = 0;
    if (!std::getline(lines, line)) return false;
    std::istringstream first(line);
    if (!(first >> version >> status) || (version != "HTTP/1.1" && version != "HTTP/1.0" && version != "HTTP/2.0") || status != 101) return false;
    std::map<std::string, std::string> fields;
    while (std::getline(lines, line)) {
        if (line == "\r" || line.empty()) break;
        const auto colon = line.find(':');
        if (colon == std::string::npos) return false;
        // aiohttp's CIMultiDict.get uses the first duplicate field.
        fields.emplace(lower(line.substr(0, colon)), trim(line.substr(colon + 1)));
    }
    return lower(fields["upgrade"]) == "websocket" && lower(fields["connection"]) == "upgrade"
        && fields["sec-websocket-accept"] == tunnel_websocket_accept(key);
}
std::string websocket_client_masked_frame(unsigned char opcode, const std::string &payload) {
    std::array<unsigned char, 4> mask{};
    std::random_device rd;
    for (auto &byte : mask) {
        byte = static_cast<unsigned char>(rd());
    }
    std::string frame;
    frame.push_back(static_cast<char>(0x80U | (opcode & 0x0fU)));
    const uint64_t len = payload.size();
    if (len < 126) {
        frame.push_back(static_cast<char>(0x80U | static_cast<unsigned char>(len)));
    } else if (len <= 0xffff) {
        frame.push_back(static_cast<char>(0x80U | 126U));
        frame.push_back(static_cast<char>((len >> 8) & 0xff));
        frame.push_back(static_cast<char>(len & 0xff));
    } else {
        frame.push_back(static_cast<char>(0x80U | 127U));
        for (int shift = 56; shift >= 0; shift -= 8) {
            frame.push_back(static_cast<char>((len >> shift) & 0xff));
        }
    }
    for (unsigned char byte : mask) {
        frame.push_back(static_cast<char>(byte));
    }
    for (size_t i = 0; i < payload.size(); ++i) {
        frame.push_back(static_cast<char>(static_cast<unsigned char>(payload[i]) ^ mask[i % mask.size()]));
    }
    return frame;
}

class TunnelWebSocketConnection::Impl {
public:
    explicit Impl(std::function<bool()> cancelled) : cancelled_(std::move(cancelled)) {}
    ~Impl() {
        close_all();
    }

    bool connect_to(
        const std::string &url,
        const std::string &worker_id,
        const std::string &api_key,
        std::string *initial_buffer,
        std::string *error,
        bool endpoint = false
    ) {
        close_all();
        bounded_upgrade_ = endpoint;
        if (initial_buffer) initial_buffer->clear();
        try {
            auto first = tunnel_connection_url(url, worker_id, api_key);
            if (endpoint) first.url.query = HttpUrl::parse(url).query;
            auto logical = HttpUrl::parse(url);
            logical.remove_credentials();
            logical.fragment.clear();
            auto target = std::move(first.url);
            auto authorization = std::move(first.authorization);
            bool header_authorization = !api_key.empty() && api_key != "none";
            HttpCookieJar cookies;
            std::array<unsigned char, 16> key_bytes{};
            std::random_device random;
            for (auto &byte : key_bytes) byte = static_cast<unsigned char>(random());
            const auto ws_key = base64(key_bytes.data(), key_bytes.size());
            unsigned redirects = 0;
            for (;;) {
                if (authorization.find_first_of("\r\n") != std::string::npos)
                    throw std::invalid_argument("Invalid websocket authorization");
                close_all();
                parsed_ = target;
                std::string raw;
                if (!connect_once(authorization, cookies.header(target), ws_key, &raw, error)) {
                    close_all();
                    return false;
                }
                const auto header_end = raw.find("\r\n\r\n");
                std::istringstream lines(raw.substr(0, header_end + 2));
                std::string line, version;
                int status = 0;
                std::getline(lines, line);
                std::istringstream status_line(line);
                if (!(status_line >> version >> status) ||
                    (version != "HTTP/1.1" && version != "HTTP/1.0" && version != "HTTP/2.0"))
                    throw std::invalid_argument("Invalid websocket response status");
                std::map<std::string, std::string> fields;
                std::vector<std::string> set_cookies;
                while (std::getline(lines, line)) {
                    if (line.empty() || line == "\r") break;
                    const auto colon = line.find(':');
                    if (colon == std::string::npos)
                        throw std::invalid_argument("Invalid websocket response header");
                    const auto name = lower(line.substr(0, colon));
                    const auto value = trim(line.substr(colon + 1));
                    if (name.size() > 8190 || value.size() > 8190)
                        throw std::invalid_argument("Websocket response header too large");
                    fields.emplace(name, value); // CIMultiDict.get uses the first field.
                    if (name == "set-cookie") set_cookies.push_back(value);
                }
                cookies.update_from_headers(set_cookies, target);
                if (status == 301 || status == 302 || status == 303 || status == 307 || status == 308) {
                    if (++redirects >= 10)
                        throw std::invalid_argument("Too many websocket redirects");
                    auto location = fields.find("location");
                    if (location == fields.end() || location->second.empty()) location = fields.find("uri");
                    if (location != fields.end()) {
                        auto reference = HttpUrl::parse(location->second);
                        // aiohttp permits relative references and absolute HTTP(S)
                        // redirects, but rejects absolute ws:/wss: redirect URLs.
                        if (!reference.scheme.empty() && reference.scheme != "http" && reference.scheme != "https")
                            throw std::invalid_argument("Non-HTTP websocket redirect");
                        if (reference.scheme == logical.scheme && !reference.authority) reference.scheme.clear();
                        auto next = logical.resolve(reference.str());
                        auto transport = next;
                        if (transport.scheme == "ws") transport.scheme = "http";
                        if (transport.scheme == "wss") transport.scheme = "https";
                        transport.require_http();
                        // Retain the logical ws/wss scheme for origin comparison:
                        // ws://host -> http://host also removes Authorization.
                        if (logical.scheme != next.scheme || !target.same_origin(transport)) {
                            authorization.clear();
                            header_authorization = false;
                        }
                        if (const auto credentials = next.credentials()) {
                            if (header_authorization)
                                throw std::invalid_argument("Conflicting websocket authorization");
                            const auto plain = credentials->first + ":" + credentials->second;
                            authorization = "Basic " + base64(reinterpret_cast<const unsigned char *>(plain.data()), plain.size());
                        }
                        next.remove_credentials(); transport.remove_credentials();
                        next.fragment.clear(); transport.fragment.clear();
                        logical = std::move(next);
                        target = std::move(transport);
                        // worker_id is an initial request parameter, not a query
                        // parameter to add again to each redirect destination.
                        continue;
                    }
                }
                if (!tunnel_validate_upgrade(raw.substr(0, header_end + 4), ws_key))
                    throw std::invalid_argument("Invalid websocket upgrade response");
                if (initial_buffer) *initial_buffer = raw.substr(header_end + 4);
                return true;
            }
        } catch (const std::exception &exception) {
            close_all();
            if (error) *error = exception.what();
            return false;
        }
    }

    bool connect_once(const std::string &authorization, const std::string &cookie,
                      const std::string &ws_key, std::string *response, std::string *error) {
        addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = IPPROTO_TCP;
        HttpDeadline resolving(0, false, cancelled_);
        const auto resolved = http_resolve(parsed_.host, std::to_string(parsed_.port()), hints, resolving);
        if (!resolved.addresses) {
            if (error) *error = "websocket DNS resolution failed or cancelled";
            return false;
        }
        // aiohttp's sock_connect budget covers TCP and TLS together, after DNS.
        HttpDeadline connecting(15, true, cancelled_);
        for (addrinfo *item = resolved.addresses.get(); item; item = item->ai_next) {
            fd_ = socket(item->ai_family, item->ai_socktype, item->ai_protocol);
            if (fd_ < 0) {
                continue;
            }
            if (http_connect(fd_, item->ai_addr, item->ai_addrlen, connecting)) {
                break;
            }
            close(fd_);
            fd_ = -1;
        }
        if (fd_ < 0) {
            if (error) {
                *error = "connect failed";
            }
            return false;
        }
        if (parsed_.scheme == "https") {
            OPENSSL_init_ssl(0, nullptr);
            ssl_ctx_ = SSL_CTX_new(TLS_client_method());
            if (!ssl_ctx_) {
                if (error) {
                    *error = "failed to create TLS context";
                }
                return false;
            }
            SSL_CTX_set_min_proto_version(ssl_ctx_, TLS1_2_VERSION);
            SSL_CTX_set_verify(ssl_ctx_, SSL_VERIFY_PEER, nullptr);
            SSL_CTX_set_default_verify_paths(ssl_ctx_);
            const char *cert_file = getenv("SSL_CERT_FILE");
            const char *cert_dir = getenv("SSL_CERT_DIR");
            if ((cert_file && *cert_file) || (cert_dir && *cert_dir)) {
                SSL_CTX_load_verify_locations(
                    ssl_ctx_,
                    cert_file && *cert_file ? cert_file : nullptr,
                    cert_dir && *cert_dir ? cert_dir : nullptr
                );
            }
            ssl_ = SSL_new(ssl_ctx_);
            if (!ssl_) {
                if (error) {
                    *error = "failed to create TLS connection";
                }
                return false;
            }
            in6_addr numeric{};
            const bool ip = inet_pton(AF_INET, parsed_.host.c_str(), &numeric) == 1 || inet_pton(AF_INET6, parsed_.host.c_str(), &numeric) == 1;
            if (!ip) {
                SSL_set_tlsext_host_name(ssl_, parsed_.host.c_str());
                X509_VERIFY_PARAM_set1_host(SSL_get0_param(ssl_), parsed_.host.c_str(), 0);
            } else {
                X509_VERIFY_PARAM_set1_ip_asc(SSL_get0_param(ssl_), parsed_.host.c_str());
            }
            SSL_set_fd(ssl_, fd_);
            if (!http_tls_connect(ssl_, fd_, connecting)) {
                if (error) {
                    *error = "failed to establish TLS";
                }
                return false;
            }
            const long verify_result = SSL_get_verify_result(ssl_);
            if (verify_result != X509_V_OK) {
                if (error) {
                    *error = X509_verify_cert_error_string(verify_result);
                }
                return false;
            }
        }

        const auto path = parsed_.request_target();
        const auto host_header = parsed_.host_header();
        std::ostringstream request;
        request << "GET " << path << " HTTP/1.1\r\n"
            << "Host: " << host_header << "\r\n"
            << "Upgrade: websocket\r\n"
            << "Connection: Upgrade\r\n"
            << "Sec-WebSocket-Version: 13\r\n"
            << "Sec-WebSocket-Key: " << ws_key << "\r\n";
        if (!authorization.empty()) {
            request << "Authorization: " << authorization << "\r\n";
        }
        if (!cookie.empty()) request << "Cookie: " << cookie << "\r\n";
        request << "\r\n";
        if (!send_all(request.str())) {
            if (error) {
                *error = "failed to send websocket handshake";
            }
            return false;
        }

        HttpDeadline upgrading(bounded_upgrade_ ? 15 : 0, false, cancelled_);
        std::string raw;
        char buffer[4096];
        size_t header_end = std::string::npos;
        while ((header_end = raw.find("\r\n\r\n")) == std::string::npos) {
            const ssize_t n = http_read(fd_, ssl_, buffer, sizeof(buffer), upgrading);
            if (n <= 0) {
                if (error) {
                    *error = "websocket handshake closed, failed or cancelled";
                }
                return false;
            }
            raw.append(buffer, static_cast<size_t>(n));
            if (raw.size() > 65536) {
                if (error) {
                    *error = "websocket handshake too large";
                }
                return false;
            }
        }
        *response = std::move(raw);
        return true;
    }

    bool send_text(const std::string &payload) {
        return send_all(websocket_client_masked_frame(0x1U, payload));
    }

    bool send_pong(const std::string &payload = "") {
        return send_all(websocket_client_masked_frame(0xAU, payload));
    }

    bool readable(int milliseconds) const {
        if (ssl_ && SSL_pending(ssl_) > 0) return true;
        pollfd descriptor{fd_, POLLIN, 0};
        return poll(&descriptor, 1, milliseconds) > 0;
    }

    void queue_frame(unsigned char opcode, const std::string &payload) {
        if (!pending_.empty()) throw std::logic_error("WebSocket write already pending");
        pending_ = websocket_client_masked_frame(opcode, payload);
        sent_ = 0;
    }
    bool writing() const { return !pending_.empty(); }
    void queue_raw(std::string bytes) {
        if (writing()) throw std::logic_error("WebSocket write already pending");
        pending_ = std::move(bytes); sent_ = 0;
    }
    int flush_frame() {
        HttpDeadline slice(.005, false, cancelled_);
        while (sent_ < pending_.size()) {
            const auto count = http_write(fd_, ssl_, pending_.data() + sent_, pending_.size() - sent_, slice);
            if (count < 0) return errno == ETIMEDOUT ? 0 : -1;
            if (!count) return -1;
            sent_ += static_cast<size_t>(count);
        }
        pending_.clear(); sent_ = 0;
        return 1;
    }
    bool send_close(int code = 1000) {
        // Never append a close frame inside an unfinished data frame or retry
        // an OpenSSL write with a different buffer. TCP teardown is safe here.
        if (writing()) return false;
        std::string payload;
        payload += static_cast<char>((code >> 8) & 255);
        payload += static_cast<char>(code & 255);
        // Best-effort graceful close must still run after cancellation.
        HttpDeadline closing(1, false);
        return send_all(websocket_client_masked_frame(0x8U, payload), closing);
    }

    ssize_t read_some(char *buffer, size_t size) {
        // An incomplete TLS record must not block dispatch of queued responses.
        HttpDeadline reading(.02, false, cancelled_);
        const auto count = http_read(fd_, ssl_, buffer, size, reading);
        if (count >= 0) return count;
        return errno == ETIMEDOUT || errno == ECANCELED ? -1 : 0;
    }

    void close_all() {
        pending_.clear(); sent_ = 0;
        if (ssl_) {
            SSL_free(ssl_);
            ssl_ = nullptr;
        }
        if (ssl_ctx_) {
            SSL_CTX_free(ssl_ctx_);
            ssl_ctx_ = nullptr;
        }
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }

private:
    HttpUrl parsed_;
    bool bounded_upgrade_ = false;
    int fd_ = -1;
    SSL_CTX *ssl_ctx_ = nullptr;
    SSL *ssl_ = nullptr;

    std::function<bool()> cancelled_;
    std::string pending_;
    size_t sent_ = 0;
    bool send_all(const std::string &data) {
        HttpDeadline writing(0, false, cancelled_);
        return send_all(data, writing);
    }
    bool send_all(const std::string &data, const HttpDeadline &deadline) {
        size_t sent = 0;
        while (sent < data.size()) {
            const auto n = http_write(fd_, ssl_, data.data() + sent, data.size() - sent, deadline);
            if (n <= 0) return false;
            sent += static_cast<size_t>(n);
        }
        return true;
    }
};
TunnelWebSocketConnection::TunnelWebSocketConnection(std::function<bool()> cancelled) : impl_(std::make_unique<Impl>(std::move(cancelled))) {}
TunnelWebSocketConnection::~TunnelWebSocketConnection() = default;
bool TunnelWebSocketConnection::connect_endpoint(const std::string &url, const std::string &key, std::string *buffer, std::string *error) { return impl_->connect_to(url, "", key, buffer, error, true); }
void TunnelWebSocketConnection::queue_raw(std::string bytes) { impl_->queue_raw(std::move(bytes)); }
bool TunnelWebSocketConnection::connect_to(const std::string &url, const std::string &worker, const std::string &key, std::string *buffer, std::string *error) { return impl_->connect_to(url, worker, key, buffer, error); }
bool TunnelWebSocketConnection::send_text(const std::string &payload) { return impl_->send_text(payload); }
bool TunnelWebSocketConnection::send_pong(const std::string &payload) { return impl_->send_pong(payload); }
bool TunnelWebSocketConnection::send_close(int code) { return impl_->send_close(code); }
bool TunnelWebSocketConnection::readable(int milliseconds) const { return impl_->readable(milliseconds); }
ssize_t TunnelWebSocketConnection::read_some(char *buffer, size_t size) { return impl_->read_some(buffer, size); }
void TunnelWebSocketConnection::close_all() { impl_->close_all(); }
void TunnelWebSocketConnection::queue_frame(unsigned char opcode, const std::string &payload) { impl_->queue_frame(opcode, payload); }
int TunnelWebSocketConnection::flush_frame() { return impl_->flush_frame(); }
bool TunnelWebSocketConnection::writing() const { return impl_->writing(); }

}
