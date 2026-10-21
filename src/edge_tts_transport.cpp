#include "ezlocalai/edge_tts.hpp"
#include "ezlocalai/http_deadline.hpp"
#include "ezlocalai/http_url.hpp"
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <zlib.h>
#include <unicode/utf8.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <regex>
#include <sstream>
#include <unistd.h>
namespace ezlocalai {
namespace {
std::string lower(std::string value) { for (auto &c : value) if (c >= 'A' && c <= 'Z') c += 32; return value; }
std::string trim(const std::string &value) { const auto a = value.find_first_not_of(" \t\r\n"), b = value.find_last_not_of(" \t\r\n"); return a == std::string::npos ? "" : value.substr(a, b - a + 1); }
std::string env(const char *name, const char *upper) { const char *value = std::getenv(name); if (!value) value = std::getenv(upper); return value ? value : ""; }
std::string base64(const unsigned char *bytes, size_t size) { std::string result(4 * ((size + 2) / 3), '\0'); EVP_EncodeBlock(reinterpret_cast<unsigned char *>(result.data()), bytes, int(size)); return result; }
void random_bytes(unsigned char *bytes, size_t size) { if (RAND_bytes(bytes, int(size)) != 1) throw std::runtime_error("Edge random byte generation failed"); }
std::string identifier(bool uuid) {
    std::array<unsigned char, 16> bytes{}; random_bytes(bytes.data(), bytes.size());
    if (uuid) { bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128; }
    std::ostringstream out; if (!uuid) out << std::uppercase; for (auto b : bytes) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(b); return out.str();
}
double now() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }
[[noreturn]] void network_error(const std::string &message) { throw TtsError("WebSocketError", message); }
struct Connection {
    int fd = -1; SSL_CTX *context = nullptr; SSL *ssl = nullptr; std::string buffered;
    z_stream compressor{}; bool compress_ready = false, no_takeover = false;
    ~Connection() { if (compress_ready) deflateEnd(&compressor); if (ssl) SSL_free(ssl); if (context) SSL_CTX_free(context); if (fd >= 0) close(fd); }
    void compression(const std::string &extension) {
        int window = 15; std::smatch match;
        if (std::regex_search(extension, match, std::regex("client_max_window_bits=([0-9]+)"))) window = std::stoi(match[1]);
        if (window < 9 || window > 15 || deflateInit2(&compressor, Z_BEST_SPEED, Z_DEFLATED, -window, 8, Z_DEFAULT_STRATEGY) != Z_OK) network_error("Invalid Edge compression negotiation");
        compress_ready = true; no_takeover = extension.find("client_no_context_takeover") != std::string::npos;
    }
    void connect(const HttpUrl &url) {
        HttpDeadline deadline(10, false); addrinfo hints{}; hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
        const auto resolved = http_resolve(url.host, std::to_string(url.port()), hints, deadline);
        if (!resolved.addresses) network_error("Edge DNS resolution failed");
        for (auto item = resolved.addresses.get(); item; item = item->ai_next) {
            fd = socket(item->ai_family, item->ai_socktype, item->ai_protocol); if (fd < 0) continue;
            if (http_connect(fd, item->ai_addr, item->ai_addrlen, deadline)) return; close(fd); fd = -1;
        }
        network_error("Edge connection failed");
    }
    void tls(const std::string &host) {
        context = SSL_CTX_new(TLS_client_method()); if (!context) network_error("Edge TLS context failed");
        SSL_CTX_set_verify(context, SSL_VERIFY_PEER, nullptr); SSL_CTX_set_default_verify_paths(context);
        ssl = SSL_new(context); if (!ssl) network_error("Edge TLS connection failed");
        SSL_set_tlsext_host_name(ssl, host.c_str()); SSL_set1_host(ssl, host.c_str()); SSL_set_fd(ssl, fd);
        HttpDeadline deadline(10, false); if (!http_tls_connect(ssl, fd, deadline)) network_error("Edge TLS handshake failed");
    }
    void send(const std::string &bytes) {
        HttpDeadline deadline(60, false); size_t offset = 0;
        while (offset < bytes.size()) { const auto count = http_write(fd, ssl, bytes.data() + offset, bytes.size() - offset, deadline); if (count <= 0) network_error("Edge write failed"); offset += count; }
    }
    bool receive() {
        std::array<char, 16384> buffer{}; HttpDeadline deadline(60, false); const auto count = http_read(fd, ssl, buffer.data(), buffer.size(), deadline);
        if (count < 0) network_error("Edge read failed or timed out"); if (!count) return false; buffered.append(buffer.data(), size_t(count)); return true;
    }
    std::string take(size_t size) {
        while (buffered.size() < size) if (!receive()) network_error("Edge WebSocket connection closed mid-frame");
        auto result = buffered.substr(0, size); buffered.erase(0, size); return result;
    }
    EdgeTtsResponse response() {
        size_t end;
        while ((end = buffered.find("\r\n\r\n")) == std::string::npos) { if (buffered.size() > 65536) network_error("Edge handshake header too large"); if (!receive()) network_error("Edge handshake closed"); }
        std::istringstream lines(take(end + 4)); std::string line, version; std::getline(lines, line); std::istringstream status(line); EdgeTtsResponse result;
        if (!(status >> version >> result.status) || version.rfind("HTTP/", 0) != 0 || result.status < 100) network_error("Invalid Edge HTTP response");
        while (std::getline(lines, line)) { const auto colon = line.find(':'); if (colon != std::string::npos) result.headers[lower(line.substr(0, colon))] = trim(line.substr(colon + 1)); }
        return result;
    }
    void frame(unsigned opcode, const std::string &payload) {
        std::string data = payload; const bool compressed = compress_ready && opcode < 8;
        if (compressed) {
            compressor.next_in = reinterpret_cast<Bytef *>(data.data()); compressor.avail_in = uInt(data.size()); std::string output;
            do { std::array<char, 16384> buffer{}; compressor.next_out = reinterpret_cast<Bytef *>(buffer.data()); compressor.avail_out = buffer.size();
                if (deflate(&compressor, no_takeover ? Z_FULL_FLUSH : Z_SYNC_FLUSH) != Z_OK) network_error("Edge compression failed"); output.append(buffer.data(), buffer.size() - compressor.avail_out);
            } while (compressor.avail_in || !compressor.avail_out);
            if (output.size() >= 4 && output.compare(output.size() - 4, 4, std::string("\0\0\xff\xff", 4)) == 0) output.resize(output.size() - 4); data = std::move(output);
        }
        std::string bytes(1, char(128 | opcode | (compressed ? 64 : 0))); const auto size = data.size();
        if (size < 126) bytes += char(128 | size);
        else { const unsigned width = size <= 65535 ? 2 : 8; bytes += char(128 | (width == 2 ? 126 : 127)); for (int i = int(width) - 1; i >= 0; --i) bytes += char((uint64_t(size) >> (i * 8)) & 255); }
        std::array<unsigned char, 4> mask{}; random_bytes(mask.data(), mask.size()); for (auto byte : mask) bytes += char(byte);
        for (size_t i = 0; i < size; ++i) bytes += char(uint8_t(data[i]) ^ mask[i % 4]); send(bytes);
    }
};
bool bypass(const HttpUrl &url) {
    std::istringstream input(lower(env("no_proxy", "NO_PROXY"))); const auto host = lower(url.host), authority = lower(url.host_header());
    for (std::string item; std::getline(input, item, ',');) { item = trim(item); if (item == "*") return true; if (!item.empty() && item[0] == '.') item.erase(0, 1);
        if (item == host || item == authority || (!item.empty() && host.size() > item.size() && host.compare(host.size() - item.size(), item.size(), item) == 0 && host[host.size() - item.size() - 1] == '.')) return true; }
    return false;
}
double server_time(const std::string &date) {
    std::tm tm{}; std::istringstream input(date); input.imbue(std::locale::classic()); input >> std::get_time(&tm, "%a, %d %b %Y %H:%M:%S"); std::string zone; input >> zone;
    if (input.fail() || (zone != "GMT" && zone != "UTC")) throw TtsError("SkewAdjustmentError", "Failed to parse server date: " + date);
    const auto original = tm; const auto time = timegm(&tm);
    if (tm.tm_year != original.tm_year || tm.tm_mon != original.tm_mon || tm.tm_mday != original.tm_mday) throw TtsError("SkewAdjustmentError", "Failed to parse server date: " + date);
    return double(time);
}
}
EdgeTtsResponse edge_tts_exchange(const EdgeTtsRequest &request, EdgeTtsStream &stream) {
    const bool secure = request.url.rfind("wss://", 0) == 0; auto url = HttpUrl::parse((secure ? "https://" : "http://") + request.url.substr(secure ? 6 : 5));
    Connection connection; auto proxy = bypass(url) ? "" : secure ? env("wss_proxy", "WSS_PROXY") : env("ws_proxy", "WS_PROXY");
    if (!proxy.empty()) {
        auto server = HttpUrl::parse(proxy); if (server.scheme != "http") throw TtsError("WebSocketError", "Unsupported Edge proxy scheme"); connection.connect(server);
        std::string connect = "CONNECT " + url.host + ":" + std::to_string(url.port()) + " HTTP/1.1\r\nHost: " + url.host_header() + "\r\n";
        if (const auto auth = server.credentials()) { const auto credentials = auth->first + ":" + auth->second; connect += "Proxy-Authorization: Basic " + base64(reinterpret_cast<const unsigned char *>(credentials.data()), credentials.size()) + "\r\n"; }
        connection.send(connect + "\r\n"); const auto result = connection.response(); if (result.status != 200) network_error("Edge proxy CONNECT failed");
    } else connection.connect(url);
    if (secure) connection.tls(url.host);
    std::array<unsigned char, 16> key{}; random_bytes(key.data(), key.size()); const auto encoded = base64(key.data(), key.size());
    std::string handshake = "GET " + url.request_target() + " HTTP/1.1\r\nHost: " + url.host_header() + "\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: " + encoded + "\r\nSec-WebSocket-Extensions: permessage-deflate; client_max_window_bits\r\n";
    for (const auto &[name, value] : request.headers) handshake += name + ": " + value + "\r\n";
    connection.send(handshake + "\r\n"); const auto response = connection.response(); if (response.status != 101) return response;
    const auto accept_input = encoded + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"; unsigned char digest[SHA_DIGEST_LENGTH]; SHA1(reinterpret_cast<const unsigned char *>(accept_input.data()), accept_input.size(), digest);
    auto field = [&](const char *key) { const auto it = response.headers.find(key); return it == response.headers.end() ? std::string() : it->second; };
    if (field("sec-websocket-accept") != base64(digest, sizeof(digest)) || lower(field("upgrade")) != "websocket" || lower(field("connection")).find("upgrade") == std::string::npos) network_error("Invalid Edge WebSocket handshake");
    const auto extensions = lower(field("sec-websocket-extensions")); const bool compression = extensions.find("permessage-deflate") != std::string::npos;
    if (compression) connection.compression(extensions);
    struct Inflater { z_stream state{}; bool ready = false; ~Inflater() { if (ready) inflateEnd(&state); } } inflater;
    for (const auto &message : request.messages) connection.frame(1, message);
    unsigned message_type = 0; bool compressed = false; std::string message;
    for (;;) {
        if (connection.buffered.empty() && !connection.receive()) { if (message_type) network_error("Edge WebSocket closed during fragmented message"); break; }
        const auto head = connection.take(2); const auto first = uint8_t(head[0]), second = uint8_t(head[1]); const unsigned opcode = first & 15; const bool final = first & 128;
        if ((first & 48) || (second & 128) || ((first & 64) && (!compression || opcode == 0 || opcode >= 8))) network_error("Invalid Edge WebSocket frame flags");
        uint64_t length = second & 127;
        if (length >= 126) { const auto extended = connection.take(length == 126 ? 2 : 8); length = 0; for (unsigned char c : extended) length = (length << 8) | c; }
        if (length > 4 * 1024 * 1024 || (opcode >= 8 && (!final || length > 125))) network_error("Invalid Edge WebSocket frame size");
        const auto payload = connection.take(size_t(length));
        if (opcode == 8) { connection.frame(8, payload); break; }
        if (opcode == 9) { connection.frame(10, payload); continue; } if (opcode == 10) continue;
        if (opcode == 1 || opcode == 2) { if (message_type) network_error("Nested Edge WebSocket message"); message_type = opcode; compressed = first & 64; }
        else if (opcode != 0 || !message_type) network_error("Unexpected Edge WebSocket continuation");
        message += payload; if (message.size() > 4 * 1024 * 1024) network_error("Edge WebSocket message too large"); if (!final) continue;
        if (compressed) {
            if (!inflater.ready) { if (inflateInit2(&inflater.state, -15) != Z_OK) network_error("Edge decompressor initialization failed"); inflater.ready = true; }
            auto input = message + std::string("\0\0\xff\xff", 4); inflater.state.next_in = reinterpret_cast<Bytef *>(input.data()); inflater.state.avail_in = uInt(input.size()); message.clear();
            do { const auto before = inflater.state.avail_in; std::array<char, 16384> output{}; inflater.state.next_out = reinterpret_cast<Bytef *>(output.data()); inflater.state.avail_out = output.size(); const int status = inflate(&inflater.state, Z_SYNC_FLUSH);
                if (status != Z_OK && status != Z_BUF_ERROR && status != Z_STREAM_END) network_error("Invalid compressed Edge message"); message.append(output.data(), output.size() - inflater.state.avail_out);
                if (message.size() > 4 * 1024 * 1024) network_error("Decompressed Edge message too large"); if (!inflater.state.avail_in && inflater.state.avail_out) break;
                if (inflater.state.avail_in == before && inflater.state.avail_out == output.size()) network_error("Invalid compressed Edge message");
            } while (true);
            if (extensions.find("server_no_context_takeover") != std::string::npos) inflateReset(&inflater.state);
        }
        if (message_type == 1) {
            int32_t offset = 0; const int32_t size = int32_t(message.size());
            while (offset < size) { UChar32 codepoint; U8_NEXT(message.data(), offset, size, codepoint); if (codepoint < 0) network_error("Invalid UTF-8 Edge text frame"); }
        }
        if (!stream.accept(int(message_type), message)) { connection.frame(8, std::string("\x03\xe8", 2)); break; }
        message_type = 0; compressed = false; message.clear();
    }
    return response;
}
std::string edge_tts_mp3(const std::string &text, const std::string &voice, EdgeTtsOperations operations) {
    static auto global_skew = std::make_shared<std::atomic<double>>(0);
    if (!operations.time) operations.time = now; if (!operations.identifier) operations.identifier = identifier;
    if (!operations.exchange) operations.exchange = edge_tts_exchange; if (!operations.clock_skew) operations.clock_skew = global_skew;
    (void)edge_tts_ssml("", voice); std::string audio;
    for (const auto &chunk : edge_tts_chunks(text)) {
        for (int attempt = 0; attempt < 2; ++attempt) {
            EdgeTtsStream stream; const auto connection = operations.identifier(true), request = operations.identifier(true), muid = operations.identifier(false);
            const auto call = edge_tts_request(chunk, voice, operations.time(), operations.clock_skew->load(), connection, request, muid);
            const auto response = operations.exchange(call, stream);
            if (response.status == 403 && !attempt) {
                std::string date; for (const auto &[key, value] : response.headers) if (lower(key) == "date") date = value;
                if (date.empty()) throw TtsError("SkewAdjustmentError", "No server date in headers.");
                const auto server = server_time(date); operations.clock_skew->store(server - operations.time()); continue;
            }
            if (response.status != 101) throw TtsError("WSServerHandshakeError", "Edge WebSocket handshake returned HTTP " + std::to_string(response.status));
            audio += stream.finish(); break;
        }
    }
    return audio;
}
}
