#include "ezlocalai/edge_tts.hpp"
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <zlib.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <array>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <thread>
namespace {
void check(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
void send_all(int fd, const std::string &value) { size_t at = 0; while (at < value.size()) { const auto size = send(fd, value.data() + at, value.size() - at, MSG_NOSIGNAL); check(size > 0, "Fixture send failed"); at += size; } }
std::string take(int fd, size_t size) { std::string value(size, '\0'); size_t at = 0; while (at < size) { const auto count = recv(fd, value.data() + at, size - at, 0); check(count > 0, "Fixture receive failed"); at += count; } return value; }
std::pair<unsigned, std::string> receive_frame(int fd, z_stream *inflater = nullptr) {
    auto header = take(fd, 2); check(uint8_t(header[0]) & 128, "Client message must be complete"); check(uint8_t(header[1]) & 128, "Client frames must be masked"); uint64_t length = uint8_t(header[1]) & 127;
    if (length >= 126) { const auto extra = take(fd, length == 126 ? 2 : 8); length = 0; for (unsigned char c : extra) length = (length << 8) | c; }
    check(length < 65536, "Unexpected client frame length"); const auto mask = take(fd, 4); auto body = take(fd, length);
    for (size_t i = 0; i < body.size(); ++i) body[i] = char(uint8_t(body[i]) ^ uint8_t(mask[i % 4]));
    if (inflater) {
        check(uint8_t(header[0]) & 64, "Negotiated client text must be compressed like aiohttp"); body += std::string("\0\0\xff\xff", 4); inflater->next_in = reinterpret_cast<Bytef *>(body.data()); inflater->avail_in = body.size(); std::string decoded;
        do { std::array<char, 16384> bytes{}; inflater->next_out = reinterpret_cast<Bytef *>(bytes.data()); inflater->avail_out = bytes.size(); const auto result = inflate(inflater, Z_SYNC_FLUSH); check(result == Z_OK || result == Z_BUF_ERROR, "Client compression"); decoded.append(bytes.data(), bytes.size() - inflater->avail_out); } while (inflater->avail_in || !inflater->avail_out);
        body = std::move(decoded);
    } else check(!(uint8_t(header[0]) & 64), "Unexpected compressed frame");
    return {uint8_t(header[0]) & 15u, body};
}
std::string frame(unsigned opcode, const std::string &body, bool final = true, bool compressed = false) {
    std::string result(1, char(opcode | (final ? 128 : 0) | (compressed ? 64 : 0))); const auto size = body.size();
    if (size < 126) result += char(size); else { const int width = size <= 65535 ? 2 : 8; result += char(width == 2 ? 126 : 127); for (int i = width - 1; i >= 0; --i) result += char((uint64_t(size) >> (i * 8)) & 255); }
    return result + body;
}
std::string audio(const std::string &bytes) { const std::string header = "X:1\r\nPath:audio\r\nContent-Type:audio/mpeg\r\n"; return std::string({char(header.size() >> 8), char(header.size() & 255)}) + header + bytes; }
struct Compressor {
    z_stream state{};
    Compressor() { check(deflateInit2(&state, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY) == Z_OK, "Fixture compressor"); }
    ~Compressor() { deflateEnd(&state); }
    std::string encode(const std::string &value) {
        state.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(value.data())); state.avail_in = value.size(); std::string output;
        do { std::array<char, 16384> bytes{}; state.next_out = reinterpret_cast<Bytef *>(bytes.data()); state.avail_out = bytes.size(); check(deflate(&state, Z_SYNC_FLUSH) == Z_OK, "Fixture compression"); output.append(bytes.data(), bytes.size() - state.avail_out); } while (state.avail_in || !state.avail_out);
        check(output.size() >= 4, "Fixture compression trailer"); output.resize(output.size() - 4); return output;
    }
};
struct Server {
    int listener = -1; unsigned port = 0; std::thread worker; std::exception_ptr error;
    explicit Server(std::function<void(int)> serve) {
        listener = socket(AF_INET, SOCK_STREAM, 0); check(listener >= 0, "Fixture socket"); sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        check(bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0 && listen(listener, 1) == 0, "Fixture listen"); socklen_t length = sizeof(address); getsockname(listener, reinterpret_cast<sockaddr *>(&address), &length); port = ntohs(address.sin_port);
        worker = std::thread([this, serve] { int fd = accept(listener, nullptr, nullptr); try { check(fd >= 0, "Fixture accept"); timeval timeout{5, 0}; setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)); setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)); serve(fd); } catch (...) { error = std::current_exception(); } if (fd >= 0) close(fd); });
    }
    ~Server() { shutdown(listener, SHUT_RDWR); if (worker.joinable()) worker.join(); if (listener >= 0) close(listener); }
    void finish() { worker.join(); if (error) std::rethrow_exception(error); }
};
void handshake(int fd, bool compressed, bool invalid, const std::string &extra = "") {
    std::string request; while (request.find("\r\n\r\n") == std::string::npos) { request += take(fd, 1); check(request.size() < 65536, "Fixture handshake size"); }
    const std::string prefix = "Sec-WebSocket-Key: "; const auto at = request.find(prefix); check(at != std::string::npos, "Handshake key missing"); const auto end = request.find("\r\n", at); const auto key = request.substr(at + prefix.size(), end - at - prefix.size()) + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    unsigned char digest[SHA_DIGEST_LENGTH]; SHA1(reinterpret_cast<const unsigned char *>(key.data()), key.size(), digest); std::array<unsigned char, 29> encoded{}; EVP_EncodeBlock(encoded.data(), digest, sizeof(digest));
    send_all(fd, std::string("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ") + (invalid ? "bad" : reinterpret_cast<const char *>(encoded.data())) + (compressed ? "\r\nSec-WebSocket-Extensions: permessage-deflate" + extra : "") + "\r\n\r\n");
}
}
int main() {
    try {
        for (const std::string mode : {"plain", "compressed", "no_takeover", "large", "bad_accept", "empty", "truncated", "bad_compressed", "masked", "oversize", "invalid_utf8"}) {
            const bool compressed = mode == "compressed" || mode == "no_takeover" || mode == "bad_compressed";
            const bool good = mode == "plain" || mode == "compressed" || mode == "no_takeover" || mode == "large";
            const auto expected = mode == "large" ? std::string(70000, 'a') : "abc";
            auto request = ezlocalai::edge_tts_request("hello", "en-US-GuyNeural", 1792400000, 0, std::string(32, '0'), std::string(32, '0'), std::string(32, 'A'));
            Server server([&](int fd) {
                handshake(fd, compressed, mode == "bad_accept", mode == "no_takeover" ? "; client_no_context_takeover; server_no_context_takeover; client_max_window_bits=9" : ""); if (mode == "bad_accept") return;
                struct Inflater { z_stream state{}; Inflater() { check(inflateInit2(&state, -15) == Z_OK, "Fixture inflater"); } ~Inflater() { inflateEnd(&state); } } inflater;
                for (const auto &message : request.messages) { const auto actual = receive_frame(fd, compressed ? &inflater.state : nullptr); check(actual.first == 1 && actual.second == message, "WebSocket request content"); }
                if (mode == "empty") { send_all(fd, frame(8, std::string("\x03\xe8", 2))); (void)receive_frame(fd); return; }
                if (mode == "truncated") { send_all(fd, frame(2, "partial", false)); return; }
                if (mode == "bad_compressed") { send_all(fd, frame(2, "not-deflate", true, true)); return; }
                if (mode == "masked") { auto bad = frame(2, "bad"); bad[1] = char(uint8_t(bad[1]) | 128); send_all(fd, bad); return; }
                if (mode == "oversize") { send_all(fd, std::string("\x82\x7f\0\0\0\0\1\0\0\0", 10)); return; }
                if (mode == "invalid_utf8") { send_all(fd, frame(1, std::string("Path:turn.end\r\n\r\n") + char(255))); return; }
                const std::string ending = "Path:turn.end\r\n\r\n"; std::string response;
                if (compressed) {
                    Compressor compressor; const auto bytes = compressor.encode(audio(expected)); const auto half = bytes.size() / 2;
                    if (mode == "no_takeover") deflateReset(&compressor.state);
                    response = frame(2, bytes.substr(0, half), false, true) + frame(9, "ping") + frame(0, bytes.substr(half)) + frame(1, compressor.encode(ending), true, true);
                } else response = frame(2, audio(expected)) + frame(9, "ping") + frame(1, ending);
                send_all(fd, response); const auto pong = receive_frame(fd); check(pong.first == 10 && pong.second == "ping", "WebSocket pong"); check(receive_frame(fd).first == 8, "WebSocket close");
            });
            request.url = "ws://127.0.0.1:" + std::to_string(server.port) + "/edge"; bool failed = false;
            try { ezlocalai::EdgeTtsStream stream; check(ezlocalai::edge_tts_exchange(request, stream).status == 101, "Handshake status"); check(stream.finish() == expected, "Network audio payload"); }
            catch (const ezlocalai::TtsError &) { failed = true; }
            server.finish(); check(failed != good, "Network case outcome: " + mode); std::cout << mode << " passed\n";
        }
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
