#include "process_fixture.hpp"
#include "fixture_ports.hpp"
#include "ezlocalai/process.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/tunnel_connection.hpp"
#include "ezlocalai/video_runtime.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <openssl/err.h>
#include <openssl/ssl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <poll.h>

static void expect(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

static bool launch_uses_model_file(const std::string &args, const std::filesystem::path &expected) {
    const auto begin = args.find("-m\n");
    if (begin == std::string::npos) return false;
    const auto end = args.find('\n', begin + 3);
    std::error_code error;
    return std::filesystem::equivalent(args.substr(begin + 3, end - begin - 3), expected, error) && !error;
}

class SlowInferenceBackend final : public ezlocalai::InferenceBackend {
public:
    std::promise<void> entered;
    std::shared_future<void> release;
    explicit SlowInferenceBackend(std::shared_future<void> release) : release(std::move(release)) {}
    // This fixture models a resident LLM. Host GPU pressure must not turn the
    // queue-holder request into a memory-admission fallback request.
    std::string loaded_models_json() const override { return R"({"llm":{"name":"slow"}})"; }
    std::string chat(const std::string &request_body) override {
        entered.set_value();
        expect(release.wait_for(std::chrono::seconds(10)) == std::future_status::ready, "release held inference");
        return "{\"id\":\"chatcmpl-slow\",\"object\":\"chat.completion\",\"model\":\"slow\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"slow local\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}";
    }

    std::string completion(const std::string &request_body) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        return "{\"id\":\"cmpl-slow\",\"object\":\"text_completion\",\"model\":\"slow\",\"choices\":[{\"text\":\"slow local\",\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}";
    }

    std::string embedding(const std::string &request_body) override {
        return "{\"object\":\"list\",\"model\":\"slow\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0]}],\"usage\":{\"prompt_tokens\":0,\"total_tokens\":0}}";
    }
};

static std::string request(const std::string &raw) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(fd >= 0, "socket");

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(18091);
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "inet_pton");
    expect(connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "connect");

    size_t sent = 0;
    while (sent < raw.size()) {
        ssize_t n = send(fd, raw.data() + sent, raw.size() - sent, 0);
        expect(n > 0, "send");
        sent += static_cast<size_t>(n);
    }
    if (raw.rfind("POST /v1/audio/speech/stream ",0)!=0) shutdown(fd, SHUT_WR);

    std::string response;
    char buffer[4096];
    for (;;) {
        ssize_t n = recv(fd, buffer, sizeof(buffer), 0);
        if (n <= 0) {
            break;
        }
        response.append(buffer, static_cast<size_t>(n));
    }
    close(fd);
    return response;
}

static std::string request_observe_first_chunk(const std::string &raw, const std::string &needle, long long *first_seen_ms) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(fd >= 0, "stream observe socket");
    timeval timeout{};
    timeout.tv_sec = 3;
    timeout.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(18091);
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "stream observe inet_pton");
    expect(connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "stream observe connect");

    size_t sent = 0;
    while (sent < raw.size()) {
        ssize_t n = send(fd, raw.data() + sent, raw.size() - sent, 0);
        expect(n > 0, "stream observe send");
        sent += static_cast<size_t>(n);
    }
    if (raw.rfind("POST /v1/audio/speech/stream ",0)!=0) shutdown(fd, SHUT_WR);

    const auto start = std::chrono::steady_clock::now();
    bool seen = false;
    std::string response;
    char buffer[4096];
    for (;;) {
        ssize_t n = recv(fd, buffer, sizeof(buffer), 0);
        if (n <= 0) {
            break;
        }
        response.append(buffer, static_cast<size_t>(n));
        if (!seen && response.find(needle) != std::string::npos) {
            seen = true;
            const auto elapsed = std::chrono::steady_clock::now() - start;
            *first_seen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
        }
    }
    close(fd);
    expect(seen, "stream observe saw first chunk");
    return response;
}

static int open_websocket(const std::string &raw, std::string *handshake_out) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(fd >= 0, "websocket socket");
    timeval timeout{};
    timeout.tv_sec = 2;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(18091);
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "websocket inet_pton");
    expect(connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "websocket connect");

    size_t sent = 0;
    while (sent < raw.size()) {
        ssize_t n = send(fd, raw.data() + sent, raw.size() - sent, 0);
        expect(n > 0, "websocket send");
        sent += static_cast<size_t>(n);
    }

    std::string response;
    char buffer[1];
    // Leave any coalesced WebSocket frame in the socket for the frame reader.
    while (response.find("\r\n\r\n") == std::string::npos) {
        ssize_t n = recv(fd, buffer, sizeof(buffer), 0);
        expect(n > 0, "websocket handshake recv");
        response.append(buffer, static_cast<size_t>(n));
    }
    if (handshake_out) {
        *handshake_out = response;
    }
    return fd;
}

static std::string post_json(const std::string &path, const std::string &body, const std::string &authorization = "Bearer secret") {
    return request(
        "POST " + path + " HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: " + authorization + "\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body
    );
}

static std::string response_header(const std::string &response, const std::string &name) {
    const std::string prefix = name + ": ";
    size_t line_start = response.find("\r\n");
    if (line_start == std::string::npos) {
        return "";
    }
    line_start += 2;
    while (line_start < response.size()) {
        const size_t line_end = response.find("\r\n", line_start);
        if (line_end == std::string::npos || line_end == line_start) {
            break;
        }
        const std::string line = response.substr(line_start, line_end - line_start);
        if (line.rfind(prefix, 0) == 0) {
            return line.substr(prefix.size());
        }
        line_start = line_end + 2;
    }
    return "";
}

static std::string response_body(const std::string &response) {
    const size_t split = response.find("\r\n\r\n");
    if (split == std::string::npos) {
        return "";
    }
    return response.substr(split + 4);
}

static bool response_header_before(const std::string &response, const std::string &first, const std::string &second) {
    const size_t header_end = response.find("\r\n\r\n");
    const size_t first_pos = response.find("\r\n" + first + ": ");
    const size_t second_pos = response.find("\r\n" + second + ": ");
    return first_pos != std::string::npos
        && second_pos != std::string::npos
        && (header_end == std::string::npos || (first_pos < header_end && second_pos < header_end))
        && first_pos < second_pos;
}

static bool has_websocket_close_code(const std::string &response, int code) {
    const unsigned char hi = static_cast<unsigned char>((code >> 8) & 0xff);
    const unsigned char lo = static_cast<unsigned char>(code & 0xff);
    for (size_t i = 0; i + 3 < response.size(); ++i) {
        if (static_cast<unsigned char>(response[i]) != 0x88) {
            continue;
        }
        const size_t len = static_cast<unsigned char>(response[i + 1]) & 0x7fU;
        if (len >= 2
            && i + 2 + len <= response.size()
            && static_cast<unsigned char>(response[i + 2]) == hi
            && static_cast<unsigned char>(response[i + 3]) == lo) {
            return true;
        }
    }
    return false;
}

static std::string json_string_field(const std::string &json, const std::string &key) {
    const std::string marker = "\"" + key + "\":\"";
    const size_t start = json.find(marker);
    if (start == std::string::npos) {
        return "";
    }
    const size_t value_start = start + marker.size();
    const size_t end = json.find('"', value_start);
    return end == std::string::npos ? "" : json.substr(value_start, end - value_start);
}

static int test_base64_decode_value(char ch) {
    if (ch >= 'A' && ch <= 'Z') {
        return ch - 'A';
    }
    if (ch >= 'a' && ch <= 'z') {
        return ch - 'a' + 26;
    }
    if (ch >= '0' && ch <= '9') {
        return ch - '0' + 52;
    }
    if (ch == '+') {
        return 62;
    }
    if (ch == '/') {
        return 63;
    }
    return -1;
}

static std::string test_base64_decode(const std::string &input) {
    std::string out;
    int buffer = 0;
    int bits = -8;
    for (char ch : input) {
        if (ch == '=') {
            break;
        }
        const int value = test_base64_decode_value(ch);
        if (value < 0) {
            continue;
        }
        buffer = (buffer << 6) | value;
        bits += 6;
        if (bits >= 0) {
            out.push_back(static_cast<char>((buffer >> bits) & 0xff));
            bits -= 8;
        }
    }
    return out;
}

static void append_le16(std::string *out, uint16_t value) {
    out->push_back(static_cast<char>(value & 0xffU));
    out->push_back(static_cast<char>((value >> 8) & 0xffU));
}

static void append_le32(std::string *out, uint32_t value) {
    out->push_back(static_cast<char>(value & 0xffU));
    out->push_back(static_cast<char>((value >> 8) & 0xffU));
    out->push_back(static_cast<char>((value >> 16) & 0xffU));
    out->push_back(static_cast<char>((value >> 24) & 0xffU));
}

static std::string make_test_wav_pcm16() {
    constexpr int sample_rate = 8000;
    constexpr int seconds = 4;
    constexpr int frames = sample_rate * seconds;
    std::vector<int16_t> samples;
    samples.reserve(frames);
    for (int i = 0; i < frames; ++i) {
        const double t = static_cast<double>(i) / sample_rate;
        double value = 0.02 * std::sin(2.0 * 3.14159265358979323846 * 110.0 * t);
        if (t >= 0.0 && t < 1.0) {
            value = 0.18 * std::sin(2.0 * 3.14159265358979323846 * 220.0 * t);
        } else if (t >= 1.5 && t < 2.5) {
            value = 0.75 * std::sin(2.0 * 3.14159265358979323846 * 880.0 * t);
        }
        samples.push_back(static_cast<int16_t>(std::max(-1.0, std::min(1.0, value)) * 32767.0));
    }

    std::string wav;
    const uint32_t data_size = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    wav.append("RIFF", 4);
    append_le32(&wav, 36U + data_size);
    wav.append("WAVE", 4);
    wav.append("fmt ", 4);
    append_le32(&wav, 16);
    append_le16(&wav, 1);
    append_le16(&wav, 1);
    append_le32(&wav, sample_rate);
    append_le32(&wav, sample_rate * 2);
    append_le16(&wav, 2);
    append_le16(&wav, 16);
    wav.append("data", 4);
    append_le32(&wav, data_size);
    for (const auto sample : samples) {
        append_le16(&wav, static_cast<uint16_t>(sample));
    }
    return wav;
}

static std::string recv_websocket_text(int fd) {
    std::string buffer;
    char chunk[4096];
    // This helper has no persistent buffer: consume exactly one frame, even
    // when TCP coalesces a keepalive and the next request into the same read.
    auto read_to = [&](size_t size) {
        while (buffer.size() < size) {
            ssize_t n = recv(fd, chunk, std::min(sizeof(chunk), size - buffer.size()), 0);
            expect(n > 0, "websocket frame recv");
            buffer.append(chunk, static_cast<size_t>(n));
        }
    };
    read_to(2);
    const unsigned char second = static_cast<unsigned char>(buffer[1]);
    uint64_t len = second & 0x7fU;
    size_t pos = 2;
    if (len == 126) {
        read_to(pos + 2);
        len = (static_cast<uint64_t>(static_cast<unsigned char>(buffer[pos])) << 8)
            | static_cast<unsigned char>(buffer[pos + 1]);
        pos += 2;
    } else if (len == 127) {
        read_to(pos + 8);
        len = 0;
        for (int i = 0; i < 8; ++i) {
            len = (len << 8) | static_cast<unsigned char>(buffer[pos + static_cast<size_t>(i)]);
        }
        pos += 8;
    }
    read_to(pos + static_cast<size_t>(len));
    return buffer.substr(pos, static_cast<size_t>(len));
}

static std::string recv_websocket_payload(int fd, std::string *buffer) {
    char chunk[4096];
    auto read_more = [&]() {
        ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
        expect(n > 0, "websocket payload recv");
        buffer->append(chunk, static_cast<size_t>(n));
    };
    while (buffer->size() < 2) {
        read_more();
    }
    const unsigned char second = static_cast<unsigned char>((*buffer)[1]);
    uint64_t len = second & 0x7fU;
    size_t pos = 2;
    if (len == 126) {
        while (buffer->size() < pos + 2) {
            read_more();
        }
        len = (static_cast<uint64_t>(static_cast<unsigned char>((*buffer)[pos])) << 8)
            | static_cast<unsigned char>((*buffer)[pos + 1]);
        pos += 2;
    } else if (len == 127) {
        while (buffer->size() < pos + 8) {
            read_more();
        }
        len = 0;
        for (int i = 0; i < 8; ++i) {
            len = (len << 8) | static_cast<unsigned char>((*buffer)[pos + static_cast<size_t>(i)]);
        }
        pos += 8;
    }
    while (buffer->size() < pos + static_cast<size_t>(len)) {
        read_more();
    }
    std::string payload = buffer->substr(pos, static_cast<size_t>(len));
    buffer->erase(0, pos + static_cast<size_t>(len));
    return payload;
}

struct TestWebSocketFrame {
    unsigned char opcode = 0;
    bool masked = false;
    std::string payload;
};

static TestWebSocketFrame recv_websocket_frame_maybe_masked(int fd, std::string *buffer) {
    char chunk[4096];
    auto read_more = [&]() {
        ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
        expect(n > 0, "websocket masked payload recv");
        buffer->append(chunk, static_cast<size_t>(n));
    };
    while (buffer->size() < 2) {
        read_more();
    }
    const unsigned char first = static_cast<unsigned char>((*buffer)[0]);
    const unsigned char second = static_cast<unsigned char>((*buffer)[1]);
    uint64_t len = second & 0x7fU;
    size_t pos = 2;
    if (len == 126) {
        while (buffer->size() < pos + 2) {
            read_more();
        }
        len = (static_cast<uint64_t>(static_cast<unsigned char>((*buffer)[pos])) << 8)
            | static_cast<unsigned char>((*buffer)[pos + 1]);
        pos += 2;
    } else if (len == 127) {
        while (buffer->size() < pos + 8) {
            read_more();
        }
        len = 0;
        for (int i = 0; i < 8; ++i) {
            len = (len << 8) | static_cast<unsigned char>((*buffer)[pos + static_cast<size_t>(i)]);
        }
        pos += 8;
    }
    const bool masked = (second & 0x80U) != 0;
    unsigned char mask[4] = {0, 0, 0, 0};
    if (masked) {
        while (buffer->size() < pos + 4) {
            read_more();
        }
        for (int i = 0; i < 4; ++i) {
            mask[i] = static_cast<unsigned char>((*buffer)[pos + static_cast<size_t>(i)]);
        }
        pos += 4;
    }
    while (buffer->size() < pos + static_cast<size_t>(len)) {
        read_more();
    }

    TestWebSocketFrame frame;
    frame.opcode = first & 0x0fU;
    frame.masked = masked;
    frame.payload = buffer->substr(pos, static_cast<size_t>(len));
    if (masked) {
        for (size_t i = 0; i < frame.payload.size(); ++i) {
            frame.payload[i] = static_cast<char>(static_cast<unsigned char>(frame.payload[i]) ^ mask[i % 4]);
        }
    }
    buffer->erase(0, pos + static_cast<size_t>(len));
    return frame;
}

static void send_websocket_text(int fd, const std::string &payload) {
    std::string frame;
    frame.push_back(static_cast<char>(0x81));
    const uint64_t len = payload.size();
    if (len < 126) {
        frame.push_back(static_cast<char>(0x80 | len));
    } else if (len <= 0xffff) {
        frame.push_back(static_cast<char>(0x80 | 126));
        frame.push_back(static_cast<char>((len >> 8) & 0xff));
        frame.push_back(static_cast<char>(len & 0xff));
    } else {
        frame.push_back(static_cast<char>(0x80 | 127));
        for (int shift = 56; shift >= 0; shift -= 8) {
            frame.push_back(static_cast<char>((len >> shift) & 0xff));
        }
    }
    const unsigned char mask[4] = {0x11, 0x22, 0x33, 0x44};
    for (unsigned char ch : mask) {
        frame.push_back(static_cast<char>(ch));
    }
    for (size_t i = 0; i < payload.size(); ++i) {
        frame.push_back(static_cast<char>(static_cast<unsigned char>(payload[i]) ^ mask[i % 4]));
    }
    size_t sent = 0;
    while (sent < frame.size()) {
        ssize_t n = send(fd, frame.data() + sent, frame.size() - sent, 0);
        expect(n > 0, "websocket frame send");
        sent += static_cast<size_t>(n);
    }
}

static void send_websocket_text_unmasked(int fd, const std::string &payload) {
    std::string frame;
    frame.push_back(static_cast<char>(0x81));
    const uint64_t len = payload.size();
    if (len < 126) {
        frame.push_back(static_cast<char>(len));
    } else if (len <= 0xffff) {
        frame.push_back(static_cast<char>(126));
        frame.push_back(static_cast<char>((len >> 8) & 0xff));
        frame.push_back(static_cast<char>(len & 0xff));
    } else {
        frame.push_back(static_cast<char>(127));
        for (int shift = 56; shift >= 0; shift -= 8) {
            frame.push_back(static_cast<char>((len >> shift) & 0xff));
        }
    }
    frame += payload;
    size_t sent = 0;
    while (sent < frame.size()) {
        ssize_t n = send(fd, frame.data() + sent, frame.size() - sent, 0);
        expect(n > 0, "websocket unmasked frame send");
        sent += static_cast<size_t>(n);
    }
}

static std::string recv_http_request_with_body(int fd) {
    std::string raw;
    char buffer[4096];
    size_t header_end = std::string::npos;
    while ((header_end = raw.find("\r\n\r\n")) == std::string::npos) {
        ssize_t n = recv(fd, buffer, sizeof(buffer), 0);
        expect(n > 0, "http request header recv");
        raw.append(buffer, static_cast<size_t>(n));
    }

    const std::string headers = raw.substr(0, header_end + 4);
    std::string lower_headers = headers;
    std::transform(lower_headers.begin(), lower_headers.end(), lower_headers.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    size_t content_length = 0;
    const std::string marker = "\r\ncontent-length:";
    const size_t content_length_pos = lower_headers.find(marker);
    if (content_length_pos != std::string::npos) {
        size_t value_start = content_length_pos + marker.size();
        while (value_start < headers.size() && (headers[value_start] == ' ' || headers[value_start] == '\t')) {
            ++value_start;
        }
        const size_t value_end = headers.find("\r\n", value_start);
        content_length = static_cast<size_t>(std::stoull(headers.substr(value_start, value_end - value_start)));
    }
    while (raw.size() < header_end + 4 + content_length) {
        ssize_t n = recv(fd, buffer, sizeof(buffer), 0);
        expect(n > 0, "http request body recv");
        raw.append(buffer, static_cast<size_t>(n));
    }
    return raw;
}

struct TestTlsCertPaths {
    std::string cert;
    std::string key;
};

static TestTlsCertPaths ensure_test_tls_cert() {
    static std::mutex tls_cert_mutex;
    std::lock_guard<std::mutex> lock(tls_cert_mutex);
    const auto dir = std::filesystem::absolute("native-test-tls");
    std::filesystem::create_directories(dir);
    const auto cert = dir / "worker-cert.pem";
    const auto key = dir / "worker-key.pem";
    const auto config = dir / "openssl.cnf";
    const auto marker = dir / "cert-version";
    std::string marker_value;
    if (std::filesystem::exists(marker)) {
        std::ifstream in(marker);
        std::getline(in, marker_value);
    }
    if (!std::filesystem::exists(cert) || !std::filesystem::exists(key) || marker_value != "v3") {
        {
            std::ofstream out(config);
            out << "[req]\n"
                << "distinguished_name=req_distinguished_name\n"
                << "prompt=no\n"
                << "x509_extensions=v3_req\n"
                << "[req_distinguished_name]\n"
                << "CN=127.0.0.1\n"
                << "[v3_req]\n"
                << "basicConstraints=critical,CA:TRUE\n"
                << "keyUsage=critical,keyCertSign,digitalSignature,keyEncipherment\n"
                << "extendedKeyUsage=serverAuth\n"
                << "subjectAltName=@alt_names\n"
                << "[alt_names]\n"
                << "IP.1=127.0.0.1\n"
                << "IP.2=127.0.0.2\n";
        }
        expect(ezlocalai::run_process({"openssl", "req", "-x509", "-nodes", "-newkey", "rsa:2048", "-days", "1", "-sha256",
            "-keyout", key.string(), "-out", cert.string(), "-config", config.string(), "-extensions", "v3_req"}, true, true).status == 0,
            "generate test tls cert");
        std::ofstream out(marker);
        out << "v3\n";
    }
    return {cert.string(), key.string()};
}

static void ssl_write_all(SSL *ssl, const std::string &data) {
    size_t sent = 0;
    while (sent < data.size()) {
        const int n = SSL_write(ssl, data.data() + sent, static_cast<int>(std::min<size_t>(data.size() - sent, 16384)));
        expect(n > 0, "ssl write");
        sent += static_cast<size_t>(n);
    }
}

struct TestListenerReady {
    int port;
    std::chrono::milliseconds delay;
    std::promise<void> signal;
    std::future<void> started = signal.get_future();
    explicit TestListenerReady(int port, std::chrono::milliseconds delay = std::chrono::milliseconds(0))
        : port(port), delay(delay) {}
    void wait() {
        expect(started.wait_for(std::chrono::seconds(10)) == std::future_status::ready,
            ("HTTPS fixture did not start on port " + std::to_string(port)).c_str());
        started.get();
    }
};

static int accept_test_tls_connection(int server_fd, int port) {
    pollfd listener{server_fd, POLLIN, 0};
    expect(poll(&listener, 1, 10000) > 0,
        ("HTTPS fixture received no connection on port " + std::to_string(port)).c_str());
    int client_fd = accept(server_fd, nullptr, nullptr);
    expect(client_fd >= 0, "fake https worker accept");
    timeval timeout{10, 0};
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    return client_fd;
}

static void fake_https_json_worker_once(
    int port,
    const std::string &expected_request_line,
    const std::string &expected_authorization,
    const std::string &body,
    TestListenerReady *ready
) {
    std::this_thread::sleep_for(ready->delay);
    const auto cert = ensure_test_tls_cert();
    OPENSSL_init_ssl(0, nullptr);
    SSL_CTX *ctx = SSL_CTX_new(TLS_server_method());
    expect(ctx != nullptr, "fake https ctx");
    expect(SSL_CTX_use_certificate_file(ctx, cert.cert.c_str(), SSL_FILETYPE_PEM) == 1, "fake https cert");
    expect(SSL_CTX_use_PrivateKey_file(ctx, cert.key.c_str(), SSL_FILETYPE_PEM) == 1, "fake https key");

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake https worker socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake https worker bind");
    expect(listen(server_fd, 1) == 0, "fake https worker listen");
    ready->signal.set_value();

    int client_fd = accept_test_tls_connection(server_fd, port);
    SSL *ssl = SSL_new(ctx);
    expect(ssl != nullptr, "fake https ssl");
    SSL_set_fd(ssl, client_fd);
    if (SSL_accept(ssl) != 1) {
        const int ssl_error = SSL_get_error(ssl, -1);
        unsigned long err = ERR_get_error();
        char err_buf[256];
        ERR_error_string_n(err, err_buf, sizeof(err_buf));
        std::cerr << "fake https accept failed: ssl_error=" << ssl_error << " " << err_buf << "\n";
        expect(false, "fake https accept");
    }
    std::string raw;
    char buffer[2048];
    while (raw.find("\r\n\r\n") == std::string::npos) {
        int n = SSL_read(ssl, buffer, sizeof(buffer));
        expect(n > 0, "fake https recv");
        raw.append(buffer, static_cast<size_t>(n));
    }
    expect(raw.find(expected_request_line) != std::string::npos, "fake https request line");
    if (!expected_authorization.empty()) {
        expect(raw.find(expected_authorization) != std::string::npos, "fake https auth");
    }
    std::string response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
    ssl_write_all(ssl, response);
    SSL_shutdown(ssl);
    SSL_free(ssl);
    close(client_fd);
    close(server_fd);
    SSL_CTX_free(ctx);
}

static void fake_https_expect_worker_once_with_absence(
    int port,
    const std::string &expected_request_line,
    const std::string &expected_authorization,
    const std::vector<std::string> &expected_fragments,
    const std::vector<std::string> &unexpected_fragments,
    const std::string &response_body,
    TestListenerReady *ready
) {
    std::this_thread::sleep_for(ready->delay);
    const auto cert = ensure_test_tls_cert();
    OPENSSL_init_ssl(0, nullptr);
    SSL_CTX *ctx = SSL_CTX_new(TLS_server_method());
    expect(ctx != nullptr, "fake https expect ctx");
    expect(SSL_CTX_use_certificate_file(ctx, cert.cert.c_str(), SSL_FILETYPE_PEM) == 1, "fake https expect cert");
    expect(SSL_CTX_use_PrivateKey_file(ctx, cert.key.c_str(), SSL_FILETYPE_PEM) == 1, "fake https expect key");

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake https expect worker socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake https expect worker bind");
    expect(listen(server_fd, 1) == 0, "fake https expect worker listen");
    ready->signal.set_value();
    int client_fd = accept_test_tls_connection(server_fd, port);
    SSL *ssl = SSL_new(ctx);
    expect(ssl != nullptr, "fake https expect ssl");
    SSL_set_fd(ssl, client_fd);
    if (SSL_accept(ssl) != 1) {
        const int ssl_error = SSL_get_error(ssl, -1);
        unsigned long err = ERR_get_error();
        char err_buf[256];
        ERR_error_string_n(err, err_buf, sizeof(err_buf));
        std::cerr << "fake https expect accept failed: ssl_error=" << ssl_error << " " << err_buf << "\n";
        expect(false, "fake https expect accept");
    }

    std::string raw;
    char buffer[4096];
    while (raw.find("\r\n\r\n") == std::string::npos) {
        int n = SSL_read(ssl, buffer, sizeof(buffer));
        expect(n > 0, "fake https expect recv headers");
        raw.append(buffer, static_cast<size_t>(n));
    }
    const auto header_end = raw.find("\r\n\r\n");
    size_t content_length = 0;
    const auto content_length_pos = raw.find("\r\nContent-Length: ");
    if (content_length_pos != std::string::npos) {
        const auto value_pos = content_length_pos + std::string("\r\nContent-Length: ").size();
        const auto value_end = raw.find("\r\n", value_pos);
        content_length = static_cast<size_t>(std::stoul(raw.substr(value_pos, value_end - value_pos)));
    }
    while (raw.size() < header_end + 4 + content_length) {
        int n = SSL_read(ssl, buffer, sizeof(buffer));
        expect(n > 0, "fake https expect recv body");
        raw.append(buffer, static_cast<size_t>(n));
    }

    expect(raw.find(expected_request_line) != std::string::npos, "fake https expect request line");
    if (!expected_authorization.empty()) {
        expect(raw.find(expected_authorization) != std::string::npos, "fake https expect auth");
    }
    for (const auto &fragment : expected_fragments) {
        expect(raw.find(fragment) != std::string::npos, "fake https expect body fragment");
    }
    for (const auto &fragment : unexpected_fragments) {
        expect(raw.find(fragment) == std::string::npos, "fake https expect absent body fragment");
    }

    std::string response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + std::to_string(response_body.size()) + "\r\nConnection: close\r\n\r\n" + response_body;
    ssl_write_all(ssl, response);
    SSL_shutdown(ssl);
    SSL_free(ssl);
    close(client_fd);
    close(server_fd);
    SSL_CTX_free(ctx);
}

static void fake_https_json_worker_many(
    int port,
    const std::string &expected_authorization,
    const std::vector<std::pair<std::string, std::string>> &requests,
    TestListenerReady *ready
) {
    std::this_thread::sleep_for(ready->delay);
    const auto cert = ensure_test_tls_cert();
    OPENSSL_init_ssl(0, nullptr);
    SSL_CTX *ctx = SSL_CTX_new(TLS_server_method());
    expect(ctx != nullptr, "fake https many ctx");
    expect(SSL_CTX_use_certificate_file(ctx, cert.cert.c_str(), SSL_FILETYPE_PEM) == 1, "fake https many cert");
    expect(SSL_CTX_use_PrivateKey_file(ctx, cert.key.c_str(), SSL_FILETYPE_PEM) == 1, "fake https many key");

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake https many worker socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake https many worker bind");
    expect(listen(server_fd, 4) == 0, "fake https many worker listen");
    ready->signal.set_value();

    for (const auto &entry : requests) {
        int client_fd = accept_test_tls_connection(server_fd, port);
        SSL *ssl = SSL_new(ctx);
        expect(ssl != nullptr, "fake https many ssl");
        SSL_set_fd(ssl, client_fd);
        if (SSL_accept(ssl) != 1) {
            const int ssl_error = SSL_get_error(ssl, -1);
            unsigned long err = ERR_get_error();
            char err_buf[256];
            ERR_error_string_n(err, err_buf, sizeof(err_buf));
            std::cerr << "fake https many accept failed: ssl_error=" << ssl_error << " " << err_buf << "\n";
            expect(false, "fake https many accept");
        }
        std::string raw;
        char buffer[2048];
        while (raw.find("\r\n\r\n") == std::string::npos) {
            int n = SSL_read(ssl, buffer, sizeof(buffer));
            expect(n > 0, "fake https many recv");
            raw.append(buffer, static_cast<size_t>(n));
        }
        expect(raw.find(entry.first) != std::string::npos, "fake https many request line");
        if (!expected_authorization.empty()) {
            expect(raw.find(expected_authorization) != std::string::npos, "fake https many auth");
        }
        const auto &body = entry.second;
        std::string response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
            + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
        ssl_write_all(ssl, response);
        SSL_shutdown(ssl);
        SSL_free(ssl);
        close(client_fd);
    }
    close(server_fd);
    SSL_CTX_free(ctx);
}

static void fake_json_worker_once(int port, const std::string &body) {
    std::vector<std::string> bodies = {body};
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake worker socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake worker inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake worker bind");
    expect(listen(server_fd, 1) == 0, "fake worker listen");
    for (const auto &response_body : bodies) {
        int client_fd = accept(server_fd, nullptr, nullptr);
        expect(client_fd >= 0, "fake worker accept");
        char buffer[2048];
        recv(client_fd, buffer, sizeof(buffer), 0);
        std::string response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
            + std::to_string(response_body.size()) + "\r\nConnection: close\r\n\r\n" + response_body;
        send(client_fd, response.data(), response.size(), 0);
        close(client_fd);
    }
    close(server_fd);
}

static void fake_chunked_worker_once(int port, const std::string &content_type, const std::vector<std::string> &chunks) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake chunked worker socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake chunked worker inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake chunked worker bind");
    expect(listen(server_fd, 1) == 0, "fake chunked worker listen");
    int client_fd = accept(server_fd, nullptr, nullptr);
    expect(client_fd >= 0, "fake chunked worker accept");
    char buffer[2048];
    recv(client_fd, buffer, sizeof(buffer), 0);

    std::ostringstream response;
    response << "HTTP/1.1 200 OK\r\n"
        << "Content-Type: " << content_type << "\r\n"
        << "Transfer-Encoding: chunked\r\n"
        << "Connection: close\r\n\r\n";
    for (const auto &chunk : chunks) {
        response << std::hex << chunk.size() << "\r\n" << chunk << "\r\n";
    }
    response << "0\r\n\r\n";
    const std::string raw = response.str();
    send(client_fd, raw.data(), raw.size(), 0);
    close(client_fd);
    close(server_fd);
}

static void fake_delayed_chunked_worker_once(
    int port,
    const std::string &content_type,
    const std::string &first_chunk,
    const std::string &second_chunk,
    int delay_ms
) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake delayed chunked worker socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake delayed chunked worker inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake delayed chunked worker bind");
    expect(listen(server_fd, 1) == 0, "fake delayed chunked worker listen");
    int client_fd = accept(server_fd, nullptr, nullptr);
    expect(client_fd >= 0, "fake delayed chunked worker accept");
    char buffer[2048];
    recv(client_fd, buffer, sizeof(buffer), 0);

    auto send_chunk = [&](const std::string &chunk) {
        std::ostringstream out;
        out << std::hex << chunk.size() << "\r\n" << chunk << "\r\n";
        const std::string raw = out.str();
        send(client_fd, raw.data(), raw.size(), 0);
    };

    std::ostringstream headers;
    headers << "HTTP/1.1 200 OK\r\n"
        << "Content-Type: " << content_type << "\r\n"
        << "Transfer-Encoding: chunked\r\n"
        << "Connection: close\r\n\r\n";
    const std::string raw_headers = headers.str();
    send(client_fd, raw_headers.data(), raw_headers.size(), 0);
    send_chunk(first_chunk);
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    send_chunk(second_chunk);
    send(client_fd, "0\r\n\r\n", 5, 0);
    close(client_fd);
    close(server_fd);
}

static void fake_chunked_worker_many(
    int port,
    const std::string &content_type,
    const std::vector<std::vector<std::string>> &responses
) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake chunked worker socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake chunked worker inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake chunked worker bind");
    expect(listen(server_fd, 1) == 0, "fake chunked worker listen");
    for (const auto &chunks : responses) {
        int client_fd = accept(server_fd, nullptr, nullptr);
        expect(client_fd >= 0, "fake chunked worker accept");
        char buffer[2048];
        recv(client_fd, buffer, sizeof(buffer), 0);

        std::ostringstream response;
        response << "HTTP/1.1 200 OK\r\n"
            << "Content-Type: " << content_type << "\r\n"
            << "Transfer-Encoding: chunked\r\n"
            << "Connection: close\r\n\r\n";
        for (const auto &chunk : chunks) {
            response << std::hex << chunk.size() << "\r\n" << chunk << "\r\n";
        }
        response << "0\r\n\r\n";
        const std::string raw = response.str();
        send(client_fd, raw.data(), raw.size(), 0);
        close(client_fd);
    }
    close(server_fd);
}

static void fake_json_worker_once_status(int port, int status, const std::string &body) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake status worker socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake status worker inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake status worker bind");
    expect(listen(server_fd, 1) == 0, "fake status worker listen");
    int client_fd = accept(server_fd, nullptr, nullptr);
    expect(client_fd >= 0, "fake status worker accept");
    char buffer[2048];
    recv(client_fd, buffer, sizeof(buffer), 0);
    std::string reason = status == 500 ? "Internal Server Error" : "OK";
    std::string response = "HTTP/1.1 " + std::to_string(status) + " " + reason
        + "\r\nContent-Type: application/json\r\nContent-Length: "
        + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
    send(client_fd, response.data(), response.size(), 0);
    close(client_fd);
    close(server_fd);
}

static void fake_json_worker_once_status_timeout(int port, int status, const std::string &body, int timeout_ms) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake timeout status worker socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake timeout status worker inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake timeout status worker bind");
    expect(listen(server_fd, 1) == 0, "fake timeout status worker listen");

    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(server_fd, &read_fds);
    timeval timeout{};
    timeout.tv_sec = timeout_ms / 1000;
    timeout.tv_usec = (timeout_ms % 1000) * 1000;
    int ready = select(server_fd + 1, &read_fds, nullptr, nullptr, &timeout);
    if (ready <= 0) {
        close(server_fd);
        return;
    }

    int client_fd = accept(server_fd, nullptr, nullptr);
    expect(client_fd >= 0, "fake timeout status worker accept");
    char buffer[2048];
    recv(client_fd, buffer, sizeof(buffer), 0);
    std::string reason = status == 500 ? "Internal Server Error" : "OK";
    std::string response = "HTTP/1.1 " + std::to_string(status) + " " + reason
        + "\r\nContent-Type: application/json\r\nContent-Length: "
        + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
    send(client_fd, response.data(), response.size(), 0);
    close(client_fd);
    close(server_fd);
}

static void fake_expect_worker_once_with_type(
    int port,
    const std::string &expected_request_line,
    const std::string &expected_content_type,
    const std::string &expected_body_fragment,
    const std::string &response_content_type,
    const std::string &response_body
) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake expect worker socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake expect worker inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake expect worker bind");
    expect(listen(server_fd, 1) == 0, "fake expect worker listen");
    int client_fd = accept(server_fd, nullptr, nullptr);
    expect(client_fd >= 0, "fake expect worker accept");

    std::string raw;
    char buffer[4096];
    for (;;) {
        ssize_t n = recv(client_fd, buffer, sizeof(buffer), 0);
        if (n <= 0) {
            break;
        }
        raw.append(buffer, static_cast<size_t>(n));
        if (raw.find(expected_body_fragment) != std::string::npos) {
            break;
        }
    }
    expect(raw.find(expected_request_line) != std::string::npos, "fake expect worker request line");
    expect(raw.find(expected_content_type) != std::string::npos, "fake expect worker content type");
    expect(raw.find(expected_body_fragment) != std::string::npos, "fake expect worker body fragment");

    std::string response = "HTTP/1.1 200 OK\r\nContent-Type: " + response_content_type + "\r\nContent-Length: "
        + std::to_string(response_body.size()) + "\r\nConnection: close\r\n\r\n" + response_body;
    send(client_fd, response.data(), response.size(), 0);
    close(client_fd);
    close(server_fd);
}

static void fake_expect_worker_once(
    int port,
    const std::string &expected_request_line,
    const std::string &expected_content_type,
    const std::string &expected_body_fragment,
    const std::string &response_body
) {
    fake_expect_worker_once_with_type(
        port,
        expected_request_line,
        expected_content_type,
        expected_body_fragment,
        "application/json",
        response_body
    );
}

static void fake_expect_worker_once_with_absence(
    int port,
    const std::string &expected_request_line,
    const std::string &expected_content_type,
    const std::string &expected_body_fragment,
    const std::string &second_expected_body_fragment,
    const std::string &third_expected_body_fragment,
    const std::string &fourth_expected_body_fragment,
    const std::string &fifth_expected_body_fragment,
    const std::string &unexpected_body_fragment,
    const std::string &second_unexpected_body_fragment,
    const std::string &response_body
) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake expect absent worker socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake expect absent worker inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake expect absent worker bind");
    expect(listen(server_fd, 1) == 0, "fake expect absent worker listen");
    int client_fd = accept(server_fd, nullptr, nullptr);
    expect(client_fd >= 0, "fake expect absent worker accept");
    std::string raw = recv_http_request_with_body(client_fd);
    expect(raw.find(expected_request_line) != std::string::npos, "fake expect absent worker request line");
    expect(raw.find(expected_content_type) != std::string::npos, "fake expect absent worker content type");
    expect(raw.find(expected_body_fragment) != std::string::npos, "fake expect absent worker body fragment");
    expect(raw.find(second_expected_body_fragment) != std::string::npos, "fake expect absent worker second body fragment");
    if (!third_expected_body_fragment.empty()) {
        expect(raw.find(third_expected_body_fragment) != std::string::npos, "fake expect absent worker third body fragment");
    }
    if (!fourth_expected_body_fragment.empty()) {
        expect(raw.find(fourth_expected_body_fragment) != std::string::npos, "fake expect absent worker fourth body fragment");
    }
    if (!fifth_expected_body_fragment.empty()) {
        expect(raw.find(fifth_expected_body_fragment) != std::string::npos, "fake expect absent worker fifth body fragment");
    }
    expect(raw.find(unexpected_body_fragment) == std::string::npos, "fake expect absent worker absent body fragment");
    if (!second_unexpected_body_fragment.empty()) {
        expect(raw.find(second_unexpected_body_fragment) == std::string::npos, "fake expect absent worker second absent body fragment");
    }
    std::string response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + std::to_string(response_body.size()) + "\r\nConnection: close\r\n\r\n" + response_body;
    send(client_fd, response.data(), response.size(), 0);
    close(client_fd);
    close(server_fd);
}

static void fake_voice_forward_once(
    int port,
    const std::string &expected_post_line,
    const std::string &expected_content_type,
    const std::string &expected_body_fragment,
    const std::string &unexpected_body_fragment,
    const std::string &response_content_type,
    const std::string &response_body,
    const std::string &expected_authorization
) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake voice socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake voice inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake voice bind");
    expect(listen(server_fd, 2) == 0, "fake voice listen");

    int availability_fd = accept(server_fd, nullptr, nullptr);
    expect(availability_fd >= 0, "fake voice availability accept");
    std::string availability_raw = recv_http_request_with_body(availability_fd);
    expect(availability_raw.find("GET /v1/audio/voices HTTP/1.1") != std::string::npos, "fake voice availability path");
    expect(availability_raw.find("Content-Length:") == std::string::npos, "fake voice availability omits content length");
    if (!expected_authorization.empty()) {
        expect(availability_raw.find(expected_authorization) != std::string::npos, "fake voice availability auth");
    }
    const std::string voices_body = "{\"voices\":[\"remote\"]}";
    std::string voices_response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + std::to_string(voices_body.size()) + "\r\nConnection: close\r\n\r\n" + voices_body;
    send(availability_fd, voices_response.data(), voices_response.size(), 0);
    close(availability_fd);

    int post_fd = accept(server_fd, nullptr, nullptr);
    expect(post_fd >= 0, "fake voice post accept");
    std::string post_raw = recv_http_request_with_body(post_fd);
    expect(post_raw.find(expected_post_line) != std::string::npos, "fake voice post path");
    expect(post_raw.find(expected_content_type) != std::string::npos, "fake voice content type");
    expect(post_raw.find(expected_body_fragment) != std::string::npos, "fake voice body fragment");
    if (!unexpected_body_fragment.empty()) {
        expect(post_raw.find(unexpected_body_fragment) == std::string::npos, "fake voice absent body fragment");
    }
    if (!expected_authorization.empty()) {
        expect(post_raw.find(expected_authorization) != std::string::npos, "fake voice post auth");
    }
    std::string post_response = "HTTP/1.1 200 OK\r\nContent-Type: " + response_content_type
        + "\r\nContent-Length: " + std::to_string(response_body.size())
        + "\r\nConnection: close\r\n\r\n" + response_body;
    send(post_fd, post_response.data(), post_response.size(), 0);
    close(post_fd);
    close(server_fd);
}

static void fake_voice_delayed_stream_once(
    int port,
    const std::string &expected_post_line,
    const std::string &expected_content_type,
    const std::string &expected_body_fragment,
    const std::string &unexpected_body_fragment,
    const std::string &first_chunk,
    const std::string &second_chunk,
    int delay_ms,
    const std::string &expected_authorization
) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake voice delayed socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake voice delayed inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake voice delayed bind");
    expect(listen(server_fd, 2) == 0, "fake voice delayed listen");

    int availability_fd = accept(server_fd, nullptr, nullptr);
    expect(availability_fd >= 0, "fake voice delayed availability accept");
    std::string availability_raw = recv_http_request_with_body(availability_fd);
    expect(availability_raw.find("GET /v1/audio/voices HTTP/1.1") != std::string::npos, "fake voice delayed availability path");
    if (!expected_authorization.empty()) {
        expect(availability_raw.find(expected_authorization) != std::string::npos, "fake voice delayed availability auth");
    }
    const std::string voices_body = "{\"voices\":[\"remote\"]}";
    std::string voices_response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + std::to_string(voices_body.size()) + "\r\nConnection: close\r\n\r\n" + voices_body;
    send(availability_fd, voices_response.data(), voices_response.size(), 0);
    close(availability_fd);

    int post_fd = accept(server_fd, nullptr, nullptr);
    expect(post_fd >= 0, "fake voice delayed post accept");
    std::string post_raw = recv_http_request_with_body(post_fd);
    expect(post_raw.find(expected_post_line) != std::string::npos, "fake voice delayed post path");
    expect(post_raw.find(expected_content_type) != std::string::npos, "fake voice delayed content type");
    expect(post_raw.find(expected_body_fragment) != std::string::npos, "fake voice delayed body fragment");
    if (!unexpected_body_fragment.empty()) {
        expect(post_raw.find(unexpected_body_fragment) == std::string::npos, "fake voice delayed absent body fragment");
    }
    if (!expected_authorization.empty()) {
        expect(post_raw.find(expected_authorization) != std::string::npos, "fake voice delayed post auth");
    }

    auto send_chunk = [&](const std::string &chunk) {
        std::ostringstream out;
        out << std::hex << chunk.size() << "\r\n" << chunk << "\r\n";
        const std::string raw = out.str();
        send(post_fd, raw.data(), raw.size(), 0);
    };
    const std::string headers =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/octet-stream\r\n"
        "Transfer-Encoding: chunked\r\n"
        "Connection: close\r\n\r\n";
    send(post_fd, headers.data(), headers.size(), 0);
    send_chunk(first_chunk);
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    send_chunk(second_chunk);
    send(post_fd, "0\r\n\r\n", 5, 0);
    close(post_fd);
    close(server_fd);
}

static void fake_fallback_forward_once(
    int port,
    const std::string &expected_post_line,
    const std::string &expected_content_type,
    const std::string &expected_body_fragment,
    const std::string &unexpected_body_fragment,
    const std::string &response_content_type,
    const std::string &response_body,
    const std::string &expected_authorization
) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake fallback socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake fallback inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake fallback bind");
    expect(listen(server_fd, 2) == 0, "fake fallback listen");

    int resources_fd = accept(server_fd, nullptr, nullptr);
    expect(resources_fd >= 0, "fake fallback resources accept");
    std::string resources_raw = recv_http_request_with_body(resources_fd);
    expect(resources_raw.find("GET /v1/resources HTTP/1.1") != std::string::npos, "fake fallback resources path");
    expect(resources_raw.find("Content-Length:") == std::string::npos, "fake fallback resources omits content length");
    if (!expected_authorization.empty()) {
        expect(resources_raw.find(expected_authorization) != std::string::npos, "fake fallback resources auth");
    }
    const std::string resources_body = "{\"free_vram_gb\":4.5}";
    std::string resources_response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + std::to_string(resources_body.size()) + "\r\nConnection: close\r\n\r\n" + resources_body;
    send(resources_fd, resources_response.data(), resources_response.size(), 0);
    close(resources_fd);

    int post_fd = accept(server_fd, nullptr, nullptr);
    expect(post_fd >= 0, "fake fallback post accept");
    std::string post_raw = recv_http_request_with_body(post_fd);
    expect(post_raw.find(expected_post_line) != std::string::npos, "fake fallback post path");
    expect(post_raw.find(expected_content_type) != std::string::npos, "fake fallback content type");
    expect(post_raw.find(expected_body_fragment) != std::string::npos, "fake fallback body fragment");
    if (!unexpected_body_fragment.empty()) {
        expect(post_raw.find(unexpected_body_fragment) == std::string::npos, "fake fallback absent body fragment");
    }
    if (!expected_authorization.empty()) {
        expect(post_raw.find(expected_authorization) != std::string::npos, "fake fallback post auth");
    }
    std::string post_response = "HTTP/1.1 200 OK\r\nContent-Type: " + response_content_type
        + "\r\nContent-Length: " + std::to_string(response_body.size())
        + "\r\nConnection: close\r\n\r\n" + response_body;
    send(post_fd, post_response.data(), post_response.size(), 0);
    close(post_fd);
    close(server_fd);
}

static void fake_slow_resources_once(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake slow resources socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake slow resources inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake slow resources bind");
    expect(listen(server_fd, 1) == 0, "fake slow resources listen");

    int client_fd = accept(server_fd, nullptr, nullptr);
    expect(client_fd >= 0, "fake slow resources accept");
    std::string raw = recv_http_request_with_body(client_fd);
    expect(raw.find("GET /v1/resources HTTP/1.1") != std::string::npos, "fake slow resources path");
    std::this_thread::sleep_for(std::chrono::milliseconds(5500));
    close(client_fd);
    close(server_fd);
}

static void fake_worker_router_register_and_deregister(
    int port,
    std::vector<std::string> *requests
) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake worker router socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    timeval timeout{};
    timeout.tv_sec = 5;
    timeout.tv_usec = 0;
    setsockopt(server_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake worker router inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake worker router bind");
    expect(listen(server_fd, 3) == 0, "fake worker router listen");

    for (int i = 0; i < 3; ++i) {
        int client_fd = accept(server_fd, nullptr, nullptr);
        expect(client_fd >= 0, "fake worker router accept");
        std::string raw = recv_http_request_with_body(client_fd);
        if (requests) {
            requests->push_back(raw);
        }
        const std::string body = "{\"ok\":true,\"worker\":{}}";
        std::string response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
            + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
        send(client_fd, response.data(), response.size(), 0);
        close(client_fd);
    }
    close(server_fd);
}

static void fake_worker_router_with_tunnel(
    int port,
    std::vector<std::string> *requests,
    std::atomic<bool> *registered,
    std::atomic<bool> *heartbeated,
    std::atomic<bool> *tunnel_done
) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake worker tunnel router socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    timeval timeout{};
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;
    setsockopt(server_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake worker tunnel router inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake worker tunnel router bind");
    expect(listen(server_fd, 8) == 0, "fake worker tunnel router listen");

    bool saw_tunnel = false;
    bool saw_register = false;
    bool saw_heartbeat = false;
    bool saw_deregister = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(12);
    while (std::chrono::steady_clock::now() < deadline && !saw_deregister) {
        int client_fd = accept(server_fd, nullptr, nullptr);
        if (client_fd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
                continue;
            }
            expect(false, "fake worker tunnel router accept");
        }
        timeval client_timeout{};
        client_timeout.tv_sec = 3;
        client_timeout.tv_usec = 0;
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &client_timeout, sizeof(client_timeout));
        setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &client_timeout, sizeof(client_timeout));

        std::string raw = recv_http_request_with_body(client_fd);
        if (requests) {
            requests->push_back(raw);
        }
        if (raw.rfind("GET /v1/router/tunnel?worker_id=TunnelWorker-", 0) == 0) {
            expect(raw.find("Authorization: Bearer router-secret") != std::string::npos, "worker tunnel router auth");
            expect(raw.find("Upgrade: websocket") != std::string::npos, "worker tunnel router upgrade");
            const std::string key_prefix = "Sec-WebSocket-Key: ";
            const auto key_begin = raw.find(key_prefix) + key_prefix.size();
            const auto key = raw.substr(key_begin, raw.find("\r\n", key_begin) - key_begin);
            const std::string handshake =
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                "Sec-WebSocket-Accept: " + ezlocalai::tunnel_websocket_accept(key) + "\r\n\r\n";
            send(client_fd, handshake.data(), handshake.size(), 0);

            send_websocket_text_unmasked(
                client_fd,
                "{\"t\":\"req\",\"id\":\"outbound-1\",\"method\":\"GET\",\"path\":\"/health\",\"headers\":{},\"body_b64\":\"\"}"
            );
            std::string frame_buffer;
            bool saw_start = false;
            bool saw_chunk = false;
            bool saw_end = false;
            std::string decoded_body;
            while (!saw_end) {
                const auto frame = recv_websocket_frame_maybe_masked(client_fd, &frame_buffer);
                expect(frame.opcode == 0x1U, "worker tunnel response text frame");
                expect(frame.masked, "worker tunnel client masks response frame");
                if (frame.payload.find("\"t\":\"resp_start\"") != std::string::npos) {
                    expect(frame.payload.find("\"id\":\"outbound-1\"") != std::string::npos, "worker tunnel response start id");
                    expect(frame.payload.find("\"status\":200") != std::string::npos, "worker tunnel response start status");
                    expect(frame.payload.find("\"stream\":true") != std::string::npos, "worker tunnel response start stream flag");
                    saw_start = true;
                } else if (frame.payload.find("\"t\":\"resp_chunk\"") != std::string::npos) {
                    expect(saw_start, "worker tunnel chunk follows start");
                    expect(frame.payload.find("\"id\":\"outbound-1\"") != std::string::npos, "worker tunnel chunk id");
                    decoded_body += test_base64_decode(json_string_field(frame.payload, "data_b64"));
                    saw_chunk = true;
                } else if (frame.payload.find("\"t\":\"resp_end\"") != std::string::npos) {
                    expect(saw_start, "worker tunnel end follows start");
                    expect(frame.payload.find("\"id\":\"outbound-1\"") != std::string::npos, "worker tunnel end id");
                    saw_end = true;
                }
            }
            expect(saw_start, "worker tunnel saw response start");
            expect(saw_chunk, "worker tunnel saw response chunk");
            expect(decoded_body.find("\"status\":\"healthy\"") != std::string::npos, "worker tunnel forwards local health body");
            tunnel_done->store(true);
            saw_tunnel = true;
            close(client_fd);
            continue;
        }

        const std::string body = "{\"ok\":true,\"worker\":{}}";
        std::string response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
            + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
        send(client_fd, response.data(), response.size(), 0);
        close(client_fd);

        if (raw.find("POST /v1/router/register HTTP/1.1") != std::string::npos) {
            saw_register = true;
            registered->store(true);
        } else if (raw.find("POST /v1/router/heartbeat HTTP/1.1") != std::string::npos) {
            saw_heartbeat = true;
            heartbeated->store(true);
        } else if (raw.find("POST /v1/router/deregister HTTP/1.1") != std::string::npos) {
            saw_deregister = true;
        }
    }
    close(server_fd);
    expect(saw_register, "worker tunnel router saw register");
    expect(saw_heartbeat, "worker tunnel router saw heartbeat");
    expect(saw_tunnel, "worker tunnel router saw websocket tunnel");
    expect(saw_deregister, "worker tunnel router saw deregister");
}

static void fake_json_service_forward_once(
    int port,
    const std::string &expected_post_line,
    const std::string &expected_body_fragment,
    const std::string &unexpected_body_fragment,
    const std::string &response_body,
    const std::string &expected_authorization,
    const std::string &probe_path,
    const std::string &models_body
) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake image server socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake image server inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake image server bind");
    expect(listen(server_fd, 2) == 0, "fake image server listen");

    int models_fd = accept(server_fd, nullptr, nullptr);
    expect(models_fd >= 0, "fake image server models accept");
    std::string models_raw = recv_http_request_with_body(models_fd);
    expect(models_raw.find("GET " + probe_path + " HTTP/1.1") != std::string::npos, "fake JSON service probe path");
    expect(models_raw.find("Content-Length:") == std::string::npos, "fake image server models omits content length");
    if (!expected_authorization.empty()) {
        expect(models_raw.find(expected_authorization) != std::string::npos, "fake image server models auth");
    }
    std::string models_response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + std::to_string(models_body.size()) + "\r\nConnection: close\r\n\r\n" + models_body;
    send(models_fd, models_response.data(), models_response.size(), 0);
    close(models_fd);

    int post_fd = accept(server_fd, nullptr, nullptr);
    expect(post_fd >= 0, "fake image server post accept");
    std::string post_raw = recv_http_request_with_body(post_fd);
    expect(post_raw.find(expected_post_line) != std::string::npos, "fake image server post path");
    expect(post_raw.find("Content-Type: application/json") != std::string::npos, "fake image server content type");
    expect(post_raw.find(expected_body_fragment) != std::string::npos, "fake image server body fragment");
    if (!unexpected_body_fragment.empty()) {
        expect(post_raw.find(unexpected_body_fragment) == std::string::npos, "fake image server absent body fragment");
    }
    if (!expected_authorization.empty()) {
        expect(post_raw.find(expected_authorization) != std::string::npos, "fake image server post auth");
    }
    std::string post_response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + std::to_string(response_body.size()) + "\r\nConnection: close\r\n\r\n" + response_body;
    send(post_fd, post_response.data(), post_response.size(), 0);
    close(post_fd);
    close(server_fd);
}

static void fake_image_server_forward_once(
    int port,
    const std::string &expected_post_line,
    const std::string &expected_body_fragment,
    const std::string &unexpected_body_fragment,
    const std::string &response_body,
    const std::string &expected_authorization
) {
    fake_json_service_forward_once(port, expected_post_line, expected_body_fragment, unexpected_body_fragment,
        response_body, expected_authorization, "/v1/models", "{\"object\":\"list\",\"data\":[{\"id\":\"image-worker\"}]}");
}

static void fake_image_server_delayed_forward_once(
    int port,
    const std::string &expected_post_line,
    const std::string &expected_body_fragment,
    const std::string &unexpected_body_fragment,
    const std::string &response_body,
    const std::string &expected_authorization,
    int delay_ms
) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake delayed image server socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake delayed image server inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake delayed image server bind");
    expect(listen(server_fd, 2) == 0, "fake delayed image server listen");

    int models_fd = accept(server_fd, nullptr, nullptr);
    expect(models_fd >= 0, "fake delayed image server models accept");
    std::string models_raw = recv_http_request_with_body(models_fd);
    expect(models_raw.find("GET /v1/models HTTP/1.1") != std::string::npos, "fake delayed image server models path");
    if (!expected_authorization.empty()) {
        expect(models_raw.find(expected_authorization) != std::string::npos, "fake delayed image server models auth");
    }
    const std::string models_body = "{\"object\":\"list\",\"data\":[{\"id\":\"image-worker\"}]}";
    std::string models_response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + std::to_string(models_body.size()) + "\r\nConnection: close\r\n\r\n" + models_body;
    send(models_fd, models_response.data(), models_response.size(), 0);
    close(models_fd);

    int post_fd = accept(server_fd, nullptr, nullptr);
    expect(post_fd >= 0, "fake delayed image server post accept");
    std::string post_raw = recv_http_request_with_body(post_fd);
    expect(post_raw.find(expected_post_line) != std::string::npos, "fake delayed image server post path");
    expect(post_raw.find("Content-Type: application/json") != std::string::npos, "fake delayed image server content type");
    expect(post_raw.find(expected_body_fragment) != std::string::npos, "fake delayed image server body fragment");
    if (!unexpected_body_fragment.empty()) {
        expect(post_raw.find(unexpected_body_fragment) == std::string::npos, "fake delayed image server absent body fragment");
    }
    if (!expected_authorization.empty()) {
        expect(post_raw.find(expected_authorization) != std::string::npos, "fake delayed image server post auth");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    std::string post_response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + std::to_string(response_body.size()) + "\r\nConnection: close\r\n\r\n" + response_body;
    send(post_fd, post_response.data(), post_response.size(), 0);
    close(post_fd);
    close(server_fd);
}

static void fake_asset_worker_once(
    int port,
    const std::string &post_path,
    const std::string &post_response_body,
    const std::string &asset_path,
    const std::string &asset_body
) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake asset worker socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake asset worker inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake asset worker bind");
    expect(listen(server_fd, 2) == 0, "fake asset worker listen");

    int post_fd = accept(server_fd, nullptr, nullptr);
    expect(post_fd >= 0, "fake asset worker post accept");
    char buffer[4096];
    std::string post_raw;
    ssize_t post_n = recv(post_fd, buffer, sizeof(buffer), 0);
    if (post_n > 0) {
        post_raw.append(buffer, static_cast<size_t>(post_n));
    }
    expect(post_raw.find("POST " + post_path + " HTTP/1.1") != std::string::npos, "fake asset worker post path");
    std::string post_response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + std::to_string(post_response_body.size()) + "\r\nConnection: close\r\n\r\n" + post_response_body;
    send(post_fd, post_response.data(), post_response.size(), 0);
    close(post_fd);

    int get_fd = accept(server_fd, nullptr, nullptr);
    expect(get_fd >= 0, "fake asset worker get accept");
    std::string get_raw;
    ssize_t get_n = recv(get_fd, buffer, sizeof(buffer), 0);
    if (get_n > 0) {
        get_raw.append(buffer, static_cast<size_t>(get_n));
    }
    expect(get_raw.find("GET " + asset_path + " HTTP/1.1") != std::string::npos, "fake asset worker get path");
    std::string get_response = "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: "
        + std::to_string(asset_body.size()) + "\r\nConnection: close\r\n\r\n" + asset_body;
    send(get_fd, get_response.data(), get_response.size(), 0);
    close(get_fd);
    close(server_fd);
}

static void fake_json_worker_many(int port, const std::vector<std::string> &bodies) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake worker socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake worker inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake worker bind");
    expect(listen(server_fd, 1) == 0, "fake worker listen");
    for (const auto &body : bodies) {
        int client_fd = accept(server_fd, nullptr, nullptr);
        expect(client_fd >= 0, "fake worker accept");
        char buffer[2048];
        recv(client_fd, buffer, sizeof(buffer), 0);
        std::string response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
            + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
        send(client_fd, response.data(), response.size(), 0);
        close(client_fd);
    }
    close(server_fd);
}

static void fake_get_json_worker_many_no_content_length(int port, const std::vector<std::string> &bodies) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake get worker socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake get worker inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake get worker bind");
    expect(listen(server_fd, 2) == 0, "fake get worker listen");
    for (const auto &body : bodies) {
        int client_fd = accept(server_fd, nullptr, nullptr);
        expect(client_fd >= 0, "fake get worker accept");
        std::string raw;
        char buffer[2048];
        while (raw.find("\r\n\r\n") == std::string::npos) {
            ssize_t n = recv(client_fd, buffer, sizeof(buffer), 0);
            expect(n > 0, "fake get worker recv");
            raw.append(buffer, static_cast<size_t>(n));
        }
        expect(raw.find("GET ") == 0, "fake get worker method");
        expect(raw.find("Content-Length:") == std::string::npos, "fake get worker omits content length");
        std::string response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
            + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
        send(client_fd, response.data(), response.size(), 0);
        close(client_fd);
    }
    close(server_fd);
}

static void fake_get_json_worker_once_no_content_length(
    int port,
    const std::string &expected_request_line,
    const std::string &body
) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake get once worker socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake get once worker inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake get once worker bind");
    expect(listen(server_fd, 1) == 0, "fake get once worker listen");
    int client_fd = accept(server_fd, nullptr, nullptr);
    expect(client_fd >= 0, "fake get once worker accept");
    std::string raw;
    char buffer[2048];
    while (raw.find("\r\n\r\n") == std::string::npos) {
        ssize_t n = recv(client_fd, buffer, sizeof(buffer), 0);
        expect(n > 0, "fake get once worker recv");
        raw.append(buffer, static_cast<size_t>(n));
    }
    expect(raw.find(expected_request_line) != std::string::npos, "fake get once worker request line");
    expect(raw.find("Content-Length:") == std::string::npos, "fake get once worker omits content length");
    std::string response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
    send(client_fd, response.data(), response.size(), 0);
    close(client_fd);
    close(server_fd);
}

static void serve_llama_startup_probe(int listener) {
    const int peer = accept(listener, nullptr, nullptr);
    expect(peer >= 0, "managed startup probe accepted");
    const auto request = recv_http_request_with_body(peer);
    expect(request.find("POST /v1/completions HTTP/1.1") == 0, "managed startup completion path");
    const auto body = request.find("\r\n\r\n");
    expect(body != std::string::npos && request.substr(body + 4) == R"({"prompt":"Hi","max_tokens":1})", "source startup probe payload");
    const std::string response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}";
    send(peer, response.data(), response.size(), 0); close(peer);
}

static void fake_llama_managed_server_once(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake managed llama socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake managed llama inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake managed llama bind");
    expect(listen(server_fd, 2) == 0, "fake managed llama listen");

    int models_fd = accept(server_fd, nullptr, nullptr);
    expect(models_fd >= 0, "fake managed llama models accept");
    std::string models_raw;
    char buffer[2048];
    while (models_raw.find("\r\n\r\n") == std::string::npos) {
        ssize_t n = recv(models_fd, buffer, sizeof(buffer), 0);
        expect(n > 0, "fake managed llama models recv");
        models_raw.append(buffer, static_cast<size_t>(n));
    }
    expect(models_raw.find("GET /v1/models HTTP/1.1") != std::string::npos, "fake managed llama models path");
    expect(models_raw.find("Content-Length:") == std::string::npos, "fake managed llama models omits content length");
    const std::string models_body = "{\"object\":\"list\",\"data\":[{\"id\":\"native-test-model\"}]}";
    std::string models_response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + std::to_string(models_body.size()) + "\r\nConnection: close\r\n\r\n" + models_body;
    send(models_fd, models_response.data(), models_response.size(), 0);
    close(models_fd);
        serve_llama_startup_probe(server_fd);

    int chat_fd = accept(server_fd, nullptr, nullptr);
    expect(chat_fd >= 0, "fake managed llama chat accept");
    std::string chat_raw = recv_http_request_with_body(chat_fd);
    expect(chat_raw.find("POST /v1/chat/completions HTTP/1.1") != std::string::npos, "fake managed llama chat path");
    expect(chat_raw.find("\"messages\"") != std::string::npos, "fake managed llama chat body");
    const std::string chat_body = "{\"id\":\"chatcmpl-managed\",\"object\":\"chat.completion\",\"model\":\"native-test-model\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"managed llama\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}";
    std::string chat_response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + std::to_string(chat_body.size()) + "\r\nConnection: close\r\n\r\n" + chat_body;
    send(chat_fd, chat_response.data(), chat_response.size(), 0);
    close(chat_fd);
    close(server_fd);
}

static void fake_llama_managed_server_once_wait_args(int port, const std::filesystem::path &args_path) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake managed llama wait socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake managed llama wait inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake managed llama wait bind");
    expect(listen(server_fd, 2) == 0, "fake managed llama wait listen");

    int models_fd = accept(server_fd, nullptr, nullptr);
    expect(models_fd >= 0, "fake managed llama wait models accept");
    std::string models_raw;
    char buffer[2048];
    while (models_raw.find("\r\n\r\n") == std::string::npos) {
        ssize_t n = recv(models_fd, buffer, sizeof(buffer), 0);
        expect(n > 0, "fake managed llama wait models recv");
        models_raw.append(buffer, static_cast<size_t>(n));
    }
    expect(models_raw.find("GET /v1/models HTTP/1.1") != std::string::npos, "fake managed llama wait models path");
    for (int wait = 0; wait < 40 && !std::filesystem::exists(args_path); ++wait) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    expect(std::filesystem::exists(args_path), "fake managed llama wait args recorded");
    const std::string models_body = "{\"object\":\"list\",\"data\":[{\"id\":\"native-test-model\"}]}";
    std::string models_response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + std::to_string(models_body.size()) + "\r\nConnection: close\r\n\r\n" + models_body;
    send(models_fd, models_response.data(), models_response.size(), 0);
    close(models_fd);
        serve_llama_startup_probe(server_fd);

    int chat_fd = accept(server_fd, nullptr, nullptr);
    expect(chat_fd >= 0, "fake managed llama wait chat accept");
    std::string chat_raw = recv_http_request_with_body(chat_fd);
    expect(chat_raw.find("POST /v1/chat/completions HTTP/1.1") != std::string::npos, "fake managed llama wait chat path");
    expect(chat_raw.find("\"messages\"") != std::string::npos, "fake managed llama wait chat body");
    const std::string chat_body = "{\"id\":\"chatcmpl-managed\",\"object\":\"chat.completion\",\"model\":\"native-test-model\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"managed llama\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}";
    std::string chat_response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + std::to_string(chat_body.size()) + "\r\nConnection: close\r\n\r\n" + chat_body;
    send(chat_fd, chat_response.data(), chat_response.size(), 0);
    close(chat_fd);
    close(server_fd);
}

static int count_context_retry_launches(const std::filesystem::path &args_path) {
    std::ifstream args_file(args_path, std::ios::binary);
    std::string args((std::istreambuf_iterator<char>(args_file)), std::istreambuf_iterator<char>());
    int launches = 0;
    size_t pos = 0;
    while ((pos = args.find("launch\n", pos)) != std::string::npos) {
        ++launches;
        pos += 7;
    }
    return launches;
}

static void fake_llama_managed_context_retry_server(
    int port,
    std::filesystem::path args_path,
    std::string expected_request_line,
    std::string expected_body_fragment,
    std::string failure_body,
    std::string success_body,
    int cycles
) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake managed llama retry socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake managed llama retry inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake managed llama retry bind");
    expect(listen(server_fd, 4) == 0, "fake managed llama retry listen");

    const auto send_json = [](int fd, int status, const std::string &body) {
        const std::string reason = status == 200 ? "OK" : "Internal Server Error";
        const std::string response = "HTTP/1.1 " + std::to_string(status) + " " + reason
            + "\r\nContent-Type: application/json\r\nContent-Length: "
            + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
        send(fd, response.data(), response.size(), 0);
    };

    for (int cycle = 0; cycle < cycles; ++cycle) {
        int models_fd = accept(server_fd, nullptr, nullptr);
        expect(models_fd >= 0, "fake managed llama retry models accept");
        std::string models_raw;
        char buffer[2048];
        while (models_raw.find("\r\n\r\n") == std::string::npos) {
            ssize_t n = recv(models_fd, buffer, sizeof(buffer), 0);
            expect(n > 0, "fake managed llama retry models recv");
            models_raw.append(buffer, static_cast<size_t>(n));
        }
        expect(models_raw.find("GET /v1/models HTTP/1.1") != std::string::npos, "fake managed llama retry models path");
        for (int wait = 0; wait < 40 && count_context_retry_launches(args_path) < cycle + 1; ++wait) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        expect(count_context_retry_launches(args_path) >= cycle + 1, "fake managed llama retry launch recorded");
        send_json(models_fd, 200, "{\"object\":\"list\",\"data\":[{\"id\":\"native-test-model\"}]}");
        close(models_fd);
        serve_llama_startup_probe(server_fd);

        int chat_fd = accept(server_fd, nullptr, nullptr);
        expect(chat_fd >= 0, "fake managed llama retry chat accept");
        std::string chat_raw = recv_http_request_with_body(chat_fd);
        expect(chat_raw.find(expected_request_line) != std::string::npos, "fake managed llama retry request path");
        expect(chat_raw.find(expected_body_fragment) != std::string::npos, "fake managed llama retry body");
        if (cycle == 0) {
            send_json(chat_fd, 500, failure_body);
        } else {
            send_json(chat_fd, 200, success_body);
        }
        close(chat_fd);
    }
    close(server_fd);
}

static void fake_llama_managed_embedding_server_once(int port, std::filesystem::path args_path) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    expect(server_fd >= 0, "fake managed embedding socket");
    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    expect(inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fake managed embedding inet_pton");
    expect(bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "fake managed embedding bind");
    expect(listen(server_fd, 2) == 0, "fake managed embedding listen");

    int models_fd = accept(server_fd, nullptr, nullptr);
    expect(models_fd >= 0, "fake managed embedding models accept");
    std::string models_raw;
    char buffer[2048];
    while (models_raw.find("\r\n\r\n") == std::string::npos) {
        ssize_t n = recv(models_fd, buffer, sizeof(buffer), 0);
        expect(n > 0, "fake managed embedding models recv");
        models_raw.append(buffer, static_cast<size_t>(n));
    }
    expect(models_raw.find("GET /v1/models HTTP/1.1") != std::string::npos, "fake managed embedding models path");
    for (int wait = 0; wait < 40 && count_context_retry_launches(args_path) < 1; ++wait) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    expect(count_context_retry_launches(args_path) >= 1, "fake managed embedding launch recorded");
    const std::string models_body = "{\"object\":\"list\",\"data\":[{\"id\":\"bge-m3\"}]}";
    std::string models_response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + std::to_string(models_body.size()) + "\r\nConnection: close\r\n\r\n" + models_body;
    send(models_fd, models_response.data(), models_response.size(), 0);
    close(models_fd);

    int embedding_fd = accept(server_fd, nullptr, nullptr);
    expect(embedding_fd >= 0, "fake managed embedding post accept");
    std::string embedding_raw = recv_http_request_with_body(embedding_fd);
    expect(embedding_raw.find("POST /v1/embeddings HTTP/1.1") != std::string::npos, "fake managed embedding post path");
    expect(embedding_raw.find("\"input\":\"auto embed\"") != std::string::npos, "fake managed embedding post body");
    const std::string embedding_body = "{\"object\":\"list\",\"model\":\"bge-m3\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.9]}],\"usage\":{\"prompt_tokens\":2,\"total_tokens\":2}}";
    std::string embedding_response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + std::to_string(embedding_body.size()) + "\r\nConnection: close\r\n\r\n" + embedding_body;
    send(embedding_fd, embedding_response.data(), embedding_response.size(), 0);
    close(embedding_fd);
    close(server_fd);
}

int main() {
    {
        int sockets[2];
        expect(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "coalesced websocket socketpair");
        const std::string frames = std::string("\x81\x03") + "one" + "\x81\x03" + "two";
        expect(send(sockets[0], frames.data(), frames.size(), 0) == static_cast<ssize_t>(frames.size()),
               "coalesced websocket send");
        shutdown(sockets[0], SHUT_WR);
        expect(recv_websocket_text(sockets[1]) == "one", "coalesced websocket first frame");
        expect(recv_websocket_text(sockets[1]) == "two", "coalesced websocket second frame");
        close(sockets[0]);
        close(sockets[1]);
    }
    const auto original_cwd = std::filesystem::current_path();
    const auto test_cwd = std::filesystem::temp_directory_path()
        / ("ezlocalai-native-http-" + std::to_string(static_cast<long long>(getpid())));
    std::filesystem::remove_all(test_cwd);
    std::filesystem::create_directories(test_cwd);
    std::filesystem::current_path(test_cwd);
    // Model fixtures belong to this test's temporary directory, including
    // inside images whose production defaults point at /app/models.
    setenv("MODELS_PATH", "./models", 1);
    setenv("HF_HUB_OFFLINE", "1", 1);
    setenv("ROUTER_DRAIN_FILE", "router-drains.json", 1);

    std::filesystem::remove_all("voices");
    std::filesystem::remove_all("outputs");
    std::filesystem::remove_all("wakeword_models");
    std::filesystem::remove_all("wakeword_jobs");
    std::filesystem::remove("usage-native-test.json");
    std::filesystem::remove("usage-native-test.history.json");
    setenv("USAGE_FILE", "usage-native-test.json", 1);
    {
        std::ofstream usage("usage-native-test.json", std::ios::binary);
        usage << "{\"Z Prior\":{\"custom\":{\"first\":true}},\"Prior Worker\":{\"llm\":{\"z-model\":{\"requests\":1,\"prompt_tokens\":1,\"completion_tokens\":1,\"prompt_tps_sum\":0,\"prompt_tps_n\":0,\"predicted_tps_sum\":0,\"predicted_tps_n\":0},\"prior-model\":{\"requests\":2,\"prompt_tokens\":11,\"completion_tokens\":13,\"prompt_tps_sum\":0,\"prompt_tps_n\":0,\"predicted_tps_sum\":0,\"predicted_tps_n\":0}},\"image\":{\"requests\":3},\"custom\":{\"kept\":true}}}";
        std::ofstream history("usage-native-test.history.json", std::ios::binary);
        history << "[{\"ts\":1,\"worker\":\"Prior Worker\",\"model\":\"prior-model\",\"prompt_tokens\":11,\"completion_tokens\":13,\"prompt_ms\":0,\"predicted_ms\":0,\"total_ms\":0,\"prompt_tps\":0,\"predicted_tps\":0,\"note\":\"kept\"}]";
    }

    ezl_env_config config{};
    ezl_load_config(&config);
    std::snprintf(config.api_key, sizeof(config.api_key), "%s", "secret");
    setenv("EZLOCALAI_API_KEY", "secret", 1);
    std::snprintf(config.default_model, sizeof(config.default_model), "%s", "native-test-model@4096,native-test-model,second-model");
    config.request_timeout_ms = 2000;
    config.max_concurrent_requests = 1;
    config.max_queue_size = 8;

    const char *saved_native_backend_env = getenv("EZLOCALAI_NATIVE_BACKEND");
    const bool had_native_backend_env = saved_native_backend_env != nullptr;
    const std::string saved_native_backend = saved_native_backend_env ? saved_native_backend_env : "";
    unsetenv("EZLOCALAI_NATIVE_BACKEND");
    expect(ezlocalai::LlamaCppInferenceBackend::should_use_from_env(config), "native defaults to llama.cpp backend");
    setenv("EZLOCALAI_NATIVE_BACKEND", "echo", 1);
    expect(!ezlocalai::LlamaCppInferenceBackend::should_use_from_env(config), "explicit echo backend opt-out");
    setenv("EZLOCALAI_NATIVE_BACKEND", "llama.cpp", 1);
    expect(ezlocalai::LlamaCppInferenceBackend::should_use_from_env(config), "explicit llama.cpp backend");
    if (had_native_backend_env) {
        setenv("EZLOCALAI_NATIVE_BACKEND", saved_native_backend.c_str(), 1);
    } else {
        unsetenv("EZLOCALAI_NATIVE_BACKEND");
    }

    std::filesystem::create_directories("models/vision-model");
    {
        std::ofstream mmproj("models/vision-model/mmproj-test.gguf", std::ios::binary);
        mmproj << "GGUF";
    }
    const char *saved_n_parallel_env = getenv("N_PARALLEL");
    const bool had_n_parallel_env = saved_n_parallel_env != nullptr;
    const std::string saved_n_parallel = saved_n_parallel_env ? saved_n_parallel_env : "";
    const char *saved_tts_enabled_env = getenv("TTS_ENABLED");
    const bool had_tts_enabled_env = saved_tts_enabled_env != nullptr;
    const std::string saved_tts_enabled = saved_tts_enabled_env ? saved_tts_enabled_env : "";
    const char *saved_stt_enabled_env = getenv("STT_ENABLED");
    const bool had_stt_enabled_env = saved_stt_enabled_env != nullptr;
    const std::string saved_stt_enabled = saved_stt_enabled_env ? saved_stt_enabled_env : "";
    setenv("N_PARALLEL", "2,3", 1);
    setenv("TTS_ENABLED", "false", 1);
    setenv("STT_ENABLED", "false", 1);
    {
        ezl_env_config vision_config = config;
        std::snprintf(vision_config.default_model, sizeof(vision_config.default_model), "%s", "org/vision-model-GGUF,plain-model");
        vision_config.max_concurrent_requests = 5;
        ezlocalai::NativeServer vision_resources_server(
            vision_config,
            std::make_unique<ezlocalai::EchoInferenceBackend>(vision_config.default_model)
        );
        const auto vision_resources = vision_resources_server.handle_request(
            "GET",
            "/v1/resources",
            "Bearer secret",
            "",
            ""
        );
        expect(vision_resources.status == 200, "vision resources status");
        expect(
            vision_resources.body.find("\"org/vision-model-GGUF\":{\"capacity\":2,\"in_flight\":0,\"queued\":0,\"available\":2}") != std::string::npos,
            "vision resources model slot"
        );
        expect(
            vision_resources.body.find("\"vision\":{\"capacity\":2,\"in_flight\":0,\"queued\":0,\"available\":2}") != std::string::npos,
            "vision resources cap slot"
        );
        expect(vision_resources.body.find("\"slot_total_capacity\":6") != std::string::npos, "vision resources avoids double-counted capacity");
    }
    if (had_n_parallel_env) {
        setenv("N_PARALLEL", saved_n_parallel.c_str(), 1);
    } else {
        unsetenv("N_PARALLEL");
    }
    if (had_tts_enabled_env) {
        setenv("TTS_ENABLED", saved_tts_enabled.c_str(), 1);
    } else {
        unsetenv("TTS_ENABLED");
    }
    if (had_stt_enabled_env) {
        setenv("STT_ENABLED", saved_stt_enabled.c_str(), 1);
    } else {
        unsetenv("STT_ENABLED");
    }

    setenv("LLAMA_CPP_SERVER_URL", "http://127.0.0.1:18139", 1);
    std::thread fake_llama_server(
        fake_expect_worker_once_with_type,
        18139,
        "POST /v1/chat/completions HTTP/1.1",
        "Content-Type: application/json",
        "\"messages\"",
        "application/json",
        "{\"id\":\"chatcmpl-llama\",\"object\":\"chat.completion\",\"model\":\"native-test-model\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"llama.cpp local\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3,\"total_tokens\":5}}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ezlocalai::LlamaCppInferenceBackend llama_backend(config.default_model);
    const auto llama_chat = llama_backend.chat("{\"model\":\"native-test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}");
    fake_llama_server.join();
    expect(llama_chat.find("\"content\":\"llama.cpp local\"") != std::string::npos, "llama.cpp backend forwards chat");
    unsetenv("LLAMA_CPP_SERVER_URL");

    setenv("LLAMA_CPP_SERVER_URL", "http://127.0.0.1:18140", 1);
    std::thread fake_llama_error_server(
        fake_json_worker_once_status,
        18140,
        503,
        "{\"error\":{\"message\":\"llama load failed\",\"type\":\"server_error\",\"code\":503}}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ezlocalai::NativeServer llama_error_server(
        config,
        std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model)
    );
    const auto llama_error = llama_error_server.handle_request(
        "POST",
        "/v1/chat/completions",
        "Bearer secret",
        "{\"model\":\"native-test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}",
        "application/json"
    );
    fake_llama_error_server.join();
    // Python returns a 200 "Local resources exhausted" completion here, which
    // WorkerRouter treats as success. A retryable status lets it fail over.
    expect(llama_error.status == 503, "llama.cpp inference error without a fallback server is retryable");
    expect(llama_error.body == "{\"detail\":{\"type\":\"inference_error\",\"message\":\"llama load failed\"}}", "llama.cpp unconfigured post-inference failure body");
    const auto llama_error_resources = llama_error_server.handle_request(
        "GET",
        "/v1/resources",
        "Bearer secret",
        "",
        ""
    );
    expect(llama_error_resources.status == 200, "llama.cpp error resources status");
    expect(llama_error_resources.body.find("\"loaded_models\":{}") != std::string::npos, "llama.cpp failed request does not mark model loaded");
    unsetenv("LLAMA_CPP_SERVER_URL");

    setenv("LLAMA_CPP_SERVER_URL", "http://127.0.0.1:18142", 1);
    std::thread fake_llama_top_error_server(
        fake_json_worker_once_status,
        18142,
        404,
        "{\"code\":404,\"message\":\"model not found\",\"type\":\"not_found_error\"}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ezlocalai::NativeServer llama_top_error_server(
        config,
        std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model)
    );
    const auto llama_top_error = llama_top_error_server.handle_request(
        "POST",
        "/v1/chat/completions",
        "Bearer secret",
        "{\"model\":\"native-test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}",
        "application/json"
    );
    fake_llama_top_error_server.join();
    expect(llama_top_error.status == 200, "buffered wrapper ignores top-level codes other than 400");
    expect(llama_top_error.body.find("\"type\":\"not_found_error\"") != std::string::npos, "llama.cpp top-level error body");
    unsetenv("LLAMA_CPP_SERVER_URL");

    setenv("LLAMA_CPP_SERVER_URL", "http://127.0.0.1:18141", 1);
    std::thread fake_llama_stream_server(
        fake_expect_worker_once_with_type,
        18141,
        "POST /v1/chat/completions HTTP/1.1",
        "Content-Type: application/json",
        "\"stream\":true",
        "text/event-stream",
        "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ezlocalai::NativeServer llama_stream_server(
        config,
        std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model)
    );
    const auto llama_stream = llama_stream_server.handle_request(
        "POST",
        "/v1/chat/completions",
        "Bearer secret",
        "{\"model\":\"native-test-model\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}",
        "application/json"
    );
    fake_llama_stream_server.join();
    expect(llama_stream.status == 200, "llama.cpp stream status");
    expect(llama_stream.content_type == "text/event-stream; charset=utf-8", "llama.cpp stream content type");
    expect(llama_stream.body.find("data: data:") == std::string::npos, "llama.cpp stream is not double wrapped");
    expect(llama_stream.body.find("delta") != std::string::npos, "llama.cpp stream body");
    unsetenv("LLAMA_CPP_SERVER_URL");

    setenv("LLAMA_CPP_SERVER_URL", "http://127.0.0.1:18147", 1);
    std::thread fake_llama_chunked_stream_server(
        fake_chunked_worker_once,
        18147,
        "text/event-stream",
        std::vector<std::string>{
            "data: {\"choices\":[{\"delta\":{\"content\":\"chunked\"}}]}\n\n",
            "data: [DONE]\n\n"
        }
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ezlocalai::NativeServer llama_chunked_stream_server(
        config,
        std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model)
    );
    const auto llama_chunked_stream = llama_chunked_stream_server.handle_request(
        "POST",
        "/v1/chat/completions",
        "Bearer secret",
        "{\"model\":\"native-test-model\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}",
        "application/json"
    );
    fake_llama_chunked_stream_server.join();
    expect(llama_chunked_stream.status == 200, "llama.cpp chunked stream status");
    expect(llama_chunked_stream.content_type == "text/event-stream; charset=utf-8", "llama.cpp chunked stream content type");
    expect(llama_chunked_stream.body.find("chunked") != std::string::npos, "llama.cpp chunked stream body decoded");
    expect(llama_chunked_stream.body.find("\r\n0\r\n\r\n") == std::string::npos, "llama.cpp chunked stream strips transfer coding");
    unsetenv("LLAMA_CPP_SERVER_URL");

    const auto fake_hf_llama_bin = test_cwd / "fake-hf-llama-server";
    const auto fake_hf_llama_args = test_cwd / "llama-hf-args.txt";
    {
        write_llama_fixture(fake_hf_llama_bin, fake_hf_llama_args, false, 0, false);
    }
    std::filesystem::permissions(
        fake_hf_llama_bin,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write | std::filesystem::perms::owner_exec,
        std::filesystem::perm_options::replace
    );
    setenv("LLAMA_CPP_SERVER_BIN", fake_hf_llama_bin.string().c_str(), 1);
    setenv("LLAMA_CPP_SERVER_PORT", "18183", 1);
    setenv("LLAMA_CPP_START_TIMEOUT", "2", 1);
    unsetenv("LLAMA_CPP_MODEL_PATH");
    unsetenv("LLAMA_CPP_MODEL_PATHS");
    unsetenv("LLAMA_CPP_USE_HF");
    std::filesystem::create_directories("models/native-test-model");
    { std::ofstream out("models/native-test-model/native-test-model-Q4_K_XL.gguf"); out << "GGUF fixture"; }
    { std::ofstream out("models/native-test-model/mmproj-local.gguf"); out << "projector fixture"; }
    std::thread fake_hf_llama_server(fake_llama_managed_server_once_wait_args, 18183, fake_hf_llama_args);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    {
        ezlocalai::LlamaCppInferenceBackend hf_llama_backend(config.default_model);
        const auto hf_llama_chat = hf_llama_backend.chat("{\"model\":\"native-test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"hf managed\"}]}");
        expect(hf_llama_chat.find("\"content\":\"managed llama\"") != std::string::npos, "managed llama.cpp defaults to hf loader");
    }
    fake_hf_llama_server.join();
    {
        std::ifstream args_file(fake_hf_llama_args, std::ios::binary);
        std::string args((std::istreambuf_iterator<char>(args_file)), std::istreambuf_iterator<char>());
        expect(args.find("-hf\n") == std::string::npos, "managed text acquisition precedes engine launch");
        expect(args.find("-m\n./models/native-test-model/native-test-model-Q4_K_XL.gguf") != std::string::npos, "managed text uses source-selected local model");
        expect(args.find("--mmproj\n./models/native-test-model/mmproj-local.gguf") != std::string::npos, "managed text discovers local projector");
        expect(args.find("--alias\nnative-test-model") != std::string::npos, "managed llama.cpp hf alias flag");
        expect(args.find("-c\n65536") != std::string::npos, "managed llama.cpp uses configured reference context");
        expect(args.find("-b\n") != std::string::npos, "managed llama.cpp passes auto batch size");
        expect(args.find("-ub\n") != std::string::npos, "managed llama.cpp passes auto ubatch size");
        expect(args.find("-fa\non") != std::string::npos, "managed llama.cpp enables flash attention by default");
        expect(args.find("-ctk\nq4_0") != std::string::npos, "managed llama.cpp defaults k cache to q4_0");
        expect(args.find("-ctv\nq4_0") != std::string::npos, "managed llama.cpp defaults v cache to q4_0");
    }
    unsetenv("LLAMA_CPP_SERVER_BIN");
    unsetenv("LLAMA_CPP_SERVER_PORT");
    unsetenv("LLAMA_CPP_START_TIMEOUT");

    const auto fake_managed_llama_bin = test_cwd / "fake-managed-llama-server";
    {
        write_llama_fixture(fake_managed_llama_bin, {}, false, 0, false);
    }
    std::filesystem::permissions(
        fake_managed_llama_bin,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write | std::filesystem::perms::owner_exec,
        std::filesystem::perm_options::replace
    );
    setenv("LLAMA_CPP_SERVER_BIN", fake_managed_llama_bin.string().c_str(), 1);
    setenv("LLAMA_CPP_MODEL_PATH", "managed-model.gguf", 1);
    setenv("LLAMA_CPP_SERVER_PORT", "18151", 1);
    setenv("LLAMA_CPP_START_TIMEOUT", "2", 1);
    std::thread fake_managed_llama_server(fake_llama_managed_server_once, 18151);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    {
        ezlocalai::LlamaCppInferenceBackend managed_llama_backend(config.default_model);
        const auto managed_llama_chat = managed_llama_backend.chat("{\"model\":\"native-test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"managed\"}]}");
        expect(managed_llama_chat.find("\"content\":\"managed llama\"") != std::string::npos, "managed llama.cpp backend forwards after readiness");
    }
    fake_managed_llama_server.join();
    unsetenv("LLAMA_CPP_SERVER_BIN");
    unsetenv("LLAMA_CPP_MODEL_PATH");
    unsetenv("LLAMA_CPP_SERVER_PORT");
    unsetenv("LLAMA_CPP_START_TIMEOUT");

    const auto fake_context_retry_llama_bin = test_cwd / "fake-context-retry-llama-server";
    const auto fake_context_retry_args = test_cwd / "llama-context-retry-args.txt";
    {
        write_llama_fixture(fake_context_retry_llama_bin, fake_context_retry_args, true, 0, false);
    }
    std::filesystem::permissions(
        fake_context_retry_llama_bin,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write | std::filesystem::perms::owner_exec,
        std::filesystem::perm_options::replace
    );
    setenv("LLAMA_CPP_SERVER_BIN", fake_context_retry_llama_bin.string().c_str(), 1);
    setenv("LLAMA_CPP_MODEL_PATH", "managed-model.gguf", 1);
    setenv("LLAMA_CPP_SERVER_PORT", "18173", 1);
    setenv("LLAMA_CPP_CONTEXT", "4096", 1);
    setenv("LLAMA_CPP_START_TIMEOUT", "2", 1);
    std::thread fake_context_retry_server(
        fake_llama_managed_context_retry_server,
        18173,
        fake_context_retry_args,
        "POST /v1/chat/completions HTTP/1.1",
        "\"messages\"",
        "{\"error\":{\"message\":\"prompt exceeds context [n_prompt_tokens=21922, n_ctx=4096]\",\"type\":\"invalid_request_error\",\"code\":500}}",
        "{\"id\":\"chatcmpl-managed-retry\",\"object\":\"chat.completion\",\"model\":\"native-test-model\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"resized context\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}",
        1
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    {
        ezlocalai::NativeServer managed_retry_native_server(
            config,
            std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model)
        );
        const auto managed_retry_chat = managed_retry_native_server.handle_request(
            "POST",
            "/v1/chat/completions",
            "Bearer secret",
            "{\"model\":\"native-test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"large context\"}]}",
            "application/json"
        );
        // Python answers 200 "Unable to process request." here. WorkerRouter
        // needs the source's context_capacity_error shape to stop treating the
        // overflow as a successful completion.
        expect(managed_retry_chat.status == 413, "managed context error returns context capacity status");
        expect(managed_retry_chat.body == "{\"detail\":{\"type\":\"context_capacity_error\",\"message\":\"prompt exceeds context [n_prompt_tokens=21922, n_ctx=4096]\",\"n_prompt_tokens\":21922,\"n_ctx\":4096,\"compact_and_retry\":true}}", "managed context error reports token counts without resizing");
        const auto managed_retry_resources = managed_retry_native_server.handle_request(
            "GET",
            "/v1/resources",
            "Bearer secret",
            "",
            ""
        );
        expect(managed_retry_resources.body.find("\"parallel\":{\"n_parallel\":1,\"n_ctx\":4096,\"per_slot_context\":4096") != std::string::npos, "managed llama.cpp resources retains context ceiling");
    }
    fake_context_retry_server.join();
    {
        std::ifstream args_file(fake_context_retry_args, std::ios::binary);
        std::string args((std::istreambuf_iterator<char>(args_file)), std::istreambuf_iterator<char>());
        expect(args.find("-c\n4096") != std::string::npos, "managed llama.cpp initial context flag");
        expect(args.find("-c\n30114") == std::string::npos && count_context_retry_launches(fake_context_retry_args) == 1, "managed context error does not relaunch above ceiling");
    }
    unsetenv("LLAMA_CPP_SERVER_BIN");
    unsetenv("LLAMA_CPP_MODEL_PATH");
    unsetenv("LLAMA_CPP_SERVER_PORT");
    unsetenv("LLAMA_CPP_CONTEXT");
    unsetenv("LLAMA_CPP_START_TIMEOUT");

    const auto fake_completion_retry_llama_bin = test_cwd / "fake-completion-retry-llama-server";
    const auto fake_completion_retry_args = test_cwd / "llama-completion-retry-args.txt";
    {
        write_llama_fixture(fake_completion_retry_llama_bin, fake_completion_retry_args, true, 0, false);
    }
    std::filesystem::permissions(
        fake_completion_retry_llama_bin,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write | std::filesystem::perms::owner_exec,
        std::filesystem::perm_options::replace
    );
    setenv("LLAMA_CPP_SERVER_BIN", fake_completion_retry_llama_bin.string().c_str(), 1);
    setenv("LLAMA_CPP_MODEL_PATH", "managed-model.gguf", 1);
    setenv("LLAMA_CPP_SERVER_PORT", "18174", 1);
    setenv("LLAMA_CPP_CONTEXT", "4096", 1);
    setenv("LLAMA_CPP_START_TIMEOUT", "2", 1);
    std::thread fake_completion_retry_server(
        fake_llama_managed_context_retry_server,
        18174,
        fake_completion_retry_args,
        "POST /v1/completions HTTP/1.1",
        "\"prompt\":\"large completion\"",
        "{\"error\":{\"message\":\"prompt exceeds context [n_prompt_tokens=21922, n_ctx=4096]\",\"type\":\"invalid_request_error\",\"code\":500}}",
        "{\"id\":\"cmpl-managed-retry\",\"object\":\"text_completion\",\"model\":\"native-test-model\",\"choices\":[{\"text\":\"resized completion\",\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}",
        1
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    {
        ezlocalai::NativeServer managed_completion_retry_server(
            config,
            std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model)
        );
        const auto managed_retry_completion = managed_completion_retry_server.handle_request(
            "POST",
            "/v1/completions",
            "Bearer secret",
            "{\"model\":\"native-test-model\",\"prompt\":\"large completion\"}",
            "application/json"
        );
        expect(managed_retry_completion.status == 413, "managed completion context error returns context capacity status");
        expect(managed_retry_completion.body.find("\"type\":\"context_capacity_error\"") != std::string::npos
            && managed_retry_completion.body.find("\"n_ctx\":4096,\"compact_and_retry\":true") != std::string::npos, "managed completion context error reports token counts without resizing");
        const auto managed_completion_retry_resources = managed_completion_retry_server.handle_request(
            "GET",
            "/v1/resources",
            "Bearer secret",
            "",
            ""
        );
        expect(managed_completion_retry_resources.body.find("\"parallel\":{\"n_parallel\":1,\"n_ctx\":4096,\"per_slot_context\":4096") != std::string::npos, "managed llama.cpp completion resources retains context ceiling");
    }
    fake_completion_retry_server.join();
    {
        std::ifstream args_file(fake_completion_retry_args, std::ios::binary);
        std::string args((std::istreambuf_iterator<char>(args_file)), std::istreambuf_iterator<char>());
        expect(args.find("-c\n4096") != std::string::npos, "managed llama.cpp completion initial context flag");
        expect(args.find("-c\n30114") == std::string::npos && count_context_retry_launches(fake_completion_retry_args) == 1, "managed completion error does not relaunch above ceiling");
    }
    unsetenv("LLAMA_CPP_SERVER_BIN");
    unsetenv("LLAMA_CPP_MODEL_PATH");
    unsetenv("LLAMA_CPP_SERVER_PORT");
    unsetenv("LLAMA_CPP_CONTEXT");
    unsetenv("LLAMA_CPP_START_TIMEOUT");

    const auto fake_oom_retry_llama_bin = test_cwd / "fake-oom-retry-llama-server";
    const auto fake_oom_retry_args = test_cwd / "llama-oom-retry-args.txt";
    {
        write_llama_fixture(fake_oom_retry_llama_bin, fake_oom_retry_args, true, 0, false);
    }
    std::filesystem::permissions(
        fake_oom_retry_llama_bin,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write | std::filesystem::perms::owner_exec,
        std::filesystem::perm_options::replace
    );
    const auto fake_oom_model_path = test_cwd / "managed-oom-model.gguf";
    {
        std::ofstream gguf(fake_oom_model_path, std::ios::binary);
        const auto write_u32 = [&gguf](uint32_t value) {
            gguf.write(reinterpret_cast<const char *>(&value), sizeof(value));
        };
        const auto write_u64 = [&gguf](uint64_t value) {
            gguf.write(reinterpret_cast<const char *>(&value), sizeof(value));
        };
        const std::string key = "llama.block_count";
        gguf.write("GGUF", 4);
        write_u32(3);
        write_u64(0);
        write_u64(1);
        write_u64(key.size());
        gguf.write(key.data(), static_cast<std::streamsize>(key.size()));
        write_u32(4);
        write_u32(31);
    }
    setenv("LLAMA_CPP_SERVER_BIN", fake_oom_retry_llama_bin.string().c_str(), 1);
    setenv("LLAMA_CPP_MODEL_PATH", fake_oom_model_path.string().c_str(), 1);
    setenv("LLAMA_CPP_SERVER_PORT", "18175", 1);
    setenv("LLAMA_CPP_CONTEXT", "4096", 1);
    setenv("LLAMA_CPP_START_TIMEOUT", "2", 1);
    setenv("GPU_LAYERS", "-1", 1);
    std::thread fake_oom_retry_server(
        fake_llama_managed_context_retry_server,
        18175,
        fake_oom_retry_args,
        "POST /v1/chat/completions HTTP/1.1",
        "\"messages\"",
        "{\"error\":{\"message\":\"CUDA out of memory while allocating KV cache\",\"type\":\"server_error\",\"code\":500}}",
        "{\"id\":\"chatcmpl-managed-oom-retry\",\"object\":\"chat.completion\",\"model\":\"native-test-model\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"cpu retry\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}",
        2
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    {
        auto single_model_config = config;
        std::snprintf(single_model_config.default_model, sizeof(single_model_config.default_model), "%s", "native-test-model");
        ezlocalai::NativeServer managed_oom_retry_server(
            single_model_config,
            std::make_unique<ezlocalai::LlamaCppInferenceBackend>(single_model_config.default_model)
        );
        const auto managed_oom_retry_chat = managed_oom_retry_server.handle_request(
            "POST",
            "/v1/chat/completions",
            "Bearer secret",
            "{\"model\":\"native-test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"oom retry\"}]}",
            "application/json"
        );
        expect(managed_oom_retry_chat.status == 200, "managed llama.cpp oom retry native status");
        expect(managed_oom_retry_chat.body.find("\"content\":\"cpu retry\"") != std::string::npos, "managed llama.cpp retries oom errors");
    }
    fake_oom_retry_server.join();
    {
        std::ifstream args_file(fake_oom_retry_args, std::ios::binary);
        std::string args((std::istreambuf_iterator<char>(args_file)), std::istreambuf_iterator<char>());
        expect(args.find("-c\n4096") != std::string::npos, "managed llama.cpp oom initial context flag");
        expect(args.find("-ngl\n24") != std::string::npos, "managed llama.cpp oom retry reduces gpu layers from gguf layer count");
        expect(args.find("-ngl\n0") == std::string::npos, "managed llama.cpp oom retry does not jump straight to cpu");
    }
    unsetenv("LLAMA_CPP_SERVER_BIN");
    unsetenv("LLAMA_CPP_MODEL_PATH");
    unsetenv("LLAMA_CPP_SERVER_PORT");
    unsetenv("LLAMA_CPP_CONTEXT");
    unsetenv("LLAMA_CPP_START_TIMEOUT");
    unsetenv("GPU_LAYERS");

    const auto fake_startup_oom_llama_bin = test_cwd / "fake-startup-oom-llama-server";
    const auto fake_startup_oom_args = test_cwd / "llama-startup-oom-args.txt";
    {
        write_llama_fixture(fake_startup_oom_llama_bin, fake_startup_oom_args, true, 0, true);
    }
    std::filesystem::permissions(
        fake_startup_oom_llama_bin,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write | std::filesystem::perms::owner_exec,
        std::filesystem::perm_options::replace
    );
    setenv("LLAMA_CPP_SERVER_BIN", fake_startup_oom_llama_bin.string().c_str(), 1);
    setenv("LLAMA_CPP_MODEL_PATH", fake_oom_model_path.string().c_str(), 1);
    setenv("LLAMA_CPP_SERVER_PORT", "18178", 1);
    setenv("LLAMA_CPP_CONTEXT", "4096", 1);
    setenv("LLAMA_CPP_START_TIMEOUT", "2", 1);
    setenv("GPU_LAYERS", "-1", 1);
    std::thread fake_startup_oom_server([&fake_startup_oom_args]() {
        for (int wait = 0; wait < 40 && count_context_retry_launches(fake_startup_oom_args) < 2; ++wait) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        expect(count_context_retry_launches(fake_startup_oom_args) >= 2, "managed llama.cpp startup oom retry relaunch recorded");
        fake_llama_managed_server_once(18178);
    });
    {
        ezlocalai::NativeServer managed_startup_oom_server(
            config,
            std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model)
        );
        const auto managed_startup_oom_chat = managed_startup_oom_server.handle_request(
            "POST",
            "/v1/chat/completions",
            "Bearer secret",
            "{\"model\":\"native-test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"startup oom retry\"}]}",
            "application/json"
        );
        expect(managed_startup_oom_chat.status == 200, "managed llama.cpp startup oom retry native status");
        expect(managed_startup_oom_chat.body.find("\"content\":\"managed llama\"") != std::string::npos, "managed llama.cpp retries startup oom errors");
    }
    fake_startup_oom_server.join();
    {
        std::ifstream args_file(fake_startup_oom_args, std::ios::binary);
        std::string args((std::istreambuf_iterator<char>(args_file)), std::istreambuf_iterator<char>());
        expect(count_context_retry_launches(fake_startup_oom_args) == 2, "managed llama.cpp startup oom launches twice");
        expect(args.find("-c\n4096") != std::string::npos, "managed llama.cpp startup oom context flag");
        expect(args.find("-ngl\n24") == std::string::npos && args.find("-ngl\n-1") != std::string::npos, "managed llama.cpp startup retry retains full GPU residency");
        std::vector<int> ubatches;
        std::istringstream flags(args);
        for (std::string flag; std::getline(flags, flag);) if (flag == "-ub" && std::getline(flags, flag)) ubatches.push_back(std::stoi(flag));
        expect(ubatches.size() == 2 && ubatches[1] == std::max(128, ubatches[0] / 2), "managed llama.cpp startup retry halves micro-batch before reducing GPU layers");
        expect(args.find("-ngl\n0") == std::string::npos, "managed llama.cpp startup oom retry does not jump straight to cpu");
    }
    unsetenv("LLAMA_CPP_SERVER_BIN");
    unsetenv("LLAMA_CPP_MODEL_PATH");
    unsetenv("LLAMA_CPP_SERVER_PORT");
    unsetenv("LLAMA_CPP_CONTEXT");
    unsetenv("LLAMA_CPP_START_TIMEOUT");
    unsetenv("GPU_LAYERS");

    const auto fake_auto_gpu_layers_llama_bin = test_cwd / "fake-auto-gpu-layers-llama-server";
    const auto fake_auto_gpu_layers_args = test_cwd / "llama-auto-gpu-layers-args.txt";
    {
        write_llama_fixture(fake_auto_gpu_layers_llama_bin, fake_auto_gpu_layers_args, false, 127, false);
    }
    std::filesystem::permissions(
        fake_auto_gpu_layers_llama_bin,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write | std::filesystem::perms::owner_exec,
        std::filesystem::perm_options::replace
    );
    const std::string saved_auto_ngl_path = getenv("PATH") ? getenv("PATH") : "";
    {
        write_output_fixture("nvidia-smi", "8192, 4096\n");
    }
    std::filesystem::permissions(
        "nvidia-smi",
        std::filesystem::perms::owner_exec | std::filesystem::perms::group_exec | std::filesystem::perms::others_exec,
        std::filesystem::perm_options::add
    );
    setenv("PATH", test_cwd.string().c_str(), 1);
    setenv("LLAMA_CPP_SERVER_BIN", fake_auto_gpu_layers_llama_bin.string().c_str(), 1);
    setenv("LLAMA_CPP_MODEL_PATH", "managed-model.gguf", 1);
    setenv("LLAMA_CPP_SERVER_PORT", "18186", 1);
    setenv("LLAMA_CPP_START_TIMEOUT", "1", 1);
    unsetenv("GPU_LAYERS");
    {
        ezlocalai::NativeServer auto_gpu_layers_server(
            config,
            std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model)
        );
        const auto auto_gpu_layers_response = auto_gpu_layers_server.handle_request(
            "POST",
            "/v1/chat/completions",
            "Bearer secret",
            "{\"model\":\"native-test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"auto layers\"}]}",
            "application/json"
        );
        expect(auto_gpu_layers_response.status == 503, "managed llama.cpp auto gpu layers fake exit status");
    }
    {
        std::ifstream args_file(fake_auto_gpu_layers_args, std::ios::binary);
        std::string args((std::istreambuf_iterator<char>(args_file)), std::istreambuf_iterator<char>());
        expect(args.find("-ngl\n-1") != std::string::npos, "managed llama.cpp auto gpu layers uses all layers with free gpu");
    }
    std::filesystem::remove("nvidia-smi");
    if (saved_auto_ngl_path.empty()) {
        unsetenv("PATH");
    } else {
        setenv("PATH", saved_auto_ngl_path.c_str(), 1);
    }
    unsetenv("LLAMA_CPP_SERVER_BIN");
    unsetenv("LLAMA_CPP_MODEL_PATH");
    unsetenv("LLAMA_CPP_SERVER_PORT");
    unsetenv("LLAMA_CPP_START_TIMEOUT");

    const auto fake_gpu_layers_llama_bin = test_cwd / "fake-gpu-layers-llama-server";
    const auto fake_gpu_layers_args = test_cwd / "llama-gpu-layers-args.txt";
    {
        write_llama_fixture(fake_gpu_layers_llama_bin, fake_gpu_layers_args, false, 127, false);
    }
    std::filesystem::permissions(
        fake_gpu_layers_llama_bin,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write | std::filesystem::perms::owner_exec,
        std::filesystem::perm_options::replace
    );
    setenv("LLAMA_CPP_SERVER_BIN", fake_gpu_layers_llama_bin.string().c_str(), 1);
    setenv("LLAMA_CPP_MODEL_PATH", "managed-model.gguf", 1);
    setenv("LLAMA_CPP_SERVER_PORT", "18177", 1);
    setenv("LLAMA_CPP_START_TIMEOUT", "1", 1);
    setenv("GPU_LAYERS", "-2", 1);
    {
        ezlocalai::NativeServer gpu_layers_server(
            config,
            std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model)
        );
        const auto gpu_layers_response = gpu_layers_server.handle_request(
            "POST",
            "/v1/chat/completions",
            "Bearer secret",
            "{\"model\":\"native-test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"layers\"}]}",
            "application/json"
        );
        expect(gpu_layers_response.status == 503, "managed llama.cpp gpu layers fake exit status");
    }
    {
        std::ifstream args_file(fake_gpu_layers_args, std::ios::binary);
        std::string args((std::istreambuf_iterator<char>(args_file)), std::istreambuf_iterator<char>());
        expect(args.find("-ngl\n-1") != std::string::npos, "managed llama.cpp maps GPU_LAYERS=-2 to all-layers flag");
    }
    unsetenv("LLAMA_CPP_SERVER_BIN");
    unsetenv("LLAMA_CPP_MODEL_PATH");
    unsetenv("LLAMA_CPP_SERVER_PORT");
    unsetenv("LLAMA_CPP_START_TIMEOUT");
    unsetenv("GPU_LAYERS");

    const auto fake_llama_config_bin = test_cwd / "fake-llama-config-server";
    const auto fake_llama_config_args = test_cwd / "llama-config-args.txt";
    {
        write_llama_fixture(fake_llama_config_bin, fake_llama_config_args, false, 127, false);
    }
    std::filesystem::permissions(
        fake_llama_config_bin,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write | std::filesystem::perms::owner_exec,
        std::filesystem::perm_options::replace
    );
    setenv("LLAMA_CPP_SERVER_BIN", fake_llama_config_bin.string().c_str(), 1);
    setenv("LLAMA_CPP_MODEL_PATHS", "first.gguf,second.gguf,third.gguf", 1);
    setenv("LLAMA_CPP_SERVER_PORT", "18179", 1);
    setenv("LLAMA_CPP_CONTEXT", "1024,2048,4096", 1);
    setenv("MAIN_GPU", "0,1,2", 1);
    setenv("N_PARALLEL", "1,2,3", 1);
    setenv("TENSOR_SPLIT", "0.2,0.8", 1);
    setenv("LLM_BATCH_SIZE", "2048", 1);
    setenv("LLM_UBATCH_SIZE", "512", 1);
    setenv("LLAMA_CPP_START_TIMEOUT", "1", 1);
    {
        ezlocalai::NativeServer llama_config_server(
            config,
            std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model)
        );
        const auto llama_config_response = llama_config_server.handle_request(
            "POST",
            "/v1/chat/completions",
            "Bearer secret",
            "{\"model\":\"second-model\",\"messages\":[{\"role\":\"user\",\"content\":\"config\"}]}",
            "application/json"
        );
        expect(llama_config_response.status == 503, "managed llama.cpp config fake exit status");
    }
    {
        std::ifstream args_file(fake_llama_config_args, std::ios::binary);
        std::string args((std::istreambuf_iterator<char>(args_file)), std::istreambuf_iterator<char>());
        expect(args.find("-m\nthird.gguf") != std::string::npos, "managed llama.cpp config selects per-model path");
        expect(args.find("--port\n18181") != std::string::npos, "managed llama.cpp config selects per-model port");
        expect(args.find("-c\n4096") != std::string::npos, "managed llama.cpp config selects per-model context");
        expect(args.find("-np\n3") != std::string::npos, "managed llama.cpp config passes per-model parallel slots");
        expect(args.find("--main-gpu\n2") != std::string::npos, "managed llama.cpp config passes per-model main gpu");
        expect(args.find("--tensor-split\n0.2,0.8") != std::string::npos, "managed llama.cpp config passes tensor split");
        expect(args.find("-b\n2048") != std::string::npos, "managed llama.cpp config passes batch size");
        expect(args.find("-ub\n512") != std::string::npos, "managed llama.cpp config passes ubatch size");
    }
    unsetenv("LLAMA_CPP_SERVER_BIN");
    unsetenv("LLAMA_CPP_MODEL_PATHS");
    unsetenv("LLAMA_CPP_SERVER_PORT");
    unsetenv("LLAMA_CPP_CONTEXT");
    unsetenv("MAIN_GPU");
    unsetenv("N_PARALLEL");
    unsetenv("TENSOR_SPLIT");
    unsetenv("LLM_BATCH_SIZE");
    unsetenv("LLM_UBATCH_SIZE");
    unsetenv("LLAMA_CPP_START_TIMEOUT");

    const auto fake_llama_hf_quant_bin = test_cwd / "fake-llama-hf-quant-server";
    const auto fake_llama_hf_quant_args = test_cwd / "llama-hf-quant-args.txt";
    {
        write_llama_fixture(fake_llama_hf_quant_bin, fake_llama_hf_quant_args, false, 127, false);
    }
    std::filesystem::permissions(
        fake_llama_hf_quant_bin,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write | std::filesystem::perms::owner_exec,
        std::filesystem::perm_options::replace
    );
    setenv("LLAMA_CPP_SERVER_BIN", fake_llama_hf_quant_bin.string().c_str(), 1);
    setenv("LLAMA_CPP_SERVER_PORT", "18183", 1);
    setenv("QUANT_TYPE", "Q4_K_M,Q5_K_M", 1);
    setenv("LLAMA_CPP_START_TIMEOUT", "1", 1);
    std::filesystem::create_directories("models/second");
    { std::ofstream out("models/second/model-Q4_K_M.gguf"); out << "wrong quant fixture"; }
    { std::ofstream out("models/second/model-Q5_K_M.gguf"); out << "requested quant fixture"; }
    { std::ofstream out("models/second/mmproj-local.gguf"); out << "projector fixture"; }
    {
        ezl_env_config hf_quant_config = config;
        std::snprintf(hf_quant_config.default_model, sizeof(hf_quant_config.default_model), "%s", "org/first-GGUF,org/second-GGUF");
        ezlocalai::NativeServer llama_hf_quant_server(
            hf_quant_config,
            std::make_unique<ezlocalai::LlamaCppInferenceBackend>(hf_quant_config.default_model)
        );
        const auto llama_hf_quant_response = llama_hf_quant_server.handle_request(
            "POST",
            "/v1/chat/completions",
            "Bearer secret",
            "{\"model\":\"org/second-GGUF\",\"messages\":[{\"role\":\"user\",\"content\":\"quant\"}]}",
            "application/json"
        );
        expect(llama_hf_quant_response.status == 503, "managed llama.cpp hf quant fake exit status");
    }
    {
        std::ifstream args_file(fake_llama_hf_quant_args, std::ios::binary);
        std::string args((std::istreambuf_iterator<char>(args_file)), std::istreambuf_iterator<char>());
        expect(args.find("-hf\n") == std::string::npos && args.find("-m\n./models/second/model-Q5_K_M.gguf") != std::string::npos, "managed text file selection uses captured per-model QUANT_TYPE");
        expect(args.find("--alias\norg/second-GGUF") != std::string::npos, "managed llama.cpp hf keeps public alias without quant suffix");
    }
    unsetenv("LLAMA_CPP_SERVER_BIN");
    unsetenv("LLAMA_CPP_SERVER_PORT");
    unsetenv("QUANT_TYPE");
    unsetenv("LLAMA_CPP_START_TIMEOUT");

    const auto fake_llama_auto_split_bin = test_cwd / "fake-llama-auto-split-server";
    const auto fake_llama_auto_split_args = test_cwd / "llama-auto-split-args.txt";
    {
        write_llama_fixture(fake_llama_auto_split_bin, fake_llama_auto_split_args, false, 127, false);
    }
    std::filesystem::permissions(
        fake_llama_auto_split_bin,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write | std::filesystem::perms::owner_exec,
        std::filesystem::perm_options::replace
    );
    const std::string saved_auto_split_path = getenv("PATH") ? getenv("PATH") : "";
    {
        write_output_fixture("nvidia-smi", "24576, 20000\n24576, 20000\n");
    }
    std::filesystem::permissions(
        "nvidia-smi",
        std::filesystem::perms::owner_exec | std::filesystem::perms::group_exec | std::filesystem::perms::others_exec,
        std::filesystem::perm_options::add
    );
    const std::string auto_split_path = test_cwd.string() + (saved_auto_split_path.empty() ? "" : ":" + saved_auto_split_path);
    setenv("PATH", auto_split_path.c_str(), 1);
    setenv("LLAMA_CPP_SERVER_BIN", fake_llama_auto_split_bin.string().c_str(), 1);
    setenv("LLAMA_CPP_MODEL_PATHS", "first.gguf,second.gguf,third.gguf", 1);
    setenv("LLAMA_CPP_SERVER_PORT", "18182", 1);
    setenv("MAIN_GPU", "0,1,1", 1);
    setenv("LLAMA_CPP_START_TIMEOUT", "1", 1);
    {
        ezlocalai::NativeServer llama_auto_split_server(
            config,
            std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model)
        );
        const auto llama_auto_split_response = llama_auto_split_server.handle_request(
            "POST",
            "/v1/chat/completions",
            "Bearer secret",
            "{\"model\":\"second-model\",\"messages\":[{\"role\":\"user\",\"content\":\"auto split\"}]}",
            "application/json"
        );
        expect(llama_auto_split_response.status == 503, "managed llama.cpp auto split fake exit status");
    }
    {
        std::ifstream args_file(fake_llama_auto_split_args, std::ios::binary);
        std::string args((std::istreambuf_iterator<char>(args_file)), std::istreambuf_iterator<char>());
        expect(args.find("--main-gpu\n1") != std::string::npos, "managed llama.cpp auto split passes per-model main gpu");
        expect(args.find("--tensor-split") == std::string::npos, "source worker placement leaves native tensor ratios unset");
        const bool multiple_cuda = ezlocalai::visible_cuda_device_count() > 1;
        expect((args.find("--split-mode\nnone") != std::string::npos) == multiple_cuda,
            "source worker split mode follows visible CUDA devices, independently of nvidia-smi output");
        if (multiple_cuda) expect(args.find("--fit\noff") != std::string::npos, "source worker placement disables automatic fitting");
    }
    std::filesystem::remove("nvidia-smi");
    if (saved_auto_split_path.empty()) {
        unsetenv("PATH");
    } else {
        setenv("PATH", saved_auto_split_path.c_str(), 1);
    }
    unsetenv("LLAMA_CPP_SERVER_BIN");
    unsetenv("LLAMA_CPP_MODEL_PATHS");
    unsetenv("LLAMA_CPP_SERVER_PORT");
    unsetenv("MAIN_GPU");
    unsetenv("LLAMA_CPP_START_TIMEOUT");

    setenv("LLAMA_CPP_SERVER_URL", "http://127.0.0.1:18143", 1);
    setenv("LLAMA_CPP_CONTEXT", "4096", 1);
    setenv("N_PARALLEL", "2", 1);
    std::thread fake_llama_resources_server(
        fake_expect_worker_once_with_type,
        18143,
        "POST /v1/chat/completions HTTP/1.1",
        "Content-Type: application/json",
        "\"messages\"",
        "application/json",
        "{\"id\":\"chatcmpl-resource\",\"object\":\"chat.completion\",\"model\":\"llama-resource-model\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"resource\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ezlocalai::NativeServer llama_resources_server(
        config,
        std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model)
    );
    const auto llama_resource_chat = llama_resources_server.handle_request(
        "POST",
        "/v1/chat/completions",
        "Bearer secret",
        "{\"model\":\"native-test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}",
        "application/json"
    );
    fake_llama_resources_server.join();
    expect(llama_resource_chat.status == 200, "llama.cpp resource chat status");
    const auto llama_resources = llama_resources_server.handle_request(
        "GET",
        "/v1/resources",
        "Bearer secret",
        "",
        ""
    );
    expect(llama_resources.status == 200, "llama.cpp resources status");
    expect(llama_resources.body.find("\"loaded_models\":{\"llm\":{") != std::string::npos, "llama.cpp resources loaded llm");
    expect(llama_resources.body.find("\"name\":\"native-test-model\"") != std::string::npos, "llama.cpp resources retain configured source model name");
    expect(llama_resources.body.find("\"device\":\"llama.cpp\"") != std::string::npos, "llama.cpp resources device");
    expect(llama_resources.body.find("\"parallel\":{\"n_parallel\":2,\"n_ctx\":4096,\"per_slot_context\":2048") != std::string::npos, "llama.cpp resources context fields");
    unsetenv("LLAMA_CPP_SERVER_URL");
    unsetenv("LLAMA_CPP_CONTEXT");
    unsetenv("N_PARALLEL");

    setenv("LLAMA_CPP_SERVER_URL", "http://127.0.0.1:18184", 1);
    std::thread fake_llama_vision_resources_server(
        fake_expect_worker_once_with_type,
        18184,
        "POST /v1/chat/completions HTTP/1.1",
        "Content-Type: application/json",
        "\"messages\"",
        "application/json",
        "{\"id\":\"chatcmpl-vision-resource\",\"object\":\"chat.completion\",\"model\":\"vision-resource-model\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"vision\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ezl_env_config vision_llama_config = config;
    std::snprintf(vision_llama_config.default_model, sizeof(vision_llama_config.default_model), "%s", "org/vision-model-GGUF");
    ezlocalai::NativeServer llama_vision_resources_server(
        vision_llama_config,
        std::make_unique<ezlocalai::LlamaCppInferenceBackend>(vision_llama_config.default_model)
    );
    const auto llama_vision_chat = llama_vision_resources_server.handle_request(
        "POST",
        "/v1/chat/completions",
        "Bearer secret",
        "{\"model\":\"org/vision-model-GGUF\",\"messages\":[{\"role\":\"user\",\"content\":\"look\"}]}",
        "application/json"
    );
    fake_llama_vision_resources_server.join();
    expect(llama_vision_chat.status == 200, "llama.cpp vision resource chat status");
    const auto llama_vision_resources = llama_vision_resources_server.handle_request(
        "GET",
        "/v1/resources",
        "Bearer secret",
        "",
        ""
    );
    expect(llama_vision_resources.status == 200, "llama.cpp vision resources status");
    expect(llama_vision_resources.body.find("\"loaded_models\":{\"vision_llm\":{") != std::string::npos, "llama.cpp resources loaded vision llm");
    expect(llama_vision_resources.body.find("\"name\":\"org/vision-model-GGUF\"") != std::string::npos, "llama.cpp vision resources retain configured source model name");
    unsetenv("LLAMA_CPP_SERVER_URL");

    setenv("LLAMA_CPP_EMBEDDING_SERVER_URL", "http://127.0.0.1:18144", 1);
    std::thread fake_llama_embedding_resources_server(
        fake_expect_worker_once_with_type,
        18144,
        "POST /v1/embeddings HTTP/1.1",
        "Content-Type: application/json",
        "\"input\":\"embed me\"",
        "application/json",
        "{\"object\":\"list\",\"model\":\"native-test-model\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.1,0.2]}],\"usage\":{\"prompt_tokens\":2,\"total_tokens\":2}}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ezlocalai::NativeServer llama_embedding_resources_server(
        config,
        std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model)
    );
    const auto llama_resource_embedding = llama_embedding_resources_server.handle_request(
        "POST",
        "/v1/embeddings",
        "Bearer secret",
        "{\"model\":\"native-test-model\",\"input\":\"embed me\"}",
        "application/json"
    );
    fake_llama_embedding_resources_server.join();
    expect(llama_resource_embedding.status == 200, "llama.cpp resource embedding status");
    const auto llama_embedding_resources = llama_embedding_resources_server.handle_request(
        "GET",
        "/v1/resources",
        "Bearer secret",
        "",
        ""
    );
    expect(llama_embedding_resources.status == 200, "llama.cpp embedding resources status");
    expect(llama_embedding_resources.body.find("\"loaded_models\":{\"embedding\":{") != std::string::npos, "llama.cpp resources loaded embedding");
    expect(llama_embedding_resources.body.find("\"device\":\"llama.cpp\"") != std::string::npos, "llama.cpp embedding resources device");
    expect(llama_embedding_resources.body.find("\"parallel\":{\"n_parallel\":1,\"n_ctx\":0,\"per_slot_context\":0") != std::string::npos, "llama.cpp embedding resources leaves llm context empty");
    unsetenv("LLAMA_CPP_EMBEDDING_SERVER_URL");

    setenv("LLAMA_CPP_SERVER_URL", "http://127.0.0.1:18146", 1);
    setenv("LLAMA_CPP_EMBEDDING_SERVER_URL", "http://127.0.0.1:18145", 1);
    setenv("LLAMA_CPP_EMBEDDING_MODEL", "BAAI/bge-m3", 1);
    std::thread fake_llama_dedicated_embedding_server(
        fake_expect_worker_once_with_type,
        18145,
        "POST /v1/embeddings HTTP/1.1",
        "Content-Type: application/json",
        "\"input\":\"dedicated embed\"",
        "application/json",
        "{\"object\":\"list\",\"model\":\"bge-m3\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.7,0.8]}],\"usage\":{\"prompt_tokens\":2,\"total_tokens\":2}}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ezlocalai::NativeServer llama_dedicated_embedding_server(
        config,
        std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model)
    );
    const auto llama_dedicated_embedding = llama_dedicated_embedding_server.handle_request(
        "POST",
        "/v1/embeddings",
        "Bearer secret",
        "{\"model\":\"native-test-model\",\"input\":\"dedicated embed\"}",
        "application/json"
    );
    fake_llama_dedicated_embedding_server.join();
    expect(llama_dedicated_embedding.status == 200, "llama.cpp dedicated embedding status");
    expect(llama_dedicated_embedding.body.find("\"model\":\"native-test-model\"") != std::string::npos, "llama.cpp dedicated embedding response model");
    unsetenv("LLAMA_CPP_SERVER_URL");
    unsetenv("LLAMA_CPP_EMBEDDING_SERVER_URL");
    unsetenv("LLAMA_CPP_EMBEDDING_MODEL");

    const auto auto_embedding_model = test_cwd / "models/Qwen3-Embedding-0.6B" / "Qwen3-Embedding-0.6B-Q8_0.gguf";
    std::filesystem::create_directories(auto_embedding_model.parent_path());
    {
        std::ofstream model(auto_embedding_model, std::ios::binary);
        model << "GGUF";
    }
    const auto fake_auto_embedding_bin = test_cwd / "fake-auto-embedding-llama-server";
    const auto fake_auto_embedding_args = test_cwd / "llama-auto-embedding-args.txt";
    {
        write_llama_fixture(fake_auto_embedding_bin, fake_auto_embedding_args, true, 0, false);
    }
    std::filesystem::permissions(
        fake_auto_embedding_bin,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write | std::filesystem::perms::owner_exec,
        std::filesystem::perm_options::replace
    );
    setenv("EMBEDDING_GPU_LAYERS", "0", 1);
    setenv("LLAMA_CPP_SERVER_BIN", fake_auto_embedding_bin.string().c_str(), 1);
    setenv("LLAMA_CPP_EMBEDDING_SERVER_PORT", "18176", 1);
    unsetenv("LLAMA_CPP_EMBEDDING_SERVER_URL");
    unsetenv("LLAMA_CPP_EMBEDDING_MODEL");
    unsetenv("LLAMA_CPP_EMBEDDING_MODEL_PATH");
    unsetenv("LLAMA_CPP_EMBEDDING_ALIAS");
    std::thread fake_auto_embedding_server(fake_llama_managed_embedding_server_once, 18176, fake_auto_embedding_args);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    {
        ezlocalai::NativeServer auto_embedding_server(
            config,
            std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model)
        );
        const auto auto_embedding = auto_embedding_server.handle_request(
            "POST",
            "/v1/embeddings",
            "Bearer secret",
            "{\"model\":\"Qwen3-Embedding-0.6B\",\"input\":\"auto embed\"}",
            "application/json"
        );
        expect(auto_embedding.status == 200, "llama.cpp auto embedding status");
        expect(auto_embedding.body.find("\"model\":\"Qwen3-Embedding-0.6B\"") != std::string::npos, "llama.cpp auto embedding response model");
    }
    fake_auto_embedding_server.join();
    {
        std::ifstream args_file(fake_auto_embedding_args, std::ios::binary);
        std::string args((std::istreambuf_iterator<char>(args_file)), std::istreambuf_iterator<char>());
        expect(args.find("--embedding") != std::string::npos, "llama.cpp auto embedding server flag");
        for (const auto &flag : {"-c\n10000", "-np\n1", "--pooling\nlast", "-b\n512", "-ub\n512", "-ctk\nf16", "-ctv\nf16", "-ngl\n0", "--no-kv-offload"}) {
            expect(args.find(flag) != std::string::npos, ("current embedding runtime default " + std::string(flag)).c_str());
        }
        expect(args.find("--alias\nQwen3-Embedding-0.6B") != std::string::npos, "llama.cpp auto embedding alias");
        expect(launch_uses_model_file(args, auto_embedding_model), "llama.cpp auto embedding selects the fixture file");
    }
    unsetenv("EMBEDDING_GPU_LAYERS");
    unsetenv("LLAMA_CPP_SERVER_BIN");
    unsetenv("LLAMA_CPP_EMBEDDING_SERVER_PORT");

    const auto fake_llama_bin = test_cwd / "fake-llama-server";
    const auto fake_llama_args = test_cwd / "llama-embedding-args.txt";
    {
        write_llama_fixture(fake_llama_bin, fake_llama_args, false, 127, false);
    }
    std::filesystem::permissions(
        fake_llama_bin,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write | std::filesystem::perms::owner_exec,
        std::filesystem::perm_options::replace
    );
    setenv("LLAMA_CPP_SERVER_BIN", fake_llama_bin.string().c_str(), 1);
    // Placement reads the model's file size before launching the process.
    { std::ofstream model("embedding-model.gguf", std::ios::binary); model << "GGUF"; }
    setenv("LLAMA_CPP_EMBEDDING_MODEL_PATH", "embedding-model.gguf", 1);
    setenv("LLAMA_CPP_EMBEDDING_MODEL", "BAAI/bge-m3", 1);
    setenv("LLAMA_CPP_EMBEDDING_ALIAS", "bge-m3", 1);
    setenv("LLAMA_CPP_EMBEDDING_CONTEXT", "8192", 1);
    setenv("LLAMA_CPP_EMBEDDING_POOLING", "cls", 1);
    setenv("LLAMA_CPP_EMBEDDING_UBATCH", "8192", 1);
    setenv("LLAMA_CPP_EMBEDDING_EXTRA_ARGS", "--no-webui", 1);
    setenv("LLAMA_CPP_START_TIMEOUT", "1", 1);
    ezlocalai::NativeServer llama_embedding_flag_server(
        config,
        std::make_unique<ezlocalai::LlamaCppInferenceBackend>(config.default_model)
    );
    const auto llama_embedding_flag_response = llama_embedding_flag_server.handle_request(
        "POST",
        "/v1/embeddings",
        "Bearer secret",
        "{\"model\":\"native-test-model\",\"input\":\"flag embed\"}",
        "application/json"
    );
    expect(llama_embedding_flag_response.status == 500, "llama.cpp fake embedding server failure is a runtime error");
    {
        std::ifstream args_file(fake_llama_args, std::ios::binary);
        std::string args((std::istreambuf_iterator<char>(args_file)), std::istreambuf_iterator<char>());
        expect(args.find("--embedding") != std::string::npos, "llama.cpp embedding server flag");
        expect(args.find("--pooling\ncls") != std::string::npos, "llama.cpp embedding pooling flag");
        expect(args.find("-ub\n8192") != std::string::npos, "llama.cpp embedding ubatch flag");
        expect(args.find("-c\n8192") != std::string::npos, "llama.cpp embedding context flag");
        expect(args.find("--no-webui") != std::string::npos, "llama.cpp embedding extra args");
        expect(args.find("--alias\nbge-m3") != std::string::npos, "llama.cpp embedding alias");
    }
    unsetenv("LLAMA_CPP_SERVER_BIN");
    unsetenv("LLAMA_CPP_EMBEDDING_MODEL_PATH");
    unsetenv("LLAMA_CPP_EMBEDDING_MODEL");
    unsetenv("LLAMA_CPP_EMBEDDING_ALIAS");
    unsetenv("LLAMA_CPP_EMBEDDING_CONTEXT");
    unsetenv("LLAMA_CPP_EMBEDDING_POOLING");
    unsetenv("LLAMA_CPP_EMBEDDING_UBATCH");
    unsetenv("LLAMA_CPP_EMBEDDING_EXTRA_ARGS");
    unsetenv("LLAMA_CPP_START_TIMEOUT");

    std::filesystem::remove("embedding-model.gguf");
    std::filesystem::remove(fake_llama_args);
    setenv("LLAMA_CPP_SERVER_BIN", fake_llama_bin.string().c_str(), 1);
    setenv("EMBEDDING_CONTEXT_LENGTH", "128", 1);
    const auto isolated_embedding_port = std::to_string(fixture_ports());
    setenv("LLAMA_CPP_EMBEDDING_SERVER_PORT", isolated_embedding_port.c_str(), 1);
    setenv("EMBEDDING_BATCH_SIZE", "2048", 1);
    setenv("EMBEDDING_UBATCH_SIZE", "512", 1);
    setenv("EMBEDDING_GPU_LAYERS", "0", 1);
    setenv("EMBEDDING_MAIN_GPU", "1", 1);
    setenv("LLAMA_CPP_START_TIMEOUT", "1", 1);
    ezl_env_config embedding_only_config = config;
    std::snprintf(embedding_only_config.default_model, sizeof(embedding_only_config.default_model), "%s", "none");
    expect(ezlocalai::LlamaCppInferenceBackend::should_use_from_env(embedding_only_config), "embedding-only service uses native inference");
    auto embedding_only_backend = std::make_unique<ezlocalai::LlamaCppInferenceBackend>(embedding_only_config.default_model);
    expect(embedding_only_backend->warmup(), "embedding-only service has no LLM to warm up");
    expect(!std::filesystem::exists(fake_llama_args), "disabled text model does not start an LLM process");
    ezlocalai::NativeServer embedding_only_server(embedding_only_config, std::move(embedding_only_backend));
    const auto embedding_only_response = embedding_only_server.handle_request("POST", "/v1/embeddings", "Bearer secret", "{\"input\":\"embedding only\"}", "application/json");
    expect(embedding_only_response.status == 500, "fixture embedding process exits with runtime error");
    {
        std::ifstream input(fake_llama_args);
        std::string args((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        for (const auto &flag : {"--alias\nQwen3-Embedding-0.6B", "-c\n128", "-b\n128", "-ub\n128", "--main-gpu\n1"}) {
            expect(args.find(flag) != std::string::npos, ("embedding configuration flag " + std::string(flag)).c_str());
        }
        expect(launch_uses_model_file(args, auto_embedding_model), "embedding discovery uses its model-specific local GGUF");
        expect(args.find("-hf\n") == std::string::npos, "embedding acquisition precedes engine launch");
    }
    std::filesystem::remove(auto_embedding_model);
    unsetenv("LLAMA_CPP_EMBEDDING_SERVER_PORT");
    for (const auto *name : {"LLAMA_CPP_SERVER_BIN", "EMBEDDING_CONTEXT_LENGTH", "EMBEDDING_BATCH_SIZE", "EMBEDDING_UBATCH_SIZE", "EMBEDDING_GPU_LAYERS", "EMBEDDING_MAIN_GPU", "LLAMA_CPP_START_TIMEOUT"}) unsetenv(name);

    auto server = std::make_unique<ezlocalai::NativeServer>(
        config,
        std::make_unique<ezlocalai::EchoInferenceBackend>(config.default_model)
    );
    std::string error;
    expect(server->start("127.0.0.1", 18091, &error), error.c_str());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    // These scenarios model distinct application configurations. The source
    // Service singletons capture URL/key once, so changing them requires a
    // fresh application, while threshold changes remain live within a case.
    auto fallback_settings = [] {
        auto value = [](const char *key) { const auto *raw = getenv(key); return std::string(raw ? raw : ""); };
        return std::vector<std::string>{value("FALLBACK_SERVER"), value("FALLBACK_API_KEY"), value("VOICE_SERVER"), value("VOICE_SERVER_API_KEY"), value("IMAGE_SERVER"), value("IMAGE_SERVER_API_KEY"), value("TEXT_SERVER"), value("TEXT_SERVER_API_KEY"), value("EMBEDDING_SERVER"), value("EMBEDDING_SERVER_API_KEY")};
    };
    auto captured_fallback_settings = fallback_settings();
    bool restart_app_for_config = true;
    std::mutex app_config_mutex;
    auto ensure_app_config = [&] {
        {
            std::lock_guard<std::mutex> lock(app_config_mutex);
            const auto settings = fallback_settings();
            if (restart_app_for_config && settings != captured_fallback_settings) {
                server.reset();
                server = std::make_unique<ezlocalai::NativeServer>(config, std::make_unique<ezlocalai::EchoInferenceBackend>(config.default_model));
                expect(server->start("127.0.0.1", 18091, &error), error.c_str());
                captured_fallback_settings = settings;
            }
        }
    };
    auto request = [&](const std::string &raw) { ensure_app_config(); return ::request(raw); };
    auto request_observe_first_chunk = [&](const std::string &raw,const std::string &needle,long long *seen) {
        ensure_app_config(); return ::request_observe_first_chunk(raw,needle,seen);
    };
    const std::string missing_route = "/v1/__not_an_ezlocalai_endpoint__";

    auto health = request("GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(health.find("200 OK") != std::string::npos, "health status");
    expect(health.find("{\"status\":\"healthy\"}") != std::string::npos, "health body");
    expect(health.find("Access-Control-Allow-Origin") == std::string::npos, "no simple cors without origin");
    auto health_origin = request("GET /health HTTP/1.1\r\nHost: localhost\r\nOrigin: http://example.com\r\n\r\n");
    expect(health_origin.find("200 OK") != std::string::npos, "health origin status");
    expect(health_origin.find("Access-Control-Allow-Origin: *") != std::string::npos, "simple cors wildcard origin");
    expect(health_origin.find("Access-Control-Allow-Credentials: true") != std::string::npos, "simple cors credentials header");
    expect(response_header_before(health_origin, "Content-Length", "Content-Type"), "simple cors content length before type");
    expect(response_header_before(health_origin, "Content-Type", "Access-Control-Allow-Origin"), "simple cors type before origin");
    expect(response_header_before(health_origin, "Access-Control-Allow-Origin", "Access-Control-Allow-Credentials"), "simple cors origin before credentials");
    expect(health_origin.find("Access-Control-Allow-Methods") == std::string::npos, "simple cors omits methods");
    expect(health_origin.find("Access-Control-Allow-Headers") == std::string::npos, "simple cors omits headers");
    auto health_cookie_origin = request("GET /health HTTP/1.1\r\nHost: localhost\r\nOrigin: http://example.com\r\nCookie: sid=1\r\n\r\n");
    expect(health_cookie_origin.find("Access-Control-Allow-Origin: http://example.com") != std::string::npos, "credentialed simple cors echoes origin");
    expect(response_header(health_cookie_origin, "Vary") == "Origin, Origin", "credentialed api simple cors duplicates vary origin");
    expect(response_header_before(health_cookie_origin, "Access-Control-Allow-Credentials", "Vary"), "credentialed simple cors credentials before vary");
    auto health_query = request("GET /health?probe=1 HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(health_query.find("200 OK") != std::string::npos, "health query status");
    auto post_health = request("POST /health HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n");
    expect(post_health.find("405 Method Not Allowed") != std::string::npos, "health post method not allowed");
    expect(post_health.find("Allow: GET") != std::string::npos, "health post allow header");
    expect(post_health.find("{\"detail\":\"Method Not Allowed\"}") != std::string::npos, "health post method body");
    auto head_health = request("HEAD /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(head_health.find("405 Method Not Allowed") != std::string::npos, "health head method not allowed");
    expect(head_health.find("Allow: GET") != std::string::npos, "health head allow header");
    expect(head_health.rfind("\r\n\r\n") == head_health.size() - 4, "health head omits body");

    auto cors_preflight = request("OPTIONS /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Origin: http://example.com\r\n"
        "Access-Control-Request-Method: POST\r\n"
        "Access-Control-Request-Headers: Authorization, Content-Type\r\n\r\n");
    expect(cors_preflight.find("200 OK") != std::string::npos, "cors preflight status");
    expect(cors_preflight.find("Content-Type: text/plain; charset=utf-8") != std::string::npos, "cors preflight content type");
    expect(cors_preflight.find("Access-Control-Allow-Origin: http://example.com") != std::string::npos, "cors preflight origin echo");
    expect(cors_preflight.find("Access-Control-Allow-Credentials: true") != std::string::npos, "cors preflight credentials");
    expect(cors_preflight.find("Access-Control-Allow-Methods: DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT") != std::string::npos, "cors preflight methods");
    expect(cors_preflight.find("Access-Control-Max-Age: 600") != std::string::npos, "cors preflight max age");
    expect(cors_preflight.find("Access-Control-Allow-Headers: Authorization, Content-Type") != std::string::npos, "cors preflight headers echo");
    expect(response_header_before(cors_preflight, "Vary", "Access-Control-Allow-Methods"), "cors preflight vary before methods");
    expect(response_header_before(cors_preflight, "Access-Control-Allow-Methods", "Access-Control-Max-Age"), "cors preflight methods before max age");
    expect(response_header_before(cors_preflight, "Access-Control-Max-Age", "Access-Control-Allow-Credentials"), "cors preflight max age before credentials");
    expect(response_header_before(cors_preflight, "Access-Control-Allow-Credentials", "Access-Control-Allow-Origin"), "cors preflight credentials before origin");
    expect(response_header_before(cors_preflight, "Access-Control-Allow-Origin", "Access-Control-Allow-Headers"), "cors preflight origin before headers");
    expect(response_header_before(cors_preflight, "Access-Control-Allow-Headers", "Content-Length"), "cors preflight headers before content length");
    expect(response_header_before(cors_preflight, "Content-Length", "Content-Type"), "cors preflight content length before type");
    expect(cors_preflight.rfind("\r\n\r\nOK") == cors_preflight.size() - 6, "cors preflight body");

    auto options_health = request("OPTIONS /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(options_health.find("405 Method Not Allowed") != std::string::npos, "plain options known get route status");
    expect(options_health.find("Allow: GET") != std::string::npos, "plain options known get route allow");
    expect(response_header_before(options_health, "Allow", "Content-Length"), "plain options known get route allow before length");
    expect(response_header_before(options_health, "Content-Length", "Content-Type"), "plain options known get route length before type");
    expect(options_health.find("{\"detail\":\"Method Not Allowed\"}") != std::string::npos, "plain options known route body");
    auto options_chat = request("OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(options_chat.find("405 Method Not Allowed") != std::string::npos, "plain options known post route status");
    expect(options_chat.find("Allow: POST") != std::string::npos, "plain options known post route allow");
    expect(response_header_before(options_chat, "Allow", "Content-Length"), "plain options known post route allow before length");
    auto options_docs = request("OPTIONS / HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(options_docs.find("405 Method Not Allowed") != std::string::npos, "plain options docs status");
    expect(options_docs.find("Allow: HEAD, GET") != std::string::npos, "plain options docs allow");
    expect(response_header_before(options_docs, "Allow", "Content-Length"), "plain options docs allow before length");
    auto options_unknown = request("OPTIONS " + missing_route + " HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(options_unknown.find("404 Not Found") != std::string::npos, "plain options unknown route status");
    expect(options_unknown.find("{\"detail\":\"Not Found\"}") != std::string::npos, "plain options unknown route body");

    auto docs = request("GET / HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(docs.find("200 OK") != std::string::npos, "docs status");
    expect(docs.find("Content-Type: text/html; charset=utf-8") != std::string::npos, "docs content type");
    expect(docs.find("Content-Length: 1015") != std::string::npos, "docs content length");
    expect(docs.find("<title>ezlocalai Server - Swagger UI</title>") != std::string::npos, "docs title");
    expect(docs.find("SwaggerUIBundle") != std::string::npos, "docs swagger ui");
    expect(docs.find("url: '/openapi.json'") != std::string::npos, "docs openapi url");
    expect(response_body(docs).find("<!-- `SwaggerUIBundle` is now available on the page -->") != std::string::npos, "docs fastapi swagger comment");

    auto redoc = request("GET /redoc HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(redoc.find("200 OK") != std::string::npos, "redoc status");
    expect(redoc.find("Content-Length: 897") != std::string::npos, "redoc content length");
    expect(redoc.find("<title>ezlocalai Server - ReDoc</title>") != std::string::npos, "redoc title");
    expect(redoc.find("spec-url=\"/openapi.json\"") != std::string::npos, "redoc openapi url");
    expect(redoc.find("redoc@2/bundles/redoc.standalone.js") != std::string::npos, "redoc fastapi script version");

    auto oauth_redirect = request("GET /docs/oauth2-redirect HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(oauth_redirect.find("200 OK") != std::string::npos, "oauth redirect status");
    expect(oauth_redirect.find("Content-Length: 3012") != std::string::npos, "oauth redirect content length");
    expect(oauth_redirect.find("Swagger UI: OAuth2 Redirect") != std::string::npos, "oauth redirect title");
    expect(response_body(oauth_redirect).find("var sentState = oauth2.state;") != std::string::npos, "oauth redirect sent state");
    expect(response_body(oauth_redirect).find("oauth2.auth.schema.get(\"flow\") === \"authorization_code\"") != std::string::npos, "oauth redirect auth code flow");
    expect(response_body(oauth_redirect).find("document.addEventListener('DOMContentLoaded'") != std::string::npos, "oauth redirect dom loaded handler");

    auto openapi = request("GET /openapi.json HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(openapi.find("200 OK") != std::string::npos, "openapi status");
    expect(openapi.find("\"title\":\"ezlocalai Server\"") != std::string::npos, "openapi title");
    expect(openapi.find("\"/v1/chat/completions\"") != std::string::npos, "openapi api path");
    expect(openapi.find("\"operationId\":\"speech_to_text_v1_audio_transcriptions_post\"") != std::string::npos, "openapi api operation id");
    expect(openapi.find("\"summary\":\"Speech To Text\"") != std::string::npos, "openapi api generated summary");
    expect(openapi.find("\"components\":{\"schemas\"") != std::string::npos, "openapi api generated schemas");
    expect(openapi.find("\"ChatCompletions\"") != std::string::npos, "openapi api chat schema");
    expect(openapi.find("\"Body_generate_subtitles_v1_audio_subtitles_post\"") != std::string::npos, "openapi api multipart schema");
    expect(openapi.find("\"tags\":[\"Completions\"]") != std::string::npos, "openapi api python route tag");
    expect(openapi.find("\"/v1/router/register\"") == std::string::npos, "openapi api excludes router route path");
    expect(openapi.find("\"/dashboard\"") == std::string::npos, "openapi api excludes router dashboard path");
    expect(openapi.find(missing_route) == std::string::npos, "openapi excludes negative-test route");
    const std::string nope_route = std::string("/v1/") + "nope";
    expect(openapi.find("\"" + nope_route + "\"") == std::string::npos, "openapi excludes nonexistent nope route");

    auto docs_head = request("HEAD / HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(docs_head.find("200 OK") != std::string::npos, "docs head status");
    expect(docs_head.find("Content-Type: text/html; charset=utf-8") != std::string::npos, "docs head content type");
    expect(docs_head.find("Content-Length: 1015") != std::string::npos, "docs head content length");
    expect(docs_head.rfind("\r\n\r\n") == docs_head.size() - 4, "docs head omits body");
    auto openapi_head = request("HEAD /openapi.json HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(openapi_head.find("200 OK") != std::string::npos, "openapi head status");
    expect(openapi_head.find("Content-Type: application/json") != std::string::npos, "openapi head content type");
    expect(openapi_head.find("Content-Length: " + std::to_string(response_body(openapi).size()) + "\r\n") != std::string::npos, "openapi HEAD length matches complete GET schema");
    expect(openapi_head.rfind("\r\n\r\n") == openapi_head.size() - 4, "openapi head omits body");
    auto redoc_head = request("HEAD /redoc HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(redoc_head.find("200 OK") != std::string::npos, "redoc head status");
    expect(redoc_head.find("Content-Length: 897") != std::string::npos, "redoc head content length");
    expect(redoc_head.rfind("\r\n\r\n") == redoc_head.size() - 4, "redoc head omits body");
    auto oauth_head = request("HEAD /docs/oauth2-redirect HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(oauth_head.find("200 OK") != std::string::npos, "oauth head status");
    expect(oauth_head.find("Content-Length: 3012") != std::string::npos, "oauth head content length");
    expect(oauth_head.rfind("\r\n\r\n") == oauth_head.size() - 4, "oauth head omits body");

    auto unauthorized = request("GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(unauthorized.find("401 Unauthorized") != std::string::npos, "auth required");
    expect(unauthorized.find("\"detail\":\"Invalid API Key\"") != std::string::npos, "auth error body");
    auto raw_key_auth = request("GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: secret\r\n\r\n");
    expect(raw_key_auth.find("200 OK") != std::string::npos, "api raw key auth accepted");
    auto wrong_scheme_auth = request("GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Token secret\r\n\r\n");
    expect(wrong_scheme_auth.find("401 Unauthorized") != std::string::npos, "api non-bearer auth scheme rejected");
    expect(wrong_scheme_auth.find("\"detail\":\"Invalid API Key\"") != std::string::npos, "api non-bearer auth body");
    auto unknown_without_auth = request("GET " + missing_route + " HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(unknown_without_auth.find("404 Not Found") != std::string::npos, "unknown route bypasses auth status");
    expect(unknown_without_auth.find("{\"detail\":\"Not Found\"}") != std::string::npos, "unknown route bypasses auth body");
    auto nope_without_auth = request("GET " + nope_route + " HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(nope_without_auth.find("404 Not Found") != std::string::npos, "nonexistent nope route status");
    expect(nope_without_auth.find("{\"detail\":\"Not Found\"}") != std::string::npos, "nonexistent nope route body");
    auto malformed_http = request("not http at all");
    expect(malformed_http.find("400 Bad Request") != std::string::npos, "malformed http status");
    expect(malformed_http.find("Content-Type: text/plain; charset=utf-8") != std::string::npos, "malformed http content type");
    expect(malformed_http.find("Invalid HTTP request received.") != std::string::npos, "malformed http body");
    auto invalid_http_version = request("GET / HTTP/9.9\r\nHost: localhost\r\n\r\n");
    expect(invalid_http_version.find("400 Bad Request") != std::string::npos, "invalid http version status");
    expect(invalid_http_version.find("Invalid HTTP request received.") != std::string::npos, "invalid http version body");
    auto post_models_without_auth = request("POST /v1/models HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n");
    expect(post_models_without_auth.find("405 Method Not Allowed") != std::string::npos, "wrong method bypasses auth status");
    expect(post_models_without_auth.find("Allow: GET") != std::string::npos, "wrong method allow header");
    auto get_chat_without_auth = request("GET /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(get_chat_without_auth.find("405 Method Not Allowed") != std::string::npos, "post route get method bypasses auth status");
    expect(get_chat_without_auth.find("Allow: POST") != std::string::npos, "post route get allow header");
    auto head_models_without_auth = request("HEAD /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(head_models_without_auth.find("405 Method Not Allowed") != std::string::npos, "head models wrong method status");
    expect(head_models_without_auth.find("Allow: GET") != std::string::npos, "head models allow header");
    expect(head_models_without_auth.rfind("\r\n\r\n") == head_models_without_auth.size() - 4, "head models wrong method omits body");

    auto api_models = request("GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(api_models.find("200 OK") != std::string::npos, "api models status");
    expect(api_models.find("\"id\":\"native-test-model\"") != std::string::npos, "api models strips legacy token suffix");
    expect(api_models.find("\"id\":\"second-model\"") != std::string::npos, "api models includes second configured model");
    expect(api_models.find("native-test-model#2") == std::string::npos, "api models hides duplicate internal id");
    expect(api_models.find("\"owned_by\":\"ezlocalai\"") != std::string::npos, "api models owner");

    auto api_v1_root = request("GET /v1 HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(api_v1_root.find("404 Not Found") != std::string::npos, "api /v1 root matches FastAPI missing route");

    const std::string body = "{\"model\":\"native-test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}";
    std::string raw = "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
    auto chat = request(raw);
    expect(chat.find("200 OK") != std::string::npos, "chat status");
    expect(chat.find("chat.completion") != std::string::npos, "chat response object");
    expect(chat.find("ping") != std::string::npos, "chat echoes prompt");
    const std::string chunked_chat_body = "{\"model\":\"native-test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"chunked ping\"}]}";
    std::ostringstream chunked_chat;
    chunked_chat << std::hex << chunked_chat_body.size() << "\r\n" << chunked_chat_body << "\r\n0\r\n\r\n";
    raw = "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Transfer-Encoding: chunked\r\n\r\n" + chunked_chat.str();
    auto chunked_chat_response = request(raw);
    expect(chunked_chat_response.find("200 OK") != std::string::npos, "chunked chat request status");
    expect(chunked_chat_response.find("chunked ping") != std::string::npos, "chunked chat request body decoded");
    const std::string stream_chat_body = "{\"model\":\"native-test-model\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"stream ping\"}]}";
    raw = "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(stream_chat_body.size()) + "\r\n\r\n" + stream_chat_body;
    auto stream_chat = request(raw);
    expect(stream_chat.find("200 OK") != std::string::npos, "stream chat status");
    expect(stream_chat.find("Content-Type: text/event-stream") != std::string::npos, "stream chat content type");
    expect(stream_chat.find("Transfer-Encoding: chunked") != std::string::npos, "stream chat chunked transfer");
    expect(stream_chat.find("Content-Length:") == std::string::npos, "stream chat omits content length");
    expect(stream_chat.find("data: {\"id\":\"chatcmpl-native-") != std::string::npos, "stream chat data chunk");
    expect(stream_chat.find("data: [DONE]") != std::string::npos, "stream chat done chunk");
    const std::string stream_chat_string_bool_body = "{\"model\":\"native-test-model\",\"stream\":\"t\",\"messages\":[{\"role\":\"user\",\"content\":\"stream truthy\"}]}";
    raw = "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(stream_chat_string_bool_body.size()) + "\r\n\r\n" + stream_chat_string_bool_body;
    auto stream_chat_string_bool = request(raw);
    expect(stream_chat_string_bool.find("200 OK") != std::string::npos, "stream chat string bool status");
    expect(stream_chat_string_bool.find("Content-Type: text/event-stream") != std::string::npos, "stream chat string bool content type");
    const std::string stream_completion_body = "{\"model\":\"native-test-model\",\"stream\":true,\"prompt\":\"stream complete\"}";
    raw = "POST /v1/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(stream_completion_body.size()) + "\r\n\r\n" + stream_completion_body;
    auto stream_completion = request(raw);
    expect(stream_completion.find("200 OK") != std::string::npos, "stream completion status");
    expect(stream_completion.find("Content-Type: text/event-stream") != std::string::npos, "stream completion content type");
    expect(stream_completion.find("Transfer-Encoding: chunked") != std::string::npos, "stream completion chunked transfer");
    expect(stream_completion.find("Native ezlocalai echo backend received: stream complete") != std::string::npos, "stream completion body");

    auto chat_missing_body = request("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Type: application/json\r\nContent-Length: 0\r\n\r\n");
    expect(chat_missing_body.find("422 Unprocessable Entity") != std::string::npos, "chat missing body status");
    expect(chat_missing_body.find("\"loc\":[\"body\"]") != std::string::npos, "chat missing body loc");
    const std::string chat_bad_messages_body = "{\"messages\":{}}";
    raw = "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(chat_bad_messages_body.size()) + "\r\n\r\n" + chat_bad_messages_body;
    auto chat_bad_messages = request(raw);
    expect(chat_bad_messages.find("422 Unprocessable Entity") != std::string::npos, "chat bad messages status");
    expect(chat_bad_messages.find("\"loc\":[\"body\",\"messages\"]") != std::string::npos, "chat bad messages loc");
    const std::string chat_bad_stop_body = "{\"stop\":[1]}";
    raw = "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(chat_bad_stop_body.size()) + "\r\n\r\n" + chat_bad_stop_body;
    auto chat_bad_stop = request(raw);
    expect(chat_bad_stop.find("422 Unprocessable Entity") != std::string::npos, "chat bad stop status");
    expect(chat_bad_stop.find("\"loc\":[\"body\",\"stop\",0]") != std::string::npos, "chat bad stop loc");
    const std::string completion_array_body = "[]";
    raw = "POST /v1/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(completion_array_body.size()) + "\r\n\r\n" + completion_array_body;
    auto completion_array = request(raw);
    expect(completion_array.find("422 Unprocessable Entity") != std::string::npos, "completion array status");
    expect(completion_array.find("\"type\":\"model_attributes_type\"") != std::string::npos, "completion array type");
    const std::string completion_bad_fields_body = "{\"prompt\":null,\"stream\":\"x\"}";
    raw = "POST /v1/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(completion_bad_fields_body.size()) + "\r\n\r\n" + completion_bad_fields_body;
    auto completion_bad_fields = request(raw);
    expect(completion_bad_fields.find("422 Unprocessable Entity") != std::string::npos, "completion bad fields status");
    expect(completion_bad_fields.find("\"loc\":[\"body\",\"prompt\"]") != std::string::npos, "completion bad prompt loc");
    expect(completion_bad_fields.find("\"type\":\"bool_parsing\"") != std::string::npos, "completion bad stream type");
    const std::string embedding_missing_input_body = "{}";
    raw = "POST /v1/embeddings HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(embedding_missing_input_body.size()) + "\r\n\r\n" + embedding_missing_input_body;
    auto embedding_missing_input = request(raw);
    expect(embedding_missing_input.find("422 Unprocessable Entity") != std::string::npos, "embedding missing input status");
    expect(embedding_missing_input.find("\"loc\":[\"body\",\"input\"]") != std::string::npos, "embedding missing input loc");
    const std::string embedding_bad_input_body = "{\"input\":123}";
    raw = "POST /v1/embeddings HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(embedding_bad_input_body.size()) + "\r\n\r\n" + embedding_bad_input_body;
    auto embedding_bad_input = request(raw);
    expect(embedding_bad_input.find("422 Unprocessable Entity") != std::string::npos, "embedding bad input status");
    expect(embedding_bad_input.find("\"loc\":[\"body\",\"input\",\"str\"]") != std::string::npos, "embedding bad input str loc");
    expect(embedding_bad_input.find("\"loc\":[\"body\",\"input\",\"list[str]\"]") != std::string::npos, "embedding bad input list loc");
    const std::string embedding_bad_list_body = "{\"input\":[\"a\",1]}";
    raw = "POST /v1/embeddings HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(embedding_bad_list_body.size()) + "\r\n\r\n" + embedding_bad_list_body;
    auto embedding_bad_list = request(raw);
    expect(embedding_bad_list.find("422 Unprocessable Entity") != std::string::npos, "embedding bad list status");
    expect(embedding_bad_list.find("\"loc\":[\"body\",\"input\",\"list[str]\",1]") != std::string::npos, "embedding bad list loc");

    auto queue_status = request("GET /v1/queue/status HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(queue_status.find("\"total_queued\":5") != std::string::npos, "queue saw chat requests");
    expect(queue_status.find("\"queue_size\":0") != std::string::npos, "queue status has python key");

    auto queue_alias = request("GET /v1/queue HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(queue_alias.find("200 OK") != std::string::npos, "queue alias status");
    auto non_python_queue_alias = request("GET /queue/status HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(non_python_queue_alias.find("404 Not Found") != std::string::npos, "non-python queue alias absent");
    expect(non_python_queue_alias.find("{\"detail\":\"Not Found\"}") != std::string::npos, "non-python queue alias body");

    auto missing_request = request("GET /v1/queue/request/not-found HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(missing_request.find("404 Not Found") != std::string::npos, "missing queue request status");
    expect(missing_request.find("\"detail\":\"Request not-found not found\"") != std::string::npos, "missing queue request shape");

    const std::string embedding_list_body = "{\"model\":\"native-test-model\",\"input\":[\"alpha\",\"beta\"]}";
    raw = "POST /v1/embeddings HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(embedding_list_body.size()) + "\r\n\r\n" + embedding_list_body;
    auto embedding_list = request(raw);
    expect(embedding_list.find("200 OK") != std::string::npos, "embedding list status");
    expect(embedding_list.find("\"object\":\"list\"") != std::string::npos, "embedding list object");
    expect(embedding_list.find("\"model\":\"native-test-model\"") != std::string::npos, "embedding local response model");
    expect(embedding_list.find("\"index\":0") != std::string::npos, "embedding list first item");
    expect(embedding_list.find("\"index\":1") != std::string::npos, "embedding list second item");

    auto fallback_status = request("GET /v1/fallback/status HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(fallback_status.find("200 OK") != std::string::npos, "fallback status");
    expect(fallback_status.find("\"configured\":false") != std::string::npos, "fallback not configured shape");

    auto resources = request("GET /v1/resources HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(resources.find("200 OK") != std::string::npos, "resources status");
    expect(resources.find("\"gpu_count\":") != std::string::npos, "resources gpu count field");
    expect(resources.find("\"free_per_gpu_gb\":[") != std::string::npos, "resources free per gpu field");
    expect(resources.find("\"fallback\":{") != std::string::npos, "resources fallback block");
    expect(resources.find("\"memory_threshold_gb\":8.0") != std::string::npos, "resources explicit fallback memory default");
    expect(resources.find("\"queue_wait_timeout\":30.0") != std::string::npos, "resources explicit queue wait default");
    expect(resources.find("\"slots\":{") != std::string::npos, "resources slots block");
    expect(resources.find("\"model_slots\":{\"native-test-model\"") != std::string::npos, "resources model slot");
    expect(resources.find("\"torch_cuda\":[") != std::string::npos, "resources CUDA allocator stats field");
    // ResourceManager now uses process-visible CUDA ordinals and caches total
    // capacity at construction. Controlled multi-GPU values and failures are
    // exercised through the real driver ABI in test_resource_manager.

    setenv("TEXT_SERVER", "http://localhost:18114", 1);
    setenv("TEXT_SERVER_API_KEY", "text-secret", 1);
    std::thread fake_text_chat(
        fake_image_server_forward_once,
        18114,
        "POST /v1/chat/completions HTTP/1.1",
        "\"temperature\":0.2",
        "",
        "{\"id\":\"chatcmpl-text\",\"object\":\"chat.completion\",\"model\":\"remote-text\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"remote chat\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}",
        "Authorization: Bearer text-secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string text_chat_body = "{\"model\":\"native-test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"text remote\"}],\"temperature\":0.2}";
    raw = "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(text_chat_body.size()) + "\r\n\r\n" + text_chat_body;
    auto text_chat = request(raw);
    fake_text_chat.join();
    expect(text_chat.find("200 OK") != std::string::npos, "text server chat status");
    expect(text_chat.find("remote chat") != std::string::npos, "text server chat body");
    std::thread fake_text_cached_chat(
        fake_expect_worker_once,
        18114,
        "POST /v1/chat/completions HTTP/1.1",
        "Content-Type: application/json",
        "\"content\":\"text cache\"",
        "{\"id\":\"chatcmpl-text-cache\",\"object\":\"chat.completion\",\"model\":\"remote-text\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"cached remote chat\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string text_cached_chat_body = "{\"model\":\"native-test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"text cache\"}]}";
    raw = "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(text_cached_chat_body.size()) + "\r\n\r\n" + text_cached_chat_body;
    auto text_cached_chat = request(raw);
    fake_text_cached_chat.join();
    expect(text_cached_chat.find("200 OK") != std::string::npos, "text server cached chat status");
    expect(text_cached_chat.find("cached remote chat") != std::string::npos, "text server availability cache skips second probe");
    unsetenv("TEXT_SERVER");
    unsetenv("TEXT_SERVER_API_KEY");

    const char *text_saved_ssl_cert_file_raw = getenv("SSL_CERT_FILE");
    const bool text_had_ssl_cert_file = text_saved_ssl_cert_file_raw != nullptr;
    const std::string text_saved_ssl_cert_file = text_saved_ssl_cert_file_raw ? text_saved_ssl_cert_file_raw : "";
    const auto text_tls_cert = ensure_test_tls_cert();
    setenv("SSL_CERT_FILE", text_tls_cert.cert.c_str(), 1);
    setenv("TEXT_SERVER", "https://127.0.0.2:18161", 1);
    setenv("TEXT_SERVER_API_KEY", "text-secret", 1);
    // Deliberately exceed the former fixed 50 ms sleep: requests must wait
    // for the listener, including when TLS setup is slower under emulation.
    TestListenerReady https_text_ready(18161, std::chrono::milliseconds(250));
    std::thread fake_https_text_chat(
        fake_https_json_worker_many,
        18161,
        "Authorization: Bearer text-secret",
        std::vector<std::pair<std::string, std::string>>{
            {"GET /v1/models HTTP/1.1", "{\"object\":\"list\",\"data\":[{\"id\":\"https-text\"}]}"},
            {"POST /v1/chat/completions HTTP/1.1", "{\"id\":\"chatcmpl-https-text\",\"object\":\"chat.completion\",\"model\":\"https-text\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"https remote chat\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}"}
        },
        &https_text_ready
    );
    https_text_ready.wait();
    const std::string https_text_chat_body = "{\"model\":\"native-test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"https text remote\"}]}";
    raw = "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(https_text_chat_body.size()) + "\r\n\r\n" + https_text_chat_body;
    auto https_text_chat = request(raw);
    fake_https_text_chat.join();
    expect(https_text_chat.find("200 OK") != std::string::npos, "https text server chat status");
    expect(https_text_chat.find("https remote chat") != std::string::npos, "https text server chat body");
    unsetenv("TEXT_SERVER");
    unsetenv("TEXT_SERVER_API_KEY");
    if (text_had_ssl_cert_file) {
        setenv("SSL_CERT_FILE", text_saved_ssl_cert_file.c_str(), 1);
    } else {
        unsetenv("SSL_CERT_FILE");
    }

    setenv("TEXT_SERVER", "http://127.0.0.1:18115", 1);
    std::thread fake_text_completion(
        fake_image_server_forward_once,
        18115,
        "POST /v1/completions HTTP/1.1",
        "\"temperature\":0.3",
        "",
        "{\"id\":\"cmpl-text\",\"object\":\"text_completion\",\"model\":\"remote-text\",\"choices\":[{\"text\":\"remote completion\",\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}",
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string text_completion_body = "{\"model\":\"native-test-model\",\"prompt\":\"complete remote\",\"temperature\":0.3}";
    raw = "POST /v1/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(text_completion_body.size()) + "\r\n\r\n" + text_completion_body;
    auto text_completion = request(raw);
    fake_text_completion.join();
    expect(text_completion.find("200 OK") != std::string::npos, "text server completion status");
    expect(text_completion.find("remote completion") != std::string::npos, "text server completion body");
    unsetenv("TEXT_SERVER");

    setenv("EMBEDDING_SERVER", "http://127.0.0.1:18116", 1);
    setenv("EMBEDDING_SERVER_API_KEY", "embedding-secret", 1);
    std::thread fake_dedicated_embeddings(
        fake_image_server_forward_once,
        18116,
        "POST /v1/embeddings HTTP/1.1",
        "\"user\":\"keep-me\",\"dimensions\":32,\"encoding_format\":\"base64\"",
        "\"extra\"",
        "{\"object\":\"list\",\"model\":\"remote-embed\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.1]}],\"usage\":{\"prompt_tokens\":1,\"total_tokens\":1}}",
        "Authorization: Bearer embedding-secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string dedicated_embedding_body = "{\"input\":[\"a\",\"b\"],\"model\":\"native-test-model\",\"user\":\"keep-me\",\"dimensions\":\"32\",\"encoding_format\":\"base64\",\"extra\":\"drop-me\"}";
    raw = "POST /v1/embeddings HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(dedicated_embedding_body.size()) + "\r\n\r\n" + dedicated_embedding_body;
    auto dedicated_embeddings = request(raw);
    fake_dedicated_embeddings.join();
    expect(dedicated_embeddings.find("200 OK") != std::string::npos, "dedicated embedding server status");
    expect(dedicated_embeddings.find("\"model\":\"remote-embed\"") != std::string::npos, "dedicated embedding server body");
    std::thread fake_cached_embeddings(
        fake_expect_worker_once_with_absence,
        18116,
        "POST /v1/embeddings HTTP/1.1",
        "Authorization: Bearer embedding-secret",
        "\"model\":\"Qwen/Qwen3-Embedding-0.6B-GGUF\"",
        "\"input\":\"cached\"",
        "Content-Type: application/json",
        "",
        "",
        "\"user\"",
        "\"dimensions\"",
        "{\"object\":\"list\",\"data\":[],\"model\":\"cached-embedding\"}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto cached_embeddings = post_json("/v1/embeddings", "{\"input\":\"cached\"}");
    fake_cached_embeddings.join();
    expect(cached_embeddings.find("cached-embedding") != std::string::npos, "embedding availability cache skips second probe and uses current model default");
    unsetenv("EMBEDDING_SERVER");
    unsetenv("EMBEDDING_SERVER_API_KEY");

    setenv("EMBEDDING_SERVER", "http://127.0.0.1:18136", 1);
    std::thread fake_dedicated_embeddings_null_model(
        fake_image_server_forward_once,
        18136,
        "POST /v1/embeddings HTTP/1.1",
        "\"input\":\"null model\"",
        "\"model\"",
        "{\"object\":\"list\",\"model\":\"remote-embed-null\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.4]}],\"usage\":{\"prompt_tokens\":1,\"total_tokens\":1}}",
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string dedicated_embedding_null_model_body = "{\"input\":\"null model\",\"model\":null}";
    raw = "POST /v1/embeddings HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(dedicated_embedding_null_model_body.size()) + "\r\n\r\n" + dedicated_embedding_null_model_body;
    auto dedicated_embeddings_null_model = request(raw);
    fake_dedicated_embeddings_null_model.join();
    expect(dedicated_embeddings_null_model.find("200 OK") != std::string::npos, "dedicated embedding server null model status");
    expect(dedicated_embeddings_null_model.find("\"model\":\"remote-embed-null\"") != std::string::npos, "dedicated embedding server null model body");
    unsetenv("EMBEDDING_SERVER");

    setenv("EMBEDDING_SERVER", "http://127.0.0.1:18137", 1);
    std::thread fake_dedicated_embeddings_empty_model(
        fake_image_server_forward_once,
        18137,
        "POST /v1/embeddings HTTP/1.1",
        "\"model\":\"\"",
        "\"model\":\"native-test-model\"",
        "{\"object\":\"list\",\"model\":\"remote-embed-empty\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.5]}],\"usage\":{\"prompt_tokens\":1,\"total_tokens\":1}}",
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string dedicated_embedding_empty_model_body = "{\"input\":\"empty model\",\"model\":\"\"}";
    raw = "POST /v1/embeddings HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(dedicated_embedding_empty_model_body.size()) + "\r\n\r\n" + dedicated_embedding_empty_model_body;
    auto dedicated_embeddings_empty_model = request(raw);
    fake_dedicated_embeddings_empty_model.join();
    expect(dedicated_embeddings_empty_model.find("200 OK") != std::string::npos, "dedicated embedding server empty model status");
    expect(dedicated_embeddings_empty_model.find("\"model\":\"remote-embed-empty\"") != std::string::npos, "dedicated embedding server empty model body");
    unsetenv("EMBEDDING_SERVER");

    setenv("EMBEDDING_ENABLED", "false", 1);
    setenv("ROUTER_URL", "http://127.0.0.1:18138", 1);
    setenv("ROUTER_CLIENT_API_KEY", "dispatch-secret", 1);
    std::thread fake_embedding_dispatch(
        fake_json_service_forward_once,
        18138,
        "POST /v1/embeddings HTTP/1.1",
        "\"input\":\"dispatch embedding\",\"model\":\"Qwen/Qwen3-Embedding-0.6B-GGUF\"",
        "\"dimensions\"",
        "{\"object\":\"list\",\"data\":[],\"model\":\"router-embedding\"}",
        "Authorization: Bearer dispatch-secret",
        "/v1/router/health",
        "{\"status\":\"ok\",\"alive_workers\":1}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto dispatch_embedding = post_json("/v1/embeddings", "{\"input\":\"dispatch embedding\"}");
    fake_embedding_dispatch.join();
    expect(dispatch_embedding.find("router-embedding") != std::string::npos, "disabled embeddings use available router");
    setenv("ROUTER_URL", "http://127.0.0.1:18139", 1);
    std::thread fake_empty_embedding_router(
        fake_expect_worker_once,
        18139,
        "GET /v1/router/health HTTP/1.1",
        "Authorization: Bearer dispatch-secret",
        "",
        "{\"status\":\"ok\",\"alive_workers\":0}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto no_embedding_router = post_json("/v1/embeddings", "{\"input\":\"dispatch embedding\"}");
    fake_empty_embedding_router.join();
    expect(no_embedding_router.find("404 Not Found") != std::string::npos, "disabled embeddings reject router without live workers");
    unsetenv("EMBEDDING_ENABLED");
    unsetenv("ROUTER_URL");
    unsetenv("ROUTER_CLIENT_API_KEY");

    setenv("FALLBACK_SERVER", "http://127.0.0.1:18117", 1);
    setenv("FALLBACK_API_KEY", "fallback-secret", 1);
    setenv("FALLBACK_MEMORY_THRESHOLD", "999999", 1);
    std::thread fake_fallback_embeddings(
        fake_fallback_forward_once,
        18117,
        "POST /v1/embeddings HTTP/1.1",
        "Content-Type: application/json",
        "\"input\":\"fallback embed\",\"model\":\"native-test-model\",\"dimensions\":null,\"encoding_format\":null",
        "\"user\"",
        "application/json",
        "{\"object\":\"list\",\"model\":\"fallback-embed\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.2]}],\"usage\":{\"prompt_tokens\":1,\"total_tokens\":1}}",
        "Authorization: Bearer fallback-secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string fallback_embedding_body = "{\"input\":\"fallback embed\",\"model\":\"native-test-model\",\"user\":\"drop-me\"}";
    raw = "POST /v1/embeddings HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(fallback_embedding_body.size()) + "\r\n\r\n" + fallback_embedding_body;
    auto fallback_embeddings = request(raw);
    fake_fallback_embeddings.join();
    expect(fallback_embeddings.find("200 OK") != std::string::npos, "fallback embeddings status");
    expect(fallback_embeddings.find("\"model\":\"fallback-embed\"") != std::string::npos, "fallback embeddings body");
    std::thread fake_cached_fallback_embeddings(
        fake_expect_worker_once,
        18117,
        "POST /v1/embeddings HTTP/1.1",
        "Content-Type: application/json",
        "\"input\":\"fallback cached embed\"",
        "{\"object\":\"list\",\"model\":\"fallback-embed-cache\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.3]}],\"usage\":{\"prompt_tokens\":1,\"total_tokens\":1}}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string fallback_cached_embedding_body = "{\"input\":\"fallback cached embed\",\"model\":\"native-test-model\"}";
    raw = "POST /v1/embeddings HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(fallback_cached_embedding_body.size()) + "\r\n\r\n" + fallback_cached_embedding_body;
    auto fallback_cached_embeddings = request(raw);
    fake_cached_fallback_embeddings.join();
    expect(fallback_cached_embeddings.find("200 OK") != std::string::npos, "fallback cached embeddings status");
    expect(fallback_cached_embeddings.find("\"model\":\"fallback-embed-cache\"") != std::string::npos, "fallback availability cache skips second probe");
    unsetenv("FALLBACK_SERVER");
    unsetenv("FALLBACK_API_KEY");
    unsetenv("FALLBACK_MEMORY_THRESHOLD");

    setenv("DEFAULT_MODEL", "", 1);
    setenv("VISION_MODEL", "", 1);
    auto chat_disabled = request("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Length: 2\r\n\r\n{}");
    expect(chat_disabled.find("404 Not Found") != std::string::npos, "chat disabled status");
    expect(chat_disabled.find("\"detail\":\"Chat completions are disabled.\"") != std::string::npos, "chat disabled body");
    auto completion_disabled = request("POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Length: 2\r\n\r\n{}");
    expect(completion_disabled.find("404 Not Found") != std::string::npos, "completion disabled status");
    expect(completion_disabled.find("\"detail\":\"Completions are disabled.\"") != std::string::npos, "completion disabled body");
    unsetenv("DEFAULT_MODEL");
    unsetenv("VISION_MODEL");

    setenv("FALLBACK_SERVER", "http://127.0.0.1:18094", 1);
    setenv("FALLBACK_MEMORY_THRESHOLD", "999999", 1);
    std::thread fallback_server(fake_json_worker_many, 18094, std::vector<std::string>{
        "{\"free_vram_gb\":4.5}",
        "{\"object\":\"list\",\"data\":[{\"id\":\"fallback-model\"},{\"name\":\"named-fallback\"}]}"
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto configured_fallback_status = request("GET /v1/fallback/status HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    fallback_server.join();
    expect(configured_fallback_status.find("200 OK") != std::string::npos, "configured fallback status");
    expect(configured_fallback_status.find("\"configured\":true") != std::string::npos, "configured fallback configured");
    expect(configured_fallback_status.find("\"available\":true") != std::string::npos, "configured fallback available");
    expect(configured_fallback_status.find("Fallback server available (4.5GB VRAM free)") != std::string::npos, "configured fallback available reason format");
    expect(configured_fallback_status.find("\"should_use\":true") != std::string::npos, "configured fallback should use threshold");
    expect(configured_fallback_status.find("\"remote_models\":[\"fallback-model\",\"named-fallback\"]") != std::string::npos, "configured fallback models");
    std::thread cached_fallback_models(fake_json_worker_once, 18094, "{\"object\":\"list\",\"data\":[{\"id\":\"cached-fallback\"}]}");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto cached_fallback_status = request("GET /v1/fallback/status HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    cached_fallback_models.join();
    expect(cached_fallback_status.find("200 OK") != std::string::npos, "cached fallback status");
    expect(cached_fallback_status.find("\"availability_reason\":\"cached\"") != std::string::npos, "cached fallback reason");
    expect(cached_fallback_status.find("\"remote_models\":[\"cached-fallback\"]") != std::string::npos, "cached fallback still fetches models");
    setenv("FALLBACK_SERVER", "http://127.0.0.1:18148", 1);
    std::thread chunked_fallback_server(fake_chunked_worker_many, 18148, "application/json", std::vector<std::vector<std::string>>{
        {"{\"free_v", "ram_gb\":4.5}"},
        {"{\"object\":\"list\",\"data\":[", "{\"id\":\"chunked-fallback\"}", "]}"}
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto chunked_fallback_status = request("GET /v1/fallback/status HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    chunked_fallback_server.join();
    expect(chunked_fallback_status.find("200 OK") != std::string::npos, "chunked fallback status");
    expect(chunked_fallback_status.find("\"available\":true") != std::string::npos, "chunked fallback available");
    expect(chunked_fallback_status.find("\"remote_models\":[\"chunked-fallback\"]") != std::string::npos, "chunked fallback models decode");
    expect(chunked_fallback_status.find("\r\n0\r\n\r\n") == std::string::npos, "chunked fallback response omits raw chunk terminator");
    setenv("FALLBACK_SERVER", "http://127.0.0.1:18150", 1);
    std::thread fallback_bodyless_get_server(fake_get_json_worker_many_no_content_length, 18150, std::vector<std::string>{
        "{\"free_vram_gb\":4.5}",
        "{\"object\":\"list\",\"data\":[{\"id\":\"bodyless-get-fallback\"}]}"
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto bodyless_get_fallback_status = request("GET /v1/fallback/status HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    fallback_bodyless_get_server.join();
    expect(bodyless_get_fallback_status.find("200 OK") != std::string::npos, "fallback bodyless get status");
    expect(bodyless_get_fallback_status.find("\"remote_models\":[\"bodyless-get-fallback\"]") != std::string::npos, "fallback bodyless get models");
    const char *fallback_saved_ssl_cert_file_raw = getenv("SSL_CERT_FILE");
    const bool fallback_had_ssl_cert_file = fallback_saved_ssl_cert_file_raw != nullptr;
    const std::string fallback_saved_ssl_cert_file = fallback_saved_ssl_cert_file_raw ? fallback_saved_ssl_cert_file_raw : "";
    const auto fallback_tls_cert = ensure_test_tls_cert();
    setenv("SSL_CERT_FILE", fallback_tls_cert.cert.c_str(), 1);
    setenv("FALLBACK_SERVER", "https://127.0.0.2:18162", 1);
    setenv("FALLBACK_API_KEY", "fallback-secret", 1);
    TestListenerReady https_fallback_ready(18162);
    std::thread https_fallback_status_server(
        fake_https_json_worker_many,
        18162,
        "Authorization: Bearer fallback-secret",
        std::vector<std::pair<std::string, std::string>>{
            {"GET /v1/resources HTTP/1.1", "{\"free_vram_gb\":4.5}"},
            {"GET /v1/models HTTP/1.1", "{\"object\":\"list\",\"data\":[{\"id\":\"https-fallback\"}]}"}
        },
        &https_fallback_ready
    );
    https_fallback_ready.wait();
    auto https_fallback_status = request("GET /v1/fallback/status HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    https_fallback_status_server.join();
    expect(https_fallback_status.find("200 OK") != std::string::npos, "https fallback status");
    expect(https_fallback_status.find("\"available\":true") != std::string::npos, "https fallback available");
    expect(https_fallback_status.find("\"remote_models\":[\"https-fallback\"]") != std::string::npos, "https fallback models");
    unsetenv("FALLBACK_API_KEY");
    if (fallback_had_ssl_cert_file) {
        setenv("SSL_CERT_FILE", fallback_saved_ssl_cert_file.c_str(), 1);
    } else {
        unsetenv("SSL_CERT_FILE");
    }
    setenv("FALLBACK_SERVER", "http://127.0.0.1:18145", 1);
    std::thread typed_models_fallback_server(fake_json_worker_many, 18145, std::vector<std::string>{
        "{\"free_vram_gb\":4.5}",
        "{\"object\":\"list\",\"data\":[{\"id\":null,\"name\":\"ignored-name\"},{\"id\":\"\"},{\"name\":null},{}]}"
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto typed_models_fallback_status = request("GET /v1/fallback/status HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    typed_models_fallback_server.join();
    expect(typed_models_fallback_status.find("200 OK") != std::string::npos, "typed models fallback status");
    expect(typed_models_fallback_status.find("\"remote_models\":[null,\"\",null,\"unknown\"]") != std::string::npos, "typed models fallback preserves python get values");
    setenv("FALLBACK_SERVER", "http://127.0.0.1:18144", 1);
    std::thread bad_models_fallback_server(fake_json_worker_many, 18144, std::vector<std::string>{
        "{\"free_vram_gb\":4.5}",
        "{\"object\":\"list\",\"data\":[{\"id\":\"good-before-bad\"},\"bad-model-entry\"]}"
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto bad_models_fallback_status = request("GET /v1/fallback/status HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    bad_models_fallback_server.join();
    expect(bad_models_fallback_status.find("200 OK") != std::string::npos, "bad models fallback status");
    expect(bad_models_fallback_status.find("\"available\":true") != std::string::npos, "bad models fallback still available");
    expect(bad_models_fallback_status.find("\"remote_models\":[]") != std::string::npos, "bad models fallback mirrors python except");
    setenv("FALLBACK_SERVER", "http://127.0.0.1:18125", 1);
    setenv("FALLBACK_VRAM_THRESHOLD", "8.0", 1);
    std::thread low_vram_fallback_server(fake_json_worker_many, 18125, std::vector<std::string>{
        "{\"free_vram_gb\":4.5}"
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto low_vram_fallback_status = request("GET /v1/fallback/status HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    low_vram_fallback_server.join();
    expect(low_vram_fallback_status.find("200 OK") != std::string::npos, "low vram fallback status");
    expect(low_vram_fallback_status.find("\"available\":false") != std::string::npos, "low vram fallback unavailable");
    expect(low_vram_fallback_status.find("Fallback server low on VRAM (4.5GB < 8.0GB)") != std::string::npos, "low vram fallback reason format");
    unsetenv("FALLBACK_VRAM_THRESHOLD");
    setenv("FALLBACK_SERVER", "http://127.0.0.1:18156", 1);
    setenv("FALLBACK_VRAM_THRESHOLD", "bad-threshold", 1);
    std::thread bad_threshold_fallback_server(fake_json_worker_many, 18156, std::vector<std::string>{
        "{\"free_vram_gb\":4.5}"
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto bad_threshold_fallback_status = request("GET /v1/fallback/status HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    bad_threshold_fallback_server.join();
    expect(bad_threshold_fallback_status.find("200 OK") != std::string::npos, "bad threshold fallback status");
    expect(bad_threshold_fallback_status.find("\"available\":false") != std::string::npos, "bad threshold fallback unavailable");
    expect(bad_threshold_fallback_status.find("\"availability_reason\":\"Fallback server unreachable\"") != std::string::npos, "bad threshold fallback reason");
    expect(bad_threshold_fallback_status.find("\"remote_models\":[]") != std::string::npos, "bad threshold skips fallback models");
    unsetenv("FALLBACK_VRAM_THRESHOLD");
    setenv("FALLBACK_SERVER", "http://127.0.0.1:18157", 1);
    std::thread string_vram_fallback_server(fake_json_worker_many, 18157, std::vector<std::string>{
        "{\"free_vram_gb\":\"4.5\"}"
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto string_vram_fallback_status = request("GET /v1/fallback/status HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    string_vram_fallback_server.join();
    expect(string_vram_fallback_status.find("200 OK") != std::string::npos, "string vram fallback status");
    expect(string_vram_fallback_status.find("\"available\":false") != std::string::npos, "string vram fallback unavailable");
    expect(string_vram_fallback_status.find("\"availability_reason\":\"Fallback server unreachable\"") != std::string::npos, "string vram fallback reason");
    expect(string_vram_fallback_status.find("\"remote_models\":[]") != std::string::npos, "string vram skips fallback models");
    setenv("FALLBACK_SERVER", "http://127.0.0.1:18120", 1);
    auto unreachable_fallback_status = request("GET /v1/fallback/status HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(unreachable_fallback_status.find("200 OK") != std::string::npos, "unreachable fallback status");
    expect(unreachable_fallback_status.find("\"available\":false") != std::string::npos, "unreachable fallback unavailable");
    expect(unreachable_fallback_status.find("\"availability_reason\":\"Fallback server unreachable\"") != std::string::npos, "unreachable fallback reason");
    setenv("FALLBACK_SERVER", "http://127.0.0.1:18143", 1);
    std::thread slow_fallback_server(fake_slow_resources_once, 18143);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto slow_fallback_status = request("GET /v1/fallback/status HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    slow_fallback_server.join();
    expect(slow_fallback_status.find("200 OK") != std::string::npos, "slow fallback status");
    expect(slow_fallback_status.find("\"available\":false") != std::string::npos, "slow fallback unavailable");
    expect(slow_fallback_status.find("\"availability_reason\":\"Fallback server unreachable\"") != std::string::npos, "slow fallback timeout reason");
    unsetenv("FALLBACK_SERVER");
    unsetenv("FALLBACK_MEMORY_THRESHOLD");

    auto known_missing = request("POST /v1/audio/transcriptions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Length: 0\r\n\r\n");
    expect(known_missing.find("422 Unprocessable Entity") != std::string::npos, "transcription missing file validates first");
    expect(known_missing.find("\"loc\":[\"body\",\"file\"]") != std::string::npos, "transcription missing file loc");
    const std::string plain_file_field_boundary = "plain-file-field-boundary";
    const std::string plain_file_field_body =
        "--" + plain_file_field_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"\r\n\r\n"
        "not-upload\r\n"
        "--" + plain_file_field_boundary + "--\r\n";
    raw = "POST /v1/audio/transcriptions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + plain_file_field_boundary + "\r\n"
        "Content-Length: " + std::to_string(plain_file_field_body.size()) + "\r\n\r\n" + plain_file_field_body;
    auto plain_file_field = request(raw);
    expect(plain_file_field.find("422 Unprocessable Entity") != std::string::npos, "plain file form field rejects as upload");
    expect(plain_file_field.find("Expected UploadFile") != std::string::npos, "plain file form field uploadfile message");
    expect(plain_file_field.find("\"input\":\"not-upload\"") != std::string::npos, "plain file form field input");
    const std::string plain_file_bad_temp_body =
        "--" + plain_file_field_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"\r\n\r\n"
        "not-upload\r\n"
        "--" + plain_file_field_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"temperature\"\r\n\r\n"
        "bad\r\n"
        "--" + plain_file_field_boundary + "--\r\n";
    raw = "POST /v1/audio/transcriptions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + plain_file_field_boundary + "\r\n"
        "Content-Length: " + std::to_string(plain_file_bad_temp_body.size()) + "\r\n\r\n" + plain_file_bad_temp_body;
    auto plain_file_bad_temp = request(raw);
    expect(plain_file_bad_temp.find("Expected UploadFile") != std::string::npos, "plain file bad temp includes uploadfile error");
    expect(plain_file_bad_temp.find("\"loc\":[\"body\",\"temperature\"]") != std::string::npos, "plain file bad temp includes temperature error");
    auto translation_missing = request("POST /v1/audio/translations HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Length: 0\r\n\r\n");
    expect(translation_missing.find("422 Unprocessable Entity") != std::string::npos, "translation missing file validates first");
    auto subtitles_missing = request("POST /v1/audio/subtitles HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Length: 0\r\n\r\n");
    expect(subtitles_missing.find("422 Unprocessable Entity") != std::string::npos, "subtitles missing file validates first");
    const std::string subtitle_bad_boundary = "subtitle-bad-boundary";
    const std::string subtitles_bad_beam_body =
        "--" + subtitle_bad_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "data\r\n"
        "--" + subtitle_bad_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"beam_size\"\r\n\r\n"
        "bad\r\n"
        "--" + subtitle_bad_boundary + "--\r\n";
    raw = "POST /v1/audio/subtitles HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + subtitle_bad_boundary + "\r\n"
        "Content-Length: " + std::to_string(subtitles_bad_beam_body.size()) + "\r\n\r\n" + subtitles_bad_beam_body;
    auto subtitles_bad_beam = request(raw);
    expect(subtitles_bad_beam.find("422 Unprocessable Entity") != std::string::npos, "subtitles bad beam status");
    expect(subtitles_bad_beam.find("\"loc\":[\"body\",\"beam_size\"]") != std::string::npos, "subtitles bad beam loc");

    setenv("STT_ENABLED", "false", 1);
    auto stt_disabled = request("POST /v1/audio/transcriptions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Length: 0\r\n\r\n");
    expect(stt_disabled.find("422 Unprocessable Entity") != std::string::npos, "stt disabled missing file validates first");
    const std::string stt_disabled_boundary = "stt-disabled-boundary";
    const std::string stt_disabled_body =
        "--" + stt_disabled_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "data\r\n"
        "--" + stt_disabled_boundary + "--\r\n";
    const std::string stt_disabled_bad_temp_body =
        "--" + stt_disabled_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "data\r\n"
        "--" + stt_disabled_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"temperature\"\r\n\r\n"
        "bad\r\n"
        "--" + stt_disabled_boundary + "--\r\n";
    raw = "POST /v1/audio/transcriptions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + stt_disabled_boundary + "\r\n"
        "Content-Length: " + std::to_string(stt_disabled_bad_temp_body.size()) + "\r\n\r\n" + stt_disabled_bad_temp_body;
    auto stt_disabled_bad_temp = request(raw);
    expect(stt_disabled_bad_temp.find("422 Unprocessable Entity") != std::string::npos, "stt disabled bad temp validates first");
    expect(stt_disabled_bad_temp.find("\"loc\":[\"body\",\"temperature\"]") != std::string::npos, "stt disabled bad temp loc");
    std::ostringstream chunked_bad_temp;
    chunked_bad_temp << std::hex << 17 << "\r\n" << stt_disabled_bad_temp_body.substr(0, 17) << "\r\n"
        << std::hex << (stt_disabled_bad_temp_body.size() - 17) << "\r\n" << stt_disabled_bad_temp_body.substr(17) << "\r\n"
        << "0\r\n\r\n";
    raw = "POST /v1/audio/transcriptions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + stt_disabled_boundary + "\r\n"
        "Transfer-Encoding: chunked\r\n\r\n" + chunked_bad_temp.str();
    auto chunked_stt_bad_temp = request(raw);
    expect(chunked_stt_bad_temp.find("422 Unprocessable Entity") != std::string::npos, "chunked stt multipart bad temp validates first");
    expect(chunked_stt_bad_temp.find("\"loc\":[\"body\",\"temperature\"]") != std::string::npos, "chunked stt multipart body decoded");
    const std::string stt_disabled_underscore_temp_body =
        "--" + stt_disabled_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "data\r\n"
        "--" + stt_disabled_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"temperature\"\r\n\r\n"
        "1_000.5\r\n"
        "--" + stt_disabled_boundary + "--\r\n";
    raw = "POST /v1/audio/transcriptions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + stt_disabled_boundary + "\r\n"
        "Content-Length: " + std::to_string(stt_disabled_underscore_temp_body.size()) + "\r\n\r\n" + stt_disabled_underscore_temp_body;
    auto stt_disabled_underscore_temp = request(raw);
    expect(stt_disabled_underscore_temp.find("404 Not Found") != std::string::npos, "stt disabled underscore temp validates");
    raw = "POST /v1/audio/transcriptions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + stt_disabled_boundary + "\r\n"
        "Content-Length: " + std::to_string(stt_disabled_body.size()) + "\r\n\r\n" + stt_disabled_body;
    auto stt_disabled_valid_file = request(raw);
    expect(stt_disabled_valid_file.find("404 Not Found") != std::string::npos, "stt disabled valid file status");
    expect(stt_disabled_valid_file.find("\"detail\":\"Speech to text is disabled.\"") != std::string::npos, "stt disabled valid file body");
    auto wakeword_disabled = request("GET /v1/wakeword/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_disabled.find("404 Not Found") != std::string::npos, "wakeword disabled status");
    expect(wakeword_disabled.find("Wake word detection is disabled") != std::string::npos, "wakeword disabled body");
    const std::string wakeword_disabled_invalid_train_body = "{}";
    raw = "POST /v1/wakeword/train HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(wakeword_disabled_invalid_train_body.size()) + "\r\n\r\n" + wakeword_disabled_invalid_train_body;
    auto wakeword_disabled_invalid_train = request(raw);
    expect(wakeword_disabled_invalid_train.find("422 Unprocessable Entity") != std::string::npos, "wakeword disabled invalid train validates first");
    expect(wakeword_disabled_invalid_train.find("\"loc\":[\"body\",\"word\"]") != std::string::npos, "wakeword disabled invalid train loc");
    const std::string wakeword_disabled_valid_train_body = "{\"word\":\"jarvis\"}";
    raw = "POST /v1/wakeword/train HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(wakeword_disabled_valid_train_body.size()) + "\r\n\r\n" + wakeword_disabled_valid_train_body;
    auto wakeword_disabled_valid_train = request(raw);
    expect(wakeword_disabled_valid_train.find("404 Not Found") != std::string::npos, "wakeword disabled valid train status");
    expect(wakeword_disabled_valid_train.find("Wake word detection is disabled") != std::string::npos, "wakeword disabled valid train body");
    auto wakeword_disabled_invalid_limit = request("GET /v1/wakeword/jobs?limit=x HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_disabled_invalid_limit.find("422 Unprocessable Entity") != std::string::npos, "wakeword disabled invalid limit validates first");
    expect(wakeword_disabled_invalid_limit.find("\"loc\":[\"query\",\"limit\"]") != std::string::npos, "wakeword disabled invalid limit loc");
    const std::string wakeword_disabled_missing_file_body = "{}";
    raw = "POST /v1/wakeword/predict/jarvis/file HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(wakeword_disabled_missing_file_body.size()) + "\r\n\r\n" + wakeword_disabled_missing_file_body;
    auto wakeword_disabled_missing_file = request(raw);
    expect(wakeword_disabled_missing_file.find("422 Unprocessable Entity") != std::string::npos, "wakeword disabled missing file validates first");
    expect(wakeword_disabled_missing_file.find("\"loc\":[\"body\",\"file\"]") != std::string::npos, "wakeword disabled missing file loc");
    raw = "POST /v1/wakeword/predict/jarvis/file HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + plain_file_field_boundary + "\r\n"
        "Content-Length: " + std::to_string(plain_file_field_body.size()) + "\r\n\r\n" + plain_file_field_body;
    auto wakeword_plain_file_field = request(raw);
    expect(wakeword_plain_file_field.find("422 Unprocessable Entity") != std::string::npos, "wakeword plain file form field rejects as upload");
    expect(wakeword_plain_file_field.find("Expected UploadFile") != std::string::npos, "wakeword plain file form field uploadfile message");
    unsetenv("STT_ENABLED");

    const char *saved_whisper_cpp_bin_env = getenv("WHISPER_CPP_BIN");
    const char *saved_whisper_model_env = getenv("WHISPER_MODEL");
    const char *saved_voice_server_for_stt_env = getenv("VOICE_SERVER");
    const char *saved_fallback_server_for_stt_env = getenv("FALLBACK_SERVER");
    const bool had_whisper_cpp_bin_env = saved_whisper_cpp_bin_env != nullptr;
    const bool had_whisper_model_env = saved_whisper_model_env != nullptr;
    const bool had_voice_server_for_stt_env = saved_voice_server_for_stt_env != nullptr;
    const bool had_fallback_server_for_stt_env = saved_fallback_server_for_stt_env != nullptr;
    const std::string saved_whisper_cpp_bin = saved_whisper_cpp_bin_env ? saved_whisper_cpp_bin_env : "";
    const std::string saved_whisper_model = saved_whisper_model_env ? saved_whisper_model_env : "";
    const std::string saved_voice_server_for_stt = saved_voice_server_for_stt_env ? saved_voice_server_for_stt_env : "";
    const std::string saved_fallback_server_for_stt = saved_fallback_server_for_stt_env ? saved_fallback_server_for_stt_env : "";
    const auto fake_whisper_bin = test_cwd / "fake-whisper-cli";
    const auto fake_whisper_args = test_cwd / "fake-whisper-args.txt";
    const auto fake_whisper_model = test_cwd / "fake-whisper-model.bin";
    {
        write_whisper_fixture(fake_whisper_bin, "{\"language\":\"en\",\"text\":\"native local transcript\",\"segments\":[{\"id\":0,\"start\":1.25,\"end\":2.5,\"text\":\"native local transcript\"}]}", fake_whisper_args, 0, "ignored stdout");
        std::ofstream model(fake_whisper_model, std::ios::binary);
        model << "fake-model";
    }
    std::filesystem::permissions(
        fake_whisper_bin,
        std::filesystem::perms::owner_exec | std::filesystem::perms::group_exec | std::filesystem::perms::others_exec,
        std::filesystem::perm_options::add
    );
    setenv("STT_ENABLED", "true", 1);
    setenv("WHISPER_CPP_BIN", fake_whisper_bin.c_str(), 1);
    setenv("WHISPER_MODEL", fake_whisper_model.c_str(), 1);
    setenv("VOICE_SERVER", "", 1);
    setenv("FALLBACK_SERVER", "", 1);
    const std::string stt_local_boundary = "stt-local-boundary";
    const std::string stt_local_body =
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "RIFF-WAVE\r\n"
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"language\"\r\n\r\n"
        "en\r\n"
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"prompt\"\r\n\r\n"
        "context words\r\n"
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"temperature\"\r\n\r\n"
        "0.3\r\n"
        "--" + stt_local_boundary + "--\r\n";
    raw = "POST /v1/audio/transcriptions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + stt_local_boundary + "\r\n"
        "Content-Length: " + std::to_string(stt_local_body.size()) + "\r\n\r\n" + stt_local_body;
    auto stt_local = request(raw);
    expect(stt_local.find("200 OK") != std::string::npos, "local whisper stt status");
    expect(stt_local.find("\"text\":\"native local transcript\"") != std::string::npos, "local whisper stt json body");
    {
        std::ifstream args_file(fake_whisper_args, std::ios::binary);
        std::string args((std::istreambuf_iterator<char>(args_file)), std::istreambuf_iterator<char>());
        expect(args.find("-tp\n0.3") != std::string::npos, "local whisper stt passes temperature");
        expect(args.find("-bs\n1") != std::string::npos, "local whisper stt uses python beam default");
    }
    const std::string stt_local_text_body =
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "RIFF-WAVE\r\n"
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"response_format\"\r\n\r\n"
        "text\r\n"
        "--" + stt_local_boundary + "--\r\n";
    raw = "POST /v1/audio/transcriptions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + stt_local_boundary + "\r\n"
        "Content-Length: " + std::to_string(stt_local_text_body.size()) + "\r\n\r\n" + stt_local_text_body;
    auto stt_local_text = request(raw);
    expect(stt_local_text.find("200 OK") != std::string::npos, "local whisper stt text status");
    expect(stt_local_text.find("Content-Type: text/plain") != std::string::npos, "local whisper stt text content type");
    expect(stt_local_text.find("\r\n\r\nnative local transcript") != std::string::npos, "local whisper stt text body");
    const std::string stt_local_verbose_body =
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "RIFF-WAVE\r\n"
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"response_format\"\r\n\r\n"
        "verbose_json\r\n"
        "--" + stt_local_boundary + "--\r\n";
    raw = "POST /v1/audio/transcriptions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + stt_local_boundary + "\r\n"
        "Content-Length: " + std::to_string(stt_local_verbose_body.size()) + "\r\n\r\n" + stt_local_verbose_body;
    auto stt_local_verbose = request(raw);
    expect(stt_local_verbose.find("200 OK") != std::string::npos, "local whisper stt verbose status");
    expect(stt_local_verbose.find("\"segments\":[{\"id\":0,\"start\":1.25,\"end\":2.5,\"text\":\"native local transcript\"}]") != std::string::npos, "local whisper stt verbose segments");
    const std::string stt_local_diarized_body =
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "RIFF-WAVE\r\n"
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"response_format\"\r\n\r\n"
        "verbose_json\r\n"
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"enable_diarization\"\r\n\r\n"
        "true\r\n"
        "--" + stt_local_boundary + "--\r\n";
    raw = "POST /v1/audio/transcriptions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + stt_local_boundary + "\r\n"
        "Content-Length: " + std::to_string(stt_local_diarized_body.size()) + "\r\n\r\n" + stt_local_diarized_body;
    auto stt_local_diarized = request(raw);
    expect(stt_local_diarized.find("200 OK") != std::string::npos, "local whisper stt diarized status");
    expect(stt_local_diarized.find("\"text\":\"[SPEAKER_00]: native local transcript\"") != std::string::npos, "local whisper stt diarized text");
    expect(stt_local_diarized.find("\"speaker\":\"SPEAKER_00\"") != std::string::npos, "local whisper stt diarized segment speaker");
    const auto fake_whisper_diarize_bin = test_cwd / "fake-whisper-diarize-cli";
    {
        write_whisper_fixture(fake_whisper_diarize_bin, "{\"language\":\"en\",\"text\":\"alpha beta\",\"segments\":[{\"id\":0,\"start\":0.0,\"end\":1.0,\"text\":\"alpha\"},{\"id\":1,\"start\":1.5,\"end\":2.5,\"text\":\"beta\"}]}", {}, 0, "");
    }
    std::filesystem::permissions(
        fake_whisper_diarize_bin,
        std::filesystem::perms::owner_exec | std::filesystem::perms::group_exec | std::filesystem::perms::others_exec,
        std::filesystem::perm_options::add
    );
    setenv("WHISPER_CPP_BIN", fake_whisper_diarize_bin.c_str(), 1);
    const std::string diarize_wav = make_test_wav_pcm16();
    const std::string stt_local_clustered_body =
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        + diarize_wav + "\r\n"
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"response_format\"\r\n\r\n"
        "verbose_json\r\n"
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"enable_diarization\"\r\n\r\n"
        "true\r\n"
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"num_speakers\"\r\n\r\n"
        "2\r\n"
        "--" + stt_local_boundary + "--\r\n";
    raw = "POST /v1/audio/transcriptions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + stt_local_boundary + "\r\n"
        "Content-Length: " + std::to_string(stt_local_clustered_body.size()) + "\r\n\r\n" + stt_local_clustered_body;
    auto stt_local_clustered = request(raw);
    expect(stt_local_clustered.find("200 OK") != std::string::npos, "local whisper stt clustered diarization status");
    expect(stt_local_clustered.find("[SPEAKER_00]: alpha") != std::string::npos, "local whisper stt clustered first speaker text");
    expect(stt_local_clustered.find("[SPEAKER_01]: beta") != std::string::npos, "local whisper stt clustered second speaker text");
    expect(stt_local_clustered.find("\"speaker\":\"SPEAKER_00\"") != std::string::npos, "local whisper stt clustered first speaker");
    expect(stt_local_clustered.find("\"speaker\":\"SPEAKER_01\"") != std::string::npos, "local whisper stt clustered second speaker");
    setenv("WHISPER_CPP_BIN", fake_whisper_bin.c_str(), 1);
    const std::string stt_local_srt_body =
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "RIFF-WAVE\r\n"
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"response_format\"\r\n\r\n"
        "srt\r\n"
        "--" + stt_local_boundary + "--\r\n";
    raw = "POST /v1/audio/transcriptions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + stt_local_boundary + "\r\n"
        "Content-Length: " + std::to_string(stt_local_srt_body.size()) + "\r\n\r\n" + stt_local_srt_body;
    auto stt_local_srt = request(raw);
    expect(stt_local_srt.find("200 OK") != std::string::npos, "local whisper stt srt status");
    expect(stt_local_srt.find("Content-Type: text/plain") != std::string::npos, "local whisper stt srt content type");
    expect(stt_local_srt.find("1\n00:00:01,250 --> 00:00:02,500\nnative local transcript") != std::string::npos, "local whisper stt srt body");
    const std::string stt_local_translation_body =
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "RIFF-WAVE\r\n"
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"language\"\r\n\r\n"
        "es\r\n"
        "--" + stt_local_boundary + "--\r\n";
    raw = "POST /v1/audio/translations HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + stt_local_boundary + "\r\n"
        "Content-Length: " + std::to_string(stt_local_translation_body.size()) + "\r\n\r\n" + stt_local_translation_body;
    auto stt_local_translation = request(raw);
    expect(stt_local_translation.find("200 OK") != std::string::npos, "local whisper translation status");
    expect(stt_local_translation.find("\"text\":\"native local transcript\"") != std::string::npos, "local whisper translation body");
    const std::string stt_local_subtitles_body =
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "RIFF-WAVE\r\n"
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"target_languages\"\r\n\r\n"
        "en\r\n"
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"format\"\r\n\r\n"
        "vtt\r\n"
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"beam_size\"\r\n\r\n"
        "7\r\n"
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"fast_mode\"\r\n\r\n"
        "true\r\n"
        "--" + stt_local_boundary + "--\r\n";
    raw = "POST /v1/audio/subtitles HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + stt_local_boundary + "\r\n"
        "Content-Length: " + std::to_string(stt_local_subtitles_body.size()) + "\r\n\r\n" + stt_local_subtitles_body;
    auto stt_local_subtitles = request(raw);
    expect(stt_local_subtitles.find("200 OK") != std::string::npos, "local whisper subtitles status");
    expect(stt_local_subtitles.find("\"source_language\":\"en\"") != std::string::npos, "local whisper subtitles source language");
    expect(stt_local_subtitles.find("WEBVTT\\n\\n00:00:01.250 --&gt;") == std::string::npos, "local whisper subtitles does not html escape");
    expect(stt_local_subtitles.find("WEBVTT\\n\\n00:00:01.250 --> 00:00:02.500\\nnative local transcript") != std::string::npos, "local whisper subtitles vtt body");
    {
        std::ifstream args_file(fake_whisper_args, std::ios::binary);
        std::string args((std::istreambuf_iterator<char>(args_file)), std::istreambuf_iterator<char>());
        expect(args.find("-bs\n1") != std::string::npos, "local whisper subtitles fast mode uses beam one");
        expect(args.find("-mc\n0") != std::string::npos, "local whisper subtitles fast mode disables context");
    }
    const std::string stt_local_subtitles_json_body =
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "RIFF-WAVE\r\n"
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"format\"\r\n\r\n"
        "json\r\n"
        "--" + stt_local_boundary + "--\r\n";
    raw = "POST /v1/audio/subtitles HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + stt_local_boundary + "\r\n"
        "Content-Length: " + std::to_string(stt_local_subtitles_json_body.size()) + "\r\n\r\n" + stt_local_subtitles_json_body;
    auto stt_local_subtitles_json = request(raw);
    expect(stt_local_subtitles_json.find("200 OK") != std::string::npos, "local whisper subtitles json status");
    expect(stt_local_subtitles_json.find("\"subtitles\":{\"en\":[{\"id\":0,\"start\":1.25,\"end\":2.5,\"text\":\"native local transcript\"}]}") != std::string::npos, "local whisper subtitles json segments");
    const std::string stt_local_subtitles_translate_body =
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "RIFF-WAVE\r\n"
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"target_languages\"\r\n\r\n"
        "en,es\r\n"
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"format\"\r\n\r\n"
        "json\r\n"
        "--" + stt_local_boundary + "--\r\n";
    raw = "POST /v1/audio/subtitles HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + stt_local_boundary + "\r\n"
        "Content-Length: " + std::to_string(stt_local_subtitles_translate_body.size()) + "\r\n\r\n" + stt_local_subtitles_translate_body;
    auto stt_local_subtitles_translate = request(raw);
    expect(stt_local_subtitles_translate.find("200 OK") != std::string::npos, "local whisper subtitles translation status");
    expect(stt_local_subtitles_translate.find("\"en\":[{\"id\":0,\"start\":1.25,\"end\":2.5,\"text\":\"native local transcript\"}]") != std::string::npos, "local whisper subtitles translation original language");
    expect(stt_local_subtitles_translate.find("\"es\":[{\"id\":0,\"start\":1.25,\"end\":2.5,\"text\":\"translated native local transcript\"}]") != std::string::npos, "local whisper subtitles translated language");
    const std::string chat_audio_format_body = "{\"model\":\"native-test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"UklGRi1XQVZF\"}],\"audio_format\":\"wav\"}";
    setenv("STT_UNLOAD_LLM_DURING_GENERATION", "false", 1);
    raw = "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(chat_audio_format_body.size()) + "\r\n\r\n" + chat_audio_format_body;
    auto chat_audio_format = request(raw);
    expect(chat_audio_format.find("200 OK") != std::string::npos, "chat audio_format status");
    expect(chat_audio_format.find("Native ezlocalai echo backend received: native local transcript") != std::string::npos, "chat audio_format transcribed before llm");
    expect(chat_audio_format.find("UklGRi1XQVZF") == std::string::npos, "chat audio_format does not pass base64 to llm");
    const std::string completion_audio_format_body = "{\"model\":\"native-test-model\",\"prompt\":\"UklGRi1XQVZF\",\"audio_format\":\"wav\"}";
    raw = "POST /v1/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(completion_audio_format_body.size()) + "\r\n\r\n" + completion_audio_format_body;
    auto completion_audio_format = request(raw);
    expect(completion_audio_format.find("200 OK") != std::string::npos, "completion audio_format status");
    expect(completion_audio_format.find("Native ezlocalai echo backend received: native local transcript") != std::string::npos, "completion audio_format transcribed before llm");
    expect(completion_audio_format.find("UklGRi1XQVZF") == std::string::npos, "completion audio_format does not pass base64 to llm");
    unsetenv("STT_UNLOAD_LLM_DURING_GENERATION");

    setenv("VOICE_SERVER", "http://127.0.0.1:18187", 1);
    std::thread fake_chat_voice_tts(
        fake_voice_forward_once,
        18187,
        "POST /v1/audio/speech HTTP/1.1",
        "Content-Type: application/json",
        "\"input\":\"Native ezlocalai echo backend received: voice chat\"",
        "",
        "audio/wav",
        "RIFFCHATVOICE",
        ""
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string chat_voice_body = "{\"model\":\"native-test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"voice chat\"}],\"voice\":\"amy\",\"language\":\"en\"}";
    raw = "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(chat_voice_body.size()) + "\r\n\r\n" + chat_voice_body;
    auto chat_voice_response = request(raw);
    fake_chat_voice_tts.join();
    expect(chat_voice_response.find("200 OK") != std::string::npos, "chat voice audio response status");
    expect(chat_voice_response.find("Native ezlocalai echo backend received: voice chat\\nhttp://localhost:8091/outputs/") != std::string::npos, "chat voice appends output URL");
    const std::string chat_voice_response_body = response_body(chat_voice_response);
    const size_t chat_voice_output = chat_voice_response_body.find("/outputs/");
    const size_t chat_voice_wav = chat_voice_response_body.find(".wav", chat_voice_output);
    expect(chat_voice_output != std::string::npos && chat_voice_wav != std::string::npos, "chat voice output path");
    const std::string chat_voice_path = chat_voice_response_body.substr(chat_voice_output, chat_voice_wav + 4 - chat_voice_output);
    auto chat_voice_asset = request("GET " + chat_voice_path + " HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(chat_voice_asset.find("200 OK") != std::string::npos, "chat voice output served");
    expect(chat_voice_asset.find("RIFFCHATVOICE") != std::string::npos, "chat voice output body");

    setenv("VOICE_SERVER", "http://127.0.0.1:18188", 1);
    std::thread fake_completion_voice_tts(
        fake_voice_forward_once,
        18188,
        "POST /v1/audio/speech HTTP/1.1",
        "Content-Type: application/json",
        "\"input\":\"Native ezlocalai echo backend received: voice completion\"",
        "",
        "audio/wav",
        "RIFFCOMPLETIONVOICE",
        ""
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string completion_voice_body = "{\"model\":\"native-test-model\",\"prompt\":\"voice completion\",\"voice\":\"amy\",\"language\":\"en\"}";
    raw = "POST /v1/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(completion_voice_body.size()) + "\r\n\r\n" + completion_voice_body;
    auto completion_voice_response = request(raw);
    fake_completion_voice_tts.join();
    expect(completion_voice_response.find("200 OK") != std::string::npos, "completion voice audio response status");
    expect(completion_voice_response.find("Native ezlocalai echo backend received: voice completion\\nhttp://localhost:8091/outputs/") != std::string::npos, "completion voice appends output URL");
    const std::string completion_voice_response_body = response_body(completion_voice_response);
    const size_t completion_voice_output = completion_voice_response_body.find("/outputs/");
    const size_t completion_voice_wav = completion_voice_response_body.find(".wav", completion_voice_output);
    expect(completion_voice_output != std::string::npos && completion_voice_wav != std::string::npos, "completion voice output path");
    const std::string completion_voice_path = completion_voice_response_body.substr(completion_voice_output, completion_voice_wav + 4 - completion_voice_output);
    auto completion_voice_asset = request("GET " + completion_voice_path + " HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(completion_voice_asset.find("200 OK") != std::string::npos, "completion voice output served");
    expect(completion_voice_asset.find("RIFFCOMPLETIONVOICE") != std::string::npos, "completion voice output body");
    setenv("VOICE_SERVER", "", 1);

    const auto fake_slow_whisper_bin = test_cwd / "fake-slow-whisper-cli";
    {
        write_whisper_fixture(fake_slow_whisper_bin, "{\"language\":\"en\",\"text\":\"slow native transcript\",\"segments\":[{\"id\":0,\"start\":0.0,\"end\":1.0,\"text\":\"slow native transcript\"}]}", {}, 1000, "");
    }
    std::filesystem::permissions(
        fake_slow_whisper_bin,
        std::filesystem::perms::owner_exec | std::filesystem::perms::group_exec | std::filesystem::perms::others_exec,
        std::filesystem::perm_options::add
    );
    setenv("WHISPER_CPP_BIN", fake_slow_whisper_bin.c_str(), 1);
    std::string slow_stt_response;
    const std::string slow_stt_body =
        "--" + stt_local_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "RIFF-WAVE\r\n"
        "--" + stt_local_boundary + "--\r\n";
    const std::string slow_stt_raw = "POST /v1/audio/transcriptions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + stt_local_boundary + "\r\n"
        "Content-Length: " + std::to_string(slow_stt_body.size()) + "\r\n\r\n" + slow_stt_body;
    std::thread slow_stt_thread([&]() {
        slow_stt_response = request(slow_stt_raw);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    auto resources_during_stt = request("GET /v1/resources HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(resources_during_stt.find("200 OK") != std::string::npos, "resources during stt status");
    expect(resources_during_stt.find("\"loaded_models\":{\"stt\":{") != std::string::npos, "resources during stt loaded model");
    expect(resources_during_stt.find("\"name\":\"fake-whisper-model.bin\"") != std::string::npos, "resources during stt model name");
    expect(resources_during_stt.find("\"device\":\"whisper.cpp\"") != std::string::npos, "resources during stt device");
    expect(resources_during_stt.find("\"stt\":{\"capacity\":1,\"in_flight\":1,\"queued\":0,\"available\":0}") != std::string::npos, "resources during stt slot busy");
    slow_stt_thread.join();
    expect(slow_stt_response.find("200 OK") != std::string::npos, "slow local whisper stt status");
    expect(slow_stt_response.find("\"text\":\"slow native transcript\"") != std::string::npos, "slow local whisper stt body");
    setenv("WHISPER_CPP_BIN", fake_whisper_bin.c_str(), 1);
    if (had_whisper_cpp_bin_env) { setenv("WHISPER_CPP_BIN", saved_whisper_cpp_bin.c_str(), 1); } else { unsetenv("WHISPER_CPP_BIN"); }
    if (had_whisper_model_env) { setenv("WHISPER_MODEL", saved_whisper_model.c_str(), 1); } else { unsetenv("WHISPER_MODEL"); }
    if (had_voice_server_for_stt_env) { setenv("VOICE_SERVER", saved_voice_server_for_stt.c_str(), 1); } else { unsetenv("VOICE_SERVER"); }
    if (had_fallback_server_for_stt_env) { setenv("FALLBACK_SERVER", saved_fallback_server_for_stt.c_str(), 1); } else { unsetenv("FALLBACK_SERVER"); }
    unsetenv("STT_ENABLED");

    auto wakeword_default_enabled = request("GET /v1/wakeword/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_default_enabled.find("200 OK") != std::string::npos, "wakeword default enabled status");
    expect(wakeword_default_enabled.find("\"models\":[]") != std::string::npos, "wakeword default enabled body");
    setenv("STT_ENABLED", "true", 1);
    auto wakeword_models_empty = request("GET /v1/wakeword/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_models_empty.find("200 OK") != std::string::npos, "wakeword empty models status");
    expect(wakeword_models_empty.find("\"models\":[]") != std::string::npos, "wakeword empty models body");
    expect(wakeword_models_empty.find("\"total\":0") != std::string::npos, "wakeword empty models total");
    auto wakeword_model_missing = request("GET /v1/wakeword/models/jarvis HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_model_missing.find("404 Not Found") != std::string::npos, "wakeword model missing status");
    expect(wakeword_model_missing.find("No model found for word 'jarvis'. Use POST /v1/wakeword/train to create one.") != std::string::npos, "wakeword model missing body");
    auto wakeword_config_missing = request("GET /v1/wakeword/models/jarvis/config HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_config_missing.find("404 Not Found") != std::string::npos, "wakeword config missing status");
    expect(wakeword_config_missing.find("No model found for word 'jarvis'") != std::string::npos, "wakeword config missing body");
    auto wakeword_delete_missing = request("DELETE /v1/wakeword/models/jarvis HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_delete_missing.find("404 Not Found") != std::string::npos, "wakeword delete missing status");
    expect(wakeword_delete_missing.find("No model found for word 'jarvis'") != std::string::npos, "wakeword delete missing body");
    auto wakeword_jobs_empty = request("GET /v1/wakeword/jobs HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_jobs_empty.find("200 OK") != std::string::npos, "wakeword empty jobs status");
    expect(wakeword_jobs_empty.find("\r\n[]") != std::string::npos, "wakeword empty jobs body");
    auto wakeword_jobs_invalid = request("GET /v1/wakeword/jobs?status=nope HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_jobs_invalid.find("400 Bad Request") != std::string::npos, "wakeword invalid status code");
    expect(wakeword_jobs_invalid.find("Invalid status. Must be one of: ['pending', 'generating_samples', 'augmenting', 'training', 'exporting', 'completed', 'failed', 'cancelled']") != std::string::npos, "wakeword invalid status body");
    auto wakeword_jobs_bad_limit = request("GET /v1/wakeword/jobs?limit=x HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_jobs_bad_limit.find("422 Unprocessable Entity") != std::string::npos, "wakeword bad limit status");
    expect(wakeword_jobs_bad_limit.find("\"type\":\"int_parsing\"") != std::string::npos, "wakeword bad limit type");
    auto wakeword_job_missing = request("GET /v1/wakeword/jobs/job-1 HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_job_missing.find("404 Not Found") != std::string::npos, "wakeword job missing status");
    expect(wakeword_job_missing.find("\"detail\":\"Job not found\"") != std::string::npos, "wakeword job missing body");
    auto wakeword_cancel_missing = request("DELETE /v1/wakeword/jobs/job-1 HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_cancel_missing.find("400 Bad Request") != std::string::npos, "wakeword cancel missing status");
    expect(wakeword_cancel_missing.find("Could not cancel job (may already be completed or cancelled)") != std::string::npos, "wakeword cancel missing body");
    std::filesystem::create_directories("wakeword_jobs");
    {
        std::ofstream old_job("wakeword_jobs/job-old.json", std::ios::binary);
        old_job << "{\"job_id\":\"job-old\",\"word\":\"old\",\"status\":\"completed\",\"created_at\":\"2026-01-01T00:00:00\",\"updated_at\":\"2026-01-01T00:05:00\",\"progress\":1.0,\"current_stage\":\"done\",\"error_message\":null,\"estimated_completion\":null,\"model_path\":\"wakeword_models/old\",\"metrics\":{\"accuracy\":0.9}}";
        std::ofstream running_job("wakeword_jobs/job-running.json", std::ios::binary);
        running_job << "{\"job_id\":\"job-running\",\"word\":\"hey ez\",\"status\":\"training\",\"created_at\":\"2026-02-01T00:00:00\",\"updated_at\":\"2026-02-01T00:03:00\",\"progress\":0.5,\"current_stage\":\"training\",\"error_message\":null,\"estimated_completion\":\"2026-02-01T00:15:00\",\"model_path\":null,\"metrics\":{\"loss\":0.1}}";
        std::ofstream computer_job("wakeword_jobs/job-computer.json", std::ios::binary);
        computer_job << "{\"job_id\":\"job-computer\",\"word\":\"computer\",\"status\":\"pending\",\"created_at\":\"2026-01-15T00:00:00\",\"updated_at\":\"2026-01-15T00:01:00\",\"progress\":0.0,\"current_stage\":\"queued\",\"error_message\":null,\"estimated_completion\":\"2026-01-15T00:15:00\",\"model_path\":null,\"metrics\":{}}";
        std::ofstream invalid_status_job("wakeword_jobs/job-invalid-status.json", std::ios::binary);
        invalid_status_job << "{\"job_id\":\"job-invalid-status\",\"word\":\"ghost\",\"status\":\"nope\",\"created_at\":\"2026-03-01T00:00:00\",\"updated_at\":\"2026-03-01T00:01:00\",\"progress\":0.0,\"current_stage\":\"queued\",\"error_message\":null,\"estimated_completion\":null,\"model_path\":null,\"metrics\":{}}";
        std::ofstream invalid_json_job("wakeword_jobs/job-invalid-json.json", std::ios::binary);
        invalid_json_job << "{\"job_id\":\"job-invalid-json\",\"word\":\"ghost\",\"status\":\"pending\"";
    }
    // Python's manager loads persisted records once. External file changes are
    // visible after a new application is constructed, not on every request.
    auto wakeword_jobs_cached = request("GET /v1/wakeword/jobs HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_jobs_cached.find("\r\n[]") != std::string::npos, "wakeword manager retains cached empty jobs");
    server.reset();
    server = std::make_unique<ezlocalai::NativeServer>(config, std::make_unique<ezlocalai::EchoInferenceBackend>(config.default_model));
    expect(server->start("127.0.0.1", 18091, &error), error.c_str());
    auto wakeword_jobs_file = request("GET /v1/wakeword/jobs HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_jobs_file.find("200 OK") != std::string::npos, "wakeword file jobs status");
    expect(wakeword_jobs_file.find("\"job_id\":\"job-running\"") != std::string::npos, "wakeword file jobs running");
    expect(wakeword_jobs_file.find("\"job_id\":\"job-old\"") != std::string::npos, "wakeword file jobs old");
    expect(wakeword_jobs_file.find("\"job_id\":\"job-invalid-status\"") == std::string::npos, "wakeword skips invalid status jobs");
    expect(wakeword_jobs_file.find("\"job_id\":\"job-invalid-json\"") == std::string::npos, "wakeword skips invalid json jobs");
    expect(wakeword_jobs_file.find("\"job_id\":\"job-running\"") < wakeword_jobs_file.find("\"job_id\":\"job-old\""), "wakeword file jobs created desc");
    auto wakeword_jobs_completed = request("GET /v1/wakeword/jobs?status=completed HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_jobs_completed.find("200 OK") != std::string::npos, "wakeword completed jobs status");
    expect(wakeword_jobs_completed.find("\"job_id\":\"job-old\"") != std::string::npos, "wakeword completed jobs includes old");
    expect(wakeword_jobs_completed.find("\"job_id\":\"job-running\"") == std::string::npos, "wakeword completed jobs filters running");
    auto wakeword_jobs_limited = request("GET /v1/wakeword/jobs?limit=1 HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_jobs_limited.find("\"job_id\":\"job-running\"") != std::string::npos, "wakeword limited jobs includes newest");
    expect(wakeword_jobs_limited.find("\"job_id\":\"job-old\"") == std::string::npos, "wakeword limited jobs excludes older");
    auto wakeword_jobs_negative_limit = request("GET /v1/wakeword/jobs?limit=-1 HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_jobs_negative_limit.find("\"job_id\":\"job-running\"") != std::string::npos, "wakeword negative limit includes newest");
    expect(wakeword_jobs_negative_limit.find("\"job_id\":\"job-computer\"") != std::string::npos, "wakeword negative limit includes middle");
    expect(wakeword_jobs_negative_limit.find("\"job_id\":\"job-old\"") == std::string::npos, "wakeword negative limit excludes last");
    auto wakeword_job_file = request("GET /v1/wakeword/jobs/job-running HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_job_file.find("200 OK") != std::string::npos, "wakeword job file status");
    expect(wakeword_job_file.find("\"current_stage\":\"training\"") != std::string::npos, "wakeword job file body");
    auto wakeword_model_training = request("GET /v1/wakeword/models/computer HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_model_training.find("200 OK") != std::string::npos, "wakeword model training status");
    expect(wakeword_model_training.find("\"status\":\"training_in_progress\"") != std::string::npos, "wakeword model training status body");
    expect(wakeword_model_training.find("\"job_id\":\"job-computer\"") != std::string::npos, "wakeword model training job id");
    expect(wakeword_model_training.find("Model for 'computer' is being trained. Check back soon.") != std::string::npos, "wakeword model training message");
    auto wakeword_invalid_status_not_training = request("GET /v1/wakeword/models/ghost HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_invalid_status_not_training.find("404 Not Found") != std::string::npos, "wakeword invalid status job not active");
    expect(wakeword_invalid_status_not_training.find("training_in_progress") == std::string::npos, "wakeword invalid status no training response");
    const std::string wakeword_train_existing_job_body = "{\"word\":\"computer\"}";
    raw = "POST /v1/wakeword/train HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(wakeword_train_existing_job_body.size()) + "\r\n\r\n" + wakeword_train_existing_job_body;
    auto wakeword_train_existing_job = request(raw);
    expect(wakeword_train_existing_job.find("200 OK") != std::string::npos, "wakeword train existing job status");
    expect(wakeword_train_existing_job.find("\"job_id\":\"job-computer\"") != std::string::npos, "wakeword train existing job id");
    expect(wakeword_train_existing_job.find("Training already in progress for 'computer'") != std::string::npos, "wakeword train existing job message");
    const std::string wakeword_train_missing_word_body = "{}";
    raw = "POST /v1/wakeword/train HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(wakeword_train_missing_word_body.size()) + "\r\n\r\n" + wakeword_train_missing_word_body;
    auto wakeword_train_missing_word = request(raw);
    expect(wakeword_train_missing_word.find("422 Unprocessable Entity") != std::string::npos, "wakeword train missing word status");
    expect(wakeword_train_missing_word.find("\"loc\":[\"body\",\"word\"]") != std::string::npos, "wakeword train missing word loc");
    expect(wakeword_train_missing_word.find("\"input\":{}") != std::string::npos, "wakeword train missing word input");
    const std::string wakeword_train_array_body = "[]";
    raw = "POST /v1/wakeword/train HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(wakeword_train_array_body.size()) + "\r\n\r\n" + wakeword_train_array_body;
    auto wakeword_train_array = request(raw);
    expect(wakeword_train_array.find("422 Unprocessable Entity") != std::string::npos, "wakeword train array status");
    expect(wakeword_train_array.find("\"type\":\"model_attributes_type\"") != std::string::npos, "wakeword train array type");
    expect(wakeword_train_array.find("Input should be a valid dictionary or object to extract fields from") != std::string::npos, "wakeword train array message");
    const std::string wakeword_train_bad_fields_body = "{\"word\":123,\"sample_count\":\"x\"}";
    raw = "POST /v1/wakeword/train HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(wakeword_train_bad_fields_body.size()) + "\r\n\r\n" + wakeword_train_bad_fields_body;
    auto wakeword_train_bad_fields = request(raw);
    expect(wakeword_train_bad_fields.find("422 Unprocessable Entity") != std::string::npos, "wakeword train bad fields status");
    expect(wakeword_train_bad_fields.find("\"type\":\"string_type\"") != std::string::npos, "wakeword train bad word type");
    expect(wakeword_train_bad_fields.find("\"type\":\"int_parsing\"") != std::string::npos, "wakeword train bad sample type");
    const std::string wakeword_train_empty_word_body = "{\"word\":\"\"}";
    raw = "POST /v1/wakeword/train HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(wakeword_train_empty_word_body.size()) + "\r\n\r\n" + wakeword_train_empty_word_body;
    auto wakeword_train_empty_word = request(raw);
    expect(wakeword_train_empty_word.find("400 Bad Request") != std::string::npos, "wakeword train empty word rejected");
    expect(wakeword_train_empty_word.find("Invalid wake word name") != std::string::npos, "wakeword train empty word detail");
    auto wakeword_cancel_file = request("DELETE /v1/wakeword/jobs/job-running HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_cancel_file.find("200 OK") != std::string::npos, "wakeword cancel file status");
    expect(wakeword_cancel_file.find("\"message\":\"Job cancelled successfully\"") != std::string::npos, "wakeword cancel file body");
    auto wakeword_cancelled_job = request("GET /v1/wakeword/jobs/job-running HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_cancelled_job.find("\"status\":\"cancelled\"") != std::string::npos, "wakeword cancelled job persisted");
    const std::string wakeword_train_new_body = "{\"word\":\"newwake\",\"sample_count\":10,\"epochs\":1,\"batch_size\":2}";
    raw = "POST /v1/wakeword/train HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(wakeword_train_new_body.size()) + "\r\n\r\n" + wakeword_train_new_body;
    auto wakeword_train_new = request(raw);
    expect(wakeword_train_new.find("200 OK") != std::string::npos, "wakeword train new status");
    expect(wakeword_train_new.find("\"job_id\":\"") != std::string::npos, "wakeword train new job id");
    expect(wakeword_train_new.find("\"job_id\":\"native-") == std::string::npos, "wakeword train new uuid shaped job id");
    expect(wakeword_train_new.find("\"status\":\"pending\"") != std::string::npos, "wakeword train new pending");
    expect(wakeword_train_new.find("Training started for 'newwake'. This typically takes 10-20 minutes.") != std::string::npos, "wakeword train new message");
    expect(wakeword_train_new.find("\"estimated_minutes\":15") != std::string::npos, "wakeword train new estimate");
    expect(wakeword_train_new.find("\"check_status_url\":\"/v1/wakeword/jobs/") != std::string::npos, "wakeword train new status url");
    auto wakeword_jobs_after_new = request("GET /v1/wakeword/jobs HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_jobs_after_new.find("\"word\":\"newwake\"") != std::string::npos, "wakeword train new persists job");
    expect(wakeword_jobs_after_new.find("\"estimated_completion\":\"") != std::string::npos, "wakeword train new persists estimate");
    // This echo fixture deliberately has no trainer. A background job now
    // reports that failure instead of remaining pending forever.
    std::string wakeword_failed_new;
    for (int attempt = 0; attempt < 100; ++attempt) {
        wakeword_failed_new = request("GET /v1/wakeword/jobs?status=failed HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
        if (wakeword_failed_new.find("\"word\":\"newwake\"") != std::string::npos) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    expect(wakeword_failed_new.find("\"word\":\"newwake\"") != std::string::npos, "wakeword unsupported trainer job terminates");
    expect(wakeword_failed_new.find("Native wake-word training is unavailable in this build") != std::string::npos, "wakeword unsupported trainer failure detail");
    auto wakeword_model_new_training = request("GET /v1/wakeword/models/newwake HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_model_new_training.find("404 Not Found") != std::string::npos, "failed wakeword job does not claim training is active");
    const std::string wakeword_predict_missing_field_body = "{}";
    raw = "POST /v1/wakeword/predict/jarvis HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(wakeword_predict_missing_field_body.size()) + "\r\n\r\n" + wakeword_predict_missing_field_body;
    auto wakeword_predict_missing_field = request(raw);
    expect(wakeword_predict_missing_field.find("422 Unprocessable Entity") != std::string::npos, "wakeword predict missing field status");
    expect(wakeword_predict_missing_field.find("\"loc\":[\"body\",\"audio_base64\"]") != std::string::npos, "wakeword predict missing field loc");
    const std::string wakeword_predict_array_body = "[]";
    raw = "POST /v1/wakeword/predict/jarvis HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(wakeword_predict_array_body.size()) + "\r\n\r\n" + wakeword_predict_array_body;
    auto wakeword_predict_array = request(raw);
    expect(wakeword_predict_array.find("422 Unprocessable Entity") != std::string::npos, "wakeword predict array status");
    expect(wakeword_predict_array.find("\"type\":\"model_attributes_type\"") != std::string::npos, "wakeword predict array type");
    const std::string wakeword_predict_bad_type_body = "{\"audio_base64\":123}";
    raw = "POST /v1/wakeword/predict/jarvis HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(wakeword_predict_bad_type_body.size()) + "\r\n\r\n" + wakeword_predict_bad_type_body;
    auto wakeword_predict_bad_type = request(raw);
    expect(wakeword_predict_bad_type.find("422 Unprocessable Entity") != std::string::npos, "wakeword predict bad type status");
    expect(wakeword_predict_bad_type.find("\"type\":\"string_type\"") != std::string::npos, "wakeword predict bad type");
    const std::string wakeword_predict_body = "{\"audio_base64\":\"AA==\"}";
    auto wakeword_predict_missing = request("POST /v1/wakeword/predict/jarvis HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(wakeword_predict_body.size()) + "\r\n\r\n" + wakeword_predict_body);
    expect(wakeword_predict_missing.find("404 Not Found") != std::string::npos, "wakeword predict missing status");
    expect(wakeword_predict_missing.find("No model found for word 'jarvis'. Train one first with POST /v1/wakeword/train") != std::string::npos, "wakeword predict missing body");
    const std::string wakeword_boundary = "wakeword-boundary";
    const std::string wakeword_missing_file_body =
        "--" + wakeword_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"other\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "data\r\n"
        "--" + wakeword_boundary + "--\r\n";
    raw = "POST /v1/wakeword/predict/jarvis/file HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + wakeword_boundary + "\r\n"
        "Content-Length: " + std::to_string(wakeword_missing_file_body.size()) + "\r\n\r\n" + wakeword_missing_file_body;
    auto wakeword_predict_file_missing_field = request(raw);
    expect(wakeword_predict_file_missing_field.find("422 Unprocessable Entity") != std::string::npos, "wakeword predict file missing field status");
    expect(wakeword_predict_file_missing_field.find("\"loc\":[\"body\",\"file\"]") != std::string::npos, "wakeword predict file missing field loc");
    const std::string wakeword_file_body =
        "--" + wakeword_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "RIFF\r\n"
        "--" + wakeword_boundary + "--\r\n";
    raw = "POST /v1/wakeword/predict/jarvis/file HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + wakeword_boundary + "\r\n"
        "Content-Length: " + std::to_string(wakeword_file_body.size()) + "\r\n\r\n" + wakeword_file_body;
    auto wakeword_predict_file_missing = request(raw);
    expect(wakeword_predict_file_missing.find("404 Not Found") != std::string::npos, "wakeword predict file missing status");
    expect(wakeword_predict_file_missing.find("No model found for word 'jarvis'. Train one first with POST /v1/wakeword/train") != std::string::npos, "wakeword predict file missing body");

    std::filesystem::create_directories("wakeword_models/hey_ez");
    {
        std::ofstream model("wakeword_models/hey_ez/model.pt", std::ios::binary);
        model << "PTMODEL";
        std::ofstream onnx("wakeword_models/hey_ez/model.onnx", std::ios::binary);
        onnx << "ONNXMODEL";
        std::ofstream config_file("wakeword_models/hey_ez/config.json", std::ios::binary);
        config_file << "{\"word\":\"hey ez\",\"threshold\":0.7}";
    }
    auto wakeword_models_file = request("GET /v1/wakeword/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_models_file.find("200 OK") != std::string::npos, "wakeword file models status");
    expect(wakeword_models_file.find("\"word\":\"hey ez\"") != std::string::npos, "wakeword file models word");
    expect(wakeword_models_file.find("\"pt\":{\"path\"") != std::string::npos, "wakeword file models pt");
    expect(wakeword_models_file.find("\"onnx\":{\"path\"") != std::string::npos, "wakeword file models onnx");
    expect(wakeword_models_file.find("\"total\":1") != std::string::npos, "wakeword file models total");
    auto wakeword_config_file = request("GET /v1/wakeword/models/hey_ez/config HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_config_file.find("200 OK") != std::string::npos, "wakeword file config status");
    expect(wakeword_config_file.find("\"threshold\":0.7") != std::string::npos, "wakeword file config body");
    auto wakeword_model_file = request("GET /v1/wakeword/models/hey_ez HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_model_file.find("200 OK") != std::string::npos, "wakeword file download status");
    expect(wakeword_model_file.find("Content-Type: application/octet-stream") != std::string::npos, "wakeword file download content type");
    expect(wakeword_model_file.find("Content-Disposition: attachment; filename=\"hey_ez_model.pt\"") != std::string::npos, "wakeword file download disposition");
    expect(wakeword_model_file.find("PTMODEL") != std::string::npos, "wakeword file download body");
    expect(response_header(wakeword_model_file, "Accept-Ranges") == "bytes", "wakeword download advertises ranges");
    const auto checkpoint_etag = response_header(wakeword_model_file, "ETag");
    expect(!checkpoint_etag.empty(), "wakeword download validator");
    auto checkpoint_range = request(
        "GET /v1/wakeword/models/hey_ez HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n"
        "Range: bytes=2-4\r\nIf-Range: " + checkpoint_etag + "\r\n\r\n");
    expect(checkpoint_range.find("206 Partial Content") != std::string::npos, "wakeword download range status");
    expect(response_header(checkpoint_range, "Content-Range") == "bytes 2-4/7", "wakeword download content range");
    expect(response_header(checkpoint_range, "Content-Length") == "3" && response_body(checkpoint_range) == "MOD",
           "wakeword range transfer matches declared length");
    auto checkpoint_conditional = request(
        "GET /v1/wakeword/models/hey_ez HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n"
        "If-None-Match: " + checkpoint_etag + "\r\n\r\n");
    expect(checkpoint_conditional.find("200 OK") != std::string::npos &&
           response_body(checkpoint_conditional) == "PTMODEL", "FileResponse does not apply StaticFiles 304");
    auto wakeword_model_encoded = request("GET /v1/wakeword/models/hey%20ez?format=pytorch HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_model_encoded.find("200 OK") != std::string::npos, "wakeword encoded word download status");
    expect(wakeword_model_encoded.find("Content-Disposition: attachment; filename=\"hey_ez_model.pt\"") != std::string::npos, "wakeword encoded word disposition");
    auto wakeword_model_onnx = request("GET /v1/wakeword/models/hey_ez?format=onnx HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_model_onnx.find("200 OK") != std::string::npos, "wakeword onnx download status");
    expect(wakeword_model_onnx.find("ONNXMODEL") != std::string::npos, "wakeword onnx body");
    auto wakeword_model_missing_format = request("GET /v1/wakeword/models/hey_ez?format=espdl HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_model_missing_format.find("404 Not Found") != std::string::npos, "wakeword missing format status");
    expect(wakeword_model_missing_format.find("Model format 'espdl' not available for word 'hey_ez'") != std::string::npos, "wakeword missing format body");
    raw = "POST /v1/wakeword/predict/hey_ez HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(wakeword_predict_body.size()) + "\r\n\r\n" + wakeword_predict_body;
    auto wakeword_predict_existing_invalid = request(raw);
    expect(wakeword_predict_existing_invalid.find("500 Internal Server Error") != std::string::npos, "wakeword predict existing invalid status");
    expect(wakeword_predict_existing_invalid.find("\"detail\":\"Prediction failed\"") != std::string::npos, "wakeword predict existing invalid body");
    raw = "POST /v1/wakeword/predict/hey_ez/file HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + wakeword_boundary + "\r\n"
        "Content-Length: " + std::to_string(wakeword_file_body.size()) + "\r\n\r\n" + wakeword_file_body;
    auto wakeword_predict_file_existing_invalid = request(raw);
    expect(wakeword_predict_file_existing_invalid.find("500 Internal Server Error") != std::string::npos, "wakeword predict file existing invalid status");
    expect(wakeword_predict_file_existing_invalid.find("\"detail\":\"Prediction failed\"") != std::string::npos, "wakeword predict file existing invalid body");
    const std::string wakeword_train_existing_model_body = "{\"word\":\"hey ez\"}";
    raw = "POST /v1/wakeword/train HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(wakeword_train_existing_model_body.size()) + "\r\n\r\n" + wakeword_train_existing_model_body;
    auto wakeword_train_existing_model = request(raw);
    expect(wakeword_train_existing_model.find("200 OK") != std::string::npos, "wakeword train existing model status");
    expect(wakeword_train_existing_model.find("\"status\":\"completed\"") != std::string::npos, "wakeword train existing model completed");
    expect(wakeword_train_existing_model.find("Model for 'hey ez' already exists") != std::string::npos, "wakeword train existing model message");
    auto wakeword_delete_file = request("DELETE /v1/wakeword/models/hey_ez HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(wakeword_delete_file.find("200 OK") != std::string::npos, "wakeword delete file status");
    expect(wakeword_delete_file.find("Model for 'hey_ez' deleted successfully") != std::string::npos, "wakeword delete file body");
    expect(!std::filesystem::exists("wakeword_models/hey_ez"), "wakeword delete removes model dir");
    unsetenv("STT_ENABLED");

    std::filesystem::create_directories("voices");
    {
        std::ofstream voice_file("voices/native-test.wav", std::ios::binary);
        voice_file << "RIFF";
        std::ofstream z_voice_file("voices/zz-order.wav", std::ios::binary);
        z_voice_file << "RIFF";
        std::ofstream a_voice_file("voices/aa-order.wav", std::ios::binary);
        a_voice_file << "RIFF";
    }
    auto voices = request("GET /v1/audio/voices HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(voices.find("200 OK") != std::string::npos, "voices status");
    expect(voices.find("\"voices\":[\"aa-order\",\"native-test\",\"zz-order\"]") != std::string::npos, "voices body follows sorted Path.glob stems");

    setenv("VOICE_SERVER", "http://127.0.0.1:18102", 1);
    std::thread fake_voice_server(fake_json_worker_many, 18102, std::vector<std::string>{
        "{\"voices\":[\"remote-voice\"]}",
        "{\"voices\":[\"remote-voice\"]}"
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto remote_voices = request("GET /v1/audio/voices HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    fake_voice_server.join();
    expect(remote_voices.find("200 OK") != std::string::npos, "remote voices status");
    expect(remote_voices.find("\"voices\":[\"remote-voice\"]") != std::string::npos, "remote voices body");
    unsetenv("VOICE_SERVER");

    const char *voice_saved_ssl_cert_file_raw = getenv("SSL_CERT_FILE");
    const bool voice_had_ssl_cert_file = voice_saved_ssl_cert_file_raw != nullptr;
    const std::string voice_saved_ssl_cert_file = voice_saved_ssl_cert_file_raw ? voice_saved_ssl_cert_file_raw : "";
    const auto voice_tls_cert = ensure_test_tls_cert();
    setenv("SSL_CERT_FILE", voice_tls_cert.cert.c_str(), 1);
    setenv("VOICE_SERVER", "https://127.0.0.2:18163", 1);
    setenv("VOICE_SERVER_API_KEY", "voice-secret", 1);
    TestListenerReady https_voice_api_ready(18163);
    std::thread fake_https_voice_api(
        fake_https_json_worker_many,
        18163,
        "Authorization: Bearer voice-secret",
        std::vector<std::pair<std::string, std::string>>{
            {"GET /v1/audio/voices HTTP/1.1", "{\"voices\":[\"availability\"]}"},
            {"GET /v1/audio/voices HTTP/1.1", "{\"voices\":[\"https-remote-voice\"]}"},
            {"POST /v1/audio/speech HTTP/1.1", "RIFFHTTPSVOICE"}
        },
        &https_voice_api_ready
    );
    https_voice_api_ready.wait();
    auto https_remote_voices = request("GET /v1/audio/voices HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(https_remote_voices.find("200 OK") != std::string::npos, "https remote voices status");
    expect(https_remote_voices.find("\"voices\":[\"https-remote-voice\"]") != std::string::npos, "https remote voices body");
    const std::string https_tts_body = "{\"input\":\"https speak\",\"voice\":\"amy\",\"language\":\"en\"}";
    raw = "POST /v1/audio/speech HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(https_tts_body.size()) + "\r\n\r\n" + https_tts_body;
    auto https_remote_tts = request(raw);
    fake_https_voice_api.join();
    expect(https_remote_tts.find("200 OK") != std::string::npos, "https remote tts status");
    expect(https_remote_tts.find("RIFFHTTPSVOICE") != std::string::npos, "https remote tts body");
    unsetenv("VOICE_SERVER");
    unsetenv("VOICE_SERVER_API_KEY");
    if (voice_had_ssl_cert_file) {
        setenv("SSL_CERT_FILE", voice_saved_ssl_cert_file.c_str(), 1);
    } else {
        unsetenv("SSL_CERT_FILE");
    }

    setenv("VOICE_SERVER", "http://127.0.0.1:18152", 1);
    std::thread fake_empty_voice_server(fake_json_worker_many, 18152, std::vector<std::string>{
        "{\"voices\":[\"availability\"]}",
        "{}"
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto empty_remote_voices = request("GET /v1/audio/voices HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    fake_empty_voice_server.join();
    expect(empty_remote_voices.find("200 OK") != std::string::npos, "empty remote voices fallback status");
    expect(empty_remote_voices.find("\"voices\":[\"aa-order\",\"native-test\",\"zz-order\"]") != std::string::npos, "empty remote voices falls back to sorted local voices");
    unsetenv("VOICE_SERVER");

    setenv("VOICE_SERVER", "http://127.0.0.1:18149", 1);
    std::thread fake_chunked_voice_server(fake_chunked_worker_many, 18149, "application/json", std::vector<std::vector<std::string>>{
        {"{\"voices\":[\"avail", "ability\"]}"},
        {"{\"voices\":[", "\"chunked-voice\"", "]}"}
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto chunked_remote_voices = request("GET /v1/audio/voices HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    fake_chunked_voice_server.join();
    expect(chunked_remote_voices.find("200 OK") != std::string::npos, "chunked remote voices status");
    expect(chunked_remote_voices.find("\"voices\":[\"chunked-voice\"]") != std::string::npos, "chunked remote voices body");
    expect(chunked_remote_voices.find("\r\n0\r\n\r\n") == std::string::npos, "chunked remote voices omits raw chunk terminator");
    unsetenv("VOICE_SERVER");

    setenv("VOICE_SERVER", "http://localhost:18126", 1);
    std::thread fake_voice_cache_probe(fake_json_worker_many, 18126, std::vector<std::string>{
        "{\"voices\":[\"availability\"]}",
        "{\"voices\":[\"first-cache-voice\"]}"
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto first_cached_voices = request("GET /v1/audio/voices HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    fake_voice_cache_probe.join();
    expect(first_cached_voices.find("200 OK") != std::string::npos, "first cached voices status");
    expect(first_cached_voices.find("\"voices\":[\"first-cache-voice\"]") != std::string::npos, "first cached voices body");
    std::thread fake_voice_cached_get(fake_json_worker_once, 18126, "{\"voices\":[\"cached-voice\"]}");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto second_cached_voices = request("GET /v1/audio/voices HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    fake_voice_cached_get.join();
    expect(second_cached_voices.find("200 OK") != std::string::npos, "second cached voices status");
    expect(second_cached_voices.find("\"voices\":[\"cached-voice\"]") != std::string::npos, "voice server availability cache skips second probe");
    unsetenv("VOICE_SERVER");

    setenv("VOICE_SERVER", "http://127.0.0.1:18103", 1);
    setenv("VOICE_SERVER_API_KEY", "voice-secret", 1);
    std::thread fake_voice_tts(
        fake_voice_forward_once,
        18103,
        "POST /v1/audio/speech HTTP/1.1",
        "Content-Type: application/json",
        "\"input\": \"speak\"",
        "\"model\"",
        "audio/wav",
        "RIFFREMOTE",
        "Authorization: Bearer voice-secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string tts_body = "{\"input\":\"speak\",\"voice\":\"amy\",\"language\":\"en\",\"model\":\"ignored\"}";
    raw = "POST /v1/audio/speech HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(tts_body.size()) + "\r\n\r\n" + tts_body;
    auto remote_tts = request(raw);
    fake_voice_tts.join();
    expect(remote_tts.find("200 OK") != std::string::npos, "remote tts status");
    expect(remote_tts.find("Content-Type: audio/wav") != std::string::npos, "remote tts content type");
    expect(remote_tts.find("RIFFREMOTE") != std::string::npos, "remote tts body");
    unsetenv("VOICE_SERVER");
    unsetenv("VOICE_SERVER_API_KEY");

    setenv("VOICE_SERVER", "http://127.0.0.1:18128", 1);
    std::thread fake_voice_tts_nulls(
        fake_voice_forward_once,
        18128,
        "POST /v1/audio/speech HTTP/1.1",
        "Content-Type: application/json",
        "\"voice\": null",
        "\"voice\": \"default\"",
        "audio/wav",
        "RIFFNULLVOICE",
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string tts_null_body = "{\"input\":\"nullable\",\"voice\":null,\"language\":null}";
    raw = "POST /v1/audio/speech HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(tts_null_body.size()) + "\r\n\r\n" + tts_null_body;
    auto remote_tts_nulls = request(raw);
    fake_voice_tts_nulls.join();
    expect(remote_tts_nulls.find("200 OK") != std::string::npos, "remote tts null optional status");
    expect(remote_tts_nulls.find("RIFFNULLVOICE") != std::string::npos, "remote tts null optional body");
    unsetenv("VOICE_SERVER");

    setenv("VOICE_SERVER", "http://127.0.0.1:18135", 1);
    std::thread fake_voice_tts_empty_optionals(
        fake_voice_forward_once,
        18135,
        "POST /v1/audio/speech HTTP/1.1",
        "Content-Type: application/json",
        "\"voice\": \"\", \"language\": \"\"",
        "\"voice\": \"default\"",
        "audio/wav",
        "RIFFEMPTYVOICE",
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string tts_empty_optionals_body = "{\"input\":\"empty optional\",\"voice\":\"\",\"language\":\"\"}";
    raw = "POST /v1/audio/speech HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(tts_empty_optionals_body.size()) + "\r\n\r\n" + tts_empty_optionals_body;
    auto remote_tts_empty_optionals = request(raw);
    fake_voice_tts_empty_optionals.join();
    expect(remote_tts_empty_optionals.find("200 OK") != std::string::npos, "remote tts empty optionals status");
    expect(remote_tts_empty_optionals.find("RIFFEMPTYVOICE") != std::string::npos, "remote tts empty optionals body");
    unsetenv("VOICE_SERVER");

    const char *saved_tts_data_voice_server_env = getenv("VOICE_SERVER");
    const char *saved_tts_data_pdftotext_env = getenv("PDFTOTEXT_BIN");
    const bool had_tts_data_voice_server = saved_tts_data_voice_server_env != nullptr;
    const bool had_tts_data_pdftotext = saved_tts_data_pdftotext_env != nullptr;
    const std::string saved_tts_data_voice_server = saved_tts_data_voice_server_env ? saved_tts_data_voice_server_env : "";
    const std::string saved_tts_data_pdftotext = saved_tts_data_pdftotext_env ? saved_tts_data_pdftotext_env : "";
    const auto fake_tts_data_pdftotext_bin = test_cwd / "fake-pdftotext";
    write_output_fixture(fake_tts_data_pdftotext_bin, "pdf extracted text");
    std::filesystem::permissions(fake_tts_data_pdftotext_bin,
        std::filesystem::perms::owner_exec | std::filesystem::perms::group_exec | std::filesystem::perms::others_exec,
        std::filesystem::perm_options::add);
    setenv("VOICE_SERVER", "http://127.0.0.1:1", 1);
    const std::string tts_data_uri_body = R"({"input":"data:audio/wav;base64,UklGRi1XQVZF","voice":"amy","language":"en"})";
    raw = "POST /v1/audio/speech HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(tts_data_uri_body.size()) + "\r\n\r\n" + tts_data_uri_body;
    // This legacy server fixture has no local STT/TTS factory. Data-URI audio
    // now requires those local services; source-backed factory tests cover it.
    expect(request(raw).find("500 Internal Server Error") != std::string::npos, "data-URI audio requires local services");

    setenv("PDFTOTEXT_BIN", fake_tts_data_pdftotext_bin.c_str(), 1);
    setenv("VOICE_SERVER", "http://127.0.0.1:18157", 1);
    std::thread fake_pdf_data_uri_voice(
        fake_voice_forward_once,
        18157,
        "POST /v1/audio/speech HTTP/1.1",
        "Content-Type: application/json",
        "\"input\":\"pdf extracted text\"",
        "data:application/pdf",
        "audio/wav",
        "RIFFPDFDATAURI",
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string pdf_data_uri_body = "{\"input\":\"data:application/pdf;base64,JVBERi0xLjQ=\",\"voice\":\"amy\",\"language\":\"en\",\"user\":\"../../bad\"}";
    raw = "POST /v1/audio/speech HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(pdf_data_uri_body.size()) + "\r\n\r\n" + pdf_data_uri_body;
    auto pdf_data_uri_tts = request(raw);
    fake_pdf_data_uri_voice.join();
    expect(pdf_data_uri_tts.find("200 OK") != std::string::npos, "tts data uri pdf status");
    expect(pdf_data_uri_tts.find("\"http://localhost:8091/outputs/") != std::string::npos, "tts data uri pdf output URL");
    const std::string pdf_data_uri_response_body = response_body(pdf_data_uri_tts);
    const size_t pdf_data_uri_output = pdf_data_uri_response_body.find("/outputs/");
    const size_t pdf_data_uri_end = pdf_data_uri_response_body.find('"', pdf_data_uri_output);
    expect(pdf_data_uri_output != std::string::npos && pdf_data_uri_end != std::string::npos, "tts data uri pdf output path parse");
    const std::string pdf_data_uri_path = pdf_data_uri_response_body.substr(pdf_data_uri_output, pdf_data_uri_end - pdf_data_uri_output);
    auto pdf_data_uri_file = request("GET " + pdf_data_uri_path + " HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(pdf_data_uri_file.find("200 OK") != std::string::npos, "tts data uri pdf file status");
    expect(pdf_data_uri_file.find("RIFFPDFDATAURI") != std::string::npos, "tts data uri pdf saved forwarded wav");
    if (had_tts_data_voice_server) { setenv("VOICE_SERVER", saved_tts_data_voice_server.c_str(), 1); } else { unsetenv("VOICE_SERVER"); }
    if (had_tts_data_pdftotext) { setenv("PDFTOTEXT_BIN", saved_tts_data_pdftotext.c_str(), 1); } else { unsetenv("PDFTOTEXT_BIN"); }

    setenv("VOICE_SERVER", "http://127.0.0.1:18104", 1);
    std::thread fake_voice_stream(
        fake_voice_forward_once,
        18104,
        "POST /v1/audio/speech/stream HTTP/1.1",
        "Content-Type: application/json",
        "\"voice\": \"amy\"",
        "\"model\"",
        "application/octet-stream",
        "PCM-STREAM",
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    raw = "POST /v1/audio/speech/stream HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(tts_body.size()) + "\r\n\r\n" + tts_body;
    auto remote_tts_stream = request(raw);
    fake_voice_stream.join();
    expect(remote_tts_stream.find("200 OK") != std::string::npos, "remote tts stream status");
    expect(remote_tts_stream.find("Content-Type: application/octet-stream") != std::string::npos, "remote tts stream content type");
    expect(remote_tts_stream.find("Transfer-Encoding: chunked") != std::string::npos, "remote tts stream chunked transfer");
    expect(remote_tts_stream.find("Content-Length:") == std::string::npos, "remote tts stream omits content length");
    expect(remote_tts_stream.find("X-Audio-Format: pcm") != std::string::npos, "remote tts stream audio format header");
    expect(remote_tts_stream.find("X-Sample-Rate: 24000") != std::string::npos, "remote tts stream sample rate header");
    expect(remote_tts_stream.find("X-Bits-Per-Sample: 16") != std::string::npos, "remote tts stream bits header");
    expect(remote_tts_stream.find("X-Channels: 1") != std::string::npos, "remote tts stream channels header");
    expect(remote_tts_stream.find("PCM-STREAM") != std::string::npos, "remote tts stream body");
    unsetenv("VOICE_SERVER");

    setenv("VOICE_SERVER", "http://127.0.0.1:18136", 1);
    std::thread fake_voice_delayed_stream(
        fake_voice_delayed_stream_once,
        18136,
        "POST /v1/audio/speech/stream HTTP/1.1",
        "Content-Type: application/json",
        "\"voice\": \"amy\"",
        "\"model\"",
        "PCM-EARLY",
        "PCM-LATE",
        800,
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    raw = "POST /v1/audio/speech/stream HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(tts_body.size()) + "\r\n\r\n" + tts_body;
    long long voice_first_stream_ms = 0;
    auto remote_tts_incremental_stream = request_observe_first_chunk(
        raw,
        "PCM-EARLY",
        &voice_first_stream_ms
    );
    fake_voice_delayed_stream.join();
    expect(remote_tts_incremental_stream.find("200 OK") != std::string::npos, "remote tts incremental stream status");
    expect(remote_tts_incremental_stream.find("Content-Type: application/octet-stream") != std::string::npos, "remote tts incremental stream content type");
    expect(remote_tts_incremental_stream.find("Transfer-Encoding: chunked") != std::string::npos, "remote tts incremental stream chunked transfer");
    expect(remote_tts_incremental_stream.find("X-Audio-Format: pcm") != std::string::npos, "remote tts incremental stream audio format header");
    expect(remote_tts_incremental_stream.find("PCM-EARLY") != std::string::npos, "remote tts incremental stream early chunk");
    expect(remote_tts_incremental_stream.find("PCM-LATE") != std::string::npos, "remote tts incremental stream late chunk");
    expect(voice_first_stream_ms < 600, "remote tts stream emitted first chunk before upstream completed");
    unsetenv("VOICE_SERVER");

    setenv("VOICE_SERVER", "http://127.0.0.1:18137", 1);
    std::thread fake_voice_resource_stream(
        fake_voice_delayed_stream_once,
        18137,
        "POST /v1/audio/speech/stream HTTP/1.1",
        "Content-Type: application/json",
        "\"voice\": \"amy\"",
        "\"model\"",
        "PCM-RESOURCE",
        "PCM-DONE",
        800,
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    raw = "POST /v1/audio/speech/stream HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(tts_body.size()) + "\r\n\r\n" + tts_body;
    std::string remote_tts_resource_response;
    std::thread remote_tts_resource_request([&]() {
        remote_tts_resource_response = request(raw);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    auto resources_during_tts = request("GET /v1/resources HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(resources_during_tts.find("200 OK") != std::string::npos, "resources during forwarded tts status");
    expect(resources_during_tts.find("\"loaded_models\":{}") != std::string::npos, "forwarded tts does not load local models");
    expect(resources_during_tts.find("\"tts\":{\"capacity\":") == std::string::npos, "forwarded tts has no local capability slot");
    remote_tts_resource_request.join();
    fake_voice_resource_stream.join();
    expect(remote_tts_resource_response.find("200 OK") != std::string::npos, "remote tts resource stream status");
    unsetenv("VOICE_SERVER");

    setenv("VOICE_SERVER", "http://127.0.0.1:18105", 1);
    setenv("VOICE_SERVER_API_KEY", "voice-secret", 1);
    std::thread fake_voice_stt(
        fake_voice_forward_once,
        18105,
        "POST /v1/audio/transcriptions HTTP/1.1",
        "Content-Type: multipart/form-data; boundary=ezlocalai-native-voice-boundary",
        "VOICE-AUDIO",
        "enable_diarization",
        "application/json",
        "{\"text\":\"remote transcript\"}",
        "Authorization: Bearer voice-secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string remote_stt_boundary = "native-stt-boundary";
    const std::string remote_stt_body =
        "--" + remote_stt_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "VOICE-AUDIO\r\n"
        "--" + remote_stt_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"model\"\r\n\r\n"
        "large-v3\r\n"
        "--" + remote_stt_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"response_format\"\r\n\r\n"
        "json\r\n"
        "--" + remote_stt_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"enable_diarization\"\r\n\r\n"
        "true\r\n"
        "--" + remote_stt_boundary + "--\r\n";
    raw = "POST /v1/audio/transcriptions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + remote_stt_boundary + "\r\n"
        "Content-Length: " + std::to_string(remote_stt_body.size()) + "\r\n\r\n" + remote_stt_body;
    auto remote_stt = request(raw);
    fake_voice_stt.join();
    expect(remote_stt.find("200 OK") != std::string::npos, "remote stt status");
    expect(remote_stt.find("\"text\":\"remote transcript\"") != std::string::npos, "remote stt body");
    unsetenv("VOICE_SERVER");

    setenv("VOICE_SERVER", "http://127.0.0.1:18127", 1);
    std::thread fake_voice_stt_text(
        fake_voice_forward_once,
        18127,
        "POST /v1/audio/transcriptions HTTP/1.1",
        "Content-Type: multipart/form-data; boundary=ezlocalai-native-voice-boundary",
        "response_format",
        "",
        "application/json",
        "{\"text\":\"remote transcript text\"}",
        "Authorization: Bearer voice-secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string remote_stt_text_body =
        "--" + remote_stt_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "VOICE-AUDIO\r\n"
        "--" + remote_stt_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"response_format\"\r\n\r\n"
        "text\r\n"
        "--" + remote_stt_boundary + "--\r\n";
    raw = "POST /v1/audio/transcriptions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + remote_stt_boundary + "\r\n"
        "Content-Length: " + std::to_string(remote_stt_text_body.size()) + "\r\n\r\n" + remote_stt_text_body;
    auto remote_stt_text = request(raw);
    fake_voice_stt_text.join();
    expect(remote_stt_text.find("200 OK") != std::string::npos, "remote stt text status");
    expect(remote_stt_text.find("Content-Type: text/plain") != std::string::npos, "remote stt text content type");
    expect(remote_stt_text.find("\r\n\r\nremote transcript text") != std::string::npos, "remote stt text body");
    expect(remote_stt_text.find("{\"text\"") == std::string::npos, "remote stt text unwraps json");
    unsetenv("VOICE_SERVER");
    unsetenv("VOICE_SERVER_API_KEY");

    setenv("FALLBACK_SERVER", "http://127.0.0.1:18106", 1);
    setenv("FALLBACK_API_KEY", "fallback-secret", 1);
    setenv("FALLBACK_MEMORY_THRESHOLD", "999999", 1);
    std::thread fake_fallback_tts(
        fake_fallback_forward_once,
        18106,
        "POST /v1/audio/speech HTTP/1.1",
        "Content-Type: application/json",
        "\"input\": \"speak\"",
        "\"model\"",
        "audio/wav",
        "RIFFFALLBACK",
        "Authorization: Bearer fallback-secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    raw = "POST /v1/audio/speech HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(tts_body.size()) + "\r\n\r\n" + tts_body;
    auto fallback_tts = request(raw);
    fake_fallback_tts.join();
    expect(fallback_tts.find("200 OK") != std::string::npos, "fallback tts status");
    expect(fallback_tts.find("Content-Type: audio/wav") != std::string::npos, "fallback tts content type");
    expect(fallback_tts.find("RIFFFALLBACK") != std::string::npos, "fallback tts body");
    unsetenv("FALLBACK_SERVER");
    unsetenv("FALLBACK_API_KEY");
    unsetenv("FALLBACK_MEMORY_THRESHOLD");

    setenv("FALLBACK_SERVER", "http://127.0.0.1:18107", 1);
    setenv("FALLBACK_MEMORY_THRESHOLD", "999999", 1);
    std::thread fake_fallback_stt(
        fake_fallback_forward_once,
        18107,
        "POST /v1/audio/transcriptions HTTP/1.1",
        "Content-Type: multipart/form-data; boundary=ezlocalai-native-voice-boundary",
        "VOICE-AUDIO",
        "enable_diarization",
        "application/json",
        "{\"text\":\"fallback transcript\"}",
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    raw = "POST /v1/audio/transcriptions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + remote_stt_boundary + "\r\n"
        "Content-Length: " + std::to_string(remote_stt_body.size()) + "\r\n\r\n" + remote_stt_body;
    auto fallback_stt = request(raw);
    fake_fallback_stt.join();
    expect(fallback_stt.find("200 OK") != std::string::npos, "fallback stt status");
    expect(fallback_stt.find("\"text\":\"fallback transcript\"") != std::string::npos, "fallback stt body");

    setenv("FALLBACK_SERVER", "http://127.0.0.1:18128", 1);
    std::thread fake_fallback_stt_text(
        fake_fallback_forward_once,
        18128,
        "POST /v1/audio/transcriptions HTTP/1.1",
        "Content-Type: multipart/form-data; boundary=ezlocalai-native-voice-boundary",
        "response_format",
        "",
        "application/json",
        "{\"text\":\"fallback transcript text\"}",
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    raw = "POST /v1/audio/transcriptions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + remote_stt_boundary + "\r\n"
        "Content-Length: " + std::to_string(remote_stt_text_body.size()) + "\r\n\r\n" + remote_stt_text_body;
    auto fallback_stt_text = request(raw);
    fake_fallback_stt_text.join();
    expect(fallback_stt_text.find("200 OK") != std::string::npos, "fallback stt text status");
    expect(fallback_stt_text.find("Content-Type: text/plain") != std::string::npos, "fallback stt text content type");
    expect(fallback_stt_text.find("\r\n\r\nfallback transcript text") != std::string::npos, "fallback stt text body");

    setenv("FALLBACK_SERVER", "http://127.0.0.1:18138", 1);
    std::thread fake_fallback_stt_empty_fields(
        fake_fallback_forward_once,
        18138,
        "POST /v1/audio/transcriptions HTTP/1.1",
        "Content-Type: multipart/form-data; boundary=ezlocalai-native-voice-boundary",
        "name=\"language\"",
        "enable_diarization",
        "application/json",
        "{\"text\":\"fallback empty fields\"}",
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string remote_stt_empty_fields_body =
        "--" + remote_stt_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "VOICE-AUDIO\r\n"
        "--" + remote_stt_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"language\"\r\n\r\n"
        "\r\n"
        "--" + remote_stt_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"prompt\"\r\n\r\n"
        "\r\n"
        "--" + remote_stt_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"response_format\"\r\n\r\n"
        "json\r\n"
        "--" + remote_stt_boundary + "--\r\n";
    raw = "POST /v1/audio/transcriptions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + remote_stt_boundary + "\r\n"
        "Content-Length: " + std::to_string(remote_stt_empty_fields_body.size()) + "\r\n\r\n" + remote_stt_empty_fields_body;
    auto fallback_stt_empty_fields = request(raw);
    fake_fallback_stt_empty_fields.join();
    expect(fallback_stt_empty_fields.find("200 OK") != std::string::npos, "fallback stt empty fields status");
    expect(fallback_stt_empty_fields.find("\"text\":\"fallback empty fields\"") != std::string::npos, "fallback stt empty fields body");
    unsetenv("FALLBACK_SERVER");
    unsetenv("FALLBACK_MEMORY_THRESHOLD");

    setenv("TTS_ENABLED", "false", 1);
    auto voices_disabled = request("GET /v1/audio/voices HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(voices_disabled.find("404 Not Found") != std::string::npos, "voices disabled status");
    expect(voices_disabled.find("\"detail\":\"Text to speech is disabled.\"") != std::string::npos, "voices disabled body");
    auto voice_upload_disabled_missing_file = request("POST /v1/audio/voices HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Length: 0\r\n\r\n");
    expect(voice_upload_disabled_missing_file.find("422 Unprocessable Entity") != std::string::npos, "voice upload disabled missing file validates first");
    expect(voice_upload_disabled_missing_file.find("\"loc\":[\"body\",\"file\"]") != std::string::npos, "voice upload disabled missing file loc");
    raw = "POST /v1/audio/voices HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + plain_file_field_boundary + "\r\n"
        "Content-Length: " + std::to_string(plain_file_field_body.size()) + "\r\n\r\n" + plain_file_field_body;
    auto voice_upload_plain_file_field = request(raw);
    expect(voice_upload_plain_file_field.find("422 Unprocessable Entity") != std::string::npos, "voice upload plain file form field rejects as upload");
    expect(voice_upload_plain_file_field.find("Expected UploadFile") != std::string::npos, "voice upload plain file form field uploadfile message");
    const std::string disabled_voice_boundary = "disabled-voice-boundary";
    const std::string disabled_voice_upload_body =
        "--" + disabled_voice_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "data\r\n"
        "--" + disabled_voice_boundary + "--\r\n";
    raw = "POST /v1/audio/voices HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + disabled_voice_boundary + "\r\n"
        "Content-Length: " + std::to_string(disabled_voice_upload_body.size()) + "\r\n\r\n" + disabled_voice_upload_body;
    auto voice_upload_disabled_valid_file = request(raw);
    expect(voice_upload_disabled_valid_file.find("404 Not Found") != std::string::npos, "voice upload disabled valid file status");
    expect(voice_upload_disabled_valid_file.find("\"detail\":\"Text to speech is disabled.\"") != std::string::npos, "voice upload disabled valid file body");
    auto tts_disabled = request("POST /v1/audio/speech HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Length: 0\r\n\r\n");
    expect(tts_disabled.find("422 Unprocessable Entity") != std::string::npos, "tts disabled missing body validates first");
    const std::string tts_disabled_valid_body = "{\"input\":\"hello\"}";
    raw = "POST /v1/audio/speech HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(tts_disabled_valid_body.size()) + "\r\n\r\n" + tts_disabled_valid_body;
    auto tts_disabled_valid = request(raw);
    expect(tts_disabled_valid.find("404 Not Found") != std::string::npos, "tts disabled valid body status");
    expect(tts_disabled_valid.find("\"detail\":\"Text to speech is disabled.\"") != std::string::npos, "tts disabled valid body");
    auto tts_ws_disabled = request("GET /v1/audio/speech/ws HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n");
    expect(tts_ws_disabled.find("101 Switching Protocols") != std::string::npos, "tts disabled websocket handshake");
    expect(has_websocket_close_code(tts_ws_disabled, 1008), "tts disabled websocket close code");
    expect(tts_ws_disabled.find("TTS disabled") != std::string::npos, "tts disabled websocket close reason");
    unsetenv("TTS_ENABLED");
    const std::string tts_missing_input_body = "{}";
    raw = "POST /v1/audio/speech HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(tts_missing_input_body.size()) + "\r\n\r\n" + tts_missing_input_body;
    auto tts_missing_input = request(raw);
    expect(tts_missing_input.find("422 Unprocessable Entity") != std::string::npos, "tts missing input status");
    expect(tts_missing_input.find("\"loc\":[\"body\",\"input\"]") != std::string::npos, "tts missing input loc");
    const std::string tts_array_body = "[]";
    raw = "POST /v1/audio/speech HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(tts_array_body.size()) + "\r\n\r\n" + tts_array_body;
    auto tts_array = request(raw);
    expect(tts_array.find("422 Unprocessable Entity") != std::string::npos, "tts array status");
    expect(tts_array.find("\"type\":\"model_attributes_type\"") != std::string::npos, "tts array type");
    const std::string tts_bad_fields_body = "{\"input\":123,\"voice\":123}";
    raw = "POST /v1/audio/speech/stream HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(tts_bad_fields_body.size()) + "\r\n\r\n" + tts_bad_fields_body;
    auto tts_bad_fields = request(raw);
    expect(tts_bad_fields.find("422 Unprocessable Entity") != std::string::npos, "tts bad fields status");
    expect(tts_bad_fields.find("\"loc\":[\"body\",\"input\"]") != std::string::npos, "tts bad input loc");
    expect(tts_bad_fields.find("\"loc\":[\"body\",\"voice\"]") != std::string::npos, "tts bad voice loc");

    // The source WebSocket session always uses local synthesis. An empty
    // session requires no model and still emits its complete PCM envelope.
    std::string tts_ws_handshake;
    int tts_ws_fd = open_websocket("GET /v1/audio/speech/ws HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n", &tts_ws_handshake);
    expect(tts_ws_handshake.find("101 Switching Protocols") != std::string::npos, "tts websocket handshake");
    expect(tts_ws_handshake.find("Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=") != std::string::npos, "tts websocket accept");
    send_websocket_text(tts_ws_fd, "{\"done\":true}");
    std::string tts_ws_frame_buffer;
    expect(recv_websocket_payload(tts_ws_fd, &tts_ws_frame_buffer) == std::string("\xc0\x5d\0\0\x10\0\x01\0",8), "tts websocket sends session header");
    expect(recv_websocket_payload(tts_ws_fd, &tts_ws_frame_buffer) == std::string(4,'\0'), "tts websocket sends session terminator");
    expect(recv_websocket_payload(tts_ws_fd, &tts_ws_frame_buffer).empty(), "tts websocket sends final empty frame");
    expect(recv_websocket_payload(tts_ws_fd, &tts_ws_frame_buffer) == std::string("\x03\xe8",2), "tts websocket closes normally");
    close(tts_ws_fd);

    const std::string image_missing_prompt_body = "{}";
    raw = "POST /v1/images/generations HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(image_missing_prompt_body.size()) + "\r\n\r\n" + image_missing_prompt_body;
    auto image_missing_prompt = request(raw);
    expect(image_missing_prompt.find("422 Unprocessable Entity") != std::string::npos, "image missing prompt status");
    expect(image_missing_prompt.find("\"loc\":[\"body\",\"prompt\"]") != std::string::npos, "image missing prompt loc");
    const std::string image_bad_n_body = "{\"prompt\":\"paint\",\"n\":\"bad\"}";
    raw = "POST /v1/images/generations HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(image_bad_n_body.size()) + "\r\n\r\n" + image_bad_n_body;
    auto image_bad_n = request(raw);
    expect(image_bad_n.find("422 Unprocessable Entity") != std::string::npos, "image bad n status");
    expect(image_bad_n.find("\"loc\":[\"body\",\"n\"]") != std::string::npos, "image bad n loc");
    const std::string image_edit_missing_image_body = "{\"prompt\":\"edit\"}";
    raw = "POST /v1/images/edits HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(image_edit_missing_image_body.size()) + "\r\n\r\n" + image_edit_missing_image_body;
    auto image_edit_missing_image = request(raw);
    expect(image_edit_missing_image.find("422 Unprocessable Entity") != std::string::npos, "image edit missing image status");
    expect(image_edit_missing_image.find("\"loc\":[\"body\"]") != std::string::npos
        && image_edit_missing_image.find("image or images is required") != std::string::npos, "image edit missing references validator");
    auto video_missing_body = request("POST /v1/videos/generations HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Type: application/json\r\nContent-Length: 0\r\n\r\n");
    expect(video_missing_body.find("422 Unprocessable Entity") != std::string::npos, "video missing body status");
    expect(video_missing_body.find("\"loc\":[\"body\"]") != std::string::npos, "video missing body loc");
    const std::string video_bad_guidance_body = "{\"guidance_scale\":\"bad\"}";
    raw = "POST /v1/videos/generations HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(video_bad_guidance_body.size()) + "\r\n\r\n" + video_bad_guidance_body;
    auto video_bad_guidance = request(raw);
    expect(video_bad_guidance.find("422 Unprocessable Entity") != std::string::npos, "video bad guidance status");
    expect(video_bad_guidance.find("\"loc\":[\"body\",\"guidance_scale\"]") != std::string::npos, "video bad guidance loc");
    const std::string video_underscore_guidance_body = "{\"guidance_scale\":\"1_000.0\"}";
    raw = "POST /v1/videos/generations HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(video_underscore_guidance_body.size()) + "\r\n\r\n" + video_underscore_guidance_body;
    auto video_underscore_guidance = request(raw);
    expect(video_underscore_guidance.find("200 OK") != std::string::npos, "video underscore guidance validates");
    const std::string video_bad_conditions_body = "{\"conditions\":[{},1,{\"image\":123,\"index\":\"bad\",\"strength\":\"bad\"}]}";
    raw = "POST /v1/videos/generations HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(video_bad_conditions_body.size()) + "\r\n\r\n" + video_bad_conditions_body;
    auto video_bad_conditions = request(raw);
    expect(video_bad_conditions.find("422 Unprocessable Entity") != std::string::npos, "video bad conditions status");
    expect(video_bad_conditions.find("\"loc\":[\"body\",\"conditions\",0,\"image\"]") != std::string::npos, "video bad conditions missing image loc");
    expect(video_bad_conditions.find("\"loc\":[\"body\",\"conditions\",1]") != std::string::npos, "video bad conditions item loc");
    expect(video_bad_conditions.find("\"loc\":[\"body\",\"conditions\",2,\"index\"]") != std::string::npos, "video bad conditions index loc");
    expect(video_bad_conditions.find("\"loc\":[\"body\",\"conditions\",2,\"strength\"]") != std::string::npos, "video bad conditions strength loc");

    setenv("IMAGE_SERVER", "http://127.0.0.1:18108", 1);
    setenv("IMAGE_SERVER_API_KEY", "image-secret", 1);
    std::thread fake_image_generation(
        fake_image_server_forward_once,
        18108,
        "POST /v1/images/generations HTTP/1.1",
        "\"prompt\":\"paint\"",
        "\"model\"",
        "{\"created\":123,\"data\":[{\"url\":\"/outputs/image.png\"}]}",
        "Authorization: Bearer image-secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string image_body = "{\"prompt\":\"paint\",\"model\":\"ignored\",\"quality\":\"hd\",\"style\":\"natural\",\"response_format\":\"url\",\"size\":\"512x512\",\"n\":2}";
    raw = "POST /v1/images/generations HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(image_body.size()) + "\r\n\r\n" + image_body;
    auto remote_image = request(raw);
    fake_image_generation.join();
    expect(remote_image.find("200 OK") != std::string::npos, "remote image status");
    expect(remote_image.find("\"url\":\"/outputs/image.png\"") != std::string::npos, "remote image body");
    std::thread fake_cached_image_generation(
        fake_expect_worker_once,
        18108,
        "POST /v1/images/generations HTTP/1.1",
        "Content-Type: application/json",
        "\"prompt\":\"cache paint\"",
        "{\"created\":223,\"data\":[{\"url\":\"/outputs/cached-image.png\"}]}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string cached_image_body = "{\"prompt\":\"cache paint\",\"model\":\"ignored\",\"response_format\":\"url\",\"size\":\"512x512\",\"n\":1}";
    raw = "POST /v1/images/generations HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(cached_image_body.size()) + "\r\n\r\n" + cached_image_body;
    auto cached_remote_image = request(raw);
    fake_cached_image_generation.join();
    expect(cached_remote_image.find("200 OK") != std::string::npos, "remote cached image status");
    expect(cached_remote_image.find("\"url\":\"/outputs/cached-image.png\"") != std::string::npos, "image server availability cache skips second probe");
    unsetenv("IMAGE_SERVER");
    unsetenv("IMAGE_SERVER_API_KEY");

    setenv("IMAGE_SERVER", "http://127.0.0.1:18153", 1);
    setenv("IMAGE_SERVER_API_KEY", "image-secret", 1);
    setenv("IMG_MODEL", "resource-image-model", 1);
    std::thread fake_image_resource_generation(
        fake_image_server_delayed_forward_once,
        18153,
        "POST /v1/images/generations HTTP/1.1",
        "\"prompt\":\"resource paint\"",
        "\"model\"",
        "{\"created\":323,\"data\":[{\"url\":\"/outputs/resource-image.png\"}]}",
        "Authorization: Bearer image-secret",
        800
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string image_resource_body = "{\"prompt\":\"resource paint\",\"model\":\"ignored\",\"response_format\":\"url\"}";
    raw = "POST /v1/images/generations HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(image_resource_body.size()) + "\r\n\r\n" + image_resource_body;
    std::string image_resource_response;
    std::thread image_resource_request([&]() {
        image_resource_response = request(raw);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    auto resources_during_image = request("GET /v1/resources HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(resources_during_image.find("200 OK") != std::string::npos, "resources during forwarded image status");
    expect(resources_during_image.find("\"loaded_models\":{}") != std::string::npos, "forwarded image does not load local models");
    expect(resources_during_image.find("\"image\":{") == std::string::npos, "forwarded image has no local capability slot");
    image_resource_request.join();
    fake_image_resource_generation.join();
    expect(image_resource_response.find("200 OK") != std::string::npos, "remote image resource status");
    unsetenv("IMAGE_SERVER");
    unsetenv("IMAGE_SERVER_API_KEY");
    unsetenv("IMG_MODEL");

    setenv("IMAGE_SERVER", "http://127.0.0.1:18130", 1);
    std::thread fake_image_null_optionals(
        fake_image_server_forward_once,
        18130,
        "POST /v1/images/generations HTTP/1.1",
        "\"response_format\":null,\"size\":null,\"n\":null",
        "\"response_format\":\"url\"",
        "{\"created\":131,\"data\":[{\"url\":\"/outputs/image-nulls.png\"}]}",
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string image_null_optionals_body = "{\"prompt\":\"nullable image\",\"response_format\":null,\"size\":null,\"n\":null}";
    raw = "POST /v1/images/generations HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(image_null_optionals_body.size()) + "\r\n\r\n" + image_null_optionals_body;
    auto remote_image_null_optionals = request(raw);
    fake_image_null_optionals.join();
    expect(remote_image_null_optionals.find("200 OK") != std::string::npos, "remote image null optionals status");
    expect(remote_image_null_optionals.find("\"url\":\"/outputs/image-nulls.png\"") != std::string::npos, "remote image null optionals body");
    unsetenv("IMAGE_SERVER");

    setenv("IMAGE_SERVER", "http://127.0.0.1:18109", 1);
    std::thread fake_image_edit(
        fake_image_server_forward_once,
        18109,
        "POST /v1/images/edits HTTP/1.1",
        "\"image\":\"data:image/png;base64,AAA\"",
        "\"model\"",
        "{\"created\":124,\"data\":[{\"url\":\"/outputs/edit.png\"}]}",
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string remote_edit_body = "{\"prompt\":\"edit\",\"image\":\"data:image/png;base64,AAA\",\"model\":\"ignored\",\"response_format\":\"url\",\"size\":\"1024x1024\",\"n\":1}";
    raw = "POST /v1/images/edits HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(remote_edit_body.size()) + "\r\n\r\n" + remote_edit_body;
    auto remote_edit = request(raw);
    fake_image_edit.join();
    expect(remote_edit.find("200 OK") != std::string::npos, "remote image edit status");
    expect(remote_edit.find("\"url\":\"/outputs/edit.png\"") != std::string::npos, "remote image edit body");
    unsetenv("IMAGE_SERVER");

    setenv("IMAGE_SERVER", "http://127.0.0.1:18158", 1);
    std::thread fake_multipart_image_edit(
        fake_image_server_forward_once,
        18158,
        "POST /v1/images/edits HTTP/1.1",
        "\"image\":\"data:image/png;base64,UE5HREFUQQ==\"",
        "filename=\"edit.png\"",
        "{\"created\":224,\"data\":[{\"url\":\"/outputs/multipart-edit.png\"}]}",
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string image_edit_multipart_boundary = "image-edit-multipart-boundary";
    const std::string image_edit_multipart_body =
        "--" + image_edit_multipart_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"image\"; filename=\"edit.png\"\r\n"
        "Content-Type: image/png\r\n\r\n"
        "PNGDATA\r\n"
        "--" + image_edit_multipart_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"prompt\"\r\n\r\n"
        "multipart edit\r\n"
        "--" + image_edit_multipart_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"response_format\"\r\n\r\n"
        "url\r\n"
        "--" + image_edit_multipart_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"size\"\r\n\r\n"
        "1024x1024\r\n"
        "--" + image_edit_multipart_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"n\"\r\n\r\n"
        "1\r\n"
        "--" + image_edit_multipart_boundary + "--\r\n";
    raw = "POST /v1/images/edits HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + image_edit_multipart_boundary + "\r\n"
        "Content-Length: " + std::to_string(image_edit_multipart_body.size()) + "\r\n\r\n" + image_edit_multipart_body;
    auto remote_multipart_edit = request(raw);
    fake_multipart_image_edit.join();
    expect(remote_multipart_edit.find("200 OK") != std::string::npos, "remote multipart image edit status");
    expect(remote_multipart_edit.find("\"url\":\"/outputs/multipart-edit.png\"") != std::string::npos, "remote multipart image edit body");
    unsetenv("IMAGE_SERVER");

    setenv("IMAGE_SERVER", "http://127.0.0.1:18131", 1);
    std::thread fake_image_edit_additional_images(
        fake_image_server_forward_once,
        18131,
        "POST /v1/images/edits HTTP/1.1",
        "\"images\":[\"ref-one\",\"ref-two\"],\"strength\":0.3",
        "\"image\"",
        "{\"created\":132,\"data\":[{\"url\":\"/outputs/additional-edit.png\"}]}",
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string remote_edit_additional_images_body = "{\"prompt\":\"additional images\",\"images\":[\"ref-one\",\"ref-two\"],\"strength\":0.3,\"response_format\":\"url\",\"size\":\"1024x1024\",\"n\":1}";
    raw = "POST /v1/images/edits HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(remote_edit_additional_images_body.size()) + "\r\n\r\n" + remote_edit_additional_images_body;
    auto remote_edit_additional_images = request(raw);
    fake_image_edit_additional_images.join();
    expect(remote_edit_additional_images.find("200 OK") != std::string::npos, "remote image edit additional images status");
    expect(remote_edit_additional_images.find("\"url\":\"/outputs/additional-edit.png\"") != std::string::npos, "remote image edit additional images body");
    unsetenv("IMAGE_SERVER");

    setenv("IMAGE_SERVER", "http://127.0.0.1:18110", 1);
    std::thread fake_video_generation(
        fake_image_server_forward_once,
        18110,
        "POST /v1/videos/generations HTTP/1.1",
        "\"num_frames\":16",
        "\"model\"",
        "{\"created\":125,\"data\":[{\"url\":\"/outputs/video.mp4\"}]}",
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string video_body = "{\"prompt\":\"move\",\"model\":\"ignored\",\"response_format\":\"url\",\"size\":\"512x512\",\"num_frames\":16,\"num_inference_steps\":8,\"guidance_scale\":3.5,\"frame_rate\":12}";
    raw = "POST /v1/videos/generations HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(video_body.size()) + "\r\n\r\n" + video_body;
    auto remote_video = request(raw);
    fake_video_generation.join();
    expect(remote_video.find("200 OK") != std::string::npos, "remote video status");
    expect(remote_video.find("\"url\":\"/outputs/video.mp4\"") != std::string::npos, "remote video body");
    unsetenv("IMAGE_SERVER");

    setenv("IMAGE_SERVER", "http://127.0.0.1:18154", 1);
    setenv("VIDEO_MODEL", "resource-video-model", 1);
    std::thread fake_video_resource_generation(
        fake_image_server_delayed_forward_once,
        18154,
        "POST /v1/videos/generations HTTP/1.1",
        "\"num_frames\":16",
        "\"model\"",
        "{\"created\":325,\"data\":[{\"url\":\"/outputs/resource-video.mp4\"}]}",
        "Authorization: Bearer secret",
        800
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    raw = "POST /v1/videos/generations HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(video_body.size()) + "\r\n\r\n" + video_body;
    std::string video_resource_response;
    std::thread video_resource_request([&]() {
        video_resource_response = request(raw);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    auto resources_during_video = request("GET /v1/resources HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(resources_during_video.find("200 OK") != std::string::npos, "resources during forwarded video status");
    expect(resources_during_video.find("\"loaded_models\":{}") != std::string::npos, "forwarded video does not load local models");
    expect(resources_during_video.find("\"video\":{") == std::string::npos, "forwarded video has no local capability slot");
    video_resource_request.join();
    fake_video_resource_generation.join();
    expect(video_resource_response.find("200 OK") != std::string::npos, "remote video resource status");
    unsetenv("IMAGE_SERVER");
    unsetenv("VIDEO_MODEL");

    setenv("IMAGE_SERVER", "http://127.0.0.1:18132", 1);
    std::thread fake_video_null_optionals(
        fake_image_server_forward_once,
        18132,
        "POST /v1/videos/generations HTTP/1.1",
        "\"prompt\":null,\"response_format\":null,\"size\":null,\"num_frames\":null",
        "\"num_frames\":121",
        "{\"created\":133,\"data\":[{\"url\":\"/outputs/video-nulls.mp4\"}]}",
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string video_null_optionals_body = "{\"prompt\":null,\"response_format\":null,\"size\":null,\"num_frames\":null,\"num_inference_steps\":null,\"guidance_scale\":null,\"frame_rate\":null,\"image\":null}";
    raw = "POST /v1/videos/generations HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(video_null_optionals_body.size()) + "\r\n\r\n" + video_null_optionals_body;
    auto remote_video_null_optionals = request(raw);
    fake_video_null_optionals.join();
    expect(remote_video_null_optionals.find("200 OK") != std::string::npos, "remote video null optionals status");
    expect(remote_video_null_optionals.find("\"url\":\"/outputs/video-nulls.mp4\"") != std::string::npos, "remote video null optionals body");
    unsetenv("IMAGE_SERVER");

    setenv("IMAGE_SERVER", "http://127.0.0.1:18129", 1);
    std::thread fake_video_conditions(
        fake_image_server_forward_once,
        18129,
        "POST /v1/videos/generations HTTP/1.1",
        "\"guidance_scale\":3.5",
        "\"conditions\"",
        "{\"created\":130,\"data\":[{\"url\":\"/outputs/video-conditions.mp4\"}]}",
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string video_conditions_body = "{\"prompt\":\"move\",\"response_format\":\"url\",\"conditions\":[{\"image\":\"frame\",\"index\":-1,\"strength\":0.25,\"ignored_extra\":true}]}";
    raw = "POST /v1/videos/generations HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(video_conditions_body.size()) + "\r\n\r\n" + video_conditions_body;
    auto remote_video_conditions = request(raw);
    fake_video_conditions.join();
    expect(remote_video_conditions.find("200 OK") != std::string::npos, "remote video omits conditions status");
    expect(remote_video_conditions.find("\"url\":\"/outputs/video-conditions.mp4\"") != std::string::npos, "remote video omits conditions body");
    unsetenv("IMAGE_SERVER");

    setenv("IMG_MODEL", "", 1);
    setenv("FALLBACK_SERVER", "http://127.0.0.1:18111", 1);
    setenv("FALLBACK_API_KEY", "fallback-secret", 1);
    std::thread fake_fallback_image(
        fake_fallback_forward_once,
        18111,
        "POST /v1/images/generations HTTP/1.1",
        "Content-Type: application/json",
        "\"size\":\"512x512\",\"n\":1",
        "\"model\"",
        "application/json",
        "{\"created\":126,\"data\":[{\"url\":\"/outputs/fallback-image.png\"}]}",
        "Authorization: Bearer fallback-secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    raw = "POST /v1/images/generations HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(image_body.size()) + "\r\n\r\n" + image_body;
    auto fallback_image = request(raw);
    fake_fallback_image.join();
    expect(fallback_image.find("200 OK") != std::string::npos, "fallback image status");
    expect(fallback_image.find("\"url\":\"/outputs/fallback-image.png\"") != std::string::npos, "fallback image body");
    unsetenv("FALLBACK_SERVER");
    unsetenv("FALLBACK_API_KEY");
    unsetenv("IMG_MODEL");

    setenv("IMG_MODEL", "", 1);
    setenv("FALLBACK_SERVER", "http://127.0.0.1:18121", 1);
    std::thread fake_fallback_image_default_size(
        fake_fallback_forward_once,
        18121,
        "POST /v1/images/generations HTTP/1.1",
        "Content-Type: application/json",
        "\"size\":\"1024x1024\",\"n\":1",
        "\"model\"",
        "application/json",
        "{\"created\":129,\"data\":[{\"url\":\"/outputs/fallback-image-default.png\"}]}",
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string image_default_body = "{\"prompt\":\"default size\"}";
    raw = "POST /v1/images/generations HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(image_default_body.size()) + "\r\n\r\n" + image_default_body;
    auto fallback_image_default = request(raw);
    fake_fallback_image_default_size.join();
    expect(fallback_image_default.find("200 OK") != std::string::npos, "fallback image default size status");
    expect(fallback_image_default.find("\"url\":\"/outputs/fallback-image-default.png\"") != std::string::npos, "fallback image default size body");
    unsetenv("FALLBACK_SERVER");
    unsetenv("IMG_MODEL");

    setenv("IMG_MODEL", "", 1);
    setenv("FALLBACK_SERVER", "http://127.0.0.1:18133", 1);
    std::thread fake_fallback_image_null_optionals(
        fake_fallback_forward_once,
        18133,
        "POST /v1/images/generations HTTP/1.1",
        "Content-Type: application/json",
        "\"response_format\":null,\"size\":null",
        "\"response_format\":\"url\"",
        "application/json",
        "{\"created\":134,\"data\":[{\"url\":\"/outputs/fallback-image-nulls.png\"}]}",
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    raw = "POST /v1/images/generations HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(image_null_optionals_body.size()) + "\r\n\r\n" + image_null_optionals_body;
    auto fallback_image_null_optionals = request(raw);
    fake_fallback_image_null_optionals.join();
    expect(fallback_image_null_optionals.find("200 OK") != std::string::npos, "fallback image null optionals status");
    expect(fallback_image_null_optionals.find("\"url\":\"/outputs/fallback-image-nulls.png\"") != std::string::npos, "fallback image null optionals body");
    unsetenv("FALLBACK_SERVER");
    unsetenv("IMG_MODEL");

    setenv("IMG_MODEL", "", 1);
    setenv("FALLBACK_SERVER", "http://127.0.0.1:18112", 1);
    std::thread fake_fallback_edit(
        fake_fallback_forward_once,
        18112,
        "POST /v1/images/edits HTTP/1.1",
        "Content-Type: application/json",
        "\"image\":\"data:image/png;base64,AAA\"",
        "\"model\"",
        "application/json",
        "{\"created\":127,\"data\":[{\"url\":\"/outputs/fallback-edit.png\"}]}",
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    raw = "POST /v1/images/edits HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(remote_edit_body.size()) + "\r\n\r\n" + remote_edit_body;
    auto fallback_edit = request(raw);
    fake_fallback_edit.join();
    expect(fallback_edit.find("200 OK") != std::string::npos, "fallback edit status");
    expect(fallback_edit.find("\"url\":\"/outputs/fallback-edit.png\"") != std::string::npos, "fallback edit body");
    unsetenv("FALLBACK_SERVER");
    unsetenv("IMG_MODEL");

    setenv("VIDEO_MODEL", "none", 1);
    setenv("FALLBACK_SERVER", "http://127.0.0.1:18113", 1);
    std::thread fake_fallback_video(
        fake_fallback_forward_once,
        18113,
        "POST /v1/videos/generations HTTP/1.1",
        "Content-Type: application/json",
        "\"num_frames\":16",
        "\"model\"",
        "application/json",
        "{\"created\":128,\"data\":[{\"url\":\"/outputs/fallback-video.mp4\"}]}",
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    raw = "POST /v1/videos/generations HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(video_body.size()) + "\r\n\r\n" + video_body;
    auto fallback_video = request(raw);
    fake_fallback_video.join();
    expect(fallback_video.find("200 OK") != std::string::npos, "fallback video status");
    expect(fallback_video.find("\"url\":\"/outputs/fallback-video.mp4\"") != std::string::npos, "fallback video body");
    unsetenv("FALLBACK_SERVER");
    unsetenv("VIDEO_MODEL");

    setenv("VIDEO_MODEL", "none", 1);
    setenv("FALLBACK_SERVER", "http://127.0.0.1:18134", 1);
    std::thread fake_fallback_video_null_optionals(
        fake_fallback_forward_once,
        18134,
        "POST /v1/videos/generations HTTP/1.1",
        "Content-Type: application/json",
        "\"prompt\":null,\"response_format\":null,\"size\":null,\"num_frames\":null",
        "\"num_frames\":121",
        "application/json",
        "{\"created\":135,\"data\":[{\"url\":\"/outputs/fallback-video-nulls.mp4\"}]}",
        "Authorization: Bearer secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    raw = "POST /v1/videos/generations HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(video_null_optionals_body.size()) + "\r\n\r\n" + video_null_optionals_body;
    auto fallback_video_null_optionals = request(raw);
    fake_fallback_video_null_optionals.join();
    expect(fallback_video_null_optionals.find("200 OK") != std::string::npos, "fallback video null optionals status");
    expect(fallback_video_null_optionals.find("\"url\":\"/outputs/fallback-video-nulls.mp4\"") != std::string::npos, "fallback video null optionals body");
    unsetenv("FALLBACK_SERVER");
    unsetenv("VIDEO_MODEL");

    setenv("IMG_MODEL", "", 1);
    const std::string image_disabled_body = "{\"prompt\":\"paint\"}";
    raw = "POST /v1/images/generations HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(image_disabled_body.size()) + "\r\n\r\n" + image_disabled_body;
    auto image_disabled = request(raw);
    expect(image_disabled.find("200 OK") != std::string::npos, "image empty model status");
    expect(image_disabled.find("https://demofree.sirv.com/nope-not-here.jpg") != std::string::npos, "image empty model fallback body");
    setenv("IMG_MODEL", "local-image-model", 1);
    setenv("IMAGE_ENABLED", "false", 1);
    auto image_switch_disabled = request(raw);
    expect(image_switch_disabled.find("https://demofree.sirv.com/nope-not-here.jpg") != std::string::npos, "image enable switch overrides nonempty model");
    const std::string disabled_edit_body = "{\"prompt\":\"paint\",\"image\":\"test-image\"}";
    auto image_edit_disabled = request("POST /v1/images/edits HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(disabled_edit_body.size()) + "\r\n\r\n" + disabled_edit_body);
    expect(image_edit_disabled.find("503 Service Unavailable") != std::string::npos, "disabled local image editing status");
    expect(image_edit_disabled.find("{\"detail\":\"No image editing service available\"}") != std::string::npos, "disabled local image editing detail");
    unsetenv("IMG_MODEL");
    unsetenv("IMAGE_ENABLED");

    auto video_disabled = request("POST /v1/videos/generations HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Length: 2\r\n\r\n{}");
    expect(video_disabled.find("200 OK") != std::string::npos, "video disabled status");
    expect(video_disabled.find("Video generation not available. Set VIDEO_ENABLED=true and VIDEO_MODEL to enable.") != std::string::npos, "video disabled body");

    const std::string boundary = "native-boundary";
    const std::string upload_body =
        "--" + boundary + "\r\n"
        "Content-Disposition: form-data; name=\"voice\"\r\n\r\n"
        "../../bad:name\r\n"
        "--" + boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "WAVE-DATA\r\n"
        "--" + boundary + "--\r\n";
    raw = "POST /v1/audio/voices HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + boundary + "\r\n"
        "Content-Length: " + std::to_string(upload_body.size()) + "\r\n\r\n" + upload_body;
    auto uploaded_voice = request(raw);
    expect(uploaded_voice.find("200 OK") != std::string::npos, "voice upload status");
    expect(uploaded_voice.find("\"detail\":\"Voice badname has been uploaded.\"") != std::string::npos, "voice upload body");
    expect(std::filesystem::exists("voices/badname.wav"), "voice upload file exists");
    {
        std::ifstream uploaded("voices/badname.wav", std::ios::binary);
        std::string contents((std::istreambuf_iterator<char>(uploaded)), std::istreambuf_iterator<char>());
        expect(contents == "WAVE-DATA", "voice upload file contents");
    }

    auto api_mode_router_health = request("GET /v1/router/health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(api_mode_router_health.find("404 Not Found") != std::string::npos, "api mode hides router health route");
    expect(api_mode_router_health.find("{\"detail\":\"Not Found\"}") != std::string::npos, "api mode router health 404 body");
    auto api_mode_router_register = request("POST /v1/router/register HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}");
    expect(api_mode_router_register.find("404 Not Found") != std::string::npos, "api mode hides router register route");
    auto api_mode_dashboard = request("GET /dashboard HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(api_mode_dashboard.find("404 Not Found") != std::string::npos, "api mode hides router dashboard route");
    auto websocket_missing_key = request("GET /v1/models HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n\r\n");
    expect(websocket_missing_key.find("400 Bad Request") != std::string::npos, "websocket missing key status");
    expect(websocket_missing_key.find("Content-Type: text/plain") != std::string::npos, "websocket missing key content type");
    expect(websocket_missing_key.find("missing Sec-WebSocket-Key header.") != std::string::npos, "websocket missing key body");
    auto websocket_missing_version = request("GET /v1/audio/speech/ws HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n");
    expect(websocket_missing_version.find("400 Bad Request") != std::string::npos, "websocket missing version status");
    expect(websocket_missing_version.find("missing Sec-WebSocket-Version header.") != std::string::npos, "websocket missing version body");
    auto websocket_bad_key = request("GET /v1/audio/speech/ws HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: bad\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n");
    expect(websocket_bad_key.find("400 Bad Request") != std::string::npos, "websocket invalid key status");
    expect(websocket_bad_key.find("invalid Sec-WebSocket-Key header: bad.") != std::string::npos, "websocket invalid key body");
    auto websocket_post = request("POST /v1/audio/speech/ws HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n");
    expect(websocket_post.find("400 Bad Request") != std::string::npos, "websocket post status");
    expect(websocket_post.find("did not receive a valid HTTP request.") != std::string::npos, "websocket post body");
    auto api_mode_router_tunnel_ws = request("GET /v1/router/tunnel?worker_id=home-node HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n");
    expect(api_mode_router_tunnel_ws.find("403 Forbidden") != std::string::npos, "api mode rejects router tunnel websocket route");
    expect(api_mode_router_tunnel_ws.find("Content-Type: text/plain") != std::string::npos, "api mode router tunnel websocket content type");

    setenv("ROUTER_MODE", "true", 1);
    auto router_mode_resources = request("GET /v1/resources HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(router_mode_resources.find("404 Not Found") != std::string::npos, "router mode hides api resources route");
    auto router_mode_queue = request("GET /v1/queue HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(router_mode_queue.find("404 Not Found") != std::string::npos, "router mode hides api queue route");
    auto router_mode_voice_upload = request("POST /v1/audio/voices HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Length: 0\r\n\r\n");
    expect(router_mode_voice_upload.find("405 Method Not Allowed") != std::string::npos, "router mode voices is get-only");
    expect(router_mode_voice_upload.find("Allow: GET") != std::string::npos, "router mode voices allow header");
    auto router_mode_tts_ws = request("GET /v1/audio/speech/ws HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n");
    expect(router_mode_tts_ws.find("403 Forbidden") != std::string::npos, "router mode rejects api tts websocket route");
    auto router_health_empty = request("GET /v1/router/health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(router_health_empty.find("200 OK") != std::string::npos, "router health unauthenticated");
    expect(router_health_empty.find("\"alive_workers\":0") != std::string::npos, "router health no workers");
    expect(router_health_empty.find("\"ttl_seconds\":60.0") != std::string::npos, "router health ttl route default float");
    auto router_health_query = request("GET /v1/router/health?probe=1 HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(router_health_query.find("200 OK") != std::string::npos, "router health query unauthenticated");
    auto router_docs = request("GET / HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(router_docs.find("200 OK") != std::string::npos, "router docs status");
    expect(router_docs.find("Content-Length: 1015") != std::string::npos, "router docs content length");
    expect(router_docs.find("<title>ezlocalai Router - Swagger UI</title>") != std::string::npos, "router docs title");
    auto router_docs_head = request("HEAD / HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(router_docs_head.find("200 OK") != std::string::npos, "router docs head status");
    expect(router_docs_head.find("Content-Length: 1015") != std::string::npos, "router docs head content length");
    expect(router_docs_head.rfind("\r\n\r\n") == router_docs_head.size() - 4, "router docs head omits body");
    auto router_redoc = request("GET /redoc HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(router_redoc.find("200 OK") != std::string::npos, "router redoc status");
    expect(router_redoc.find("Content-Length: 897") != std::string::npos, "router redoc content length");
    expect(router_redoc.find("<title>ezlocalai Router - ReDoc</title>") != std::string::npos, "router redoc title");
    auto router_redoc_head = request("HEAD /redoc HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(router_redoc_head.find("200 OK") != std::string::npos, "router redoc head status");
    expect(router_redoc_head.find("Content-Length: 897") != std::string::npos, "router redoc head content length");
    expect(router_redoc_head.rfind("\r\n\r\n") == router_redoc_head.size() - 4, "router redoc head omits body");
    auto router_openapi = request("GET /openapi.json HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(router_openapi.find("200 OK") != std::string::npos, "router openapi status");
    expect(router_openapi.find("\"title\":\"ezlocalai Router\"") != std::string::npos, "router openapi title");
    expect(router_openapi.find("\"/v1/router/register\"") != std::string::npos, "router openapi route path");
    expect(router_openapi.find("\"/v1/resources\"") == std::string::npos, "router openapi excludes api resources path");
    expect(router_openapi.find("\"/v1/audio/translations\"") == std::string::npos, "router openapi excludes api audio translations path");
    expect(router_openapi.find("\"operationId\":\"router_register_v1_router_register_post\"") != std::string::npos, "router openapi operation id");
    expect(router_openapi.find("\"summary\":\"Router Register\"") != std::string::npos, "router openapi generated summary");
    expect(router_openapi.find("\"components\":{\"schemas\"") != std::string::npos, "router openapi generated schemas");
    expect(router_openapi.find("\"Body_audio_transcriptions_v1_audio_transcriptions_post\"") != std::string::npos, "router openapi multipart schema");
    expect(router_openapi.find("\"additionalProperties\":true,\"title\":\"Payload\"") != std::string::npos, "router openapi dict payload schema");
    expect(router_openapi.find("private network without juggling browser auth") != std::string::npos, "router openapi python docstring");
    auto router_openapi_head = request("HEAD /openapi.json HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(router_openapi_head.find("200 OK") != std::string::npos, "router openapi head status");
    expect(router_openapi_head.rfind("\r\n\r\n") == router_openapi_head.size() - 4, "router openapi head omits body");
    auto router_oauth_redirect = request("GET /docs/oauth2-redirect HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(router_oauth_redirect.find("200 OK") != std::string::npos, "router oauth redirect status");
    expect(router_oauth_redirect.find("Content-Length: 3012") != std::string::npos, "router oauth redirect content length");
    expect(response_body(router_oauth_redirect).find("var redirectUrl = oauth2.redirectUrl;") != std::string::npos, "router oauth redirect body");
    auto router_oauth_head = request("HEAD /docs/oauth2-redirect HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(router_oauth_head.find("200 OK") != std::string::npos, "router oauth head status");
    expect(router_oauth_head.find("Content-Length: 3012") != std::string::npos, "router oauth head content length");
    expect(router_oauth_head.rfind("\r\n\r\n") == router_oauth_head.size() - 4, "router oauth head omits body");
    setenv("ROUTER_WAIT_TIMEOUT", "0", 1);
    const std::string router_tts_false_model_body = "{\"input\":\"say it\",\"model\":false}";
    raw = "POST /v1/audio/speech HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(router_tts_false_model_body.size()) + "\r\n\r\n" + router_tts_false_model_body;
    auto router_tts_false_model = request(raw);
    expect(router_tts_false_model.find("503 Service Unavailable") != std::string::npos, "router tts false model no worker status");
    expect(router_tts_false_model.find("No worker available for capability='tts'") != std::string::npos, "router tts false model omits model detail");
    expect(router_tts_false_model.find("model='false'") == std::string::npos, "router tts false model no false string");

    auto router_register_missing_auth = request("POST /v1/router/register HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}");
    expect(router_register_missing_auth.find("401 Unauthorized") != std::string::npos, "router register missing auth status");
    expect(router_register_missing_auth.find("\"detail\":\"Missing Authorization header\"") != std::string::npos, "router register missing auth body");
    auto router_register_invalid_auth = request("POST /v1/router/register HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer wrong\r\nContent-Length: 2\r\n\r\n{}");
    expect(router_register_invalid_auth.find("401 Unauthorized") != std::string::npos, "router register invalid auth status");
    expect(router_register_invalid_auth.find("\"detail\":\"Invalid API key\"") != std::string::npos, "router register invalid auth body");
    auto router_register_empty_body = request("POST /v1/router/register HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Length: 0\r\n\r\n");
    expect(router_register_empty_body.find("422 Unprocessable Entity") != std::string::npos, "router register empty body status");
    expect(router_register_empty_body.find("\"type\":\"missing\"") != std::string::npos, "router register empty body validation type");
    expect(router_register_empty_body.find("\"msg\":\"Field required\"") != std::string::npos, "router register empty body validation message");
    auto router_register_bad_json = request("POST /v1/router/register HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Length: 1\r\n\r\n{");
    expect(router_register_bad_json.find("422 Unprocessable Entity") != std::string::npos, "router register bad json status");
    expect(router_register_bad_json.find("\"type\":\"json_invalid\"") != std::string::npos, "router register bad json validation type");
    expect(router_register_bad_json.find("Expecting property name enclosed in double quotes") != std::string::npos, "router register bad json validation ctx");
    auto router_deregister_empty_body = post_json("/v1/router/deregister", "");
    expect(router_deregister_empty_body.find("422 Unprocessable Entity") != std::string::npos, "router deregister empty body status");
    expect(router_deregister_empty_body.find("\"type\":\"missing\"") != std::string::npos, "router deregister empty body validation type");
    auto router_deregister_bad_json = post_json("/v1/router/deregister", "{");
    expect(router_deregister_bad_json.find("422 Unprocessable Entity") != std::string::npos, "router deregister bad json status");
    expect(router_deregister_bad_json.find("\"type\":\"json_invalid\"") != std::string::npos, "router deregister bad json validation type");
    auto router_deregister_array_body = post_json("/v1/router/deregister", "[]");
    expect(router_deregister_array_body.find("422 Unprocessable Entity") != std::string::npos, "router deregister array body status");
    expect(router_deregister_array_body.find("\"type\":\"dict_type\"") != std::string::npos, "router deregister array body validation type");
    expect(router_deregister_array_body.find("\"input\":[]") != std::string::npos, "router deregister array body validation input");
    auto router_deregister_scalar_body = post_json("/v1/router/deregister", "true");
    expect(router_deregister_scalar_body.find("422 Unprocessable Entity") != std::string::npos, "router deregister scalar body status");
    expect(router_deregister_scalar_body.find("\"type\":\"dict_type\"") != std::string::npos, "router deregister scalar body validation type");
    expect(router_deregister_scalar_body.find("\"input\":true") != std::string::npos, "router deregister scalar body validation input");
    for (const auto *falsey_worker_body : {
             "{\"worker_id\":null}",
             "{\"worker_id\":false}",
             "{\"worker_id\":0}",
             "{\"worker_id\":\"\"}",
             "{\"worker_id\":[]}",
             "{\"worker_id\":{}}",
         }) {
        auto router_deregister_falsey_worker = post_json("/v1/router/deregister", falsey_worker_body);
        expect(router_deregister_falsey_worker.find("400 Bad Request") != std::string::npos, "router deregister falsey worker status");
        expect(router_deregister_falsey_worker.find("\"detail\":\"Missing worker_id\"") != std::string::npos, "router deregister falsey worker body");
    }
    auto router_deregister_unknown_worker = post_json("/v1/router/deregister", "{\"worker_id\":\"not-registered\"}");
    expect(router_deregister_unknown_worker.find("200 OK") != std::string::npos, "router deregister unknown worker status");
    expect(router_deregister_unknown_worker.find("\"ok\":false") != std::string::npos, "router deregister unknown worker body");
    const std::string heartbeat_array_body = "[]";
    raw = "POST /v1/router/heartbeat HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(heartbeat_array_body.size()) + "\r\n\r\n" + heartbeat_array_body;
    auto router_heartbeat_array_body = request(raw);
    expect(router_heartbeat_array_body.find("422 Unprocessable Entity") != std::string::npos, "router heartbeat array body status");
    expect(router_heartbeat_array_body.find("\"type\":\"dict_type\"") != std::string::npos, "router heartbeat array validation type");
    expect(router_heartbeat_array_body.find("\"input\":[]") != std::string::npos, "router heartbeat array validation input");
    const std::string register_null_worker_body = "{\"worker_id\":null,\"label\":\"bad\"}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_null_worker_body.size()) + "\r\n\r\n" + register_null_worker_body;
    auto router_register_null_worker = request(raw);
    expect(router_register_null_worker.find("400 Bad Request") != std::string::npos, "router register null worker status");
    expect(router_register_null_worker.find("\"detail\":\"Missing field: worker_id\"") != std::string::npos, "router register null worker body");
    const std::string register_zero_worker_body = "{\"worker_id\":0,\"url\":\"http://10.0.0.5:8091\"}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_zero_worker_body.size()) + "\r\n\r\n" + register_zero_worker_body;
    auto router_register_zero_worker = request(raw);
    expect(router_register_zero_worker.find("400 Bad Request") != std::string::npos, "router register zero worker status");
    expect(router_register_zero_worker.find("\"detail\":\"Missing field: worker_id\"") != std::string::npos, "router register zero worker body");
    const std::string register_numeric_worker_body = "{\"worker_id\":123,\"url\":\"http://10.0.0.5:8091\"}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_numeric_worker_body.size()) + "\r\n\r\n" + register_numeric_worker_body;
    auto router_register_numeric_worker = request(raw);
    expect(router_register_numeric_worker.find("200 OK") != std::string::npos, "router register numeric worker status");
    expect(router_register_numeric_worker.find("\"worker_id\":\"123\"") != std::string::npos, "router register numeric worker id");
    const std::string deregister_numeric_worker_body = "{\"worker_id\":123}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_numeric_worker_body.size()) + "\r\n\r\n" + deregister_numeric_worker_body;
    auto router_deregister_numeric_worker = request(raw);
    expect(router_deregister_numeric_worker.find("\"ok\":true") != std::string::npos, "router deregister numeric worker");

    const std::string register_true_worker_body = "{\"worker_id\":true,\"url\":\"http://10.0.0.5:8091\"}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_true_worker_body.size()) + "\r\n\r\n" + register_true_worker_body;
    auto router_register_true_worker = request(raw);
    expect(router_register_true_worker.find("200 OK") != std::string::npos, "router register true worker status");
    expect(router_register_true_worker.find("\"worker_id\":\"True\"") != std::string::npos, "router register true worker id");
    const std::string deregister_true_worker_body = "{\"worker_id\":true}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_true_worker_body.size()) + "\r\n\r\n" + deregister_true_worker_body;
    auto router_deregister_true_worker = request(raw);
    expect(router_deregister_true_worker.find("\"ok\":true") != std::string::npos, "router deregister true worker");

    const std::string register_list_worker_body = "{\"worker_id\":[\"list-id\"],\"url\":\"http://10.0.0.5:8091\"}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_list_worker_body.size()) + "\r\n\r\n" + register_list_worker_body;
    auto router_register_list_worker = request(raw);
    expect(router_register_list_worker.find("200 OK") != std::string::npos, "router register list worker status");
    expect(router_register_list_worker.find("\"worker_id\":\"['list-id']\"") != std::string::npos, "router register list worker id");
    const std::string deregister_list_worker_body = "{\"worker_id\":[\"list-id\"]}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_list_worker_body.size()) + "\r\n\r\n" + deregister_list_worker_body;
    auto router_deregister_list_worker = request(raw);
    expect(router_deregister_list_worker.find("\"ok\":true") != std::string::npos, "router deregister list worker");

    const std::string register_quote_list_worker_body = "{\"worker_id\":[\"a'b\"],\"url\":\"http://10.0.0.5:8091\"}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_quote_list_worker_body.size()) + "\r\n\r\n" + register_quote_list_worker_body;
    auto router_register_quote_list_worker = request(raw);
    expect(router_register_quote_list_worker.find("200 OK") != std::string::npos, "router register quote list worker status");
    expect(router_register_quote_list_worker.find("\"worker_id\":\"[\\\"a'b\\\"]\"") != std::string::npos, "router register quote list worker id");
    const std::string deregister_quote_list_worker_body = "{\"worker_id\":[\"a'b\"]}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_quote_list_worker_body.size()) + "\r\n\r\n" + deregister_quote_list_worker_body;
    auto router_deregister_quote_list_worker = request(raw);
    expect(router_deregister_quote_list_worker.find("\"ok\":true") != std::string::npos, "router deregister quote list worker");

    const std::string register_numeric_list_worker_body = "{\"worker_id\":[1e2,-0],\"url\":\"http://10.0.0.5:8091\"}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_numeric_list_worker_body.size()) + "\r\n\r\n" + register_numeric_list_worker_body;
    auto router_register_numeric_list_worker = request(raw);
    expect(router_register_numeric_list_worker.find("200 OK") != std::string::npos, "router register numeric list worker status");
    expect(router_register_numeric_list_worker.find("\"worker_id\":\"[100.0, 0]\"") != std::string::npos, "router register numeric list worker id");
    const std::string deregister_numeric_list_worker_body = "{\"worker_id\":[1e2,-0]}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_numeric_list_worker_body.size()) + "\r\n\r\n" + deregister_numeric_list_worker_body;
    auto router_deregister_numeric_list_worker = request(raw);
    expect(router_deregister_numeric_list_worker.find("\"ok\":true") != std::string::npos, "router deregister numeric list worker");

    const std::string register_dict_worker_body = "{\"worker_id\":{\"a\":1,\"b\":\"x\"},\"url\":\"http://10.0.0.5:8091\"}";
    auto router_register_dict_worker = post_json("/v1/router/register", register_dict_worker_body);
    expect(router_register_dict_worker.find("200 OK") != std::string::npos, "router register dict worker status");
    expect(router_register_dict_worker.find("\"worker_id\":\"{'a': 1, 'b': 'x'}\"") != std::string::npos, "router register dict worker id");
    const std::string deregister_dict_worker_body = "{\"worker_id\":{\"a\":1,\"b\":\"x\"}}";
    auto router_deregister_dict_worker = post_json("/v1/router/deregister", deregister_dict_worker_body);
    expect(router_deregister_dict_worker.find("200 OK") != std::string::npos, "router deregister dict worker status");
    expect(router_deregister_dict_worker.find("\"ok\":true") != std::string::npos, "router deregister dict worker");

    const std::string register_string_fields_body = "{\"worker_id\":\"string-fields\",\"url\":\"http://10.0.0.5:8091\",\"label\":true,\"api_key\":true,\"version\":[\"abc\"]}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_string_fields_body.size()) + "\r\n\r\n" + register_string_fields_body;
    auto router_register_string_fields = request(raw);
    expect(router_register_string_fields.find("200 OK") != std::string::npos, "router register string fields status");
    expect(router_register_string_fields.find("\"label\":\"True\"") != std::string::npos, "router register bool label string");
    expect(router_register_string_fields.find("\"version\":\"['abc']\"") != std::string::npos, "router register list version string");
    expect(router_register_string_fields.find("\"circuit_open_until\":0.0") != std::string::npos, "router register public circuit float");
    const std::string heartbeat_string_fields_body = "{\"worker_id\":\"string-fields\",\"version\":true}";
    raw = "POST /v1/router/heartbeat HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(heartbeat_string_fields_body.size()) + "\r\n\r\n" + heartbeat_string_fields_body;
    auto router_heartbeat_string_fields = request(raw);
    expect(router_heartbeat_string_fields.find("200 OK") != std::string::npos, "router heartbeat string fields status");
    expect(router_heartbeat_string_fields.find("\"version\":\"True\"") != std::string::npos, "router heartbeat bool version string");
    const std::string deregister_string_fields_body = "{\"worker_id\":\"string-fields\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_string_fields_body.size()) + "\r\n\r\n" + deregister_string_fields_body;
    auto router_deregister_string_fields = request(raw);
    expect(router_deregister_string_fields.find("\"ok\":true") != std::string::npos, "router deregister string fields");

    const std::string register_string_maps_body = "{\"worker_id\":\"string-maps\",\"url\":\"http://10.0.0.5:8091\",\"model_quant\":{\"m\":true,\"empty\":\"\",\"zero\":0,\"arr\":[1]}}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_string_maps_body.size()) + "\r\n\r\n" + register_string_maps_body;
    auto router_register_string_maps = request(raw);
    expect(router_register_string_maps.find("200 OK") != std::string::npos, "router register string maps status");
    expect(router_register_string_maps.find("\"model_quant\":{") != std::string::npos, "router register model quant string map");
    expect(router_register_string_maps.find("\"m\":\"True\"") != std::string::npos, "router register model quant bool string");
    expect(router_register_string_maps.find("\"arr\":\"[1]\"") != std::string::npos, "router register model quant list string");
    expect(router_register_string_maps.find("\"zero\"") == std::string::npos, "router register model quant drops falsey");
    const std::string heartbeat_string_maps_body = "{\"worker_id\":\"string-maps\",\"cap_models\":{\"tts\":true,\"skip\":false}}";
    raw = "POST /v1/router/heartbeat HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(heartbeat_string_maps_body.size()) + "\r\n\r\n" + heartbeat_string_maps_body;
    auto router_heartbeat_string_maps = request(raw);
    expect(router_heartbeat_string_maps.find("200 OK") != std::string::npos, "router heartbeat string maps status");
    expect(router_heartbeat_string_maps.find("\"cap_models\":{\"tts\":\"True\"}") != std::string::npos, "router heartbeat cap models string map");
    const std::string deregister_string_maps_body = "{\"worker_id\":\"string-maps\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_string_maps_body.size()) + "\r\n\r\n" + deregister_string_maps_body;
    auto router_deregister_string_maps = request(raw);
    expect(router_deregister_string_maps.find("\"ok\":true") != std::string::npos, "router deregister string maps");

    const std::string register_falsey_slots_body = "{\"worker_id\":\"falsey-slots\",\"url\":\"http://10.0.0.5:8091\",\"cap_slots\":{\"text\":{\"capacity\":null,\"in_flight\":true,\"queued\":false}}}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_falsey_slots_body.size()) + "\r\n\r\n" + register_falsey_slots_body;
    auto router_register_falsey_slots = request(raw);
    expect(router_register_falsey_slots.find("200 OK") != std::string::npos, "router register falsey slots status");
    expect(router_register_falsey_slots.find("\"queue_capacity\":1,\"in_flight\":0,\"reported_in_flight\":0") != std::string::npos, "router register nested slot in_flight not top level");
    expect(router_register_falsey_slots.find("\"cap_slots\":{\"text\":{\"capacity\":0,\"in_flight\":1,\"queued\":0,\"available\":0}}") != std::string::npos, "router register falsey slots normalized");
    expect(router_register_falsey_slots.find("\"slot_total_capacity\":0") != std::string::npos, "router register falsey slots total capacity");
    const std::string deregister_falsey_slots_body = "{\"worker_id\":\"falsey-slots\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_falsey_slots_body.size()) + "\r\n\r\n" + deregister_falsey_slots_body;
    auto router_deregister_falsey_slots = request(raw);
    expect(router_deregister_falsey_slots.find("\"ok\":true") != std::string::npos, "router deregister falsey slots");

    const std::string register_ordered_slots_body = "{\"worker_id\":\"ordered-slots\",\"url\":\"http://10.0.0.5:8091\",\"cap_slots\":{\"tts\":{\"capacity\":1},\"text\":{\"capacity\":2}},\"model_slots\":{\"z-model\":{\"capacity\":1},\"a-model\":{\"capacity\":2}}}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_ordered_slots_body.size()) + "\r\n\r\n" + register_ordered_slots_body;
    auto router_register_ordered_slots = request(raw);
    expect(router_register_ordered_slots.find("200 OK") != std::string::npos, "router register ordered slots status");
    expect(router_register_ordered_slots.find("\"cap_slots\":{\"tts\":{\"capacity\":1,\"in_flight\":0,\"queued\":0,\"available\":1},\"text\":{\"capacity\":2,\"in_flight\":0,\"queued\":0,\"available\":2}}") != std::string::npos, "router register preserves cap slot order");
    expect(router_register_ordered_slots.find("\"model_slots\":{\"z-model\":{\"capacity\":1,\"in_flight\":0,\"queued\":0,\"available\":1},\"a-model\":{\"capacity\":2,\"in_flight\":0,\"queued\":0,\"available\":2}}") != std::string::npos, "router register preserves model slot order");
    const std::string heartbeat_ordered_slots_body = "{\"worker_id\":\"ordered-slots\",\"cap_slots\":{\"video\":{\"capacity\":1},\"image\":{\"capacity\":2}},\"model_slots\":{\"new-z\":{\"capacity\":1},\"new-a\":{\"capacity\":2}}}";
    raw = "POST /v1/router/heartbeat HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(heartbeat_ordered_slots_body.size()) + "\r\n\r\n" + heartbeat_ordered_slots_body;
    auto router_heartbeat_ordered_slots = request(raw);
    expect(router_heartbeat_ordered_slots.find("200 OK") != std::string::npos, "router heartbeat ordered slots status");
    expect(router_heartbeat_ordered_slots.find("\"cap_slots\":{\"video\":{\"capacity\":1,\"in_flight\":0,\"queued\":0,\"available\":1},\"image\":{\"capacity\":2,\"in_flight\":0,\"queued\":0,\"available\":2}}") != std::string::npos, "router heartbeat preserves cap slot order");
    expect(router_heartbeat_ordered_slots.find("\"model_slots\":{\"new-z\":{\"capacity\":1,\"in_flight\":0,\"queued\":0,\"available\":1},\"new-a\":{\"capacity\":2,\"in_flight\":0,\"queued\":0,\"available\":2}}") != std::string::npos, "router heartbeat preserves model slot order");
    const std::string deregister_ordered_slots_body = "{\"worker_id\":\"ordered-slots\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_ordered_slots_body.size()) + "\r\n\r\n" + deregister_ordered_slots_body;
    auto router_deregister_ordered_slots = request(raw);
    expect(router_deregister_ordered_slots.find("\"ok\":true") != std::string::npos, "router deregister ordered slots");

    const std::string register_underscore_ints_body = "{\"worker_id\":\"underscore-ints\",\"url\":\"http://10.0.0.5:8091\",\"free_vram_gb\":\"1_000.5\",\"total_vram_gb\":\"1_000.0\",\"queue_capacity\":\"1_000\",\"cap_slots\":{\"text\":{\"capacity\":\"1_000\",\"in_flight\":\"1_0\",\"queued\":\"0\"}},\"model_context\":{\"ctx\":\"2_048\"}}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_underscore_ints_body.size()) + "\r\n\r\n" + register_underscore_ints_body;
    auto router_register_underscore_ints = request(raw);
    expect(router_register_underscore_ints.find("200 OK") != std::string::npos, "router register underscore ints status");
    expect(router_register_underscore_ints.find("\"free_vram_gb\":1000.5") != std::string::npos, "router register underscore free vram");
    expect(router_register_underscore_ints.find("\"total_vram_gb\":1000.0") != std::string::npos, "router register underscore total vram");
    expect(router_register_underscore_ints.find("\"queue_capacity\":1000") != std::string::npos, "router register underscore queue capacity");
    expect(router_register_underscore_ints.find("\"cap_slots\":{\"text\":{\"capacity\":1000,\"in_flight\":10,\"queued\":0,\"available\":990}}") != std::string::npos, "router register underscore slot ints");
    expect(router_register_underscore_ints.find("\"model_context\":{\"ctx\":2048}") != std::string::npos, "router register underscore model context");
    const std::string heartbeat_underscore_ints_body = "{\"worker_id\":\"underscore-ints\",\"free_ram_gb\":\"2_000.25\",\"queue_depth\":\"1_2\",\"cap_slots\":{\"text\":{\"capacity\":\"2_000\",\"in_flight\":\"2_0\",\"queued\":\"1_0\"}},\"model_context\":{\"ctx\":\"4_096\"}}";
    raw = "POST /v1/router/heartbeat HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(heartbeat_underscore_ints_body.size()) + "\r\n\r\n" + heartbeat_underscore_ints_body;
    auto router_heartbeat_underscore_ints = request(raw);
    expect(router_heartbeat_underscore_ints.find("200 OK") != std::string::npos, "router heartbeat underscore ints status");
    expect(router_heartbeat_underscore_ints.find("\"free_ram_gb\":2000.25") != std::string::npos, "router heartbeat underscore free ram");
    expect(router_heartbeat_underscore_ints.find("\"queue_depth\":12") != std::string::npos, "router heartbeat underscore queue depth");
    expect(router_heartbeat_underscore_ints.find("\"cap_slots\":{\"text\":{\"capacity\":2000,\"in_flight\":20,\"queued\":10,\"available\":1970}}") != std::string::npos, "router heartbeat underscore slot ints");
    expect(router_heartbeat_underscore_ints.find("\"model_context\":{\"ctx\":4096}") != std::string::npos, "router heartbeat underscore model context");
    const std::string deregister_underscore_ints_body = "{\"worker_id\":\"underscore-ints\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_underscore_ints_body.size()) + "\r\n\r\n" + deregister_underscore_ints_body;
    auto router_deregister_underscore_ints = request(raw);
    expect(router_deregister_underscore_ints.find("\"ok\":true") != std::string::npos, "router deregister underscore ints");

    const std::string register_model_context_body = "{\"worker_id\":\"context-map\",\"url\":\"http://10.0.0.5:8091\",\"model_context\":{\"truthy\":true,\"falsey\":false,\"quoted\":\"2048\",\"floaty\":12.9}}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_model_context_body.size()) + "\r\n\r\n" + register_model_context_body;
    auto router_register_model_context = request(raw);
    expect(router_register_model_context.find("200 OK") != std::string::npos, "router register model context status");
    expect(router_register_model_context.find("\"model_context\":{\"truthy\":1,\"falsey\":0,\"quoted\":2048,\"floaty\":12}") != std::string::npos, "router register model context int map");
    const std::string heartbeat_model_context_body = "{\"worker_id\":\"context-map\",\"model_context\":{\"updated\":\"4096\"}}";
    raw = "POST /v1/router/heartbeat HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(heartbeat_model_context_body.size()) + "\r\n\r\n" + heartbeat_model_context_body;
    auto router_heartbeat_model_context = request(raw);
    expect(router_heartbeat_model_context.find("200 OK") != std::string::npos, "router heartbeat model context status");
    expect(router_heartbeat_model_context.find("\"model_context\":{\"updated\":4096}") != std::string::npos, "router heartbeat model context int map");
    const std::string deregister_model_context_body = "{\"worker_id\":\"context-map\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_model_context_body.size()) + "\r\n\r\n" + deregister_model_context_body;
    auto router_deregister_model_context = request(raw);
    expect(router_deregister_model_context.find("\"ok\":true") != std::string::npos, "router deregister model context");

    const std::string register_gpus_string_body = "{\"worker_id\":\"gpus-string\",\"url\":\"http://10.0.0.5:8091\",\"gpus\":\"ab\",\"best_tier\":99}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_gpus_string_body.size()) + "\r\n\r\n" + register_gpus_string_body;
    auto router_register_gpus_string = request(raw);
    expect(router_register_gpus_string.find("500 Internal Server Error") != std::string::npos, "router register gpus string status");
    expect(router_register_gpus_string.find("\r\n\r\nInternal Server Error") != std::string::npos, "router register gpus string body");
    const std::string register_gpus_nonobject_list_body = "{\"worker_id\":\"gpus-list-str\",\"url\":\"http://10.0.0.5:8091\",\"gpus\":[\"bad\"],\"best_tier\":99}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_gpus_nonobject_list_body.size()) + "\r\n\r\n" + register_gpus_nonobject_list_body;
    auto router_register_gpus_nonobject_list = request(raw);
    expect(router_register_gpus_nonobject_list.find("500 Internal Server Error") != std::string::npos, "router register gpus list string status");
    const std::string register_gpus_string_total_body = "{\"worker_id\":\"gpus-string-total\",\"url\":\"http://10.0.0.5:8091\",\"gpus\":[{\"name\":\"GPU\",\"total_vram_gb\":\"4\"}],\"best_tier\":99}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_gpus_string_total_body.size()) + "\r\n\r\n" + register_gpus_string_total_body;
    auto router_register_gpus_string_total = request(raw);
    expect(router_register_gpus_string_total.find("500 Internal Server Error") != std::string::npos, "router register gpus string total status");

    const std::string register_string_lists_body = "{\"worker_id\":\"string-lists\",\"capabilities\":\"text\",\"models\":\"ab\"}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_string_lists_body.size()) + "\r\n\r\n" + register_string_lists_body;
    auto router_register_string_lists = request(raw);
    expect(router_register_string_lists.find("200 OK") != std::string::npos, "router register string lists status");
    expect(router_register_string_lists.find("\"capabilities\":[\"t\",\"e\",\"x\",\"t\"]") != std::string::npos, "router register list string capabilities");
    expect(router_register_string_lists.find("\"models\":[\"a\",\"b\"]") != std::string::npos, "router register list string models");
    const std::string deregister_string_lists_body = "{\"worker_id\":\"string-lists\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_string_lists_body.size()) + "\r\n\r\n" + deregister_string_lists_body;
    auto router_deregister_string_lists = request(raw);
    expect(router_deregister_string_lists.find("\"ok\":true") != std::string::npos, "router deregister string lists");

    const std::string register_dict_lists_body = "{\"worker_id\":\"dict-lists\",\"url\":\"http://10.0.0.5:8091\",\"capabilities\":{\"z\":1,\"x\":2},\"models\":{\"b\":1,\"a\":2}}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_dict_lists_body.size()) + "\r\n\r\n" + register_dict_lists_body;
    auto router_register_dict_lists = request(raw);
    expect(router_register_dict_lists.find("200 OK") != std::string::npos, "router register dict lists status");
    expect(router_register_dict_lists.find("\"capabilities\":[\"z\",\"x\"]") != std::string::npos, "router register dict capabilities preserve order");
    expect(router_register_dict_lists.find("\"models\":[\"b\",\"a\"]") != std::string::npos, "router register dict models preserve order");
    const std::string heartbeat_dict_lists_body = "{\"worker_id\":\"dict-lists\",\"models\":{\"new-b\":1,\"new-a\":2},\"capabilities\":{\"new-z\":1,\"new-x\":2}}";
    raw = "POST /v1/router/heartbeat HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(heartbeat_dict_lists_body.size()) + "\r\n\r\n" + heartbeat_dict_lists_body;
    auto router_heartbeat_dict_lists = request(raw);
    expect(router_heartbeat_dict_lists.find("200 OK") != std::string::npos, "router heartbeat dict lists status");
    expect(router_heartbeat_dict_lists.find("\"capabilities\":[\"new-z\",\"new-x\"]") != std::string::npos, "router heartbeat dict capabilities preserve order");
    expect(router_heartbeat_dict_lists.find("\"models\":[\"new-b\",\"new-a\"]") != std::string::npos, "router heartbeat dict models preserve order");
    const std::string deregister_dict_lists_body = "{\"worker_id\":\"dict-lists\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_dict_lists_body.size()) + "\r\n\r\n" + deregister_dict_lists_body;
    auto router_deregister_dict_lists = request(raw);
    expect(router_deregister_dict_lists.find("\"ok\":true") != std::string::npos, "router deregister dict lists");

    const std::string register_mixed_lists_body = "{\"worker_id\":\"mixed-lists\",\"url\":\"http://10.0.0.5:8091\",\"capabilities\":[1,\"x\"],\"models\":[\"a\",1,true,null,{\"k\":2}]}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_mixed_lists_body.size()) + "\r\n\r\n" + register_mixed_lists_body;
    auto router_register_mixed_lists = request(raw);
    expect(router_register_mixed_lists.find("200 OK") != std::string::npos, "router register mixed lists status");
    expect(router_register_mixed_lists.find("\"capabilities\":[1,\"x\"]") != std::string::npos, "router register mixed capabilities public JSON");
    expect(router_register_mixed_lists.find("\"models\":[\"a\",1,true,null,{\"k\":2}]") != std::string::npos, "router register mixed models public JSON");
    const std::string heartbeat_false_models_body = "{\"worker_id\":\"mixed-lists\",\"models\":false}";
    raw = "POST /v1/router/heartbeat HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(heartbeat_false_models_body.size()) + "\r\n\r\n" + heartbeat_false_models_body;
    auto router_heartbeat_false_models = request(raw);
    expect(router_heartbeat_false_models.find("500 Internal Server Error") != std::string::npos, "router heartbeat false models status");
    const std::string deregister_mixed_lists_body = "{\"worker_id\":\"mixed-lists\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_mixed_lists_body.size()) + "\r\n\r\n" + deregister_mixed_lists_body;
    auto router_deregister_mixed_lists = request(raw);
    expect(router_deregister_mixed_lists.find("\"ok\":true") != std::string::npos, "router deregister mixed lists");
    const std::string register_bad_float_body = "{\"worker_id\":\"bad-float\",\"free_vram_gb\":\"bad\"}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_bad_float_body.size()) + "\r\n\r\n" + register_bad_float_body;
    auto router_register_bad_float = request(raw);
    expect(router_register_bad_float.find("500 Internal Server Error") != std::string::npos, "router register bad float status");
    expect(router_register_bad_float.find("Content-Type: text/plain; charset=utf-8") != std::string::npos, "router register bad float content type");
    expect(router_register_bad_float.find("\r\n\r\nInternal Server Error") != std::string::npos, "router register bad float body");
    const std::string register_bad_context_body = "{\"worker_id\":\"bad-context\",\"model_context\":{\"m\":\"bad\"}}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_bad_context_body.size()) + "\r\n\r\n" + register_bad_context_body;
    auto router_register_bad_context = request(raw);
    expect(router_register_bad_context.find("500 Internal Server Error") != std::string::npos, "router register bad context status");
    const std::string register_bad_slots_body = "{\"worker_id\":\"bad-slots\",\"cap_slots\":[1]}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_bad_slots_body.size()) + "\r\n\r\n" + register_bad_slots_body;
    auto router_register_bad_slots = request(raw);
    expect(router_register_bad_slots.find("500 Internal Server Error") != std::string::npos, "router register bad slots status");
    const std::string register_bad_slot_value_body = "{\"worker_id\":\"bad-slot-value\",\"cap_slots\":{\"text\":{\"capacity\":\"bad\"}}}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_bad_slot_value_body.size()) + "\r\n\r\n" + register_bad_slot_value_body;
    auto router_register_bad_slot_value = request(raw);
    expect(router_register_bad_slot_value.find("500 Internal Server Error") != std::string::npos, "router register bad slot value status");
    const std::string register_bad_gpus_body = "{\"worker_id\":\"bad-gpus\",\"gpus\":true,\"best_tier\":99}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_bad_gpus_body.size()) + "\r\n\r\n" + register_bad_gpus_body;
    auto router_register_bad_gpus = request(raw);
    expect(router_register_bad_gpus.find("500 Internal Server Error") != std::string::npos, "router register bad gpus scalar status");

    auto router_tunnel_missing_auth_ws = request("GET /v1/router/tunnel?worker_id=home-node HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n");
    expect(router_tunnel_missing_auth_ws.find("101 Switching Protocols") != std::string::npos, "router tunnel missing auth websocket handshake");
    expect(has_websocket_close_code(router_tunnel_missing_auth_ws, 4401), "router tunnel missing auth close code");
    auto router_tunnel_plain_auth_ws = request("GET /v1/router/tunnel?worker_id=home-node HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: secret\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n");
    expect(has_websocket_close_code(router_tunnel_plain_auth_ws, 4401), "router tunnel plain auth rejected");
    auto router_tunnel_missing_worker_ws = request("GET /v1/router/tunnel HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n");
    expect(router_tunnel_missing_worker_ws.find("101 Switching Protocols") != std::string::npos, "router tunnel missing worker websocket handshake");
    expect(has_websocket_close_code(router_tunnel_missing_worker_ws, 4400), "router tunnel missing worker close code");
    auto router_tunnel_ws = request("GET /v1/router/tunnel?worker_id=probe-node HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n");
    expect(router_tunnel_ws.find("101 Switching Protocols") != std::string::npos, "router tunnel websocket handshake");
    expect(router_tunnel_ws.find("Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=") != std::string::npos, "router tunnel websocket accept");
    expect(!has_websocket_close_code(router_tunnel_ws, 1011), "router tunnel accepted websocket does not use placeholder close");

    const std::string register_body =
        "{\"worker_id\":\"worker-1\","
        "\"label\":\"Worker One\","
        "\"url\":\"http://10.0.0.2:8091\","
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"text\",\"vision\"],"
        "\"models\":[\"model-a\"],"
        "\"free_vram_gb\":\"12.5\","
        "\"total_vram_gb\":\"24\","
        "\"queue_depth\":\"1\","
        "\"queue_capacity\":\"4\","
        "\"best_tier\":\"80\","
        "\"gpus\":[{\"index\":0,\"name\":\"RTX 4090\",\"total_vram_gb\":24,\"tier\":80,\"backend\":\"cuda\"}],"
        "\"version\":\"abc123\"}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_body.size()) + "\r\n\r\n" + register_body;
    auto registered = request(raw);
    expect(registered.find("200 OK") != std::string::npos, "router register status");
    expect(registered.find("\"ok\":true") != std::string::npos, "router register ok");
    expect(registered.find("\"worker_id\":\"worker-1\"") != std::string::npos, "router register worker id");
    expect(registered.find("\"capabilities\":[\"text\",\"vision\"]") != std::string::npos, "router register capabilities");
    expect(registered.find("\"free_vram_gb\":12.5") != std::string::npos, "router register coerces quoted free vram");
    expect(registered.find("\"queue_capacity\":4") != std::string::npos, "router register coerces quoted queue capacity");

    const std::string register_numeric_coerce_body =
        "{\"worker_id\":\"coerce-worker\","
        "\"label\":\"Coerce Worker\","
        "\"url\":\"http://10.0.0.3:8091\","
        "\"capabilities\":[],"
        "\"models\":[],"
        "\"free_vram_gb\":true,"
        "\"queue_capacity\":0,"
        "\"in_flight\":true}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_numeric_coerce_body.size()) + "\r\n\r\n" + register_numeric_coerce_body;
    auto registered_numeric_coerce = request(raw);
    expect(registered_numeric_coerce.find("200 OK") != std::string::npos, "router register numeric coercion status");
    expect(registered_numeric_coerce.find("\"free_vram_gb\":1.0") != std::string::npos, "router register bool float coercion");
    expect(registered_numeric_coerce.find("\"queue_capacity\":1") != std::string::npos, "router register falsey queue capacity default");
    expect(registered_numeric_coerce.find("\"in_flight\":1") != std::string::npos, "router register bool int coercion");

    const std::string register_url_trim_body = "{\"worker_id\":\"url-trim-worker\",\"url\":\" http://10.0.0.8:8091/// \",\"capabilities\":[],\"models\":[]}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_url_trim_body.size()) + "\r\n\r\n" + register_url_trim_body;
    auto registered_url_trim = request(raw);
    expect(registered_url_trim.find("200 OK") != std::string::npos, "router register url trim status");
    expect(registered_url_trim.find("\"url\":\"http://10.0.0.8:8091\"") != std::string::npos, "router register strips url whitespace and slashes");

    const std::string deregister_url_trim_body = "{\"worker_id\":\"url-trim-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_url_trim_body.size()) + "\r\n\r\n" + deregister_url_trim_body;
    auto deregistered_url_trim = request(raw);
    expect(deregistered_url_trim.find("200 OK") != std::string::npos, "router deregister url trim worker status");

    const std::string heartbeat_numeric_coerce_body = "{\"worker_id\":\"coerce-worker\",\"queue_capacity\":false,\"free_ram_gb\":true}";
    raw = "POST /v1/router/heartbeat HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(heartbeat_numeric_coerce_body.size()) + "\r\n\r\n" + heartbeat_numeric_coerce_body;
    auto heartbeat_numeric_coerce = request(raw);
    expect(heartbeat_numeric_coerce.find("200 OK") != std::string::npos, "router heartbeat numeric coercion status");
    expect(heartbeat_numeric_coerce.find("\"queue_capacity\":0") != std::string::npos, "router heartbeat false bool int coercion");
    expect(heartbeat_numeric_coerce.find("\"free_ram_gb\":1.0") != std::string::npos, "router heartbeat true bool float coercion");

    const std::string deregister_numeric_coerce_body = "{\"worker_id\":\"coerce-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_numeric_coerce_body.size()) + "\r\n\r\n" + deregister_numeric_coerce_body;
    auto deregistered_numeric_coerce = request(raw);
    expect(deregistered_numeric_coerce.find("200 OK") != std::string::npos, "router deregister numeric coercion worker status");

    const std::string heartbeat_gpus_string_body = "{\"worker_id\":\"worker-1\",\"gpus\":\"ignored\"}";
    raw = "POST /v1/router/heartbeat HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(heartbeat_gpus_string_body.size()) + "\r\n\r\n" + heartbeat_gpus_string_body;
    auto heartbeat_gpus_string = request(raw);
    expect(heartbeat_gpus_string.find("200 OK") != std::string::npos, "router heartbeat gpus string status");
    expect(heartbeat_gpus_string.find("\"gpus\":[{\"index\":0") != std::string::npos, "router heartbeat gpus string ignored");
    expect(heartbeat_gpus_string.find("\"best_tier\":80") != std::string::npos, "router heartbeat gpus string preserves tier");

    const std::string register_loopback_worker_body =
        "{\"worker_id\":\"loopback-worker\","
        "\"label\":\"Loopback Worker\","
        "\"url\":\"http://localhost:9999\","
        "\"port\":18119,"
        "\"capabilities\":[\"text\"],"
        "\"models\":[\"loopback-model\"],"
        "\"gpus\":[{\"index\":0,\"name\":\"NVIDIA GeForce RTX 4090\",\"tier\":80,\"backend\":\"cuda\"},{\"index\":-1,\"name\":\"CPU\",\"tier\":2,\"backend\":\"cpu\"}],"
        "\"rack\":\"a1\","
        "\"az\":\"z9\","
        "\"extra\":{\"note\":\"nested\"}}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_loopback_worker_body.size()) + "\r\n\r\n" + register_loopback_worker_body;
    auto registered_loopback = request(raw);
    expect(registered_loopback.find("200 OK") != std::string::npos, "router loopback worker register status");
    expect(registered_loopback.find("\"url\":\"http://127.0.0.1:18119\"") != std::string::npos, "router loopback worker source ip url");
    expect(registered_loopback.find("\"total_vram_gb\":0.0") != std::string::npos, "router register gpus do not populate top-level total vram");
    expect(registered_loopback.find("\"best_tier\":80") != std::string::npos, "router register derives best gpu tier");
    expect(registered_loopback.find("\"rack\":\"a1\"") != std::string::npos, "router register preserves unknown extra field");
    expect(registered_loopback.find("\"note\":\"nested\"") != std::string::npos, "router register preserves explicit extra as unknown field");
    expect(registered_loopback.find("\"extra\":{\"rack\":\"a1\",\"az\":\"z9\",\"note\":\"nested\"}") != std::string::npos, "router register extra merges explicit metadata in python insertion order");
    const std::string heartbeat_loopback_body = "{\"worker_id\":\"loopback-worker\",\"gpus\":[{\"index\":0,\"name\":\"AMD Radeon RX 7900 XTX\",\"tier\":55,\"backend\":\"rocm\"}],\"extra\":{\"zone\":\"west\",\"rack\":\"b2\"}}";
    raw = "POST /v1/router/heartbeat HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(heartbeat_loopback_body.size()) + "\r\n\r\n" + heartbeat_loopback_body;
    auto heartbeat_loopback = request(raw);
    expect(heartbeat_loopback.find("200 OK") != std::string::npos, "router loopback worker heartbeat status");
    expect(heartbeat_loopback.find("\"best_tier\":55") != std::string::npos, "router heartbeat derives best gpu tier");
    expect(heartbeat_loopback.find("\"rack\":\"b2\"") != std::string::npos, "router heartbeat updates existing extra");
    expect(heartbeat_loopback.find("\"zone\":\"west\"") != std::string::npos, "router heartbeat merges explicit extra");
    expect(heartbeat_loopback.find("\"extra\":{\"rack\":\"b2\",\"az\":\"z9\",\"note\":\"nested\",\"zone\":\"west\"}") != std::string::npos, "router heartbeat extra preserves update order");
    const std::string deregister_loopback_body = "{\"worker_id\":\"loopback-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_loopback_body.size()) + "\r\n\r\n" + deregister_loopback_body;
    auto deregistered_loopback = request(raw);
    expect(deregistered_loopback.find("200 OK") != std::string::npos, "router loopback worker deregister status");

    const std::string register_tunnel_worker_body =
        "{\"worker_id\":\"tunnel-worker\","
        "\"label\":\"Tunnel Worker\","
        "\"url\":\"tunnel://home-node\","
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"text\"],"
        "\"models\":[\"tunnel-model\"],"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_tunnel_worker_body.size()) + "\r\n\r\n" + register_tunnel_worker_body;
    auto registered_tunnel = request(raw);
    expect(registered_tunnel.find("200 OK") != std::string::npos, "router tunnel worker register status");
    expect(registered_tunnel.find("\"url\":\"tunnel://home-node\"") != std::string::npos, "router tunnel url preserved");
    expect(registered_tunnel.find("\"tunnel\":") == std::string::npos, "router register public omits dashboard tunnel flag");
    expect(registered_tunnel.find("\"tunnel_connected\":") == std::string::npos, "router register public omits dashboard tunnel connection");

    auto workers = request("GET /v1/router/workers HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(workers.find("200 OK") != std::string::npos, "router workers status");
    expect(workers.find("\"object\":\"list\"") != std::string::npos, "router workers object");
    expect(workers.find("\"ttl_seconds\":60.0") != std::string::npos, "router workers ttl route default float");
    expect(workers.find("\"worker_id\":\"worker-1\"") != std::string::npos, "router workers contains worker");
    expect(workers.find("\"worker_id\":\"tunnel-worker\"") != std::string::npos, "router workers contains tunnel worker");
    expect(
        workers.find("\"worker_id\":\"worker-1\"") < workers.find("\"worker_id\":\"tunnel-worker\""),
        "router workers preserves registration order"
    );
    expect(workers.find("\"tunnel\":") == std::string::npos, "router workers public omits dashboard tunnel flag");
    expect(workers.find("\"tunnel_connected\":") == std::string::npos, "router workers public omits dashboard tunnel connection");

    auto registered_z_order = post_json(
        "/v1/router/register",
        "{\"worker_id\":\"z-order-worker\",\"label\":\"Z Order\",\"url\":\"http://10.0.0.9:8091\",\"capabilities\":[],\"models\":[]}"
    );
    expect(registered_z_order.find("200 OK") != std::string::npos, "router register z order worker status");
    auto registered_a_order = post_json(
        "/v1/router/register",
        "{\"worker_id\":\"a-order-worker\",\"label\":\"A Order\",\"url\":\"http://10.0.0.10:8091\",\"capabilities\":[],\"models\":[]}"
    );
    expect(registered_a_order.find("200 OK") != std::string::npos, "router register a order worker status");
    auto order_workers = request("GET /v1/router/workers HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(
        order_workers.find("\"worker_id\":\"z-order-worker\"") < order_workers.find("\"worker_id\":\"a-order-worker\""),
        "router workers uses insertion order, not lexical order"
    );
    auto reregistered_z_order = post_json(
        "/v1/router/register",
        "{\"worker_id\":\"z-order-worker\",\"label\":\"Z ReRegistered\",\"url\":\"http://10.0.0.11:8091\",\"capabilities\":[],\"models\":[]}"
    );
    expect(reregistered_z_order.find("200 OK") != std::string::npos, "router reregister z order worker status");
    order_workers = request("GET /v1/router/workers HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(
        order_workers.find("\"worker_id\":\"z-order-worker\"") < order_workers.find("\"worker_id\":\"a-order-worker\""),
        "router workers preserves order across reregister"
    );
    expect(order_workers.find("\"label\":\"Z ReRegistered\"") != std::string::npos, "router workers reregister updates fields");
    auto deregister_z_order = post_json("/v1/router/deregister", "{\"worker_id\":\"z-order-worker\"}");
    expect(deregister_z_order.find("\"ok\":true") != std::string::npos, "router deregister z order worker");
    registered_z_order = post_json(
        "/v1/router/register",
        "{\"worker_id\":\"z-order-worker\",\"label\":\"Z Order\",\"url\":\"http://10.0.0.9:8091\",\"capabilities\":[],\"models\":[]}"
    );
    expect(registered_z_order.find("200 OK") != std::string::npos, "router register z order worker after deregister status");
    order_workers = request("GET /v1/router/workers HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(
        order_workers.find("\"worker_id\":\"a-order-worker\"") < order_workers.find("\"worker_id\":\"z-order-worker\""),
        "router workers moves deregistered worker to new insertion position"
    );
    auto deregister_a_order = post_json("/v1/router/deregister", "{\"worker_id\":\"a-order-worker\"}");
    expect(deregister_a_order.find("\"ok\":true") != std::string::npos, "router deregister a order worker");
    deregister_z_order = post_json("/v1/router/deregister", "{\"worker_id\":\"z-order-worker\"}");
    expect(deregister_z_order.find("\"ok\":true") != std::string::npos, "router deregister z order worker cleanup");

    const std::string heartbeat_body = "{\"worker_id\":\"worker-1\",\"queue_depth\":\"0\",\"in_flight\":\"2\",\"models\":[\"model-b\"]}";
    raw = "POST /v1/router/heartbeat HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(heartbeat_body.size()) + "\r\n\r\n" + heartbeat_body;
    auto heartbeat = request(raw);
    expect(heartbeat.find("200 OK") != std::string::npos, "router heartbeat status");
    expect(heartbeat.find("\"queue_depth\":0") != std::string::npos, "router heartbeat coerces quoted queue depth");
    expect(heartbeat.find("\"in_flight\":2") != std::string::npos, "router heartbeat updates in flight");
    expect(heartbeat.find("\"models\":[\"model-b\"]") != std::string::npos, "router heartbeat updates models");

    const std::string string_lists_heartbeat_body = "{\"worker_id\":\"worker-1\",\"models\":\"xy\",\"capabilities\":\"go\"}";
    raw = "POST /v1/router/heartbeat HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(string_lists_heartbeat_body.size()) + "\r\n\r\n" + string_lists_heartbeat_body;
    auto string_lists_heartbeat = request(raw);
    expect(string_lists_heartbeat.find("200 OK") != std::string::npos, "router heartbeat string lists status");
    expect(string_lists_heartbeat.find("\"models\":[\"x\",\"y\"]") != std::string::npos, "router heartbeat list string models");
    expect(string_lists_heartbeat.find("\"capabilities\":[\"g\",\"o\"]") != std::string::npos, "router heartbeat list string capabilities");
    const std::string bad_queue_heartbeat_body = "{\"worker_id\":\"worker-1\",\"queue_depth\":\"bad\"}";
    raw = "POST /v1/router/heartbeat HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(bad_queue_heartbeat_body.size()) + "\r\n\r\n" + bad_queue_heartbeat_body;
    auto bad_queue_heartbeat = request(raw);
    expect(bad_queue_heartbeat.find("500 Internal Server Error") != std::string::npos, "router heartbeat bad queue status");
    expect(bad_queue_heartbeat.find("\r\n\r\nInternal Server Error") != std::string::npos, "router heartbeat bad queue body");
    const std::string bad_context_heartbeat_body = "{\"worker_id\":\"worker-1\",\"model_context\":{\"m\":\"bad\"}}";
    raw = "POST /v1/router/heartbeat HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(bad_context_heartbeat_body.size()) + "\r\n\r\n" + bad_context_heartbeat_body;
    auto bad_context_heartbeat = request(raw);
    expect(bad_context_heartbeat.find("500 Internal Server Error") != std::string::npos, "router heartbeat bad context status");
    const std::string bad_slot_heartbeat_body = "{\"worker_id\":\"worker-1\",\"cap_slots\":{\"text\":{\"queued\":\"bad\"}}}";
    raw = "POST /v1/router/heartbeat HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(bad_slot_heartbeat_body.size()) + "\r\n\r\n" + bad_slot_heartbeat_body;
    auto bad_slot_heartbeat = request(raw);
    expect(bad_slot_heartbeat.find("500 Internal Server Error") != std::string::npos, "router heartbeat bad slot status");

    const std::string empty_lists_heartbeat_body = "{\"worker_id\":\"worker-1\",\"models\":[],\"capabilities\":[]}";
    raw = "POST /v1/router/heartbeat HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(empty_lists_heartbeat_body.size()) + "\r\n\r\n" + empty_lists_heartbeat_body;
    auto empty_lists_heartbeat = request(raw);
    expect(empty_lists_heartbeat.find("200 OK") != std::string::npos, "router heartbeat empty lists status");
    expect(empty_lists_heartbeat.find("\"models\":[]") != std::string::npos, "router heartbeat clears models");
    expect(empty_lists_heartbeat.find("\"capabilities\":[]") != std::string::npos, "router heartbeat clears capabilities");

    const std::string restore_heartbeat_body = "{\"worker_id\":\"worker-1\",\"models\":[\"model-b\"],\"capabilities\":[\"text\",\"vision\"]}";
    raw = "POST /v1/router/heartbeat HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(restore_heartbeat_body.size()) + "\r\n\r\n" + restore_heartbeat_body;
    auto restored_heartbeat = request(raw);
    expect(restored_heartbeat.find("200 OK") != std::string::npos, "router heartbeat restore lists status");

    setenv("ROUTER_MODE", "true", 1);
    auto router_models_missing_auth = request("GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(router_models_missing_auth.find("401 Unauthorized") != std::string::npos, "router models missing auth status");
    expect(router_models_missing_auth.find("\"detail\":\"Missing Authorization header\"") != std::string::npos, "router models missing auth body");
    auto router_models_invalid_auth = request("GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer wrong\r\n\r\n");
    expect(router_models_invalid_auth.find("401 Unauthorized") != std::string::npos, "router models invalid auth status");
    expect(router_models_invalid_auth.find("\"detail\":\"Invalid API key\"") != std::string::npos, "router models invalid auth body");
    auto router_models_split_auth = request("GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Token secret\r\n\r\n");
    expect(router_models_split_auth.find("200 OK") != std::string::npos, "router models split-token auth accepted");
    auto router_models = request("GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(router_models.find("200 OK") != std::string::npos, "router models status");
    expect(router_models.find("\"id\":\"model-b\"") != std::string::npos, "router models aggregates worker model");
    expect(router_models.find("\"workers\":[\"Worker One\"]") != std::string::npos, "router models worker label");

    auto router_chat_missing_body = request("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Type: application/json\r\nContent-Length: 0\r\n\r\n");
    expect(router_chat_missing_body.find("422 Unprocessable Entity") != std::string::npos, "router chat missing body status");
    expect(router_chat_missing_body.find("\"loc\":[\"body\"]") != std::string::npos, "router chat missing body loc");
    const std::string router_completion_array_body = "[]";
    raw = "POST /v1/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(router_completion_array_body.size()) + "\r\n\r\n" + router_completion_array_body;
    auto router_completion_array = request(raw);
    expect(router_completion_array.find("422 Unprocessable Entity") != std::string::npos, "router completion array body status");
    expect(router_completion_array.find("\"type\":\"dict_type\"") != std::string::npos, "router completion array body type");
    const std::string router_embedding_bad_json_body = "{";
    raw = "POST /v1/embeddings HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(router_embedding_bad_json_body.size()) + "\r\n\r\n" + router_embedding_bad_json_body;
    auto router_embedding_bad_json = request(raw);
    expect(router_embedding_bad_json.find("422 Unprocessable Entity") != std::string::npos, "router embedding bad json status");
    expect(router_embedding_bad_json.find("\"type\":\"json_invalid\"") != std::string::npos, "router embedding bad json type");
    raw = "POST /v1/audio/speech HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(router_completion_array_body.size()) + "\r\n\r\n" + router_completion_array_body;
    auto router_tts_array = request(raw);
    expect(router_tts_array.find("422 Unprocessable Entity") != std::string::npos, "router tts array body status");
    expect(router_tts_array.find("\"type\":\"dict_type\"") != std::string::npos, "router tts array body type");
    raw = "POST /v1/images/generations HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(router_completion_array_body.size()) + "\r\n\r\n" + router_completion_array_body;
    auto router_image_array = request(raw);
    expect(router_image_array.find("422 Unprocessable Entity") != std::string::npos, "router image array body status");
    expect(router_image_array.find("\"type\":\"dict_type\"") != std::string::npos, "router image array body type");
    raw = "POST /v1/videos/generations HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(router_completion_array_body.size()) + "\r\n\r\n" + router_completion_array_body;
    auto router_video_array = request(raw);
    expect(router_video_array.find("422 Unprocessable Entity") != std::string::npos, "router video array body status");
    expect(router_video_array.find("\"type\":\"dict_type\"") != std::string::npos, "router video array body type");

    const std::string router_chat_body = "{\"model\":\"missing-model\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}";
    raw = "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(router_chat_body.size()) + "\r\n\r\n" + router_chat_body;
    auto router_chat_no_worker = request(raw);
    expect(router_chat_no_worker.find("503 Service Unavailable") != std::string::npos, "router chat no worker status");
    expect(router_chat_no_worker.find("No worker available for capability='text' model='missing-model'") != std::string::npos, "router chat no worker body");
    unsetenv("ROUTER_WAIT_TIMEOUT");

    const std::string register_literal_image_worker_body =
        "{\"worker_id\":\"literal-image-worker\","
        "\"label\":\"Literal Image Worker\","
        "\"url\":\"http://127.0.0.1:18155\","
        "\"port\":18155,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"text\"],"
        "\"models\":[\"literal-image-model\"],"
        "\"best_tier\":70,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_literal_image_worker_body.size()) + "\r\n\r\n" + register_literal_image_worker_body;
    auto registered_literal_image_worker = request(raw);
    expect(registered_literal_image_worker.find("200 OK") != std::string::npos, "router literal image worker register status");
    std::thread fake_literal_image_worker(
        fake_json_worker_once,
        18155,
        "{\"id\":\"chatcmpl-literal-image\",\"object\":\"chat.completion\",\"model\":\"literal-image-model\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"literal text routed\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string literal_image_chat_body = "{\"model\":\"literal-image-model\",\"messages\":[{\"role\":\"user\",\"content\":\"image_url\"}]}";
    raw = "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(literal_image_chat_body.size()) + "\r\n\r\n" + literal_image_chat_body;
    auto literal_image_chat = request(raw);
    fake_literal_image_worker.join();
    expect(literal_image_chat.find("200 OK") != std::string::npos, "router literal image text route status");
    expect(literal_image_chat.find("literal text routed") != std::string::npos, "router literal image text route body");
    const std::string deregister_literal_image_worker_body = "{\"worker_id\":\"literal-image-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_literal_image_worker_body.size()) + "\r\n\r\n" + deregister_literal_image_worker_body;
    auto deregistered_literal_image_worker = request(raw);
    expect(deregistered_literal_image_worker.find("200 OK") != std::string::npos, "router literal image worker deregister status");

    const std::string register_usage_worker_body =
        "{\"worker_id\":\"usage-worker\","
        "\"label\":\"Usage Worker\","
        "\"url\":\"http://127.0.0.1:18093\","
        "\"port\":18093,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"text\"],"
        "\"models\":[\"usage-model-GGUF\"],"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_usage_worker_body.size()) + "\r\n\r\n" + register_usage_worker_body;
    auto registered_usage_worker = request(raw);
    expect(registered_usage_worker.find("200 OK") != std::string::npos, "router usage worker register status");

    const std::string worker_body =
        "{\"id\":\"cmpl-worker\",\"object\":\"text_completion\",\"model\":\"usage-model\","
        "\"choices\":[{\"text\":\"proxied\",\"index\":0,\"finish_reason\":\"stop\"}],"
        "\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":4,\"total_tokens\":7},"
        "\"timings\":{\"prompt_ms\":150,\"predicted_ms\":200,\"total_ms\":350}}";
    std::thread fake_worker(fake_json_worker_once, 18093, worker_body);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string router_completion_body = "{\"model\":\"usage-model-GGUF\",\"prompt\":\"hello\"}";
    raw = "POST /v1/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(router_completion_body.size()) + "\r\n\r\n" + router_completion_body;
    auto router_completion_proxy = request(raw);
    fake_worker.join();
    expect(router_completion_proxy.find("200 OK") != std::string::npos, "router completion proxy status");
    expect(router_completion_proxy.find("\"prompt_tokens\":3") != std::string::npos, "router completion proxy body");

    const std::string stream_usage_body =
        "data: {\"id\":\"chatcmpl-stream-usage\",\"object\":\"chat.completion\",\"model\":\"usage-model\","
        "\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":8,\"total_tokens\":15}}\n\n"
        "data: [DONE]\n\n";
    std::thread fake_stream_usage_worker(fake_json_worker_once, 18093, stream_usage_body);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string stream_usage_chat_body = "{\"model\":\"usage-model-GGUF\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"stream usage\"}]}";
    raw = "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(stream_usage_chat_body.size()) + "\r\n\r\n" + stream_usage_chat_body;
    auto router_stream_usage_proxy = request(raw);
    fake_stream_usage_worker.join();
    expect(router_stream_usage_proxy.find("200 OK") != std::string::npos, "router stream usage proxy status");
    expect(router_stream_usage_proxy.find("chatcmpl-stream-usage") != std::string::npos, "router stream usage proxy body");

    const std::string completion_stream_usage_body =
        "data: {\"id\":\"cmpl-stream-usage\",\"object\":\"text_completion\",\"model\":\"usage-model\","
        "\"choices\":[],\"usage\":{\"prompt_tokens\":6,\"completion_tokens\":9,\"total_tokens\":15}}\n\n"
        "data: [DONE]\n\n";
    std::thread fake_completion_stream_usage_worker(fake_json_worker_once, 18093, completion_stream_usage_body);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string completion_stream_usage_request_body = "{\"model\":\"usage-model-GGUF\",\"stream\":true,\"prompt\":\"stream usage\"}";
    raw = "POST /v1/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(completion_stream_usage_request_body.size()) + "\r\n\r\n" + completion_stream_usage_request_body;
    auto router_completion_stream_usage_proxy = request(raw);
    fake_completion_stream_usage_worker.join();
    expect(router_completion_stream_usage_proxy.find("200 OK") != std::string::npos, "router completion stream usage proxy status");
    expect(router_completion_stream_usage_proxy.find("cmpl-stream-usage") != std::string::npos, "router completion stream usage proxy body");

    const std::string timings_only_body =
        "{\"id\":\"cmpl-timings\",\"object\":\"text_completion\",\"model\":\"usage-model\","
        "\"choices\":[{\"text\":\"timings\",\"index\":0,\"finish_reason\":\"stop\"}],"
        "\"timings\":{\"prompt_n\":2,\"predicted_n\":5,\"prompt_ms\":100,\"predicted_ms\":250}}";
    std::thread fake_timings_worker(fake_json_worker_once, 18093, timings_only_body);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    raw = "POST /v1/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(router_completion_body.size()) + "\r\n\r\n" + router_completion_body;
    auto router_timings_proxy = request(raw);
    fake_timings_worker.join();
    expect(router_timings_proxy.find("200 OK") != std::string::npos, "router timings-only proxy status");

    std::thread fake_router_stream_worker(fake_json_worker_once, 18093, "data: router stream\n\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string router_stream_chat_body = "{\"model\":\"usage-model-GGUF\",\"stream\":\"false\",\"messages\":[{\"role\":\"user\",\"content\":\"stream\"}]}";
    raw = "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(router_stream_chat_body.size()) + "\r\n\r\n" + router_stream_chat_body;
    auto router_stream_chat_proxy = request(raw);
    fake_router_stream_worker.join();
    expect(router_stream_chat_proxy.find("200 OK") != std::string::npos, "router stream chat proxy status");
    expect(router_stream_chat_proxy.find("Content-Type: text/event-stream") != std::string::npos, "router stream chat proxy content type");
    expect(router_stream_chat_proxy.find("Transfer-Encoding: chunked") != std::string::npos, "router stream chat proxy chunked transfer");
    expect(router_stream_chat_proxy.find("data: router stream") != std::string::npos, "router stream chat proxy body");

    std::thread fake_chunked_router_stream_worker(
        fake_chunked_worker_once,
        18093,
        "text/event-stream",
        std::vector<std::string>{"data: chunked one\n\n", "data: chunked two\n\n"}
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    raw = "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(router_stream_chat_body.size()) + "\r\n\r\n" + router_stream_chat_body;
    auto router_chunked_stream_proxy = request(raw);
    fake_chunked_router_stream_worker.join();
    expect(router_chunked_stream_proxy.find("200 OK") != std::string::npos, "router chunked stream proxy status");
    expect(router_chunked_stream_proxy.find("Content-Type: text/event-stream") != std::string::npos, "router chunked stream proxy content type");
    expect(router_chunked_stream_proxy.find("Transfer-Encoding: chunked") != std::string::npos, "router chunked stream proxy chunked transfer");
    expect(router_chunked_stream_proxy.find("data: chunked one") != std::string::npos, "router chunked stream first chunk decoded");
    expect(router_chunked_stream_proxy.find("data: chunked two") != std::string::npos, "router chunked stream second chunk decoded");

    std::thread fake_delayed_router_stream_worker(
        fake_delayed_chunked_worker_once,
        18093,
        "text/event-stream",
        "data: early-stream\n\n",
        "data: late-stream\n\n",
        800
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    raw = "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(router_stream_chat_body.size()) + "\r\n\r\n" + router_stream_chat_body;
    long long router_first_stream_ms = 0;
    auto router_incremental_stream_proxy = request_observe_first_chunk(
        raw,
        "data: early-stream",
        &router_first_stream_ms
    );
    fake_delayed_router_stream_worker.join();
    expect(router_incremental_stream_proxy.find("200 OK") != std::string::npos, "router incremental stream status");
    expect(router_incremental_stream_proxy.find("Content-Type: text/event-stream") != std::string::npos, "router incremental stream content type");
    expect(router_incremental_stream_proxy.find("Transfer-Encoding: chunked") != std::string::npos, "router incremental stream chunked transfer");
    expect(router_incremental_stream_proxy.find("data: early-stream") != std::string::npos, "router incremental stream early chunk");
    expect(router_incremental_stream_proxy.find("data: late-stream") != std::string::npos, "router incremental stream late chunk");
    expect(router_first_stream_ms < 600, "router incremental stream emitted first chunk before upstream completed");

    const std::string register_embedding_worker_body =
        "{\"worker_id\":\"embedding-worker\","
        "\"label\":\"Embedding Worker\","
        "\"url\":\"http://127.0.0.1:18101\","
        "\"port\":18101,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"embedding\"],"
        "\"models\":[\"embed-model\"],"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_embedding_worker_body.size()) + "\r\n\r\n" + register_embedding_worker_body;
    auto registered_embedding_worker = request(raw);
    expect(registered_embedding_worker.find("200 OK") != std::string::npos, "router embedding worker register status");

    std::thread fake_embedding_worker(
        fake_expect_worker_once,
        18101,
        "POST /v1/embeddings HTTP/1.1",
        "Content-Type: application/json",
        "embed me",
        "{\"object\":\"list\",\"model\":\"embed-model\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.1]}],\"usage\":{\"prompt_tokens\":2,\"total_tokens\":2}}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string router_embedding_body = "{\"model\":\"embed-model\",\"input\":\"embed me\"}";
    raw = "POST /v1/embeddings HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(router_embedding_body.size()) + "\r\n\r\n" + router_embedding_body;
    auto router_embedding_proxy = request(raw);
    fake_embedding_worker.join();
    expect(router_embedding_proxy.find("200 OK") != std::string::npos, "router embedding proxy status");
    expect(router_embedding_proxy.find("\"embedding\":[0.1]") != std::string::npos, "router embedding proxy body");

    const char *saved_ssl_cert_file_raw = getenv("SSL_CERT_FILE");
    const bool had_ssl_cert_file = saved_ssl_cert_file_raw != nullptr;
    const std::string saved_ssl_cert_file = saved_ssl_cert_file_raw ? saved_ssl_cert_file_raw : "";
    const auto tls_cert = ensure_test_tls_cert();
    setenv("SSL_CERT_FILE", tls_cert.cert.c_str(), 1);

    const std::string register_https_embedding_worker_body =
        "{\"worker_id\":\"https-embedding-worker\","
        "\"label\":\"HTTPS Embedding Worker\","
        "\"url\":\"https://127.0.0.2:18159\","
        "\"port\":18159,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"embedding\"],"
        "\"models\":[\"https-embed-model\"],"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_https_embedding_worker_body.size()) + "\r\n\r\n" + register_https_embedding_worker_body;
    auto registered_https_embedding_worker = request(raw);
    expect(registered_https_embedding_worker.find("200 OK") != std::string::npos, "router https embedding worker register status");
    TestListenerReady https_embedding_ready(18159);
    std::thread fake_https_embedding_worker(
        fake_https_json_worker_once,
        18159,
        "POST /v1/embeddings HTTP/1.1",
        "Authorization: Bearer worker-secret",
        "{\"object\":\"list\",\"model\":\"https-embed-model\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.2]}],\"usage\":{\"prompt_tokens\":1,\"total_tokens\":1}}",
        &https_embedding_ready
    );
    https_embedding_ready.wait();
    const std::string router_https_embedding_body = "{\"model\":\"https-embed-model\",\"input\":\"https embed\"}";
    raw = "POST /v1/embeddings HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(router_https_embedding_body.size()) + "\r\n\r\n" + router_https_embedding_body;
    auto router_https_embedding_proxy = request(raw);
    fake_https_embedding_worker.join();
    expect(router_https_embedding_proxy.find("200 OK") != std::string::npos, "router https embedding proxy status");
    expect(router_https_embedding_proxy.find("\"embedding\":[0.2]") != std::string::npos, "router https embedding proxy body");
    const std::string deregister_https_embedding_worker_body = "{\"worker_id\":\"https-embedding-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_https_embedding_worker_body.size()) + "\r\n\r\n" + deregister_https_embedding_worker_body;
    auto deregistered_https_embedding_worker = request(raw);
    expect(deregistered_https_embedding_worker.find("200 OK") != std::string::npos, "router https embedding worker deregister status");

    const std::string register_tts_worker_body =
        "{\"worker_id\":\"tts-worker\","
        "\"label\":\"TTS Worker\","
        "\"url\":\"http://127.0.0.1:18095\","
        "\"port\":18095,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"tts\"],"
        "\"models\":[],"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_tts_worker_body.size()) + "\r\n\r\n" + register_tts_worker_body;
    auto registered_tts_worker = request(raw);
    expect(registered_tts_worker.find("200 OK") != std::string::npos, "router tts worker register status");

    std::thread fake_voice_worker(
        fake_get_json_worker_once_no_content_length,
        18095,
        "GET /v1/audio/voices HTTP/1.1",
        "{\"voices\":[\"amy\",{\"id\":\"bob\",\"name\":\"Bob\"}]}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto router_voice_list = request("GET /v1/audio/voices HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    fake_voice_worker.join();
    expect(router_voice_list.find("200 OK") != std::string::npos, "router voices aggregate status");
    expect(router_voice_list.find("\"voices\":[\"amy\",{\"id\":\"bob\",\"name\":\"Bob\"}]") != std::string::npos, "router voices aggregate body");
    const std::string deregister_tts_worker_body = "{\"worker_id\":\"tts-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_tts_worker_body.size()) + "\r\n\r\n" + deregister_tts_worker_body;
    auto deregistered_tts_worker = request(raw);
    expect(deregistered_tts_worker.find("200 OK") != std::string::npos, "router tts worker deregister status");

    const std::string register_https_tts_worker_body =
        "{\"worker_id\":\"https-tts-worker\","
        "\"label\":\"HTTPS TTS Worker\","
        "\"url\":\"https://127.0.0.2:18160\","
        "\"port\":18160,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"tts\"],"
        "\"models\":[],"
        "\"free_vram_gb\":999,"
        "\"best_tier\":999,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_https_tts_worker_body.size()) + "\r\n\r\n" + register_https_tts_worker_body;
    auto registered_https_tts_worker = request(raw);
    expect(registered_https_tts_worker.find("200 OK") != std::string::npos, "router https tts worker register status");
    // Exercise startup beyond the former fixed 50 ms sleep.
    TestListenerReady https_voice_ready(18160, std::chrono::milliseconds(100));
    std::thread fake_https_voice_worker(
        fake_https_json_worker_once,
        18160,
        "GET /v1/audio/voices HTTP/1.1",
        "Authorization: Bearer worker-secret",
        "{\"voices\":[\"tls-amy\"]}",
        &https_voice_ready
    );
    https_voice_ready.wait();
    auto router_https_voice_list = request("GET /v1/audio/voices HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    if (router_https_voice_list.find("\"voices\":[\"tls-amy\"]") == std::string::npos)
        std::cerr << "Unexpected HTTPS voices response before fixture join: " << router_https_voice_list << '\n';
    fake_https_voice_worker.join();
    expect(router_https_voice_list.find("200 OK") != std::string::npos, "router https voices aggregate status");
    expect(router_https_voice_list.find("\"voices\":[\"tls-amy\"]") != std::string::npos, "router https voices aggregate body");
    const std::string deregister_https_tts_worker_body = "{\"worker_id\":\"https-tts-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_https_tts_worker_body.size()) + "\r\n\r\n" + deregister_https_tts_worker_body;
    auto deregistered_https_tts_worker = request(raw);
    expect(deregistered_https_tts_worker.find("200 OK") != std::string::npos, "router https tts worker deregister status");

    const std::string register_https_tts_speech_worker_body =
        "{\"worker_id\":\"https-tts-speech-worker\","
        "\"label\":\"HTTPS TTS Speech Worker\","
        "\"url\":\"https://127.0.0.2:18164\","
        "\"port\":18164,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"tts\"],"
        "\"models\":[],"
        "\"free_vram_gb\":999,"
        "\"best_tier\":999,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_https_tts_speech_worker_body.size()) + "\r\n\r\n" + register_https_tts_speech_worker_body;
    auto registered_https_tts_speech_worker = request(raw);
    expect(registered_https_tts_speech_worker.find("200 OK") != std::string::npos, "router https tts speech worker register status");
    TestListenerReady https_tts_ready(18164);
    std::thread fake_https_tts_speech_worker(
        fake_https_json_worker_once,
        18164,
        "POST /v1/audio/speech HTTP/1.1",
        "Authorization: Bearer worker-secret",
        "RIFFROUTERHTTPSTTS",
        &https_tts_ready
    );
    https_tts_ready.wait();
    const std::string router_https_tts_body = "{\"model\":\"https-tts-model\",\"input\":\"tls say it\",\"voice\":\"amy\"}";
    raw = "POST /v1/audio/speech HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(router_https_tts_body.size()) + "\r\n\r\n" + router_https_tts_body;
    auto router_https_tts_proxy = request(raw);
    fake_https_tts_speech_worker.join();
    expect(router_https_tts_proxy.find("200 OK") != std::string::npos, "router https tts proxy status");
    expect(router_https_tts_proxy.find("RIFFROUTERHTTPSTTS") != std::string::npos, "router https tts proxy body");
    const std::string deregister_https_tts_speech_worker_body = "{\"worker_id\":\"https-tts-speech-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_https_tts_speech_worker_body.size()) + "\r\n\r\n" + deregister_https_tts_speech_worker_body;
    auto deregistered_https_tts_speech_worker = request(raw);
    expect(deregistered_https_tts_speech_worker.find("200 OK") != std::string::npos, "router https tts speech worker deregister status");
    if (had_ssl_cert_file) {
        setenv("SSL_CERT_FILE", saved_ssl_cert_file.c_str(), 1);
    } else {
        unsetenv("SSL_CERT_FILE");
    }

    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_tts_worker_body.size()) + "\r\n\r\n" + register_tts_worker_body;
    auto reregistered_tts_worker = request(raw);
    expect(reregistered_tts_worker.find("200 OK") != std::string::npos, "router tts worker reregister status");

    std::thread fake_voice_worker_no_id(
        fake_json_worker_once,
        18095,
        "{\"voices\":[{\"name\":\"No ID A\"},{\"name\":\"No ID B\"}]}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto router_voice_no_id = request("GET /v1/audio/voices HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    fake_voice_worker_no_id.join();
    expect(router_voice_no_id.find("200 OK") != std::string::npos, "router voices no-id status");
    expect(router_voice_no_id.find("\"voices\":[{\"name\":\"No ID A\"}]") != std::string::npos, "router voices no-id dedupe");
    expect(router_voice_no_id.find("No ID B") == std::string::npos, "router voices no-id skips second");

    std::thread fake_voice_worker_unhashable_id(
        fake_json_worker_once,
        18095,
        "{\"voices\":[\"good-before-bad\",{\"id\":[],\"name\":\"Bad ID\"},\"after-bad\"]}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto router_voice_unhashable_id = request("GET /v1/audio/voices HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    fake_voice_worker_unhashable_id.join();
    expect(router_voice_unhashable_id.find("200 OK") != std::string::npos, "router voices unhashable id status");
    expect(router_voice_unhashable_id.find("\"voices\":[\"good-before-bad\"]") != std::string::npos, "router voices unhashable id keeps earlier voices");
    expect(router_voice_unhashable_id.find("Bad ID") == std::string::npos, "router voices unhashable id skips bad voice");
    expect(router_voice_unhashable_id.find("after-bad") == std::string::npos, "router voices unhashable id stops worker");

    std::thread fake_voice_worker_object_payload(
        fake_json_worker_once,
        18095,
        "{\"voices\":{\"dict-a\":1,\"dict-b\":2}}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto router_voice_object_payload = request("GET /v1/audio/voices HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    fake_voice_worker_object_payload.join();
    expect(router_voice_object_payload.find("200 OK") != std::string::npos, "router voices object payload status");
    expect(router_voice_object_payload.find("\"voices\":[\"dict-a\",\"dict-b\"]") != std::string::npos, "router voices object payload iterates keys");

    std::thread fake_voice_worker_string_payload(
        fake_json_worker_once,
        18095,
        "{\"voices\":\"xy\"}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto router_voice_string_payload = request("GET /v1/audio/voices HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    fake_voice_worker_string_payload.join();
    expect(router_voice_string_payload.find("200 OK") != std::string::npos, "router voices string payload status");
    expect(router_voice_string_payload.find("\"voices\":[\"x\",\"y\"]") != std::string::npos, "router voices string payload iterates chars");

    std::thread fake_voice_worker_scalar_payload(
        fake_json_worker_once,
        18095,
        "{\"voices\":123,\"data\":[\"fallback-not-used\"]}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto router_voice_scalar_payload = request("GET /v1/audio/voices HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    fake_voice_worker_scalar_payload.join();
    expect(router_voice_scalar_payload.find("200 OK") != std::string::npos, "router voices scalar payload status");
    expect(router_voice_scalar_payload.find("\"voices\":[]") != std::string::npos, "router voices scalar payload mirrors python exception");
    expect(router_voice_scalar_payload.find("fallback-not-used") == std::string::npos, "router voices scalar payload skips data fallback");

    const std::string register_voice_transport_error_worker_body =
        "{\"worker_id\":\"voice-transport-error-worker\","
        "\"label\":\"Voice Transport Error\","
        "\"url\":\"http://127.0.0.1:18170\","
        "\"port\":18170,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"tts\"],"
        "\"models\":[],"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_voice_transport_error_worker_body.size()) + "\r\n\r\n" + register_voice_transport_error_worker_body;
    auto registered_voice_transport_error = request(raw);
    expect(registered_voice_transport_error.find("200 OK") != std::string::npos, "router voice transport error worker register status");
    std::thread fake_voice_worker_transport_keep_aggregate(
        fake_json_worker_once,
        18095,
        "{\"voices\":[\"still-aggregated\"]}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto router_voice_transport_error = request("GET /v1/audio/voices HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    fake_voice_worker_transport_keep_aggregate.join();
    expect(router_voice_transport_error.find("200 OK") != std::string::npos, "router voices transport error status");
    expect(router_voice_transport_error.find("\"voices\":[\"still-aggregated\"]") != std::string::npos, "router voices transport error keeps other voices");
    auto voice_transport_error_workers = request("GET /v1/router/workers HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    auto voice_transport_error_pos = voice_transport_error_workers.find("\"worker_id\":\"voice-transport-error-worker\"");
    auto voice_transport_next_pos = voice_transport_error_workers.find("\"worker_id\":", voice_transport_error_pos + 1);
    auto voice_transport_failures_pos = voice_transport_error_workers.find("\"connection_failures\":1", voice_transport_error_pos);
    auto voice_transport_errors_pos = voice_transport_error_workers.find("\"total_errors\":1", voice_transport_error_pos);
    auto voice_transport_kind_pos = voice_transport_error_workers.find("\"kind\":\"connection\"", voice_transport_error_pos);
    expect(voice_transport_error_pos != std::string::npos
        && voice_transport_failures_pos != std::string::npos
        && voice_transport_errors_pos != std::string::npos
        && voice_transport_kind_pos != std::string::npos
        && (voice_transport_next_pos == std::string::npos
            || (voice_transport_failures_pos < voice_transport_next_pos
                && voice_transport_errors_pos < voice_transport_next_pos
                && voice_transport_kind_pos < voice_transport_next_pos)),
        "router voices transport error records proxy failure");
    const std::string deregister_voice_transport_error_worker_body = "{\"worker_id\":\"voice-transport-error-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_voice_transport_error_worker_body.size()) + "\r\n\r\n" + deregister_voice_transport_error_worker_body;
    auto deregistered_voice_transport_error = request(raw);
    expect(deregistered_voice_transport_error.find("200 OK") != std::string::npos, "router voice transport error worker deregister status");

    const std::string register_voice_tunnel_missing_worker_body =
        "{\"worker_id\":\"voice-tunnel-missing-worker\","
        "\"label\":\"Voice Tunnel Missing\","
        "\"url\":\"tunnel://voice-missing-node\","
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"tts\"],"
        "\"models\":[],"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_voice_tunnel_missing_worker_body.size()) + "\r\n\r\n" + register_voice_tunnel_missing_worker_body;
    auto registered_voice_tunnel_missing = request(raw);
    expect(registered_voice_tunnel_missing.find("200 OK") != std::string::npos, "router voice missing tunnel worker register status");
    std::thread fake_voice_worker_tunnel_missing_keep_aggregate(
        fake_json_worker_once,
        18095,
        "{\"voices\":[\"tunnel-healthy\"]}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto router_voice_tunnel_missing = request("GET /v1/audio/voices HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    fake_voice_worker_tunnel_missing_keep_aggregate.join();
    expect(router_voice_tunnel_missing.find("200 OK") != std::string::npos, "router voices missing tunnel status");
    expect(router_voice_tunnel_missing.find("\"voices\":[\"tunnel-healthy\"]") != std::string::npos, "router voices missing tunnel keeps other voices");
    auto voice_tunnel_missing_errors = request("GET /v1/router/errors HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    auto voice_tunnel_missing_pos = voice_tunnel_missing_errors.find("\"worker_id\":\"voice-tunnel-missing-worker\"");
    auto voice_tunnel_missing_next_pos = voice_tunnel_missing_errors.find("\"worker_id\":", voice_tunnel_missing_pos + 1);
    auto voice_tunnel_missing_kind_pos = voice_tunnel_missing_errors.find("\"kind\":\"tunnel_disconnect\"", voice_tunnel_missing_pos);
    auto voice_tunnel_missing_status_pos = voice_tunnel_missing_errors.find("\"status\":503", voice_tunnel_missing_pos);
    auto voice_tunnel_missing_message_pos = voice_tunnel_missing_errors.find("\"message\":\"Tunnel voice-missing-node not connected\"", voice_tunnel_missing_pos);
    expect(voice_tunnel_missing_pos != std::string::npos
        && voice_tunnel_missing_kind_pos != std::string::npos
        && voice_tunnel_missing_status_pos != std::string::npos
        && voice_tunnel_missing_message_pos != std::string::npos
        && (voice_tunnel_missing_next_pos == std::string::npos
            || (voice_tunnel_missing_kind_pos < voice_tunnel_missing_next_pos
                && voice_tunnel_missing_status_pos < voice_tunnel_missing_next_pos
                && voice_tunnel_missing_message_pos < voice_tunnel_missing_next_pos)),
        "router errors records pre-request tunnel disconnect like python");
    const std::string deregister_voice_tunnel_missing_worker_body = "{\"worker_id\":\"voice-tunnel-missing-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_voice_tunnel_missing_worker_body.size()) + "\r\n\r\n" + deregister_voice_tunnel_missing_worker_body;
    auto deregistered_voice_tunnel_missing = request(raw);
    expect(deregistered_voice_tunnel_missing.find("200 OK") != std::string::npos, "router voice missing tunnel worker deregister status");

    const std::string register_voice_order_a_body =
        "{\"worker_id\":\"voice-order-a\","
        "\"label\":\"Voice Order A\","
        "\"url\":\"http://127.0.0.1:18123\","
        "\"port\":18123,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"tts\"],"
        "\"models\":[],"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_voice_order_a_body.size()) + "\r\n\r\n" + register_voice_order_a_body;
    auto registered_voice_order_a = request(raw);
    expect(registered_voice_order_a.find("200 OK") != std::string::npos, "router voice order a register status");

    const std::string register_voice_order_b_body =
        "{\"worker_id\":\"voice-order-b\","
        "\"label\":\"Voice Order B\","
        "\"url\":\"http://127.0.0.1:18124\","
        "\"port\":18124,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"tts\"],"
        "\"models\":[],"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_voice_order_b_body.size()) + "\r\n\r\n" + register_voice_order_b_body;
    auto registered_voice_order_b = request(raw);
    expect(registered_voice_order_b.find("200 OK") != std::string::npos, "router voice order b register status");

    std::thread fake_voice_order_a(fake_json_worker_once, 18123, "{\"voices\":[\"zeta\",\"common\"]}");
    std::thread fake_voice_order_b(fake_json_worker_once, 18124, "{\"voices\":[\"alpha\",\"common\",\"omega\"]}");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto router_voice_order = request("GET /v1/audio/voices HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    fake_voice_order_a.join();
    fake_voice_order_b.join();
    expect(router_voice_order.find("200 OK") != std::string::npos, "router voices order status");
    expect(router_voice_order.find("\"voices\":[\"zeta\",\"common\",\"alpha\",\"omega\"]") != std::string::npos, "router voices preserve first-seen order");

    const std::string register_stt_worker_body =
        "{\"worker_id\":\"stt-worker\","
        "\"label\":\"STT Worker\","
        "\"url\":\"http://127.0.0.1:18097\","
        "\"port\":18097,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"stt\"],"
        "\"models\":[],"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_stt_worker_body.size()) + "\r\n\r\n" + register_stt_worker_body;
    auto registered_stt_worker = request(raw);
    expect(registered_stt_worker.find("200 OK") != std::string::npos, "router stt worker register status");

    const std::string stt_boundary = "native-stt-boundary";
    const std::string stt_body =
        "--" + stt_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"model\"\r\n\r\n"
        "old-whisper\r\n"
        "--" + stt_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"model\"\r\n\r\n"
        "whisper-1\r\n"
        "--" + stt_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"temperature\"\r\n\r\n"
        "1\r\n"
        "--" + stt_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"timestamps\"\r\n\r\n"
        "true\r\n"
        "--" + stt_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"enable_diarization\"\r\n\r\n"
        "maybe\r\n"
        "--" + stt_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "AUDIO-DATA\r\n"
        "--" + stt_boundary + "--\r\n";
    std::thread fake_stt_worker(
        fake_expect_worker_once_with_absence,
        18097,
        "POST /v1/audio/transcriptions HTTP/1.1",
        "Content-Type: multipart/form-data; boundary=ezlocalai-native-voice-boundary",
        "filename=\"sample.wav\"\r\nContent-Type: audio/wav\r\n\r\nAUDIO-DATA",
        "name=\"response_format\"\r\n\r\njson",
        "name=\"temperature\"\r\n\r\n1.0",
        "name=\"timestamps\"\r\n\r\nTrue",
        "name=\"model\"\r\n\r\nwhisper-1",
        "enable_diarization",
        "old-whisper",
        "{\"text\":\"proxied transcript\"}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    raw = "POST /v1/audio/transcriptions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + stt_boundary + "\r\n"
        "Content-Length: " + std::to_string(stt_body.size()) + "\r\n\r\n" + stt_body;
    auto router_stt_proxy = request(raw);
    fake_stt_worker.join();
    expect(router_stt_proxy.find("200 OK") != std::string::npos, "router stt multipart proxy status");
    expect(router_stt_proxy.find("proxied transcript") != std::string::npos, "router stt multipart proxy body");

    {
        const char *stt_saved_ssl_cert_file_raw = getenv("SSL_CERT_FILE");
        const bool stt_had_ssl_cert_file = stt_saved_ssl_cert_file_raw != nullptr;
        const std::string stt_saved_ssl_cert_file = stt_saved_ssl_cert_file_raw ? stt_saved_ssl_cert_file_raw : "";
        const auto stt_tls_cert = ensure_test_tls_cert();
        setenv("SSL_CERT_FILE", stt_tls_cert.cert.c_str(), 1);
        const std::string register_https_stt_worker_body =
            "{\"worker_id\":\"https-stt-worker\","
            "\"label\":\"HTTPS STT Worker\","
            "\"url\":\"https://127.0.0.2:18166\","
            "\"port\":18166,"
            "\"api_key\":\"worker-secret\","
            "\"capabilities\":[\"stt\"],"
            "\"models\":[\"https-whisper\"],"
            "\"free_vram_gb\":998,"
            "\"best_tier\":998,"
            "\"queue_capacity\":1}";
        raw = "POST /v1/router/register HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(register_https_stt_worker_body.size()) + "\r\n\r\n" + register_https_stt_worker_body;
        auto registered_https_stt_worker = request(raw);
        expect(registered_https_stt_worker.find("200 OK") != std::string::npos, "router https stt worker register status");

        const std::string https_stt_boundary = "native-https-stt-boundary";
        const std::string https_stt_body =
            "--" + https_stt_boundary + "\r\n"
            "Content-Disposition: form-data; name=\"model\"\r\n\r\n"
            "old-https-whisper\r\n"
            "--" + https_stt_boundary + "\r\n"
            "Content-Disposition: form-data; name=\"model\"\r\n\r\n"
            "https-whisper\r\n"
            "--" + https_stt_boundary + "\r\n"
            "Content-Disposition: form-data; name=\"temperature\"\r\n\r\n"
            "0.25\r\n"
            "--" + https_stt_boundary + "\r\n"
            "Content-Disposition: form-data; name=\"timestamps\"\r\n\r\n"
            "false\r\n"
            "--" + https_stt_boundary + "\r\n"
            "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
            "Content-Type: audio/wav\r\n\r\n"
            "HTTPS-AUDIO\r\n"
            "--" + https_stt_boundary + "--\r\n";
        TestListenerReady https_stt_ready(18166);
        std::thread fake_https_stt_worker(
            fake_https_expect_worker_once_with_absence,
            18166,
            "POST /v1/audio/transcriptions HTTP/1.1",
            "Authorization: Bearer worker-secret",
            std::vector<std::string>{
                "Content-Type: multipart/form-data; boundary=ezlocalai-native-voice-boundary",
                "filename=\"sample.wav\"\r\nContent-Type: audio/wav\r\n\r\nHTTPS-AUDIO",
                "name=\"response_format\"\r\n\r\njson",
                "name=\"temperature\"\r\n\r\n0.25",
                "name=\"timestamps\"\r\n\r\nFalse",
                "name=\"model\"\r\n\r\nhttps-whisper",
            },
            std::vector<std::string>{"old-https-whisper"},
            "{\"text\":\"https transcript\"}",
            &https_stt_ready
        );
        https_stt_ready.wait();
        raw = "POST /v1/audio/transcriptions HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: multipart/form-data; boundary=" + https_stt_boundary + "\r\n"
            "Content-Length: " + std::to_string(https_stt_body.size()) + "\r\n\r\n" + https_stt_body;
        auto router_https_stt_proxy = request(raw);
        fake_https_stt_worker.join();
        expect(router_https_stt_proxy.find("200 OK") != std::string::npos, "router https stt multipart proxy status");
        expect(router_https_stt_proxy.find("https transcript") != std::string::npos, "router https stt multipart proxy body");
        const std::string deregister_https_stt_worker_body = "{\"worker_id\":\"https-stt-worker\"}";
        raw = "POST /v1/router/deregister HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(deregister_https_stt_worker_body.size()) + "\r\n\r\n" + deregister_https_stt_worker_body;
        auto deregistered_https_stt_worker = request(raw);
        expect(deregistered_https_stt_worker.find("200 OK") != std::string::npos, "router https stt worker deregister status");
        if (stt_had_ssl_cert_file) {
            setenv("SSL_CERT_FILE", stt_saved_ssl_cert_file.c_str(), 1);
        } else {
            unsetenv("SSL_CERT_FILE");
        }
    }

    const std::string register_image_worker_body =
        "{\"worker_id\":\"image-worker\","
        "\"label\":\"Image Worker\","
        "\"url\":\"http://127.0.0.1:18098\","
        "\"port\":18098,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"image\"],"
        "\"models\":[],"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_image_worker_body.size()) + "\r\n\r\n" + register_image_worker_body;
    auto registered_image_worker = request(raw);
    expect(registered_image_worker.find("200 OK") != std::string::npos, "router image worker register status");

    const std::string edit_boundary = "native-edit-boundary";
    const std::string edit_body =
        "--" + edit_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"model\"\r\n\r\n"
        "old-model\r\n"
        "--" + edit_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"model\"\r\n\r\n"
        "image-model\r\n"
        "--" + edit_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"image\"; filename=\"\"\r\n"
        "Content-Type: image/png\r\n\r\n"
        "PNG-DATA\r\n"
        "--" + edit_boundary + "--\r\n";
    std::thread fake_image_worker(
        fake_expect_worker_once_with_absence,
        18098,
        "POST /v1/images/edits HTTP/1.1",
        "Content-Type: multipart/form-data; boundary=ezlocalai-native-voice-boundary",
        "Content-Disposition: form-data; name=\"image\"; filename=\"file\"",
        "name=\"model\"\r\n\r\nimage-model",
        "PNG-DATA",
        "",
        "",
        "boundary=" + edit_boundary,
        "old-model",
        "{\"created\":1,\"data\":[{\"url\":\"worker-image.png\"}]}"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    raw = "POST /v1/images/edits HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: multipart/form-data; boundary=" + edit_boundary + "\r\n"
        "Content-Length: " + std::to_string(edit_body.size()) + "\r\n\r\n" + edit_body;
    auto router_image_edit_proxy = request(raw);
    fake_image_worker.join();
    expect(router_image_edit_proxy.find("200 OK") != std::string::npos, "router image edit multipart proxy status");
    expect(router_image_edit_proxy.find("worker-image.png") != std::string::npos, "router image edit multipart proxy body");

    {
        const char *image_saved_ssl_cert_file_raw = getenv("SSL_CERT_FILE");
        const bool image_had_ssl_cert_file = image_saved_ssl_cert_file_raw != nullptr;
        const std::string image_saved_ssl_cert_file = image_saved_ssl_cert_file_raw ? image_saved_ssl_cert_file_raw : "";
        const auto image_tls_cert = ensure_test_tls_cert();
        setenv("SSL_CERT_FILE", image_tls_cert.cert.c_str(), 1);
        const std::string register_https_image_worker_body =
            "{\"worker_id\":\"https-image-worker\","
            "\"label\":\"HTTPS Image Worker\","
            "\"url\":\"https://127.0.0.2:18165\","
            "\"port\":18165,"
            "\"api_key\":\"worker-secret\","
            "\"capabilities\":[\"image\"],"
            "\"models\":[],"
            "\"free_vram_gb\":999,"
            "\"best_tier\":999,"
            "\"queue_capacity\":1}";
        raw = "POST /v1/router/register HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(register_https_image_worker_body.size()) + "\r\n\r\n" + register_https_image_worker_body;
        auto registered_https_image_worker = request(raw);
        expect(registered_https_image_worker.find("200 OK") != std::string::npos, "router https image worker register status");
        TestListenerReady https_image_ready(18165);
        std::thread fake_https_image_worker(
            fake_https_json_worker_once,
            18165,
            "POST /v1/images/generations HTTP/1.1",
            "Authorization: Bearer worker-secret",
            "{\"created\":22,\"data\":[{\"url\":\"https-image.png\"}]}",
            &https_image_ready
        );
        https_image_ready.wait();
        const std::string https_image_generation_body = "{\"model\":\"https-image-model\",\"prompt\":\"tls image\"}";
        raw = "POST /v1/images/generations HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(https_image_generation_body.size()) + "\r\n\r\n" + https_image_generation_body;
        auto router_https_image_proxy = request(raw);
        fake_https_image_worker.join();
        expect(router_https_image_proxy.find("200 OK") != std::string::npos, "router https image proxy status");
        expect(router_https_image_proxy.find("https-image.png") != std::string::npos, "router https image proxy body");
        const std::string register_https_image_edit_worker_body =
            "{\"worker_id\":\"https-image-edit-worker\","
            "\"label\":\"HTTPS Image Edit Worker\","
            "\"url\":\"https://127.0.0.2:18167\","
            "\"port\":18167,"
            "\"api_key\":\"worker-secret\","
            "\"capabilities\":[\"image\"],"
            "\"models\":[\"https-edit-model\"],"
            "\"free_vram_gb\":1000,"
            "\"best_tier\":1000,"
            "\"queue_capacity\":1}";
        raw = "POST /v1/router/register HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(register_https_image_edit_worker_body.size()) + "\r\n\r\n" + register_https_image_edit_worker_body;
        auto registered_https_image_edit_worker = request(raw);
        expect(registered_https_image_edit_worker.find("200 OK") != std::string::npos, "router https image edit worker register status");

        const std::string https_edit_boundary = "native-https-edit-boundary";
        const std::string https_edit_body =
            "--" + https_edit_boundary + "\r\n"
            "Content-Disposition: form-data; name=\"model\"\r\n\r\n"
            "old-https-edit-model\r\n"
            "--" + https_edit_boundary + "\r\n"
            "Content-Disposition: form-data; name=\"model\"\r\n\r\n"
            "https-edit-model\r\n"
            "--" + https_edit_boundary + "\r\n"
            "Content-Disposition: form-data; name=\"prompt\"\r\n\r\n"
            "make it sharp\r\n"
            "--" + https_edit_boundary + "\r\n"
            "Content-Disposition: form-data; name=\"image\"; filename=\"\"\r\n"
            "Content-Type: image/png\r\n\r\n"
            "HTTPS-PNG-DATA\r\n"
            "--" + https_edit_boundary + "--\r\n";
        TestListenerReady https_edit_ready(18167);
        std::thread fake_https_image_edit_worker(
            fake_https_expect_worker_once_with_absence,
            18167,
            "POST /v1/images/edits HTTP/1.1",
            "Authorization: Bearer worker-secret",
            std::vector<std::string>{
                "Content-Type: multipart/form-data; boundary=ezlocalai-native-voice-boundary",
                "Content-Disposition: form-data; name=\"image\"; filename=\"file\"",
                "Content-Type: image/png\r\n\r\nHTTPS-PNG-DATA",
                "name=\"model\"\r\n\r\nhttps-edit-model",
                "name=\"prompt\"\r\n\r\nmake it sharp",
            },
            std::vector<std::string>{"old-https-edit-model", "boundary=" + https_edit_boundary},
            "{\"created\":23,\"data\":[{\"url\":\"https-edit.png\"}]}",
            &https_edit_ready
        );
        https_edit_ready.wait();
        raw = "POST /v1/images/edits HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: multipart/form-data; boundary=" + https_edit_boundary + "\r\n"
            "Content-Length: " + std::to_string(https_edit_body.size()) + "\r\n\r\n" + https_edit_body;
        auto router_https_image_edit_proxy = request(raw);
        fake_https_image_edit_worker.join();
        expect(router_https_image_edit_proxy.find("200 OK") != std::string::npos, "router https image edit proxy status");
        expect(router_https_image_edit_proxy.find("https-edit.png") != std::string::npos, "router https image edit proxy body");
        const std::string deregister_https_image_edit_worker_body = "{\"worker_id\":\"https-image-edit-worker\"}";
        raw = "POST /v1/router/deregister HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(deregister_https_image_edit_worker_body.size()) + "\r\n\r\n" + deregister_https_image_edit_worker_body;
        auto deregistered_https_image_edit_worker = request(raw);
        expect(deregistered_https_image_edit_worker.find("200 OK") != std::string::npos, "router https image edit worker deregister status");
        const std::string deregister_https_image_worker_body = "{\"worker_id\":\"https-image-worker\"}";
        raw = "POST /v1/router/deregister HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(deregister_https_image_worker_body.size()) + "\r\n\r\n" + deregister_https_image_worker_body;
        auto deregistered_https_image_worker = request(raw);
        expect(deregistered_https_image_worker.find("200 OK") != std::string::npos, "router https image worker deregister status");
        if (image_had_ssl_cert_file) {
            setenv("SSL_CERT_FILE", image_saved_ssl_cert_file.c_str(), 1);
        } else {
            unsetenv("SSL_CERT_FILE");
        }
    }

    {
        const char *video_saved_ssl_cert_file_raw = getenv("SSL_CERT_FILE");
        const bool video_had_ssl_cert_file = video_saved_ssl_cert_file_raw != nullptr;
        const std::string video_saved_ssl_cert_file = video_saved_ssl_cert_file_raw ? video_saved_ssl_cert_file_raw : "";
        const auto video_tls_cert = ensure_test_tls_cert();
        setenv("SSL_CERT_FILE", video_tls_cert.cert.c_str(), 1);
        const std::string register_https_video_worker_body =
            "{\"worker_id\":\"https-video-worker\","
            "\"label\":\"HTTPS Video Worker\","
            "\"url\":\"https://127.0.0.2:18169\","
            "\"port\":18169,"
            "\"api_key\":\"worker-secret\","
            "\"capabilities\":[\"video\"],"
            "\"models\":[\"https-video-model\"],"
            "\"free_vram_gb\":1001,"
            "\"best_tier\":1001,"
            "\"queue_capacity\":1}";
        raw = "POST /v1/router/register HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(register_https_video_worker_body.size()) + "\r\n\r\n" + register_https_video_worker_body;
        auto registered_https_video_worker = request(raw);
        expect(registered_https_video_worker.find("200 OK") != std::string::npos, "router https video worker register status");
        TestListenerReady https_video_ready(18169);
        std::thread fake_https_video_worker(
            fake_https_json_worker_once,
            18169,
            "POST /v1/videos/generations HTTP/1.1",
            "Authorization: Bearer worker-secret",
            "{\"created\":24,\"data\":[{\"url\":\"https-video.mp4\"}]}",
            &https_video_ready
        );
        https_video_ready.wait();
        const std::string https_video_body = "{\"model\":\"https-video-model\",\"prompt\":\"tls video\"}";
        raw = "POST /v1/videos/generations HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(https_video_body.size()) + "\r\n\r\n" + https_video_body;
        auto router_https_video_proxy = request(raw);
        fake_https_video_worker.join();
        expect(router_https_video_proxy.find("200 OK") != std::string::npos, "router https video proxy status");
        expect(router_https_video_proxy.find("https-video.mp4") != std::string::npos, "router https video proxy body");
        const std::string deregister_https_video_worker_body = "{\"worker_id\":\"https-video-worker\"}";
        raw = "POST /v1/router/deregister HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(deregister_https_video_worker_body.size()) + "\r\n\r\n" + deregister_https_video_worker_body;
        auto deregistered_https_video_worker = request(raw);
        expect(deregistered_https_video_worker.find("200 OK") != std::string::npos, "router https video worker deregister status");
        if (video_had_ssl_cert_file) {
            setenv("SSL_CERT_FILE", video_saved_ssl_cert_file.c_str(), 1);
        } else {
            unsetenv("SSL_CERT_FILE");
        }
    }

    const std::string register_asset_worker_body =
        "{\"worker_id\":\"asset-worker\","
        "\"label\":\"Asset Worker\","
        "\"url\":\"http://127.0.0.1:18099\","
        "\"port\":18099,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"image\"],"
        "\"models\":[],"
        "\"free_vram_gb\":99,"
        "\"best_tier\":120,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_asset_worker_body.size()) + "\r\n\r\n" + register_asset_worker_body;
    auto registered_asset_worker = request(raw);
    expect(registered_asset_worker.find("200 OK") != std::string::npos, "router asset worker register status");

    const std::string asset_worker_url = "http://127.0.0.1:18099/outputs/generated.png";
    const std::string asset_response_body = "{\"created\":1,\"data\":[{\"url\":\"" + asset_worker_url + "\"}]}";
    std::thread fake_asset_worker(
        fake_asset_worker_once,
        18099,
        "/v1/images/generations",
        asset_response_body,
        "/outputs/generated.png",
        "PNG-ASSET"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string asset_generation_body = "{\"model\":\"asset-model\",\"prompt\":\"asset\"}";
    raw = "POST /v1/images/generations HTTP/1.1\r\n"
        "Host: localhost:18091\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(asset_generation_body.size()) + "\r\n\r\n" + asset_generation_body;
    auto router_asset_proxy = request(raw);
    fake_asset_worker.join();
    expect(router_asset_proxy.find("200 OK") != std::string::npos, "router asset proxy status");
    expect(router_asset_proxy.find(asset_worker_url) == std::string::npos, "router asset proxy rewrites worker url");
    expect(router_asset_proxy.find("\"url\":\"http://localhost:18091/outputs/generated.png\"") != std::string::npos, "router asset proxy request-base url");
    expect(std::filesystem::exists("outputs/generated.png"), "router asset file saved");
    {
        std::ifstream generated("outputs/generated.png", std::ios::binary);
        std::string contents((std::istreambuf_iterator<char>(generated)), std::istreambuf_iterator<char>());
        expect(contents == "PNG-ASSET", "router asset file contents");
    }
    const std::string register_video_asset_worker_body =
        "{\"worker_id\":\"video-asset-worker\","
        "\"label\":\"Video Asset Worker\","
        "\"url\":\"http://127.0.0.1:18168\","
        "\"port\":18168,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"video\"],"
        "\"models\":[\"asset-video-model\"],"
        "\"free_vram_gb\":101,"
        "\"best_tier\":121,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_video_asset_worker_body.size()) + "\r\n\r\n" + register_video_asset_worker_body;
    auto registered_video_asset_worker = request(raw);
    expect(registered_video_asset_worker.find("200 OK") != std::string::npos, "router video asset worker register status");

    const std::string video_asset_worker_url = "http://127.0.0.1:18168/outputs/router-video.mp4";
    const std::string video_asset_response_body = "{\"created\":3,\"data\":[{\"url\":\"" + video_asset_worker_url + "\"}]}";
    std::thread fake_video_asset_worker(
        fake_asset_worker_once,
        18168,
        "/v1/videos/generations",
        video_asset_response_body,
        "/outputs/router-video.mp4",
        "MP4-ASSET"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string video_asset_body = "{\"model\":\"asset-video-model\",\"prompt\":\"video asset\"}";
    raw = "POST /v1/videos/generations HTTP/1.1\r\n"
        "Host: localhost:18091\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(video_asset_body.size()) + "\r\n\r\n" + video_asset_body;
    auto router_video_asset_proxy = request(raw);
    fake_video_asset_worker.join();
    expect(router_video_asset_proxy.find("200 OK") != std::string::npos, "router video asset proxy status");
    expect(router_video_asset_proxy.find(video_asset_worker_url) == std::string::npos, "router video asset rewrites worker url");
    expect(router_video_asset_proxy.find("\"url\":\"http://localhost:18091/outputs/router-video.mp4\"") != std::string::npos, "router video asset request-base url");
    expect(std::filesystem::exists("outputs/router-video.mp4"), "router video asset file saved");
    {
        std::ifstream generated("outputs/router-video.mp4", std::ios::binary);
        std::string contents((std::istreambuf_iterator<char>(generated)), std::istreambuf_iterator<char>());
        expect(contents == "MP4-ASSET", "router video asset file contents");
    }
    const std::string deregister_video_asset_worker_body = "{\"worker_id\":\"video-asset-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_video_asset_worker_body.size()) + "\r\n\r\n" + deregister_video_asset_worker_body;
    auto deregistered_video_asset_worker = request(raw);
    expect(deregistered_video_asset_worker.find("200 OK") != std::string::npos, "router video asset worker deregister status");
    auto served_asset = request("GET /outputs/generated.png HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(served_asset.find("200 OK") != std::string::npos, "router asset static status");
    expect(served_asset.find("PNG-ASSET") != std::string::npos, "router asset static body");
    auto served_asset_head = request("HEAD /outputs/generated.png HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(served_asset_head.find("200 OK") != std::string::npos, "router asset static head status");
    expect(served_asset_head.find("Content-Length: 9") != std::string::npos, "router asset static head content length");
    expect(served_asset_head.rfind("\r\n\r\n") == served_asset_head.size() - 4, "router asset static head omits body");
    {
        std::ofstream text_asset("outputs/readme.txt", std::ios::binary);
        text_asset << "TEXT-ASSET";
    }
    auto served_text_asset = request("GET /outputs/readme.txt HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(served_text_asset.find("200 OK") != std::string::npos, "router text asset static status");
    expect(served_text_asset.find("Content-Type: text/plain; charset=utf-8") != std::string::npos, "router text asset static content type");
    expect(served_text_asset.find("Accept-Ranges: bytes") != std::string::npos, "router text asset static accept ranges");
    expect(response_header_before(served_text_asset, "Content-Type", "Accept-Ranges"), "router text asset static content type before accept ranges");
    expect(response_header_before(served_text_asset, "Accept-Ranges", "Content-Length"), "router text asset static accept ranges before content length");
    expect(response_header_before(served_text_asset, "Content-Length", "Last-Modified"), "router text asset static content length before last modified");
    expect(response_header_before(served_text_asset, "Last-Modified", "ETag"), "router text asset static last modified before etag");
    const std::string text_asset_etag = response_header(served_text_asset, "ETag");
    const std::string text_asset_last_modified = response_header(served_text_asset, "Last-Modified");
    expect(!text_asset_etag.empty(), "router text asset static etag");
    expect(!text_asset_last_modified.empty(), "router text asset static last modified");
    expect(served_text_asset.find("TEXT-ASSET") != std::string::npos, "router text asset static body");
    auto served_text_asset_head = request("HEAD /outputs/readme.txt HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(served_text_asset_head.find("200 OK") != std::string::npos, "router text asset static head status");
    expect(served_text_asset_head.find("Content-Type: text/plain; charset=utf-8") != std::string::npos, "router text asset static head content type");
    expect(served_text_asset_head.find("Accept-Ranges: bytes") != std::string::npos, "router text asset static head accept ranges");
    expect(served_text_asset_head.find("Content-Length: 10") != std::string::npos, "router text asset static head content length");
    expect(served_text_asset_head.rfind("\r\n\r\n") == served_text_asset_head.size() - 4, "router text asset static head omits body");
    auto text_asset_not_modified = request(
        "GET /outputs/readme.txt HTTP/1.1\r\nHost: localhost\r\nIf-None-Match: " + text_asset_etag + "\r\n\r\n"
    );
    expect(text_asset_not_modified.find("304 Not Modified") != std::string::npos, "router text asset static etag not modified status");
    expect(text_asset_not_modified.find("ETag: " + text_asset_etag) != std::string::npos, "router text asset static etag not modified header");
    expect(text_asset_not_modified.find("Content-Type:") == std::string::npos, "router text asset static etag not modified omits content type");
    expect(text_asset_not_modified.find("Content-Length:") == std::string::npos, "router text asset static etag not modified omits content length");
    expect(text_asset_not_modified.rfind("\r\n\r\n") == text_asset_not_modified.size() - 4, "router text asset static etag not modified omits body");
    auto text_asset_since_not_modified = request(
        "GET /outputs/readme.txt HTTP/1.1\r\nHost: localhost\r\nIf-Modified-Since: " + text_asset_last_modified + "\r\n\r\n"
    );
    expect(text_asset_since_not_modified.find("304 Not Modified") != std::string::npos, "router text asset static last modified not modified status");
    auto text_asset_range = request("GET /outputs/readme.txt HTTP/1.1\r\nHost: localhost\r\nRange: bytes=0-3\r\n\r\n");
    expect(text_asset_range.find("206 Partial Content") != std::string::npos, "router text asset static range status");
    expect(text_asset_range.find("Content-Length: 4") != std::string::npos, "router text asset static range content length");
    expect(text_asset_range.find("Content-Range: bytes 0-3/10") != std::string::npos, "router text asset static range content range");
    expect(response_header_before(text_asset_range, "ETag", "Content-Range"), "router text asset static range etag before content range");
    expect(text_asset_range.rfind("TEXT") == text_asset_range.size() - 4, "router text asset static range body");
    auto text_asset_suffix_range = request("GET /outputs/readme.txt HTTP/1.1\r\nHost: localhost\r\nRange: bytes=-5\r\n\r\n");
    expect(text_asset_suffix_range.find("206 Partial Content") != std::string::npos, "router text asset static suffix range status");
    expect(text_asset_suffix_range.find("Content-Range: bytes 5-9/10") != std::string::npos, "router text asset static suffix range content range");
    expect(text_asset_suffix_range.rfind("ASSET") == text_asset_suffix_range.size() - 5, "router text asset static suffix range body");
    auto text_asset_multi_range = request("GET /outputs/readme.txt HTTP/1.1\r\nHost: localhost\r\nRange: bytes=0-3,5-8\r\n\r\n");
    expect(text_asset_multi_range.find("206 Partial Content") != std::string::npos, "router text asset static multipart range status");
    expect(response_header(text_asset_multi_range, "Content-Type").find("multipart/byteranges; boundary=") == 0, "router text asset static multipart range header");
    expect(text_asset_multi_range.find("Content-Range: bytes 0-3/10") != std::string::npos, "router text asset static multipart first content range");
    expect(text_asset_multi_range.find("Content-Range: bytes 5-8/10") != std::string::npos, "router text asset static multipart second content range");
    expect(text_asset_multi_range.find("\r\n\r\nTEXT\r\n") != std::string::npos, "router text asset static multipart first body");
    expect(text_asset_multi_range.find("\r\n\r\nASSE\r\n") != std::string::npos, "router text asset static multipart second body");
    expect(response_header(text_asset_multi_range, "Content-Length") ==
           std::to_string(response_body(text_asset_multi_range).size()), "multipart wire body matches declared length");
    auto text_asset_range_head = request("HEAD /outputs/readme.txt HTTP/1.1\r\nHost: localhost\r\nRange: bytes=0-3\r\n\r\n");
    expect(text_asset_range_head.find("206 Partial Content") != std::string::npos &&
           response_header(text_asset_range_head, "Content-Length") == "4" &&
           response_body(text_asset_range_head).empty(), "range HEAD keeps length and omits deferred body");
    auto text_asset_if_range_mismatch = request("GET /outputs/readme.txt HTTP/1.1\r\nHost: localhost\r\nRange: bytes=0-3\r\nIf-Range: \"mismatch\"\r\n\r\n");
    expect(text_asset_if_range_mismatch.find("200 OK") != std::string::npos, "router text asset static if-range mismatch full status");
    expect(text_asset_if_range_mismatch.find("Content-Range:") == std::string::npos, "router text asset static if-range mismatch omits content range");
    expect(text_asset_if_range_mismatch.find("TEXT-ASSET") != std::string::npos, "router text asset static if-range mismatch body");
    auto text_asset_bad_range_unit = request("GET /outputs/readme.txt HTTP/1.1\r\nHost: localhost\r\nRange: items=0-3\r\n\r\n");
    expect(text_asset_bad_range_unit.find("400 Bad Request") != std::string::npos, "router text asset static bad range unit status");
    expect(response_header_before(text_asset_bad_range_unit, "Content-Length", "Content-Type"), "router text asset static bad range unit header order");
    expect(text_asset_bad_range_unit.find("Only support bytes range") != std::string::npos, "router text asset static bad range unit body");
    auto text_asset_bad_range = request("GET /outputs/readme.txt HTTP/1.1\r\nHost: localhost\r\nRange: bytes=50-60\r\n\r\n");
    expect(text_asset_bad_range.find("416 Range Not Satisfiable") != std::string::npos, "router text asset static unsatisfiable range status");
    expect(text_asset_bad_range.find("Content-Range: bytes */10") != std::string::npos, "router text asset static unsatisfiable range header");
    expect(response_header_before(text_asset_bad_range, "Content-Range", "Content-Length"), "router text asset static unsatisfiable content range before length");
    expect(response_header_before(text_asset_bad_range, "Content-Length", "Content-Type"), "router text asset static unsatisfiable length before type");
    expect(text_asset_bad_range.rfind("\r\n\r\n") == text_asset_bad_range.size() - 4, "router text asset static unsatisfiable range omits body");
    {
        std::ofstream spaced_asset("outputs/fancy name.txt", std::ios::binary);
        spaced_asset << "SPACE-ASSET";
        std::ofstream dotted_asset("outputs/a..b.txt", std::ios::binary);
        dotted_asset << "DOTS-ASSET";
        std::filesystem::create_directories("outputs/nested");
    }
    auto spaced_text_asset = request("GET /outputs/fancy%20name.txt HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(spaced_text_asset.find("200 OK") != std::string::npos, "router text asset static url-decoded path status");
    expect(spaced_text_asset.find("SPACE-ASSET") != std::string::npos, "router text asset static url-decoded path body");
    auto dotted_text_asset = request("GET /outputs/a..b.txt HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(dotted_text_asset.find("200 OK") != std::string::npos, "router text asset static dotted segment status");
    expect(dotted_text_asset.find("DOTS-ASSET") != std::string::npos, "router text asset static dotted segment body");
    auto normalized_text_asset = request("GET /outputs/nested/../readme.txt HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(normalized_text_asset.find("200 OK") != std::string::npos, "router text asset static normalized path status");
    expect(normalized_text_asset.find("TEXT-ASSET") != std::string::npos, "router text asset static normalized path body");
    auto escaped_text_asset = request("GET /outputs/%2e%2e/app.py HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(escaped_text_asset.find("404 Not Found") != std::string::npos, "router text asset static escaped traversal status");
    auto missing_asset_head = request("HEAD /outputs/missing.png HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(missing_asset_head.find("404 Not Found") != std::string::npos, "missing asset head status");
    expect(missing_asset_head.rfind("\r\n\r\n") == missing_asset_head.size() - 4, "missing asset head omits body");
    auto post_asset = request("POST /outputs/generated.png HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n");
    expect(post_asset.find("405 Method Not Allowed") != std::string::npos, "router asset static post status");
    expect(post_asset.find("Allow:") == std::string::npos, "router asset static post omits allow");

    const std::string register_tts_asset_worker_body =
        "{\"worker_id\":\"tts-asset-worker\","
        "\"label\":\"TTS Asset Worker\","
        "\"url\":\"http://127.0.0.1:18102\","
        "\"port\":18102,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"tts\"],"
        "\"models\":[],"
        "\"free_vram_gb\":99,"
        "\"best_tier\":130,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_tts_asset_worker_body.size()) + "\r\n\r\n" + register_tts_asset_worker_body;
    auto registered_tts_asset_worker = request(raw);
    expect(registered_tts_asset_worker.find("200 OK") != std::string::npos, "router tts asset worker register status");

    const std::string tts_asset_worker_url = "http://127.0.0.1:18102/outputs/speech.wav";
    const std::string tts_asset_response_body = "{\"url\":\"" + tts_asset_worker_url + "\"}";
    std::thread fake_tts_asset_worker(
        fake_asset_worker_once,
        18102,
        "/v1/audio/speech",
        tts_asset_response_body,
        "/outputs/speech.wav",
        "WAV-ASSET"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string tts_asset_body = "{\"model\":\"tts-asset-model\",\"input\":\"say it\",\"voice\":\"amy\"}";
    raw = "POST /v1/audio/speech HTTP/1.1\r\n"
        "Host: localhost:18091\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(tts_asset_body.size()) + "\r\n\r\n" + tts_asset_body;
    auto router_tts_asset_proxy = request(raw);
    fake_tts_asset_worker.join();
    expect(router_tts_asset_proxy.find("200 OK") != std::string::npos, "router tts asset proxy status");
    expect(router_tts_asset_proxy.find(tts_asset_worker_url) == std::string::npos, "router tts asset rewrites worker url");
    expect(router_tts_asset_proxy.find("\"url\":\"http://localhost:18091/outputs/speech.wav\"") != std::string::npos, "router tts asset request-base url");
    expect(std::filesystem::exists("outputs/speech.wav"), "router tts asset file saved");
    {
        std::ifstream generated("outputs/speech.wav", std::ios::binary);
        std::string contents((std::istreambuf_iterator<char>(generated)), std::istreambuf_iterator<char>());
        expect(contents == "WAV-ASSET", "router tts asset file contents");
    }
    auto served_tts_asset = request("GET /outputs/speech.wav HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(served_tts_asset.find("200 OK") != std::string::npos, "router tts asset static status");
    expect(served_tts_asset.find("Content-Type: audio/x-wav") != std::string::npos, "router tts asset static content type");
    expect(served_tts_asset.find("WAV-ASSET") != std::string::npos, "router tts asset static body");

    const std::string register_tts_stream_worker_body =
        "{\"worker_id\":\"tts-stream-worker\","
        "\"label\":\"TTS Stream Worker\","
        "\"url\":\"http://127.0.0.1:18122\","
        "\"port\":18122,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"tts\"],"
        "\"models\":[],"
        "\"free_vram_gb\":100,"
        "\"best_tier\":140,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_tts_stream_worker_body.size()) + "\r\n\r\n" + register_tts_stream_worker_body;
    auto registered_tts_stream_worker = request(raw);
    expect(registered_tts_stream_worker.find("200 OK") != std::string::npos, "router tts stream worker register status");

    std::thread fake_tts_stream_worker(
        fake_expect_worker_once_with_type,
        18122,
        "POST /v1/audio/speech/stream HTTP/1.1",
        "Content-Type: application/json",
        "\"input\":\"stream it\"",
        "application/octet-stream",
        "STREAM-DATA"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string tts_stream_body = "{\"model\":\"tts-stream-model\",\"input\":\"stream it\",\"voice\":\"amy\"}";
    raw = "POST /v1/audio/speech/stream HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(tts_stream_body.size()) + "\r\n\r\n" + tts_stream_body;
    auto router_tts_stream_proxy = request(raw);
    fake_tts_stream_worker.join();
    expect(router_tts_stream_proxy.find("200 OK") != std::string::npos, "router tts stream proxy status");
    expect(router_tts_stream_proxy.find("Content-Type: text/event-stream") != std::string::npos, "router tts stream proxy content type");
    expect(router_tts_stream_proxy.find("Transfer-Encoding: chunked") != std::string::npos, "router tts stream proxy chunked transfer");
    expect(router_tts_stream_proxy.find("STREAM-DATA") != std::string::npos, "router tts stream proxy body");

    auto dashboard_html = request("GET /dashboard HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(dashboard_html.find("200 OK") != std::string::npos, "dashboard html status");
    expect(dashboard_html.find("text/html") != std::string::npos, "dashboard html content type");
    expect(dashboard_html.find("<title>ezlocalai Router</title>") != std::string::npos, "dashboard html title");
    expect(dashboard_html.find("<div id=\"dash\">") != std::string::npos, "dashboard html refresh container");
    expect(dashboard_html.find("auto-refresh 5s") != std::string::npos, "dashboard html refresh label");
    expect(dashboard_html.find("X-Dashboard-Refresh") != std::string::npos, "dashboard html refresh header");
    expect(dashboard_html.find("<pre>") == std::string::npos, "dashboard html is not json pre wrapper");
    expect(dashboard_html.find("Structured model rows are available") == std::string::npos, "dashboard html renders model table");
    expect(dashboard_html.find("<h2>LLM model breakdown</h2>") != std::string::npos, "dashboard html renders llm model breakdown");
    expect(dashboard_html.find("Usage Worker") != std::string::npos, "dashboard html renders usage rows");
    expect(dashboard_html.find("Text + Vision") != std::string::npos, "dashboard html renders capability cards");
    expect(dashboard_html.find("tunnel://home-node") != std::string::npos, "dashboard html renders top-level worker url");
    expect(dashboard_html.find("RTX 4090 (24GB VRAM)") != std::string::npos, "dashboard html renders worker gpu details");

    const std::string register_dashboard_special_body =
        "{\"worker_id\":\"dash-special<&\","
        "\"label\":\"Dash <& Worker\","
        "\"url\":\"http://10.0.0.20:8091/path?x=<y>&z=q\","
        "\"capabilities\":[],"
        "\"models\":[\"dash-model<&\"],"
        "\"gpus\":[{\"index\":0,\"name\":\"GPU <&\",\"total_vram_gb\":1,\"backend\":\"rocm<&\"}],"
        "\"version\":\"ver<&\"}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_dashboard_special_body.size()) + "\r\n\r\n" + register_dashboard_special_body;
    auto registered_dashboard_special = request(raw);
    expect(registered_dashboard_special.find("200 OK") != std::string::npos, "router dashboard special worker register status");
    auto dashboard_html_special = request("GET /dashboard HTTP/1.1\r\nHost: localhost\r\n\r\n");
    expect(dashboard_html_special.find("Dash <& Worker") != std::string::npos, "dashboard html preserves python raw worker label");
    expect(dashboard_html_special.find("dash-model<&") != std::string::npos, "dashboard html preserves python raw model name");
    expect(dashboard_html_special.find("GPU <& (1GB VRAM) [rocm<&]") != std::string::npos, "dashboard html preserves python raw gpu details");
    expect(dashboard_html_special.find("Dash &lt;&amp; Worker") == std::string::npos, "dashboard html does not escape worker label differently from python");
    const std::string deregister_dashboard_special_body = "{\"worker_id\":\"dash-special<&\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_dashboard_special_body.size()) + "\r\n\r\n" + deregister_dashboard_special_body;
    auto deregistered_dashboard_special = request(raw);
    expect(deregistered_dashboard_special.find("200 OK") != std::string::npos, "router dashboard special worker deregister status");

    auto router_usage = request("GET /v1/router/usage HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(router_usage.find("200 OK") != std::string::npos, "router usage status");
    expect(router_usage.find("\"Usage Worker\"") != std::string::npos, "router usage worker");
    expect(router_usage.find("\"usage-model\"") != std::string::npos, "router usage model");
    expect(router_usage.find("\"usage-model-GGUF\"") == std::string::npos, "router usage strips gguf suffix");
    expect(router_usage.find("\"requests\":4,\"prompt_tokens\":18,\"completion_tokens\":26") != std::string::npos, "router usage includes streaming and timings-only tokens");
    expect(router_usage.find("\"prompt_tps_sum\":40.0") != std::string::npos, "router usage float tps sum");
    expect(router_usage.find("\"predicted_tps_sum\":40.0") != std::string::npos, "router usage float predicted tps sum");
    expect(router_usage.find("\"Embedding Worker\"") != std::string::npos, "router embedding usage worker");
    expect(router_usage.find("\"embedding\":{\"requests\":1}") != std::string::npos, "router embedding usage cap");
    auto z_prior_usage_pos = router_usage.find("\"Z Prior\"");
    auto prior_usage_pos = router_usage.find("\"Prior Worker\"");
    auto usage_worker_usage_pos = router_usage.find("\"Usage Worker\"");
    expect(z_prior_usage_pos != std::string::npos && prior_usage_pos != std::string::npos
        && usage_worker_usage_pos != std::string::npos
        && z_prior_usage_pos < prior_usage_pos && prior_usage_pos < usage_worker_usage_pos,
        "router usage preserves worker insertion order");
    expect(router_usage.find("\"Prior Worker\"") != std::string::npos, "router usage preserves loaded worker");
    expect(router_usage.find("\"llm\":{\"z-model\"") != std::string::npos, "router usage preserves loaded llm model order");
    expect(router_usage.find("\"Prior Worker\":{\"llm\"") != std::string::npos, "router usage preserves loaded section order");
    expect(router_usage.find("\"image\":{\"requests\":3}") != std::string::npos, "router usage preserves loaded cap");
    expect(router_usage.find("\"custom\":{\"kept\":true}") != std::string::npos, "router usage preserves loaded custom section");

    auto router_history = request("GET /v1/router/history?limit=1 HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(router_history.find("200 OK") != std::string::npos, "router history status");
    expect(router_history.find("\"history\":[") != std::string::npos, "router history shape");
    expect(router_history.find("\"completion_tokens\":5") != std::string::npos, "router history timings-only completion tokens");
    expect(router_history.find("\"prompt_ms\":100.0") != std::string::npos, "router history float prompt ms");
    expect(router_history.find("\"predicted_tps\":20.0") != std::string::npos, "router history float predicted tps");
    auto router_history_stream_usage = request("GET /v1/router/history?limit=3 HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(router_history_stream_usage.find("\"prompt_tokens\":7,\"completion_tokens\":8") != std::string::npos, "router history includes streamed usage tokens");
    expect(router_history_stream_usage.find("\"prompt_tokens\":6,\"completion_tokens\":9") != std::string::npos, "router history includes streamed completion usage tokens");
    const std::string stream_timing_prefix = "\"prompt_tokens\":7,\"completion_tokens\":8,\"prompt_ms\":0.0,\"predicted_ms\":0.0,\"total_ms\":";
    const auto stream_timing_position = router_history_stream_usage.find(stream_timing_prefix);
    expect(stream_timing_position != std::string::npos && std::stod(router_history_stream_usage.substr(stream_timing_position + stream_timing_prefix.size())) > 0.0,
        "router history injects positive elapsed total_ms for streamed usage");
    auto router_history_loaded = request("GET /v1/router/history?limit=10 HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(router_history_loaded.find("\"worker\":\"Prior Worker\"") != std::string::npos, "router history preserves loaded entry");
    expect(router_history_loaded.find("\"note\":\"kept\"") != std::string::npos, "router history preserves loaded raw fields");
    auto router_history_bad_limit = request("GET /v1/router/history?limit=x HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(router_history_bad_limit.find("422 Unprocessable Entity") != std::string::npos, "router history bad limit status");
    expect(router_history_bad_limit.find("\"loc\":[\"query\",\"limit\"]") != std::string::npos, "router history bad limit loc");
    auto router_history_negative_limit = request("GET /v1/router/history?limit=-1 HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(router_history_negative_limit.find("200 OK") != std::string::npos, "router history negative limit status");

    setenv("ROUTER_MODE", "true", 1);
    setenv("ROUTER_ERROR_THRESHOLD", "1", 1);
    const std::string register_failing_worker_body =
        "{\"worker_id\":\"failing-worker\","
        "\"label\":\"Failing Worker\","
        "\"url\":\"http://127.0.0.1:18096\","
        "\"port\":18096,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"text\"],"
        "\"models\":[\"fail-model\"],"
        "\"best_tier\":70,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_failing_worker_body.size()) + "\r\n\r\n" + register_failing_worker_body;
    auto registered_failing_worker = request(raw);
    expect(registered_failing_worker.find("200 OK") != std::string::npos, "router failing worker register status");

    const std::string register_failover_worker_body =
        "{\"worker_id\":\"failover-worker\","
        "\"label\":\"Failover Worker\","
        "\"url\":\"http://127.0.0.1:18100\","
        "\"port\":18100,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"text\"],"
        "\"models\":[\"fail-model\"],"
        "\"best_tier\":20,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_failover_worker_body.size()) + "\r\n\r\n" + register_failover_worker_body;
    auto registered_failover_worker = request(raw);
    expect(registered_failover_worker.find("200 OK") != std::string::npos, "router failover worker register status");

    std::thread fake_failing_worker(fake_json_worker_once_status, 18096, 500, "{\"detail\":\"boom\"}");
    std::thread fake_failover_worker(fake_json_worker_once, 18100, "{\"id\":\"cmpl-failover\",\"object\":\"text_completion\",\"model\":\"fail-model\",\"choices\":[{\"text\":\"failover ok\",\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":6,\"total_tokens\":11}}");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string failing_completion_body = "{\"model\":\"fail-model\",\"prompt\":\"hello\"}";
    raw = "POST /v1/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(failing_completion_body.size()) + "\r\n\r\n" + failing_completion_body;
    auto failing_completion = request(raw);
    fake_failing_worker.join();
    fake_failover_worker.join();
    expect(failing_completion.find("200 OK") != std::string::npos, "router failover proxy status");
    expect(failing_completion.find("failover ok") != std::string::npos, "router failover proxy body");

    auto router_errors = request("GET /v1/router/errors HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(router_errors.find("200 OK") != std::string::npos, "router errors status");
    expect(router_errors.find("\"worker_id\":\"failing-worker\"") != std::string::npos, "router errors includes failing worker");
    expect(router_errors.find("\"kind\":\"http_5xx\"") != std::string::npos, "router errors records kind");
    expect(router_errors.find("\"circuit_open\":true") != std::string::npos, "router errors circuit opens");

    setenv("ROUTER_MAX_RETRIES", "0bad", 1);
    const std::string register_retry_env_primary_body =
        "{\"worker_id\":\"retry-env-primary\","
        "\"label\":\"Retry Env Primary\","
        "\"url\":\"http://127.0.0.1:18132\","
        "\"port\":18132,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"text\"],"
        "\"models\":[\"retry-env-model\"],"
        "\"best_tier\":75,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_retry_env_primary_body.size()) + "\r\n\r\n" + register_retry_env_primary_body;
    auto registered_retry_env_primary = request(raw);
    expect(registered_retry_env_primary.find("200 OK") != std::string::npos, "router retry env primary register status");

    const std::string register_retry_env_fallback_body =
        "{\"worker_id\":\"retry-env-fallback\","
        "\"label\":\"Retry Env Fallback\","
        "\"url\":\"http://127.0.0.1:18133\","
        "\"port\":18133,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"text\"],"
        "\"models\":[\"retry-env-model\"],"
        "\"best_tier\":20,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_retry_env_fallback_body.size()) + "\r\n\r\n" + register_retry_env_fallback_body;
    auto registered_retry_env_fallback = request(raw);
    expect(registered_retry_env_fallback.find("200 OK") != std::string::npos, "router retry env fallback register status");

    std::thread fake_retry_env_primary(fake_json_worker_once_status, 18132, 500, "{\"detail\":\"retry env boom\"}");
    std::thread fake_retry_env_fallback(fake_json_worker_once, 18133, "{\"id\":\"cmpl-retry-env\",\"object\":\"text_completion\",\"model\":\"retry-env-model\",\"choices\":[{\"text\":\"retry env ok\",\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string retry_env_completion_body = "{\"model\":\"retry-env-model\",\"prompt\":\"hello\"}";
    raw = "POST /v1/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(retry_env_completion_body.size()) + "\r\n\r\n" + retry_env_completion_body;
    auto retry_env_completion = request(raw);
    fake_retry_env_primary.join();
    fake_retry_env_fallback.join();
    expect(retry_env_completion.find("200 OK") != std::string::npos, "router retry env fallback status");
    expect(retry_env_completion.find("retry env ok") != std::string::npos, "router retry env fallback body");
    const std::string deregister_retry_env_primary_body = "{\"worker_id\":\"retry-env-primary\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_retry_env_primary_body.size()) + "\r\n\r\n" + deregister_retry_env_primary_body;
    auto deregistered_retry_env_primary = request(raw);
    expect(deregistered_retry_env_primary.find("200 OK") != std::string::npos, "router retry env primary deregister status");
    const std::string deregister_retry_env_fallback_body = "{\"worker_id\":\"retry-env-fallback\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_retry_env_fallback_body.size()) + "\r\n\r\n" + deregister_retry_env_fallback_body;
    auto deregistered_retry_env_fallback = request(raw);
    expect(deregistered_retry_env_fallback.find("200 OK") != std::string::npos, "router retry env fallback deregister status");
    unsetenv("ROUTER_MAX_RETRIES");

    setenv("ROUTER_MAX_RETRIES", "1", 1);
    const std::string register_retry_exhausted_body =
        "{\"worker_id\":\"retry-exhausted-worker\","
        "\"label\":\"Retry Exhausted\","
        "\"url\":\"http://127.0.0.1:18154\","
        "\"port\":18154,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"text\"],"
        "\"models\":[\"solo-retry-model\"],"
        "\"best_tier\":70,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_retry_exhausted_body.size()) + "\r\n\r\n" + register_retry_exhausted_body;
    auto registered_retry_exhausted = request(raw);
    expect(registered_retry_exhausted.find("200 OK") != std::string::npos, "router exhausted retry worker register status");
    std::thread fake_retry_exhausted(fake_json_worker_once_status, 18154, 500, "{\"detail\":\"only worker boom\"}");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    setenv("ROUTER_WAIT_TIMEOUT", "0", 1);
    const std::string retry_exhausted_body = "{\"model\":\"solo-retry-model\",\"prompt\":\"hello\"}";
    raw = "POST /v1/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(retry_exhausted_body.size()) + "\r\n\r\n" + retry_exhausted_body;
    auto retry_exhausted = request(raw);
    fake_retry_exhausted.join();
    expect(retry_exhausted.find("502 Bad Gateway") != std::string::npos, "router exhausted retry status");
    expect(retry_exhausted.find("All 2 worker attempts failed for /v1/completions") != std::string::npos, "router exhausted retry detail");
    expect(retry_exhausted.find("No worker available for capability='text' model='solo-retry-model'") != std::string::npos, "router exhausted retry no worker detail");
    const std::string deregister_retry_exhausted_body = "{\"worker_id\":\"retry-exhausted-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_retry_exhausted_body.size()) + "\r\n\r\n" + deregister_retry_exhausted_body;
    auto deregistered_retry_exhausted = request(raw);
    expect(deregistered_retry_exhausted.find("200 OK") != std::string::npos, "router exhausted retry worker deregister status");
    unsetenv("ROUTER_MAX_RETRIES");
    unsetenv("ROUTER_WAIT_TIMEOUT");

    setenv("ROUTER_MAX_RETRIES", "0", 1);
    const std::string register_retry_transport_body =
        "{\"worker_id\":\"retry-transport-worker\","
        "\"label\":\"Retry Transport\","
        "\"url\":\"http://127.0.0.1:18172\","
        "\"port\":18172,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"text\"],"
        "\"models\":[\"retry-transport-model\"],"
        "\"best_tier\":70,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_retry_transport_body.size()) + "\r\n\r\n" + register_retry_transport_body;
    auto registered_retry_transport = request(raw);
    expect(registered_retry_transport.find("200 OK") != std::string::npos, "router retry transport worker register status");
    const std::string retry_transport_body = "{\"model\":\"retry-transport-model\",\"prompt\":\"hello\"}";
    raw = "POST /v1/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(retry_transport_body.size()) + "\r\n\r\n" + retry_transport_body;
    auto retry_transport = request(raw);
    expect(retry_transport.find("502 Bad Gateway") != std::string::npos, "router retry transport exhausted status");
    expect(retry_transport.find("All 1 worker attempts failed for /v1/completions: ClientConnectorError: Cannot connect to host 127.0.0.1:18172 ssl:default [Connect call failed ('127.0.0.1', 18172)]") != std::string::npos,
        "router retry transport exhausted detail mirrors python client connector error");
    expect(retry_transport.find("HTTPException: 503") == std::string::npos,
        "router retry transport exhausted detail omits native transport wrapper");
    auto retry_transport_errors = request("GET /v1/router/errors HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    auto retry_transport_errors_pos = retry_transport_errors.find("\"worker_id\":\"retry-transport-worker\"");
    auto retry_transport_errors_next_pos = retry_transport_errors.find("\"worker_id\":", retry_transport_errors_pos + 1);
    auto retry_transport_status_null_pos = retry_transport_errors.find("\"status\":null", retry_transport_errors_pos);
    auto retry_transport_message_pos = retry_transport_errors.find("\"message\":\"ClientConnectorError: Cannot connect to host 127.0.0.1:18172 ssl:default [Connect call failed ('127.0.0.1', 18172)]\"", retry_transport_errors_pos);
    expect(retry_transport_errors_pos != std::string::npos
        && retry_transport_status_null_pos != std::string::npos
        && retry_transport_message_pos != std::string::npos
        && (retry_transport_errors_next_pos == std::string::npos
            || (retry_transport_status_null_pos < retry_transport_errors_next_pos
                && retry_transport_message_pos < retry_transport_errors_next_pos)),
        "router retry transport error records python-like status and message");
    const std::string deregister_retry_transport_body = "{\"worker_id\":\"retry-transport-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_retry_transport_body.size()) + "\r\n\r\n" + deregister_retry_transport_body;
    auto deregistered_retry_transport = request(raw);
    expect(deregistered_retry_transport.find("200 OK") != std::string::npos, "router retry transport worker deregister status");
    unsetenv("ROUTER_MAX_RETRIES");

    const std::string register_status600_worker_body =
        "{\"worker_id\":\"status600-worker\","
        "\"label\":\"Status600 Worker\","
        "\"url\":\"http://127.0.0.1:18131\","
        "\"port\":18131,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"text\"],"
        "\"models\":[\"status600-model\"],"
        "\"best_tier\":60,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_status600_worker_body.size()) + "\r\n\r\n" + register_status600_worker_body;
    auto registered_status600_worker = request(raw);
    expect(registered_status600_worker.find("200 OK") != std::string::npos, "router status600 worker register status");

    std::thread fake_status600_worker(fake_json_worker_once_status, 18131, 600, "{\"detail\":\"not transient\"}");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string status600_completion_body = "{\"model\":\"status600-model\",\"prompt\":\"hello\"}";
    raw = "POST /v1/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(status600_completion_body.size()) + "\r\n\r\n" + status600_completion_body;
    auto status600_completion = request(raw);
    fake_status600_worker.join();
    expect(status600_completion.find("600 OK") != std::string::npos, "router status600 does not retry status");
    expect(status600_completion.find("not transient") != std::string::npos, "router status600 body returned");
    const std::string deregister_status600_worker_body = "{\"worker_id\":\"status600-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_status600_worker_body.size()) + "\r\n\r\n" + deregister_status600_worker_body;
    auto deregistered_status600_worker = request(raw);
    expect(deregistered_status600_worker.find("200 OK") != std::string::npos, "router status600 worker deregister status");

    setenv("ROUTER_IDLE_TIER_WINDOW", "0bad", 1);
    setenv("ROUTER_CROSS_MODEL_GRACE", "0", 1);
    const std::string register_idle_window_fast_body =
        "{\"worker_id\":\"idle-window-fast\","
        "\"label\":\"Idle Window Fast\","
        "\"url\":\"http://127.0.0.1:18139\","
        "\"port\":18139,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"text\"],"
        "\"models\":[\"idle-window-model\"],"
        "\"best_tier\":90,"
        "\"queue_capacity\":2,"
        "\"cap_slots\":{\"text\":{\"capacity\":2,\"in_flight\":1,\"queued\":0}}}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_idle_window_fast_body.size()) + "\r\n\r\n" + register_idle_window_fast_body;
    auto registered_idle_window_fast = request(raw);
    expect(registered_idle_window_fast.find("200 OK") != std::string::npos, "router idle window fast register status");

    const std::string register_idle_window_slow_body =
        "{\"worker_id\":\"idle-window-slow\","
        "\"label\":\"Idle Window Slow\","
        "\"url\":\"http://127.0.0.1:18140\","
        "\"port\":18140,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"text\"],"
        "\"models\":[\"idle-window-model\"],"
        "\"best_tier\":80,"
        "\"queue_capacity\":1,"
        "\"cap_slots\":{\"text\":{\"capacity\":1,\"in_flight\":0,\"queued\":0}}}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_idle_window_slow_body.size()) + "\r\n\r\n" + register_idle_window_slow_body;
    auto registered_idle_window_slow = request(raw);
    expect(registered_idle_window_slow.find("200 OK") != std::string::npos, "router idle window slow register status");

    std::thread fake_idle_window_fast(fake_json_worker_once_status_timeout, 18139, 200, "{\"object\":\"list\",\"model\":\"fast-selected\",\"data\":[],\"usage\":{\"prompt_tokens\":1,\"total_tokens\":1}}", 1000);
    std::thread fake_idle_window_slow(fake_json_worker_once_status_timeout, 18140, 200, "{\"object\":\"list\",\"model\":\"slow-selected\",\"data\":[],\"usage\":{\"prompt_tokens\":1,\"total_tokens\":1}}", 1000);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string idle_window_embedding_body = "{\"model\":\"idle-window-model\",\"input\":\"hello\"}";
    raw = "POST /v1/embeddings HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(idle_window_embedding_body.size()) + "\r\n\r\n" + idle_window_embedding_body;
    auto idle_window_embedding = request(raw);
    fake_idle_window_fast.join();
    fake_idle_window_slow.join();
    expect(idle_window_embedding.find("200 OK") != std::string::npos, "router idle window selection status");
    expect(idle_window_embedding.find("slow-selected") != std::string::npos, "router invalid idle window uses default");
    const std::string deregister_idle_window_fast_body = "{\"worker_id\":\"idle-window-fast\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_idle_window_fast_body.size()) + "\r\n\r\n" + deregister_idle_window_fast_body;
    auto deregistered_idle_window_fast = request(raw);
    expect(deregistered_idle_window_fast.find("200 OK") != std::string::npos, "router idle window fast deregister status");
    const std::string deregister_idle_window_slow_body = "{\"worker_id\":\"idle-window-slow\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_idle_window_slow_body.size()) + "\r\n\r\n" + deregister_idle_window_slow_body;
    auto deregistered_idle_window_slow = request(raw);
    expect(deregistered_idle_window_slow.find("200 OK") != std::string::npos, "router idle window slow deregister status");
    unsetenv("ROUTER_IDLE_TIER_WINDOW");
    unsetenv("ROUTER_CROSS_MODEL_GRACE");

    setenv("ROUTER_CROSS_MODEL_GRACE", "0bad", 1);
    setenv("ROUTER_WAIT_TIMEOUT", "0", 1);
    const std::string register_cross_grace_worker_body =
        "{\"worker_id\":\"cross-grace-worker\","
        "\"label\":\"Cross Grace Worker\","
        "\"url\":\"http://127.0.0.1:18142\","
        "\"port\":18142,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"text\"],"
        "\"models\":[\"cross-grace-other\"],"
        "\"best_tier\":95,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_cross_grace_worker_body.size()) + "\r\n\r\n" + register_cross_grace_worker_body;
    auto registered_cross_grace_worker = request(raw);
    expect(registered_cross_grace_worker.find("200 OK") != std::string::npos, "router cross grace worker register status");
    const std::string cross_grace_completion_body = "{\"model\":\"cross-grace-target\",\"prompt\":\"hello\"}";
    raw = "POST /v1/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(cross_grace_completion_body.size()) + "\r\n\r\n" + cross_grace_completion_body;
    auto cross_grace_completion = request(raw);
    expect(cross_grace_completion.find("503 Service Unavailable") != std::string::npos, "router invalid cross grace uses default status");
    expect(cross_grace_completion.find("No worker available for capability='text' model='cross-grace-target'") != std::string::npos, "router invalid cross grace waits same model");
    const std::string deregister_cross_grace_worker_body = "{\"worker_id\":\"cross-grace-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_cross_grace_worker_body.size()) + "\r\n\r\n" + deregister_cross_grace_worker_body;
    auto deregistered_cross_grace_worker = request(raw);
    expect(deregistered_cross_grace_worker.find("200 OK") != std::string::npos, "router cross grace worker deregister status");
    unsetenv("ROUTER_CROSS_MODEL_GRACE");
    unsetenv("ROUTER_WAIT_TIMEOUT");

    setenv("ROUTER_WAIT_TIMEOUT", "1", 1);
    setenv("ROUTER_CROSS_MODEL_GRACE", "1", 1);
    const std::string register_wait_worker_body =
        "{\"worker_id\":\"wait-worker\","
        "\"label\":\"Wait Worker\","
        "\"url\":\"http://127.0.0.1:18179\","
        "\"port\":18179,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"text\"],"
        "\"models\":[\"wait-model\"],"
        "\"best_tier\":70,"
        "\"queue_capacity\":1,"
        "\"cap_slots\":{\"text\":{\"capacity\":1,\"in_flight\":1,\"queued\":0}},"
        "\"model_slots\":{\"wait-model\":{\"capacity\":1,\"in_flight\":1,\"queued\":0}}}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_wait_worker_body.size()) + "\r\n\r\n" + register_wait_worker_body;
    auto registered_wait_worker = request(raw);
    expect(registered_wait_worker.find("200 OK") != std::string::npos, "router wait worker register status");
    std::thread fake_wait_worker(
        fake_json_worker_once,
        18179,
        "{\"id\":\"cmpl-wait\",\"object\":\"text_completion\",\"model\":\"wait-model\",\"choices\":[{\"text\":\"wait ok\",\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}"
    );
    std::string wait_completion;
    const std::string wait_completion_body = "{\"model\":\"wait-model\",\"prompt\":\"hello\"}";
    std::thread wait_request_thread([&]() {
        wait_completion = request("POST /v1/completions HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(wait_completion_body.size()) + "\r\n\r\n" + wait_completion_body);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    const std::string wait_heartbeat_body =
        "{\"worker_id\":\"wait-worker\","
        "\"cap_slots\":{\"text\":{\"capacity\":1,\"in_flight\":0,\"queued\":0}},"
        "\"model_slots\":{\"wait-model\":{\"capacity\":1,\"in_flight\":0,\"queued\":0}}}";
    raw = "POST /v1/router/heartbeat HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(wait_heartbeat_body.size()) + "\r\n\r\n" + wait_heartbeat_body;
    auto wait_heartbeat = request(raw);
    expect(wait_heartbeat.find("200 OK") != std::string::npos, "router wait heartbeat status");
    wait_request_thread.join();
    fake_wait_worker.join();
    expect(wait_completion.find("200 OK") != std::string::npos, "router wait completion status");
    expect(wait_completion.find("wait ok") != std::string::npos, "router wait completion body");
    const std::string deregister_wait_worker_body = "{\"worker_id\":\"wait-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_wait_worker_body.size()) + "\r\n\r\n" + deregister_wait_worker_body;
    auto deregistered_wait_worker = request(raw);
    expect(deregistered_wait_worker.find("200 OK") != std::string::npos, "router wait worker deregister status");
    unsetenv("ROUTER_WAIT_TIMEOUT");
    unsetenv("ROUTER_CROSS_MODEL_GRACE");

    setenv("ROUTER_ERROR_WINDOW_SECONDS", "60bad", 1);
    setenv("ROUTER_ERROR_THRESHOLD", "1", 1);
    const std::string register_error_threshold_worker_body =
        "{\"worker_id\":\"error-threshold-worker\","
        "\"label\":\"Error Threshold Worker\","
        "\"url\":\"http://127.0.0.1:18141\","
        "\"port\":18141,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"text\"],"
        "\"models\":[\"error-threshold-model\"],"
        "\"best_tier\":65,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_error_threshold_worker_body.size()) + "\r\n\r\n" + register_error_threshold_worker_body;
    auto registered_error_threshold_worker = request(raw);
    expect(registered_error_threshold_worker.find("200 OK") != std::string::npos, "router error threshold worker register status");
    std::thread fake_error_threshold_worker(fake_json_worker_once_status, 18141, 500, "{\"detail\":\"threshold boom\"}");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string error_threshold_embedding_body = "{\"model\":\"error-threshold-model\",\"input\":\"hello\"}";
    raw = "POST /v1/embeddings HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(error_threshold_embedding_body.size()) + "\r\n\r\n" + error_threshold_embedding_body;
    auto error_threshold_embedding = request(raw);
    fake_error_threshold_worker.join();
    expect(error_threshold_embedding.find("500 Internal Server Error") != std::string::npos, "router error threshold worker response status");
    auto error_threshold_usage = request("GET /v1/router/usage HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    auto error_usage_pos = error_threshold_usage.find("\"Error Threshold Worker\"");
    auto error_embedding_usage_pos = error_threshold_usage.find("\"embedding\":{\"requests\":1}", error_usage_pos);
    expect(error_usage_pos != std::string::npos && error_embedding_usage_pos != std::string::npos,
        "router records embedding cap usage after worker HTTP error response");
    auto error_threshold_workers = request("GET /v1/router/workers HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    auto threshold_pos = error_threshold_workers.find("\"worker_id\":\"error-threshold-worker\"");
    expect(threshold_pos != std::string::npos, "router error threshold worker listed");
    auto next_worker_pos = error_threshold_workers.find("\"worker_id\":", threshold_pos + 1);
    auto circuit_false_pos = error_threshold_workers.find("\"circuit_open\":false", threshold_pos);
    auto total_errors_zero_pos = error_threshold_workers.find("\"total_errors\":0", threshold_pos);
    expect(circuit_false_pos != std::string::npos
        && (next_worker_pos == std::string::npos || circuit_false_pos < next_worker_pos),
        "router invalid error env defaults without opening circuit");
    expect(total_errors_zero_pos != std::string::npos
        && (next_worker_pos == std::string::npos || total_errors_zero_pos < next_worker_pos),
        "router worker HTTP error response does not count as proxy error for non-LLM route");

    const std::string register_transport_error_worker_body =
        "{\"worker_id\":\"embedding-transport-error-worker\","
        "\"label\":\"Embedding Transport Error\","
        "\"url\":\"http://127.0.0.1:18158\","
        "\"port\":18158,"
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"embedding\"],"
        "\"models\":[\"transport-error-model\"],"
        "\"best_tier\":60,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_transport_error_worker_body.size()) + "\r\n\r\n" + register_transport_error_worker_body;
    auto registered_transport_error_worker = request(raw);
    expect(registered_transport_error_worker.find("200 OK") != std::string::npos, "router transport error worker register status");
    const std::string transport_error_embedding_body = "{\"model\":\"transport-error-model\",\"input\":\"hello\"}";
    raw = "POST /v1/embeddings HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(transport_error_embedding_body.size()) + "\r\n\r\n" + transport_error_embedding_body;
    auto transport_error_embedding = request(raw);
    expect(transport_error_embedding.find("500 Internal Server Error") != std::string::npos, "router embedding transport error status");
    expect(transport_error_embedding.find("\r\n\r\nInternal Server Error") != std::string::npos, "router embedding transport error body");
    auto transport_error_workers = request("GET /v1/router/workers HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    auto transport_worker_pos = transport_error_workers.find("\"worker_id\":\"embedding-transport-error-worker\"");
    auto transport_next_pos = transport_error_workers.find("\"worker_id\":", transport_worker_pos + 1);
    auto transport_failures_pos = transport_error_workers.find("\"connection_failures\":1", transport_worker_pos);
    auto transport_errors_pos = transport_error_workers.find("\"total_errors\":1", transport_worker_pos);
    auto transport_kind_pos = transport_error_workers.find("\"kind\":\"connection\"", transport_worker_pos);
    auto transport_status_null_pos = transport_error_workers.find("\"status\":null", transport_worker_pos);
    auto transport_message_pos = transport_error_workers.find("\"message\":\"ClientConnectorError: Cannot connect to host 127.0.0.1:18158 ssl:default [Connect call failed ('127.0.0.1', 18158)]\"", transport_worker_pos);
    expect(transport_worker_pos != std::string::npos
        && transport_failures_pos != std::string::npos
        && transport_errors_pos != std::string::npos
        && transport_kind_pos != std::string::npos
        && transport_status_null_pos != std::string::npos
        && transport_message_pos != std::string::npos
        && (transport_next_pos == std::string::npos
            || (transport_failures_pos < transport_next_pos
                && transport_errors_pos < transport_next_pos
                && transport_kind_pos < transport_next_pos
                && transport_status_null_pos < transport_next_pos
                && transport_message_pos < transport_next_pos)),
        "router embedding transport error records python-like proxy failure");
    auto transport_error_usage = request("GET /v1/router/usage HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(transport_error_usage.find("\"Embedding Transport Error\"") == std::string::npos,
        "router embedding transport error does not record cap usage");
    const std::string deregister_transport_error_worker_body = "{\"worker_id\":\"embedding-transport-error-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_transport_error_worker_body.size()) + "\r\n\r\n" + deregister_transport_error_worker_body;
    auto deregistered_transport_error_worker = request(raw);
    expect(deregistered_transport_error_worker.find("200 OK") != std::string::npos, "router transport error worker deregister status");

    const std::string deregister_error_threshold_worker_body = "{\"worker_id\":\"error-threshold-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_error_threshold_worker_body.size()) + "\r\n\r\n" + deregister_error_threshold_worker_body;
    auto deregistered_error_threshold_worker = request(raw);
    expect(deregistered_error_threshold_worker.find("200 OK") != std::string::npos, "router error threshold worker deregister status");
    unsetenv("ROUTER_ERROR_WINDOW_SECONDS");
    unsetenv("ROUTER_ERROR_THRESHOLD");

    std::string live_tunnel_handshake;
    int live_tunnel_fd = open_websocket("GET /v1/router/tunnel?worker_id=home-node HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n", &live_tunnel_handshake);
    expect(live_tunnel_handshake.find("101 Switching Protocols") != std::string::npos, "router tunnel live websocket handshake");
    std::string orphan_tunnel_handshake;
    int orphan_tunnel_fd = open_websocket("GET /v1/router/tunnel?worker_id=orphan-node HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n", &orphan_tunnel_handshake);
    expect(orphan_tunnel_handshake.find("101 Switching Protocols") != std::string::npos, "router orphan tunnel websocket handshake");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    const std::string tunnel_chat_body = "{\"model\":\"tunnel-model\",\"messages\":[{\"role\":\"user\",\"content\":\"via tunnel\"}]}";
    std::string tunnel_proxy_response;
    std::thread tunnel_proxy_request_thread([&]() {
        tunnel_proxy_response = request("POST /v1/chat/completions HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(tunnel_chat_body.size()) + "\r\n\r\n" + tunnel_chat_body);
    });
    const std::string tunnel_request_frame = recv_websocket_text(live_tunnel_fd);
    expect(tunnel_request_frame.find("\"t\":\"req\"") != std::string::npos, "router tunnel proxy request frame type");
    expect(tunnel_request_frame.find("\"method\":\"POST\"") != std::string::npos, "router tunnel proxy request method");
    expect(tunnel_request_frame.find("\"path\":\"/v1/chat/completions\"") != std::string::npos, "router tunnel proxy request path");
    expect(tunnel_request_frame.find("\"Authorization\":\"Bearer worker-secret\"") != std::string::npos, "router tunnel proxy worker auth");
    expect(tunnel_request_frame.find("\"body_b64\":") != std::string::npos, "router tunnel proxy body encoding");
    const std::string tunnel_request_id = json_string_field(tunnel_request_frame, "id");
    expect(!tunnel_request_id.empty(), "router tunnel proxy request id");
    send_websocket_text(live_tunnel_fd, "{\"t\":\"resp_start\",\"id\":\"" + tunnel_request_id + "\",\"status\":200,\"headers\":{\"Content-Type\":\"application/json\"},\"stream\":true}");
    send_websocket_text(live_tunnel_fd, "{\"t\":\"resp_chunk\",\"id\":\"" + tunnel_request_id + "\",\"data_b64\":\"eyJpZCI6ImNoYXRjbXBsLXR1bm5lbCIsImNob2ljZXMiOltdfQ==\"}");
    send_websocket_text(live_tunnel_fd, "{\"t\":\"resp_end\",\"id\":\"" + tunnel_request_id + "\"}");
    tunnel_proxy_request_thread.join();
    expect(tunnel_proxy_response.find("200 OK") != std::string::npos, "router tunnel proxy response status");
    expect(tunnel_proxy_response.find("\"id\":\"chatcmpl-tunnel\"") != std::string::npos, "router tunnel proxy response body");

    setenv("ROUTER_MAX_RETRIES", "0", 1);
    const std::string tunnel_pre_header_error_body = "{\"model\":\"tunnel-model\",\"prompt\":\"pre header tunnel error\"}";
    std::string tunnel_pre_header_error_response;
    std::thread tunnel_pre_header_error_thread([&]() {
        tunnel_pre_header_error_response = request("POST /v1/completions HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(tunnel_pre_header_error_body.size()) + "\r\n\r\n" + tunnel_pre_header_error_body);
    });
    const std::string tunnel_pre_header_error_frame = recv_websocket_text(live_tunnel_fd);
    const std::string tunnel_pre_header_error_request_id = json_string_field(tunnel_pre_header_error_frame, "id");
    expect(!tunnel_pre_header_error_request_id.empty(), "router tunnel pre-header error request id");
    send_websocket_text(live_tunnel_fd, "{\"t\":\"resp_err\",\"id\":\"" + tunnel_pre_header_error_request_id + "\",\"error\":\"pre-header tunnel boom\"}");
    tunnel_pre_header_error_thread.join();
    expect(tunnel_pre_header_error_response.find("502 Bad Gateway") != std::string::npos, "router tunnel pre-header error response status");
    expect(tunnel_pre_header_error_response.find("RuntimeError: pre-header tunnel boom") != std::string::npos, "router tunnel pre-header error mirrors python runtime error");

    const std::string tunnel_post_header_error_body = "{\"model\":\"tunnel-model\",\"prompt\":\"post header tunnel error\"}";
    std::string tunnel_post_header_error_response;
    std::thread tunnel_post_header_error_thread([&]() {
        tunnel_post_header_error_response = request("POST /v1/completions HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(tunnel_post_header_error_body.size()) + "\r\n\r\n" + tunnel_post_header_error_body);
    });
    const std::string tunnel_post_header_error_frame = recv_websocket_text(live_tunnel_fd);
    const std::string tunnel_post_header_error_request_id = json_string_field(tunnel_post_header_error_frame, "id");
    expect(!tunnel_post_header_error_request_id.empty(), "router tunnel post-header error request id");
    send_websocket_text(live_tunnel_fd, "{\"t\":\"resp_start\",\"id\":\"" + tunnel_post_header_error_request_id + "\",\"status\":206,\"headers\":{\"Content-Type\":\"application/json\"},\"stream\":false}");
    send_websocket_text(live_tunnel_fd, "{\"t\":\"resp_chunk\",\"id\":\"" + tunnel_post_header_error_request_id + "\",\"data_b64\":\"UEFSVElBTC1UVU5ORUw=\"}");
    send_websocket_text(live_tunnel_fd, "{\"t\":\"resp_err\",\"id\":\"" + tunnel_post_header_error_request_id + "\",\"error\":\"post-header tunnel boom\"}");
    tunnel_post_header_error_thread.join();
    unsetenv("ROUTER_MAX_RETRIES");
    expect(tunnel_post_header_error_response.find("206 Partial Content") != std::string::npos, "router tunnel post-header error preserves response status");
    expect(tunnel_post_header_error_response.find("PARTIAL-TUNNEL") != std::string::npos, "router tunnel post-header error preserves response body");
    expect(tunnel_post_header_error_response.find("post-header tunnel boom") == std::string::npos, "router tunnel post-header error does not replace started response");

    const std::string tunnel_stream_chat_body = "{\"model\":\"tunnel-model\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"stream via tunnel\"}]}";
    std::string tunnel_stream_chat_response;
    long long tunnel_chat_first_stream_ms = 0;
    std::thread tunnel_stream_chat_thread([&]() {
        tunnel_stream_chat_response = request_observe_first_chunk(
            std::string("POST /v1/chat/completions HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: ") + std::to_string(tunnel_stream_chat_body.size()) + "\r\n\r\n" + tunnel_stream_chat_body,
            "data: early tunnel",
            &tunnel_chat_first_stream_ms
        );
    });
    const std::string tunnel_stream_chat_frame = recv_websocket_text(live_tunnel_fd);
    expect(tunnel_stream_chat_frame.find("\"t\":\"req\"") != std::string::npos, "router tunnel stream chat request frame type");
    expect(tunnel_stream_chat_frame.find("\"path\":\"/v1/chat/completions\"") != std::string::npos, "router tunnel stream chat request path");
    expect(tunnel_stream_chat_frame.find("\"stream\":true") != std::string::npos, "router tunnel stream chat request stream flag");
    const std::string tunnel_stream_chat_request_id = json_string_field(tunnel_stream_chat_frame, "id");
    expect(!tunnel_stream_chat_request_id.empty(), "router tunnel stream chat request id");
    send_websocket_text(live_tunnel_fd, "{\"t\":\"resp_start\",\"id\":\"" + tunnel_stream_chat_request_id + "\",\"status\":200,\"headers\":{\"Content-Type\":\"text/event-stream\"},\"stream\":true}");
    send_websocket_text(live_tunnel_fd, "{\"t\":\"resp_chunk\",\"id\":\"" + tunnel_stream_chat_request_id + "\",\"data_b64\":\"ZGF0YTogZWFybHkgdHVubmVsCgo=\"}");
    std::this_thread::sleep_for(std::chrono::milliseconds(800));
    send_websocket_text(live_tunnel_fd, "{\"t\":\"resp_chunk\",\"id\":\"" + tunnel_stream_chat_request_id + "\",\"data_b64\":\"ZGF0YTogbGF0ZSB0dW5uZWwKCg==\"}");
    send_websocket_text(live_tunnel_fd, "{\"t\":\"resp_end\",\"id\":\"" + tunnel_stream_chat_request_id + "\"}");
    tunnel_stream_chat_thread.join();
    expect(tunnel_stream_chat_response.find("200 OK") != std::string::npos, "router tunnel stream chat response status");
    expect(tunnel_stream_chat_response.find("Content-Type: text/event-stream") != std::string::npos, "router tunnel stream chat response media type");
    expect(tunnel_stream_chat_response.find("Transfer-Encoding: chunked") != std::string::npos, "router tunnel stream chat chunked transfer");
    expect(tunnel_stream_chat_response.find("data: early tunnel") != std::string::npos, "router tunnel stream chat early chunk");
    expect(tunnel_stream_chat_response.find("data: late tunnel") != std::string::npos, "router tunnel stream chat late chunk");
    expect(tunnel_chat_first_stream_ms < 600, "router tunnel stream chat emitted first chunk before tunnel completed");

    const std::string tunnel_embedding_body = "{\"model\":\"tunnel-model\",\"input\":\"embed via tunnel\"}";
    std::string tunnel_embedding_response;
    std::thread tunnel_embedding_request_thread([&]() {
        tunnel_embedding_response = request("POST /v1/embeddings HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(tunnel_embedding_body.size()) + "\r\n\r\n" + tunnel_embedding_body);
    });
    const std::string tunnel_embedding_frame = recv_websocket_text(live_tunnel_fd);
    expect(tunnel_embedding_frame.find("\"t\":\"req\"") != std::string::npos, "router embedding tunnel request frame type");
    expect(tunnel_embedding_frame.find("\"method\":\"POST\"") != std::string::npos, "router embedding tunnel request method");
    expect(tunnel_embedding_frame.find("\"path\":\"/v1/embeddings\"") != std::string::npos, "router embedding tunnel request path");
    expect(tunnel_embedding_frame.find("\"Content-Type\":\"application/json\"") != std::string::npos, "router embedding tunnel content type");
    expect(tunnel_embedding_frame.find("\"Authorization\":\"Bearer worker-secret\"") != std::string::npos, "router embedding tunnel worker auth");
    expect(test_base64_decode(json_string_field(tunnel_embedding_frame, "body_b64")).find("embed via tunnel") != std::string::npos, "router embedding tunnel body");
    const std::string tunnel_embedding_request_id = json_string_field(tunnel_embedding_frame, "id");
    expect(!tunnel_embedding_request_id.empty(), "router embedding tunnel request id");
    send_websocket_text(live_tunnel_fd, "{\"t\":\"resp_start\",\"id\":\"" + tunnel_embedding_request_id + "\",\"status\":200,\"headers\":{\"Content-Type\":\"application/json\"},\"stream\":false}");
    send_websocket_text(live_tunnel_fd, "{\"t\":\"resp_chunk\",\"id\":\"" + tunnel_embedding_request_id + "\",\"data_b64\":\"eyJvYmplY3QiOiJsaXN0IiwibW9kZWwiOiJ0dW5uZWwtbW9kZWwiLCJkYXRhIjpbeyJvYmplY3QiOiJlbWJlZGRpbmciLCJpbmRleCI6MCwiZW1iZWRkaW5nIjpbMC4zXX1dLCJ1c2FnZSI6eyJwcm9tcHRfdG9rZW5zIjoxLCJ0b3RhbF90b2tlbnMiOjF9fQ==\"}");
    send_websocket_text(live_tunnel_fd, "{\"t\":\"resp_end\",\"id\":\"" + tunnel_embedding_request_id + "\"}");
    tunnel_embedding_request_thread.join();
    expect(tunnel_embedding_response.find("200 OK") != std::string::npos, "router embedding tunnel response status");
    expect(tunnel_embedding_response.find("\"embedding\":[0.3]") != std::string::npos, "router embedding tunnel response body");

    auto router_tunnels = request("GET /v1/router/tunnels HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(router_tunnels.find("200 OK") != std::string::npos, "router tunnels status");
    expect(router_tunnels.find("\"worker_id\":\"tunnel-worker\"") != std::string::npos, "router tunnels includes tunnel worker");
    expect(router_tunnels.find("\"tunnel_worker_id\":\"home-node\"") != std::string::npos, "router tunnels worker id");
    expect(router_tunnels.find("\"connected\":true") != std::string::npos, "router tunnels connected");
    expect(router_tunnels.find("\"connected_at\":null") == std::string::npos, "router tunnels connected timestamp");
    expect(router_tunnels.find("\"connect_count\":1") != std::string::npos, "router tunnels connect count");
    expect(router_tunnels.find("\"orphan_tunnels\":[{\"tunnel_worker_id\":\"orphan-node\"") != std::string::npos, "router tunnels orphan connection");

    auto router_dashboard = request("GET /v1/router/dashboard HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(router_dashboard.find("200 OK") != std::string::npos, "router dashboard status");
    expect(router_dashboard.find("\"pool_health\":\"healthy\"") != std::string::npos, "router dashboard health");
    expect(router_dashboard.find("\"router\":{\"ttl_seconds\":60.0,\"wait_timeout\":120.0}") != std::string::npos, "router dashboard router route default float fields");
    expect(router_dashboard.find("\"alive_workers\":14") != std::string::npos, "router dashboard totals");
    expect(router_dashboard.find("\"capabilities\":[{\"capability\":\"text+vision\"") != std::string::npos, "router dashboard capability sort order");
    expect(router_dashboard.find("\"workers\":[{\"worker_id\":\"tts-stream-worker\"") != std::string::npos, "router dashboard worker tier sort order");
    expect(router_dashboard.find("\"capability\":\"text+vision\"") != std::string::npos, "router dashboard merges text vision");
    expect(router_dashboard.find("\"model\":\"model-b\"") != std::string::npos, "router dashboard model rollup");
    expect(router_dashboard.find("\"best_tier\":80") != std::string::npos, "router dashboard model best tier");
    expect(router_dashboard.find("\"slots_left\":2") != std::string::npos, "router dashboard model slots left");
    expect(router_dashboard.find("\"tunnel\":true") != std::string::npos, "router dashboard includes tunnel flag");
    expect(router_dashboard.find("\"tunnel_connected\":true") != std::string::npos, "router dashboard includes live tunnel connection");

    const std::string register_image_tunnel_worker_body =
        "{\"worker_id\":\"image-tunnel-worker\","
        "\"label\":\"Image Tunnel Worker\","
        "\"url\":\"tunnel://image-node\","
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"image\"],"
        "\"models\":[\"tunnel-image-model\"],"
        "\"free_vram_gb\":250,"
        "\"best_tier\":250,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_image_tunnel_worker_body.size()) + "\r\n\r\n" + register_image_tunnel_worker_body;
    auto registered_image_tunnel = request(raw);
    expect(registered_image_tunnel.find("200 OK") != std::string::npos, "router image tunnel worker register status");
    std::string image_tunnel_handshake;
    int image_tunnel_fd = open_websocket("GET /v1/router/tunnel?worker_id=image-node HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n", &image_tunnel_handshake);
    expect(image_tunnel_handshake.find("101 Switching Protocols") != std::string::npos, "router image tunnel websocket handshake");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::filesystem::remove("outputs/tunnel-image.png");

    const std::string tunnel_image_body = "{\"model\":\"tunnel-image-model\",\"prompt\":\"via tunnel image\"}";
    std::string tunnel_image_response;
    std::thread tunnel_image_request_thread([&]() {
        tunnel_image_response = request("POST /v1/images/generations HTTP/1.1\r\n"
            "Host: localhost:18091\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(tunnel_image_body.size()) + "\r\n\r\n" + tunnel_image_body);
    });
    const std::string image_request_frame = recv_websocket_text(image_tunnel_fd);
    expect(image_request_frame.find("\"t\":\"req\"") != std::string::npos, "router image tunnel request frame type");
    expect(image_request_frame.find("\"method\":\"POST\"") != std::string::npos, "router image tunnel request method");
    expect(image_request_frame.find("\"path\":\"/v1/images/generations\"") != std::string::npos, "router image tunnel request path");
    expect(image_request_frame.find("\"Authorization\":\"Bearer worker-secret\"") != std::string::npos, "router image tunnel worker auth");
    const std::string image_request_id = json_string_field(image_request_frame, "id");
    expect(!image_request_id.empty(), "router image tunnel request id");
    send_websocket_text(image_tunnel_fd, "{\"t\":\"resp_start\",\"id\":\"" + image_request_id + "\",\"status\":200,\"headers\":{\"Content-Type\":\"application/json\"},\"stream\":false}");
    send_websocket_text(image_tunnel_fd, "{\"t\":\"resp_chunk\",\"id\":\"" + image_request_id + "\",\"data_b64\":\"eyJjcmVhdGVkIjo5LCJkYXRhIjpbeyJ1cmwiOiJodHRwOi8vd29ya2VyLmxvY2FsL291dHB1dHMvdHVubmVsLWltYWdlLnBuZyJ9XX0=\"}");
    send_websocket_text(image_tunnel_fd, "{\"t\":\"resp_end\",\"id\":\"" + image_request_id + "\"}");

    const std::string asset_request_frame = recv_websocket_text(image_tunnel_fd);
    expect(asset_request_frame.find("\"t\":\"req\"") != std::string::npos, "router image tunnel asset request frame type");
    expect(asset_request_frame.find("\"method\":\"GET\"") != std::string::npos, "router image tunnel asset request method");
    expect(asset_request_frame.find("\"path\":\"/outputs/tunnel-image.png\"") != std::string::npos, "router image tunnel asset request path");
    expect(asset_request_frame.find("\"Authorization\":\"Bearer worker-secret\"") != std::string::npos, "router image tunnel asset worker auth");
    const std::string asset_request_id = json_string_field(asset_request_frame, "id");
    expect(!asset_request_id.empty(), "router image tunnel asset request id");
    send_websocket_text(image_tunnel_fd, "{\"t\":\"resp_start\",\"id\":\"" + asset_request_id + "\",\"status\":200,\"headers\":{\"Content-Type\":\"image/png\"},\"stream\":false}");
    send_websocket_text(image_tunnel_fd, "{\"t\":\"resp_chunk\",\"id\":\"" + asset_request_id + "\",\"data_b64\":\"VFVOTkVMLVBORw==\"}");
    send_websocket_text(image_tunnel_fd, "{\"t\":\"resp_end\",\"id\":\"" + asset_request_id + "\"}");
    tunnel_image_request_thread.join();
    expect(tunnel_image_response.find("200 OK") != std::string::npos, "router image tunnel response status");
    expect(tunnel_image_response.find("http://worker.local/outputs/tunnel-image.png") == std::string::npos, "router image tunnel rewrites worker url");
    expect(tunnel_image_response.find("\"url\":\"http://localhost:18091/outputs/tunnel-image.png\"") != std::string::npos, "router image tunnel request-base url");
    expect(std::filesystem::exists("outputs/tunnel-image.png"), "router image tunnel asset file saved");
    {
        std::ifstream generated("outputs/tunnel-image.png", std::ios::binary);
        std::string contents((std::istreambuf_iterator<char>(generated)), std::istreambuf_iterator<char>());
        expect(contents == "TUNNEL-PNG", "router image tunnel asset file contents");
    }

    const std::string tunnel_edit_boundary = "native-edit-tunnel-boundary";
    const std::string tunnel_edit_body =
        "--" + tunnel_edit_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"model\"\r\n\r\n"
        "old-tunnel-image-model\r\n"
        "--" + tunnel_edit_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"model\"\r\n\r\n"
        "tunnel-image-model\r\n"
        "--" + tunnel_edit_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"image\"; filename=\"\"\r\n"
        "Content-Type: image/png\r\n\r\n"
        "PNG-TUNNEL-EDIT\r\n"
        "--" + tunnel_edit_boundary + "--\r\n";
    std::string tunnel_edit_response;
    std::thread tunnel_edit_request_thread([&]() {
        tunnel_edit_response = request("POST /v1/images/edits HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: multipart/form-data; boundary=" + tunnel_edit_boundary + "\r\n"
            "Content-Length: " + std::to_string(tunnel_edit_body.size()) + "\r\n\r\n" + tunnel_edit_body);
    });
    const std::string edit_request_frame = recv_websocket_text(image_tunnel_fd);
    expect(edit_request_frame.find("\"t\":\"req\"") != std::string::npos, "router image edit tunnel request frame type");
    expect(edit_request_frame.find("\"method\":\"POST\"") != std::string::npos, "router image edit tunnel request method");
    expect(edit_request_frame.find("\"path\":\"/v1/images/edits\"") != std::string::npos, "router image edit tunnel request path");
    expect(edit_request_frame.find("\"Content-Type\":\"multipart/form-data; boundary=ezlocalai-native-voice-boundary\"") != std::string::npos, "router image edit tunnel normalized content type");
    expect(edit_request_frame.find("\"Authorization\":\"Bearer worker-secret\"") != std::string::npos, "router image edit tunnel worker auth");
    const std::string edit_payload = test_base64_decode(json_string_field(edit_request_frame, "body_b64"));
    expect(edit_payload.find("Content-Disposition: form-data; name=\"image\"; filename=\"file\"") != std::string::npos, "router image edit tunnel empty filename fallback");
    expect(edit_payload.find("Content-Type: image/png\r\n\r\nPNG-TUNNEL-EDIT") != std::string::npos, "router image edit tunnel file body");
    expect(edit_payload.find("name=\"model\"\r\n\r\ntunnel-image-model") != std::string::npos, "router image edit tunnel last model field");
    expect(edit_payload.find("old-tunnel-image-model") == std::string::npos, "router image edit tunnel drops overwritten model field");
    const std::string edit_request_id = json_string_field(edit_request_frame, "id");
    expect(!edit_request_id.empty(), "router image edit tunnel request id");
    send_websocket_text(image_tunnel_fd, "{\"t\":\"resp_start\",\"id\":\"" + edit_request_id + "\",\"status\":200,\"headers\":{\"Content-Type\":\"application/json\"},\"stream\":false}");
    send_websocket_text(image_tunnel_fd, "{\"t\":\"resp_chunk\",\"id\":\"" + edit_request_id + "\",\"data_b64\":\"eyJjcmVhdGVkIjoxMSwiZGF0YSI6W3sidXJsIjoidHVubmVsLWVkaXQucG5nIn1dfQ==\"}");
    send_websocket_text(image_tunnel_fd, "{\"t\":\"resp_end\",\"id\":\"" + edit_request_id + "\"}");
    tunnel_edit_request_thread.join();
    expect(tunnel_edit_response.find("200 OK") != std::string::npos, "router image edit tunnel response status");
    expect(tunnel_edit_response.find("tunnel-edit.png") != std::string::npos, "router image edit tunnel response body");
    close(image_tunnel_fd);
    const std::string deregister_image_tunnel_worker_body = "{\"worker_id\":\"image-tunnel-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_image_tunnel_worker_body.size()) + "\r\n\r\n" + deregister_image_tunnel_worker_body;
    auto deregistered_image_tunnel = request(raw);
    expect(deregistered_image_tunnel.find("200 OK") != std::string::npos, "router image tunnel worker deregister status");

    const std::string register_video_tunnel_worker_body =
        "{\"worker_id\":\"video-tunnel-worker\","
        "\"label\":\"Video Tunnel Worker\","
        "\"url\":\"tunnel://video-node\","
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"video\"],"
        "\"models\":[\"tunnel-video-model\"],"
        "\"free_vram_gb\":255,"
        "\"best_tier\":255,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_video_tunnel_worker_body.size()) + "\r\n\r\n" + register_video_tunnel_worker_body;
    auto registered_video_tunnel = request(raw);
    expect(registered_video_tunnel.find("200 OK") != std::string::npos, "router video tunnel worker register status");
    std::string video_tunnel_handshake;
    int video_tunnel_fd = open_websocket("GET /v1/router/tunnel?worker_id=video-node HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: dmlkZW8tdHVubmVsLWtleQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n", &video_tunnel_handshake);
    expect(video_tunnel_handshake.find("101 Switching Protocols") != std::string::npos, "router video tunnel websocket handshake");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::filesystem::remove("outputs/tunnel-video.mp4");

    const std::string tunnel_video_body = "{\"model\":\"tunnel-video-model\",\"prompt\":\"via tunnel video\"}";
    std::string tunnel_video_response;
    std::thread tunnel_video_request_thread([&]() {
        tunnel_video_response = request("POST /v1/videos/generations HTTP/1.1\r\n"
            "Host: localhost:18091\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(tunnel_video_body.size()) + "\r\n\r\n" + tunnel_video_body);
    });
    const std::string video_request_frame = recv_websocket_text(video_tunnel_fd);
    expect(video_request_frame.find("\"t\":\"req\"") != std::string::npos, "router video tunnel request frame type");
    expect(video_request_frame.find("\"method\":\"POST\"") != std::string::npos, "router video tunnel request method");
    expect(video_request_frame.find("\"path\":\"/v1/videos/generations\"") != std::string::npos, "router video tunnel request path");
    expect(video_request_frame.find("\"Content-Type\":\"application/json\"") != std::string::npos, "router video tunnel content type");
    expect(video_request_frame.find("\"Authorization\":\"Bearer worker-secret\"") != std::string::npos, "router video tunnel worker auth");
    expect(test_base64_decode(json_string_field(video_request_frame, "body_b64")).find("\"model\":\"tunnel-video-model\"") != std::string::npos, "router video tunnel body");
    const std::string video_request_id = json_string_field(video_request_frame, "id");
    expect(!video_request_id.empty(), "router video tunnel request id");
    send_websocket_text(video_tunnel_fd, "{\"t\":\"resp_start\",\"id\":\"" + video_request_id + "\",\"status\":200,\"headers\":{\"Content-Type\":\"application/json\"},\"stream\":false}");
    send_websocket_text(video_tunnel_fd, "{\"t\":\"resp_chunk\",\"id\":\"" + video_request_id + "\",\"data_b64\":\"eyJjcmVhdGVkIjoxMiwiZGF0YSI6W3sidXJsIjoiaHR0cDovL3dvcmtlci5sb2NhbC9vdXRwdXRzL3R1bm5lbC12aWRlby5tcDQifV19\"}");
    send_websocket_text(video_tunnel_fd, "{\"t\":\"resp_end\",\"id\":\"" + video_request_id + "\"}");

    const std::string video_asset_request_frame = recv_websocket_text(video_tunnel_fd);
    expect(video_asset_request_frame.find("\"t\":\"req\"") != std::string::npos, "router video tunnel asset request frame type");
    expect(video_asset_request_frame.find("\"method\":\"GET\"") != std::string::npos, "router video tunnel asset request method");
    expect(video_asset_request_frame.find("\"path\":\"/outputs/tunnel-video.mp4\"") != std::string::npos, "router video tunnel asset request path");
    expect(video_asset_request_frame.find("\"Authorization\":\"Bearer worker-secret\"") != std::string::npos, "router video tunnel asset worker auth");
    const std::string video_asset_request_id = json_string_field(video_asset_request_frame, "id");
    expect(!video_asset_request_id.empty(), "router video tunnel asset request id");
    send_websocket_text(video_tunnel_fd, "{\"t\":\"resp_start\",\"id\":\"" + video_asset_request_id + "\",\"status\":200,\"headers\":{\"Content-Type\":\"video/mp4\"},\"stream\":false}");
    send_websocket_text(video_tunnel_fd, "{\"t\":\"resp_chunk\",\"id\":\"" + video_asset_request_id + "\",\"data_b64\":\"VFVOTkVMLU1QNA==\"}");
    send_websocket_text(video_tunnel_fd, "{\"t\":\"resp_end\",\"id\":\"" + video_asset_request_id + "\"}");
    tunnel_video_request_thread.join();
    expect(tunnel_video_response.find("200 OK") != std::string::npos, "router video tunnel response status");
    expect(tunnel_video_response.find("http://worker.local/outputs/tunnel-video.mp4") == std::string::npos, "router video tunnel rewrites worker url");
    expect(tunnel_video_response.find("\"url\":\"http://localhost:18091/outputs/tunnel-video.mp4\"") != std::string::npos, "router video tunnel request-base url");
    expect(std::filesystem::exists("outputs/tunnel-video.mp4"), "router video tunnel asset file saved");
    {
        std::ifstream generated("outputs/tunnel-video.mp4", std::ios::binary);
        std::string contents((std::istreambuf_iterator<char>(generated)), std::istreambuf_iterator<char>());
        expect(contents == "TUNNEL-MP4", "router video tunnel asset file contents");
    }
    close(video_tunnel_fd);
    const std::string deregister_video_tunnel_worker_body = "{\"worker_id\":\"video-tunnel-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_video_tunnel_worker_body.size()) + "\r\n\r\n" + deregister_video_tunnel_worker_body;
    auto deregistered_video_tunnel = request(raw);
    expect(deregistered_video_tunnel.find("200 OK") != std::string::npos, "router video tunnel worker deregister status");

    const std::string register_stt_tunnel_worker_body =
        "{\"worker_id\":\"stt-tunnel-worker\","
        "\"label\":\"STT Tunnel Worker\","
        "\"url\":\"tunnel://stt-node\","
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"stt\"],"
        "\"models\":[\"whisper-tunnel\"],"
        "\"free_vram_gb\":260,"
        "\"best_tier\":260,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_stt_tunnel_worker_body.size()) + "\r\n\r\n" + register_stt_tunnel_worker_body;
    auto registered_stt_tunnel = request(raw);
    expect(registered_stt_tunnel.find("200 OK") != std::string::npos, "router stt tunnel worker register status");
    std::string stt_tunnel_handshake;
    int stt_tunnel_fd = open_websocket("GET /v1/router/tunnel?worker_id=stt-node HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: c3R0LXR1bm5lbC1wcm9iZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n", &stt_tunnel_handshake);
    expect(stt_tunnel_handshake.find("101 Switching Protocols") != std::string::npos, "router stt tunnel websocket handshake");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::string tunnel_stt_boundary = "native-stt-tunnel-boundary";
    const std::string tunnel_stt_body =
        "--" + tunnel_stt_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"model\"\r\n\r\n"
        "old-whisper-tunnel\r\n"
        "--" + tunnel_stt_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"model\"\r\n\r\n"
        "whisper-tunnel\r\n"
        "--" + tunnel_stt_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"temperature\"\r\n\r\n"
        "2\r\n"
        "--" + tunnel_stt_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"timestamps\"\r\n\r\n"
        "false\r\n"
        "--" + tunnel_stt_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
        "AUDIO-TUNNEL\r\n"
        "--" + tunnel_stt_boundary + "--\r\n";
    std::string tunnel_stt_response;
    std::thread tunnel_stt_request_thread([&]() {
        tunnel_stt_response = request("POST /v1/audio/transcriptions HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: multipart/form-data; boundary=" + tunnel_stt_boundary + "\r\n"
            "Content-Length: " + std::to_string(tunnel_stt_body.size()) + "\r\n\r\n" + tunnel_stt_body);
    });
    const std::string stt_request_frame = recv_websocket_text(stt_tunnel_fd);
    expect(stt_request_frame.find("\"t\":\"req\"") != std::string::npos, "router stt tunnel request frame type");
    expect(stt_request_frame.find("\"method\":\"POST\"") != std::string::npos, "router stt tunnel request method");
    expect(stt_request_frame.find("\"path\":\"/v1/audio/transcriptions\"") != std::string::npos, "router stt tunnel request path");
    expect(stt_request_frame.find("\"Content-Type\":\"multipart/form-data; boundary=ezlocalai-native-voice-boundary\"") != std::string::npos, "router stt tunnel normalized content type");
    expect(stt_request_frame.find("\"Authorization\":\"Bearer worker-secret\"") != std::string::npos, "router stt tunnel worker auth");
    const std::string stt_payload = test_base64_decode(json_string_field(stt_request_frame, "body_b64"));
    expect(stt_payload.find("Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"") != std::string::npos, "router stt tunnel filename preserved");
    expect(stt_payload.find("Content-Type: audio/wav\r\n\r\nAUDIO-TUNNEL") != std::string::npos, "router stt tunnel file body");
    expect(stt_payload.find("name=\"model\"\r\n\r\nwhisper-tunnel") != std::string::npos, "router stt tunnel last model field");
    expect(stt_payload.find("old-whisper-tunnel") == std::string::npos, "router stt tunnel drops overwritten model field");
    expect(stt_payload.find("name=\"response_format\"\r\n\r\njson") != std::string::npos, "router stt tunnel response_format default");
    expect(stt_payload.find("name=\"temperature\"\r\n\r\n2.0") != std::string::npos, "router stt tunnel temperature normalization");
    expect(stt_payload.find("name=\"timestamps\"\r\n\r\nFalse") != std::string::npos, "router stt tunnel timestamp bool normalization");
    const std::string stt_request_id = json_string_field(stt_request_frame, "id");
    expect(!stt_request_id.empty(), "router stt tunnel request id");
    send_websocket_text(stt_tunnel_fd, "{\"t\":\"resp_start\",\"id\":\"" + stt_request_id + "\",\"status\":200,\"headers\":{\"Content-Type\":\"application/json\"},\"stream\":false}");
    send_websocket_text(stt_tunnel_fd, "{\"t\":\"resp_chunk\",\"id\":\"" + stt_request_id + "\",\"data_b64\":\"eyJ0ZXh0IjoidHVubmVsZWQgdHJhbnNjcmlwdCJ9\"}");
    send_websocket_text(stt_tunnel_fd, "{\"t\":\"resp_end\",\"id\":\"" + stt_request_id + "\"}");
    tunnel_stt_request_thread.join();
    expect(tunnel_stt_response.find("200 OK") != std::string::npos, "router stt tunnel response status");
    expect(tunnel_stt_response.find("tunneled transcript") != std::string::npos, "router stt tunnel response body");
    close(stt_tunnel_fd);
    const std::string deregister_stt_tunnel_worker_body = "{\"worker_id\":\"stt-tunnel-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_stt_tunnel_worker_body.size()) + "\r\n\r\n" + deregister_stt_tunnel_worker_body;
    auto deregistered_stt_tunnel = request(raw);
    expect(deregistered_stt_tunnel.find("200 OK") != std::string::npos, "router stt tunnel worker deregister status");

    const std::string register_tts_tunnel_worker_body =
        "{\"worker_id\":\"tts-tunnel-worker\","
        "\"label\":\"TTS Tunnel Worker\","
        "\"url\":\"tunnel://tts-node\","
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"tts\"],"
        "\"models\":[\"tts-tunnel-model\"],"
        "\"free_vram_gb\":265,"
        "\"best_tier\":265,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_tts_tunnel_worker_body.size()) + "\r\n\r\n" + register_tts_tunnel_worker_body;
    auto registered_tts_tunnel = request(raw);
    expect(registered_tts_tunnel.find("200 OK") != std::string::npos, "router tts tunnel worker register status");
    std::string tts_tunnel_handshake;
    int tts_tunnel_fd = open_websocket("GET /v1/router/tunnel?worker_id=tts-node HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: dHRzLXR1bm5lbC1wcm9iZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n", &tts_tunnel_handshake);
    expect(tts_tunnel_handshake.find("101 Switching Protocols") != std::string::npos, "router tts tunnel websocket handshake");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::string tunnel_voice_response;
    std::thread tunnel_voice_request_thread([&]() {
        tunnel_voice_response = request("GET /v1/audio/voices HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    });
    const std::string voice_request_frame = recv_websocket_text(tts_tunnel_fd);
    expect(voice_request_frame.find("\"t\":\"req\"") != std::string::npos, "router voices tunnel request frame type");
    expect(voice_request_frame.find("\"method\":\"GET\"") != std::string::npos, "router voices tunnel request method");
    expect(voice_request_frame.find("\"path\":\"/v1/audio/voices\"") != std::string::npos, "router voices tunnel request path");
    expect(voice_request_frame.find("\"Authorization\":\"Bearer worker-secret\"") != std::string::npos, "router voices tunnel worker auth");
    expect(voice_request_frame.find("\"body_b64\"") == std::string::npos, "router voices tunnel omits body");
    const std::string voice_request_id = json_string_field(voice_request_frame, "id");
    expect(!voice_request_id.empty(), "router voices tunnel request id");
    send_websocket_text(tts_tunnel_fd, "{\"t\":\"resp_start\",\"id\":\"" + voice_request_id + "\",\"status\":200,\"headers\":{\"Content-Type\":\"application/json\"},\"stream\":false}");
    send_websocket_text(tts_tunnel_fd, "{\"t\":\"resp_chunk\",\"id\":\"" + voice_request_id + "\",\"data_b64\":\"eyJ2b2ljZXMiOlsidHVubmVsLWFteSIseyJpZCI6InR1bm5lbC1ib2IiLCJuYW1lIjoiVHVubmVsIEJvYiJ9XX0=\"}");
    send_websocket_text(tts_tunnel_fd, "{\"t\":\"resp_end\",\"id\":\"" + voice_request_id + "\"}");
    tunnel_voice_request_thread.join();
    expect(tunnel_voice_response.find("200 OK") != std::string::npos, "router voices tunnel response status");
    expect(tunnel_voice_response.find("\"tunnel-amy\"") != std::string::npos, "router voices tunnel string voice");
    expect(tunnel_voice_response.find("\"id\":\"tunnel-bob\"") != std::string::npos, "router voices tunnel object voice");

    const std::string tunnel_tts_stream_body = "{\"model\":\"tts-tunnel-model\",\"input\":\"stream via tunnel\",\"voice\":\"amy\"}";
    std::string tunnel_tts_stream_response;
    std::thread tunnel_tts_stream_request_thread([&]() {
        tunnel_tts_stream_response = request("POST /v1/audio/speech/stream HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(tunnel_tts_stream_body.size()) + "\r\n\r\n" + tunnel_tts_stream_body);
    });
    const std::string tts_stream_request_frame = recv_websocket_text(tts_tunnel_fd);
    expect(tts_stream_request_frame.find("\"t\":\"req\"") != std::string::npos, "router tts stream tunnel request frame type");
    expect(tts_stream_request_frame.find("\"method\":\"POST\"") != std::string::npos, "router tts stream tunnel request method");
    expect(tts_stream_request_frame.find("\"path\":\"/v1/audio/speech/stream\"") != std::string::npos, "router tts stream tunnel request path");
    expect(tts_stream_request_frame.find("\"stream\":true") != std::string::npos, "router tts stream tunnel stream flag");
    expect(tts_stream_request_frame.find("\"Authorization\":\"Bearer worker-secret\"") != std::string::npos, "router tts stream tunnel worker auth");
    const std::string tts_stream_request_id = json_string_field(tts_stream_request_frame, "id");
    expect(!tts_stream_request_id.empty(), "router tts stream tunnel request id");
    send_websocket_text(tts_tunnel_fd, "{\"t\":\"resp_start\",\"id\":\"" + tts_stream_request_id + "\",\"status\":200,\"headers\":{\"Content-Type\":\"application/octet-stream\"},\"stream\":true}");
    send_websocket_text(tts_tunnel_fd, "{\"t\":\"resp_chunk\",\"id\":\"" + tts_stream_request_id + "\",\"data_b64\":\"U1RSRUFNLVRVTk5FTA==\"}");
    send_websocket_text(tts_tunnel_fd, "{\"t\":\"resp_end\",\"id\":\"" + tts_stream_request_id + "\"}");
    tunnel_tts_stream_request_thread.join();
    expect(tunnel_tts_stream_response.find("200 OK") != std::string::npos, "router tts stream tunnel response status");
    expect(tunnel_tts_stream_response.find("Content-Type: application/octet-stream") != std::string::npos, "router tts stream tunnel response media type");
    expect(tunnel_tts_stream_response.find("Transfer-Encoding: chunked") != std::string::npos, "router tts stream tunnel chunked transfer");
    expect(tunnel_tts_stream_response.find("STREAM-TUNNEL") != std::string::npos, "router tts stream tunnel response body");

    std::string tunnel_tts_post_header_error_response;
    std::thread tunnel_tts_post_header_error_thread([&]() {
        tunnel_tts_post_header_error_response = request("POST /v1/audio/speech/stream HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(tunnel_tts_stream_body.size()) + "\r\n\r\n" + tunnel_tts_stream_body);
    });
    const std::string tts_post_header_error_frame = recv_websocket_text(tts_tunnel_fd);
    expect(tts_post_header_error_frame.find("\"path\":\"/v1/audio/speech/stream\"") != std::string::npos, "router tts post-header error tunnel request path");
    const std::string tts_post_header_error_request_id = json_string_field(tts_post_header_error_frame, "id");
    expect(!tts_post_header_error_request_id.empty(), "router tts post-header error tunnel request id");
    send_websocket_text(tts_tunnel_fd, "{\"t\":\"resp_start\",\"id\":\"" + tts_post_header_error_request_id + "\",\"status\":206,\"headers\":{\"Content-Type\":\"application/octet-stream\"},\"stream\":true}");
    send_websocket_text(tts_tunnel_fd, "{\"t\":\"resp_chunk\",\"id\":\"" + tts_post_header_error_request_id + "\",\"data_b64\":\"VFRTLVBBUlRJQUw=\"}");
    send_websocket_text(tts_tunnel_fd, "{\"t\":\"resp_err\",\"id\":\"" + tts_post_header_error_request_id + "\",\"error\":\"tts tunnel post-header boom\"}");
    tunnel_tts_post_header_error_thread.join();
    expect(tunnel_tts_post_header_error_response.find("206 Partial Content") != std::string::npos, "router tts post-header tunnel error preserves response status");
    expect(tunnel_tts_post_header_error_response.find("Content-Type: application/octet-stream") != std::string::npos, "router tts post-header tunnel error keeps stream media type");
    expect(tunnel_tts_post_header_error_response.find("TTS-PARTIAL") != std::string::npos, "router tts post-header tunnel error preserves response body");
    expect(tunnel_tts_post_header_error_response.find("tts tunnel post-header boom") == std::string::npos, "router tts post-header tunnel error does not replace started stream");

    std::string tunnel_tts_incremental_response;
    long long tunnel_first_stream_ms = 0;
    std::thread tunnel_tts_incremental_thread([&]() {
        tunnel_tts_incremental_response = request_observe_first_chunk(
            std::string("POST /v1/audio/speech/stream HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: ") + std::to_string(tunnel_tts_stream_body.size()) + "\r\n\r\n" + tunnel_tts_stream_body,
            "EARLY-TUNNEL",
            &tunnel_first_stream_ms
        );
    });
    const std::string tts_incremental_request_frame = recv_websocket_text(tts_tunnel_fd);
    expect(tts_incremental_request_frame.find("\"t\":\"req\"") != std::string::npos, "router tts incremental tunnel request frame type");
    expect(tts_incremental_request_frame.find("\"path\":\"/v1/audio/speech/stream\"") != std::string::npos, "router tts incremental tunnel request path");
    expect(tts_incremental_request_frame.find("\"stream\":true") != std::string::npos, "router tts incremental tunnel stream flag");
    const std::string tts_incremental_request_id = json_string_field(tts_incremental_request_frame, "id");
    expect(!tts_incremental_request_id.empty(), "router tts incremental tunnel request id");
    send_websocket_text(tts_tunnel_fd, "{\"t\":\"resp_start\",\"id\":\"" + tts_incremental_request_id + "\",\"status\":200,\"headers\":{\"Content-Type\":\"application/octet-stream\"},\"stream\":true}");
    send_websocket_text(tts_tunnel_fd, "{\"t\":\"resp_chunk\",\"id\":\"" + tts_incremental_request_id + "\",\"data_b64\":\"RUFSTFktVFVOTkVM\"}");
    std::this_thread::sleep_for(std::chrono::milliseconds(800));
    send_websocket_text(tts_tunnel_fd, "{\"t\":\"resp_chunk\",\"id\":\"" + tts_incremental_request_id + "\",\"data_b64\":\"TEFURS1UVU5ORUw=\"}");
    send_websocket_text(tts_tunnel_fd, "{\"t\":\"resp_end\",\"id\":\"" + tts_incremental_request_id + "\"}");
    tunnel_tts_incremental_thread.join();
    expect(tunnel_tts_incremental_response.find("200 OK") != std::string::npos, "router tts incremental tunnel response status");
    expect(tunnel_tts_incremental_response.find("Content-Type: application/octet-stream") != std::string::npos, "router tts incremental tunnel response media type");
    expect(tunnel_tts_incremental_response.find("Transfer-Encoding: chunked") != std::string::npos, "router tts incremental tunnel chunked transfer");
    expect(tunnel_tts_incremental_response.find("EARLY-TUNNEL") != std::string::npos, "router tts incremental tunnel early chunk");
    expect(tunnel_tts_incremental_response.find("LATE-TUNNEL") != std::string::npos, "router tts incremental tunnel late chunk");
    expect(tunnel_first_stream_ms < 600, "router tts tunnel stream emitted first chunk before tunnel completed");
    close(tts_tunnel_fd);
    const std::string deregister_tts_tunnel_worker_body = "{\"worker_id\":\"tts-tunnel-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_tts_tunnel_worker_body.size()) + "\r\n\r\n" + deregister_tts_tunnel_worker_body;
    auto deregistered_tts_tunnel = request(raw);
    expect(deregistered_tts_tunnel.find("200 OK") != std::string::npos, "router tts tunnel worker deregister status");

    setenv("ROUTER_TUNNEL_PING_INTERVAL", "0.05", 1);
    setenv("ROUTER_TUNNEL_PONG_TIMEOUT", "0.25", 1);
    const std::string register_keepalive_tunnel_worker_body =
        "{\"worker_id\":\"keepalive-tunnel-worker\","
        "\"label\":\"Keepalive Tunnel Worker\","
        "\"url\":\"tunnel://keepalive-node\","
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"text\"],"
        "\"models\":[\"keepalive-model\"],"
        "\"best_tier\":266,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_keepalive_tunnel_worker_body.size()) + "\r\n\r\n" + register_keepalive_tunnel_worker_body;
    auto registered_keepalive_tunnel = request(raw);
    expect(registered_keepalive_tunnel.find("200 OK") != std::string::npos, "router keepalive tunnel worker register status");
    std::string keepalive_tunnel_handshake;
    int keepalive_tunnel_fd = open_websocket("GET /v1/router/tunnel?worker_id=keepalive-node HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: a2VlcGFsaXZlLXR1bm5lbA==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n", &keepalive_tunnel_handshake);
    expect(keepalive_tunnel_handshake.find("101 Switching Protocols") != std::string::npos, "router keepalive tunnel websocket handshake");
    const std::string keepalive_ping = recv_websocket_text(keepalive_tunnel_fd);
    expect(keepalive_ping.find("\"t\":\"ping\"") != std::string::npos, "router tunnel keepalive sends json ping");
    send_websocket_text(keepalive_tunnel_fd, "{\"t\":\"pong\"}");
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    auto keepalive_connected = request("GET /v1/router/tunnels HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    auto keepalive_pos = keepalive_connected.find("\"worker_id\":\"keepalive-tunnel-worker\"");
    auto keepalive_next_pos = keepalive_connected.find("\"worker_id\":", keepalive_pos + 1);
    auto keepalive_connected_pos = keepalive_connected.find("\"connected\":true", keepalive_pos);
    expect(keepalive_pos != std::string::npos
        && keepalive_connected_pos != std::string::npos
        && (keepalive_next_pos == std::string::npos || keepalive_connected_pos < keepalive_next_pos),
        "router tunnel keepalive pong keeps tunnel connected");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    auto keepalive_closed = request("GET /v1/router/tunnels HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    keepalive_pos = keepalive_closed.find("\"worker_id\":\"keepalive-tunnel-worker\"");
    keepalive_next_pos = keepalive_closed.find("\"worker_id\":", keepalive_pos + 1);
    const auto keepalive_closed_pos = keepalive_closed.find("\"connected\":false", keepalive_pos);
    const auto keepalive_reason_pos = keepalive_closed.find("pong timeout", keepalive_pos);
    expect(keepalive_pos != std::string::npos
        && keepalive_closed_pos != std::string::npos
        && keepalive_reason_pos != std::string::npos
        && (keepalive_next_pos == std::string::npos || (keepalive_closed_pos < keepalive_next_pos && keepalive_reason_pos < keepalive_next_pos)),
        "router tunnel keepalive timeout disconnects silent socket");
    close(keepalive_tunnel_fd);
    const std::string deregister_keepalive_tunnel_worker_body = "{\"worker_id\":\"keepalive-tunnel-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_keepalive_tunnel_worker_body.size()) + "\r\n\r\n" + deregister_keepalive_tunnel_worker_body;
    auto deregistered_keepalive_tunnel = request(raw);
    expect(deregistered_keepalive_tunnel.find("200 OK") != std::string::npos, "router keepalive tunnel worker deregister status");
    unsetenv("ROUTER_TUNNEL_PING_INTERVAL");
    unsetenv("ROUTER_TUNNEL_PONG_TIMEOUT");

    const std::string register_disconnect_tunnel_worker_body =
        "{\"worker_id\":\"disconnect-tunnel-worker\","
        "\"label\":\"Disconnect Tunnel Worker\","
        "\"url\":\"tunnel://disconnect-node\","
        "\"api_key\":\"worker-secret\","
        "\"capabilities\":[\"text\"],"
        "\"models\":[\"disconnect-tunnel-model\"],"
        "\"best_tier\":270,"
        "\"queue_capacity\":1}";
    raw = "POST /v1/router/register HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(register_disconnect_tunnel_worker_body.size()) + "\r\n\r\n" + register_disconnect_tunnel_worker_body;
    auto registered_disconnect_tunnel = request(raw);
    expect(registered_disconnect_tunnel.find("200 OK") != std::string::npos, "router disconnect tunnel worker register status");
    std::string disconnect_tunnel_handshake;
    int disconnect_tunnel_fd = open_websocket("GET /v1/router/tunnel?worker_id=disconnect-node HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n", &disconnect_tunnel_handshake);
    expect(disconnect_tunnel_handshake.find("101 Switching Protocols") != std::string::npos, "router disconnect tunnel websocket handshake");
    setenv("ROUTER_MAX_RETRIES", "0", 1);
    const std::string disconnect_completion_body = "{\"model\":\"disconnect-tunnel-model\",\"prompt\":\"drop\"}";
    std::string disconnect_tunnel_response;
    std::thread disconnect_tunnel_request_thread([&]() {
        disconnect_tunnel_response = request("POST /v1/completions HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Authorization: Bearer secret\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(disconnect_completion_body.size()) + "\r\n\r\n" + disconnect_completion_body);
    });
    const std::string disconnect_request_frame = recv_websocket_text(disconnect_tunnel_fd);
    expect(disconnect_request_frame.find("\"path\":\"/v1/completions\"") != std::string::npos, "router disconnect tunnel request path");
    close(disconnect_tunnel_fd);
    disconnect_tunnel_request_thread.join();
    unsetenv("ROUTER_MAX_RETRIES");
    expect(disconnect_tunnel_response.find("502 Bad Gateway") != std::string::npos, "router disconnect tunnel response status");
    expect(disconnect_tunnel_response.find("All 1 worker attempts failed for /v1/completions") != std::string::npos, "router disconnect tunnel exhausted detail");
    expect(disconnect_tunnel_response.find("RuntimeError:") != std::string::npos, "router disconnect tunnel response mirrors python runtime error");
    expect(disconnect_tunnel_response.find("Tunnel disconnect-node disconnected") != std::string::npos, "router disconnect tunnel response body");
    auto disconnect_router_errors = request("GET /v1/router/errors HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    auto disconnect_tunnel_errors_pos = disconnect_router_errors.find("\"worker_id\":\"disconnect-tunnel-worker\"");
    auto disconnect_tunnel_errors_next_pos = disconnect_router_errors.find("\"worker_id\":", disconnect_tunnel_errors_pos + 1);
    auto disconnect_tunnel_errors_kind_pos = disconnect_router_errors.find("\"kind\":\"tunnel_disconnect\"", disconnect_tunnel_errors_pos);
    expect(disconnect_tunnel_errors_pos != std::string::npos, "router errors includes disconnect tunnel worker");
    expect(disconnect_tunnel_errors_kind_pos == std::string::npos
        || (disconnect_tunnel_errors_next_pos != std::string::npos
            && disconnect_tunnel_errors_kind_pos > disconnect_tunnel_errors_next_pos),
        "router errors does not record dropped in-flight tunnel request");
    const std::string deregister_disconnect_tunnel_worker_body = "{\"worker_id\":\"disconnect-tunnel-worker\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_disconnect_tunnel_worker_body.size()) + "\r\n\r\n" + deregister_disconnect_tunnel_worker_body;
    auto deregistered_disconnect_tunnel = request(raw);
    expect(deregistered_disconnect_tunnel.find("200 OK") != std::string::npos, "router disconnect tunnel worker deregister status");

    close(live_tunnel_fd);
    close(orphan_tunnel_fd);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto router_tunnels_closed = request("GET /v1/router/tunnels HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(router_tunnels_closed.find("\"connected\":false") != std::string::npos, "router tunnels disconnects closed socket");
    expect(router_tunnels_closed.find("\"disconnect_history\":[{") != std::string::npos, "router tunnels disconnect history");
    expect(router_tunnels_closed.find("\"orphan_tunnels\":[]") != std::string::npos, "router tunnels clears closed orphan");
    auto router_dashboard_tunnel_closed = request("GET /v1/router/dashboard HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    auto dashboard_tunnel_worker_pos = router_dashboard_tunnel_closed.rfind("\"worker_id\":\"tunnel-worker\"");
    auto dashboard_tunnel_worker_next_pos = router_dashboard_tunnel_closed.find("\"worker_id\":", dashboard_tunnel_worker_pos + 1);
    auto dashboard_tunnel_false_pos = router_dashboard_tunnel_closed.find("\"tunnel_connected\":false", dashboard_tunnel_worker_pos);
    expect(dashboard_tunnel_worker_pos != std::string::npos
        && dashboard_tunnel_false_pos != std::string::npos
        && (dashboard_tunnel_worker_next_pos == std::string::npos || dashboard_tunnel_false_pos < dashboard_tunnel_worker_next_pos),
        "router dashboard reports disconnected tunnel flag");

    const std::string unknown_heartbeat_body = "{\"worker_id\":\"missing\"}";
    raw = "POST /v1/router/heartbeat HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(unknown_heartbeat_body.size()) + "\r\n\r\n" + unknown_heartbeat_body;
    auto unknown_heartbeat = request(raw);
    expect(unknown_heartbeat.find("410 Gone") != std::string::npos, "router unknown heartbeat status");
    expect(unknown_heartbeat.find("\"should_reregister\":true") != std::string::npos, "router unknown heartbeat body");

    const std::string deregister_body = "{\"worker_id\":\"worker-1\"}";
    raw = "POST /v1/router/deregister HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(deregister_body.size()) + "\r\n\r\n" + deregister_body;
    auto deregistered = request(raw);
    expect(deregistered.find("200 OK") != std::string::npos, "router deregister status");
    expect(deregistered.find("\"ok\":true") != std::string::npos, "router deregister ok");
    unsetenv("ROUTER_MODE");
    unsetenv("ROUTER_WAIT_TIMEOUT");

    restart_app_for_config = false;
    server->stop();
    {
        std::ifstream usage_file("usage-native-test.json", std::ios::binary);
        std::string usage_contents((std::istreambuf_iterator<char>(usage_file)), std::istreambuf_iterator<char>());
        expect(usage_contents.find("\"Usage Worker\"") != std::string::npos, "router usage shutdown flush file worker");
        expect(usage_contents.find("\"embedding\":{\"requests\":1}") != std::string::npos, "router usage shutdown flush cap");
        std::ifstream history_file("usage-native-test.history.json", std::ios::binary);
        std::string history_contents((std::istreambuf_iterator<char>(history_file)), std::istreambuf_iterator<char>());
        expect(history_contents.find("\"worker\":\"Usage Worker\"") != std::string::npos, "router history shutdown flush file worker");
        expect(history_contents.find("\"completion_tokens\":5") != std::string::npos, "router history shutdown flush file tokens");
    }

    const char *saved_router_url_env = getenv("ROUTER_URL");
    const char *saved_router_api_key_env = getenv("ROUTER_API_KEY");
    const char *saved_worker_label_env = getenv("WORKER_LABEL");
    const char *saved_worker_tunnel_env = getenv("WORKER_TUNNEL");
    const char *saved_worker_interval_env = getenv("WORKER_HEARTBEAT_INTERVAL");
    const char *saved_worker_timeout_env = getenv("WORKER_HEARTBEAT_TIMEOUT");
    const char *saved_voice_server_env = getenv("VOICE_SERVER");
    const char *saved_ezlocalai_url_env = getenv("EZLOCALAI_URL");
    const bool had_router_url_env = saved_router_url_env != nullptr;
    const bool had_router_api_key_env = saved_router_api_key_env != nullptr;
    const bool had_worker_label_env = saved_worker_label_env != nullptr;
    const bool had_worker_tunnel_env = saved_worker_tunnel_env != nullptr;
    const bool had_worker_interval_env = saved_worker_interval_env != nullptr;
    const bool had_worker_timeout_env = saved_worker_timeout_env != nullptr;
    const bool had_voice_server_env = saved_voice_server_env != nullptr;
    const bool had_ezlocalai_url_env = saved_ezlocalai_url_env != nullptr;
    const std::string saved_router_url = saved_router_url_env ? saved_router_url_env : "";
    const std::string saved_router_api_key = saved_router_api_key_env ? saved_router_api_key_env : "";
    const std::string saved_worker_label = saved_worker_label_env ? saved_worker_label_env : "";
    const std::string saved_worker_tunnel = saved_worker_tunnel_env ? saved_worker_tunnel_env : "";
    const std::string saved_worker_interval = saved_worker_interval_env ? saved_worker_interval_env : "";
    const std::string saved_worker_timeout = saved_worker_timeout_env ? saved_worker_timeout_env : "";
    const std::string saved_voice_server = saved_voice_server_env ? saved_voice_server_env : "";
    const std::string saved_ezlocalai_url = saved_ezlocalai_url_env ? saved_ezlocalai_url_env : "";
    std::vector<std::string> worker_router_requests;
    std::thread fake_worker_router(
        fake_worker_router_register_and_deregister,
        18119,
        &worker_router_requests
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    setenv("ROUTER_URL", "http://127.0.0.1:18119", 1);
    setenv("ROUTER_API_KEY", "router-secret", 1);
    setenv("WORKER_LABEL", "NativeWorker", 1);
    setenv("WORKER_TUNNEL", "false", 1);
    setenv("WORKER_HEARTBEAT_INTERVAL", "2", 1);
    setenv("WORKER_HEARTBEAT_TIMEOUT", "1", 1);
    setenv("VOICE_SERVER", "https://voice.ezlocal.ai", 1);
    unsetenv("EZLOCALAI_URL");
    ezlocalai::NativeServer worker_client_server(
        config,
        std::make_unique<ezlocalai::EchoInferenceBackend>("native-test-model")
    );
    expect(worker_client_server.start("127.0.0.1", 18092, &error), error.c_str());
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    worker_client_server.stop();
    fake_worker_router.join();
    expect(worker_router_requests.size() == 3, "worker heartbeat sends register, heartbeat, and deregister");
    expect(worker_router_requests[0].find("POST /v1/router/register HTTP/1.1") != std::string::npos, "worker register path");
    expect(worker_router_requests[0].find("Authorization: Bearer router-secret") != std::string::npos, "worker register auth");
    expect(worker_router_requests[0].find("\"worker_id\":\"NativeWorker-") != std::string::npos, "worker register id");
    expect(worker_router_requests[0].find("\"label\":\"NativeWorker\"") != std::string::npos, "worker register label");
    expect(worker_router_requests[0].find("\"url\":\"http://localhost:8091\"") != std::string::npos, "worker register callback url");
    expect(worker_router_requests[0].find("\"models\":[\"native-test-model\",\"second-model\"]") != std::string::npos, "worker register models");
    expect(worker_router_requests[0].find("\"capabilities\":[\"text\",\"embedding\"]") != std::string::npos, "worker register capabilities");
    expect(worker_router_requests[0].find("\"cap_models\":{\"embedding\":\"Qwen3-Embedding-0.6B\"}") != std::string::npos, "worker register current embedding model alias");
    expect(worker_router_requests[0].find("\"cap_slots\":{") != std::string::npos, "worker register cap slots");
    expect(worker_router_requests[0].find("\"model_slots\":{") != std::string::npos, "worker register model slots");
    expect(worker_router_requests[0].find("\"gpus\":[") != std::string::npos, "worker register gpus");
    expect(worker_router_requests[1].find("POST /v1/router/heartbeat HTTP/1.1") != std::string::npos, "worker heartbeat path");
    expect(worker_router_requests[1].find("\"capabilities\":[\"text\",\"embedding\"]") != std::string::npos, "worker heartbeat shares current capability detector");
    expect(worker_router_requests[1].find("\"worker_id\":\"NativeWorker-") != std::string::npos, "worker heartbeat id");
    expect(worker_router_requests[2].find("POST /v1/router/deregister HTTP/1.1") != std::string::npos, "worker deregister path");
    expect(worker_router_requests[2].find("\"worker_id\":\"NativeWorker-") != std::string::npos, "worker deregister id");

    std::vector<std::string> worker_tunnel_router_requests;
    std::atomic<bool> tunnel_registered{false};
    std::atomic<bool> tunnel_heartbeated{false};
    std::atomic<bool> tunnel_done{false};
    std::thread fake_worker_tunnel_router(
        fake_worker_router_with_tunnel,
        18120,
        &worker_tunnel_router_requests,
        &tunnel_registered,
        &tunnel_heartbeated,
        &tunnel_done
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    setenv("ROUTER_URL", "http://127.0.0.1:18120", 1);
    setenv("ROUTER_API_KEY", "router-secret", 1);
    setenv("WORKER_LABEL", "TunnelWorker", 1);
    setenv("WORKER_TUNNEL", "true", 1);
    setenv("WORKER_HEARTBEAT_INTERVAL", "2", 1);
    setenv("WORKER_HEARTBEAT_TIMEOUT", "1", 1);
    setenv("VOICE_SERVER", "https://voice.ezlocal.ai", 1);
    setenv("EZLOCALAI_URL", "http://127.0.0.1:18093", 1);
    ezlocalai::NativeServer worker_tunnel_server(
        config,
        std::make_unique<ezlocalai::EchoInferenceBackend>("native-test-model")
    );
    expect(worker_tunnel_server.start("127.0.0.1", 18093, &error), error.c_str());
    for (int i = 0; i < 100
        && !(tunnel_registered.load() && tunnel_heartbeated.load() && tunnel_done.load());
        ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    expect(tunnel_registered.load(), "worker tunnel sends register");
    expect(tunnel_heartbeated.load(), "worker tunnel sends heartbeat");
    expect(tunnel_done.load(), "worker tunnel forwards router request");
    worker_tunnel_server.stop();
    fake_worker_tunnel_router.join();
    auto tunnel_request_seen = [&](const std::string &needle) {
        return std::any_of(worker_tunnel_router_requests.begin(), worker_tunnel_router_requests.end(), [&](const std::string &request_raw) {
            return request_raw.find(needle) != std::string::npos;
        });
    };
    expect(tunnel_request_seen("GET /v1/router/tunnel?worker_id=TunnelWorker-"), "worker tunnel websocket path");
    expect(tunnel_request_seen("Authorization: Bearer router-secret"), "worker tunnel router auth header");
    expect(tunnel_request_seen("\"url\":\"tunnel://TunnelWorker-"), "worker tunnel register callback url");
    expect(tunnel_request_seen("POST /v1/router/deregister HTTP/1.1"), "worker tunnel deregister path");

    if (had_router_url_env) { setenv("ROUTER_URL", saved_router_url.c_str(), 1); } else { unsetenv("ROUTER_URL"); }
    if (had_router_api_key_env) { setenv("ROUTER_API_KEY", saved_router_api_key.c_str(), 1); } else { unsetenv("ROUTER_API_KEY"); }
    if (had_worker_label_env) { setenv("WORKER_LABEL", saved_worker_label.c_str(), 1); } else { unsetenv("WORKER_LABEL"); }
    if (had_worker_tunnel_env) { setenv("WORKER_TUNNEL", saved_worker_tunnel.c_str(), 1); } else { unsetenv("WORKER_TUNNEL"); }
    if (had_worker_interval_env) { setenv("WORKER_HEARTBEAT_INTERVAL", saved_worker_interval.c_str(), 1); } else { unsetenv("WORKER_HEARTBEAT_INTERVAL"); }
    if (had_worker_timeout_env) { setenv("WORKER_HEARTBEAT_TIMEOUT", saved_worker_timeout.c_str(), 1); } else { unsetenv("WORKER_HEARTBEAT_TIMEOUT"); }
    if (had_voice_server_env) { setenv("VOICE_SERVER", saved_voice_server.c_str(), 1); } else { unsetenv("VOICE_SERVER"); }
    if (had_ezlocalai_url_env) { setenv("EZLOCALAI_URL", saved_ezlocalai_url.c_str(), 1); } else { unsetenv("EZLOCALAI_URL"); }

    ezl_env_config slow_config = config;
    std::snprintf(slow_config.default_model, sizeof(slow_config.default_model), "%s", "slow");
    slow_config.request_timeout_ms = 2000;
    slow_config.max_concurrent_requests = 1;
    slow_config.max_queue_size = 4;
    std::promise<void> release_slow;
    auto slow_backend = std::make_unique<SlowInferenceBackend>(release_slow.get_future().share());
    auto slow_entered = slow_backend->entered.get_future();
    ezlocalai::NativeServer slow_server(slow_config, std::move(slow_backend));
    expect(slow_server.start("127.0.0.1", 18091, &error), error.c_str());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    setenv("QUEUE_WAIT_TIMEOUT", "0.05", 1);
    setenv("FALLBACK_SERVER", "http://127.0.0.1:18118", 1);
    setenv("FALLBACK_API_KEY", "fallback-secret", 1);

    std::thread fake_queue_fallback(
        fake_fallback_forward_once,
        18118,
        "POST /v1/chat/completions HTTP/1.1",
        "Content-Type: application/json",
        "\"max_tokens\":8192",
        "",
        "application/json",
        "{\"id\":\"chatcmpl-fallback\",\"object\":\"chat.completion\",\"model\":\"fallback\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"queue fallback\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}",
        "Authorization: Bearer fallback-secret"
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    const std::string slow_first_body = "{\"model\":\"slow\",\"messages\":[{\"role\":\"user\",\"content\":\"hold queue\"}]}";
    const std::string slow_first_raw = "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(slow_first_body.size()) + "\r\n\r\n" + slow_first_body;
    std::string slow_first_response;
    std::thread slow_first_client([&]() {
        slow_first_response = request(slow_first_raw);
    });
    expect(slow_entered.wait_for(std::chrono::seconds(2)) == std::future_status::ready, "first inference holds queue");
    auto slow_resources = request("GET /v1/resources HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n\r\n");
    expect(slow_resources.find("200 OK") != std::string::npos, "slow resources status");
    expect(
        slow_resources.find("\"model_slots\":{\"slow\":{\"capacity\":1,\"in_flight\":1,\"queued\":0,\"available\":0}}") != std::string::npos,
        "resources tracks active model slot"
    );
    expect(
        slow_resources.find("\"text\":{\"capacity\":1,\"in_flight\":1,\"queued\":0,\"available\":0}") != std::string::npos,
        "resources tracks active text slot"
    );

    const std::string queued_body = "{\"model\":\"slow\",\"messages\":[{\"role\":\"user\",\"content\":\"fallback me\"}]}";
    raw = "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Authorization: Bearer secret\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(queued_body.size()) + "\r\n\r\n" + queued_body;
    auto queued_fallback = request(raw);
    release_slow.set_value();
    fake_queue_fallback.join();
    slow_first_client.join();
    expect(queued_fallback.find("200 OK") != std::string::npos, "queue fallback status");
    expect(queued_fallback.find("queue fallback") != std::string::npos, "queue fallback body");
    expect(slow_first_response.find("slow local") != std::string::npos, "slow first request completed locally");

    slow_server.stop();
    unsetenv("QUEUE_WAIT_TIMEOUT");
    unsetenv("FALLBACK_SERVER");
    unsetenv("FALLBACK_API_KEY");
    unsetenv("USAGE_FILE");
    std::filesystem::current_path(original_cwd);
    std::filesystem::remove_all(test_cwd);
    return 0;
}
