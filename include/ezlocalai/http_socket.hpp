#pragma once

#include <cerrno>
#include <charconv>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <string>
#include <string_view>
#include <sys/socket.h>

namespace ezlocalai::http_socket {

// Streaming clients should receive a complete event as soon as it is ready.
// Unsupported socket types retain their existing behavior.
inline bool low_latency(int fd) noexcept {
    const int enabled = 1;
    return setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled)) == 0;
}

inline bool send_all(int fd, std::string_view bytes) {
    size_t sent = 0;
    while (sent < bytes.size()) {
        const auto count = send(fd, bytes.data() + sent, bytes.size() - sent,
#ifdef MSG_NOSIGNAL
            MSG_NOSIGNAL
#else
            0
#endif
        );
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        sent += static_cast<size_t>(count);
    }
    return true;
}

inline bool send_chunk(int fd, std::string_view chunk) {
    if (chunk.empty()) return true;
    char length[2 * sizeof(size_t)];
    const auto converted = std::to_chars(length, length + sizeof(length), chunk.size(), 16);
    if (converted.ec != std::errc{}) return false;
    const auto length_size = static_cast<size_t>(converted.ptr - length);
    std::string frame;
    frame.reserve(length_size + chunk.size() + 4);
    frame.append(length, length_size);
    frame += "\r\n";
    frame.append(chunk.data(), chunk.size());
    frame += "\r\n";
    // Keep the HTTP length and event in the same write. Sending the tiny length
    // first can leave the event waiting for an ACK when Nagle is enabled.
    return send_all(fd, frame);
}

} // namespace ezlocalai::http_socket
