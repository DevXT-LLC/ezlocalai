#pragma once
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <limits>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

namespace ezlocalai::media_wire {
inline constexpr uint32_t max_message = 16 * 1024 * 1024;
using Clock = std::chrono::steady_clock;
using Deadline = Clock::time_point;
struct Timeout : std::runtime_error { Timeout() : std::runtime_error("media worker deadline exceeded") {} };
inline Deadline deadline(double seconds) {
    if (seconds < 0) return Deadline::max();
    const auto now = Clock::now();
    if (seconds >= std::chrono::duration<double>(Deadline::max() - now).count()) return Deadline::max();
    return now + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
}
inline void ready(int fd, short events, Deadline until) {
    for (;;) {
        int ms = -1;
        if (until != Deadline::max()) {
            const auto remaining = until - Clock::now();
            if (remaining <= Clock::duration::zero()) throw Timeout();
            const auto value = std::chrono::duration_cast<std::chrono::milliseconds>(remaining).count();
            ms = static_cast<int>(std::min<int64_t>(std::numeric_limits<int>::max(), value + 1));
        }
        pollfd p{fd, events, 0};
        const auto result = poll(&p, 1, ms);
        if (result > 0) {
            if (p.revents & (events | POLLHUP)) return;
            throw std::runtime_error("media worker socket failed");
        }
        if (result == 0) throw Timeout();
        if (errno != EINTR) throw std::runtime_error("media worker poll failed");
    }
}
inline bool transfer(int fd, void *data, size_t size, bool writing, Deadline until) {
    auto *bytes = static_cast<char *>(data); size_t used = 0;
    while (used < size) {
        ready(fd, writing ? POLLOUT : POLLIN, until);
        const auto n = writing ? send(fd, bytes + used, size - used, MSG_NOSIGNAL | MSG_DONTWAIT)
                               : recv(fd, bytes + used, size - used, MSG_DONTWAIT);
        if (n > 0) { used += static_cast<size_t>(n); continue; }
        if (n == 0 && !writing && used == 0) return false;
        if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        throw std::runtime_error("media worker disconnected");
    }
    return true;
}
inline void send(int fd, const std::string &message, Deadline until = Deadline::max()) {
    if (message.size() > max_message) throw std::length_error("media worker message too large");
    const auto size = static_cast<uint32_t>(message.size());
    unsigned char header[]{static_cast<unsigned char>(size >> 24), static_cast<unsigned char>(size >> 16), static_cast<unsigned char>(size >> 8), static_cast<unsigned char>(size)};
    transfer(fd, header, sizeof(header), true, until);
    transfer(fd, const_cast<char *>(message.data()), message.size(), true, until);
}
inline bool receive(int fd, std::string &message, Deadline until = Deadline::max()) {
    unsigned char header[4];
    if (!transfer(fd, header, sizeof(header), false, until)) return false;
    const uint32_t size = uint32_t(header[0]) << 24 | uint32_t(header[1]) << 16 | uint32_t(header[2]) << 8 | header[3];
    if (size > max_message) throw std::length_error("media worker message too large");
    message.resize(size);
    if (size && !transfer(fd, message.data(), size, false, until)) throw std::runtime_error("truncated media worker message");
    return true;
}
} // namespace ezlocalai::media_wire
