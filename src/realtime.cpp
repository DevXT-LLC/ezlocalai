#include "ezlocalai/realtime.hpp"
#include "ezlocalai/http_deadline.hpp"
#include <chrono>
#include <cerrno>
#include <poll.h>
#include <stdexcept>

namespace ezlocalai {
void relay_realtime_websocket(int client, TunnelWebSocketConnection &engine,
    std::string from_client, std::string from_engine,
    const std::function<bool()> &stopped, double idle_seconds) {
    constexpr size_t limit = 65536;
    if (from_client.size() > limit || from_engine.size() > limit)
        throw std::runtime_error("Realtime upgrade buffer exceeds limit");
    if (!from_client.empty()) engine.queue_raw(std::move(from_client));
    size_t sent = 0;
    auto last = std::chrono::steady_clock::now();
    bool engine_eof = false;
    char buffer[16384];
    while (!stopped()) {
        const auto age = std::chrono::duration<double>(std::chrono::steady_clock::now() - last).count();
        if (idle_seconds > 0 && age >= idle_seconds) break;
        if (engine.writing()) {
            const auto result = engine.flush_frame();
            if (result < 0) break;
            if (result > 0) last = std::chrono::steady_clock::now();
        }
        if (sent < from_engine.size()) {
            HttpDeadline slice(.005, false, stopped);
            const auto n = http_write(client, nullptr, from_engine.data() + sent, from_engine.size() - sent, slice);
            if (n > 0) { sent += size_t(n); last = std::chrono::steady_clock::now(); }
            else if (n == 0 || errno != ETIMEDOUT) break;
        }
        if (sent == from_engine.size()) {
            from_engine.clear(); sent = 0;
            if (engine_eof) break;
            if (engine.readable(0)) {
                const auto n = engine.read_some(buffer, sizeof(buffer));
                if (n == 0) engine_eof = true;
                if (n > 0) { from_engine.assign(buffer, size_t(n)); last = std::chrono::steady_clock::now(); }
            }
        }
        pollfd incoming{client, short(engine.writing() ? 0 : POLLIN), 0};
        const auto ready = poll(&incoming, 1, 10);
        if (ready < 0 && errno != EINTR) break;
        if (!engine.writing() && ready > 0 && incoming.revents) {
            HttpDeadline slice(.005, false, stopped);
            const auto n = http_read(client, nullptr, buffer, sizeof(buffer), slice);
            if (n > 0) { engine.queue_raw(std::string(buffer, size_t(n))); last = std::chrono::steady_clock::now(); }
            else if (n == 0 || errno != ETIMEDOUT) break;
        }
    }
}
}
