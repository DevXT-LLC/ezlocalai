#pragma once
#include "ezlocalai/http_url.hpp"
#include <string>
#include <functional>
#include <memory>
#include <sys/types.h>

namespace ezlocalai {
struct TunnelConnectionUrl {
    HttpUrl url;
    std::string authorization;
};
TunnelConnectionUrl tunnel_connection_url(const std::string &value, const std::string &worker_id, const std::string &api_key = "");
std::string tunnel_websocket_accept(const std::string &key);
bool tunnel_validate_upgrade(const std::string &headers, const std::string &key);
// All socket/TLS access stays on one thread. The cancellation predicate may
// read an atomic shutdown flag from another thread; it never closes a live fd.
class TunnelWebSocketConnection {
public:
    explicit TunnelWebSocketConnection(std::function<bool()> cancelled = {});
    ~TunnelWebSocketConnection();
    bool connect_to(const std::string &url, const std::string &worker_id,
        const std::string &api_key, std::string *initial_buffer, std::string *error);
    bool send_text(const std::string &payload);
    bool send_pong(const std::string &payload = "");
    bool send_close(int code = 1000);
    // The worker event loop retains one masked frame across short write slices.
    // A slice returns 1 when complete, 0 on backpressure, and -1 on failure.
    void queue_frame(unsigned char opcode, const std::string &payload);
    int flush_frame();
    bool writing() const;
    bool readable(int milliseconds) const;
    ssize_t read_some(char *buffer, size_t size);
    void close_all();
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
class TunnelReconnectBackoff {
public:
    double disconnected(double duration, bool failed) {
        if (!failed && duration > 30.0) seconds_ = 2.0;
        const double delay = seconds_;
        seconds_ = seconds_ * 1.5 > 15.0 ? 15.0 : seconds_ * 1.5;
        return delay;
    }
private:
    double seconds_ = 2.0;
};
}
