#include "ezlocalai/http_socket.hpp"
#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <future>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
void check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
struct Sockets {
    int listener = -1, server = -1, client = -1;
    Sockets() {
        listener = socket(AF_INET, SOCK_STREAM, 0); check(listener >= 0, "listener");
        sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        check(bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0 && listen(listener, 1) == 0, "listen");
        socklen_t size = sizeof(address); check(getsockname(listener, reinterpret_cast<sockaddr *>(&address), &size) == 0, "address");
        client = socket(AF_INET, SOCK_STREAM, 0);
        check(client >= 0 && connect(client, reinterpret_cast<sockaddr *>(&address), size) == 0, "connect");
        server = accept(listener, nullptr, nullptr); check(server >= 0, "accept");
        timeval timeout{2, 0}; setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(server, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    }
    ~Sockets() { for (auto fd : {server, client, listener}) if (fd >= 0) close(fd); }
};
std::string frame(const std::string &payload) {
    std::ostringstream value; value << std::hex << payload.size() << "\r\n" << payload << "\r\n"; return value.str();
}
double sample(bool coalesced, bool nodelay, bool *delayed_ack_requested) {
    Sockets sockets;
    if (nodelay) check(ezlocalai::http_socket::low_latency(sockets.server), "TCP_NODELAY");
    std::promise<void> ready, consumed;
    auto ready_future = ready.get_future(), consumed_future = consumed.get_future();
    const std::string payload = "data: {\"choices\":[{\"delta\":{\"content\":\"first\"}}]}\n\n";
    const auto expected = frame(payload);
    std::chrono::steady_clock::time_point started;
    bool written = false;
    std::thread writer([&] {
        const bool headers = ezlocalai::http_socket::send_all(sockets.server,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n");
        ready_future.wait();
        started = std::chrono::steady_clock::now();
        if (coalesced) written = headers && ezlocalai::http_socket::send_chunk(sockets.server, payload);
        else {
            std::ostringstream prefix; prefix << std::hex << payload.size() << "\r\n";
            written = headers && ezlocalai::http_socket::send_all(sockets.server, prefix.str())
                && ezlocalai::http_socket::send_all(sockets.server, payload)
                && ezlocalai::http_socket::send_all(sockets.server, "\r\n");
        }
        // FIN would flush buffered bytes and hide the delay being measured.
        consumed_future.wait();
    });
    std::string bytes; char buffer[4096]; bool received = true;
    while (bytes.find("\r\n\r\n") == std::string::npos) {
        const auto count = recv(sockets.client, buffer, sizeof(buffer), 0);
        if (count <= 0) { received = false; break; }
        bytes.append(buffer, static_cast<size_t>(count));
    }
#ifdef TCP_QUICKACK
    const int disabled = 0;
    *delayed_ack_requested &= setsockopt(sockets.client, IPPROTO_TCP, TCP_QUICKACK, &disabled, sizeof(disabled)) == 0;
#else
    *delayed_ack_requested = false;
#endif
    bytes.clear(); ready.set_value();
    while (received && bytes.size() < expected.size()) {
        const auto count = recv(sockets.client, buffer, sizeof(buffer), 0);
        if (count <= 0) { received = false; break; }
        bytes.append(buffer, static_cast<size_t>(count));
    }
    const auto finished = std::chrono::steady_clock::now();
    consumed.set_value(); writer.join();
    check(received && written && bytes == expected, "first event must arrive byte-exactly");
    return std::chrono::duration<double, std::milli>(finished - started).count();
}
}
int main(int argc, char **argv) {
    try {
        const int repeats = argc == 2 ? std::stoi(argv[1]) : 31;
        check(repeats > 0 && repeats <= 1000, "usage: bench_stream_socket [1..1000 repeats]");
        bool delayed_ack_requested = true;
        std::cout << std::setprecision(9) << "{\"scope\":\"controlled loopback first complete SSE event, not model TTFT\",\"variants\":[";
        bool first = true;
        for (const auto mode : {0, 1, 2, 3}) {
            const bool coalesced = mode & 1, nodelay = mode & 2;
            std::vector<double> samples;
            for (int i = 0; i < repeats; ++i) samples.push_back(sample(coalesced, nodelay, &delayed_ack_requested));
            auto sorted = samples; std::sort(sorted.begin(), sorted.end());
            if (!first) std::cout << ',';
            first = false;
            std::cout << "{\"coalesced\":" << (coalesced ? "true" : "false") << ",\"tcp_nodelay\":" << (nodelay ? "true" : "false")
                << ",\"median_ms\":" << sorted[sorted.size()/2] << ",\"samples_ms\":[";
            for (size_t i = 0; i < samples.size(); ++i) { if (i) std::cout << ','; std::cout << samples[i]; }
            std::cout << "]}";
        }
        std::cout << "],\"delayed_ack_requested\":" << (delayed_ack_requested ? "true" : "false") << ",\"wire_bytes_identical\":true}\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
