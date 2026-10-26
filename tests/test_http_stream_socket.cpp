#include "ezlocalai/http_socket.hpp"
#include <arpa/inet.h>
#include <chrono>
#include <csignal>
#include <future>
#include <iostream>
#include <pthread.h>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unistd.h>

namespace {
void check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
std::string encoded(const std::string &chunk) {
    std::ostringstream out;
    out << std::hex << chunk.size() << "\r\n" << chunk << "\r\n";
    return out.str();
}
std::string receive_all(int fd) {
    std::string bytes;
    char buffer[4096];
    for (;;) {
        const auto size = recv(fd, buffer, sizeof(buffer), 0);
        if (size < 0 && errno == EINTR) continue;
        check(size >= 0, "receive failed");
        if (!size) return bytes;
        bytes.append(buffer, static_cast<size_t>(size));
    }
}
void framing() {
    int sockets[2];
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "socketpair");
    const std::string payloads[] = {"data: {\"content\":\"雪\"}\n\n", std::string("a\0b\r\nc", 6), "data: [DONE]\n\n"};
    std::string expected;
    check(ezlocalai::http_socket::send_chunk(sockets[0], ""), "empty chunks are no-ops");
    for (const auto &payload : payloads) {
        check(ezlocalai::http_socket::send_chunk(sockets[0], payload), "send chunk");
        expected += encoded(payload);
    }
    check(ezlocalai::http_socket::send_all(sockets[0], "0\r\n\r\n"), "send end marker");
    shutdown(sockets[0], SHUT_WR);
    const auto actual = receive_all(sockets[1]);
    close(sockets[0]); close(sockets[1]);
    check(actual == expected + "0\r\n\r\n", "chunk framing must be byte-exact, including UTF-8 and NULs");
}
volatile std::sig_atomic_t signals_received = 0;
void interrupt(int) { ++signals_received; }
void interrupted_writes() {
    int sockets[2];
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "interrupted socketpair");
    const int small_buffer = 4096;
    check(setsockopt(sockets[0], SOL_SOCKET, SO_SNDBUF, &small_buffer, sizeof(small_buffer)) == 0, "small send buffer");
    timeval timeout{5, 0};
    setsockopt(sockets[0], SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    setsockopt(sockets[1], SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    struct sigaction handler{}, previous{};
    handler.sa_handler = interrupt;
    sigemptyset(&handler.sa_mask);
    check(sigaction(SIGUSR1, &handler, &previous) == 0, "install interrupt handler");
    std::string payload(1024 * 1024, '\0');
    for (size_t i = 0; i < payload.size(); ++i) payload[i] = static_cast<char>(i % 251);
    std::promise<void> started;
    bool written = false;
    std::thread writer([&] {
        started.set_value();
        written = ezlocalai::http_socket::send_chunk(sockets[0], payload);
        shutdown(sockets[0], SHUT_WR);
    });
    started.get_future().wait();
    // With no reader and a tiny send buffer the writer cannot finish. Signals
    // interrupt both a partially completed send and subsequent blocked sends.
    for (int i = 0; i < 10; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        pthread_kill(writer.native_handle(), SIGUSR1);
    }
    std::string actual;
    bool received = true;
    try { actual = receive_all(sockets[1]); } catch (...) { received = false; close(sockets[1]); sockets[1] = -1; }
    writer.join();
    sigaction(SIGUSR1, &previous, nullptr);
    close(sockets[0]); if (sockets[1] >= 0) close(sockets[1]);
    check(received && written && signals_received > 0, "interrupted sends must complete without a disconnect");
    check(actual == encoded(payload), "partial and interrupted writes must preserve every byte");
}
void low_latency() {
    int listener = socket(AF_INET, SOCK_STREAM, 0);
    check(listener >= 0, "TCP listener");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    check(bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0, "TCP bind");
    check(listen(listener, 1) == 0, "TCP listen");
    socklen_t size = sizeof(address);
    check(getsockname(listener, reinterpret_cast<sockaddr *>(&address), &size) == 0, "TCP address");
    int client = socket(AF_INET, SOCK_STREAM, 0);
    check(client >= 0 && connect(client, reinterpret_cast<sockaddr *>(&address), size) == 0, "TCP connect");
    int server = accept(listener, nullptr, nullptr);
    check(server >= 0 && ezlocalai::http_socket::low_latency(server), "enable TCP_NODELAY");
    int enabled = 0; size = sizeof(enabled);
    check(getsockopt(server, IPPROTO_TCP, TCP_NODELAY, &enabled, &size) == 0 && enabled == 1, "TCP_NODELAY applies to the stream socket");
    close(server); close(client); close(listener);
}
void disconnected() {
    int sockets[2];
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "disconnected socketpair");
    close(sockets[1]);
    const bool sent = ezlocalai::http_socket::send_chunk(sockets[0], "data: disconnected\n\n");
    close(sockets[0]);
    check(!sent, "disconnected reader must stop the stream without SIGPIPE");
}
}
int main() {
    try {
        framing(); interrupted_writes(); low_latency(); disconnected();
        std::cout << "HTTP chunk bytes, partial/interrupted writes, TCP_NODELAY and disconnect checks passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
