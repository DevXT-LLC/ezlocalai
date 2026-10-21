#pragma once
#include <arpa/inet.h>
#include <atomic>
#include <fcntl.h>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

// A worker binds after model acquisition, so bind(0)/close is insufficient:
// unrelated HTTP clients/listeners can claim that ephemeral port in between.
// Keep advisory reservations for participating fixture processes and choose
// outside Linux's automatic client/listener allocation range.
inline int fixture_ports(int count = 1) {
    struct Reservations {
        std::mutex mutex;
        std::vector<int> locks;
        ~Reservations() { for (const int fd : locks) close(fd); }
    };
    static Reservations reservations;
    static std::atomic<unsigned> sequence{0};
    int low = 32768, high = 60999;
    std::ifstream("/proc/sys/net/ipv4/ip_local_port_range") >> low >> high;
    if (count <= 0 || count > 32) throw std::runtime_error("invalid fixture port count");
    const auto seed = static_cast<unsigned>(getpid()) * 97u + sequence.fetch_add(1) * 37u;
    std::lock_guard<std::mutex> guard(reservations.mutex);
    for (unsigned attempt = 0; attempt < 64000; ++attempt) {
        const int base = 1024 + (seed + attempt) % 64000;
        if (base + count > 65536 || (base <= high && base + count - 1 >= low)) continue;
        std::vector<int> locks, sockets; bool available = true;
        for (int offset = 0; available && offset < count; ++offset) {
            const auto path = "/tmp/ezlocalai-fixture-port-" + std::to_string(getuid()) + "-" + std::to_string(base + offset);
            const int lock = open(path.c_str(),O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW,0600);
            if (lock < 0) { available = false; break; } locks.push_back(lock);
            if (flock(lock,LOCK_EX | LOCK_NB)) { available = false; break; }
            const int fd = socket(AF_INET,SOCK_STREAM | SOCK_CLOEXEC,0);
            if (fd < 0) { available = false; break; } sockets.push_back(fd);
            sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK); address.sin_port = htons(base + offset);
            available = !bind(fd,reinterpret_cast<sockaddr *>(&address),sizeof(address));
        }
        for (const int fd : sockets) close(fd);
        if (available) { reservations.locks.insert(reservations.locks.end(),locks.begin(),locks.end()); return base; }
        for (const int fd : locks) close(fd);
    }
    throw std::runtime_error("fixture worker ports unavailable outside ephemeral range");
}
