#include "ezlocalai/http_deadline.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <openssl/err.h>
#include <poll.h>
#include <pthread.h>
#include <stdexcept>
#include <thread>

namespace ezlocalai {
double http_monotonic_time() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::optional<double> http_total_deadline(double total, double started) {
    if (!(total > 0)) return {};
    double when = started + total;
    if (!std::isfinite(when)) throw std::invalid_argument("Invalid HTTP total timeout");
    if (total >= 5) when = std::ceil(when);
    return when;
}

HttpDeadline::HttpDeadline(double total, bool round_deadline, std::function<bool()> cancelled) : cancelled_(std::move(cancelled)) {
    const auto started = http_monotonic_time();
    end_ = http_total_deadline(total, started);
    if (end_ && !round_deadline) end_ = started + total;
}
double HttpDeadline::remaining() const { return end_ ? *end_ - http_monotonic_time() : std::numeric_limits<double>::infinity(); }
bool HttpDeadline::expired() const {
    if (cancelled_ && cancelled_()) { errno = ECANCELED; return true; }
    if (remaining() > 0) return false;
    errno = ETIMEDOUT;
    return true;
}
bool HttpDeadline::wait(int fd, short events) const {
    for (;;) {
        if (expired()) return false;
        const auto seconds = remaining();
        if (seconds <= 0) { errno = ETIMEDOUT; return false; }
        int milliseconds = !end_ ? -1 : seconds >= std::numeric_limits<int>::max()/1000.0 ? std::numeric_limits<int>::max() : static_cast<int>(std::ceil(seconds*1000));
        if (cancelled_) milliseconds = milliseconds < 0 ? 50 : std::min(milliseconds,50);
        pollfd item{fd,events,0};
        const int status = poll(&item,1,milliseconds);
        if (status > 0) return !expired();
        if (status < 0 && errno != EINTR) return false;
        // Interrupted waits and long/rounded poll intervals retain one deadline.
    }
}

HttpAddressResult http_resolve(const std::string &host, const std::string &port, const addrinfo &hints, const HttpDeadline &deadline) {
    struct State {
        std::mutex mutex;
        std::condition_variable cv;
        HttpAddressResult result;
        bool complete = false;
    };
    if (deadline.expired()) return {{},errno == ECANCELED ? EAI_SYSTEM : 0,errno != ECANCELED};
    in6_addr numeric{};
    if (inet_pton(AF_INET,host.c_str(),&numeric) == 1 || inet_pton(AF_INET6,host.c_str(),&numeric) == 1) {
        auto numeric_hints = hints; numeric_hints.ai_flags |= AI_NUMERICHOST | AI_NUMERICSERV;
        addrinfo *raw = nullptr;
        const auto error = getaddrinfo(host.c_str(),port.c_str(),&numeric_hints,&raw);
        return {std::shared_ptr<addrinfo>(raw,[](auto *p) { if (p) freeaddrinfo(p); }),error,deadline.expired()};
    }
    // A timed-out request must not wait for an OS resolver call to return.
    // The detached task owns its inputs and frees any late result itself.
    auto state = std::make_shared<State>();
    std::thread([state,host,port,hints] {
        addrinfo *raw = nullptr;
        const int error = getaddrinfo(host.c_str(),port.c_str(),&hints,&raw);
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->result = {std::shared_ptr<addrinfo>(raw,[](auto *p) { if (p) freeaddrinfo(p); }),error,false};
            state->complete = true;
        }
        state->cv.notify_one();
    }).detach();
    std::unique_lock<std::mutex> lock(state->mutex);
    while (!state->complete) {
        if (deadline.expired()) return {{},errno == ECANCELED ? EAI_SYSTEM : 0,errno != ECANCELED};
        const auto left = deadline.remaining();
        state->cv.wait_for(lock,std::chrono::duration<double>(std::min(left,.05)));
    }
    if (deadline.expired()) return {{},errno == ECANCELED ? EAI_SYSTEM : 0,errno != ECANCELED};
    return state->result;
}

bool http_connect(int fd, const sockaddr *address, socklen_t size, const HttpDeadline &deadline) {
    if (deadline.expired()) return false;
    const int flags = fcntl(fd,F_GETFL,0);
    if (flags < 0 || fcntl(fd,F_SETFL,flags | O_NONBLOCK) < 0) return false;
    if (connect(fd,address,size) == 0) return !deadline.expired();
    if (errno != EINPROGRESS && errno != EINTR) return false;
    if (!deadline.wait(fd,POLLOUT)) return false;
    int error = 0; socklen_t length = sizeof(error);
    if (getsockopt(fd,SOL_SOCKET,SO_ERROR,&error,&length) < 0) return false;
    if (error) { errno = error; return false; }
    return true;
}

namespace {
// OpenSSL's socket BIO does not accept MSG_NOSIGNAL. Protect this thread's
// TLS operations without changing the application's process-wide handlers.
class NoSigpipe {
public:
    explicit NoSigpipe(bool enabled) {
        if (!enabled) return;
        sigemptyset(&signals_); sigaddset(&signals_,SIGPIPE);
        active_ = pthread_sigmask(SIG_BLOCK,&signals_,&previous_) == 0;
        if (active_) {
            sigset_t pending{}; sigpending(&pending);
            consume_ = !sigismember(&pending,SIGPIPE) && !sigismember(&previous_,SIGPIPE);
        }
    }
    ~NoSigpipe() {
        if (!active_) return;
        const int saved = errno;
        if (consume_) {
            timespec immediate{};
            while (sigtimedwait(&signals_,nullptr,&immediate) < 0 && errno == EINTR) {}
        }
        pthread_sigmask(SIG_SETMASK,&previous_,nullptr);
        errno = saved;
    }
private:
    sigset_t signals_{}, previous_{};
    bool active_ = false, consume_ = false;
};
bool tls_retry(SSL *ssl, int fd, int result, const HttpDeadline &deadline) {
    const int error = SSL_get_error(ssl,result);
    if (error == SSL_ERROR_WANT_READ) return deadline.wait(fd,POLLIN);
    if (error == SSL_ERROR_WANT_WRITE) return deadline.wait(fd,POLLOUT);
    if (error == SSL_ERROR_SYSCALL && errno == EINTR) return !deadline.expired();
    errno = EIO;
    return false;
}
}

bool http_tls_connect(SSL *ssl, int fd, const HttpDeadline &deadline) {
    NoSigpipe signals(true);
    for (;;) {
        if (deadline.expired()) return false;
        errno = 0;
        ERR_clear_error();
        const int result = SSL_connect(ssl);
        if (result == 1) return !deadline.expired();
        if (!tls_retry(ssl,fd,result,deadline)) return false;
    }
}

ssize_t http_read(int fd, SSL *ssl, void *buffer, size_t size, const HttpDeadline &deadline) {
    NoSigpipe signals(ssl != nullptr);
    for (;;) {
        if (deadline.expired()) return -1;
        errno = 0;
        if (ssl) ERR_clear_error();
        const auto count = ssl ? SSL_read(ssl,buffer,static_cast<int>(std::min<size_t>(size,16384))) : recv(fd,buffer,size,0);
        if (count > 0) return count;
        if (ssl) {
            if (SSL_get_error(ssl,count) == SSL_ERROR_ZERO_RETURN) return 0;
            if (!tls_retry(ssl,fd,count,deadline)) return -1;
        } else {
            if (count == 0) return 0;
            if (errno == EINTR) continue;
            if ((errno != EAGAIN && errno != EWOULDBLOCK) || !deadline.wait(fd,POLLIN)) return -1;
        }
    }
}

ssize_t http_write(int fd, SSL *ssl, const void *buffer, size_t size, const HttpDeadline &deadline) {
    NoSigpipe signals(ssl != nullptr);
    for (;;) {
        if (deadline.expired()) return -1;
        errno = 0;
        if (ssl) ERR_clear_error();
        const auto count = ssl ? SSL_write(ssl,buffer,static_cast<int>(std::min<size_t>(size,16384))) : send(fd,buffer,size,MSG_NOSIGNAL);
        if (count > 0) return count;
        if (ssl) {
            if (!tls_retry(ssl,fd,count,deadline)) return -1;
        } else {
            if (errno == EINTR) continue;
            if ((errno != EAGAIN && errno != EWOULDBLOCK) || !deadline.wait(fd,POLLOUT)) return -1;
        }
    }
}
}
