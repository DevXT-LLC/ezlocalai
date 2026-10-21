#ifndef EZLOCALAI_HTTP_DEADLINE_HPP
#define EZLOCALAI_HTTP_DEADLINE_HPP

#include <memory>
#include <functional>
#include <netdb.h>
#include <optional>
#include <openssl/ssl.h>
#include <string>

namespace ezlocalai {

// aiohttp's total timer is monotonic and rounds deadlines for totals >= 5 s.
// Nonpositive totals and NaN disable the timer; positive infinity is invalid.
std::optional<double> http_total_deadline(double total, double started);
double http_monotonic_time();

class HttpDeadline {
public:
    explicit HttpDeadline(double total, bool round_deadline = true, std::function<bool()> cancelled = {});
    bool expired() const;
    double remaining() const;
    bool wait(int fd, short events) const;
private:
    std::optional<double> end_;
    std::function<bool()> cancelled_;
};

struct HttpAddressResult {
    std::shared_ptr<addrinfo> addresses;
    int error = 0;
    bool timed_out = false;
};

HttpAddressResult http_resolve(const std::string &host, const std::string &port, const addrinfo &hints, const HttpDeadline &deadline);
bool http_connect(int fd, const sockaddr *address, socklen_t size, const HttpDeadline &deadline);
bool http_tls_connect(SSL *ssl, int fd, const HttpDeadline &deadline);
ssize_t http_read(int fd, SSL *ssl, void *buffer, size_t size, const HttpDeadline &deadline);
ssize_t http_write(int fd, SSL *ssl, const void *buffer, size_t size, const HttpDeadline &deadline);

}
#endif
