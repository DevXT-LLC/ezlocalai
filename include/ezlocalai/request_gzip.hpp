#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ezlocalai {

// Independent limits on received gzip bytes and the decoded JSON request.
inline constexpr size_t http_request_gzip_max_bytes = 16 * 1024 * 1024;

enum class HttpRequestGzipFailure { Malformed, TooLarge };

class HttpRequestGzipError : public std::runtime_error {
public:
    explicit HttpRequestGzipError(HttpRequestGzipFailure reason)
        : std::runtime_error(reason == HttpRequestGzipFailure::TooLarge
              ? "Gzip request body exceeds 16 MiB" : "Malformed gzip request body"), reason_(reason) {}
    HttpRequestGzipFailure reason() const noexcept { return reason_; }
private:
    HttpRequestGzipFailure reason_;
};

// A request must contain exactly one complete gzip member, with a valid CRC
// and length trailer. Unlike response compatibility decoding, trailing data,
// concatenated members and incomplete gzip streams are rejected.
std::string decode_http_request_gzip(std::string_view encoded);

}
