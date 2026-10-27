#include "ezlocalai/request_gzip.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <zlib.h>

namespace ezlocalai {

std::string decode_http_request_gzip(std::string_view encoded) {
    if (encoded.size() > http_request_gzip_max_bytes)
        throw HttpRequestGzipError(HttpRequestGzipFailure::TooLarge);
    static_assert(http_request_gzip_max_bytes <= std::numeric_limits<uInt>::max());
    z_stream stream{};
    if (inflateInit2(&stream, MAX_WBITS + 16) != Z_OK)
        throw std::runtime_error("Cannot initialize gzip request decoder");
    struct Cleanup {
        z_stream &stream;
        ~Cleanup() { inflateEnd(&stream); }
    } cleanup{stream};
    stream.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(encoded.data()));
    stream.avail_in = static_cast<uInt>(encoded.size());
    std::array<char, 64 * 1024> output;
    std::string decoded;
    for (;;) {
        // Permit one extra output byte only to detect an over-limit stream.
        // No append or reserve can grow the decoded request past the limit.
        const size_t remaining = http_request_gzip_max_bytes - decoded.size();
        const size_t offered = std::min(output.size(), remaining + 1);
        stream.next_out = reinterpret_cast<Bytef *>(output.data());
        stream.avail_out = static_cast<uInt>(offered);
        const int result = inflate(&stream, Z_NO_FLUSH);
        const size_t produced = offered - stream.avail_out;
        if (produced > remaining)
            throw HttpRequestGzipError(HttpRequestGzipFailure::TooLarge);
        decoded.append(output.data(), produced);
        if (result == Z_STREAM_END) {
            if (stream.avail_in != 0)
                throw HttpRequestGzipError(HttpRequestGzipFailure::Malformed);
            return decoded;
        }
        if (result != Z_OK || (produced == 0 && stream.avail_in == 0))
            throw HttpRequestGzipError(HttpRequestGzipFailure::Malformed);
    }
}

}
