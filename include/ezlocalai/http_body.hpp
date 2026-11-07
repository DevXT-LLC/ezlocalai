#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ezlocalai {

class HttpContentDecodingError : public std::runtime_error {
public:
    HttpContentDecodingError() : std::runtime_error("Invalid HTTP content encoding") {}
};

class HttpBodySizeError : public std::runtime_error {
public:
    HttpBodySizeError() : std::runtime_error("HTTP body exceeds configured limit") {}
};

// Incremental HTTP payload framing and the reference aiohttp content decoders.
// Unknown Content-Encoding values pass through; this reference runtime supports
// gzip, deflate and Brotli, and rejects zstd when parsing the response headers.
class HttpBodyDecoder {
public:
    // max_bytes optionally bounds the framed payload before content decoding;
    // bounded callers also reject excessive chunk extensions and trailers.
    HttpBodyDecoder(std::string encoding, bool chunked, std::optional<size_t> length,
        std::optional<size_t> max_bytes = {});
    ~HttpBodyDecoder();
    void feed(std::string_view bytes);
    // Drain decoded bytes without resetting framing or decompression state.
    std::string take();
    bool complete() const;
    const std::string &finish();
private:
    struct State;
    std::unique_ptr<State> state_;
};

}
