#include "ezlocalai/request_gzip.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <zlib.h>

namespace {
void expect(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

std::string gzip(const std::string &input, int level = 1) {
    z_stream stream{};
    expect(deflateInit2(&stream, level, Z_DEFLATED, MAX_WBITS + 16, 8, Z_DEFAULT_STRATEGY) == Z_OK,
        "gzip encoder initialization");
    struct Cleanup { z_stream &stream; ~Cleanup() { deflateEnd(&stream); } } cleanup{stream};
    stream.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(input.data()));
    stream.avail_in = static_cast<uInt>(input.size());
    std::array<char, 64 * 1024> buffer;
    std::string output;
    for (;;) {
        stream.next_out = reinterpret_cast<Bytef *>(buffer.data());
        stream.avail_out = buffer.size();
        const int result = deflate(&stream, Z_FINISH);
        output.append(buffer.data(), buffer.size() - stream.avail_out);
        if (result == Z_STREAM_END) return output;
        expect(result == Z_OK, "gzip encoder finish");
    }
}

void rejected(const std::string &input, ezlocalai::HttpRequestGzipFailure expected) {
    try { (void)ezlocalai::decode_http_request_gzip(input); }
    catch (const ezlocalai::HttpRequestGzipError &error) {
        expect(error.reason() == expected, "wrong gzip rejection classification");
        return;
    }
    throw std::runtime_error("invalid or oversized gzip request was accepted");
}
}

int main() {
    try {
        using namespace ezlocalai;
        const std::string binary("\0\xff\x01\x7fquoted\"\\\n", 14);
        for (const std::string &input : {std::string(), std::string("{}"), binary,
                std::string(R"({"messages":[{"content":"café 😃"}],"disable_fallback":true})")}) {
            expect(decode_http_request_gzip(gzip(input)) == input, "decoded payload bytes changed");
        }
        const auto valid = gzip(R"({"prompt":"exact bytes","prompt_cache_key":"a","disable_fallback":true})");
        for (size_t size = 0; size < valid.size(); ++size)
            rejected(valid.substr(0, size), HttpRequestGzipFailure::Malformed);
        rejected("{}", HttpRequestGzipFailure::Malformed);
        rejected(valid + valid, HttpRequestGzipFailure::Malformed);
        rejected(valid + '\0', HttpRequestGzipFailure::Malformed);
        rejected(valid + "trailing", HttpRequestGzipFailure::Malformed);
        auto corrupt_crc = valid;
        corrupt_crc[corrupt_crc.size() - 8] ^= 1;
        rejected(corrupt_crc, HttpRequestGzipFailure::Malformed);
        auto corrupt_length = valid;
        corrupt_length.back() ^= 1;
        rejected(corrupt_length, HttpRequestGzipFailure::Malformed);
        auto reserved_flags = valid;
        reserved_flags[3] |= static_cast<char>(0xe0);
        rejected(reserved_flags, HttpRequestGzipFailure::Malformed);
        auto invalid_method = valid;
        invalid_method[2] = 9;
        rejected(invalid_method, HttpRequestGzipFailure::Malformed);

        const std::string decoded_limit(http_request_gzip_max_bytes, 'x');
        expect(decode_http_request_gzip(gzip(decoded_limit)) == decoded_limit,
            "exact decoded limit rejected");
        rejected(gzip(decoded_limit + 'x'), HttpRequestGzipFailure::TooLarge);

        // Build a valid single member with exactly the encoded limit. Optional
        // gzip FEXTRA bytes do not change its decoded payload or CRC trailer.
        const std::string near_limit(http_request_gzip_max_bytes - 8192, 'w');
        auto encoded_limit = gzip(near_limit, 0);
        const auto padding = http_request_gzip_max_bytes - encoded_limit.size();
        expect(padding >= 2 && padding - 2 <= 65535, "encoded-limit fixture padding");
        const auto extra_size = padding - 2;
        std::string extra;
        extra.push_back(static_cast<char>(extra_size & 255));
        extra.push_back(static_cast<char>((extra_size >> 8) & 255));
        extra.append(extra_size, '\0');
        encoded_limit[3] |= 4;
        encoded_limit.insert(10, extra);
        expect(encoded_limit.size() == http_request_gzip_max_bytes, "encoded-limit fixture size");
        expect(decode_http_request_gzip(encoded_limit) == near_limit, "exact encoded limit rejected");
        rejected(encoded_limit + 'x', HttpRequestGzipFailure::TooLarge);
        expect(decode_http_request_gzip(valid) == R"({"prompt":"exact bytes","prompt_cache_key":"a","disable_fallback":true})",
            "decoder state leaked after rejected requests");
        std::cout << "Request gzip identity, complete-member/CRC/trailer and independent 16 MiB boundaries passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
