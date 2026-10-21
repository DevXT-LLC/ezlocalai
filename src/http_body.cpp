#include "ezlocalai/http_body.hpp"

#include <algorithm>
#include <array>
#include <brotli/decode.h>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <zlib.h>

namespace ezlocalai {
namespace {
[[noreturn]] void invalid_body() { throw std::runtime_error("Invalid HTTP response payload"); }
[[noreturn]] void invalid_encoding() { throw HttpContentDecodingError(); }
}

struct HttpBodyDecoder::State {
    enum class Frame { Size, Data, DataEnd, Trailers, Done };
    std::string encoding, body, line;
    bool chunked, started = false, ended = false, finished = false, zlib_ready = false;
    std::optional<size_t> length;
    size_t received = 0, chunk_left = 0, end_bytes = 0;
    Frame frame = Frame::Size;
    z_stream z{};
    BrotliDecoderState *brotli = nullptr;

    State(std::string coding, bool chunks, std::optional<size_t> size) : encoding(std::move(coding)),chunked(chunks),length(size) {
        for (auto &c : encoding) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (encoding == "zstd") invalid_encoding();
        if (encoding == "br") { brotli = BrotliDecoderCreateInstance(nullptr,nullptr,nullptr); if (!brotli) throw std::bad_alloc(); }
    }
    ~State() { if (zlib_ready) inflateEnd(&z); if (brotli) BrotliDecoderDestroyInstance(brotli); }

    void decode(std::string_view bytes) {
        if (bytes.empty()) return;
        if (encoding != "gzip" && encoding != "deflate" && encoding != "br") { body.append(bytes); return; }
        std::array<unsigned char,65536> output;
        if (brotli) {
            if (ended) invalid_encoding();
            auto *input = reinterpret_cast<const uint8_t *>(bytes.data()); size_t available = bytes.size();
            for (;;) {
                auto *out = output.data(); size_t free = output.size();
                const auto result = BrotliDecoderDecompressStream(brotli,&available,&input,&free,&out,nullptr);
                body.append(reinterpret_cast<const char *>(output.data()),output.size()-free);
                if (result == BROTLI_DECODER_RESULT_ERROR) invalid_encoding();
                if (result == BROTLI_DECODER_RESULT_SUCCESS) { ended = true; if (available) invalid_encoding(); break; }
                if (result == BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT) break;
            }
            return;
        }
        // Python's zlib object ignores unused data and later gzip members.
        if (ended) return;
        if (!started) {
            const bool raw = encoding == "deflate" && (static_cast<unsigned char>(bytes.front()) & 0xf) != 8;
            const int window = encoding == "gzip" ? MAX_WBITS+16 : raw ? -MAX_WBITS : MAX_WBITS;
            if (inflateInit2(&z,window) != Z_OK) invalid_encoding();
            zlib_ready = true; started = true;
        }
        while (!bytes.empty() && !ended) {
            const auto size = std::min<size_t>(bytes.size(),std::numeric_limits<uInt>::max());
            z.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(bytes.data())); z.avail_in = static_cast<uInt>(size);
            for (;;) {
                z.next_out = output.data(); z.avail_out = output.size();
                const int result = inflate(&z,Z_SYNC_FLUSH);
                body.append(reinterpret_cast<const char *>(output.data()),output.size()-z.avail_out);
                if (result == Z_STREAM_END) { ended = true; break; }
                if (result != Z_OK && result != Z_BUF_ERROR) invalid_encoding();
                if (!z.avail_in && z.avail_out) break;
                if (result == Z_BUF_ERROR && z.avail_out) invalid_encoding();
            }
            bytes.remove_prefix(size);
        }
    }
    void finish_decoding() {
        if (finished) return;
        if (zlib_ready && !ended) {
            std::array<unsigned char,65536> output;
            z.next_in = nullptr; z.avail_in = 0;
            for (;;) {
                z.next_out = output.data(); z.avail_out = output.size();
                const int result = inflate(&z,Z_FINISH);
                body.append(reinterpret_cast<const char *>(output.data()),output.size()-z.avail_out);
                if (result == Z_STREAM_END) { ended = true; break; }
                if (result != Z_OK && result != Z_BUF_ERROR) invalid_encoding();
                if (z.avail_out) break;
            }
        }
        // aiohttp checks EOF only for nonempty deflate, not gzip or Brotli.
        if (encoding == "deflate" && started && !ended) invalid_encoding();
        finished = true;
    }
    void chunk_size() {
        auto value = line.substr(0,line.size()-2); const auto extension = value.find(';');
        if (extension != std::string::npos) value.resize(extension);
        const auto first = value.find_first_not_of(" \t"); const auto last = value.find_last_not_of(" \t");
        if (first == std::string::npos) invalid_body();
        value = value.substr(first,last-first+1); size_t size = 0;
        for (const auto c : value) {
            int digit = c >= '0' && c <= '9' ? c-'0' : c >= 'a' && c <= 'f' ? c-'a'+10 : c >= 'A' && c <= 'F' ? c-'A'+10 : -1;
            if (digit < 0 || size > (std::numeric_limits<size_t>::max()-digit)/16) invalid_body();
            size = size*16+digit;
        }
        chunk_left = size; frame = size ? Frame::Data : Frame::Trailers;
    }
    void feed(std::string_view bytes) {
        if (!chunked) {
            const auto count = length ? std::min(bytes.size(),*length-received) : bytes.size();
            decode(bytes.substr(0,count)); received += count; return;
        }
        while (!bytes.empty() && frame != Frame::Done) {
            if (frame == Frame::Data) {
                const auto count = std::min(chunk_left,bytes.size()); decode(bytes.substr(0,count)); bytes.remove_prefix(count); chunk_left -= count;
                if (!chunk_left) { frame = Frame::DataEnd; end_bytes = 0; }
            } else if (frame == Frame::DataEnd) {
                if (bytes.front() != (end_bytes ? '\n' : '\r')) invalid_body();
                bytes.remove_prefix(1); if (++end_bytes == 2) frame = Frame::Size;
            } else {
                const char next = bytes.front(); bytes.remove_prefix(1); line += next;
                if (next != '\n') continue;
                if (line.size() < 2 || line[line.size()-2] != '\r') invalid_body();
                if (frame == Frame::Size) chunk_size();
                else if (line.size() == 2) frame = Frame::Done;
                line.clear();
            }
        }
    }
    bool complete() const { return chunked ? frame == Frame::Done : length && received == *length; }
};

HttpBodyDecoder::HttpBodyDecoder(std::string encoding, bool chunked, std::optional<size_t> length) : state_(std::make_unique<State>(std::move(encoding),chunked,length)) {}
HttpBodyDecoder::~HttpBodyDecoder() = default;
void HttpBodyDecoder::feed(std::string_view bytes) { state_->feed(bytes); }
std::string HttpBodyDecoder::take() { std::string bytes; bytes.swap(state_->body); return bytes; }
bool HttpBodyDecoder::complete() const { return state_->complete(); }
const std::string &HttpBodyDecoder::finish() {
    if ((state_->chunked || state_->length) && !state_->complete()) invalid_body();
    state_->finish_decoding(); return state_->body;
}
}
