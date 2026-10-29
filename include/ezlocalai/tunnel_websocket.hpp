#pragma once
#include <cstddef>
#include <string>

namespace ezlocalai {
struct TunnelWebSocketMessage {
    unsigned char opcode = 0;
    std::string payload;
};

// The worker negotiates no compression. Match the reference aiohttp reader's
// exclusive 64 MiB message limit, including fragmented message accumulation.
class TunnelWebSocketReader {
public:
    explicit TunnelWebSocketReader(size_t limit = 64 * 1024 * 1024, int expected_mask = -1)
        : limit_(limit), expected_mask_(expected_mask) {}
    bool pop(std::string &wire, TunnelWebSocketMessage &message);
    int error_code() const { return error_code_; }
private:
    size_t limit_;
    int expected_mask_;
    int error_code_ = 0;
    int partial_opcode_ = -1;
    std::string partial_;
    bool fail(std::string &wire, int code);
};
}
