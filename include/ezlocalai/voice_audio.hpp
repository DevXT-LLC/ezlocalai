#pragma once
#include <cmath>
#include <cstdint>
#include <openssl/evp.h>
#include <stdexcept>
#include <string>
#include <vector>
namespace ezlocalai {
inline std::string voice_base64_encode(const std::string &bytes) {
    if (bytes.size() > 4 * 1024 * 1024)
        throw std::runtime_error("Voice audio block exceeds limit");
    std::string out(4 * ((bytes.size() + 2) / 3), '\0');
    if (!bytes.empty())
        EVP_EncodeBlock(reinterpret_cast<unsigned char *>(out.data()),
                        reinterpret_cast<const unsigned char *>(bytes.data()), int(bytes.size()));
    return out;
}
inline std::string voice_base64_decode(const std::string &text) {
    if (text.empty())
        return {};
    if (text.size() > 4 * 1024 * 1024 || text.size() % 4)
        throw std::runtime_error("Invalid voice audio base64 size");
    size_t padding = 0;
    if (text.back() == '=')
        ++padding;
    if (text.size() > 1 && text[text.size() - 2] == '=')
        ++padding;
    for (size_t i = 0; i < text.size() - padding; ++i) {
        const unsigned char c = text[i];
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '+' ||
              c == '/'))
            throw std::runtime_error("Invalid voice audio base64");
    }
    std::string out(text.size() / 4 * 3, '\0');
    int n = EVP_DecodeBlock(reinterpret_cast<unsigned char *>(out.data()),
                            reinterpret_cast<const unsigned char *>(text.data()), int(text.size()));
    if (n < 0)
        throw std::runtime_error("Invalid voice audio base64");
    out.resize(size_t(n) - padding);
    return out;
}
} // namespace ezlocalai
