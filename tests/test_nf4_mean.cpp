#include "ezlocalai/torch_cuda_mean.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <zlib.h>
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Fixture required");
        std::ifstream input(argv[1]);
        nlohmann::json fixture;
        input >> fixture;
        for (const auto& row : fixture.at("cases")) {
            const std::string hex = row.at("absmax_f32_zlib_hex");
            std::vector<unsigned char> compressed;
            for (size_t i = 0; i < hex.size(); i += 2)
                compressed.push_back(static_cast<unsigned char>(std::stoul(hex.substr(i, 2), nullptr, 16)));
            std::vector<float> values(row.at("count").get<size_t>());
            std::vector<unsigned char> raw(values.size() * 4);
            uLongf size = raw.size();
            if (uncompress(raw.data(), &size, compressed.data(), compressed.size()) != Z_OK || size != raw.size())
                throw std::runtime_error("Invalid statistics buffer");
            for (size_t i = 0; i < values.size(); ++i) {
                uint32_t bits = 0;
                for (size_t byte = 0; byte < 4; ++byte) bits |= uint32_t(raw[i * 4 + byte]) << (byte * 8);
                std::memcpy(&values[i], &bits, 4);
            }
            const float actual = ezlocalai::pytorch_cuda_mean(values, fixture.at("multiprocessors"),
                fixture.at("max_threads_per_multiprocessor"));
            if (actual != row.at("offset").get<float>())
                throw std::runtime_error("CUDA reduction mismatch: " + row.at("tensor").get<std::string>());
        }
        if (fixture.at("cases").size() != 4) throw std::runtime_error("All mean cases required");
        std::cout << "Four real Gemma CUDA mean reductions match exactly\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
