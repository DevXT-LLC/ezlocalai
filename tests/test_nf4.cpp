#include "ezlocalai/nf4.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

static std::vector<uint8_t> bytes(const std::string& hex) {
    if (hex.size() % 2) throw std::runtime_error("Odd hex buffer");
    std::vector<uint8_t> result;
    for (size_t i = 0; i < hex.size(); i += 2)
        result.push_back(static_cast<uint8_t>(std::stoul(hex.substr(i, 2), nullptr, 16)));
    return result;
}
static std::vector<uint16_t> bits16(const std::string& hex) {
    const auto raw = bytes(hex);
    if (raw.size() % 2) throw std::runtime_error("Odd 16-bit buffer");
    std::vector<uint16_t> result;
    for (size_t i = 0; i < raw.size(); i += 2) result.push_back(raw[i] | uint16_t(raw[i + 1]) << 8);
    return result;
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Fixture required");
        std::ifstream input(argv[1]);
        nlohmann::json fixture;
        input >> fixture;
        const auto code = fixture.at("dynamic_code").get<std::vector<float>>();
        if (code.size() != ezlocalai::nf4_detail::dynamic_code.size() ||
            !std::equal(code.begin(), code.end(), ezlocalai::nf4_detail::dynamic_code.begin()))
            throw std::runtime_error("Dynamic codebook mismatch");
        const bool half = fixture.at("dtype") == "float16";
        const std::string encoding = half ? "f16" : "bf16";
        auto quantize = half ? ezlocalai::quantize_nf4_f16 : ezlocalai::quantize_nf4_bf16;
        auto dequantize = half ? ezlocalai::dequantize_nf4_f16 : ezlocalai::dequantize_nf4_bf16;
        size_t elements = 0;
        for (const auto& row : fixture.at("cases")) {
            const auto values = bits16(row.at("input_" + encoding + "_hex"));
            const auto expected = bits16(row.at("dequantized_" + encoding + "_hex"));
            if (values.size() != row.at("size") || values.size() != expected.size())
                throw std::runtime_error("Oracle shape mismatch");
            const bool nested = row.at("nested");
            const auto actual = quantize(values.data(), values.size(), nested, std::nullopt);
            const auto cuda = quantize(values.data(), values.size(), nested,
                ezlocalai::Nf4CudaDevice{84, 1536});
            std::vector<uint16_t> decoded(values.size());
            dequantize(actual, decoded.data());
            bool equal = actual.packed == bytes(row.at("quantized_hex")) && decoded == expected;
            dequantize(cuda, decoded.data());
            equal = equal && cuda.packed == actual.packed && decoded == expected &&
                cuda.nested == actual.nested && cuda.offset == actual.offset &&
                cuda.nested_absmax == actual.nested_absmax;
            if (nested) {
                equal = equal && actual.nested == row.at("absmax").get<std::vector<uint8_t>>() &&
                    actual.nested_absmax == row.at("nested_absmax").get<std::vector<float>>() &&
                    actual.offset == row.at("offset").get<float>();
            } else {
                equal = equal && actual.scales == row.at("absmax").get<std::vector<float>>();
            }
            if (!equal) throw std::runtime_error("NF4 mismatch: " + row.at("pattern").get<std::string>() +
                " size " + std::to_string(values.size()) + " nested " + std::to_string(nested));
            elements += values.size();
        }
        if (fixture.at("cases").size() != 136) throw std::runtime_error("All NF4 cases required");
        std::cout << "136 installed CUDA NF4 cases match exactly (" << elements << " " << encoding << " values)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
