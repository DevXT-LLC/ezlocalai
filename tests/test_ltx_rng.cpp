#include "core/rng_mt19937.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::runtime_error("fixture required");
        std::ifstream input(argv[1]);
        nlohmann::json fixture;
        input >> fixture;
        float worst = 0;
        for (const auto &row : fixture.at("cases")) {
            MT19937RNG generator(row.at("seed").get<uint64_t>());
            for (const auto &draw : row.at("draws")) {
                const auto actual = generator.randn(row.at("size").get<uint32_t>());
                const auto expected = draw.get<std::vector<float>>();
                if (actual.size() != expected.size()) throw std::runtime_error("RNG shape");
                for (size_t i = 0; i < expected.size(); ++i) {
                    const float error = std::abs(actual[i] - expected[i]);
                    worst = std::max(worst, error);
                    if (!std::isfinite(actual[i]) || error > fixture.at("absolute_tolerance").get<float>())
                        throw std::runtime_error("installed CPU RNG mismatch");
                }
            }
        }
        if (fixture.at("cases").size() != 44) throw std::runtime_error("all cases required");
        std::cout << "44 CPU RNG cases / 88 consecutive draws match; max error " << worst << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
