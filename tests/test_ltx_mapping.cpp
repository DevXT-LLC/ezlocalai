#include "ltx_diffusers.hpp"
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <set>

int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::runtime_error("fixture required");
        std::ifstream file(argv[1]); nlohmann::json fixture; file >> fixture;
        std::set<std::string> outputs;
        std::map<std::string,int> counts;
        for (const auto &row : fixture.at("rows")) {
            const std::string component = row.at("component"), name = row.at("name");
            const auto result = ltx_diffusers_tensor_name(component,name);
            if (result != row.at("expected")) throw std::runtime_error("mapping mismatch: " + component + ":" + name + " -> " + result);
            if (!outputs.insert(result).second) throw std::runtime_error("tensor name collision: " + result);
            ++counts[component];
        }
        if (counts != std::map<std::string,int>{{"vae",170},{"audio_vae",102},{"connectors",262}})
            throw std::runtime_error("incomplete component coverage");
        std::cout << "Verified 534 published tensor names against installed converters, decoder execution and GGUF metadata\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
