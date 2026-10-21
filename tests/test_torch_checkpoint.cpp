#include "ezlocalai/torch_checkpoint.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using Json = nlohmann::json;
void check(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::string unhex(const std::string &value) {
    check(value.size() % 2 == 0, "hex length"); std::string result;
    for (size_t i = 0; i < value.size(); i += 2) result.push_back(static_cast<char>(std::stoi(value.substr(i, 2), nullptr, 16)));
    return result;
}
}
int main(int argc, char **argv) {
    try {
        check(argc == 2, "fixture required"); std::ifstream input(argv[1]); Json fixture; input >> fixture;
        size_t tensors = 0, rejected = 0, truncated = 0;
        for (const auto &row : fixture.at("cases")) {
            const auto name = row.at("name").get<std::string>(), pickle = unhex(row.at("pickle_hex"));
            std::map<std::string, ezlocalai::TorchCheckpointTensor> state;
            bool failed = false;
            try { state = ezlocalai::parse_torch_checkpoint_pickle(pickle); }
            catch (const std::runtime_error &) { failed = true; }
            check(failed == row.contains("error"), name + " source acceptance mismatch");
            if (failed) { ++rejected; continue; }
            check(state.size() == row.at("tensors").size(), name + " state dictionary size");
            for (const auto &entry : state) {
                const auto &value = entry.second;
                const auto &expected = row.at("tensors").at(entry.first);
                check(value.scalar_type == expected.at("dtype"), name + " dtype");
                check(Json(value.shape) == expected.at("shape") && Json(value.strides) == expected.at("strides"), name + " tensor layout");
                check(value.storage_offset == expected.at("storage_offset") && value.storage_elements == expected.at("storage_elements"), name + " storage layout");
                check(row.at("storage_sha256").at(value.storage_key) == expected.at("storage_sha256"), name + " storage identity");
                ++tensors;
            }
            for (size_t i = 0; i < pickle.size(); i += std::max<size_t>(1, pickle.size() / 80)) {
                bool error = false;
                try { ezlocalai::parse_torch_checkpoint_pickle(std::string_view(pickle).substr(0, i)); }
                catch (const std::runtime_error &) { error = true; }
                check(error, name + " truncated input must fail"); ++truncated;
            }
        }
        // A self-referencing metadata dictionary must not leak on either a
        // successful empty state load or a missing-state failure (ASan checks).
        const auto cyclic = unhex("80027d710058100000006d6f64656c5f73746174655f646963747d710173580500000065787472616800732e");
        for (int i = 0; i < 100; ++i) check(ezlocalai::parse_torch_checkpoint_pickle(cyclic).empty(), "cyclic metadata");
        std::cout << fixture.at("cases").size() << " installed Torch cases: " << tensors << " tensor layouts, " << rejected << " rejected checkpoints, " << truncated << " truncated inputs\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
