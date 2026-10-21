#include "ezlocalai/configuration.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
namespace fs = std::filesystem;
void check(bool yes, const char *message) { if (!yes) throw std::runtime_error(message); }
int main() {
    const auto root = fs::temp_directory_path() / ("ezl-embedding-discovery-" + std::to_string(getpid()));
    struct Cleanup {fs::path path; ~Cleanup(){fs::remove_all(path);}} cleanup{root};
    try {
        unsetenv("LLAMA_CPP_EMBEDDING_MODEL_PATH"); unsetenv("LLAMA_CPP_EMBEDDING_MODEL");
        setenv("EZLOCALAI_DOTENV_PATH", "", 1);
        setenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B-GGUF", 1);
        setenv("EMBEDDING_QUANT_TYPE", "Q8_0", 1); setenv("MODELS_PATH", root.c_str(), 1);
        const auto name = "Qwen3-Embedding-0.6B-Q8_0.gguf";
        fs::create_directories(root / "ace-step"); std::ofstream(root / "ace-step" / name) << "incompatible music engine checkpoint";
        fs::create_directories(root / "Qwen3-Embedding-0.6B");
        const auto expected = root / "Qwen3-Embedding-0.6B" / name;
        std::ofstream(expected) << "embedding checkpoint";
        std::ofstream(root / "Qwen3-Embedding-0.6B" / "mmproj-Q8_0.gguf") << "projector";
        std::ofstream(root / "Qwen3-Embedding-0.6B" / "Qwen3-Embedding-0.6B-Q4_0.gguf") << "wrong quantization";
        auto actual = ezlocalai::embedding_model_config();
        check(actual.path == expected && !actual.acquire, "prefer the model directory over another engine's same-named GGUF");
        setenv("LLAMA_CPP_EMBEDDING_MODEL_PATH", (root / "ace-step" / name).c_str(), 1);
        check(ezlocalai::embedding_model_config().path == root / "ace-step" / name, "explicit file override remains authoritative");
        unsetenv("LLAMA_CPP_EMBEDDING_MODEL_PATH"); fs::remove(expected);
        check(ezlocalai::embedding_model_config().path == root / "ace-step" / name, "native recursive override remains available without a scoped match");
        fs::create_directories(root / "z-copy"); std::ofstream(root / "z-copy" / name) << "another copy";
        check(ezlocalai::embedding_model_config().path == root / "ace-step" / name, "fallback tie selection is stable");
        fs::remove(root / "ace-step" / name); fs::remove(root / "z-copy" / name);
        check(ezlocalai::embedding_model_config().path.empty(), "wrong quantization and projector are not selected");
        std::cout << "Scoped embedding cache, explicit override, recursive fallback, deterministic ties and exclusions passed\n";
        return 0;
    } catch (const std::exception &error) {std::cerr << error.what() << '\n'; return 1;}
}
