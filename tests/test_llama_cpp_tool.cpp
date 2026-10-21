#include "ezlocalai/process.hpp"
#include "ezlocalai/engine_revision.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {
void expect(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}
std::string run(const std::vector<std::string> &args) {
    auto result = ezlocalai::run_process(args, true);
    expect(result.status == 0, args[0] + " failed: " + result.output);
    while (!result.output.empty() && result.output.back() == '\n') result.output.pop_back();
    return result.output;
}
void contains(const std::string &value, const std::string &expected) {
    expect(value.find(expected) != std::string::npos, "missing: " + expected + " in:\n" + value);
}
void env(const char *name, const std::string &value) {
    expect(setenv(name, value.c_str(), 1) == 0, std::string("setenv: ") + name);
}
struct TemporaryDirectory {
    fs::path path;
    TemporaryDirectory() {
        std::string pattern = (fs::temp_directory_path() / "ezlocalai-tool-XXXXXX").string();
        char *created = mkdtemp(pattern.data());
        if (!created) throw std::runtime_error("mkdtemp failed");
        path = created;
    }
    ~TemporaryDirectory() {
        std::error_code error;
        fs::remove_all(path, error);
    }
};
}

int main(int argc, char **argv) {
    try {
        expect(argc == 2, "path to llama_cpp_tool required");
        const std::string tool = fs::absolute(argv[1]).string();
        TemporaryDirectory temp;
        // The process launcher preserves metacharacters, while CMake itself
        // cannot configure builds under semicolon-containing directory names.
        expect(run({"/bin/echo", "literal ; $(no-command) `no-command`"}) == "literal ; $(no-command) `no-command`", "literal process arguments");
        const auto root = temp.path / "space directory";
        fs::create_directory(root);
        const auto remote = root / "remote.git";
        const auto seed = root / "seed";
        const auto checkout = root / "checkout";
        const auto build = root / "build";
        const auto initial_cwd = fs::current_path();
        fs::current_path(root);
        for (const char *name : {"LLAMA_CPP_MODEL_PATH", "LLAMA_CPP_MODEL_PATHS", "LLAMA_CPP_EMBEDDING_MODEL",
                "LLAMA_CPP_EMBEDDING_MODEL_PATH", "LLAMA_CPP_EMBEDDING_SERVER_URL", "LLAMA_CPP_EMBEDDING_ALIAS",
                "EMBEDDING_MODEL", "EMBEDDING_MODEL_ALIAS", "EMBEDDING_QUANT_TYPE",
                "EZLOCALAI_LLAMA_CPP_CMAKE_ARGS", "EZLOCALAI_LLAMA_CPP_BUILD_TYPE"}) unsetenv(name);
        run({"git", "init", "--bare", "--initial-branch=master", remote.string()});
        run({"git", "clone", remote.string(), seed.string()});
        auto git = [&](std::vector<std::string> args) {
            args.insert(args.begin(), {"git", "-C", seed.string()});
            return run(args);
        };
        git({"config", "user.email", "native-test@example.invalid"});
        git({"config", "user.name", "native test"});
        git({"config", "commit.gpgsign", "false"});
        std::ofstream(seed / "README.md") << "one\n";
        // Build a real native fixture to exercise configure/build/print-bin.
        std::ofstream(seed / "CMakeLists.txt") <<
            "cmake_minimum_required(VERSION 3.16)\nproject(fixture LANGUAGES CXX)\n"
            "if(NOT TEST_VALUE STREQUAL \"value with spaces\")\nmessage(FATAL_ERROR \"argument split\")\nendif()\n"
            "set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)\nadd_executable(llama-server main.cpp)\n";
        std::ofstream(seed / "main.cpp") << "int main() { return 0; }\n";
        git({"add", "."});
        git({"commit", "-m", "initial"});
        git({"push", "origin", "master"});
        env("EZLOCALAI_LLAMA_CPP_REPO", remote.string());
        env("EZLOCALAI_LLAMA_CPP_REF", "master");
        env("EZLOCALAI_LLAMA_CPP_DIR", checkout.string());
        env("EZLOCALAI_LLAMA_CPP_BUILD_DIR", build.string());
        run({tool, "update"});
        const auto first_sha = run({"git", "-C", checkout.string(), "rev-parse", "HEAD"});
        auto version = run({tool, "version"});
        contains(version, "repo=" + remote.string());
        contains(version, "ref=master");
        contains(version, "sha=" + first_sha);
        std::ofstream(seed / "README.md", std::ios::app) << "two\n";
        git({"add", "README.md"});
        git({"commit", "-m", "update"});
        git({"push", "origin", "master"});
        run({tool, "update"});
        const auto second_sha = run({"git", "-C", checkout.string(), "rev-parse", "HEAD"});
        expect(first_sha != second_sha, "update fetches new commit");
        contains(run({tool, "version"}), "sha=" + second_sha);
        env("EZLOCALAI_LLAMA_CPP_CMAKE_ARGS", "-DTEST_VALUE=\"value with spaces\"");
        env("EZLOCALAI_LLAMA_CPP_JOBS", "2");
        run({tool, "build"});
        const auto binary = run({tool, "print-bin"});
        expect(binary == (build / "bin/llama-server").string(), "print-bin resolves built executable");
        run({binary});
        auto status = run({tool, "status"});
        for (const auto &expected : {"repo=" + remote.string(), std::string("ref=master"), std::string("checkout=present"),
                "sha=" + second_sha, std::string("build=present"), std::string("version_file=present"), "server_bin=" + binary}) contains(status, expected);
        unsetenv("EZLOCALAI_LLAMA_CPP_REF");
        contains(run({tool, "status"}), std::string("ref=") + ezlocalai::llama_cpp_reference_revision);
        env("EZLOCALAI_LLAMA_CPP_REF", "master");

        const auto models = root / "models";
        fs::create_directory(models);
        for (const char *name : {"my-model-q4.gguf", "mmproj-my-model.gguf", "Qwen3-Embedding-0.6B-Q8_0.gguf"}) std::ofstream(models / name) << "GGUF\n";
        fs::create_directory(models / "My-Model");
        std::ofstream(models / "My-Model/model-Q4_K_M.gguf") << "GGUF\n";
        env("DEFAULT_MODEL", "org/My-Model-GGUF,org/My-Model-GGUF");
        env("QUANT_TYPE", "Q4_K_M,Q5_K_M");
        env("MODELS_PATH", models.string());
        status = run({tool, "model-status"});
        contains(status, "model_count=2");
        contains(status, "model[0].id=org/My-Model-GGUF");
        contains(status, "model[0].mode=local");
        contains(status, "model[0].path=" + (models / "My-Model/model-Q4_K_M.gguf").string());
        contains(status, "model[0].hf=org/My-Model-GGUF:Q4_K_M");
        contains(status, "model[1].id=org/My-Model-GGUF");
        contains(status, "model[1].mode=hf");
        contains(status, "model[1].path=\n");
        contains(status, "model[1].hf=org/My-Model-GGUF:Q5_K_M");
        contains(status, "embedding.mode=local");
        contains(status, "embedding.path=" + (models / "Qwen3-Embedding-0.6B-Q8_0.gguf").string());
        contains(status, "embedding.alias=Qwen3-Embedding-0.6B");
        contains(status, "embedding.hf=Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0");
        std::ofstream(models / "My-Model/model-Q5_K_M.gguf") << "GGUF\n";
        status = run({tool, "model-status"});
        contains(status, "model[1].path=" + (models / "My-Model/model-Q5_K_M.gguf").string());
        env("DEFAULT_MODEL", "org/Absent-GGUF");
        contains(run({tool, "model-status"}), "model[0].mode=hf");
        expect(!fs::exists(models / "Absent"), "model-status preview does not create download directories");
        env("DEFAULT_MODEL", "org/My-Model-GGUF,org/My-Model-GGUF");
        fs::remove(models / "Qwen3-Embedding-0.6B-Q8_0.gguf");
        contains(run({tool, "model-status"}), "embedding.mode=hf");
        env("DEFAULT_MODEL", "org/Missing-GGUF");
        env("LLAMA_CPP_MODEL_PATHS", (models / "missing.gguf").string());
        status = run({tool, "model-status"});
        contains(status, "model[0].mode=missing-local");
        contains(status, "model[0].path=" + (models / "missing.gguf").string());
        env("LLAMA_CPP_EMBEDDING_SERVER_URL", "http://localhost:9090");
        contains(run({tool, "model-status"}), "embedding.mode=remote");
        env("DEFAULT_MODEL", "none");
        contains(run({tool, "model-status"}), "model_count=0");
        env("EZLOCALAI_LLAMA_CPP_REF", first_sha);
        run({tool, "update"});
        contains(run({tool, "version"}), "sha=" + first_sha);
        contains(run({tool, "status"}), "branch=HEAD");
        run({tool, "clean"});
        expect(!fs::exists(build), "clean removes build");
        expect(fs::exists(checkout / ".git"), "clean preserves checkout");
        expect(ezlocalai::run_process({tool, "print-bin"}, true).status == 1, "missing binary returns 1");
        expect(ezlocalai::run_process({tool, "definitely-not-a-command"}, true).status == 2, "invalid command returns 2");
        env("EZLOCALAI_LLAMA_CPP_BUILD_DIR", checkout.string());
        expect(ezlocalai::run_process({tool, "clean"}, true).status != 0, "clean rejects checkout directory");
        fs::current_path(initial_cwd);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
