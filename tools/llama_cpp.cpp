#include "ezlocalai/process.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/engine_revision.hpp"
#include "ezlocalai/llm_catalog.hpp"
#include "ezlocalai/model_download.hpp"
#include "ezlocalai/ezlocalai_c.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

namespace fs = std::filesystem;

namespace {
std::string env(const char *name, const std::string &fallback = {}) {
    const char *value = std::getenv(name);
    return value && *value ? value : fallback;
}

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n\v\f");
    return first == std::string::npos ? "" : value.substr(first, value.find_last_not_of(" \t\r\n\v\f") - first + 1);
}

std::string lower(std::string value) {
    for (char &ch : value) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return value;
}

std::vector<std::string> csv(const std::string &value) {
    std::vector<std::string> out;
    std::istringstream input(value);
    for (std::string part; std::getline(input, part, ',');) {
        part = trim(part);
        if (!part.empty()) out.push_back(part);
    }
    return out;
}

std::string find_model(const std::string &model, const std::optional<std::string> &quant) {
    // Preview exactly the runtime's local selection without creating folders,
    // querying the Hub, or downloading optional projectors.
    ezlocalai::ModelDownloadHooks io;
    io.mkdir = [](const auto &) {};
    io.list_remote = [](const auto &) { return std::vector<std::string>{}; };
    io.download = [](const auto &, const auto &, const auto &) { throw std::logic_error("model-status cannot download files"); };
    auto root = ezlocalai::strip_config_value(ezl_getenv_default("MODELS_PATH"));
    if (root.empty()) root = "./models";
    try { return ezlocalai::download_gguf_model(model, root, quant, true, io).model; }
    catch (const std::exception &) { return {}; }
}

std::string explicit_model_path(size_t index) {
    const auto paths = csv(env("LLAMA_CPP_MODEL_PATHS"));
    return index < paths.size() ? paths[index] : trim(env("LLAMA_CPP_MODEL_PATH"));
}

std::string resolve_path(const std::string &model, const std::string &explicit_path, const std::optional<std::string> &quant) {
    if (!explicit_path.empty()) return explicit_path;
    if (!model.empty() && fs::exists(model)) return model;
    return find_model(model, quant);
}

std::string path_mode(const std::string &path, bool explicit_path, const std::string &model) {
    if (!path.empty()) return fs::exists(path) || !explicit_path ? "local" : "missing-local";
    return model.empty() || fs::path(lower(model)).extension() == ".gguf" ? "missing-local" : "hf";
}

std::string hf_model(std::string source, const std::optional<std::string> &quantization) {
    if (source.find('/') == std::string::npos) source = "unsloth/" + source + "-GGUF";
    auto quant = quantization.value_or(ezl_getenv_default("QUANT_TYPE"));
    const auto comma = quant.find(',');
    if (comma != std::string::npos) quant = ezlocalai::strip_config_value(quant.substr(0, comma));
    return quant.empty() ? source : source + ":" + quant;
}

void model_status() {
    const std::string models = ezl_getenv_default("DEFAULT_MODEL");
    std::cout << "default_model=" << models << '\n';
    ezlocalai::LlmCatalogSettings settings;
    settings.models = models; settings.main_gpus = ezl_getenv_default("MAIN_GPU");
    settings.contexts = ezl_getenv_default("LLM_MAX_TOKENS"); settings.parallels = ezl_getenv_default("N_PARALLEL");
    settings.quants = ezl_getenv_default("QUANT_TYPE"); settings.speculative_type = ezl_getenv_default("LLM_SPECULATIVE_TYPE");
    const auto catalog = ezlocalai::parse_llm_catalog(settings);
    std::cout << "model_count=" << catalog.available.size() << '\n';
    for (size_t i = 0; i < catalog.available.size(); ++i) {
        const auto &id = catalog.available[i];
        const auto source = catalog.source(id); const auto &config = catalog.configs.at(id);
        const auto explicit_path = explicit_model_path(config.original_index);
        const auto path = resolve_path(source, explicit_path, config.quant);
        const auto prefix = "model[" + std::to_string(i) + "].";
        std::cout << prefix << "id=" << source << '\n'
                  << prefix << "mode=" << path_mode(path, !explicit_path.empty(), source) << '\n'
                  << prefix << "path=" << path << '\n'
                  << prefix << "hf=" << hf_model(source, config.quant) << '\n';
    }
    const auto embedding = ezlocalai::embedding_model_config();
    const auto explicit_path = trim(env("LLAMA_CPP_EMBEDDING_MODEL_PATH"));
    const auto mode = !trim(env("LLAMA_CPP_EMBEDDING_SERVER_URL")).empty()
        ? "remote" : path_mode(embedding.path, !explicit_path.empty(), embedding.model);
    std::cout << "embedding_model=" << embedding.model << '\n'
              << "embedding.mode=" << mode << '\n'
              << "embedding.path=" << embedding.path << '\n'
              << "embedding.alias=" << embedding.alias << '\n'
              << "embedding.hf=" << embedding.hf_argument << '\n';
}

// Accept quoted arguments without invoking a shell, expansion, or globbing.
std::vector<std::string> split_arguments(const std::string &value) {
    std::vector<std::string> out;
    std::string part;
    char quote = 0;
    bool escaped = false, started = false;
    for (const char ch : value) {
        if (escaped) { part += ch; escaped = false; started = true; }
        else if (ch == '\\' && quote != '\'') { escaped = true; started = true; }
        else if (quote) { if (ch == quote) quote = 0; else part += ch; }
        else if (ch == '\'' || ch == '"') { quote = ch; started = true; }
        else if (std::isspace(static_cast<unsigned char>(ch))) {
            if (started) { out.push_back(part); part.clear(); started = false; }
        } else { part += ch; started = true; }
    }
    if (quote || escaped) throw std::runtime_error("unclosed quote or escape in EZLOCALAI_LLAMA_CPP_CMAKE_ARGS");
    if (started) out.push_back(part);
    return out;
}

struct Tool {
    fs::path native = env("EZLOCALAI_NATIVE_DIR", EZLOCALAI_NATIVE_SOURCE_DIR);
    std::string repo = env("EZLOCALAI_LLAMA_CPP_REPO", "https://github.com/ggml-org/llama.cpp.git");
    std::string ref = env("EZLOCALAI_LLAMA_CPP_REF", ezlocalai::llama_cpp_reference_revision);
    fs::path checkout = env("EZLOCALAI_LLAMA_CPP_DIR", (native / "third_party/llama.cpp").string());
    fs::path build = env("EZLOCALAI_LLAMA_CPP_BUILD_DIR", (native / "build/llama.cpp").string());

    static std::string checked(const std::vector<std::string> &args, bool capture = false) {
        auto result = ezlocalai::run_process(args, capture);
        if (result.status != 0) throw std::runtime_error(args.front() + " failed with status " + std::to_string(result.status));
        while (!result.output.empty() && result.output.back() == '\n') result.output.pop_back();
        return result.output;
    }

    std::string git(const std::vector<std::string> &args, bool capture = false) const {
        std::vector<std::string> command{"git", "-C", checkout.string()};
        command.insert(command.end(), args.begin(), args.end());
        return checked(command, capture);
    }

    void write_version() const {
        const auto sha = git({"rev-parse", "HEAD"}, true);
        const auto date = git({"show", "-s", "--format=%cI", "HEAD"}, true);
        fs::create_directories(build);
        std::ofstream out(build / "ezlocalai-llama-cpp-version.txt");
        out << "repo=" << repo << "\nref=" << ref << "\nsha=" << sha << "\ndate=" << date << '\n';
        if (!out) throw std::runtime_error("could not write llama.cpp version file");
    }

    void ensure() const {
        if (!checkout.parent_path().empty()) fs::create_directories(checkout.parent_path());
        if (!fs::exists(checkout / ".git")) checked({"git", "clone", repo, checkout.string()});
        else git({"remote", "set-url", "origin", repo});
        git({"fetch", "--tags", "--prune", "origin"});
        const auto remote = ezlocalai::run_process({"git", "-C", checkout.string(), "rev-parse", "--verify", "--quiet", "refs/remotes/origin/" + ref}, true);
        if (remote.status == 0) git({"checkout", "-B", ref, "origin/" + ref});
        else git({"checkout", ref});
        const auto branch = git({"rev-parse", "--abbrev-ref", "HEAD"}, true);
        if (branch != "HEAD") git({"pull", "--ff-only", "origin", branch});
        write_version();
    }

    void build_server() const {
        ensure();
        std::vector<std::string> configure{"cmake", "-S", checkout.string(), "-B", build.string(), "-DCMAKE_BUILD_TYPE=" + env("EZLOCALAI_LLAMA_CPP_BUILD_TYPE", "Release")};
        const auto extra = split_arguments(env("EZLOCALAI_LLAMA_CPP_CMAKE_ARGS"));
        configure.insert(configure.end(), extra.begin(), extra.end());
        checked(configure);
        std::vector<std::string> command{"cmake", "--build", build.string(), "--target", "llama-server"};
        const auto jobs = env("EZLOCALAI_LLAMA_CPP_JOBS");
        if (!jobs.empty()) { command.push_back("--parallel"); command.push_back(jobs); }
        checked(command);
    }

    std::string binary() const {
        for (const char *name : {"bin/llama-server", "tools/server/llama-server", "llama-server"}) {
            const auto path = build / name;
            if (fs::is_regular_file(path) && access(path.c_str(), X_OK) == 0) return path.string();
        }
        return {};
    }

    void status() const {
        std::cout << "repo=" << repo << "\nref=" << ref << "\ncheckout_dir=" << checkout.string() << "\nbuild_dir=" << build.string() << '\n';
        if (fs::exists(checkout / ".git")) {
            const auto sha = git({"rev-parse", "HEAD"}, true);
            const auto branch = git({"rev-parse", "--abbrev-ref", "HEAD"}, true);
            std::cout << "checkout=present\nsha=" << sha << "\nbranch=" << branch << '\n';
        } else std::cout << "checkout=missing\n";
        std::cout << "build=" << (fs::is_directory(build) ? "present" : "missing")
                  << "\nversion_file=" << (fs::is_regular_file(build / "ezlocalai-llama-cpp-version.txt") ? "present" : "missing")
                  << "\nserver_bin=" << binary() << '\n';
    }

    void clean() const {
        const auto target = fs::weakly_canonical(build);
        const auto source = fs::weakly_canonical(checkout);
        const auto relative = source.lexically_relative(target);
        if (target == target.root_path() || target == fs::current_path() ||
            relative.empty() || *relative.begin() != "..") {
            throw std::runtime_error("refusing to remove a root, working directory, or llama.cpp source directory");
        }
        fs::remove_all(build);
    }
};

void usage(std::ostream &out) {
    out << "Usage: llama_cpp_tool <command>\n\n"
        << "Commands: ensure, update, build, print-bin, version, status, model-status, clean, help\n\n"
        << "Environment: EZLOCALAI_LLAMA_CPP_REPO, EZLOCALAI_LLAMA_CPP_REF,\n"
        << "  EZLOCALAI_LLAMA_CPP_DIR, EZLOCALAI_LLAMA_CPP_BUILD_DIR,\n"
        << "  EZLOCALAI_LLAMA_CPP_BUILD_TYPE, EZLOCALAI_LLAMA_CPP_CMAKE_ARGS,\n"
        << "  EZLOCALAI_LLAMA_CPP_JOBS, EZLOCALAI_NATIVE_DIR, DEFAULT_MODEL, QUANT_TYPE,\n"
        << "  MODELS_PATH, LLAMA_CACHE, LLAMA_CPP_MODEL_PATH, LLAMA_CPP_MODEL_PATHS,\n"
        << "  LLAMA_CPP_EMBEDDING_MODEL, LLAMA_CPP_EMBEDDING_MODEL_PATH,\n"
        << "  LLAMA_CPP_EMBEDDING_SERVER_URL, LLAMA_CPP_EMBEDDING_ALIAS\n";
}
} // namespace

int main(int argc, char **argv) {
    try {
        const std::string command = argc > 1 ? argv[1] : "help";
        const Tool tool;
        if (command == "ensure" || command == "update") tool.ensure();
        else if (command == "build") tool.build_server();
        else if (command == "print-bin") {
            const auto path = tool.binary();
            if (path.empty()) return 1;
            std::cout << path << '\n';
        } else if (command == "version") {
            std::ifstream in(tool.build / "ezlocalai-llama-cpp-version.txt");
            if (!in) throw std::runtime_error("llama.cpp version file is missing");
            std::cout << in.rdbuf();
        } else if (command == "status") tool.status();
        else if (command == "model-status") model_status();
        else if (command == "clean") tool.clean();
        else if (command == "help" || command == "-h" || command == "--help") usage(std::cout);
        else { usage(std::cerr); return 2; }
    } catch (const std::exception &error) {
        std::cerr << "llama_cpp_tool: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
