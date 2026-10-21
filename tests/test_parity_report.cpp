#include "ezlocalai/process.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

int main(int argc, char **argv) {
    fs::path root;
    try {
        if (argc != 2) throw std::runtime_error("parity_report binary required");
        std::string pattern = (fs::temp_directory_path() / "ezlocalai-parity-XXXXXX").string();
        char *created = mkdtemp(pattern.data());
        if (!created) throw std::runtime_error("mkdtemp failed");
        root = created;
        const auto source = root / "reference";
        fs::create_directory(source);
        const auto manifest = root / "manifest.tsv";
        auto write_manifest = [&](const std::string &status) {
            std::ofstream out(manifest);
            out << "mode\tmethod\tpath\tpython_source\tnative_status\tnotes\n"
                << "api\tGET\t/health\tapp.py:1\t" << status << "\tfixture\n"
                << "router\tGET\t/health\trouter_app.py:1\timplemented\tfixture\n";
        };
        auto check = [&](int expected, std::vector<std::string> args) {
            args.insert(args.begin(), argv[1]);
            args.push_back(manifest.string());
            const auto result = ezlocalai::run_process(args, true, true);
            if (result.status != expected) throw std::runtime_error("unexpected parity result: " + result.output);
        };
        write_manifest("implemented");
        check(1, {"--enforce", "--source-root", source.string()});
        // These are scanner input data, never executed as Python.
        std::ofstream(source / "app.py") << "@app.get(\n    \"/health\", tags=[\"System\"]\n)\n";
        check(0, {"--enforce", "--worker-only", "--source-root", source.string()});
        // Worker scope must still enforce worker-side protocol/module gaps.
        std::ofstream(manifest, std::ios::app) << "module\tC++\tworker-router-protocol\tTunnel.py\tpartial\tfixture\n";
        check(1, {"--enforce", "--worker-only", "--source-root", source.string()});
        write_manifest("implemented");
        std::ofstream(source / "router_app.py") << "@app.get(\"/health\")\n";
        check(0, {"--enforce", "--source-root", source.string()});
        write_manifest("partial");
        check(0, {"--source-root", source.string()});
        check(1, {"--enforce", "--source-root", source.string()});
        write_manifest("unsupported-status");
        check(2, {"--enforce", "--source-root", source.string()});
        write_manifest("implemented");
        std::ofstream(source / "app.py", std::ios::app) << "@app.post(\"/new-route\")\n";
        check(2, {"--source-root", source.string()});
        std::ofstream(manifest, std::ios::app) << "api\tPOST\t/new-route\tapp.py:4\tmissing\tfixture\n";
        check(0, {"--source-root", source.string()});
        check(1, {"--enforce", "--source-root", source.string()});
        fs::remove_all(root);
        return 0;
    } catch (const std::exception &error) {
        if (!root.empty()) { std::error_code ignored; fs::remove_all(root, ignored); }
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
