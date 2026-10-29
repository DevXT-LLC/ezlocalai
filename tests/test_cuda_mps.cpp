#include "ezlocalai/cuda_mps.hpp"

#include <cstdlib>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <chrono>
#include <unistd.h>

namespace fs = std::filesystem;
namespace {
void check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
std::string read(const fs::path &path) {
    std::ifstream file(path); return {std::istreambuf_iterator<char>(file), {}};
}
int control(int argc, char **argv) {
    std::ofstream log(std::getenv("TEST_MPS_COMMAND_LOG"), std::ios::app);
    const fs::path pipes = std::getenv("CUDA_MPS_PIPE_DIRECTORY");
    if (argc == 2 && std::string(argv[1]) == "-f") {
        log << "daemon\n"; log.flush();
        if (std::getenv("TEST_MPS_FAIL_START")) return 7;
        { std::ofstream ready(pipes / "fixture-ready"); ready << getpid(); }
        while (!fs::exists(pipes / "fixture-quit")) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return 0;
    }
    if (!fs::exists(pipes / "fixture-ready")) return 2;
    std::string line; std::getline(std::cin, line); log << line << '\n';
    if (line.find("start_server -uid ") == 0) { std::ofstream marker(pipes / "fixture-server-started"); marker << "ready"; }
    if (line == "get_server_list") {
        if (std::getenv("TEST_MPS_FAIL_QUERY") && fs::exists(pipes / "fixture-server-started")) return 9;
        std::cout << "1234\n";
    }
    if (line == "quit") { std::ofstream quit(pipes / "fixture-quit"); quit << "done"; }
    return 0;
}
}

int main(int argc, char **argv) {
    if (fs::path(argv[0]).filename() == "nvidia-cuda-mps-control") return control(argc, argv);
    if (argc == 2 && std::string(argv[1]) == "--worker-fixture") {
        ezlocalai::CudaMpsService disabled;
        check(!disabled.active() && std::string(std::getenv("EZLOCALAI_CUDA_MPS")) == "false", "worker recursively started MPS");
        check(fs::exists(fs::path(std::getenv("CUDA_MPS_PIPE_DIRECTORY")) / "fixture-ready"), "worker lost private pipe directory");
        return 23;
    }
    if (argc == 2 && std::string(argv[1]) == "--waiting-worker-fixture") {
        for (;;) pause();
    }
    char pattern[] = "/tmp/test-cuda-mps-XXXXXX";
    const auto created = mkdtemp(pattern);
    if (!created) return 1;
    const fs::path root = created;
    try {
        check(!ezlocalai::llm_mps_client_priority("", 0), "unset per-model priority did not inherit");
        check(ezlocalai::llm_mps_client_priority("0, 1", 0) == 0 &&
              ezlocalai::llm_mps_client_priority("0, 1", 7) == 1, "priority CSV indexing/fallback changed");
        check(!ezlocalai::llm_mps_client_priority("0,,1", 1), "empty priority did not inherit");
        bool invalid_priority = false;
        try { (void)ezlocalai::llm_mps_client_priority("0,2", 0); }
        catch (const std::invalid_argument &) { invalid_priority = true; }
        check(invalid_priority, "invalid priority in another model entry accepted");
        const auto self = fs::canonical(argv[0]).string();
        fs::create_directory(root / "bin");
        fs::create_symlink(self, root / "bin/nvidia-cuda-mps-control");
        const std::string path = (root / "bin").string() + ":/usr/bin:/bin";
        setenv("PATH", path.c_str(), 1);
        setenv("TEST_MPS_COMMAND_LOG", (root / "commands").c_str(), 1);
        setenv("EZLOCALAI_CUDA_MPS", "false", 1);
        setenv("CUDA_MPS_PIPE_DIRECTORY", "external-pipes", 1);
        setenv("CUDA_MPS_LOG_DIRECTORY", "external-logs", 1);
        { ezlocalai::CudaMpsService service; }
        check(std::string(std::getenv("CUDA_MPS_PIPE_DIRECTORY")) == "external-pipes", "disabled MPS changed external setup");
        check(!fs::exists(root / "commands"), "disabled MPS launched a process");
        setenv("EZLOCALAI_CUDA_MPS", "typo", 1);
        bool rejected = false;
        try { ezlocalai::CudaMpsService service; } catch (...) { rejected = true; }
        check(rejected, "invalid option accepted");
        setenv("EZLOCALAI_CUDA_MPS", "true", 1);
        rejected = false;
        try { ezlocalai::CudaMpsService service; } catch (...) { rejected = true; }
        check(rejected && !fs::exists(root / "commands"), "managed MPS commandeered external setup");
        unsetenv("CUDA_MPS_PIPE_DIRECTORY"); unsetenv("CUDA_MPS_LOG_DIRECTORY");
        setenv("CUDA_VISIBLE_DEVICES", "2,0", 1);
        rejected = false;
        try { ezlocalai::CudaMpsService service; } catch (...) { rejected = true; }
        check(rejected && !fs::exists(root / "commands"), "unsafe MPS device remapping accepted");
        setenv("CUDA_VISIBLE_DEVICES", "0", 1);
        std::string directory;
        {
            ezlocalai::CudaMpsService service;
            directory = std::getenv("CUDA_MPS_PIPE_DIRECTORY");
            check(std::string(std::getenv("CUDA_MPS_LOG_DIRECTORY")) == directory + "/log", "private logs missing");
            struct stat info{};
            check(stat(directory.c_str(), &info) == 0 && (info.st_mode & 0777) == 0700, "MPS pipes are not private");
            check(service.supervise({self, "--worker-fixture"}, [] { return false; }) == 23, "worker exit status not propagated");
            check(service.supervise({self, "--waiting-worker-fixture"}, [] { return true; }) == 128 + SIGTERM, "shutdown signal not forwarded");
            check(std::string(std::getenv("EZLOCALAI_CUDA_MPS")) == "true", "supervisor configuration changed");
        }
        check(!std::getenv("CUDA_MPS_PIPE_DIRECTORY") && !std::getenv("CUDA_MPS_LOG_DIRECTORY"), "MPS environment leaked");
        check(read(root / "commands") == "daemon\nget_server_list\nstart_server -uid " + std::to_string(getuid()) + "\nget_server_list\nquit\n", "startup/readiness/shutdown order changed");
        const auto pid = std::stoi(read(fs::path(directory) / "fixture-ready"));
        check(kill(pid, 0) != 0 && errno == ESRCH, "foreground MPS child was not reaped");
        fs::remove_all(directory);
        for (const char *failure : {"TEST_MPS_FAIL_START", "TEST_MPS_FAIL_QUERY"}) {
            setenv(failure, "1", 1);
            rejected = false;
            try { ezlocalai::CudaMpsService service; }
            catch (const std::exception &error) {
                rejected = true;
                const std::string message = error.what();
                const auto pos = message.find("; logs: ");
                check(pos != std::string::npos, "failure omitted diagnostic location");
                const auto logs = message.substr(pos + 8);
                check(logs.find("/tmp/ezlocalai-mps-") == 0, "unexpected diagnostic directory");
                check(fs::exists(logs), "failure diagnostics were discarded");
                fs::remove_all(logs);
            }
            unsetenv(failure);
            check(rejected, "MPS failure silently fell back to ordinary CUDA");
            check(!std::getenv("CUDA_MPS_PIPE_DIRECTORY"), "failed startup leaked MPS environment");
        }
        fs::remove_all(root);
        std::cout << "CUDA MPS: disabled/external configuration, private startup, readiness, shutdown and failure cleanup passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\nArtifacts: " << root << '\n'; return 1;
    }
}
