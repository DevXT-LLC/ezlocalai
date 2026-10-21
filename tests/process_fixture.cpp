#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

int main(int argc, char **argv) {
    const auto executable = std::filesystem::read_symlink("/proc/self/exe");
    std::ifstream config(executable.string() + ".fixture");
    std::string arguments_path, output, whisper_json;
    bool append_launch = false, fail_first_launch = false;
    int delay_ms = 0, exit_status = 0;
    if (!(config >> std::quoted(arguments_path) >> append_launch >> delay_ms >> exit_status >> fail_first_launch
                 >> std::quoted(output) >> std::quoted(whisper_json))) return 2;
    if (!arguments_path.empty()) {
        std::ofstream args(arguments_path, append_launch ? std::ios::app : std::ios::trunc);
        if (append_launch) args << "launch\n";
        for (int i = 1; i < argc; ++i) args << argv[i] << '\n';
        args.close();
        if (fail_first_launch) {
            std::ifstream in(arguments_path);
            int launches = 0;
            for (std::string line; std::getline(in, line);) if (line == "launch") ++launches;
            if (launches == 1) {
                std::cerr << "CUDA out of memory while loading model\n";
                return 1;
            }
        }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    if (!whisper_json.empty()) {
        std::string prefix;
        for (int i = 1; i + 1 < argc; ++i) if (std::string(argv[i]) == "-of") prefix = argv[++i];
        if (prefix.empty()) return 2;
        std::ofstream out(prefix + ".json");
        out << whisper_json;
        if (!out) return 2;
    }
    std::cout << output;
    return exit_status;
}
