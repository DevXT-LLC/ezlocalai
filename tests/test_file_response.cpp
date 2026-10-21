#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/native_server.hpp"
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
void check(bool ok, const std::string &message) {
    if (!ok)
        throw std::runtime_error(message);
}
std::string lower(std::string s) {
    for (auto &c : s)
        c = std::tolower(static_cast<unsigned char>(c));
    return s;
}
void replace(std::string &s, const std::string &from, const std::string &to) {
    for (size_t at = 0; (at = s.find(from, at)) != std::string::npos; at += to.size())
        s.replace(at, from.size(), to);
}
void write(const fs::path &path, const std::string &data, int64_t ns) {
    std::ofstream(path, std::ios::binary) << data;
    timespec times[2]{{ns / 1000000000, ns % 1000000000}, {ns / 1000000000, ns % 1000000000}};
    check(utimensat(AT_FDCWD, path.c_str(), times, 0) == 0, "set fixture timestamp");
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--check-inherited-file") {
            for (const auto &fd : fs::directory_iterator("/proc/self/fd")) {
                std::error_code error;
                const auto target = fs::read_symlink(fd.path(), error).string();
                if (!error && target.find(argv[2]) != std::string::npos)
                    return 1;
            }
            return 0;
        }
        check(argc == 2, "source response fixture required");
        std::ifstream input(argv[1]);
        const auto fixture = Json::parse(input);
        char pattern[] = "/tmp/ezlocalai-downloads-XXXXXX";
        check(mkdtemp(pattern), "temporary directory");
        const auto previous = fs::current_path();
        fs::current_path(pattern);
        struct Cleanup {
            fs::path previous, path;
            ~Cleanup() {
                fs::current_path(previous);
                fs::remove_all(path);
            }
        } cleanup{previous, pattern};
        for (const auto &item : ezlocalai::environment_defaults)
            unsetenv(std::string(item.first).c_str());
        for (const auto *key :
             {"TTS_ENABLED", "EMBEDDING_ENABLED", "IMAGE_ENABLED", "VIDEO_ENABLED", "MUSIC_ENABLED",
              "EZLOCALAI_PRELOAD_MODEL_ON_START", "EZLOCALAI_MODEL_STATUS_ON_START"})
            setenv(key, "false", 1);
        setenv("STT_ENABLED", "true", 1);
        setenv("EZLOCALAI_API_KEY", "", 1);
        setenv("ROUTER_URL", "", 1);
        setenv("DEFAULT_MODEL", "", 1);
        setenv("CUDA_VISIBLE_DEVICES", "", 1);
        fs::create_directories("outputs");
        fs::create_directories("wakeword_models/héllo");
        ezl_env_config config{};
        ezl_load_config(&config);
        ezlocalai::NativeServer server(config, std::make_unique<ezlocalai::EchoInferenceBackend>(""));
        for (const auto &row : fixture.at("cases")) {
            const auto name = row.at("name").get<std::string>();
            const auto bytes = row.at("file") == "empty.bin" ? "" : fixture.at("data").get<std::string>();
            const auto path = row.at("static") == true
                                  ? fs::path("outputs") / row.at("file").get<std::string>()
                                  : fs::path("wakeword_models/héllo/model.pt");
            write(path, bytes, fixture.at("mtime_ns"));
            const auto route = row.at("static") == true ? "/outputs/" + row.at("file").get<std::string>()
                                                        : "/v1/wakeword/models/h%C3%A9llo";
            const auto headers = row.at("headers").get<std::unordered_map<std::string, std::string>>();
            for (bool deferred : {false, true}) {
                auto response = server.handle_request("GET", route, "", "", "", "", "", headers, deferred);
                check(response.status == row.at("status"),
                      name + ": status " + std::to_string(response.status));
                Json actual = Json::object();
                for (const auto &h : response.headers)
                    actual[lower(h.first)] = h.second;
                if (!response.content_type.empty())
                    actual["content-type"] = response.content_type;
                if (response.status != 304)
                    actual["content-length"] =
                        std::to_string(response.content_length.value_or(response.body.size()));
                std::string body = response.body;
                if (response.stream_writer) {
                    check(deferred && body.empty(), "streaming file body is not buffered");
                    int sockets[2];
                    check(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "socket pair");
                    response.stream_writer(sockets[0]);
                    shutdown(sockets[0], SHUT_WR);
                    char buffer[2048];
                    for (ssize_t n; (n = read(sockets[1], buffer, sizeof(buffer))) > 0;)
                        body.append(buffer, n);
                    close(sockets[0]);
                    close(sockets[1]);
                }
                const auto type = actual.value("content-type", "");
                if (type.rfind("multipart/byteranges; boundary=", 0) == 0) {
                    const auto boundary = type.substr(type.find("boundary=") + 9);
                    check(boundary.size() == 26, "multipart boundary entropy length");
                    actual["content-type"] = "multipart/byteranges; boundary=" + std::string(26, 'B');
                    replace(body, boundary, std::string(26, 'B'));
                }
                check(actual == row.at("response_headers"),
                      name + ": headers " + actual.dump() + " expected " + row.at("response_headers").dump());
                check(body == row.at("body"), name + ": response body");
            }
        }
        // A small range in a sparse 5 GiB file must not allocate/read the whole file.
        {
            std::ofstream f("outputs/large.bin", std::ios::binary);
            f.seekp(int64_t(5) * 1024 * 1024 * 1024);
            f << "tail";
        }
        auto range = server.handle_request("GET", "/outputs/large.bin", "", "", "", "", "",
                                           {{"range", "bytes=-4"}}, true);
        check(range.status == 206 && range.body.empty() && range.content_length == 4 &&
                  bool(range.stream_writer),
              "large file stays deferred");
        fs::remove("outputs/large.bin");
        // Model engines are fork/exec children. An in-progress download must
        // not leave its open (possibly deleted) inode in a long-lived engine.
        const auto inherited_path = (fs::path(pattern) / "outputs/large.bin").string();
        const auto child = fork();
        check(child >= 0, "fork descriptor inheritance probe");
        if (child == 0) {
            execl("/proc/self/exe", "test_file_response", "--check-inherited-file",
                  inherited_path.c_str(), nullptr);
            _exit(127);
        }
        int status = 0;
        pid_t waited;
        do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
        check(waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
              "file descriptor is not inherited across model process exec");
        int sockets[2];
        check(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "large file socket pair");
        range.stream_writer(sockets[0]);
        shutdown(sockets[0], SHUT_WR);
        char tail[8]{};
        const auto tail_size = read(sockets[1], tail, sizeof(tail));
        close(sockets[0]);
        close(sockets[1]);
        check(tail_size == 4 && std::string(tail, 4) == "tail",
              "deferred range retains opened inode and reads beyond 32-bit offsets");
        check(server.handle_request("HEAD", "/v1/wakeword/models/h%C3%A9llo", "", "").status == 405,
              "FastAPI GET route does not add HEAD");
        std::cout
            << fixture.at("cases").size()
            << " source file responses matched in buffered and deferred modes; large-file range passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
