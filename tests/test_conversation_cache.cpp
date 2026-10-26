#include "ezlocalai/llm_request.hpp"
#include "ezlocalai/process.hpp"
#include "server-task.h"
#include <arpa/inet.h>
#include <openssl/sha.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/socket.h>

static void check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
static server_prompt prompt(int branch, int count = 128) {
    llama_tokens tokens;
    for (int i = 0; i < count; ++i) tokens.push_back(i < 64 ? i : branch * 1000 + i);
    return {server_tokens(tokens, false), {}};
}
static void policy() {
    auto a = prompt(1), b = prompt(2), c = prompt(3);
    server_prompt_cache cache(2, 4096);
    auto *saved_a = cache.alloc(a, 768 * 1024, 0);
    check(saved_a != nullptr, "save conversation A");
    check(cache.alloc(b, 768 * 1024, 0) != nullptr, "save conversation B");
    check(cache.alloc(c, 768 * 1024, 0, &a.tokens) != nullptr, "save C without evicting requested A");
    auto best = cache.find_best(c, a.tokens);
    check(best != cache.states.end() && &*best == saved_a, "requested cache survives eviction");
    check(cache.states.size() == 2 && cache.size() <= 2 * 1024 * 1024, "cache remains bounded");
    auto short_resident = prompt(0, 8);
    check(cache.find_best(short_resident, a.tokens) == best, "a short resident prefix cannot hide a longer cache hit");
    check(cache.find_best(a, a.tokens) == cache.states.end(), "resident continuation avoids a redundant restore");
    server_prompt_cache tight(1, 4096);
    check(tight.alloc(a, 400 * 1024, 200 * 1024) != nullptr, "cache includes target and draft state");
    check(tight.alloc(b, 400 * 1024, 200 * 1024, &a.tokens) == nullptr, "skip outgoing snapshot when both states exceed budget");
    check(tight.find_best(b, a.tokens) != tight.states.end(), "tight budget preserves requested state");
    check(tight.size() == 600 * 1024, "no transient cache-budget overrun");
    check(tight.alloc(c, 2 * 1024 * 1024, 0, &a.tokens) == nullptr, "oversize snapshot is rejected");
    check(tight.states.size() == 1, "oversize snapshot does not disturb requested cache");

    const std::string base = R"({"messages":[{"role":"user","content":"unchanged"}],"temperature":0,)";
    const auto keyed = ezlocalai::llm_engine_request(base + R"("prompt_cache_key":"workconductor:agent:conversation"})", true, "16", "");
    check(keyed.find(R"("prompt_cache_key":"workconductor:agent:conversation")") != std::string::npos,
          "conversation key reaches the engine");
    check(keyed.find(R"("content":"unchanged")") != std::string::npos, "cache hint does not rewrite messages");
    for (const auto value : {"null", "false", "7", "[]", "{}", "\"\""}) {
        const auto unkeyed = ezlocalai::llm_engine_request(base + "\"prompt_cache_key\":" + value + '}', true, "16", "");
        check(unkeyed.find("prompt_cache_key") == std::string::npos, "invalid/empty cache hints retain legacy behavior");
    }
}
static int port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0); check(fd >= 0, "listener");
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    check(bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0, "bind");
    socklen_t size = sizeof(address); getsockname(fd, reinterpret_cast<sockaddr *>(&address), &size); close(fd);
    return ntohs(address.sin_port);
}
static void engine(const std::string &binary, const std::string &model) {
    std::ifstream input(model, std::ios::binary);
    const std::string bytes{std::istreambuf_iterator<char>(input), {}};
    unsigned char digest[32]; SHA256(reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size(), digest);
    std::ostringstream hash; for (auto byte : digest) hash << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    check(hash.str() == "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d", "pinned CPU model checksum");
    const auto number = std::to_string(port());
    const auto logfile = "/tmp/ezlocalai-cache-test-" + std::to_string(getpid()) + ".log";
    const pid_t pid = fork(); check(pid >= 0, "fork");
    if (!pid) {
        setsid(); const int fd = open(logfile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        dup2(fd, STDOUT_FILENO); dup2(fd, STDERR_FILENO); close(fd);
        execl(binary.c_str(), binary.c_str(), "-m", model.c_str(), "-ngl", "0", "-c", "2048", "-np", "1",
              "-b", "512", "-ub", "128", "--cache-ram", "32", "--threads", "2", "--host", "127.0.0.1",
              "--port", number.c_str(), static_cast<char *>(nullptr)); _exit(127);
    }
    struct Child {
        pid_t pid;
        ~Child() {
            kill(-pid, SIGTERM); int status;
            for (int i = 0; i < 100; ++i) {
                if (waitpid(pid, &status, WNOHANG) == pid) return;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            kill(-pid, SIGKILL); waitpid(pid, &status, 0);
        }
    } child{pid};
    const auto base = "http://127.0.0.1:" + number;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    bool ready = false;
    while (std::chrono::steady_clock::now() < deadline) {
        if (ezlocalai::run_process({"curl", "-sf", "--max-time", "1", base + "/health"}, true).status == 0) { ready = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    check(ready, "CPU cache engine health; inspect /tmp/ezlocalai-cache-test-*.log");
    std::string shared = "Reference section: ";
    for (int i = 0; i < 30; ++i) shared += "The forest is green and the river runs beside the village. ";
    std::map<std::string, std::string> answers;
    int turn = 0;
    for (const std::string name : {"A", "B", "A", "B", "A"}) {
        auto text = shared + "\nConversation " + name + ": ";
        for (int i = 0; i < 16; ++i) text += name == "A" ? "cobalt books are on the shelf. " : "violet cups are beside the door. ";
        text += "\nReply briefly.";
        auto messages = json::array();
        messages.push_back(json{{"role", "system"}, {"content", "You read stories."}});
        messages.push_back(json{{"role", "user"}, {"content", text}});
        const json body{{"messages", messages},
            {"prompt_cache_key", "conversation-" + name}, {"max_tokens", 1}, {"temperature", 0},
            {"seed", 42}, {"cache_prompt", true}};
        const auto result = ezlocalai::run_process({"curl", "-sf", "--max-time", "30", base + "/v1/chat/completions",
            "-H", "Content-Type: application/json", "--data-binary", body.dump()}, true);
        check(result.status == 0, "CPU cache engine inference"); const auto response = json::parse(result.output);
        const auto answer = response.at("choices").at(0).at("message").at("content").get<std::string>();
        if (turn < 2) answers[name] = answer;
        else {
            check(response.at("timings").at("prompt_n").get<int>() <= 4, "returning conversation must restore its full cached tail");
            check(answer == answers[name], "restored conversation produces its original greedy answer");
        }
        std::cout << "conversation " << name << ": " << response.at("timings").dump() << '\n'; ++turn;
    }
    std::filesystem::remove(logfile);
}
int main(int argc, char **argv) {
    try {
        policy();
        if (argc == 4 && std::string(argv[1]) == "--engine") engine(argv[2], argv[3]);
        else check(argc == 1, "usage: test_conversation_cache [--engine LLAMA_SERVER PINNED_MODEL]");
        std::cout << "Conversation cache metadata, prefix selection and bounded eviction passed\n"; return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
