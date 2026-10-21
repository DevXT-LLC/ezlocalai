#include "ezlocalai/process.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;
using Json = nlohmann::json;
std::string read(const fs::path &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot read " + path.string());
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
void write(const fs::path &path, const std::string &text) { std::ofstream file(path); file << text; if (!file) throw std::runtime_error("Cannot write evidence"); }
std::string sha256(const std::string &text) {
    unsigned char bytes[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char *>(text.data()), text.size(), bytes);
    std::ostringstream result;
    for (auto byte : bytes) result << std::hex << std::setw(2) << std::setfill('0') << int(byte);
    return result.str();
}
void check(bool condition, const std::string &message) { if (!condition) throw std::runtime_error(message); }
int main(int argc, char **argv) {
    Json record;
    fs::path output;
    try {
        check(argc == 4, "usage: test_speculative_model CASES CONFIG OUTPUT_DIRECTORY");
        const auto fixture = Json::parse(read(argv[1])), config = Json::parse(read(argv[2]));
        output = fs::absolute(argv[3]); fs::create_directories(output);
        const auto mode = config.at("backend").get<std::string>();
        record = {{"backend", mode}, {"greedy", Json::array()}, {"cases", Json::array()}};
        size_t sequence = 0;
        auto request = [&](const std::string &base, const std::string &path, const std::string &key, const Json &body = nullptr) {
            const auto response = output / ("response-" + std::to_string(sequence++) + ".txt");
            std::vector<std::string> command{"/usr/bin/curl", "-sS", "--max-time", "240", "-o", response.string(), "-w", "%{http_code}", "-H", "Content-Type: application/json"};
            if (!key.empty()) command.insert(command.end(), {"-H", "Authorization: Bearer " + key});
            if (!body.is_null()) {
                write(output / "request.json", body.dump());
                command.insert(command.end(), {"--data-binary", "@" + (output / "request.json").string()});
            }
            command.push_back(base + path);
            const auto result = ezlocalai::run_process(command, true, true);
            check(result.status == 0, "HTTP transport failed; response " + response.string());
            check(result.output == "200", "HTTP " + result.output + "; response " + response.string());
            return read(response);
        };
        const auto worker = config.at("worker_url").get<std::string>(), router = config.at("router_url").get<std::string>(), engine = config.at("engine_url").get<std::string>();
        const auto worker_key = config.value("worker_key", ""), router_key = config.value("router_key", ""), engine_key = config.value("engine_key", "");
        record["before"] = Json::parse(request(worker, "/v1/resources", worker_key));
        const auto &runtime = record["before"]["model_lifecycle"]["loaded_llm_runtime"];
        check(runtime.size() == 1 && runtime[0]["speculative_type"] == mode, "runtime speculative backend differs from requested backend");
        check(record["before"]["parallel"]["n_ctx"] == config.at("context"), "allocated context differs from requested context");
        request(engine, "/v1/completions", engine_key, {{"prompt", "Warmup: count to ten."}, {"max_tokens", 32}, {"temperature", 0}});
        for (const auto &test : fixture.at("greedy_cases")) {
            const auto started = std::chrono::steady_clock::now();
            const auto response = Json::parse(request(engine, "/v1/chat/completions", engine_key, test.at("request")));
            const auto &message = response.at("choices").at(0).at("message");
            const auto content = (message.contains("reasoning_content") && message["reasoning_content"].is_string() ? message["reasoning_content"].get<std::string>() : "") + message.at("content").get<std::string>();
            check(!content.empty(), "empty greedy response");
            record["greedy"].push_back({{"name", test.at("name")}, {"sha256", sha256(content)}, {"answer", content}, {"timings", response.value("timings", Json::object())}, {"usage", response.value("usage", Json::object())}, {"seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()}});
            write(output / "result.json", record.dump(2));
            std::cout << mode << " greedy " << test.at("name") << " completed\n" << std::flush;
        }
        long long drafted = 0, accepted = 0;
        bool passed = true;
        for (const bool stream : {false, true}) for (const auto &test : fixture.at("cases")) {
            const auto started = std::chrono::steady_clock::now();
            Json item{{"name", test.at("name")}, {"stream", stream}, {"passed", false}};
            try {
                Json body{{"model", config.at("model")}, {"worker", config.at("label")}, {"messages", Json::array({{{"role", "user"}, {"content", test.at("content")}}})}, {"stream", stream}, {"max_tokens", test.at("max_tokens")}, {"temperature", 0}, {"seed", 42}, {"cache_prompt", true}, {"chat_template_kwargs", {{"enable_thinking", false}}}};
                const auto response = request(router, "/v1/chat/completions", router_key, body);
                std::string answer;
                Json timings = Json::object();
                if (stream) {
                    bool done = false;
                    size_t chunks = 0;
                    std::istringstream lines(response);
                    for (std::string line; std::getline(lines, line);) {
                        if (!line.empty() && line.back() == '\r') line.pop_back();
                        if (line.compare(0, 5, "data:")) continue;
                        auto data = line.substr(5); while (!data.empty() && data.front() == ' ') data.erase(0, 1);
                        if (data == "[DONE]") { done = true; break; }
                        const auto chunk = Json::parse(data);
                        check(!chunk.contains("error"), "SSE error event");
                        ++chunks;
                        for (const auto &choice : chunk.value("choices", Json::array())) {
                            const auto delta = choice.value("delta", Json::object());
                            if (delta.contains("content") && delta["content"].is_string()) answer += delta["content"].get<std::string>();
                        }
                        if (chunk.contains("timings")) timings = chunk["timings"];
                    }
                    check(done && chunks > 0, "SSE did not complete with DONE");
                    item["chunks"] = chunks;
                } else {
                    const auto decoded = Json::parse(response);
                    answer = decoded.at("choices").at(0).at("message").at("content");
                    timings = decoded.value("timings", Json::object());
                    item["usage"] = decoded.value("usage", Json::object());
                }
                item["answer"] = answer; item["timings"] = timings;
                drafted += timings.value("draft_n", 0LL); accepted += timings.value("draft_n_accepted", 0LL);
                auto flattened = answer;
                for (auto &ch : flattened) if (ch == '\n' || ch == '\r') ch = ' ';
                check(std::regex_search(flattened, std::regex(test.at("expected").get<std::string>(), std::regex::icase)), "answer did not match the source fixture");
                item["passed"] = true;
            } catch (const std::exception &error) { item["error"] = error.what(); passed = false; }
            item["seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            record["cases"].push_back(item);
            write(output / "result.json", record.dump(2));
            std::cout << mode << ' ' << test.at("name") << " stream=" << stream << " passed=" << item["passed"] << '\n' << std::flush;
        }
        record["drafted"] = drafted; record["accepted"] = accepted;
        record["after"] = Json::parse(request(worker, "/v1/resources", worker_key));
        record["queue"] = Json::parse(request(worker, "/v1/queue/status", worker_key));
        check(record["queue"]["active_requests"] == 0 && record["queue"]["processing_count"] == 0, "queue did not drain");
        check(mode == "none" ? drafted == 0 : drafted > 0 && accepted > 0, "speculative execution counters do not match backend");
        record["passed"] = passed;
        write(output / "result.json", record.dump(2));
        check(passed, "one or more source vision cases failed");
        return 0;
    } catch (const std::exception &error) {
        if (!output.empty()) { record["error"] = error.what(); record["passed"] = false; write(output / "result.json", record.dump(2)); }
        std::cerr << error.what() << '\n';
        return 1;
    }
}
