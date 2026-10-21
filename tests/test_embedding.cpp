#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <openssl/sha.h>

namespace {
void expect(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::string read_file(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    expect(bool(input), "could not read " + path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
std::vector<std::string> split(const std::string &value, char separator) {
    std::vector<std::string> parts;
    size_t begin = 0;
    do {
        const auto end = value.find(separator, begin);
        parts.push_back(value.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    } while (true);
    return parts;
}
std::string unhex(const std::string &value) {
    expect(value.size() % 2 == 0, "invalid hex fixture");
    std::string result;
    for (size_t i = 0; i < value.size(); i += 2) result += static_cast<char>(std::stoi(value.substr(i, 2), nullptr, 16));
    return result;
}
std::string sha256(const std::string &value) {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char *>(value.data()), value.size(), digest);
    std::ostringstream result;
    for (const auto byte : digest) result << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    return result.str();
}
class FixtureBackend : public ezlocalai::InferenceBackend {
public:
    std::string response;
    std::string request;
    std::string chat(const std::string &) override { throw std::runtime_error("unexpected chat"); }
    std::string completion(const std::string &) override { throw std::runtime_error("unexpected completion"); }
    std::string embedding(const std::string &body) override { request = body; return response; }
};
}

int main(int argc, char **argv) {
    try {
        expect(argc == 3, "reference checkout and embedding fixture required");
        setenv("EZLOCALAI_DOTENV_PATH", "", 1);
        for (const auto &entry : ezlocalai::environment_defaults) unsetenv(entry.first.data());
        ezl_env_config config{};
        ezl_load_config(&config);
        std::snprintf(config.api_key, sizeof(config.api_key), "%s", "test-secret");
        auto backend = std::make_unique<FixtureBackend>();
        auto *fixture = backend.get();
        ezlocalai::NativeServer server(config, std::move(backend));
        size_t count = 0;
        std::istringstream lines(read_file(argv[2]));
        for (std::string line; std::getline(lines, line);) {
            if (line.empty()) continue;
            if (line.front() == '#') {
                const auto marker = line.find(" sha256=");
                if (marker != std::string::npos) {
                    const auto path = std::string(argv[1]) + "/" + line.substr(2, marker - 2);
                    expect(sha256(read_file(path)) == line.substr(marker + 8), "reference changed; refresh embedding oracle: " + path);
                }
                continue;
            }
            const auto columns = split(line, '\t');
            expect(columns.size() == 7, "malformed embedding fixture");
            for (const auto &entry : ezlocalai::environment_defaults) unsetenv(entry.first.data());
            for (const auto &entry : split(columns[1], ';')) {
                if (entry.empty()) continue;
                const auto equals = entry.find('=');
                setenv(entry.substr(0, equals).c_str(), unhex(entry.substr(equals + 1)).c_str(), 1);
            }
            fixture->response = unhex(columns[3]);
            fixture->request.clear();
            const auto response = server.handle_request("POST", "/v1/embeddings", "Bearer test-secret", unhex(columns[2]), "application/json");
            expect(response.status == std::stoi(columns[4]), columns[0] + " status: " + std::to_string(response.status));
            expect(response.body == unhex(columns[5]), columns[0] + " response: " + response.body + " expected " + unhex(columns[5]));
            expect(fixture->request == (columns[6] == "-" ? "" : unhex(columns[6])), columns[0] + " backend request: " + fixture->request);
            ++count;
        }
        expect(count == 86, "all embedding reference cases executed");
        setenv("EMBEDDING_ENABLED", "FALSE", 1);
        const auto disabled = server.handle_request("POST", "/v1/embeddings", "Bearer test-secret", "{\"input\":\"hello\"}", "application/json");
        expect(disabled.status == 404 && disabled.body == "{\"detail\":\"Embeddings are disabled.\"}", "disabled embedding response");
        const auto invalid = server.handle_request("POST", "/v1/embeddings", "Bearer test-secret", "{}", "application/json");
        expect(invalid.status == 422, "embedding validation precedes disabled check");
        const auto queue = server.handle_request("GET", "/v1/queue", "Bearer test-secret", "");
        expect(queue.body.find("\"total_queued\":0") != std::string::npos, "embeddings do not enter the LLM queue");
        std::cout << "Verified " << count << " reference embedding contract cases, disabled handling, and queue isolation\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
