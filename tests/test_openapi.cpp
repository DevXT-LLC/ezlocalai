#include "ezlocalai/native_server.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using Json = nlohmann::json;
void check(bool value, const std::string &message) {
    if (!value) throw std::runtime_error(message);
}
int main(int argc, char **argv) {
    try {
        check(argc == 2, "reference fixture required");
        std::ifstream input(argv[1]);
        const auto fixture = Json::parse(input);
        setenv("EZLOCALAI_DOTENV_PATH", "", 1);
        setenv("ROUTER_MODE", "false", 1);
        setenv("ROUTER_URL", "", 1);
        setenv("VOICE_SERVER", "", 1);
        setenv("LLM_MODEL_RESIDENCY", "resident", 1);
        setenv("N_PARALLEL", "1", 1);
        for (const auto &test : fixture.at("cases")) {
            for (const auto *name : {"DEFAULT_MODEL", "EMBEDDING_MODEL", "WHISPER_MODEL"}) {
                if (test.at("env").contains(name)) setenv(name, test["env"][name].get<std::string>().c_str(), 1);
                else unsetenv(name);
            }
            ezl_env_config config{};
            ezl_load_config(&config);
            ezlocalai::NativeServer server(config, std::make_unique<ezlocalai::EchoInferenceBackend>(config.default_model));
            // Even the first lazy schema request must use captured startup defaults.
            for (const auto *name : {"DEFAULT_MODEL", "EMBEDDING_MODEL", "WHISPER_MODEL"})
                setenv(name, "changed-after-initialization", 1);
            const auto response = server.handle_request("GET", "/openapi.json", "", "");
            const auto expected = test.at("response").get<std::string>();
            check(response.status == 200 && response.content_type == "application/json", "schema HTTP metadata");
            check(Json::parse(response.body) == Json::parse(expected), test.at("name").get<std::string>() + ": complete source schema differs");
            check(response.body == expected, test.at("name").get<std::string>() + ": source serialization differs");
            check(server.handle_request("GET", "/openapi.json", "", "").body == expected, "cached schema changed");
        }
        for (const auto *name : {"DEFAULT_MODEL", "EMBEDDING_MODEL", "WHISPER_MODEL"}) unsetenv(name);
        ezl_env_config config{};
        ezl_load_config(&config);
        ezlocalai::NativeServer server(config, std::make_unique<ezlocalai::EchoInferenceBackend>(config.default_model));
        for (const auto &test : fixture.at("documentation_http")) {
            // The HTTP transport suppresses HEAD bodies; its exact length/body
            // behavior is exercised separately by native_http over TCP.
            if (test.at("method") != "GET") continue;
            const auto response = server.handle_request("GET", test.at("path"), "", "");
            check(response.status == test.at("status"), "documentation status");
            check(response.content_type == test.at("content_type"), "documentation content type");
            check(response.body == test.at("body"), test.at("path").get<std::string>() + ": documentation bytes differ");
            check(std::to_string(response.body.size()) == test.at("content_length"), "documentation length");
        }
        std::cout << fixture.at("cases").size() << " complete source schemas and startup-default lifetimes passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
