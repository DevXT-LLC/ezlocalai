#include "ezlocalai/tunnel_connection.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
using Json = nlohmann::json;
int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::runtime_error("fixture required");
        std::ifstream input(argv[1]); const auto fixture = Json::parse(input);
        const std::string key = "dGhlIHNhbXBsZSBub25jZQ==";
        if (ezlocalai::tunnel_websocket_accept(key) != "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=") throw std::runtime_error("standard challenge");
        for (const auto &row : fixture.at("handshakes")) {
            std::string headers = row.value("version", "HTTP/1.1") + " " + std::to_string(row.value("status", 101)) + " Test\r\n";
            for (const auto &pair : row.at("headers")) {
                auto value = pair[1].get<std::string>();
                size_t at;
                while ((at = value.find("$accept")) != std::string::npos) value.replace(at, 7, ezlocalai::tunnel_websocket_accept(key));
                headers += pair[0].get<std::string>() + ": " + value + "\r\n";
            }
            if (ezlocalai::tunnel_validate_upgrade(headers + "\r\n", key) != row.at("accepted").get<bool>())
                throw std::runtime_error("handshake: " + row.at("name").get<std::string>());
        }
        for (const auto &row : fixture.at("urls")) {
            auto target = ezlocalai::tunnel_connection_url(row.at("input"), row.at("worker_id"));
            Json actual{{"host", target.url.host}, {"port", target.url.port()}, {"target", target.url.request_target()},
                {"host_header", target.url.host_header()}, {"authorization", target.authorization}};
            for (auto it = actual.begin(); it != actual.end(); ++it) if (it.value() != row.at(it.key()))
                throw std::runtime_error(row.at("input").get<std::string>() + ": " + actual.dump());
        }
        for (const auto &row : fixture.at("backoff")) {
            ezlocalai::TunnelReconnectBackoff backoff;
            for (size_t i = 0; i < row.at("waits").size(); ++i)
                if (backoff.disconnected(row.at("durations")[i], row.at("failures")[i]) != row.at("waits")[i].get<double>())
                    throw std::runtime_error("source backoff sequence");
        }
        std::cout << fixture.at("handshakes").size() << " source handshakes, " << fixture.at("urls").size()
            << " URLs and " << fixture.at("backoff").size() << " reconnect traces passed\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
