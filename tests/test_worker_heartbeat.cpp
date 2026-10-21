#include "ezlocalai/worker_heartbeat.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

int main(int argc, char **argv) {
    using Json = nlohmann::json;
    try {
        if (argc != 2) throw std::runtime_error("reference fixture required");
        std::ifstream file(argv[1]);
        const auto fixture = Json::parse(file);
        for (const auto &test : fixture.at("cases")) {
            Json trace = Json::array();
            size_t post = 0, waits = 0;
            bool stopped = false;
            ezlocalai::WorkerHeartbeatOperations ops;
            ops.stopped = [&] { return stopped; };
            ops.post = [&](const std::string &path) {
                const auto &outcomes = test.at("outcomes");
                const bool ok = post < outcomes.size() ? outcomes.at(post).get<bool>() : true;
                ++post; trace.push_back({{"post", path}, {"ok", ok}}); return ok;
            };
            ops.wait = [&](double seconds) {
                trace.push_back({{"wait", seconds}});
                stopped = ++waits >= test.at("wait_limit").get<size_t>();
            };
            ezlocalai::run_worker_heartbeat(test.at("interval"), ops);
            if (trace != test.at("trace")) throw std::runtime_error(test.at("name").get<std::string>() + ": heartbeat trace differs: " + trace.dump());
        }
        std::cout << fixture.at("cases").size() << " complete Python heartbeat/recovery/shutdown traces passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
