#include "ezlocalai/resource_manager.hpp"
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
using Json = nlohmann::json;
using namespace ezlocalai;
namespace {
void check(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
std::string read(const std::string &path) { std::ifstream in(path); check(bool(in), "Missing " + path); return {std::istreambuf_iterator<char>(in), {}}; }
std::string sha256(const std::string &bytes) {
    unsigned char digest[SHA256_DIGEST_LENGTH]; SHA256(reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size(), digest);
    std::ostringstream out; for (auto b : digest) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(b); return out.str();
}
}
int main(int argc, char **argv) {
    try {
        check(argc == 3, "Reference root and captured cases required");
        const auto fixture = Json::parse(read(argv[2]));
        const auto source = std::string(argv[1]) + "/Pipes.py";
        if (std::filesystem::exists(source)) check(sha256(read(source)) == fixture.at("source_sha256"), "Python source changed");
        else std::cout << "Reference checkout absent; testing captured source results\n";
        size_t count = 0;
        for (const auto &row : fixture.at("cases")) {
            Json events = Json::array(), result = Json::array(); bool failed = false;
            const auto probe = [&](const std::string &name) { events.push_back(name); if (row.at("fail") == name) throw std::runtime_error("fixture failure"); };
            const auto bytes = [&](int i, const char *name) { probe(std::string(name) + ':' + std::to_string(i)); return row.at("data").at(i).at(name).get<double>(); };
            try {
                const auto stats = collect_cuda_allocator_stats({
                    [&] { probe("available"); return row.at("available").get<bool>(); },
                    [&] { probe("count"); return row.at("count").get<int>(); },
                    [&](int i) { probe("info:" + std::to_string(i)); const auto &d = row.at("data").at(i); return std::pair<double, double>{d.at("free"), d.at("total")}; },
                    [&](int i) { return bytes(i, "allocated"); }, [&](int i) { return bytes(i, "reserved"); }, [&](int i) { return bytes(i, "peak"); }
                });
                for (const auto &s : stats) {
                    if (!s.error.empty()) result.push_back({{"index", s.index}, {"error", s.error}});
                    else result.push_back({{"index", s.index}, {"allocated_gb", s.allocated_gb}, {"reserved_gb", s.reserved_gb}, {"max_allocated_gb", s.max_allocated_gb}, {"free_gb", s.free_gb}, {"total_gb", s.total_gb}});
                }
            } catch (const std::exception &) { failed = true; result = nullptr; }
            const auto label = "case " + std::to_string(count++);
            check(failed == row.at("error"), label + " exception boundary");
            check(events == row.at("events"), label + " probe order");
            check(result == row.at("result"), label + " counters/rounding");
        }
        std::cout << count << " Python allocator cases passed\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
