#include "ezlocalai/edge_tts.hpp"
#include "ezlocalai/wakeword_tts.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace {
using Json = nlohmann::json; namespace fs = std::filesystem;
void check(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
std::string hex(const std::string &bytes) { const char *digits = "0123456789abcdef"; std::string output; for (unsigned char c : bytes) { output += digits[c >> 4]; output += digits[c & 15]; } return output; }
std::string unhex(const std::string &value) { std::string output; for (size_t i = 0; i < value.size(); i += 2) output += char(std::stoul(value.substr(i, 2), nullptr, 16)); return output; }
std::string audio_frame() { const std::string header = "X:1\r\nPath:audio\r\nContent-Type:audio/mpeg\r\n"; return std::string({char(header.size() >> 8), char(header.size() & 255)}) + header + "abc"; }
}
int main(int argc, char **argv) {
    try {
        check(argc >= 2, "Edge reference root required"); const auto root = fs::path(argv[1]) / "edge";
        if (argc > 2 && std::string(argv[2]) == "live") {
            Json record{{"attempts", Json::array()}}; ezlocalai::EdgeTtsOperations operations; operations.clock_skew = std::make_shared<std::atomic<double>>(0);
            operations.exchange = [&](const auto &request, auto &stream) { const auto result = ezlocalai::edge_tts_exchange(request, stream); record["attempts"].push_back(Json{{"status", result.status}, {"date", result.headers.count("date") ? result.headers.at("date") : ""}}); return result; };
            try {
                const auto mp3 = ezlocalai::edge_tts_mp3("hello ezlocalai", "en-US-GuyNeural", operations), wav = ezlocalai::wakeword_decode_mp3(mp3);
                std::ofstream(root / "native_live.mp3", std::ios::binary).write(mp3.data(), mp3.size()); std::ofstream(root / "native_live.wav", std::ios::binary).write(wav.data(), wav.size());
                record["mp3_bytes"] = mp3.size(); record["wav_bytes"] = wav.size(); std::ofstream(root / "native_live.json") << record.dump(2) << '\n'; std::cout << record.dump() << '\n'; return 0;
            } catch (const ezlocalai::TtsError &error) { record["error"] = error.category; record["message"] = error.what(); record["clock_skew"] = operations.clock_skew->load(); std::ofstream(root / "native_live.json") << record.dump(2) << '\n'; throw; }
        }
        std::ifstream file(root / "reference.json"); const auto fixture = Json::parse(file);
        for (const auto &row : fixture.at("text")) check(Json(ezlocalai::edge_tts_chunks(row.at("input"))) == row.at("chunks"), "Edge text splitting");
        for (const auto &row : fixture.at("ssml")) {
            try { const auto actual = ezlocalai::edge_tts_ssml("hello &amp; 猫", row.at("voice")); check(!row.contains("error") && actual == row.at("ssml"), "Edge SSML mismatch"); }
            catch (const ezlocalai::TtsError &error) { check(row.contains("error") && error.category == row.at("error") && std::string(error.what()) == row.at("message"), "Edge voice validation mismatch"); }
        }
        for (const auto &row : fixture.at("tokens")) check(ezlocalai::edge_tts_token(row.at("time")) == row.at("token"), "Edge token mismatch");
        const auto request = ezlocalai::edge_tts_request("hello", "en-US-GuyNeural", 1792400000, 0, std::string(32, '0'), std::string(32, '0'), std::string(32, 'A'));
        check(Json{{"url", request.url}, {"headers", request.headers}, {"messages", request.messages}} == fixture.at("request"), "Edge request wire data mismatch");
        for (const auto &row : fixture.at("streams")) {
            ezlocalai::EdgeTtsStream stream;
            try {
                for (const auto &frame : row.at("frames")) if (!stream.accept(frame.at("type") == "text" ? 1 : 2, frame.at("type") == "text" ? frame.at("data").get<std::string>() : unhex(frame.at("data")))) break;
                const auto audio = stream.finish(); check(!row.contains("error") && hex(audio) == row.at("audio"), "Edge audio mismatch");
            } catch (const ezlocalai::TtsError &error) { check(row.contains("error") && error.category == row.at("error") && std::string(error.what()) == row.at("message"), "Edge stream error mismatch: " + row.at("name").get<std::string>() + ": " + error.what()); }
        }
        for (const bool missing_date : {false, true}) {
            int calls = 0; ezlocalai::EdgeTtsOperations operations; operations.time = [] { return 1792400000.; }; operations.identifier = [](bool uuid) { return std::string(32, uuid ? '0' : 'A'); }; operations.clock_skew = std::make_shared<std::atomic<double>>(0);
            operations.exchange = [&](const auto &request, auto &stream) {
                ++calls; if (calls == 1) return ezlocalai::EdgeTtsResponse{403, missing_date ? std::map<std::string, std::string>{} : std::map<std::string, std::string>{{"Date", "Mon, 19 Oct 2026 12:00:00 GMT"}}};
                check(request.url.find(ezlocalai::edge_tts_token(1792411200.)) != std::string::npos, "Corrected clock token"); stream.accept(2, audio_frame()); return ezlocalai::EdgeTtsResponse{};
            };
            try { check(ezlocalai::edge_tts_mp3("hello", "en-US-GuyNeural", operations) == "abc" && !missing_date && calls == 2, "Edge retry result"); }
            catch (const ezlocalai::TtsError &error) { check(missing_date && calls == 1 && error.category == "SkewAdjustmentError", "Edge missing-date failure"); }
        }
        std::cout << "10 text, 7 SSML, 8 token, 11 stream, exact request and clock-skew retry checks passed\n"; return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
