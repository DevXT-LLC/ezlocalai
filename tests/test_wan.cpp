#include "ezlocalai/wan_video.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/process.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/image_processing.hpp"

#include <arpa/inet.h>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <netinet/in.h>
#include <openssl/sha.h>
#include <poll.h>
#include <sstream>
#include <sys/socket.h>

namespace fs = std::filesystem;
namespace {
void expect(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::string read(const fs::path &path) { std::ifstream in(path, std::ios::binary); return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()}; }
std::vector<std::string> split(const std::string &value, char delimiter) {
    std::vector<std::string> out; size_t start = 0;
    do { const auto end = value.find(delimiter, start); out.push_back(value.substr(start, end == std::string::npos ? end : end - start)); if (end == std::string::npos) break; start = end + 1; } while (true);
    return out;
}
std::string unhex(const std::string &value) { std::string out; for (size_t i = 0; i < value.size(); i += 2) out += static_cast<char>(std::stoi(value.substr(i, 2), nullptr, 16)); return out; }
std::vector<std::string> vector_value(const std::string &value) { std::vector<std::string> out; if (value != "-") for (auto item : split(value, ',')) out.push_back(unhex(item)); return out; }
std::string sha256(const std::string &value) {
    unsigned char hash[SHA256_DIGEST_LENGTH]; SHA256(reinterpret_cast<const unsigned char *>(value.data()), value.size(), hash);
    std::ostringstream out; for (auto byte : hash) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte); return out.str();
}
void environment() {
    for (const auto &entry : ezlocalai::environment_defaults) unsetenv(entry.first.data());
    for (const char *name : {"HF_HUB_CACHE", "HF_HUB_OFFLINE", "HF_ENDPOINT", "WAN_FIXTURE_ARGUMENTS", "WAN_FIXTURE_DELAY", "WAN_FIXTURE_EXIT", "WAN_FIXTURE_MISSING_FRAME", "WAN_FIXTURE_INPUT_CAPTURE"}) unsetenv(name);
    setenv("EZLOCALAI_DOTENV_PATH", "", 1); setenv("ROUTER_DRAIN_FILE", "", 1); setenv("USAGE_FILE", "usage.json", 1);
    setenv("DEFAULT_MODEL", "none", 1); setenv("VIDEO_ENABLED", "true", 1); setenv("VIDEO_MODEL", "wan2.2", 1);
    setenv("SDCPP_BIN", TEST_WAN_FIXTURE_BIN, 1); setenv("EZLOCALAI_URL", "", 1);
}
size_t reference_cases(const fs::path &root, const fs::path &fixture) {
    std::istringstream lines(read(fixture)); size_t count = 0;
    const std::map<std::string, std::string> assets = {{"high", "high model.gguf"}, {"low", "low model.gguf"}, {"encoder", "encoder.gguf"}, {"vae", "vae.safetensors"}};
    for (std::string line; std::getline(lines, line);) {
        if (line.empty()) continue;
        if (line[0] == '#') { const auto marker = line.find(" sha256="); if (marker != std::string::npos) expect(sha256(read(root / line.substr(2, marker - 2))) == line.substr(marker + 8), "Wan source changed; refresh reference cases"); continue; }
        const auto c = split(line, '\t'); environment(); ++count;
        if (c[1] == "command") {
            for (const auto &entry : split(c[2], ';')) { const auto equal = entry.find('='); setenv(entry.substr(0, equal).c_str(), unhex(entry.substr(equal + 1)).c_str(), 1); }
            ezlocalai::WanVideoSettings s; s.steps = std::stoll(c[4]); s.guidance = std::stod(c[5]);
            const auto command = ezlocalai::wan_video_command("sd cli", unhex(c[3]), assets, "frame_%06d.png", "literal $(touch nope); ' music", "negative", s, unhex(c[6]));
            expect(command == vector_value(c[7]), c[0] + " argv mismatch");
        } else if (c[1] == "settings") {
            ezlocalai::WanVideoSettings s; s.size = unhex(c[2]); s.frames = std::stoll(c[3]); s.fps = std::stoll(c[4]); s.steps = std::stoll(c[5]); s.guidance = std::stod(c[6]);
            std::string actual;
            try { s = ezlocalai::normalize_wan_settings(s); actual = std::to_string(s.width) + "," + std::to_string(s.height) + "," + std::to_string(s.frames) + "," + std::to_string(s.fps); }
            catch (const std::exception &e) { actual = "ERROR:" + std::string(e.what()); }
            expect(actual == unhex(c[7]), c[0] + ": " + actual + " expected " + unhex(c[7]));
        } else if (c[1] == "assets") {
            std::vector<std::string> actual; std::string error;
            try { for (const auto &asset : ezlocalai::wan_model_assets(c[2] == "1", unhex(c[3]))) actual.insert(actual.end(), {asset.repository, asset.filename, asset.revision}); }
            catch (const std::exception &e) { error = e.what(); }
            if (c[4].rfind("ERROR:", 0) == 0) expect(error == unhex(c[4].substr(6)), c[0] + " asset error");
            else expect(error.empty() && actual == vector_value(c[4]), c[0] + " pinned assets");
        } else throw std::runtime_error("unknown fixture");
    }
    expect(count == 818, "all Wan reference cases"); return count;
}
class HubServer {
public:
    HubServer() {
        fd_ = socket(AF_INET, SOCK_STREAM, 0); expect(fd_ >= 0, "Hub socket");
        sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        expect(bind(fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0, "Hub bind");
        socklen_t size = sizeof(address); getsockname(fd_, reinterpret_cast<sockaddr *>(&address), &size);
        url = "http://127.0.0.1:" + std::to_string(ntohs(address.sin_port));
        expect(listen(fd_, 8) == 0, "Hub listen");
        thread_ = std::thread([this] { serve(); });
    }
    ~HubServer() { stopped_ = true; shutdown(fd_, SHUT_RDWR); thread_.join(); close(fd_); }
    std::string url;
    std::atomic<int> heads{0}, gets{0};
    std::atomic<bool> fail{false}, auth_missing{false};
private:
    void serve() {
        while (!stopped_) {
            pollfd poller{fd_, POLLIN, 0}; if (poll(&poller, 1, 100) <= 0) continue;
            const int peer = accept(fd_, nullptr, nullptr); if (peer < 0) continue;
            timeval timeout{2, 0}; setsockopt(peer, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            std::string raw; char buffer[4096];
            while (raw.find("\r\n\r\n") == std::string::npos) { const auto n = recv(peer, buffer, sizeof(buffer), 0); if (n <= 0) break; raw.append(buffer, n); }
            std::istringstream line(raw); std::string method, path; line >> method >> path;
            if (raw.find("Authorization: Bearer test-readonly\r\n") == std::string::npos) auth_missing = true;
            if (method == "HEAD") ++heads; else ++gets;
            const auto content = "fixture:" + path;
            const int status = fail && method == "GET" ? 503 : 200;
            std::string response = "HTTP/1.1 " + std::to_string(status) + " Response\r\nETag: \"" + sha256(path) + "\"\r\nContent-Length: " + std::to_string(content.size()) + "\r\nConnection: close\r\n\r\n";
            if (method != "HEAD") response += content;
            send(peer, response.data(), response.size(), MSG_NOSIGNAL); shutdown(peer, SHUT_RDWR); close(peer);
        }
    }
    int fd_; std::atomic<bool> stopped_{false}; std::thread thread_;
};
void cache_cases() {
    environment(); HubServer server;
    setenv("HF_ENDPOINT", server.url.c_str(), 1); setenv("HF_TOKEN", "test-readonly", 1);
    const auto asset = ezlocalai::wan_model_assets(false, "Q4_K_M").front();
    auto first = std::async(std::launch::async, [&] { return ezlocalai::download_hub_asset(asset, "cache with spaces"); });
    auto second = std::async(std::launch::async, [&] { return ezlocalai::download_hub_asset(asset, "cache with spaces"); });
    const auto path = first.get(); expect(second.get() == path, "concurrent download path");
    expect(server.heads == 1 && server.gets == 1 && !server.auth_missing, "single authenticated download under concurrent calls");
    expect(fs::is_symlink(path) && read(path).find(asset.filename) != std::string::npos, "Hub blob/snapshot symlink and bytes");
    setenv("HF_HUB_OFFLINE", "1", 1);
    expect(ezlocalai::download_hub_asset(asset, "cache with spaces") == path, "offline cached model");
    auto missing = asset; missing.filename = "missing.gguf";
    bool failed = false; try { ezlocalai::download_hub_asset(missing, "cache with spaces"); } catch (...) { failed = true; }
    expect(failed && server.heads == 1, "offline miss does not connect");
    unsetenv("HF_HUB_OFFLINE"); server.fail = true; failed = false;
    try { ezlocalai::download_hub_asset(missing, "cache with spaces"); } catch (...) { failed = true; }
    expect(failed && ezlocalai::cached_hub_asset(missing, "cache with spaces").empty(), "failed download not published");
    server.fail = false;
    expect(!ezlocalai::download_hub_asset(missing, "cache with spaces").empty(), "retry failed download");
    unsetenv("HF_TOKEN"); unsetenv("HF_ENDPOINT");
}
void seed_assets(const fs::path &cache) {
    for (const bool image : {false, true}) for (const auto &asset : ezlocalai::wan_model_assets(image, "Q4_K_M")) {
        auto repo = asset.repository; size_t pos = 0; while ((pos = repo.find('/', pos)) != std::string::npos) { repo.replace(pos, 1, "--"); pos += 2; }
        const auto path = cache / ("models--" + repo) / "snapshots" / asset.revision / asset.filename;
        fs::create_directories(path.parent_path()); std::ofstream(path) << "fixture model";
    }
}
size_t output_count() { size_t count = 0; for (const auto &entry : fs::directory_iterator("outputs")) if (entry.path().extension() == ".mp4") ++count; return count; }
void check_conditioning(const fs::path &fixtures, const fs::path &capture) {
    const auto captured = ezlocalai::decode_rgb_image(read(capture));
    std::istringstream cases(read(fixtures / "image_cases.tsv")); std::string expected;
    for (std::string line; std::getline(cases, line);) {
        const auto c = split(line, '\t');
        if (c.size() == 8 && c[0] == "palette-transparent.png" && c[1] == "64" && c[2] == "64") expected = c[7];
    }
    expect(captured.width == 64 && captured.height == 64 && !expected.empty()
        && sha256(std::string(captured.pixels.begin(), captured.pixels.end())) == expected, "sd-cli receives exact Pillow conditioning pixels");
}
void generation_cases(const fs::path &fixtures) {
    environment(); seed_assets("models"); setenv("HF_HUB_OFFLINE", "1", 1); setenv("WAN_FIXTURE_ARGUMENTS", "sd-argv", 1);
    ezlocalai::WanVideo backend("wan2.2", "cpu");
    ezlocalai::WanVideoSettings settings; settings.size = "65x79"; settings.frames = 10; settings.fps = 12; settings.steps = 18;
    const auto output = backend.generate("literal $(touch must-not-exist)", settings, "");
    expect(fs::file_size(output) > 0 && !fs::exists("must-not-exist"), "generated real MP4 without shell expansion");
    const auto probe = ezlocalai::run_process({"ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames,r_frame_rate", "-of", "json", output}, true);
    expect(probe.status == 0 && probe.output.find("\"width\": 64") != std::string::npos && probe.output.find("\"height\": 64") != std::string::npos && probe.output.find("\"nb_frames\": \"9\"") != std::string::npos && probe.output.find("12/1") != std::string::npos, "MP4 dimensions, aligned frame count and FPS: " + probe.output);
    const auto before = output_count();
    for (const auto &failure : {std::pair{"WAN_FIXTURE_EXIT", "7"}, {"WAN_FIXTURE_MISSING_FRAME", "1"}, {"WAN_FIXTURE_DELAY", "2000"}}) {
        setenv(failure.first, failure.second, 1); setenv("VIDEO_GENERATION_TIMEOUT", failure.first == std::string("WAN_FIXTURE_DELAY") ? "0.05" : "10", 1);
        bool caught = false; try { backend.generate("test", settings, ""); } catch (const std::runtime_error &) { caught = true; }
        expect(caught && output_count() == before, "generation failure cleans partial MP4"); unsetenv(failure.first);
    }
    const auto args = split(read("sd-argv"), '\n');
    for (size_t i = 0; i + 1 < args.size(); ++i) if (args[i] == "-o") expect(!fs::exists(fs::path(args[i + 1]).parent_path()), "generation temporary files cleaned");
    setenv("VIDEO_GENERATION_TIMEOUT", "10", 1);
    const auto url = backend.generate("test", settings, "http://example.test///");
    expect(url.rfind("http://example.test/outputs/", 0) == 0, "trailing slash normalized output URL");
    bool caught = false; try { backend.generate("test", settings, "", "", true); } catch (const std::invalid_argument &e) { caught = std::string(e.what()).find("use image instead of conditions") != std::string::npos; }
    expect(caught, "Wan condition frame rejection");
    ezlocalai::WanVideo image_model("wan2.2-i2v", "cpu"); caught = false;
    try { image_model.generate("test", settings, ""); } catch (const std::invalid_argument &e) { caught = std::string(e.what()) == "The configured Wan I2V model requires an image."; }
    expect(caught, "I2V model requires image");
    // Capture the actual input consumed by sd-cli and compare with the Pillow
    // oracle, exercising decode, fractional center crop and resize end to end.
    setenv("WAN_FIXTURE_INPUT_CAPTURE", "conditioning.png", 1);
    const auto image_output = image_model.generate("test", settings, "", read(fixtures / "images/palette-transparent.png"));
    expect(fs::file_size(image_output) > 0 && read("sd-argv").find("I2V-A14B") != std::string::npos, "I2V assets and MP4 path");
    check_conditioning(fixtures, "conditioning.png");
}
class TrackedBackend : public ezlocalai::InferenceBackend {
public:
    std::atomic<bool> loaded{true}; std::atomic<int> unloads{0}, restores{0};
    std::string chat(const std::string &) override { return "{\"choices\":[]}"; }
    std::string completion(const std::string &body) override { return chat(body); }
    std::string embedding(const std::string &) override { return "{}"; }
    std::string loaded_models_json() const override { return loaded ? "{\"llm\":{\"name\":\"fixture\"}}" : "{}"; }
    std::function<void(bool)> unload_llms_for_media() override { ++unloads; loaded = false; return [this](bool reload) { ++restores; loaded = reload; }; }
};
void api_cases(const fs::path &fixtures) {
    environment(); seed_assets("models"); setenv("HF_HUB_OFFLINE", "1", 1);
    ezl_env_config config{}; ezl_load_config(&config);
    ezlocalai::NativeServer server(config, std::make_unique<ezlocalai::EchoInferenceBackend>("model"));
    const std::string body = R"({"prompt":"test","size":"64x64","num_frames":5,"num_inference_steps":2,"frame_rate":12,"response_format":"b64_json","n":2})";
    auto response = server.handle_request("POST", "/v1/videos/generations", "", body);
    expect(response.status == 200 && response.body.find("\"b64_json\":") != std::string::npos, "local video API success: " + response.body);
    const auto first = response.body.find("\"b64_json\":"); expect(response.body.find("\"b64_json\":", first + 1) != std::string::npos, "n generates separate video items");
    std::istringstream image_cases(read(fixtures / "video_image_cases.tsv")); std::string image_data;
    for (std::string line; std::getline(image_cases, line);) {
        const auto c = split(line, '\t');
        if (c.size() == 5 && c[0] == "palette-transparent.png-1") image_data = unhex(c[1]);
    }
    expect(!image_data.empty(), "image data URL fixture");
    setenv("WAN_FIXTURE_INPUT_CAPTURE", "api-conditioning.png", 1);
    response = server.handle_request("POST", "/v1/videos/generations", "", "{\"prompt\":\"test\",\"size\":\"64x64\",\"num_frames\":5,\"image\":\"" + image_data + "\"}");
    expect(response.status == 200, "image data URL passes through video API");
    check_conditioning(fixtures, "api-conditioning.png");
    response = server.handle_request("POST", "/v1/videos/generations", "", "{\"prompt\":\"test\",\"conditions\":[{\"image\":\"" + image_data + "\"}]}");
    expect(response.status == 400 && response.body.find("use image instead of conditions") != std::string::npos, "valid decoded conditions reach Wan rejection");
    response = server.handle_request("POST", "/v1/videos/generations", "", R"({"prompt":"test","size":"bad"})");
    expect(response.status == 400 && response.body.find("Invalid video dimensions") != std::string::npos, "invalid local video settings");
    response = server.handle_request("POST", "/v1/videos/generations", "", R"({"prompt":"test","image":"http://127.0.0.1/private"})");
    expect(response.status == 500, "private image URL rejected before fetch");
    setenv("WAN_FIXTURE_EXIT", "3", 1);
    response = server.handle_request("POST", "/v1/videos/generations", "", R"({"prompt":"test","size":"64x64","num_frames":5})");
    expect(response.status == 500, "native child failure maps to API error");
    unsetenv("WAN_FIXTURE_EXIT");
    setenv("SDCPP_BIN", "/missing/sd-cli", 1);
    ezlocalai::NativeServer missing(config, std::make_unique<ezlocalai::EchoInferenceBackend>("model"));
    response = missing.handle_request("POST", "/v1/videos/generations", "", body);
    expect(response.status == 500 && response.body == "{\"detail\":\"Video generation produced no output\"}", "failed lazy backend load");

    environment(); setenv("DEFAULT_MODEL", "fixture", 1); setenv("VIDEO_UNLOAD_LLM_DURING_GENERATION", "true", 1);
    setenv("WAN_FIXTURE_ARGUMENTS", "api-video-argv", 1); setenv("WAN_FIXTURE_DELAY", "400", 1); setenv("HF_HUB_OFFLINE", "1", 1);
    ezl_load_config(&config);
    auto tracked = std::make_unique<TrackedBackend>(); auto *backend = tracked.get();
    ezlocalai::NativeServer handoff(config, std::move(tracked));
    for (const auto &invalid : {R"({"prompt":"test","image":"bm90IGFuIGltYWdl"})",
                               R"({"prompt":"test","size":"bad","image":"bm90IGFuIGltYWdl"})",
                               R"({"prompt":"test","conditions":[{"image":""}]})",
                               R"({"prompt":"test","conditions":[{"image":"bm90IGFuIGltYWdl"}]})"}) {
        response = handoff.handle_request("POST", "/v1/videos/generations", "", invalid);
        expect(response.status == 500 && response.content_type == "text/plain; charset=utf-8" && backend->unloads == 0
            && !fs::exists("api-video-argv"), "invalid image rejected before Wan validation and LLM handoff: " + response.body);
    }
    auto video = std::async(std::launch::async, [&] { return handoff.handle_request("POST", "/v1/videos/generations", "", R"({"prompt":"test","size":"64x64","num_frames":5})"); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!fs::exists("api-video-argv") && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    const auto resources = handoff.handle_request("GET", "/v1/resources", "", "");
    expect(resources.body.find("\"text\":{\"capacity\":0") != std::string::npos && resources.body.find("\"video\":{\"capacity\":1,\"in_flight\":1") != std::string::npos, "video handoff resource slots");
    response = handoff.handle_request("POST", "/v1/chat/completions", "", "{\"messages\":[]}");
    expect(response.body.find("temporarily_unavailable") != std::string::npos, "text admission during video");
    expect(video.get().status == 200 && backend->unloads == 1 && backend->restores == 1 && backend->loaded, "video restores prior LLM residency");
    unsetenv("WAN_FIXTURE_DELAY"); setenv("WAN_FIXTURE_EXIT", "5", 1);
    response = handoff.handle_request("POST", "/v1/videos/generations", "", "{\"prompt\":\"test\"}");
    expect(response.status == 500 && backend->unloads == 2 && backend->restores == 2 && backend->loaded, "video failure restores LLM residency");
}
} // namespace
int main(int argc, char **argv) {
    try {
        expect(argc == 3, "reference and fixture required"); const auto root = fs::absolute(argv[1]), fixture = fs::absolute(argv[2]);
        const auto previous = fs::current_path(); char temporary[] = "/tmp/ezlocalai-wan-XXXXXX"; expect(mkdtemp(temporary), "temporary directory"); fs::current_path(temporary);
        const auto count = reference_cases(root, fixture); cache_cases(); generation_cases(fixture.parent_path()); api_cases(fixture.parent_path());
        fs::current_path(previous); fs::remove_all(temporary);
        std::cout << "Verified " << count << " Wan reference cases, cache transfers, subprocess cleanup and real MP4 generation\n"; return 0;
    } catch (const std::exception &e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
