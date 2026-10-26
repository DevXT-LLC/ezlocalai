#include "ezlocalai/process.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "fixture_ports.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <functional>
#include <future>
#include <arpa/inet.h>
#include <sys/socket.h>

namespace fs = std::filesystem;
using Json = nlohmann::json;
using namespace std::chrono_literals;
static void require(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
static void env(const char *name, const std::string &value) { setenv(name, value.c_str(), 1); }
struct Child {
    pid_t pid = -1;
    void start(const fs::path &binary, const fs::path &log) {
        // All children are launched before this test has any threads.
        pid = fork(); require(pid >= 0, "fork");
        if (!pid) {
            int fd = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
            if (fd < 0) _exit(126);
            dup2(fd, STDOUT_FILENO); dup2(fd, STDERR_FILENO); close(fd);
            execl(binary.c_str(), binary.c_str(), nullptr); _exit(127);
        }
    }
    void stop() {
        if (pid <= 0) return;
        kill(pid, SIGTERM);
        // Python permits five seconds for the heartbeat task, followed by a
        // separate ten-second deregistration request. The focused shutdown
        // test checks those boundaries; allow owned-engine cleanup here too.
        const auto deadline = std::chrono::steady_clock::now() + 20s;
        while (std::chrono::steady_clock::now() < deadline) {
            int status = 0;
            if (waitpid(pid, &status, WNOHANG) == pid) {
                pid = -1; require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "clean process exit"); return;
            }
            std::this_thread::sleep_for(20ms);
        }
        throw std::runtime_error("process shutdown exceeded twenty seconds");
    }
    ~Child() {
        // A failed assertion must still let the worker reap its owned engines.
        try { stop(); } catch (...) {}
        if (pid > 0) {
            kill(pid, SIGKILL);
            while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {}
        }
    }
};
static std::string request(const std::string &base, const std::string &path, const Json &body = nullptr,
                           const std::string &key = "client-test-key") {
    std::vector<std::string> args{"curl", "--silent", "--show-error", "--fail-with-body", "--max-time", "10",
        "-H", "Authorization: Bearer " + key, base + path};
    fs::path body_file;
    if (!body.is_null()) {
        auto serialized = body.dump();
        if (serialized.size() > 65536) {
            std::string pattern = (fs::temp_directory_path() / "ezlocalai-router-body-XXXXXX").string();
            const int fd = mkstemp(pattern.data()); require(fd >= 0, "request body temporary file"); close(fd);
            body_file = pattern;
            std::ofstream(body_file, std::ios::binary) << serialized;
            serialized = "@" + body_file.string();
        }
        args.insert(args.end(), {"-H", "Content-Type: application/json", "--data-binary", serialized});
    }
    auto response = ezlocalai::run_process(args, true, true);
    if (!body_file.empty()) fs::remove(body_file);
    require(response.status == 0, path + ": " + response.output);
    return response.output;
}
static void until(const std::function<bool()> &ready, const std::string &message) {
    const auto end = std::chrono::steady_clock::now() + 15s;
    do {
        if (ready()) return;
        std::this_thread::sleep_for(50ms);
    } while (std::chrono::steady_clock::now() < end);
    throw std::runtime_error(message);
}
static void cancel_chat_after_first_chunk(int port, const Json &body, const std::function<void()> &before_close) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    require(fd >= 0, "stream cancellation client socket");
    struct Close { int fd; ~Close() { shutdown(fd, SHUT_RDWR); close(fd); } } cleanup{fd};
    timeval timeout{10, 0};
    require(setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0 &&
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == 0,
        "bound cancellation client socket waits");
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    require(connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0, "connect cancellation client");
    const auto payload = body.dump();
    const auto wire = "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
        "Authorization: Bearer client-test-key\r\nContent-Type: application/json\r\nContent-Length: "
        + std::to_string(payload.size()) + "\r\n\r\n" + payload;
    for (size_t sent = 0; sent < wire.size();) {
        const auto count = send(fd, wire.data() + sent, wire.size() - sent, MSG_NOSIGNAL);
        if (count < 0 && errno == EINTR) continue;
        require(count > 0, "send cancellation request"); sent += static_cast<size_t>(count);
    }
    std::string response; char buffer[4096];
    while (response.find("fixture-stream-cancel-output") == std::string::npos) {
        const auto count = recv(fd, buffer, sizeof(buffer), 0);
        if (count < 0 && errno == EINTR) continue;
        require(count > 0 && response.size() < 1024 * 1024, "receive first live inference chunk");
        response.append(buffer, static_cast<size_t>(count));
    }
    require(response.find("200 OK") != std::string::npos && response.find("[DONE]") == std::string::npos,
        "cancel an active successful stream, not a completed response");
    before_close();
}
static void heartbeat_release_cases(const fs::path &worker_binary, const std::string &router_url,
                                    const fs::path &temporary, bool benchmark = false) {
    // A worker heartbeat sampled during generation must not keep the completed
    // worker busy until its next 30-second periodic publication. Exercise both
    // queue-owned buffered work and the separately owned lazy-stream lease.
    for (const bool tunnel : {false, true}) for (const bool streaming : {false, true}) {
        const std::string name = std::string("heartbeat-release-") + (tunnel ? "tunnel-" : "direct-")
            + (streaming ? "stream" : "buffered");
        const auto directory = temporary / name; fs::create_directories(directory);
        const auto worker_port = std::to_string(fixture_ports());
        const auto engine_port = std::to_string(fixture_ports());
        const auto model_file = directory / "model-Q6_K.gguf"; std::ofstream(model_file) << "GGUF fixture";
        env("PORT", worker_port); env("EZLOCALAI_URL", "http://127.0.0.1:" + worker_port);
        env("EZLOCALAI_NATIVE_BACKEND", "llama.cpp"); env("EZLOCALAI_PRELOAD_MODEL_ON_START", "true");
        env("DEFAULT_MODEL", "integration-model"); env("LLM_MODEL_RESIDENCY", "resident");
        env("LLAMA_CPP_MODEL_PATH", model_file.string()); env("LLAMA_CPP_MMPROJ_PATH", "");
        env("LLAMA_CPP_SERVER_BIN", TEST_EMBEDDING_FIXTURE_BIN); env("LLAMA_CPP_SERVER_PORT", engine_port);
        env("LLAMA_CPP_EXTRA_ARGS", ""); env("LLAMA_CPP_START_TIMEOUT", "2");
        env("GPU_LAYERS", "0"); env("LLM_SPECULATIVE_TYPE", "none"); env("N_PARALLEL", "1");
        env("EMBEDDING_FIXTURE_DIR", directory.string()); env("EZLOCALAI_API_KEY", "local-test-key");
        env("ROUTER_URL", router_url); env("ROUTER_API_KEY", "register-test-key");
        const auto interval = benchmark ? "2" : "30";
        env("WORKER_LABEL", name); env("WORKER_HEARTBEAT_INTERVAL", interval);
        env("WORKER_TUNNEL", tunnel ? "true" : "false");
        for (auto flag : {"TTS_ENABLED", "STT_ENABLED", "EMBEDDING_ENABLED", "IMAGE_ENABLED", "VIDEO_ENABLED", "MUSIC_ENABLED"}) env(flag, "false");
        const std::string sse = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"fixture\"},\"finish_reason\":null}]}\n\n"
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
            "data: [DONE]\n\n";
        env("LLM_FIXTURE_INFERENCE_RESULT", streaming ? sse : R"({"choices":[{"message":{"content":"fixture"},"finish_reason":"stop"}]})");
        env("LLM_FIXTURE_INFERENCE_STATUS", "200");
        Child worker; worker.start(worker_binary, temporary / (name + ".log"));
        Json current;
        const auto visible = [&] {
            const auto workers = Json::parse(request(router_url, "/v1/router/workers")).at("data");
            for (const auto &entry : workers)
                if (entry.at("label") == name) { current = entry; return true; }
            return false;
        };
        until(visible, "heartbeat-release worker registered");
        require(current.at("model_slots").at("integration-model").at("capacity") == 1,
            "heartbeat regression requires one actual model slot");
        const auto id = current.at("worker_id");
        Json first_body{{"model", "integration-model"}, {"worker", name}, {"stream", streaming},
            {"disable_fallback", true}, {"max_tokens", 8}, {"prompt_cache_key", "heartbeat-release-affinity"},
            {"messages", Json::array({{{"role", "user"}, {"content", "heartbeat-release-block"}}})}};
        auto next_body = first_body;
        next_body["messages"] = Json::array({{{"role", "user"}, {"content", "heartbeat-release-follow-up"}}});
        std::future<std::string> first, next;
        // Release before future destructors join on any failed assertion.
        struct Release {
            fs::path path;
            ~Release() { std::ofstream(path) << "release"; }
        } release{directory / ("release-" + engine_port)};
        first = std::async(std::launch::async, [&] { return request(router_url, "/v1/chat/completions", first_body); });
        until([&] { return fs::exists(directory / ("seen-" + engine_port)); }, "first heartbeat regression request reaches engine");
        // Establish exactly the state produced by a periodic busy heartbeat.
        // This setup avoids spending 30 seconds waiting for that sample, and
        // requires the actual worker completion event to clear it afterwards.
        const Json slot{{"capacity", 1}, {"in_flight", 1}, {"queued", 0}};
        request(router_url, "/v1/router/heartbeat",
            {{"worker_id", id}, {"in_flight", 1}, {"queue_depth", 1}, {"queue_capacity", 1},
             {"cap_slots", {{"text", slot}}}, {"model_slots", {{"integration-model", slot}}}}, "register-test-key");
        until([&] {
            return visible() && current.at("reported_in_flight") == 1
                && current.at("router_in_flight") == 1;
        }, "Rust sees busy worker report and active reservation");
        next = std::async(std::launch::async, [&] { return request(router_url, "/v1/chat/completions", next_body); });
        until([&] {
            return Json::parse(request(router_url, "/v1/router/dashboard")).at("router").at("waiting_requests").get<int>() > 0;
        }, "strictly pinned follow-up waits for the occupied worker");
        if (next.wait_for(0ms) == std::future_status::ready)
            throw std::runtime_error("follow-up finished before release: " + next.get());
        const auto released = std::chrono::steady_clock::now();
        std::ofstream(release.path) << "release";
        require(next.wait_for(benchmark ? 5s : 1s) == std::future_status::ready,
            name + ": completed worker remained unavailable beyond one second (periodic interval is thirty seconds)");
        const auto latency = std::chrono::duration<double>(std::chrono::steady_clock::now() - released).count();
        const auto first_response = first.get(), next_response = next.get();
        if (streaming) {
            require(first_response.find("fixture") != std::string::npos && first_response.find("[DONE]") != std::string::npos,
                "initial stream completes normally");
            require(next_response.find("fixture") != std::string::npos && next_response.find("[DONE]") != std::string::npos,
                "follow-up stream completes normally");
        } else {
            require(!Json::parse(first_response).at("choices").empty() && !Json::parse(next_response).at("choices").empty(),
                "buffered requests complete after event-driven idle publication");
        }
        until([&] {
            return visible() && current.at("reported_in_flight") == 0 && current.at("router_in_flight") == 0;
        }, "event publication clears worker and router busy counts");
        worker.stop(); until([&] { return !visible(); }, "heartbeat-release worker deregisters");
        std::cout << name << ": queued follow-up completed " << latency
            << " seconds after release with a " << interval << "-second periodic heartbeat\n" << std::flush;
        unsetenv("LLM_FIXTURE_INFERENCE_RESULT"); unsetenv("LLM_FIXTURE_INFERENCE_STATUS");
    }
}
int main(int argc, char **argv) {
    fs::path temporary;
    const auto cwd = fs::current_path();
    try {
        require(argc == 3 || (argc == 4 && (std::string(argv[3]) == "--heartbeat-release-only" ||
            std::string(argv[3]) == "--heartbeat-release-benchmark")),
            "native worker and Rust WorkerRouter executables required");
        const auto worker_binary = fs::absolute(argv[1]), router_binary = fs::absolute(argv[2]);
        std::string pattern = (fs::temp_directory_path() / "ezlocalai-worker-router-XXXXXX").string();
        const auto created = mkdtemp(pattern.data()); require(created != nullptr, "temporary directory"); temporary = created;
        fs::current_path(temporary); std::ofstream(".env");
        for (const auto &entry : ezlocalai::environment_defaults) env(std::string(entry.first).c_str(), std::string(entry.second));
        env("HOST", "127.0.0.1"); env("LOG_LEVEL", "WARNING"); env("ROUTER_APP_DIR", temporary.string());
        env("TOKIO_WORKER_THREADS", "2"); env("ROUTER_MODE", "false");
        env("CHUTES_API_KEY", ""); env("OPENROUTER_API_KEY", ""); env("ROUTER_WAIT_TIMEOUT", "3");
        env("USAGE_FILE", (temporary / "usage.json").string()); env("ROUTER_DRAIN_FILE", (temporary / "drain.json").string());
        env("ROUTER_ERROR_FILE", (temporary / "errors.json").string()); env("ROUTER_OUTPUTS_DIR", (temporary / "outputs").string());
        const auto router_port = std::to_string(fixture_ports());
        const auto router_url = "http://127.0.0.1:" + router_port;
        Child router;
        auto start_router = [&] {
            env("PORT", router_port); env("EZLOCALAI_API_KEY", "client-test-key"); env("ROUTER_REGISTER_KEY", "register-test-key");
            router.start(router_binary, temporary / "router.log");
            until([&] { try { return Json::parse(request(router_url, "/health")).is_object(); } catch (...) { return false; } }, "router ready");
        };
        start_router();
        if (argc == 4) {
            heartbeat_release_cases(worker_binary, router_url, temporary,
                std::string(argv[3]) == "--heartbeat-release-benchmark");
            router.stop(); fs::current_path(cwd); fs::remove_all(temporary); return 0;
        }
        for (const bool swap : {false, true}) for (const bool tunnel : {false, true}) {
            const std::string name = std::string(tunnel ? "native-tunnel" : "native-direct") + (swap ? "-swap" : "");
            const auto worker_port = std::to_string(fixture_ports());
            env("PORT", worker_port); env("EZLOCALAI_URL", "http://127.0.0.1:" + worker_port);
            env("EZLOCALAI_NATIVE_BACKEND", "echo"); env("EZLOCALAI_PRELOAD_MODEL_ON_START", "false");
            env("DEFAULT_MODEL", "integration-model"); env("EZLOCALAI_API_KEY", "local-test-key");
            env("LLM_MODEL_RESIDENCY", swap ? "swap" : "resident");
            env("LLAMA_CPP_MMPROJ_PATH", "");
            if (swap) {
                env("EZLOCALAI_NATIVE_BACKEND", "llama.cpp"); env("EZLOCALAI_PRELOAD_MODEL_ON_START", "true");
                env("DEFAULT_MODEL", "integration-model,integration-other");
                const auto model = temporary / "model-Q6_K.gguf"; std::ofstream(model) << "GGUF fixture";
                const auto projector = temporary / "explicit-projector.gguf"; std::ofstream(projector) << "GGUF fixture";
                env("LLAMA_CPP_MMPROJ_PATH", projector.string());
                env("LLAMA_CPP_MODEL_PATH", model.string()); env("LLAMA_CPP_SERVER_BIN", TEST_EMBEDDING_FIXTURE_BIN);
                env("EMBEDDING_FIXTURE_DIR", (temporary / name).string()); env("LLAMA_CPP_SERVER_PORT", std::to_string(fixture_ports(3)));
                env("LLAMA_CPP_START_TIMEOUT", "2"); env("GPU_LAYERS", "0"); env("LLM_SPECULATIVE_TYPE", "none");
            }
            env("ROUTER_URL", router_url); env("ROUTER_API_KEY", "register-test-key"); env("WORKER_LABEL", name);
            env("EZLOCALAI_VERSION", "ezlocalai-router-contract-test");
            env("WORKER_HEARTBEAT_INTERVAL", "2"); env("WORKER_TUNNEL", tunnel ? "true" : "false");
            env("QUANT_TYPE", "Q4_K_XL");
            env("LLM_MAX_TOKENS", "4096"); env("N_PARALLEL", "2");
            for (auto flag : {"TTS_ENABLED", "STT_ENABLED", "EMBEDDING_ENABLED", "IMAGE_ENABLED", "VIDEO_ENABLED", "MUSIC_ENABLED"}) env(flag, "false");
            env("TTS_ENABLED", "true"); env("STT_ENABLED", "true");
            env("VOICE_SERVER", ""); env("LAZY_LOAD_VOICE", "true");
            env("TTS_N_PARALLEL", "3"); env("STT_N_PARALLEL", "4");
            env("TTS_UNLOAD_LLM_DURING_GENERATION", "true");
            env("STT_UNLOAD_LLM_DURING_GENERATION", "true");
            env("MUSIC_UNLOAD_LLM_DURING_GENERATION", "false");
            env("IMAGE_UNLOAD_LLM_DURING_GENERATION", "false");
            env("VIDEO_UNLOAD_LLM_DURING_GENERATION", "false");
            Child worker; worker.start(worker_binary, temporary / (std::string(name) + ".log"));
            Json current;
            const auto visible = [&](bool present) {
                auto workers = Json::parse(request(router_url, "/v1/router/workers")).at("data");
                for (const auto &entry : workers) if (entry.at("label") == name) { current = entry; return present; }
                return !present;
            };
            until([&] { return visible(true); }, "worker registered");
            const auto id = current.at("worker_id");
            require(current.at("models") == (swap ? Json::array({"integration-model", "integration-other"}) : Json::array({"integration-model"})), "advertised models");
            require(current.at("capabilities").size() == (swap ? 4 : 3), "advertised text, speech and loaded explicit-projector capabilities");
            require(request(router_url, "/dashboard").find("ezlocalai-router-contract-test") != std::string::npos, "Rust dashboard displays native build version");
            const auto speech_metadata = [&] {
                require(current.at("version") == "ezlocalai-router-contract-test", "Rust registration/heartbeat retains native version");
                for(const auto &model:current.at("extra").at("loaded_llms")) require(current.at("model_quant").at(model.get<std::string>()) == "Q6_K", "Rust sees loaded file quantization instead of configured fallback");
                require(current.at("cap_slots").at("tts").at("capacity") == 1, "Rust router sees shared TTS capacity");
                require(current.at("cap_slots").at("stt").at("capacity") == 1, "Rust router sees shared STT capacity");
                require(current.at("extra").at("llm_unload_dependent_capabilities") == Json::array({"tts", "stt"}), "Rust router retains speech dependencies");
                require(current.at("slot_total_capacity") == (swap ? 1 : 2), "Rust router counts shared text/speech capacity once");
                require(current.at("extra").at("llm_model_residency") == (swap ? "swap" : "resident"), "Rust router receives backend residency mode");
                if (swap) {
                    const auto &capabilities = current.at("capabilities");
                    require(std::find(capabilities.begin(), capabilities.end(), "vision") != capabilities.end(),
                        "owned engine's explicit projector advertises vision to Rust");
                    std::ifstream arguments(temporary / name / ("args-" + std::string(std::getenv("LLAMA_CPP_SERVER_PORT"))));
                    const std::string launch_arguments(std::istreambuf_iterator<char>(arguments), {});
                    require(launch_arguments.find("--mmproj\n" + std::string(std::getenv("LLAMA_CPP_MMPROJ_PATH")) + "\n") != std::string::npos,
                        "reported projector is passed to the owned engine");
                    require(current.at("extra").at("loaded_llms") == Json::array({"integration-model"}) && current.at("extra").at("active_llm") == "integration-model", "router sees only resident engine");
                    require(current.at("queue_capacity") == 1 && current.at("cap_slots").at("text").at("capacity") == 1, "router sees exclusive swap capacity");
                } else require(current.at("extra").at("loaded_llms") == Json::array() && current.at("extra").at("active_llm").is_null(), "echo backend does not advertise resident model engines");
            };
            speech_metadata();
            auto models = Json::parse(request(router_url, "/v1/models")).at("data");
            bool found = false; for (const auto &m : models) if (m.at("id") == "integration-model") found = true;
            require(found, "router model discovery");
            Json chat{{"model", "integration-model"}, {"messages", Json::array({{{"role", "user"}, {"content", "Hello café 日本語"}}})},
                {"max_tokens", 16}, {"prompt_cache_key", "integration-conversation"}};
            auto value = Json::parse(request(router_url, "/v1/chat/completions", chat));
            require(!value.at("choices").empty(), "buffered chat");
            chat["stream"] = true; auto stream = request(router_url, "/v1/chat/completions", chat);
            require(stream.find("data:") != std::string::npos && stream.find("[DONE]") != std::string::npos, "streamed chat");
            if (swap) {
                std::ifstream seen(temporary / name / ("seen-" + std::string(std::getenv("LLAMA_CPP_SERVER_PORT"))));
                const std::string requests(std::istreambuf_iterator<char>(seen), {});
                const std::string cache_hint = "\"prompt_cache_key\":\"integration-conversation\"";
                const auto first = requests.find(cache_hint);
                require(first != std::string::npos && requests.find(cache_hint, first + cache_hint.size()) != std::string::npos,
                    "conversation cache hint reaches owned engine through buffered and streamed Rust routing");
                auto cancelled = chat;
                cancelled["messages"] = Json::array({{{"role", "user"}, {"content", "fixture-stream-cancel"}}});
                cancelled["disable_fallback"] = true;
                const std::string engine_port = std::getenv("LLAMA_CPP_SERVER_PORT");
                const auto launches = [&] {
                    std::ifstream file(temporary / name / ("launch-" + engine_port));
                    require(bool(file), "owned engine launch record");
                    return std::string(std::istreambuf_iterator<char>(file), {});
                };
                const auto initial_launches = launches();
                cancel_chat_after_first_chunk(std::stoi(router_port), cancelled, [&] {
                    require(visible(true) && current.at("router_in_flight").get<int>() > 0,
                        "Rust reserves the active inference stream");
                    const auto resources = Json::parse(request("http://127.0.0.1:" + worker_port,
                        "/v1/resources", nullptr, "local-test-key"));
                    require(resources.at("slots").at("slot_total_in_flight").get<int>() > 0,
                        "worker reserves the active inference stream");
                });
                const auto marker = temporary / name / ("cancelled-stream-" + engine_port);
                until([&] { return fs::exists(marker); }, "client cancellation reaches the owned inference engine");
                until([&] {
                    if (!visible(true)) return false;
                    const auto resources = Json::parse(request("http://127.0.0.1:" + worker_port,
                        "/v1/resources", nullptr, "local-test-key"));
                    return current.at("router_in_flight") == 0 && current.at("active_requests") == 0 &&
                        current.at("in_flight") == 0 && resources.at("slots").at("slot_total_in_flight") == 0;
                }, "cancelled stream releases router and worker admission");
                auto recovered = chat; recovered["stream"] = false; recovered["disable_fallback"] = true;
                value = Json::parse(request(router_url, "/v1/chat/completions", recovered));
                require(!value.at("choices").empty(), "inference succeeds after streaming cancellation");
                require(launches() == initial_launches, "cancellation does not restart the owned engine");
            }
            if (!swap) {
                auto large = chat;
                large["stream"] = false;
                // Router-to-worker JSON/base64 exceeds the old 16 MiB frame limit.
                // Keep the prompt small so the echo response stays small as well.
                large["transport_test_padding"] = std::string(13 * 1024 * 1024, 'p');
                value = Json::parse(request(router_url, "/v1/chat/completions", large));
                require(!value.at("choices").empty(), "large request through actual Rust router");
            }
            router.stop(); start_router();
            until([&] { return visible(true); }, "worker re-registers after router restart");
            require(current.at("worker_id") == id, "stable process identity");
            speech_metadata();
            value = Json::parse(request(router_url, "/v1/completions", {{"model", "integration-model"}, {"prompt", "ping"}, {"max_tokens", 8}}));
            require(!value.at("choices").empty(), "completion after restart");
            if (swap) {
                value = Json::parse(request(router_url, "/v1/completions", {{"model", "integration-other"}, {"prompt", "switch"}, {"max_tokens", 8}}));
                require(value.at("model") == "integration-other", "router switches model through native worker");
                until([&] { return visible(true) && current.at("extra").at("loaded_llms") == Json::array({"integration-other"}) && current.at("extra").at("active_llm") == "integration-other"; }, "heartbeat follows model switch");
            }
            worker.stop(); until([&] { return visible(false); }, "worker deregisters on shutdown");
            std::cout << name << ": registration/auth, model discovery, JSON/SSE, restart and shutdown passed\n";
        }
        env("LLAMA_CPP_MMPROJ_PATH", "");
        for (const bool runtime_override : {false, true}) for (const bool tunnel : {false, true}) {
            const std::string name = std::string(runtime_override ? "runtime-slots" : "preload-failure") + (tunnel ? "-tunnel" : "-direct");
            const std::string model = runtime_override ? "integration-model" : "integration-model-mtp";
            const auto worker_port = std::to_string(fixture_ports());
            const auto engine_port = fixture_ports(4);
            const auto directory = temporary / name; fs::create_directories(directory);
            if (!runtime_override) std::ofstream(directory / ("fail-launch-" + std::to_string(engine_port + 1))) << "fail";
            env("PORT", worker_port); env("EZLOCALAI_URL", "http://127.0.0.1:" + worker_port);
            env("EZLOCALAI_NATIVE_BACKEND", "llama.cpp"); env("EZLOCALAI_PRELOAD_MODEL_ON_START", "true");
            env("EZLOCALAI_API_KEY", "local-test-key"); env("DEFAULT_MODEL", model);
            env("LLM_MODEL_RESIDENCY", "resident"); env("LLM_SPECULATIVE_TYPE", runtime_override ? "none" : "mtp"); env("N_PARALLEL", runtime_override ? "4" : "3");
            env("LLAMA_CPP_EXTRA_ARGS", runtime_override ? "-np 2" : "");
            env("EMBEDDING_FIXTURE_DIR", directory.string()); env("LLAMA_CPP_SERVER_PORT", std::to_string(engine_port));
            env("ROUTER_URL", router_url); env("ROUTER_API_KEY", "register-test-key"); env("WORKER_LABEL", name);
            env("WORKER_TUNNEL", tunnel ? "true" : "false"); env("TTS_ENABLED", "false"); env("STT_ENABLED", "false");
            Child worker; worker.start(worker_binary, temporary / (name + ".log"));
            Json current;
            const auto visible = [&](bool present) {
                const auto workers = Json::parse(request(router_url, "/v1/router/workers")).at("data");
                for (const auto &entry : workers) if (entry.at("label") == name) { current = entry; return present; }
                return !present;
            };
            const auto check_capacity = [&] {
                require(current.at("queue_capacity") == 2 && current.at("slot_total_capacity") == 2, "failed replica reduces Rust queue/total capacity");
                require(current.at("cap_slots").at("text").at("capacity") == 2, "failed replica reduces text capacity");
                require(current.at("model_slots").at(model).at("capacity") == 2, "failed replica reduces public model capacity");
                require(current.at("extra").at("loaded_llms") == (runtime_override ? Json::array({model}) : Json::array({model, model})), "loaded engine identities");
                if (runtime_override) {
                    require(current.at("model_context").at(model) == 2048, "router per-slot context uses resolved parallel count");
                    require(!current.at("model_slots").at(model).contains("instances"), "two slots in one engine do not advertise multiple instances");
                } else {
                    require(current.at("model_slots").at(model).at("instances") == 2, "Rust router receives two surviving resident instances");
                    require(current.at("cap_slots").at("text").at("instances") == 2, "single-model text slots inherit resident instance count");
                }
            };
            until([&] { return visible(true); }, "worker starts despite failed automatic replica"); check_capacity();
            const auto value = Json::parse(request(router_url, "/v1/completions", {{"model", model}, {"prompt", "test"}, {"max_tokens", 8}}));
            require(!value.at("choices").empty(), "router serves surviving replica");
            if (!runtime_override) {
                for (int i : {0, 2}) fs::remove(directory / ("seen-" + std::to_string(engine_port + i)));
                std::future<std::string> first, second;
                struct Release { fs::path directory; int port; ~Release() { for (int i : {0, 2}) std::ofstream(directory / ("release-" + std::to_string(port + i))) << "release"; } } release{directory, engine_port};
                const auto invoke = [&] { return request(router_url, "/v1/completions", {{"model", model}, {"prompt", "block"}, {"max_tokens", 8}}); };
                first = std::async(std::launch::async, invoke); second = std::async(std::launch::async, invoke);
                until([&] { return fs::exists(directory / ("seen-" + std::to_string(engine_port))) && fs::exists(directory / ("seen-" + std::to_string(engine_port + 2))); }, "Rust routes concurrent requests to both surviving instances");
                until([&] { return visible(true) && current.at("router_in_flight") == 2; }, "Rust retains both concurrent reservations");
                for (int i : {0, 2}) std::ofstream(directory / ("release-" + std::to_string(engine_port + i))) << "release";
                require(!Json::parse(first.get()).at("choices").empty() && !Json::parse(second.get()).at("choices").empty(), "both routed concurrent requests complete");
            }
            router.stop(); start_router();
            until([&] { return visible(true); }, "partially preloaded worker re-registers"); check_capacity();
            worker.stop(); until([&] { return visible(false); }, "partially preloaded worker deregisters");
            std::cout << name << ": continued startup, surviving capacity, routed inference and restart passed\n";
        }
        for (const bool tunnel : {false, true}) {
            const std::string name = tunnel ? "embedding-report-tunnel" : "embedding-report-direct";
            const auto worker_port = std::to_string(fixture_ports());
            env("PORT", worker_port); env("EZLOCALAI_URL", "http://127.0.0.1:" + worker_port);
            env("EZLOCALAI_NATIVE_BACKEND", "echo"); env("DEFAULT_MODEL", "none");
            env("EZLOCALAI_PRELOAD_MODEL_ON_START", "false"); env("LLM_MODEL_RESIDENCY", "resident");
            env("EZLOCALAI_API_KEY", "local-test-key"); env("WORKER_LABEL", name);
            env("ROUTER_URL", router_url); env("ROUTER_API_KEY", "register-test-key");
            env("WORKER_TUNNEL", tunnel ? "true" : "false");
            env("EMBEDDING_ENABLED", "true"); env("EMBEDDING_SERVER", "");
            env("EMBEDDING_MODEL", "org/embedding-model"); env("EMBEDDING_MODEL_ALIAS", "vectors");
            env("EMBEDDING_QUANT_TYPE", "Q8_0"); env("EMBEDDING_CONTEXT_LENGTH", "8192");
            Child worker; worker.start(worker_binary, temporary / (name + ".log")); Json current;
            const auto visible = [&] {
                const auto inventory = Json::parse(request(router_url, "/v1/router/workers"));
                for (const auto &entry : inventory.at("data"))
                    if (entry.at("label") == name) { current = entry; return true; }
                return false;
            };
            until(visible, "embedding worker registered");
            const auto verify = [&] {
                require(current.at("capabilities") == Json::array({"embedding"}), "embedding-only capabilities");
                require(current.at("cap_models").at("embedding") == "vectors", "embedding alias for routing/dashboard");
                for (const auto &model : {"org/embedding-model", "vectors"}) {
                    require(current.at("model_quant").at(model) == "Q8_0", "embedding model/alias quantization reaches Rust");
                    require(current.at("model_context").at(model) == 8192, "embedding model/alias context reaches Rust");
                }
            };
            verify();
            const auto html = request(router_url, "/dashboard");
            require(html.find("vectors") != std::string::npos && html.find("Q8_0") != std::string::npos, "Rust dashboard displays embedding alias and quantization");
            router.stop(); start_router(); until(visible, "embedding metadata re-registers after router restart"); verify();
            worker.stop(); until([&] { return !visible(); }, "embedding worker deregisters");
            env("EMBEDDING_ENABLED", "false");
            std::cout << name << ": alias, context, quantization, dashboard and restart passed\n";
        }
        heartbeat_release_cases(worker_binary, router_url, temporary);
        router.stop(); fs::current_path(cwd); fs::remove_all(temporary); return 0;
    } catch (const std::exception &e) {
        fs::current_path(cwd);
        std::cerr << e.what() << "\nLogs retained in " << temporary << '\n'; return 1;
    }
}
