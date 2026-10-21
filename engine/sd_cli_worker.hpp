#pragma once
// Included by the pinned sd-cli after its public API/resource-owner headers.
#include "ezlocalai/media_worker_protocol.hpp"
#include <csignal>
#include <fcntl.h>
#include <memory>
#include <set>
#include <sys/prctl.h>

namespace ezlocalai_sd_worker {
inline bool active = false, preload = false;
inline std::shared_ptr<sd_ctx_t> context;
inline std::vector<std::string> context_arguments;
inline std::shared_ptr<sd_ctx_t> acquire(sd_ctx_params_t parameters) {
    if (!active) return {new_sd_ctx(&parameters), free_sd_ctx};
    if (!context) {
        parameters.eager_load = true;
        context = {new_sd_ctx(&parameters), free_sd_ctx};
        if (context) std::cerr << "[media-worker] context ready\n";
    }
#ifdef EZLOCALAI_LTX_OFFLOAD_POLICY
    if (context && !sd_set_ltx_offload_mode(context.get(), parameters.ltx_offload_mode))
        throw std::invalid_argument("invalid LTX offload mode");
#endif
    return context;
}
inline void clear_callbacks() {
    sd_set_log_callback(nullptr, nullptr);
    sd_set_progress_callback(nullptr, nullptr);
    sd_set_preview_callback(nullptr, PREVIEW_NONE, 1, false, false, nullptr);
}
inline std::vector<std::string> model_arguments(const std::vector<std::string> &args) {
    // Everything except these generation-only options identifies the context.
    // Unknown future model options therefore cause a reload instead of unsafe reuse.
    static const std::set<std::string> generation{
        "-p", "--prompt", "-n", "--negative-prompt", "-W", "--width", "-H", "--height",
        "--video-frames", "--fps", "--steps", "--cfg-scale", "--sampling-method",
        "--scheduler", "--seed", "--output-begin-idx", "-o", "--output",
        "-i", "--init-img", "--strength", "--ltx-conditions", "-r", "--ref-image",
        "--extra-sample-args", "--ltx-offload-mode"};
    std::vector<std::string> result;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i && generation.count(args[i])) {
            if (++i >= args.size()) throw std::invalid_argument("missing generation option value");
        } else result.push_back(args[i]);
    }
    return result;
}
template<class Execute> int serve(int fd, pid_t expected_parent, Execute execute) {
    if (prctl(PR_SET_PDEATHSIG, SIGKILL) || getppid() != expected_parent) return 2;
    active = true;
    try {
        std::string message;
        while (ezlocalai::media_wire::receive(fd, message)) {
            int status = 1;
            try {
                const auto request = nlohmann::json::parse(message);
                if (request.at("version") != 1) throw std::invalid_argument("unsupported media protocol");
                auto args = request.at("args").template get<std::vector<std::string>>();
                if (args.empty() || args.size() > 4096) throw std::invalid_argument("invalid argument count");
                for (const auto &arg : args) if (arg.find('\0') != std::string::npos) throw std::invalid_argument("embedded null byte");
                const auto log = request.at("log").template get<std::string>();
                if (log.find('\0') != std::string::npos) throw std::invalid_argument("embedded null byte");
                const int output = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
                if (output < 0) throw std::runtime_error("cannot open media log");
                const bool redirected = dup2(output, STDOUT_FILENO) >= 0 && dup2(output, STDERR_FILENO) >= 0;
                close(output);
                if (!redirected) throw std::runtime_error("cannot redirect media log");
                const auto key = model_arguments(args);
                if (key != context_arguments) { context.reset(); context_arguments = key; }
                preload = request.at("preload").template get<bool>();
                std::vector<const char *> argv; for (const auto &arg : args) argv.push_back(arg.c_str()); argv.push_back(nullptr);
                status = execute(static_cast<int>(args.size()), argv.data());
                clear_callbacks();
                if (status) context.reset();
            } catch (const std::exception &error) {
                clear_callbacks(); context.reset();
                std::cerr << "media worker: " << error.what() << '\n';
            }
            std::cout.flush(); std::cerr.flush(); fflush(nullptr);
            ezlocalai::media_wire::send(fd, nlohmann::json{{"version", 1}, {"status", status}}.dump());
        }
    } catch (const std::exception &) { clear_callbacks(); context.reset(); return 2; }
    clear_callbacks(); context.reset(); return 0;
}
} // namespace ezlocalai_sd_worker
