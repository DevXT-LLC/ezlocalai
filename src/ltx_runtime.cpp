#include "ezlocalai/ltx_runtime.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/hub_snapshot.hpp"
#include "ezlocalai/image_processing.hpp"
#include "ezlocalai/process.hpp"
#include "ezlocalai/video_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <openssl/rand.h>
#include <set>

namespace ezlocalai {
std::string python_float_string(double value);
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
std::string env(const char *key, const char *fallback = "") { const char *p = std::getenv(key); return p ? p : fallback; }
std::string lower(std::string s) { for (auto &c : s) if (c >= 'A' && c <= 'Z') c += 'a' - 'A'; return s; }
void write(const fs::path &path, const std::string &bytes) {
    std::ofstream out(path, std::ios::binary); out.write(bytes.data(), bytes.size());
    if (!out) throw std::runtime_error("Cannot write LTX input");
}
std::string tail(const fs::path &path) {
    std::ifstream in(path, std::ios::binary); in.seekg(0, std::ios::end);
    const auto length = in.tellg(); if (length < 0) return "";
    in.seekg(std::max<std::streamoff>(0, static_cast<std::streamoff>(length) - 8192));
    return {std::istreambuf_iterator<char>(in), {}};
}
struct WorkDirectory {
    fs::path path;
    WorkDirectory() {
        auto name = (fs::temp_directory_path() / "ltx-video-XXXXXX").string();
        if (!mkdtemp(name.data())) throw std::system_error(errno, std::generic_category(), "LTX temporary directory");
        path = name;
    }
    ~WorkDirectory() { std::error_code ignored; fs::remove_all(path, ignored); }
};
std::string uuid() {
    unsigned char bytes[16]; if (RAND_bytes(bytes, sizeof(bytes)) != 1) throw std::runtime_error("LTX output identifier");
    bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128;
    const char *hex = "0123456789abcdef"; std::string out;
    for (int i = 0; i < 16; ++i) { if (i == 4 || i == 6 || i == 8 || i == 10) out += '-'; out += hex[bytes[i] >> 4]; out += hex[bytes[i] & 15]; }
    return out;
}
void validate_component(const fs::path &root, const std::string &component) {
    const auto directory = root / component;
    // Same preference order as the pinned engine adapter. Checking every
    // indexed shard prevents an incomplete offline snapshot reaching ggml.
    for (const auto *name : {"model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json",
                             "model.safetensors", "diffusion_pytorch_model.safetensors"}) {
        const auto path = directory / name;
        if (!fs::is_regular_file(path)) continue;
        if (path.extension() == ".json") {
            std::ifstream in(path); const auto index = Json::parse(in);
            const auto &weights = index.at("weight_map");
            if (!weights.is_object() || weights.empty()) throw std::runtime_error("Empty LTX component index: " + component);
            std::set<std::string> shards;
            for (const auto &weight : weights.items()) shards.insert(weight.value().get<std::string>());
            for (const auto &shard : shards) {
                const fs::path relative(shard);
                if (relative.empty() || relative.is_absolute()) throw std::runtime_error("Invalid LTX shard path");
                for (const auto &part : relative) if (part == "..") throw std::runtime_error("Invalid LTX shard path");
                if (!fs::is_regular_file(directory / relative)) throw std::runtime_error("Missing LTX shard: " + component + "/" + shard);
            }
        }
        return;
    }
    throw std::runtime_error("Missing LTX component: " + component);
}
}
bool is_ltx_model(const std::string &model) { return lower(model).find("ltx") != std::string::npos; }
LtxAssets prepare_ltx_assets(const std::string &model) {
    HubSnapshotOptions options; options.cache_directory = env("HF_HUB_CACHE", "models");
    LtxAssets assets;
    assets.transformer = hub_snapshot_file(model, "ltx-2.3-22b-dev-Q4_K_M.gguf", options);
    const auto repository = env("LTX2_CONFIG_REPO", "diffusers/LTX-2.3-Diffusers");
    options.allow_patterns = {"text_encoder/*.safetensors*", "vae/*.safetensors*", "audio_vae/*.safetensors*",
        "vocoder/*.safetensors*", "connectors/*.safetensors*", "tokenizer/tokenizer.json"};
    assets.components = fs::is_directory(repository) ? repository : hub_snapshot_download(repository, options);
    for (const auto *component : {"text_encoder", "vae", "audio_vae", "vocoder", "connectors"}) validate_component(assets.components, component);
    assets.tokenizer = (fs::path(assets.components) / "tokenizer/tokenizer.json").string();
    if (!fs::is_regular_file(assets.tokenizer)) throw std::runtime_error("Missing LTX tokenizer");
    return assets;
}
std::vector<std::string> ltx_video_command(const std::string &binary, const std::string &device,
    const LtxAssets &assets, const std::string &output, const std::string &prompt,
    const LtxVideoSettings &s, const std::string &residency, const std::string &image_path,
    const std::string &conditions_json, const LtxTextEncoderSettings &text_encoder) {
    if (text_encoder.weight_type != "bf16" && text_encoder.weight_type != "f16")
        throw std::invalid_argument("Unsupported LTX text encoder precision");
    // Limit this memory mode to the child; changing process-global environment
    // during concurrent requests would affect unrelated media engines.
    std::vector<std::string> args{"/usr/bin/env", "GGML_CUDA_NO_PINNED=1", binary, "-M", "vid_gen",
        "--diffusion-model", assets.transformer, "--ltx-diffusers-root", assets.components,
        "--tokenizer", assets.tokenizer, "--mmap", "--tensor-type-rules", "^text_encoders.llm.*=" + text_encoder.weight_type,
        "--diffusion-fa", "--vae-tiling", "--temporal-tiling", "--sampling-method", "euler",
        "--extra-sample-args", std::string(ltx_reference_scheduler_args),
        "--rng", "cpu", "--seed", "42", "-p", prompt, "-n",
        "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, blurry, shaky",
        "-W", std::to_string(s.width), "-H", std::to_string(s.height), "--video-frames", std::to_string(s.frames),
        "--fps", std::to_string(s.fps), "--steps", std::to_string(s.steps), "--cfg-scale", python_float_string(s.guidance),
        "--output-begin-idx", "0", "-o", output, "--backend"};
    if (device == "cpu") args.push_back("cpu");
    else {
        auto policy = ltx_device_policy(device, true, 8, 0);
        args.push_back("cuda" + std::to_string(policy.gpu_index));
    }
    if (text_encoder.nf4_sm_count || text_encoder.nf4_threads_per_sm) {
        if (device == "cpu" || text_encoder.nf4_sm_count <= 0 || text_encoder.nf4_threads_per_sm < 512)
            throw std::invalid_argument("Invalid LTX NF4 CUDA geometry");
        args.insert(args.end(), {"--ltx-nf4-sm-count", std::to_string(text_encoder.nf4_sm_count),
            "--ltx-nf4-threads-per-sm", std::to_string(text_encoder.nf4_threads_per_sm)});
    }
    // Quantized Gemma retains its own GPU parameters independently of this
    // policy for the remaining modules. Sequential execution disables prefetch.
    if (device != "cpu" && residency != "full") args.push_back("--offload-to-cpu");
    if (device != "cpu") args.insert(args.end(), {"--ltx-offload-mode", residency == "sequential" ? "2" : "1"});
    if (!conditions_json.empty()) args.insert(args.end(), {"--ltx-conditions", conditions_json});
    else if (!image_path.empty()) args.insert(args.end(), {"-i", image_path, "--strength", "1"});
    for (const auto &arg : args) if (arg.find('\0') != std::string::npos) throw std::invalid_argument("embedded null byte");
    return args;
}
LtxVideo::LtxVideo(std::string model, std::string device, std::optional<LtxVideoSettings> hint)
    : model_(std::move(model)), device_(std::move(device)),
      binary_(env("SDCPP_BIN", "/opt/stable-diffusion.cpp/build/bin/sd-cli")),
      engine_({"/usr/bin/env", "GGML_CUDA_NO_PINNED=1", binary_}) {
    if (!is_ltx_model(model_)) throw std::invalid_argument("Unsupported LTX model: " + model_);
    if (!fs::is_regular_file(binary_)) throw std::runtime_error("LTX requires the patched stable-diffusion.cpp sd-cli binary");
    if (lower(strip_config_value(env("VIDEO_CONNECTORS_SOURCE", "diffusers"))) == "gguf")
        throw std::runtime_error("LTX GGUF connector reconstruction is not implemented");
    assets_ = prepare_ltx_assets(model_);
    auto &observation = placement_observation_;
    if (hint) {
        observation.frames = hint->frames;
        const auto size = lower(hint->size); const auto separator = size.find('x');
        if (separator != std::string::npos) try {
            const auto width = parse_configuration_integer(size.substr(0, separator));
            const auto height = parse_configuration_integer(size.substr(separator + 1));
            if (width && height) { observation.width = std::stoll(*width); observation.height = std::stoll(*height); }
        } catch (const std::exception &) { observation.width.reset(); observation.height.reset(); }
    }
    auto &text_encoder = text_encoder_;
    if (device_ != "cpu") try {
        const auto ordinal = ltx_device_policy(device_, true, 8, 0).gpu_index;
        const auto properties = cuda_device_properties(static_cast<int>(ordinal));
        text_encoder.weight_type = properties.major >= 8 ? "bf16" : "f16";
        const auto memory = cuda_memory_info(static_cast<int>(ordinal));
        observation.free_gb = memory.free_bytes / double(1ULL << 30); observation.total_gb = memory.total_bytes / double(1ULL << 30);
        if (ltx_device_policy(device_, true, properties.major, observation.free_gb).try_bnb) {
            const auto geometry = cuda_multiprocessor_geometry(static_cast<int>(ordinal));
            text_encoder.nf4_sm_count = geometry.first;
            text_encoder.nf4_threads_per_sm = geometry.second;
        }
    } catch (const std::exception &) {}
    // Load Gemma first, with the remaining parameters on CPU. Placement is
    // selected below from the GPU memory left after that load, as in Python.
    residency_ = device_ == "cpu" ? "cpu" : "sequential";
    warm();
}
FileProcessResult LtxVideo::execute(const std::vector<std::string> &arguments, bool preload,
                                  const std::string &log, double timeout) {
    // ltx_video_command includes env and its assignment before the executable.
    const std::vector<std::string> argv(arguments.begin() + 2, arguments.end());
    FileProcessResult result;
    try { result = engine_.request(argv, preload, log, timeout); }
    catch (const std::exception &error) {
        engine_.stop();
        std::cerr << "[VIDEO] LTX engine transport failed: " << error.what() << '\n';
        loaded_ = false;
        return {1, false};
    }
    loaded_ = !result.status && !result.timed_out;
    return result;
}
void LtxVideo::warm() {
    if (loaded_) return;
    double timeout;
    if (!parse_configuration_float(env("VIDEO_GENERATION_TIMEOUT", "3600"), &timeout) || !std::isfinite(timeout))
        throw std::invalid_argument("Invalid VIDEO_GENERATION_TIMEOUT");
    for (;;) {
        WorkDirectory work;
        LtxVideoSettings settings; settings.size = "64x64"; settings.width = settings.height = 64;
        settings.frames = 9; settings.steps = 2;
        const auto log = (work.path / "preload.log").string();
        const auto result = execute(ltx_video_command(binary_, device_, assets_, (work.path / "frame_%06d.png").string(),
            "preload", settings, residency_, "", "", text_encoder_), true, log, std::max(0.0, timeout));
        if (result.timed_out) throw std::runtime_error("LTX preload timed out");
        if (!result.status) {
            if (!placement_selected_ && device_ != "cpu") {
                auto observation = placement_observation_;
                observation.free_gb = observation.total_gb = 0;
                observation.text_encoder_on_gpu = text_encoder_.nf4_sm_count > 0;
                try {
                    const auto ordinal = ltx_device_policy(device_, true, 8, 0).gpu_index;
                    const auto memory = cuda_memory_info(static_cast<int>(ordinal));
                    observation.free_gb = memory.free_bytes / double(1ULL << 30);
                    observation.total_gb = memory.total_bytes / double(1ULL << 30);
                } catch (const std::exception &) {}
                const auto selected = choose_video_gpu_residency(ltx_residency_policy_from_environment(observation));
                placement_selected_ = true;
                if (selected != residency_) { residency_ = selected; continue; }
            }
            return;
        }
        const auto diagnostic = tail(log);
        if (text_encoder_.nf4_sm_count && diagnostic.find("LTX NF4 conversion failed") != std::string::npos) {
            text_encoder_.nf4_sm_count = text_encoder_.nf4_threads_per_sm = 0;
            std::cerr << "[VIDEO] NF4 preload failed; retrying unquantized text weights\n";
            continue;
        }
        const auto message = lower(diagnostic);
        const bool cuda_failure = message.find("out of memory") != std::string::npos
            || message.find("model manager cannot make enough memory available") != std::string::npos
            || message.find("cuda error") != std::string::npos || message.find("cuda failure") != std::string::npos;
        if (device_ != "cpu" && cuda_failure && residency_ == "full") { residency_ = "model_offload"; continue; }
        if (device_ != "cpu" && cuda_failure && residency_ == "model_offload") { residency_ = "sequential"; continue; }
        throw std::runtime_error("LTX preload failed: " + diagnostic);
    }
}
std::string LtxVideo::generate(const std::string &prompt, LtxVideoSettings settings,
    const std::string &local_uri, const std::string &image_bytes,
    const std::vector<LtxCondition> &conditions, bool include_audio) {
    settings = normalize_ltx_settings(settings);
    if (settings.width <= 0 || settings.height <= 0 || settings.frames <= 0 || settings.fps <= 0 || !std::isfinite(settings.guidance))
        throw std::invalid_argument("Invalid LTX dimensions, frame count or generation settings");
    // The alternative GGUF connector reconstruction is not implemented yet;
    // do not silently run different weights for an explicitly selected source.
    if (lower(strip_config_value(env("VIDEO_CONNECTORS_SOURCE", "diffusers"))) == "gguf")
        throw std::runtime_error("LTX GGUF connector reconstruction is not implemented");
    const auto &assets = assets_;
    auto &text_encoder = text_encoder_;
    auto &residency = residency_;
    double timeout;
    if (!parse_configuration_float(env("VIDEO_GENERATION_TIMEOUT", "3600"), &timeout) || !std::isfinite(timeout)) throw std::invalid_argument("Invalid VIDEO_GENERATION_TIMEOUT");
    fs::create_directories("outputs"); const auto output = fs::path("outputs") / (uuid() + ".mp4");
    bool reduced = false;
    try {
        for (;;) {
            WorkDirectory work;
            std::string image_path, conditions_json;
            if (!reduced && !conditions.empty()) {
                auto array = Json::array();
                for (size_t i = 0; i < conditions.size(); ++i) {
                    const auto &condition = conditions[i];
                    if (!std::isfinite(condition.strength)) throw std::invalid_argument("LTX condition strength must be finite");
                    const auto path = work.path / ("condition-" + std::to_string(i) + ".png");
                    write(path, encode_rgb_png(decode_rgb_image(condition.image_bytes)));
                    array.push_back({{"image", path.string()}, {"index", condition.index}, {"strength", condition.strength}});
                }
                conditions_json = array.dump();
            } else if (!reduced && !image_bytes.empty()) {
                image_path = (work.path / "image.png").string();
                write(image_path, encode_rgb_png(resize_rgb_bilinear(decode_rgb_image(image_bytes), settings.width, settings.height)));
            }
            const auto pattern = (work.path / "frame_%06d.png").string(), log = (work.path / "generation.log").string();
            const auto result = execute(ltx_video_command(binary_, device_, assets, pattern, prompt, settings, residency, image_path, conditions_json, text_encoder), false, log, std::max(0.0, timeout));
            if (result.timed_out) throw std::runtime_error("LTX generation timed out");
            if (result.status) {
                const auto diagnostic = tail(log); std::cerr << "[VIDEO] LTX exit " << result.status << ": " << diagnostic << '\n';
                if (text_encoder.nf4_sm_count && diagnostic.find("LTX NF4 conversion failed") != std::string::npos) {
                    // Restart in a fresh child so partially converted weights
                    // cannot survive a quantizer failure. Quanto is still absent.
                    text_encoder.nf4_sm_count = text_encoder.nf4_threads_per_sm = 0;
                    std::cerr << "[VIDEO] NF4 conversion failed; retrying unquantized text weights\n";
                    continue;
                }
                // Startup banners contain CUDA even for unrelated failures. Only
                // error diagnostics should activate the CUDA recovery policy.
                const auto message = lower(diagnostic);
                const bool cuda_failure = message.find("out of memory") != std::string::npos
                    || message.find("model manager cannot make enough memory available") != std::string::npos
                    || message.find("cuda error") != std::string::npos || message.find("cuda failure") != std::string::npos;
                const auto action = cuda_failure ? ltx_failure_action(true, diagnostic, residency,
                    env("VIDEO_RETRY_SEQUENTIAL_ON_OOM", "true")) : LtxFailureAction::propagate;
                if (!reduced && device_ != "cpu" && action == LtxFailureAction::reload_sequential) { residency = "sequential"; continue; }
                if (!reduced && device_ != "cpu" && action == LtxFailureAction::reduce_on_gpu) { settings = reduced_ltx_settings(settings); reduced = true; continue; }
                return "";
            }
            for (int64_t i = 0; i < settings.frames; ++i) {
                auto number = std::to_string(i); number.insert(0, number.size() < 6 ? 6 - number.size() : 0, '0');
                if (!fs::is_regular_file(work.path / ("frame_" + number + ".png"))) throw std::runtime_error("LTX did not produce the requested frames");
            }
            const auto audio = work.path / "frame_%06d.wav";
            auto encode = [&](bool with_audio) {
                std::vector<std::string> command{"ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", std::to_string(settings.fps), "-start_number", "0", "-i", pattern};
                if (with_audio) command.insert(command.end(), {"-i", audio.string(), "-ar", "24000", "-c:a", "aac"});
                command.insert(command.end(), {"-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", output.string()});
                return run_process_to_files(command, "/dev/null", (work.path / "encoding.log").string(), 180);
            };
            // Retain the 24 kHz output format, but resample the native 48 kHz
            // waveform. Relabeling it doubles duration and halves pitch.
            const bool with_audio = include_audio && fs::is_regular_file(audio);
            auto encoded = encode(with_audio);
            if ((encoded.status || encoded.timed_out) && with_audio) encoded = encode(false);
            if (encoded.status || encoded.timed_out || !fs::is_regular_file(output) || fs::file_size(output) == 0)
                throw std::runtime_error("LTX video encoding failed");
            auto base = local_uri; while (!base.empty() && base.back() == '/') base.pop_back();
            return base.empty() ? output.generic_string() : base + "/" + output.generic_string();
        }
    } catch (...) { std::error_code ignored; fs::remove(output, ignored); throw; }
}
}
