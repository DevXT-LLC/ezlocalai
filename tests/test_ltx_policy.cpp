#include "ezlocalai/ltx_video.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <sstream>

namespace {
using Json = nlohmann::json;
void check(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::string read(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary); check(bool(input), "read " + path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
std::string sha256(const std::string &value) {
    unsigned char bytes[EVP_MAX_MD_SIZE]; unsigned count;
    check(EVP_Digest(value.data(), value.size(), bytes, &count, EVP_sha256(), nullptr) == 1, "SHA256");
    std::ostringstream result;
    for (unsigned i = 0; i < count; ++i) result << std::hex << std::setw(2) << std::setfill('0') << unsigned(bytes[i]);
    return result.str();
}
ezlocalai::LtxVideoSettings settings(const Json &request) {
    ezlocalai::LtxVideoSettings value;
    value.size = request.value("size", value.size);
    value.frames = request.value("num_frames", value.frames);
    value.fps = request.value("frame_rate", value.fps);
    value.steps = request.value("num_inference_steps", value.steps);
    value.guidance = request.value("guidance_scale", value.guidance);
    return value;
}
void compare(const ezlocalai::LtxVideoSettings &s, const Json &expected) {
    const Json actual = {{"width",s.width},{"height",s.height},{"num_frames",s.frames},
        {"frame_rate",s.fps},{"num_inference_steps",s.steps},{"guidance_scale",s.guidance}};
    for (const auto &[key,value] : actual.items()) check(value == expected.at(key), key + ": " + actual.dump());
}
}
int main(int argc, char **argv) {
    try {
        check(argc == 3, "reference root and fixture required");
        const auto fixture = Json::parse(read(argv[2]));
        for (const auto &[path,digest] : fixture.at("sources").items())
            check(sha256(read(std::filesystem::path(argv[1])/path)) == digest.get<std::string>(), "source changed: " + path);
        size_t count = 0;
        for (const auto &row : fixture.at("normalization")) {
            std::string error;
            ezlocalai::LtxVideoSettings result;
            try { result = ezlocalai::normalize_ltx_settings(settings(row.at("request"))); }
            catch (const std::invalid_argument &) { error = "ValueError"; }
            catch (const std::overflow_error &) { error = "OverflowError"; }
            check(error == (row.at("error").is_null() ? "" : row.at("error").get<std::string>()), "normalization error: " + row.dump());
            if (error.empty()) compare(result, row.at("trace").at(0));
            ++count;
        }
        for (const auto &row : fixture.at("modes")) {
            const auto &f = row.at("flags");
            check(ezlocalai::ltx_generation_mode(f[0],f[1],f[2],f[3]) == row["trace"][0]["mode"], "mode selection");
            ++count;
        }
        for (const auto &row : fixture.at("failures")) {
            const auto action = ezlocalai::ltx_failure_action(row.at("runtime_error"),row.at("message"),row.at("residency"),row.at("retry"));
            const std::string name = action == ezlocalai::LtxFailureAction::propagate ? "propagate" :
                action == ezlocalai::LtxFailureAction::reload_sequential ? "reload_sequential" : "reduce_on_gpu";
            check(name == row.at("action"), "failure policy: " + row.dump());
            if (name == "reduce_on_gpu") compare(ezlocalai::reduced_ltx_settings(ezlocalai::normalize_ltx_settings(settings(row.at("request")))),row["trace"][1]);
            ++count;
        }
        for (const auto &row : fixture.at("residency")) {
            const auto &input = row.at("input"); ezlocalai::VideoResidencyPolicy p;
            p.configured_mode = input.at("configured_mode"); p.total_gb = input.at("total_gb"); p.free_gb = input.at("free_gb");
            p.text_encoder_on_gpu = input.at("text_encoder_on_gpu");
            if (input.contains("num_frames") && !input.at("num_frames").is_null()) p.frames = input.at("num_frames").get<int64_t>();
            if (input.contains("width") && !input.at("width").is_null()) p.width = input.at("width").get<int64_t>();
            if (input.contains("height") && !input.at("height").is_null()) p.height = input.at("height").get<int64_t>();
            p.short_min_free_gb = input.value("short_model_offload_min_free_gb",12.);
            check(ezlocalai::choose_video_gpu_residency(p) == row.at("result"), "residency: " + row.dump());
            ++count;
        }
        for (const auto &row : fixture.at("devices")) {
            const auto result = ezlocalai::ltx_device_policy(row.at("device"),row.at("available"),row.at("major"),row.at("free_gb"));
            check(result.cuda == row.at("cuda") && result.gpu_index == row.at("gpu_index") &&
                result.dtype == row.at("dtype") && result.try_bnb == row.at("try_bnb"), "device policy: " + row.dump());
            ++count;
        }
        for (const auto &row : fixture.at("environment")) {
            for (const char *key : {"VIDEO_GPU_RESIDENCY","VIDEO_MODEL_OFFLOAD_MIN_FREE_GB","VIDEO_FULL_GPU_MIN_FREE_GB",
                "VIDEO_SHORT_MODEL_OFFLOAD_MIN_FREE_GB","VIDEO_MODEL_OFFLOAD_MAX_FRAMES","VIDEO_MODEL_OFFLOAD_MAX_PIXELS"}) unsetenv(key);
            for (const auto &[key,value] : row.at("environment").items()) setenv(key.c_str(),value.get<std::string>().c_str(),1);
            ezlocalai::VideoResidencyPolicy input; input.total_gb=24; input.free_gb=row.at("free_gb"); input.text_encoder_on_gpu=row.at("text_encoder_on_gpu");
            input.frames=121; input.width=512; input.height=512;
            check(ezlocalai::choose_video_gpu_residency(ezlocalai::ltx_residency_policy_from_environment(input)) == row.at("result"), "environment policy: " + row.dump());
            ++count;
        }
        check(count == 2006, "all LTX source policy cases");
        std::cout << "Verified " << count << " LTX settings, mode, residency, device and OOM policy cases\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
