// Compare the engine's actual timestep graph against hooks on the installed
// Python transformer. No model weights or GPU are required.
#include "ltx_timesteps.hpp"
#include "ggml-cpu.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>

int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::runtime_error("fixture required");
        std::ifstream input(argv[1]);
        nlohmann::json fixture;
        input >> fixture;
        int cases = 0;
        for (const auto &row : fixture.at("cases")) {
            // ezlocalai never enables the optional opposite-modality mode.
            if (row.at("use_cross_timestep").get<bool>()) continue;
            const float sigma = row.at("sigma");
            const auto mask = row.at("denoise_mask").get<std::vector<float>>();
            for (bool python : {true, false}) {
                std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
                    ggml_init({1024 * 1024, nullptr, false}), ggml_free);
                if (!context) throw std::runtime_error("ggml context");
                auto *ctx = context.get();
                auto *video = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mask.size());
                auto *audio = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
                for (size_t i = 0; i < mask.size(); ++i)
                    static_cast<float *>(video->data)[i] = sigma * mask[i];
                *static_cast<float *>(audio->data) = sigma;
                const auto result = ltx_av_cross_timesteps(ctx, video, audio, 1000.f, python);
                auto *graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, result.first);
                ggml_build_forward_expand(graph, result.second);
                if (ggml_graph_compute_with_ctx(ctx, graph, 1) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("timestep graph failed");
                for (const auto &[name, tensor] : std::vector<std::pair<std::string, ggml_tensor *>>{
                        {"av_cross_attn_video_scale_shift", result.first},
                        {"av_cross_attn_video_a2v_gate", result.first},
                        {"av_cross_attn_audio_scale_shift", result.second},
                        {"av_cross_attn_audio_v2a_gate", result.second}}) {
                    auto expected = row.at("module_inputs").at(name).get<std::vector<float>>();
                    if (!python) std::fill(expected.begin(), expected.end(), sigma);
                    if (ggml_nelements(tensor) != static_cast<int64_t>(expected.size()))
                        throw std::runtime_error("timestep shape mismatch: " + name);
                    const auto *actual = static_cast<const float *>(tensor->data);
                    for (size_t i = 0; i < expected.size(); ++i)
                        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 1e-4f)
                            throw std::runtime_error("timestep mismatch: " + name);
                }
            }
            ++cases;
        }
        if (cases != 24) throw std::runtime_error("all worker cases required");
        std::cout << "24 installed transformer traces match all four cross-attention inputs; legacy behavior preserved\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
