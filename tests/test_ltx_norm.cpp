// Compile against the patched engine's actual RMS block; no model weights.
#include "model/te/llm.hpp"
#include "ggml-cpu.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>

int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::runtime_error("fixture required");
        std::ifstream in(argv[1]);
        nlohmann::json fixture;
        in >> fixture;
        float worst = 0;
        for (const auto &row : fixture.at("cases")) {
            for (bool raw : {true, false}) {
                const int width = row.at("width"), scale = row.at("scale");
                const std::string mode = row.at("mode");
                std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
                    ggml_init({16 * 1024 * 1024, nullptr, false}), ggml_free);
                if (!context) throw std::runtime_error("ggml context");
                auto *ctx = context.get();
                LLM::LLMRMSNorm norm(width, 1e-6f, raw);
                norm.init(ctx);
                std::map<std::string, ggml_tensor *> params;
                norm.get_param_tensors(params);
                auto *weight = static_cast<float *>(params.at("weight")->data);
                for (int i = 0; i < width; ++i) {
                    weight[i] = (mode == "zero" ? 0.f : mode == "quarter" ? .25f : (i % 13 - 6) / 16.f)
                        + (raw ? 0.f : 1.f);
                }
                auto *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 2);
                auto *data = static_cast<float *>(x->data);
                for (int i = 0; i < width * 2; ++i) data[i] = (i % 37 - 18) / 8.f * scale;
                GGMLRunnerContext runner;
                runner.ggml_ctx = ctx;
                auto *result = norm.forward(&runner, x);
                auto *graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, result);
                if (ggml_graph_compute_with_ctx(ctx, graph, 1) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("ggml graph failed");
                const auto expected = row.at("expected").get<std::vector<float>>();
                if (expected.size() != static_cast<size_t>(width * 2))
                    throw std::runtime_error("fixture shape mismatch");
                auto *actual = static_cast<float *>(result->data);
                for (size_t i = 0; i < expected.size(); ++i) {
                    const float error = std::abs(actual[i] - expected[i]);
                    worst = std::max(worst, error);
                    if (!std::isfinite(actual[i]) || error > fixture.at("absolute_tolerance").get<float>())
                        throw std::runtime_error("RMS unit-offset mismatch");
                }
            }
        }
        if (fixture.at("cases").size() != 36) throw std::runtime_error("all cases required");
        std::cout << "Verified 36 installed Gemma RMS cases for raw and folded weights (72 graphs); max error "
                  << worst << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
