#include "model/diffusion/ltxv.hpp"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <iostream>
#include <stdexcept>

static void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
static std::vector<float> read(ggml_tensor* tensor) {
    check(tensor != nullptr, "committed tensor exists");
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}
static void run(ggml_backend_t backend) {
    // The already-processed conditioning path needs no model weights, but uses
    // the real connector graph, allocator, device copies and RunnerCache.
    LTXV::LTXAVRunner runner(backend);
    runner.set_scale_overrides(1, 1);
    runner.config.cross_attention_dim = 4;
    runner.config.audio_cross_attention_dim = 6;
    runner.config.use_connector = runner.config.use_audio_connector = false;
    runner.model.config = runner.config;
    sd::Tensor<float> positive({10, 1024}), negative({10, 1024});
    for (size_t i = 0; i < positive.values().size(); ++i) {
        positive.values()[i] = float(i % 97) / 97.f;
        negative.values()[i] = -positive.values()[i];
    }
    const auto first = runner.prepare_connector_context(2, positive, true);
    check(first.has_value(), "prepare positive conditioning");
    auto* video = runner.get_cache_tensor_by_name(*first + ".video");
    auto* audio = runner.get_cache_tensor_by_name(*first + ".audio");
    const auto video_values = read(video), audio_values = read(audio);
    for (size_t t = 0; t < 1024; ++t) {
        for (size_t c = 0; c < 4; ++c) check(video_values[t * 4 + c] == positive.values()[t * 10 + c], "video slice preserves strided input");
        for (size_t c = 0; c < 6; ++c) check(audio_values[t * 6 + c] == positive.values()[t * 10 + 4 + c], "audio slice preserves strided input");
    }
    const auto second = runner.prepare_connector_context(2, negative, true);
    check(second && second != first, "negative conditioning gets independent storage");
    check(runner.prepare_connector_context(2, positive, true) == first &&
          runner.get_cache_tensor_by_name(*first + ".video") == video, "unchanged condition skips graph and allocation");
    positive.values()[0] = 9;
    const auto changed = runner.prepare_connector_context(2, positive, true);
    check(changed && read(runner.get_cache_tensor_by_name(*changed + ".video"))[0] == 9, "changed input recomputes device output");
    const auto video_only = runner.prepare_connector_context(2, positive, false);
    check(video_only && runner.connector_inputs.find(positive.shape(), positive.values(), false, 1, 1), "audio mode invalidates cached identity");
    runner.free_cache_ctx_and_buffer();
    const auto after_flush = runner.prepare_connector_context(2, positive, false);
    check(after_flush && read(runner.get_cache_tensor_by_name(*after_flush + ".video"))[0] == 9, "device cache flush cannot leave stale input identity active");
    GGMLRunner* base = &runner;
    base->runner_end();
    check(!runner.runner_started() && !runner.get_cache_tensor_by_name(*changed + ".video") &&
          !runner.connector_inputs.find(positive.shape(), positive.values(), true, 1, 1), "base-pointer session cleanup releases device and input caches");
    check(runner.prepare_connector_context(2, negative, true).has_value(), "next sampling session prepares afresh");
    base->runner_end();
}
int main(int argc, char**) try {
    auto backend = argc > 1 ? ggml_backend_cuda_init(0) : ggml_backend_cpu_init();
    check(backend != nullptr, "initialize backend");
    run(backend);
    ggml_backend_free(backend);
    std::cout << "Real LTX connector graph slicing, cache reuse, invalidation and session cleanup passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
