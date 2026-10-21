// Persistent, private stdio worker around llama.cpp's tools/tts pipeline.
// Model-specific synthesis stays in upstream libmtmd; no public network listener.
#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "nlohmann/json.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using json = nlohmann::json;

static std::string encode64(const char * data, size_t size) {
    static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((size + 2) / 3) * 4);
    for (size_t i = 0; i < size; i += 3) {
        uint32_t bits = (uint32_t) (unsigned char) data[i] << 16;
        if (i + 1 < size) bits |= (uint32_t) (unsigned char) data[i + 1] << 8;
        if (i + 2 < size) bits |= (unsigned char) data[i + 2];
        out += alphabet[(bits >> 18) & 63];
        out += alphabet[(bits >> 12) & 63];
        out += i + 1 < size ? alphabet[(bits >> 6) & 63] : '=';
        out += i + 2 < size ? alphabet[bits & 63] : '=';
    }
    return out;
}

static json synthesize(const json & request, const common_params & params,
                       llama_context * ctx, mtmd_context * mctx,
                       common_sampler * sampler, mtmd_helper::gen_audio & gen) {
    const std::string prompt = request.at("text").get<std::string>();
    const std::string speaker = request.at("speaker").get<std::string>();
    const bool streaming = request.value("stream", false);
    const std::string output = streaming ? "" : request.at("output").get<std::string>();
    const std::string lang = request.value("language", std::string("en"));
    const int max_frames = request.value("max_new_tokens", 320);
    if (prompt.empty() || prompt.size() > 65536 || max_frames < 1 || max_frames > 4096) {
        throw std::runtime_error("invalid text or max_new_tokens (1..4096)");
    }
    llama_memory_clear(llama_get_memory(ctx), true);
    common_sampler_reset(sampler);
    mtmd::bitmap_ptr bitmap;
    auto wrapper = mtmd_helper_bitmap_init_from_file(
        mctx, speaker.c_str(), false, mtmd_helper_init_opt_default());
    if (!wrapper.bitmap) throw std::runtime_error("cannot load reference audio");
    bitmap.reset(wrapper.bitmap);

    mtmd_helper_gen_audio_inp inp{};
    inp.seq_id = 0;
    inp.prompt = prompt.c_str();
    inp.prompt_len = prompt.size();
    inp.speaker_ref = bitmap.get();
    inp.lang = lang.c_str();
    inp.top_k = params.sampling.top_k;
    inp.top_p = params.sampling.top_p;
    inp.seed = params.sampling.seed;
    inp.out_type = streaming ? MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM : MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    if (gen.set_input(&inp) != 0) throw std::runtime_error("TTS set_input failed");
    for (;;) {
        int ret = gen.step_prompt(params.n_batch);
        if (ret < 0) throw std::runtime_error("TTS prompt processing failed");
        if (ret == 0) break;
    }
    auto sample = [&]() {
        llama_token token = common_sampler_sample(sampler, ctx, -1);
        common_sampler_accept(sampler, token, true);
        return token;
    };
    llama_token token = sample();
    const float * hidden = llama_get_embeddings_ith(ctx, -1);
    bool stop = false;
    int frames = 0;
    size_t sent_bytes = 0;
    auto emit_audio = [&]() {
        int32_t rate = 0;
        const char * pcm = nullptr;
        size_t size = 0;
        int64_t samples = 0;
        if (gen.get_output(&rate, &pcm, &size, &samples) != 0) {
            throw std::runtime_error("TTS streaming output failed");
        }
        if (size > sent_bytes) {
            std::cout << json({{"pcm_f32", encode64(pcm + sent_bytes, size - sent_bytes)},
                               {"sample_rate", rate}}).dump() << std::endl;
            sent_bytes = size;
        }
    };
    while (!stop && frames < max_frames) {
        const float * next = nullptr;
        if (gen.step_gen(token, hidden, &next, &stop) != 0) {
            throw std::runtime_error("TTS generation failed");
        }
        if (!next) break;
        ++frames;
        // The pinned Qwen libmtmd pipeline flushes its vocoder at 72 frames.
        // Drain only at that existing boundary: flushing a partial window here
        // would change the decoder's attention context and therefore the audio.
        if (streaming && frames % 72 == 0) emit_audio();
        hidden = next;
        token = sample();
    }
    if (streaming) {
        emit_audio(); // Finish the final partial window exactly as WAV mode does.
        return {{"ok", true}, {"frames", frames},
                {"limit_reached", !stop && frames >= max_frames}};
    }
    int32_t rate = 0;
    const char * data = nullptr;
    size_t size = 0;
    int64_t samples = 0;
    if (gen.get_output(&rate, &data, &size, &samples) != 0 || samples <= 0) {
        throw std::runtime_error("TTS returned no audio");
    }
    std::ofstream wav(output, std::ios::binary | std::ios::trunc);
    wav.write(data, size);
    wav.close();
    if (!wav) throw std::runtime_error("cannot write output WAV");
    return {{"ok", true}, {"sample_rate", rate}, {"frames", frames},
            {"limit_reached", !stop && frames >= max_frames}};
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TTS)) return 1;
    mtmd_helper_log_set(common_log_default_callback, nullptr);
    params.embedding = true;
    llama_backend_init();
    llama_numa_init(params.numa);
    try {
        auto init = common_init_from_params(params);
        auto model = init->model();
        auto ctx = init->context();
        if (!model || !ctx) throw std::runtime_error("cannot load TTS backbone");
        auto mp = mtmd_context_params_default();
        mp.use_gpu = params.mmproj_use_gpu;
        mp.device = params.mmproj_device;
        mtmd::context_ptr mctx(mtmd_init_from_file(params.mmproj.path.c_str(), model, mp));
        if (!mctx || mtmd_gen_audio_get_info(mctx.get()).type == MTMD_GEN_AUDIO_TYPE_NONE) {
            throw std::runtime_error("cannot load TTS audio encoder/decoder");
        }
        std::cout << json({{"ready", true}}).dump() << std::endl;
        // set_input resets per-utterance state while preserving the pipeline's
        // cached embedding matrix. Recreating it for every chunk is expensive.
        mtmd_helper::gen_audio gen(ctx, mctx.get());
        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                std::cout << synthesize(json::parse(line), params, ctx, mctx.get(),
                                        init->sampler(0), gen).dump() << std::endl;
            } catch (const std::exception & error) {
                std::cout << json({{"error", error.what()}}).dump() << std::endl;
            }
        }
    } catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    llama_backend_free();
    return 0;
}
