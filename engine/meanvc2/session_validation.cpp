// Optional real-model acceptance; no weights or reference audio enter images.
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/runtime/registry.h"
#include "ezlocalai/voice_resample.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <nlohmann/json.hpp>
#include <unistd.h>
namespace rt = engine::runtime;
int main(int argc, char **argv) {
    int report = dup(STDOUT_FILENO);
    dup2(STDERR_FILENO, STDOUT_FILENO);
    try {
        if (argc != 5)
            throw std::runtime_error("MODEL SOURCE_WAV VOICE_A VOICE_B required");
        auto registry = rt::make_default_registry();
        rt::ModelLoadRequest load;
        load.model_path = argv[1];
        load.family_hint = "meanvc2";
        auto model = registry.load(load);
        rt::SessionOptions options;
        options.backend.type = engine::core::BackendType::Cpu;
        options.backend.threads = 6;
        auto owner =
            model->create_task_session({rt::VoiceTaskKind::VoiceConversion, rt::RunMode::Streaming}, options);
        auto *session = dynamic_cast<rt::IStreamingVoiceTaskSession *>(owner.get());
        if (!session)
            throw std::runtime_error("Missing streaming interface");
        session->prepare(rt::SessionPreparationRequest{});
        const auto source = engine::audio::read_wav_f32(std::filesystem::path(argv[2]));
        if (source.channels != 1 || source.sample_rate != 24000)
            throw std::runtime_error("24kHz mono source required");
        nlohmann::json records = nlohmann::json::array();
        std::vector<std::vector<float>> outputs;
        for (int voice : {0, 0, 1, 1, 0}) {
            auto reference = engine::audio::read_wav_f32(std::filesystem::path(argv[3 + voice]));
            rt::TaskRequest request;
            request.voice = rt::VoiceCondition{};
            request.voice->speaker = rt::VoiceReference{};
            request.voice->speaker->audio =
                rt::AudioBuffer{reference.sample_rate, reference.channels, reference.samples};
            session->reset();
            auto started = std::chrono::steady_clock::now();
            session->start_stream(request);
            auto setup = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            ezlocalai::VoiceResampler down(24000, 16000);
            ezlocalai::VoiceBlockBuffer blocks;
            auto feed = [&](const std::vector<float> &v, uint64_t at) {
                rt::AudioChunk chunk;
                chunk.sample_rate = 16000;
                chunk.channels = 1;
                chunk.samples = v;
                chunk.start_sample = at;
                session->process_audio_chunk(chunk);
            };
            for (size_t at = 0; at < source.samples.size(); at += 3840)
                blocks.push(down.push(std::vector<float>(source.samples.begin() + at,
                                                         source.samples.begin() +
                                                             std::min(source.samples.size(), at + 3840))),
                            feed);
            blocks.push(down.finish(), feed);
            blocks.finish(feed);
            auto final = session->finish_stream();
            if (!final.audio_output || final.audio_output->samples.empty())
                throw std::runtime_error("Empty conversion");
            outputs.push_back(final.audio_output->samples);
            records.push_back(
                {{"voice", voice}, {"start_seconds", setup}, {"samples", outputs.back().size()}});
        }
        auto same = [&](int a, int b) {
            return outputs[a].size() == outputs[b].size() &&
                   !std::memcmp(outputs[a].data(), outputs[b].data(), outputs[a].size() * sizeof(float));
        };
        std::cerr << nlohmann::json{{"a_repeat", same(0, 1)},
                                    {"a_back", same(0, 4)},
                                    {"b_repeat", same(2, 3)},
                                    {"a_equals_b", same(0, 2)},
                                    {"records", records}}
                         .dump()
                  << "\n";
        if (!same(0, 1) || !same(2, 3) || same(0, 2) || same(2, 4))
            throw std::runtime_error("Conditioning reuse/invalidation changed expected output");
        const auto result = nlohmann::json{{"passed", true},
                                           {"bit_identical_cached_output", true},
                                           {"different_reference_changes_output", true},
                                           {"reference_invalidation_verified", true},
                                           {"recomputed_a_matches_original", same(0, 4)},
                                           {"records", records}}
                                .dump() +
                            "\n";
        if (write(report, result.data(), result.size()) != ssize_t(result.size()))
            throw std::runtime_error("Cannot write validation report");
        close(report);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
