#pragma once
#include <functional>
#include <string>
#include <vector>
#include "ezlocalai/llm_video.hpp"

namespace ezlocalai {
struct LlmInputModel {
    std::string source;
    bool vision = false;
    std::string stops = R"(["<|im_end|","<|im_end|>","</|im_end|>","</s>","<s>","User:","### \n###","[/INST]","<|eot_id|>","<|end_of_text|>","assistant\n\n"])";
};
struct LlmInputImage {
    std::string bytes;
    std::string content_type = "image/jpeg";
};
struct LlmInputOperations {
    std::string local_uri;
    std::function<LlmInputModel()> load;
    std::function<LlmInputImage(const std::string &, const std::string &)> fetch_image;
    std::function<std::string(const std::string &)> png;
    // JSON result, including null when no description was produced.
    std::function<std::string(const std::vector<std::string> &, const std::string &)> describe;
    std::function<std::string(const std::string &)> fetch_audio;
    // JSON arguments and result preserve the source's raw input types.
    std::function<std::string(const std::string &, const std::string &)> transcribe;
    std::function<std::string(const std::string &, const std::string &)> transcribe_segments;
    std::string video_fps = "1.0", video_max_frames = "16";
    std::function<LlmInputVideo(const std::string &, double, int64_t)> extract_video;
    std::function<std::string(const std::string &)> read_audio;
    std::function<void(const std::string &)> remove_audio;
};
// All JSON text/image/audio/video items use the queued preparation phase.
bool llm_text_image_input_supported(const std::string &body);
// Pipes' multimodal preprocessing, including the model-load boundary. Throws
// LlmResponseError for Python input errors, before wrapper/inference fallback.
std::string llm_prepare_text_image_input(const std::string &body, bool chat,
    const LlmInputOperations &operations);
std::string llm_apply_audio_format(const std::string &body, bool chat,
    const LlmInputOperations &operations);
std::vector<std::string> llm_prepare_images(const std::vector<std::string> &images,
    const LlmInputOperations &operations, bool description = false);
}
