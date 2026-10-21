#pragma once
#include <optional>
#include <memory>
#include <string>
#include <utility>

namespace ezlocalai {
class LlmInferenceStream;
struct LlmResponseError { std::string category, message; };
std::optional<LlmResponseError> llm_response_error(const std::string &json, bool chat_stream = false);
std::string llm_stream_error_event(const std::string &message);
struct InferenceResult {
    std::string body;
    std::optional<LlmResponseError> failure;
    bool lazy_chat_stream = false;
    // Inference-phase results (including wrapper/binding failures), distinct
    // from model construction failures. No engine call is required on failure.
    bool from_engine = false;
    std::string stream_prefix;
    // Pipes' mutated input is also used by its post-inference fallback.
    std::string prepared_request;
    std::shared_ptr<LlmInferenceStream> stream;
    InferenceResult(std::string value) : body(std::move(value)) {}
    InferenceResult(std::string value, LlmResponseError error, bool lazy = false)
        : body(std::move(value)), failure(std::move(error)), lazy_chat_stream(lazy), from_engine(true) {}
};
// The buffered LLM wrapper cleans only the first choice and reports its
// configured source model. Completion streaming skips cleaning, but still
// checks errors and sets the model on dictionary results.
InferenceResult llm_buffered_response(const std::string &json, const std::string &model,
    bool chat_mode, bool completion_stream = false);
}
