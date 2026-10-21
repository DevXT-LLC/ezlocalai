#pragma once
#include <string>

namespace ezlocalai {
bool llm_qwen38_model(const std::string &model);
bool llm_minicpm5_model(const std::string &model);
// model must be the source name resolved from the selected replica.
std::string llm_model_request_settings(const std::string &body, const std::string &model);
// LLM.chat/completion's payload, before the native binding. Stream values keep
// their original JSON type; completion's callback guard is checked separately.
std::string llm_engine_request(const std::string &body, bool chat_mode,
    const std::string &output_tokens = "8192", const std::string &system_message = "");
void llm_completion_callback_guard(const std::string &payload);
}
