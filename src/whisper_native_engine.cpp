#include "ezlocalai/whisper_transcription.hpp"
#include <stdexcept>
#include "ezlocalai/stt_runtime.hpp"

namespace ezlocalai {
WhisperNativeEngine::WhisperNativeEngine(const std::string &path,const std::string &device,const std::string &compute,size_t threads) {
    char error[4096]{}; auto *handle = ezl_whisper_kernel_create(path.c_str(),device.c_str(),compute.c_str(),threads,error,sizeof(error));
    if (!handle) throw SttError(ezl_whisper_kernel_error_category(),error);
    model_ = std::shared_ptr<void>(handle,ezl_whisper_kernel_destroy);
}
WhisperEngineOperations WhisperNativeEngine::operations() const {
    WhisperEngineOperations operations; char error[4096]{}; ezl_whisper_metadata metadata;
    if (ezl_whisper_kernel_metadata(model_.get(),&metadata,error,sizeof(error))) throw SttError(ezl_whisper_kernel_error_category(),error);
    operations.bands = metadata.bands; operations.multilingual = metadata.multilingual; operations.languages = metadata.languages;
    operations.encode = [model = model_](const WhisperFeatures &features) {
        char error[4096]{}; auto *result = ezl_whisper_kernel_encode(model.get(),features.values.data(),features.bands,features.frames,false,error,sizeof(error));
        if (!result) throw SttError(ezl_whisper_kernel_error_category(),error); return WhisperEncoded(result,ezl_whisper_encoded_destroy);
    };
    operations.generate = [model = model_](const WhisperEncoded &encoded,const std::vector<uint32_t> &prompt,const ezl_whisper_options &options) {
        struct Result { WhisperGeneration value; size_t calls = 0; } result;
        const auto callback = [](const uint64_t *tokens,size_t count,float score,float no_speech,void *context) {
            auto &result = *static_cast<Result *>(context); if (result.calls++) return;
            if (count) result.value.tokens.assign(tokens,tokens + count); result.value.score = score; result.value.no_speech_prob = no_speech;
        };
        const std::vector<uint64_t> tokens(prompt.begin(),prompt.end()); char error[4096]{};
        if (ezl_whisper_kernel_generate_encoded(model.get(),encoded.get(),tokens.data(),tokens.size(),&options,callback,&result,error,sizeof(error))) throw SttError(ezl_whisper_kernel_error_category(),error);
        if (!result.calls) throw std::runtime_error("Whisper generation returned no hypothesis"); return result.value;
    };
    operations.detect_language = [model = model_](const WhisperEncoded &encoded) {
        WhisperLanguages result; char error[4096]{};
        const auto callback = [](const char *language,float probability,void *context) { static_cast<WhisperLanguages *>(context)->emplace_back(language,probability); };
        if (ezl_whisper_kernel_detect_language(model.get(),encoded.get(),callback,&result,error,sizeof(error))) throw SttError(ezl_whisper_kernel_error_category(),error); return result;
    };
    operations.align = [model = model_](const WhisperEncoded &encoded,const std::vector<uint32_t> &start,const std::vector<uint32_t> &text,size_t frames,int64_t width) {
        WhisperAlignment result; char error[4096]{}; const std::vector<uint64_t> sequence(start.begin(),start.end()), tokens(text.begin(),text.end());
        const auto callback = [](const int64_t *pairs,size_t count,const float *probabilities,size_t probability_count,void *context) {
            auto &result = *static_cast<WhisperAlignment *>(context); for (size_t i = 0; i < count; ++i) result.indices.emplace_back(pairs[i * 2],pairs[i * 2 + 1]);
            if (probability_count) result.probabilities.assign(probabilities,probabilities + probability_count);
        };
        if (ezl_whisper_kernel_align(model.get(),encoded.get(),sequence.data(),sequence.size(),tokens.data(),tokens.size(),frames,width,callback,&result,error,sizeof(error))) throw SttError(ezl_whisper_kernel_error_category(),error); return result;
    };
    operations.encode_batch = [model = model_](const std::vector<WhisperFeatures> &batch) {
        if (batch.empty()) throw std::invalid_argument("Empty Whisper feature batch");
        const auto bands = batch[0].bands, frames = batch[0].frames; std::vector<float> values;
        values.reserve(batch.size() * bands * frames);
        for (const auto &features : batch) {
            if (features.bands != bands || features.frames != frames || features.values.size() != bands * frames) throw std::invalid_argument("Inconsistent Whisper batch feature shape");
            values.insert(values.end(),features.values.begin(),features.values.end());
        }
        char error[4096]{}; auto *result = ezl_whisper_kernel_encode_batch(model.get(),values.data(),batch.size(),bands,frames,false,error,sizeof(error));
        if (!result) throw SttError(ezl_whisper_kernel_error_category(),error); return WhisperEncoded(result,ezl_whisper_encoded_destroy);
    };
    operations.generate_batch = [model = model_](const WhisperEncoded &encoded,const std::vector<std::vector<uint32_t>> &prompts,const ezl_whisper_options &options) {
        std::vector<std::vector<uint64_t>> tokens; for (const auto &prompt : prompts) tokens.emplace_back(prompt.begin(),prompt.end());
        std::vector<const uint64_t *> pointers; std::vector<size_t> sizes;
        for (const auto &prompt : tokens) { pointers.push_back(prompt.data()); sizes.push_back(prompt.size()); }
        std::vector<WhisperGeneration> result; char error[4096]{};
        const auto callback = [](size_t index,const uint64_t *tokens,size_t count,float score,float no_speech,void *context) {
            auto &output = *static_cast<std::vector<WhisperGeneration> *>(context); if (output.size() <= index) output.resize(index + 1);
            auto &item = output[index]; if (count) item.tokens.assign(tokens,tokens + count); item.score = score; item.no_speech_prob = no_speech;
        };
        if (ezl_whisper_kernel_generate_batch(model.get(),encoded.get(),pointers.data(),sizes.data(),prompts.size(),&options,callback,&result,error,sizeof(error))) throw SttError(ezl_whisper_kernel_error_category(),error);
        return result;
    };
    operations.detect_language_batch = [model = model_](const WhisperEncoded &encoded) {
        std::vector<WhisperLanguages> result; char error[4096]{};
        const auto callback = [](size_t index,const char *language,float probability,void *context) {
            auto &output = *static_cast<std::vector<WhisperLanguages> *>(context); if (output.size() <= index) output.resize(index + 1); output[index].emplace_back(language,probability);
        };
        if (ezl_whisper_kernel_detect_language_batch(model.get(),encoded.get(),callback,&result,error,sizeof(error))) throw SttError(ezl_whisper_kernel_error_category(),error); return result;
    };
    operations.align_batch = [model = model_](const WhisperEncoded &encoded,const std::vector<uint32_t> &start,const std::vector<std::vector<uint32_t>> &text,const std::vector<size_t> &frames,int64_t width) {
        if (text.size() != frames.size()) throw std::invalid_argument("Inconsistent Whisper alignment batch shape");
        const std::vector<uint64_t> sequence(start.begin(),start.end()); std::vector<std::vector<uint64_t>> tokens;
        for (const auto &item : text) tokens.emplace_back(item.begin(),item.end());
        std::vector<const uint64_t *> pointers; std::vector<size_t> counts;
        for (const auto &item : tokens) { pointers.push_back(item.data()); counts.push_back(item.size()); }
        std::vector<WhisperAlignment> result; char error[4096]{};
        const auto callback = [](size_t index,const int64_t *pairs,size_t count,const float *probabilities,size_t probability_count,void *context) {
            auto &output = *static_cast<std::vector<WhisperAlignment> *>(context); if (output.size() <= index) output.resize(index + 1); auto &item = output[index];
            for (size_t i = 0; i < count; ++i) item.indices.emplace_back(pairs[i * 2],pairs[i * 2 + 1]);
            if (probability_count) item.probabilities.assign(probabilities,probabilities + probability_count);
        };
        if (ezl_whisper_kernel_align_batch(model.get(),encoded.get(),sequence.data(),sequence.size(),pointers.data(),counts.data(),frames.data(),text.size(),width,callback,&result,error,sizeof(error))) throw SttError(ezl_whisper_kernel_error_category(),error);
        return result;
    };
    return operations;
}
}
