#include "ezlocalai/whisper_kernel.h"
#include <ctranslate2/models/whisper.h>
#include <cstdio>
#include <memory>

namespace {
thread_local const char *error_category = "RuntimeError";
void failure(char *output,size_t size,const std::exception &error) {
    error_category = dynamic_cast<const std::bad_alloc *>(&error) ? "MemoryError" :
        dynamic_cast<const std::out_of_range *>(&error) ? "IndexError" :
        dynamic_cast<const std::overflow_error *>(&error) ? "OverflowError" :
        (dynamic_cast<const std::invalid_argument *>(&error) || dynamic_cast<const std::length_error *>(&error) || dynamic_cast<const std::domain_error *>(&error)) ? "ValueError" : "RuntimeError";
    if (output && size) std::snprintf(output,size,"%s",error.what());
}
ctranslate2::models::WhisperOptions options_from(const ezl_whisper_options *requested) {
    ctranslate2::models::WhisperOptions options;
    if (requested) {
        options.beam_size=requested->beam_size;options.max_length=requested->max_length;options.sampling_topk=requested->sampling_topk;options.num_hypotheses=requested->num_hypotheses;
        options.patience=requested->patience;options.length_penalty=requested->length_penalty;options.repetition_penalty=requested->repetition_penalty;options.sampling_temperature=requested->sampling_temperature;
        options.no_repeat_ngram_size=requested->no_repeat_ngram_size;options.max_initial_timestamp_index=requested->max_initial_timestamp_index;options.suppress_blank=requested->suppress_blank;
        if (requested->suppress_tokens || requested->suppress_tokens_is_set) {
            options.suppress_tokens.clear();
            if (requested->suppress_count) {
                if (!requested->suppress_tokens) throw std::invalid_argument("Missing Whisper suppression tokens");
                options.suppress_tokens.assign(requested->suppress_tokens,requested->suppress_tokens+requested->suppress_count);
            }
        }
    }
    options.return_scores=true;options.return_no_speech_prob=true; return options;
}
void generate(ctranslate2::models::Whisper &model,const ctranslate2::StorageView &input,
    const uint64_t *prompt,size_t prompt_size,const ezl_whisper_options *requested,ezl_whisper_result_callback callback,void *user) {
    if (!prompt || !prompt_size || !callback) throw std::invalid_argument("Invalid Whisper generation arguments");
    std::vector<size_t> tokens(prompt,prompt+prompt_size);
    const auto result=model.generate(input,std::vector<std::vector<size_t>>{std::move(tokens)},options_from(requested)).front().get();
    for (size_t i=0;i<result.sequences_ids.size();++i) {
        const std::vector<uint64_t> sequence(result.sequences_ids[i].begin(),result.sequences_ids[i].end());
        callback(sequence.data(),sequence.size(),result.scores.at(i),result.no_speech_prob,user);
    }
}
}
extern "C" const char *ezl_whisper_kernel_error_category() { return error_category; }
extern "C" void *ezl_whisper_kernel_create(const char *model,const char *device,const char *compute,size_t threads,char *error,size_t size) {
    try {
        ctranslate2::ReplicaPoolConfig pool; pool.num_threads_per_replica = threads;
        return new ctranslate2::models::Whisper(model,ctranslate2::str_to_device(device),ctranslate2::str_to_compute_type(compute),{0},false,pool);
    } catch (const std::exception &exception) { failure(error,size,exception); return nullptr; }
}
extern "C" void ezl_whisper_kernel_destroy(void *handle) { delete static_cast<ctranslate2::models::Whisper *>(handle); }
extern "C" int ezl_whisper_kernel_metadata(void *handle,ezl_whisper_metadata *metadata,char *error,size_t size) {
    try {
        if (!handle || !metadata) throw std::invalid_argument("Invalid Whisper metadata arguments");
        auto &model=*static_cast<ctranslate2::models::Whisper *>(handle);
        metadata->bands=model.n_mels();metadata->languages=model.num_languages();metadata->multilingual=model.is_multilingual();return 0;
    } catch (const std::exception &exception) { failure(error,size,exception); return 1; }
}
extern "C" void *ezl_whisper_kernel_encode(void *handle,const float *features,size_t bands,size_t frames,bool to_cpu,char *error,size_t size) {
    return ezl_whisper_kernel_encode_batch(handle,features,1,bands,frames,to_cpu,error,size);
}
extern "C" void *ezl_whisper_kernel_encode_batch(void *handle,const float *features,size_t batch,size_t bands,size_t frames,bool to_cpu,char *error,size_t size) {
    try {
        if (!handle || !features || !batch || !bands || !frames) throw std::invalid_argument("Invalid Whisper encoding arguments");
        auto &model=*static_cast<ctranslate2::models::Whisper *>(handle);
        const ctranslate2::StorageView input({static_cast<ctranslate2::dim_t>(batch),static_cast<ctranslate2::dim_t>(bands),static_cast<ctranslate2::dim_t>(frames)},const_cast<float *>(features));
        return new ctranslate2::StorageView(model.encode(input,to_cpu).get());
    } catch (const std::exception &exception) { failure(error,size,exception); return nullptr; }
}
extern "C" void ezl_whisper_encoded_destroy(void *encoded) { delete static_cast<ctranslate2::StorageView *>(encoded); }
extern "C" int ezl_whisper_kernel_generate_batch(void *handle,void *encoded,const uint64_t *const *prompts,const size_t *sizes,size_t batch,
    const ezl_whisper_options *requested,ezl_whisper_batch_result_callback callback,void *user,char *error,size_t size) {
    try {
        if (!handle || !encoded || !prompts || !sizes || !batch || !callback) throw std::invalid_argument("Invalid Whisper batch generation arguments");
        std::vector<std::vector<size_t>> tokens;
        for (size_t i = 0; i < batch; ++i) { if (!prompts[i] || !sizes[i]) throw std::invalid_argument("Invalid Whisper batch prompt"); tokens.emplace_back(prompts[i],prompts[i] + sizes[i]); }
        auto futures = static_cast<ctranslate2::models::Whisper *>(handle)->generate(*static_cast<ctranslate2::StorageView *>(encoded),std::move(tokens),options_from(requested));
        for (size_t i = 0; i < futures.size(); ++i) {
            const auto result = futures[i].get(); const auto &first = result.sequences_ids.at(0); const std::vector<uint64_t> sequence(first.begin(),first.end());
            callback(i,sequence.data(),sequence.size(),result.scores.at(0),result.no_speech_prob,user);
        }
        return 0;
    } catch (const std::exception &exception) { failure(error,size,exception); return 1; }
}
extern "C" int ezl_whisper_kernel_detect_language_batch(void *handle,void *encoded,ezl_whisper_batch_language_callback callback,void *user,char *error,size_t size) {
    try {
        if (!handle || !encoded || !callback) throw std::invalid_argument("Invalid Whisper batch language detection arguments");
        auto futures = static_cast<ctranslate2::models::Whisper *>(handle)->detect_language(*static_cast<ctranslate2::StorageView *>(encoded));
        for (size_t i = 0; i < futures.size(); ++i) for (const auto &item : futures[i].get()) callback(i,item.first.c_str(),item.second,user);
        return 0;
    } catch (const std::exception &exception) { failure(error,size,exception); return 1; }
}
extern "C" int ezl_whisper_kernel_align_batch(void *handle,void *encoded,const uint64_t *start,size_t start_size,
    const uint64_t *const *tokens,const size_t *counts,const size_t *frames,size_t batch,int64_t width,
    ezl_whisper_batch_alignment_callback callback,void *user,char *error,size_t size) {
    try {
        if (!handle || !encoded || !start || !start_size || !tokens || !counts || !frames || !batch || !callback) throw std::invalid_argument("Invalid Whisper batch alignment arguments");
        std::vector<std::vector<size_t>> text;
        for (size_t i = 0; i < batch; ++i) { text.emplace_back(); if (counts[i]) { if (!tokens[i]) throw std::invalid_argument("Missing Whisper batch alignment tokens"); text.back().assign(tokens[i],tokens[i] + counts[i]); } }
        auto futures = static_cast<ctranslate2::models::Whisper *>(handle)->align(*static_cast<ctranslate2::StorageView *>(encoded),
            std::vector<size_t>(start,start + start_size),std::move(text),std::vector<size_t>(frames,frames + batch),width);
        for (size_t i = 0; i < futures.size(); ++i) {
            const auto result = futures[i].get(); std::vector<int64_t> pairs;
            for (const auto &pair : result.alignments) { pairs.push_back(pair.first); pairs.push_back(pair.second); }
            callback(i,pairs.data(),result.alignments.size(),result.text_token_probs.data(),result.text_token_probs.size(),user);
        }
        return 0;
    } catch (const std::exception &exception) { failure(error,size,exception); return 1; }
}
extern "C" int ezl_whisper_kernel_generate_encoded(void *handle,void *encoded,const uint64_t *prompt,size_t prompt_size,
    const ezl_whisper_options *requested,ezl_whisper_result_callback callback,void *user,char *error,size_t size) {
    try {
        if (!handle || !encoded) throw std::invalid_argument("Invalid encoded Whisper input");
        generate(*static_cast<ctranslate2::models::Whisper *>(handle),*static_cast<ctranslate2::StorageView *>(encoded),prompt,prompt_size,requested,callback,user);return 0;
    } catch (const std::exception &exception) { failure(error,size,exception); return 1; }
}
extern "C" int ezl_whisper_kernel_detect_language(void *handle,void *encoded,ezl_whisper_language_callback callback,void *user,char *error,size_t size) {
    try {
        if (!handle || !encoded || !callback) throw std::invalid_argument("Invalid Whisper language detection arguments");
        const auto result=static_cast<ctranslate2::models::Whisper *>(handle)->detect_language(*static_cast<ctranslate2::StorageView *>(encoded)).front().get();
        for (const auto &item:result) callback(item.first.c_str(),item.second,user);return 0;
    } catch (const std::exception &exception) { failure(error,size,exception); return 1; }
}
extern "C" int ezl_whisper_kernel_align(void *handle,void *encoded,const uint64_t *start,size_t start_size,
    const uint64_t *tokens,size_t token_count,size_t frames,int64_t median_filter_width,ezl_whisper_alignment_callback callback,void *user,char *error,size_t size) {
    try {
        if (!handle || !encoded || !start || !start_size || (!tokens && token_count) || !callback) throw std::invalid_argument("Invalid Whisper alignment arguments");
        std::vector<size_t> text;if(token_count)text.assign(tokens,tokens+token_count);
        const auto result=static_cast<ctranslate2::models::Whisper *>(handle)->align(*static_cast<ctranslate2::StorageView *>(encoded),
            std::vector<size_t>(start,start+start_size),{std::move(text)},{frames},median_filter_width).front().get();
        std::vector<int64_t> pairs;for(const auto &pair:result.alignments){pairs.push_back(pair.first);pairs.push_back(pair.second);}
        callback(pairs.data(),result.alignments.size(),result.text_token_probs.data(),result.text_token_probs.size(),user);return 0;
    } catch (const std::exception &exception) { failure(error,size,exception); return 1; }
}
extern "C" int ezl_whisper_kernel_generate(void *handle,const float *features,size_t bands,size_t frames,
    const uint64_t *prompt,size_t prompt_size,const ezl_whisper_options *requested,
    ezl_whisper_result_callback callback,void *user,char *error,size_t size) {
    try {
        if (!handle || !features || !bands || !frames || !prompt || !prompt_size || !callback) throw std::invalid_argument("Invalid Whisper generation arguments");
        auto &model = *static_cast<ctranslate2::models::Whisper *>(handle);
        if (bands != model.n_mels()) throw std::invalid_argument("Whisper feature band count does not match model");
        const ctranslate2::StorageView input({1,static_cast<ctranslate2::dim_t>(bands),static_cast<ctranslate2::dim_t>(frames)},const_cast<float *>(features));
        generate(model,input,prompt,prompt_size,requested,callback,user);
        return 0;
    } catch (const std::exception &exception) { failure(error,size,exception); return 1; }
}
