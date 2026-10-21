#pragma once
#include <cstddef>
#include <cstdint>
// This boundary keeps CTranslate2's libstdc++ ABI independent of the server.
extern "C" {
struct ezl_whisper_options {
    size_t beam_size = 1, max_length = 448, sampling_topk = 1, num_hypotheses = 1;
    float patience = 1, length_penalty = 1, repetition_penalty = 1, sampling_temperature = 1;
    size_t no_repeat_ngram_size = 0, max_initial_timestamp_index = 50;
    bool suppress_blank = true;
    const int *suppress_tokens = nullptr;
    size_t suppress_count = 0;
    bool suppress_tokens_is_set = false;
};
using ezl_whisper_result_callback = void (*)(const uint64_t *,size_t,float,float,void *);
using ezl_whisper_language_callback = void (*)(const char *,float,void *);
using ezl_whisper_alignment_callback = void (*)(const int64_t *,size_t,const float *,size_t,void *);
using ezl_whisper_batch_result_callback = void (*)(size_t,const uint64_t *,size_t,float,float,void *);
using ezl_whisper_batch_language_callback = void (*)(size_t,const char *,float,void *);
using ezl_whisper_batch_alignment_callback = void (*)(size_t,const int64_t *,size_t,const float *,size_t,void *);
struct ezl_whisper_metadata { size_t bands = 0, languages = 0; bool multilingual = false; };
// Category of the last failed call on this thread, matching pybind11 translation.
const char *ezl_whisper_kernel_error_category();
void *ezl_whisper_kernel_create(const char *model,const char *device,const char *compute,size_t threads,char *error,size_t error_size);
void ezl_whisper_kernel_destroy(void *handle);
int ezl_whisper_kernel_metadata(void *handle,ezl_whisper_metadata *metadata,char *error,size_t error_size);
void *ezl_whisper_kernel_encode(void *handle,const float *features,size_t bands,size_t frames,
    bool to_cpu,char *error,size_t error_size);
void ezl_whisper_encoded_destroy(void *encoded);
void *ezl_whisper_kernel_encode_batch(void *handle,const float *features,size_t batch,size_t bands,size_t frames,
    bool to_cpu,char *error,size_t error_size);
int ezl_whisper_kernel_generate_batch(void *handle,void *encoded,const uint64_t *const *prompts,const size_t *prompt_sizes,size_t batch,
    const ezl_whisper_options *options,ezl_whisper_batch_result_callback callback,void *user,char *error,size_t error_size);
int ezl_whisper_kernel_detect_language_batch(void *handle,void *encoded,ezl_whisper_batch_language_callback callback,
    void *user,char *error,size_t error_size);
int ezl_whisper_kernel_align_batch(void *handle,void *encoded,const uint64_t *start,size_t start_size,
    const uint64_t *const *tokens,const size_t *token_counts,const size_t *frames,size_t batch,int64_t median_filter_width,
    ezl_whisper_batch_alignment_callback callback,void *user,char *error,size_t error_size);
int ezl_whisper_kernel_generate_encoded(void *handle,void *encoded,const uint64_t *prompt,size_t prompt_size,
    const ezl_whisper_options *options,ezl_whisper_result_callback callback,void *user,char *error,size_t error_size);
int ezl_whisper_kernel_detect_language(void *handle,void *encoded,ezl_whisper_language_callback callback,
    void *user,char *error,size_t error_size);
int ezl_whisper_kernel_align(void *handle,void *encoded,const uint64_t *start,size_t start_size,
    const uint64_t *tokens,size_t token_count,size_t frames,int64_t median_filter_width,
    ezl_whisper_alignment_callback callback,void *user,char *error,size_t error_size);
int ezl_whisper_kernel_generate(void *handle,const float *features,size_t bands,size_t frames,
    const uint64_t *prompt,size_t prompt_size,const ezl_whisper_options *options,
    ezl_whisper_result_callback callback,void *user,char *error,size_t error_size);
}
