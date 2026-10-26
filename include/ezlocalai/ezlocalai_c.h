#ifndef EZLOCALAI_C_H
#define EZLOCALAI_C_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ezl_env_config {
    char url[256];
    char api_key[256];
    char default_model[512];
    char log_level[64];
    size_t max_queue_size;
    size_t max_concurrent_requests;
    long request_timeout_ms;
} ezl_env_config;

typedef enum ezl_request_status {
    EZL_REQUEST_QUEUED = 0,
    EZL_REQUEST_PROCESSING = 1,
    EZL_REQUEST_COMPLETED = 2,
    EZL_REQUEST_FAILED = 3,
    EZL_REQUEST_CANCELLED = 4
} ezl_request_status;

typedef struct ezl_queue_snapshot {
    size_t max_queue_size;
    size_t max_concurrent_requests;
    size_t queued;
    size_t processing;
    size_t active;
    unsigned long long total_queued;
    unsigned long long total_processed;
    unsigned long long total_failed;
} ezl_queue_snapshot;

typedef struct ezl_request_info {
    char id[64];
    ezl_request_status status;
    char completion_type[64];
    double timestamp;
    char error[256];
} ezl_request_info;

typedef struct ezl_string_list {
    size_t count;
    char **items;
} ezl_string_list;

typedef struct ezl_queue ezl_queue;

typedef void (*ezl_processor_fn)(
    const char *payload,
    char *result,
    size_t result_size,
    void *user_data
);

const char *ezl_getenv_default(const char *name);
// Match Globals.getenv(name, fallback): an empty fallback uses built-in
// defaults; an explicitly set environment value (including empty) wins.
const char *ezl_getenv_fallback(const char *name, const char *fallback);
int ezl_load_dotenv(void);
int ezl_load_dotenv_file(const char *path);
void ezl_load_config(ezl_env_config *out);

int ezl_format_timestamp_srt(double seconds, char *out, size_t out_size);
int ezl_format_timestamp_vtt(double seconds, char *out, size_t out_size);
// CTTS helpers accept UTF-8; chunk limits count code points and zero clamps to one.
int ezl_number_to_words(long n, char *out, size_t out_size);
int ezl_normalize_text_for_tts(const char *text, char *out, size_t out_size);
int ezl_split_text_into_chunks(const char *text, size_t max_chars, ezl_string_list *out);
void ezl_string_list_free(ezl_string_list *list);
int ezl_gpu_tier_for_name(const char *name);
int ezl_detect_local_capabilities(ezl_string_list *out);

ezl_queue *ezl_queue_create(size_t max_concurrent_requests, size_t max_queue_size);
void ezl_queue_destroy(ezl_queue *queue);

int ezl_queue_start(ezl_queue *queue, ezl_processor_fn processor, void *user_data);
void ezl_queue_stop(ezl_queue *queue);
int ezl_queue_is_running(ezl_queue *queue);
/* Resize admission without cancelling active work; zero selects one slot. */
int ezl_queue_set_concurrency(ezl_queue *queue, size_t max_concurrent_requests);

int ezl_queue_enqueue(
    ezl_queue *queue,
    const char *payload,
    char *request_id_out,
    size_t request_id_size
);

int ezl_queue_enqueue_typed(
    ezl_queue *queue,
    const char *payload,
    const char *completion_type,
    char *request_id_out,
    size_t request_id_size
);

int ezl_queue_wait(
    ezl_queue *queue,
    const char *request_id,
    long timeout_ms,
    char *result_out,
    size_t result_size,
    char *error_out,
    size_t error_size
);

int ezl_queue_cancel(ezl_queue *queue, const char *request_id);
// Install while stopped. Called after queue state changes, outside its mutex;
// observers are read-only: they may read snapshots but must not enqueue,
// cancel, reconfigure, stop or destroy the queue themselves. Clearing/replacing
// the observer waits for an in-flight callback.
typedef void (*ezl_queue_state_fn)(void *user_data);
int ezl_queue_set_state_observer(ezl_queue *queue, ezl_queue_state_fn observer, void *user_data);
int ezl_queue_get_request_status(ezl_queue *queue, const char *request_id, ezl_request_status *status_out);
int ezl_queue_get_request_info(ezl_queue *queue, const char *request_id, ezl_request_info *info_out);
void ezl_queue_get_snapshot(ezl_queue *queue, ezl_queue_snapshot *snapshot_out);

const char *ezl_request_status_name(ezl_request_status status);

#ifdef __cplusplus
}
#endif

#endif
