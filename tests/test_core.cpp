#define _POSIX_C_SOURCE 200809L

#include "ezlocalai/ezlocalai_c.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cerrno>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <future>
#include <vector>

static void expect(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        _Exit(1);
    }
}

static void echo_processor(const char *payload, char *result, size_t result_size, void *user_data) {
    (void)user_data;
    snprintf(result, result_size, "processed:%s", payload);
}

struct BlockingProcessor {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false;
    bool released = false;
    int calls = 0;
};

struct QueueObserver {
    ezl_queue *queue = nullptr;
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<ezl_queue_snapshot> snapshots;
    bool hold = false, entered = false, released = false;
};
static void observe_queue(void *data) {
    auto &state = *static_cast<QueueObserver *>(data);
    ezl_queue_snapshot snapshot{};
    // This would deadlock if observers were called under the queue state lock.
    ezl_queue_get_snapshot(state.queue, &snapshot);
    std::unique_lock<std::mutex> lock(state.mutex);
    state.snapshots.push_back(snapshot); state.entered = true; state.changed.notify_all();
    if (state.hold) state.changed.wait(lock, [&] { return state.released; });
}
static void test_queue_observer_lifetime() {
    auto *queue = ezl_queue_create(1, 1);
    QueueObserver observer; observer.queue = queue; observer.hold = true;
    expect(ezl_queue_set_state_observer(queue, observe_queue, &observer) == 0, "install stopped queue observer");
    auto update = std::async(std::launch::async, [&] { return ezl_queue_set_concurrency(queue, 2); });
    {
        std::unique_lock<std::mutex> lock(observer.mutex);
        expect(observer.changed.wait_for(lock, std::chrono::seconds(2), [&] { return observer.entered; }), "observer can inspect the updated state");
    }
    auto unregister = std::async(std::launch::async, [&] { return ezl_queue_set_state_observer(queue, nullptr, nullptr); });
    expect(unregister.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
        "unregister waits for an in-flight observer before freeing userdata");
    { std::lock_guard<std::mutex> lock(observer.mutex); observer.released = true; observer.changed.notify_all(); }
    expect(update.get() == 0 && unregister.get() == 0, "observer update and unregister finish");
    expect(ezl_queue_set_concurrency(queue, 1) == 0 && observer.snapshots.size() == 1, "cleared observer is never called again");
    ezl_queue_destroy(queue);
}

static void blocking_processor(const char *, char *result, size_t size, void *data) {
    auto &state = *static_cast<BlockingProcessor *>(data);
    std::unique_lock<std::mutex> lock(state.mutex);
    ++state.calls;
    state.entered = true;
    state.changed.notify_all();
    state.changed.wait(lock, [&] { return state.released; });
    snprintf(result, size, "released");
}

static void failing_processor(const char *, char *, size_t, void *) {
    throw std::runtime_error("processor exception");
}

static void test_queue_lifecycle() {
    BlockingProcessor state;
    auto *queue = ezl_queue_create(1, 1);
    expect(queue != nullptr, "create bounded queue");
    QueueObserver observer; observer.queue = queue;
    expect(ezl_queue_set_state_observer(queue, observe_queue, &observer) == 0, "install queue transition observer");
    expect(ezl_queue_start(queue, blocking_processor, &state) == 0, "start blocking processor");
    expect(ezl_queue_set_state_observer(queue, nullptr, nullptr) == EBUSY, "running queue observer cannot be replaced");
    char first[64], second[64], rejected[64], result[64], error[64];
    expect(ezl_queue_enqueue(queue, "first", first, sizeof(first)) == 0, "enqueue blocking request");
    {
        std::unique_lock<std::mutex> lock(state.mutex);
        expect(state.changed.wait_for(lock, std::chrono::seconds(2), [&] { return state.entered; }), "worker starts request");
    }
    expect(ezl_queue_wait(queue, first, 5, result, sizeof(result), error, sizeof(error)) == ETIMEDOUT, "wait deadline expires");
    expect(ezl_queue_enqueue(queue, "second", second, sizeof(second)) == 0, "enqueue waiting request");
    expect(ezl_queue_enqueue(queue, "overflow", rejected, sizeof(rejected)) == ENOSPC, "queue capacity enforced");
    expect(ezl_queue_cancel(queue, second) == 0, "cancel pending request");
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.released = true;
        state.changed.notify_all();
    }
    expect(ezl_queue_wait(queue, first, 2000, result, sizeof(result), error, sizeof(error)) == 0, "timed out waiter can observe completion");
    expect(ezl_queue_wait(queue, second, 2000, result, sizeof(result), error, sizeof(error)) == ECANCELED, "cancelled request never executes");
    ezl_queue_stop(queue);
    expect(state.calls == 1, "only first callback ran");
    ezl_queue_snapshot snapshot{};
    ezl_queue_get_snapshot(queue, &snapshot);
    expect(snapshot.active == 0 && snapshot.processing == 0 && snapshot.queued == 0, "cancelled work drains accounting");
    expect(snapshot.total_processed == 1 && snapshot.total_failed == 0, "cancelled work is not counted as a failure");
    {
        std::lock_guard<std::mutex> lock(observer.mutex);
        expect(!observer.snapshots.empty() && observer.snapshots.back().active == 0 && observer.snapshots.back().processing == 0,
            "final observer sees released processing and active counters after timed-out work and cancellation");
    }
    expect(ezl_queue_enqueue(queue, "stopped", rejected, sizeof(rejected)) == EINVAL, "stopped queue rejects submissions");
    expect(ezl_queue_start(queue, failing_processor, nullptr) == 0, "restart queue");
    expect(ezl_queue_enqueue(queue, "failure", first, sizeof(first)) == 0, "enqueue failing request");
    expect(ezl_queue_wait(queue, first, 2000, result, sizeof(result), error, sizeof(error)) == EIO, "callback exception becomes request failure");
    expect(strcmp(error, "processor exception") == 0, "callback failure preserves error");
    ezl_queue_stop(queue);
    expect(ezl_queue_start(queue, echo_processor, nullptr) == 0, "restart after callback failure");
    expect(ezl_queue_enqueue(queue, "healthy", first, sizeof(first)) == 0, "enqueue after restart");
    expect(ezl_queue_wait(queue, first, 2000, result, sizeof(result), error, sizeof(error)) == 0, "queue recovers after callback failure");
    expect(strcmp(result, "processed:healthy") == 0, "restarted queue uses new processor");
    ezl_queue_destroy(queue);
}

int main(void) {
    test_queue_observer_lifetime();
    remove("native-c-api-empty.env");
    setenv("EZLOCALAI_DOTENV_PATH", "native-c-api-empty.env", 1);

    struct {
        const char *name;
        const char *value;
    } defaults[] = {
        {"EZLOCALAI_URL", "http://localhost:8091"},
        {"EZLOCALAI_API_KEY", "none"},
        {"ALLOWED_DOMAINS", "*"},
        {"DEFAULT_MODEL", "unsloth/Qwen3.5-4B-GGUF"},
        {"WHISPER_MODEL", "large-v3-turbo"},
        {"IMG_MODEL", "none"},
        {"VIDEO_MODEL", "QuantStack/Wan2.2-T2V-A14B-GGUF"},
        {"TTS_ENABLED", "true"},
        {"TTS_PROVIDER", "qwen"},
        {"STT_ENABLED", "true"},
        {"NGROK_TOKEN", ""},
        {"LOG_LEVEL", "INFO"},
        {"LOG_FORMAT", "%(asctime)s | %(levelname)s | %(message)s"},
        {"UVICORN_WORKERS", "1"},
        {"MAIN_GPU", "0"},
        {"TENSOR_SPLIT", ""},
        {"QUANT_TYPE", "Q4_K_XL"},
        {"LLM_BATCH_SIZE", "auto"},
        {"LLM_MAX_TOKENS", "65536"},
        {"REASONING_BUDGET", "-1"},
        {"N_PARALLEL", "1"},
        {"MAX_CONCURRENT_REQUESTS", ""},
        {"VLM_MAX_TOKENS", "8192"},
        {"MAX_QUEUE_SIZE", "100"},
        {"REQUEST_TIMEOUT", "120"},
        {"QUEUE_WAIT_TIMEOUT", "10"},
        {"FALLBACK_SERVER", ""},
        {"FALLBACK_API_KEY", ""},
        {"FALLBACK_MODEL", ""},
        {"FALLBACK_MEMORY_THRESHOLD", "1.0"},
        {"VOICE_SERVER", ""},
        {"VOICE_SERVER_API_KEY", ""},
        {"IMAGE_SERVER", ""},
        {"IMAGE_SERVER_API_KEY", ""},
        {"TEXT_SERVER", ""},
        {"TEXT_SERVER_API_KEY", ""},
        {"LAZY_LOAD_VOICE", "true"},
        {"ROUTER_MODE", "false"},
        {"ROUTER_REGISTER_KEY", ""},
        {"ROUTER_WORKER_TTL", "30"},
        {"ROUTER_WAIT_TIMEOUT", "0"},
        {"ROUTER_URL", ""},
        {"ROUTER_API_KEY", ""},
        {"WORKER_LABEL", ""},
        {"WORKER_HEARTBEAT_INTERVAL", "10"},
        {"WORKER_TUNNEL", "false"},
    };
    for (size_t i = 0; i < sizeof(defaults) / sizeof(defaults[0]); ++i) {
        unsetenv(defaults[i].name);
        expect(strcmp(ezl_getenv_default(defaults[i].name), defaults[i].value) == 0, defaults[i].name);
    }

    char ts[32];
    expect(ezl_format_timestamp_srt(3661.234, ts, sizeof(ts)) == 0, "format srt succeeds");
    expect(strcmp(ts, "01:01:01,233") == 0 || strcmp(ts, "01:01:01,234") == 0, "srt timestamp shape");
    expect(ezl_format_timestamp_vtt(5.006, ts, sizeof(ts)) == 0, "format vtt succeeds");
    expect(strcmp(ts, "00:00:05.006") == 0, "vtt timestamp shape");

    char words[512];
    expect(ezl_number_to_words(0, words, sizeof(words)) == 0, "zero to words");
    expect(strcmp(words, "zero") == 0, "zero word");
    expect(ezl_number_to_words(123456, words, sizeof(words)) == 0, "large number to words");
    expect(strcmp(words, "one hundred twenty three thousand four hundred fifty six") == 0, "large number words");
    expect(ezl_number_to_words(1000000, words, sizeof(words)) == 0, "very large number to string");
    expect(strcmp(words, "1000000") == 0, "very large number string");

    expect(ezl_normalize_text_for_tts("Meet at 10:05 PM on 12/24/2025 with 42 guests.", words, sizeof(words)) == 0, "normalize tts");
    expect(
        strcmp(
            words,
            "Meet at ten oh five P M on December twenty fourth two thousand twenty five with forty two guests."
        ) == 0,
        "normalize tts output"
    );
    expect(ezl_normalize_text_for_tts("Room A12 has 7 seats and code x9.", words, sizeof(words)) == 0, "normalize boundaries");
    expect(strcmp(words, "Room A12 has seven seats and code x9.") == 0, "normalize number boundaries");

    ezl_string_list chunks;
    expect(ezl_split_text_into_chunks("First sentence. Second sentence is longer. Third sentence.", 35, &chunks) == 0, "split sentences");
    expect(chunks.count == 3, "sentence chunk count");
    expect(strcmp(chunks.items[0], "First sentence.") == 0, "sentence chunk 0");
    expect(strcmp(chunks.items[1], "Second sentence is longer.") == 0, "sentence chunk 1");
    expect(strcmp(chunks.items[2], "Third sentence.") == 0, "sentence chunk 2");
    ezl_string_list_free(&chunks);

    expect(ezl_split_text_into_chunks("alpha, beta, gamma, delta", 13, &chunks) == 0, "split commas");
    expect(chunks.count == 3, "comma chunk count");
    expect(strcmp(chunks.items[0], "alpha , beta") == 0, "comma chunk 0");
    expect(strcmp(chunks.items[1], ", gamma ,") == 0, "comma chunk 1");
    expect(strcmp(chunks.items[2], "delta") == 0, "comma chunk 2");
    ezl_string_list_free(&chunks);

    expect(ezl_split_text_into_chunks("one two three four five", 11, &chunks) == 0, "force split words");
    expect(chunks.count == 3, "word chunk count");
    expect(strcmp(chunks.items[0], "one two") == 0, "word chunk 0");
    expect(strcmp(chunks.items[1], "three four") == 0, "word chunk 1");
    expect(strcmp(chunks.items[2], "five") == 0, "word chunk 2");
    ezl_string_list_free(&chunks);

    expect(ezl_gpu_tier_for_name(NULL) == 2, "gpu empty is cpu");
    expect(ezl_gpu_tier_for_name("") == 2, "gpu blank is cpu");
    expect(ezl_gpu_tier_for_name("NVIDIA GeForce RTX 4090") == 80, "4090 tier");
    expect(ezl_gpu_tier_for_name("NVIDIA GeForce RTX 4070 Ti SUPER") == 55, "ordered 4070 ti super tier");
    expect(ezl_gpu_tier_for_name("Jetson Orin Nano") == 12, "ordered orin nano tier");
    expect(ezl_gpu_tier_for_name("AMD Radeon RX 7900 XTX") == 55, "amd tier");
    expect(ezl_gpu_tier_for_name("Mystery Accelerator") == 20, "default gpu tier");

    ezl_string_list caps;
    unsetenv("IMAGE_ENABLED");
    unsetenv("VIDEO_ENABLED");
    unsetenv("MUSIC_ENABLED");
    unsetenv("EMBEDDING_ENABLED");
    unsetenv("EMBEDDING_SERVER");
    unsetenv("DEFAULT_MODEL");
    unsetenv("VOICE_SERVER");
    unsetenv("IMAGE_SERVER");
    unsetenv("TEXT_SERVER");
    unsetenv("IMG_MODEL");
    unsetenv("VIDEO_MODEL");
    unsetenv("TTS_ENABLED");
    unsetenv("STT_ENABLED");
    expect(ezl_detect_local_capabilities(&caps) == 0, "default capabilities");
    expect(caps.count == 4, "default capability count");
    expect(strcmp(caps.items[0], "text") == 0, "default cap text");
    expect(strcmp(caps.items[1], "tts") == 0, "default cap tts");
    expect(strcmp(caps.items[2], "stt") == 0, "default cap stt");
    expect(strcmp(caps.items[3], "embedding") == 0, "default cap embedding");
    ezl_string_list_free(&caps);

    setenv("DEFAULT_MODEL", "vendor/My-Vision-Model-GGUF", 1);
    setenv("VOICE_SERVER", "http://voice-worker:8091", 1);
    setenv("IMG_MODEL", "unsloth/FLUX.2-klein-4B-GGUF", 1);
    setenv("IMAGE_ENABLED", "true", 1);
    setenv("VIDEO_ENABLED", "true", 1);
    setenv("VIDEO_MODEL", "unsloth/LTX-2.3-GGUF", 1);
    expect(ezl_detect_local_capabilities(&caps) == 0, "custom capabilities");
    expect(caps.count == 5, "custom capability count");
    expect(strcmp(caps.items[0], "text") == 0, "custom cap text");
    expect(strcmp(caps.items[1], "vision") == 0, "custom cap vision");
    expect(strcmp(caps.items[2], "image") == 0, "custom cap image");
    expect(strcmp(caps.items[3], "video") == 0, "custom cap video");
    expect(strcmp(caps.items[4], "embedding") == 0, "custom cap embedding");
    ezl_string_list_free(&caps);
    unsetenv("IMAGE_ENABLED");
    unsetenv("VIDEO_ENABLED");
    unsetenv("MUSIC_ENABLED");
    unsetenv("EMBEDDING_ENABLED");
    unsetenv("EMBEDDING_SERVER");
    unsetenv("DEFAULT_MODEL");
    unsetenv("VOICE_SERVER");
    unsetenv("IMG_MODEL");
    unsetenv("VIDEO_MODEL");

    unsetenv("DEFAULT_MODEL");
    unsetenv("MAX_CONCURRENT_REQUESTS");
    unsetenv("N_PARALLEL");
    unsetenv("TEXT_SERVER_API_KEY");
    unsetenv("REQUEST_TIMEOUT");
    FILE *dotenv = fopen("native-c-api.env", "w");
    expect(dotenv != NULL, "create dotenv fixture");
    fprintf(dotenv, "# native dotenv fixture\n");
    fprintf(dotenv, "DEFAULT_MODEL=dotenv-model\n");
    fprintf(dotenv, "MAX_CONCURRENT_REQUESTS=3\n");
    fprintf(dotenv, "TEXT_SERVER_API_KEY=\"quoted value\"\n");
    fprintf(dotenv, "export REQUEST_TIMEOUT=2.5\n");
    fclose(dotenv);
    setenv("DEFAULT_MODEL", "process-model", 1);
    expect(ezl_load_dotenv_file("native-c-api.env") == 0, "load dotenv fixture");
    expect(strcmp(getenv("DEFAULT_MODEL"), "process-model") == 0, "dotenv does not override process env");
    expect(strcmp(getenv("MAX_CONCURRENT_REQUESTS"), "3") == 0, "dotenv max concurrent");
    expect(strcmp(getenv("TEXT_SERVER_API_KEY"), "quoted value") == 0, "dotenv quoted value");
    ezl_env_config dotenv_config;
    ezl_load_config(&dotenv_config);
    expect(strcmp(dotenv_config.default_model, "process-model") == 0, "dotenv config env precedence");
    expect(dotenv_config.max_concurrent_requests == 3, "max concurrent fallback feeds native concurrency");
    expect(dotenv_config.request_timeout_ms == 2500, "dotenv decimal seconds");
    remove("native-c-api.env");
    unsetenv("DEFAULT_MODEL");
    unsetenv("MAX_CONCURRENT_REQUESTS");
    unsetenv("N_PARALLEL");
    unsetenv("TEXT_SERVER_API_KEY");
    unsetenv("REQUEST_TIMEOUT");

    ezl_env_config config;
    ezl_load_config(&config);
    expect(config.max_queue_size > 0, "config queue size");
    expect(config.max_concurrent_requests > 0, "config concurrency");

    ezl_queue *queue = ezl_queue_create(1, 4);
    expect(queue != NULL, "queue create");
    expect(ezl_queue_is_running(queue) == 0, "queue initially stopped");
    expect(ezl_queue_start(queue, echo_processor, NULL) == 0, "queue start");
    expect(ezl_queue_is_running(queue) == 1, "queue running after start");

    char id[64];
    expect(ezl_queue_enqueue_typed(queue, "hello", "chat", id, sizeof(id)) == 0, "queue enqueue");

    char result[128];
    char error[128];
    expect(ezl_queue_wait(queue, id, 2000, result, sizeof(result), error, sizeof(error)) == 0, "queue wait");
    expect(strcmp(result, "processed:hello") == 0, "queue result");

    ezl_request_info info;
    expect(ezl_queue_get_request_info(queue, id, &info) == 0, "queue request info");
    expect(strcmp(info.id, id) == 0, "request info id");
    expect(strcmp(info.completion_type, "chat") == 0, "request info completion type");
    expect(info.status == EZL_REQUEST_COMPLETED, "request info status");
    expect(info.timestamp > 0.0, "request info timestamp");

    ezl_queue_snapshot snapshot;
    ezl_queue_get_snapshot(queue, &snapshot);
    expect(snapshot.total_queued == 1, "snapshot queued");
    expect(snapshot.total_processed == 1, "snapshot processed");
    expect(snapshot.active == 0, "snapshot active drained");

    ezl_queue_stop(queue);
    expect(ezl_queue_is_running(queue) == 0, "queue stopped");

    ezl_queue_destroy(queue);
    test_queue_lifecycle();
    return 0;
}
