#ifndef EZLOCALAI_NATIVE_SERVER_HPP
#define EZLOCALAI_NATIVE_SERVER_HPP
#include "ezlocalai/llm_response.hpp"
#include "ezlocalai/llm_stream.hpp"
#include "ezlocalai/worker_heartbeat.hpp"
#include "ezlocalai/stt_service.hpp"
#include "ezlocalai/tts_service.hpp"
#include "ezlocalai/qwen_image.hpp"
#include "ezlocalai/wakeword_model.hpp"
#include "ezlocalai/wakeword_jobs.hpp"

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <deque>
#include <functional>
#include <mutex>
#include <memory>
#include <optional>
#include <set>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ezlocalai/ezlocalai_c.h"
#include "ezlocalai/video_runtime.hpp"
#include "ezlocalai/ltx_video.hpp"
#include "ezlocalai/embedding_runtime.hpp"
#include "ezlocalai/resource_manager.hpp"
#include "ezlocalai/model_lifecycle.hpp"
#include "ezlocalai/llm_catalog.hpp"
#include "ezlocalai/tts_routing.hpp"
#include "ezlocalai/configuration.hpp"

namespace ezlocalai {

class WanVideo;
class LtxVideo;
class HttpDeadline;

struct HttpResponse {
    enum class HeaderOrder {
        Default,
        StaticFile,
        StaticRangeError,
        StaticUnsatisfiable,
    };

    int status = 200;
    std::string content_type = "application/json";
    std::string body;
    std::vector<std::pair<std::string, std::string>> headers;
    bool add_default_cors_headers = true;
    HeaderOrder header_order = HeaderOrder::Default;
    bool chunked = false;
    bool transport_error = false;
    std::function<void(int)> stream_writer;
    std::shared_ptr<std::string> stream_capture;
    // Known length for deferred file transfers; body remains empty.
    std::optional<size_t> content_length;
};

struct RouterSlotState {
    int capacity = 0;
    int in_flight = 0;
    int queued = 0;
    int available = 0;
    int instances = -1;
};

struct RealtimeEngineLease {
    std::string url, api_key;
    std::function<void()> release;
    RealtimeEngineLease() = default;
    RealtimeEngineLease(const RealtimeEngineLease &) = delete;
    RealtimeEngineLease &operator=(const RealtimeEngineLease &) = delete;
    ~RealtimeEngineLease() { if (release) release(); }
};

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;
    virtual std::string chat(const std::string &request_body) = 0;
    virtual std::string completion(const std::string &request_body) = 0;
    virtual std::string embedding(const std::string &request_body) = 0;
    virtual std::unique_ptr<RealtimeEngineLease> open_realtime(const std::string &, const std::string &) {
        throw std::runtime_error("Realtime inference is unavailable on this backend");
    }
    virtual std::string chat_for_replica(const std::string &body, const std::string &) { return chat(body); }
    virtual std::string completion_for_replica(const std::string &body, const std::string &) { return completion(body); }
    virtual InferenceResult text_result(const std::string &body, const std::string &replica, bool chat_mode) {
        return chat_mode ? chat_for_replica(body, replica) : completion_for_replica(body, replica);
    }
    virtual InferenceResult lazy_text_result(const std::string &body, const std::string &replica, bool chat_mode) {
        return text_result(body, replica, chat_mode);
    }
    virtual InferenceResult pipeline_text_result(const std::string &body, const std::string &replica, bool chat_mode,
        const struct LlmInputOperations &input);
    virtual std::string loaded_models_json() const { return "{}"; }
    // Status-only observation may use a coherent snapshot during a model
    // transition. Admission/handoff decisions retain loaded_models_json().
    virtual std::string observed_loaded_models_json() const { return loaded_models_json(); }
    virtual std::string model_lifecycle_json() const { return ""; }
    virtual std::shared_ptr<ResourceManager> resource_manager() const { return {}; }
    virtual int llm_context_size() const { return 0; }
    virtual std::optional<int64_t> llm_parallel_count() const { return std::nullopt; }
    virtual std::map<std::string, int64_t> loaded_model_contexts() const { return {}; }
    virtual std::map<std::string, std::string> loaded_model_quants() const { return {}; }
    virtual bool loaded_vision_model() const { return false; }
    virtual bool swap_llm_residency() const { return false; }
    virtual std::optional<std::vector<LlmReplica>> llm_replica_state() const { return std::nullopt; }
    // Configure before serving requests; the application owns auxiliary pools.
    virtual void set_llm_reclaimer(std::function<void(double)>) {}
    // Detach loaded text/vision engines. The callback restores that residency
    // when requested; embedding engines are independent of this handoff.
    virtual std::function<void(bool)> unload_llms_for_media() { return {}; }
    virtual std::function<void(bool)> unload_llms_for_video() { return unload_llms_for_media(); }
    // Return true only when an idle resident model was actually detached.
    virtual bool unload_aux_model(const std::string &) { return false; }
    virtual void warm_aux_model(const std::string &) {}
    virtual bool embedding_slot_state(RouterSlotState *) const { return false; }
    virtual std::string embedding_pool_json() const { return ""; }
    virtual bool warmup(std::string *error_out = nullptr) {
        (void)error_out;
        return true;
    }
};

class EchoInferenceBackend final : public InferenceBackend {
public:
    explicit EchoInferenceBackend(std::string default_model);

    std::string chat(const std::string &request_body) override;
    std::string completion(const std::string &request_body) override;
    std::string embedding(const std::string &request_body) override;

private:
    std::string default_model_;
};

class LlamaCppInferenceBackend final : public InferenceBackend {
public:
    explicit LlamaCppInferenceBackend(std::string default_model, std::shared_ptr<ResourceManager> resources = {});
    ~LlamaCppInferenceBackend() override;

    static bool should_use_from_env(const ezl_env_config &config);

    std::string chat(const std::string &request_body) override;
    std::string completion(const std::string &request_body) override;
    std::string embedding(const std::string &request_body) override;
    std::unique_ptr<RealtimeEngineLease> open_realtime(const std::string &model, const std::string &replica) override;
    std::string chat_for_replica(const std::string &body, const std::string &replica) override;
    std::string completion_for_replica(const std::string &body, const std::string &replica) override;
    InferenceResult text_result(const std::string &body, const std::string &replica, bool chat_mode) override;
    InferenceResult lazy_text_result(const std::string &body, const std::string &replica, bool chat_mode) override;
    InferenceResult pipeline_text_result(const std::string &body, const std::string &replica, bool chat_mode,
        const struct LlmInputOperations &input) override;
    std::string loaded_models_json() const override;
    std::string observed_loaded_models_json() const override;
    std::string model_lifecycle_json() const override;
    std::shared_ptr<ResourceManager> resource_manager() const override { return resources_; }
    int llm_context_size() const override;
    std::optional<int64_t> llm_parallel_count() const override;
    std::map<std::string, int64_t> loaded_model_contexts() const override;
    std::map<std::string, std::string> loaded_model_quants() const override;
    bool loaded_vision_model() const override;
    bool swap_llm_residency() const override { return swap_llm_residency_; }
    std::optional<std::vector<LlmReplica>> llm_replica_state() const override;
    void set_llm_reclaimer(std::function<void(double)> reclaim) override { llm_reclaimer_ = std::move(reclaim); }
    bool warmup(std::string *error_out = nullptr) override;
    std::function<void(bool)> unload_llms_for_media() override;
    std::function<void(bool)> unload_llms_for_video() override;
    bool unload_aux_model(const std::string &model) override;
    void warm_aux_model(const std::string &model) override;
    bool embedding_slot_state(RouterSlotState *state) const override;
    std::string embedding_pool_json() const override;

private:
    struct Server {
        std::string model_id;
        std::string public_model;
        std::string model_path;
        std::string projector_path;
        std::string url;
        size_t config_index = 0;
        int port = 0;
        int pid = -1;
        bool owns_process = false;
        bool started = false;
        bool embedding_mode = false;
        bool embedding_acquire = false;
        bool llm_acquire = false;
        bool llm_loaded = false;
        std::optional<double> llm_request_estimate_gb;
        std::optional<int64_t> llm_parallel;
        bool embedding_loaded = false;
        std::optional<int> embedding_main_gpu;
        int active_requests = 0;
        int context_size = 0;
        int64_t lifecycle_context = 0;
        int configured_gpu_layers = -1000000;
        int current_gpu_layers = -1000000;
        int retry_gpu_layers = -1000000;
        int64_t known_load_context = 0;
        std::optional<int64_t> known_gpu_layers;
        int total_layers = 0;
        bool cpu_only_retry = false;
        std::string embedding_device = "llama.cpp";
        double embedding_vram_gb = 0.0;
        std::optional<double> llm_vram_load_delta_gb;
        double last_used = 0.0;
        std::string runtime_json = "{}";
        std::string output_tokens = "8192";
    };

    struct Observation {
        std::deque<Server> servers;
        LlmCatalog catalog;
        std::vector<size_t> load_order;
        std::string active;
        std::map<std::string, double> estimates;
        bool swap = false;
    };
    // Observers use the last coherent metadata while process startup/shutdown
    // holds mutex_. Publish immediately before those potentially long waits.
    mutable std::shared_ptr<const Observation> observation_;
    std::shared_ptr<const Observation> publish_observation_locked() const;
    std::shared_ptr<const Observation> observe() const;
    std::string loaded_models_from_observation(const std::shared_ptr<const Observation> &snapshot) const;

    std::string default_model_;
    std::map<std::string, double> llm_vram_estimates_;
    std::string server_bin_;
    std::string api_key_;
    // References used by active requests remain valid as the embedding pool grows.
    std::deque<Server> servers_;
    Server embedding_template_;
    std::unique_ptr<EmbeddingPool> embedding_pool_;
    std::shared_ptr<ResourceManager> resources_;
    ModelLifecycle lifecycle_;
    LlmCatalog llm_catalog_;
    std::string optimal_context_;
    std::vector<LlmReplica> llm_replicas_;
    mutable std::mutex replica_state_mutex_;
    std::vector<size_t> llm_load_order_;
    std::string active_llm_;
    bool swap_llm_residency_ = false;
    std::function<void(double)> llm_reclaimer_;
    // Lock before mutex_. Auxiliary cleanup releases mutex_ while retaining
    // this gate, so another LLM load cannot enter during reclamation.
    std::mutex llm_transition_mutex_;
    size_t llm_server_count_ = 0;
    size_t embedding_server_begin_ = 0;
    mutable std::mutex mutex_;
    std::vector<std::pair<std::weak_ptr<void>, std::function<void()>>> streams_;
    std::unordered_map<std::string, bool> vision_model_cache_;

    Server *select_server_locked(const std::string &request_body, bool embedding_request, const std::string &replica);
    bool ensure_server_locked(Server *server, std::string *error_out, std::optional<int64_t> requested_context = {});
    bool load_server_locked(Server *server, std::string *error_out, std::optional<int64_t> requested_context);
    bool load_server_once_locked(Server *server, std::string *error_out, std::optional<int64_t> requested_context,
        std::optional<int64_t> gpu_layers = {}, std::optional<int64_t> ubatch = {}, const HttpDeadline *probe_deadline = nullptr);
    void note_llm_loaded_locked(Server *server);
    void publish_llm_replica_locked(const Server &server);
    void unload_other_llms_for_swap_locked(Server *target);
    double estimate_configured_llm_locked(Server &server, int64_t context);
    void prepare_request_llm_locked(Server *server, int64_t context, std::unique_lock<std::mutex> &lock);
    ModelType llm_resource_type(const Server &server) const;
    InferenceResult forward(const std::string &path, const std::string &request_body, const std::string &replica = {},
        bool model_profile = true, std::optional<int64_t> context = {});
    void stop_owned_server_locked(Server *server);
    int64_t configured_load_context(const Server &server) const;
    void stop_owned_servers();
    std::function<void(bool)> make_media_handoff(bool video);
    EmbeddingPool::Handle load_embedding_instance(size_t slot_index);
    void destroy_embedding_instance(EmbeddingPool::Handle handle);
    void register_embedding_resources(const EmbeddingPool::Snapshot &pool);
};

class NativeServer {
public:
    NativeServer(ezl_env_config config, std::unique_ptr<InferenceBackend> backend, std::shared_ptr<ResourceManager> resources = {},SttPoolFactory stt_factory = {},TtsPoolFactory tts_factory = {},TtsRequestOperations tts_request = {},std::optional<bool> precache_done = {});
    ~NativeServer();

    NativeServer(const NativeServer &) = delete;
    NativeServer &operator=(const NativeServer &) = delete;

    bool start(const std::string &host, int port, std::string *error_out = nullptr);
    void stop();
    bool running() const;
    // Startup-only: prepare subprocess media backends after speech/embeddings.
    void preload_media();
    // Configure before starting request handling.
    void set_wakeword_factory(WakewordModelFactory factory) { wakeword_factory_ = std::move(factory); }
    void set_wakeword_trainer(std::function<std::shared_ptr<WakewordTrainableModel>(const std::string &)> factory) { wakeword_trainer_ = std::move(factory); }
    std::optional<SttPoolSnapshot> stt_pool_snapshot() const { return stt_service_ ? std::optional<SttPoolSnapshot>(stt_service_->snapshot()) : std::nullopt; }
    std::optional<TtsPoolSnapshot> tts_pool_snapshot() const { return tts_service_ ? std::optional<TtsPoolSnapshot>(tts_service_->snapshot()) : std::nullopt; }
    std::optional<VoicePermitSnapshot> tts_permits() const { return tts_service_ ? std::optional<VoicePermitSnapshot>(tts_service_->permits()) : std::nullopt; }

    HttpResponse handle_request(
        const std::string &method,
        const std::string &path,
        const std::string &authorization,
        const std::string &body,
        const std::string &content_type = "",
        const std::string &client_host = "",
        const std::string &host_header = "",
        const std::unordered_map<std::string, std::string> &request_headers = {},
        bool allow_deferred_streaming = false
    );

private:
    WakewordModelFactory wakeword_factory_;
    std::function<std::shared_ptr<WakewordTrainableModel>(const std::string &)> wakeword_trainer_;
    std::mutex wakeword_jobs_mutex_;
    std::unique_ptr<WakewordJobs> wakeword_jobs_;
    WakewordJobs &wakeword_jobs_manager();
    std::mutex wakeword_models_mutex_;
    std::map<std::string, std::shared_ptr<WakewordModel>> wakeword_loaded_models_;
    HttpResponse predict_wakeword(const std::string &word, const std::string &directory, const std::string &audio);
    std::unique_ptr<SttService> stt_service_;
    std::unique_ptr<TtsService> tts_service_;
    int64_t tts_pool_configured_ = 1;
    int embedding_pool_configured_ = 1;
    std::function<void(bool)> tts_restore_llm_;
    bool native_tts_handoff() const;
    bool native_tts_preload() const;
    bool local_native_speech(const std::string &body,bool streaming,bool deferred,HttpResponse *out);
    std::shared_ptr<VoicePermitQueue> voice_handoff_ = std::make_shared<VoicePermitQueue>(1,true);
    std::function<void(bool)> stt_restore_llm_;
    bool native_stt_handoff() const;
    bool native_stt_preload() const;
    bool local_native_transcription(const std::string &body,const std::string &content_type,HttpResponse *out,bool translate);
    bool local_native_subtitles(const std::string &body,const std::string &content_type,HttpResponse *out);
    std::string direct_llm_chat(const std::string &body);
    struct ScopedCapabilityUse {
        const NativeServer *server = nullptr;
        std::string capability;

        ScopedCapabilityUse(const NativeServer *server_in, std::string capability_in);
        ~ScopedCapabilityUse();
    };

    struct RouterWorker {
        struct ActiveRequest {
            std::string token;
            std::string capability;
            std::string model;
            std::chrono::steady_clock::time_point expires_at;
        };
        struct RecentError {
            double ts = 0.0;
            std::string kind;
            int status = 0;
            std::string path;
            std::string message;
        };

        std::string worker_id;
        std::string label;
        std::string url;
        std::string api_key;
        std::vector<std::string> capabilities;
        std::vector<std::string> models;
        std::string capabilities_json = "[]";
        std::string models_json = "[]";
        double free_vram_gb = 0.0;
        double total_vram_gb = 0.0;
        double free_ram_gb = 0.0;
        double total_ram_gb = 0.0;
        int queue_depth = 0;
        int queue_capacity = 1;
        int in_flight = 0;
        std::map<std::string, RouterSlotState> cap_slots;
        std::map<std::string, RouterSlotState> model_slots;
        std::vector<std::string> cap_slot_order;
        std::vector<std::string> model_slot_order;
        std::string gpus_json = "[]";
        int best_tier = 2;
        std::string model_context_json = "{}";
        std::string model_quant_json = "{}";
        std::string cap_models_json = "{}";
        std::string version;
        std::string extra_json = "{}";
        double registered_at = 0.0;
        double last_heartbeat = 0.0;
        int connection_failures = 0;
        int total_errors = 0;
        double circuit_open_until = 0.0;
        std::string drain_owner;
        double drain_until = 0.0;
        std::vector<ActiveRequest> active_requests;
        std::vector<RecentError> recent_errors;
    };

    struct UsageLlmStats {
        int requests = 0;
        int prompt_tokens = 0;
        int completion_tokens = 0;
        double prompt_tps_sum = 0.0;
        int prompt_tps_n = 0;
        double predicted_tps_sum = 0.0;
        int predicted_tps_n = 0;
    };

    struct UsageHistoryEntry {
        double ts = 0.0;
        std::string worker;
        std::string model;
        int prompt_tokens = 0;
        int completion_tokens = 0;
        double prompt_ms = 0.0;
        double predicted_ms = 0.0;
        double total_ms = 0.0;
        double prompt_tps = 0.0;
        double predicted_tps = 0.0;
        std::string raw_json;
    };

    struct RouterTunnelDisconnect {
        double at = 0.0;
        double duration = 0.0;
        std::string reason;
    };

    struct RouterTunnelState {
        bool connected = false;
        int fd = -1;
        double connected_at = 0.0;
        double last_recv = 0.0;
        std::string last_close_reason;
        int connect_count = 0;
        uint64_t generation = 0;
        std::vector<RouterTunnelDisconnect> disconnect_history;
    };

    struct RouterTunnelPending {
        bool started = false;
        bool headers_received = false;
        bool done = false;
        int status = 0;
        std::string content_type;
        std::string body;
        std::vector<std::string> chunks;
        std::string error;
        std::string tunnel_worker_id;
        uint64_t generation = 0;
    };

    ezl_env_config config_;
    std::string worker_openapi_json_;
    std::unique_ptr<InferenceBackend> backend_;
    std::shared_ptr<ResourceManager> resources_;
    ezl_queue *queue_ = nullptr;
    struct QueuedForwardResponse {
        std::mutex mutex;
        std::optional<HttpResponse> response;
        std::optional<std::string> local_body;
        std::shared_ptr<LlmInferenceStream> stream;
        bool source_response = false;
        bool fallback_dictionary = false;
        bool internal = false;
        std::exception_ptr failure;
    };
    std::mutex queued_forward_mutex_;
    std::unordered_map<std::string, std::shared_ptr<QueuedForwardResponse>> queued_forward_responses_;
    std::vector<std::pair<std::weak_ptr<void>, std::function<void()>>> local_streams_;
    std::atomic<bool> running_{false};
    int server_fd_ = -1;
    std::thread accept_thread_;
    std::mutex client_mutex_;
    std::condition_variable client_cv_;
    std::set<int> client_fds_;
    mutable std::mutex router_mutex_;
    std::unordered_map<std::string, RouterWorker> router_workers_;
    std::vector<std::string> router_worker_order_;
    struct RouterDrain { std::string label; std::string owner; double until; };
    std::vector<RouterDrain> router_drains_;
    bool router_drains_loaded_ = false;
    std::string router_drain_path_;
    double router_reservation_ttl_ = 15.0;
    std::unordered_map<std::string, RouterTunnelState> router_tunnels_;
    uint64_t router_tunnel_generation_ = 0;
    std::mutex router_tunnel_pending_mutex_;
    std::condition_variable router_tunnel_pending_cv_;
    std::unordered_map<std::string, RouterTunnelPending> router_tunnel_pending_;
    mutable std::mutex usage_mutex_;
    mutable bool usage_loaded_ = false;
    mutable std::map<std::string, std::map<std::string, std::string>> usage_loaded_worker_values_;
    mutable std::map<std::string, std::map<std::string, UsageLlmStats>> usage_llm_;
    mutable std::map<std::string, std::map<std::string, int>> usage_caps_;
    mutable std::vector<std::string> usage_worker_order_;
    mutable std::map<std::string, std::vector<std::string>> usage_section_order_;
    mutable std::map<std::string, std::vector<std::string>> usage_llm_model_order_;
    mutable std::vector<UsageHistoryEntry> usage_history_;
    mutable bool usage_dirty_ = false;
    std::thread usage_flush_thread_;
    std::atomic<bool> worker_heartbeat_running_{false};
    std::atomic<bool> worker_heartbeat_cancelled_{false};
    bool worker_heartbeat_done_ = true; // guarded by worker_heartbeat_mutex_
    std::thread worker_heartbeat_thread_;
    mutable std::mutex worker_heartbeat_mutex_;
    std::condition_variable worker_heartbeat_cv_;
    mutable WorkerHeartbeatSignal worker_heartbeat_signal_;
    std::string worker_heartbeat_id_;
    std::string worker_heartbeat_url_;
    std::string worker_heartbeat_router_url_;
    std::string worker_heartbeat_api_key_;
    std::string worker_heartbeat_label_;
    int worker_heartbeat_port_ = 8091;
    double worker_heartbeat_interval_ = 10.0;
    std::atomic<bool> worker_tunnel_running_{false};
    std::thread worker_tunnel_thread_;
    std::string worker_tunnel_ws_url_;
    std::string worker_tunnel_local_url_;
    mutable std::mutex fallback_mutex_;
    mutable std::once_flag tts_voice_once_, tts_router_once_, tts_general_once_;
    mutable std::shared_ptr<TtsRemoteClient> tts_voice_client_, tts_router_client_, tts_general_client_;
    struct VoiceClientSettings { std::string raw, url, api_key; bool dedicated = false; };
    mutable std::once_flag voice_client_once_;
    mutable VoiceClientSettings voice_client_settings_;
    const VoiceClientSettings &voice_client() const;
    mutable std::once_flag image_client_once_, text_client_once_, embedding_client_once_;
    mutable ServiceClientSettings image_client_settings_, text_client_settings_, embedding_client_settings_;
    const ServiceClientSettings &image_client() const;
    const ServiceClientSettings &text_client() const;
    const ServiceClientSettings &embedding_client() const;
    struct FallbackClientSettings { std::string url, api_key; };
    mutable std::once_flag fallback_client_once_;
    mutable FallbackClientSettings fallback_client_settings_;
    mutable std::once_flag router_fallback_once_;
    mutable FallbackClientSettings router_fallback_settings_;
    mutable std::mutex router_fallback_mutex_;
    mutable double router_fallback_checked_ = -1;
    mutable bool router_fallback_alive_ = false;
    mutable bool fallback_cache_valid_ = false;
    mutable double fallback_cache_last_check_ = 0.0;
    mutable bool fallback_cache_available_ = false;
    mutable std::string fallback_cache_reason_;
    mutable std::mutex voice_mutex_;
    std::mutex llm_audio_handoff_mutex_, llm_audio_mutex_;
    std::condition_variable llm_audio_cv_;
    int64_t llm_audio_active_ = 0, llm_audio_capacity_ = 1;
    std::atomic<bool> llm_audio_stopping_{false};
    mutable bool voice_cache_valid_ = false;
    mutable std::string voice_cache_url_;
    mutable double voice_cache_last_check_ = 0.0;
    mutable bool voice_cache_available_ = false;
    mutable std::mutex text_mutex_;
    mutable bool text_cache_valid_ = false;
    mutable std::string text_cache_url_;
    mutable double text_cache_last_check_ = 0.0;
    mutable bool text_cache_available_ = false;
    mutable std::mutex embedding_mutex_;
    mutable std::map<std::string, std::pair<double, bool>> embedding_availability_;
    mutable std::mutex image_mutex_;
    std::mutex music_mutex_;
    std::mutex video_mutex_, llm_handoff_mutex_;
    // Observers must not acquire video_mutex_: reservation includes restoration.
    std::atomic<bool> video_session_active_{false};
    std::unique_ptr<WanVideo> video_;
    std::unique_ptr<LtxVideo> ltx_video_;
    std::mutex image_generation_mutex_;
    std::unique_ptr<QwenImage> image_;
    bool music_initialized_ = false;
    std::string music_server_url_, music_model_, music_lm_model_, music_synth_model_;
    double music_timeout_ = 1800.0, music_poll_interval_ = 1.0;
    mutable bool image_cache_valid_ = false;
    mutable std::string image_cache_url_;
    mutable double image_cache_last_check_ = 0.0;
    mutable bool image_cache_available_ = false;
    mutable std::mutex resource_mutex_;
    std::condition_variable llm_idle_cv_;
    LlmSlotState llm_slots_;
    LlmCatalog llm_catalog_;
    std::vector<LlmReplica> llm_replicas_;
    bool llm_slots_stopping_ = false;
    bool llm_temporarily_unavailable_ = false;
    bool resource_video_loaded_ = false;
    std::string resource_video_device_;
    mutable std::map<std::string, int> resource_capability_in_flight_;
    mutable std::map<std::string, double> resource_capability_last_used_;

    static void queue_processor(const char *payload, char *result, size_t result_size, void *user_data);

    bool authorized(const std::string &authorization) const;
    bool authorized_for_worker(const std::string &authorization) const;
    bool authorized_for_worker_websocket(const std::string &authorization) const;
    HttpResponse method_not_allowed_or_not_found(const std::string &path) const;
    HttpResponse route_authorized(
        const std::string &method,
        const std::string &path,
        const std::string &body,
        const std::string &content_type,
        const std::string &request_base_url,
        const std::unordered_map<std::string, std::string> &request_headers,
        bool allow_deferred_streaming
    );
    HttpResponse docs_html() const;
    HttpResponse redoc_html() const;
    HttpResponse swagger_oauth2_redirect_html() const;
    HttpResponse openapi_json() const;
    HttpResponse enqueue_inference(const std::string &kind, const std::string &body, bool allow_deferred_streaming = false);
    HttpResponse queue_status() const;
    HttpResponse resources(bool sync_queue = true) const;
    void sync_text_queue_capacity() const;
    HttpResponse music_generation(const std::string &body, bool video);
    HttpResponse request_ace_music(const std::string &payload);
    void wait_for_llm_idle_for_music(bool reserve);
    void wait_for_llm_idle_for_media(const std::string &service, const char *timeout_env, bool reserve);
    HttpResponse local_video_generation(const std::string &body);
    HttpResponse local_music_video_generation(const std::string &payload);
    // Caller retains video_mutex_ and the required llm_handoff_mutex_ scope.
    bool run_video_session(const std::string &model, std::optional<int> secondary,
        bool resident_llm, bool unload, const std::function<void()> &operation,
        std::optional<LtxVideoSettings> hint = {});
    void load_video_backend(const std::string &model, const std::string &device,
                            std::optional<LtxVideoSettings> hint = {});
    HttpResponse local_image_generation(const std::string &body, bool edit);
    struct VideoAuxiliaryState {
        VideoAuxiliaryMask unloaded{}, native_owner{};
    };
    void unload_aux_models_for_video(bool unload_llm, VideoAuxiliaryState &state);
    void reclaim_auxiliary_memory_for_llm(double required_gb);
    void restore_aux_models_after_video(const VideoAuxiliaryState &state);
    HttpResponse video_generation(const std::string &body);
    bool forward_router_media_request(const std::string &body, HttpResponse *out, const std::string &path) const;
    HttpResponse generate_ace_music(const std::string &payload);
    bool forward_music_request(const std::string &payload, bool video, bool through_router, HttpResponse *out) const;
    std::string resolve_request_slot_model(const std::string &body) const;
    bool should_use_llm_fallback() const;
    std::string begin_llm_inference(const std::string &model, const std::string &body,
        const std::function<bool()> &cancelled = {});
    void end_llm_inference(const std::string &model, const std::string &replica);
    void resource_increment_capability_in_flight(const std::string &capability, int delta) const;
    HttpResponse models() const;
    HttpResponse serve_output_file(
        const std::string &path,
        const std::unordered_map<std::string, std::string> &request_headers,
        bool deferred = false
    ) const;
    HttpResponse audio_voices() const;
    bool voice_server_available_cached(const std::string &voice_server, const std::string &api_key) const;
    bool forward_voice_server_tts(
        const std::string &path,
        const std::string &body,
        HttpResponse *out,
        bool allow_deferred_streaming = false
    ) const;
    bool forward_voice_server_transcription(const std::string &body, const std::string &content_type, HttpResponse *out) const;
    std::shared_ptr<TtsRemoteClient> tts_remote_client(TtsProvider provider) const;
    HttpResponse routed_speech(const std::string &body, bool streaming, bool deferred);
    bool forward_fallback_tts(const std::string &body, HttpResponse *out) const;
    bool forward_fallback_transcription(const std::string &body, const std::string &content_type, HttpResponse *out) const;
    bool data_uri_audio_speech(const std::string &body, HttpResponse *out) const;
    std::string transcribe_llm_audio(const std::string &audio, const std::string &format, bool return_segments = false);
    bool append_voice_audio_to_llm_response(
        const std::string &kind,
        const std::string &request_body,
        std::string *response_body
    ) const;
    bool local_whisper_transcription(const std::string &body, const std::string &content_type, HttpResponse *out, bool translate = false) const;
    bool local_whisper_subtitles(const std::string &body, const std::string &content_type, HttpResponse *out) const;
    bool fallback_server_available_cached(const std::string &fallback_server, const std::string &api_key) const;
    const FallbackClientSettings &fallback_client() const;
    std::string post_inference_fallback(const std::string &body, bool chat_mode) const;
    std::pair<bool, std::string> fallback_availability(const std::string &fallback_server, const std::string &api_key) const;
    HttpResponse fallback_status() const;
    bool text_server_available_cached(const std::string &text_server, const std::string &api_key) const;
    bool forward_text_server_request(const std::string &path, const std::string &body, HttpResponse *out) const;
    bool forward_embedding_server_request(const std::string &body, HttpResponse *out, bool through_router = false) const;
    HttpResponse local_embeddings(const std::string &body);
    bool forward_fallback_llm(const std::string &path, const std::string &body, HttpResponse *out) const;
    // Text requests fall back to ROUTER_URL when FALLBACK_SERVER is unset.
    const FallbackClientSettings &router_fallback() const;
    const std::string &effective_llm_fallback_url() const;
    bool router_fallback_available() const;
    bool forward_llm_to_router(const std::string &path, const std::string &body, HttpResponse *out,
        std::shared_ptr<LlmInferenceStream> *stream) const;
    bool forward_fallback_embeddings(const std::string &body, HttpResponse *out) const;
    bool image_server_available_cached(const std::string &image_server, const std::string &api_key) const;
    bool forward_image_server_request(const std::string &path, const std::string &body, HttpResponse *out) const;
    bool forward_fallback_image_video_request(const std::string &path, const std::string &body, HttpResponse *out) const;
    HttpResponse upload_voice(const std::string &body, const std::string &content_type) const;
    HttpResponse router_register(const std::string &body, const std::string &client_host);
    HttpResponse router_heartbeat(const std::string &body);
    HttpResponse router_deregister(const std::string &body);
    HttpResponse router_drain(const std::string &body, bool resume);
    bool ensure_router_drains_loaded();
    void save_router_drains_locked() const;
    bool router_worker_draining(const RouterWorker &worker) const;
    int router_maintenance_busy(const RouterWorker &worker) const;
    int router_reserved(const RouterWorker &worker, const std::string &capability = "", const std::string &model = "") const;
    std::string router_effective_slots_json(const RouterWorker &worker, bool models) const;
    HttpResponse router_workers() const;
    HttpResponse router_health() const;
    HttpResponse router_dashboard() const;
    HttpResponse router_dashboard_html() const;
    HttpResponse router_tunnels() const;
    HttpResponse router_usage() const;
    HttpResponse router_history(int limit = 0) const;
    HttpResponse router_errors() const;
    std::string router_worker_public_json(const RouterWorker &worker) const;
    std::string router_worker_dashboard_json(const RouterWorker &worker) const;
    std::vector<RouterWorker> router_workers_in_registration_order_locked() const;
    uint64_t router_tunnel_attach(const std::string &worker_id, int fd);
    void router_tunnel_note_recv(const std::string &worker_id, uint64_t generation);
    void router_tunnel_detach(const std::string &worker_id, uint64_t generation, const std::string &reason);
    bool router_tunnel_connected(const std::string &worker_id) const;
    void router_tunnel_handle_message(const std::string &message);
    HttpResponse router_tunnel_request(
        const RouterWorker &worker,
        const std::string &method,
        const std::string &path,
        const std::string &api_key,
        const std::string &content_type,
        const std::string &body,
        bool stream,
        int timeout_ms = 0,
        bool allow_deferred_streaming = false
    );
    double router_ttl_seconds() const;
    bool has_alive_router_worker(const std::string &capability, const std::string &model = "") const;
    bool find_alive_router_worker(
        const std::string &capability,
        const std::string &model,
        RouterWorker *out,
        const std::vector<std::string> &exclude_worker_ids = {},
        bool allow_cross_model = true
    ) const;
    bool wait_for_alive_router_worker(
        const std::string &capability,
        const std::string &model,
        RouterWorker *out,
        const std::vector<std::string> &exclude_worker_ids = {}
    ) const;
    RouterSlotState router_slot_state(
        const RouterWorker &worker,
        const std::string &capability,
        const std::string &model
    ) const;
    int router_effective_busy(const RouterWorker &worker, const std::string &capability = "", const std::string &model = "") const;
    int router_capacity_for(const RouterWorker &worker, const std::string &capability = "", const std::string &model = "") const;
    int router_slots_left(const RouterWorker &worker, const std::string &capability = "", const std::string &model = "") const;
    int router_total_capacity(const RouterWorker &worker) const;
    int router_total_busy(const RouterWorker &worker) const;
    int router_total_slots_left(const RouterWorker &worker) const;
    double router_score(const RouterWorker &worker, const std::string &capability, const std::string &model = "") const;
    bool router_reserve_request(const std::string &worker_id, const std::string &capability, const std::string &model, bool require_capacity, std::string *token, std::string *error);
    void router_release_request(const std::string &worker_id, const std::string &token);
    void router_record_error(const std::string &worker_id, const std::string &kind, const std::string &path, const std::string &message, int status);
    std::string router_recent_errors_json(const RouterWorker &worker) const;
    std::string router_aggregate_errors_json(const std::vector<RouterWorker> &workers) const;
    HttpResponse router_proxy_request(
        const std::string &capability,
        const std::string &model,
        const std::string &path,
        const std::string &body,
        const std::string &content_type = "application/json",
        const std::string &request_base_url = "",
        int timeout_seconds = 0,
        bool allow_deferred_streaming = false
    );
    std::string usage_snapshot_json() const;
    std::string usage_history_json(int limit = 0) const;
    void ensure_usage_loaded_locked() const;
    std::string usage_snapshot_json_locked() const;
    std::string usage_history_json_locked(int limit = 0) const;
    void record_router_usage(const RouterWorker &worker, const std::string &capability, const std::string &model, const HttpResponse &response);
    HttpResponse persist_response_assets(const RouterWorker &worker, const HttpResponse &response, const std::string &request_base_url = "");
    void usage_flush_loop();
    void flush_usage_if_dirty() const;
    void mark_usage_dirty_locked() const;
    void persist_usage_locked() const;
    void start_worker_heartbeat(const std::string &host, int port);
    void stop_worker_heartbeat();
    void notify_worker_state_changed() const;
    void worker_heartbeat_loop();
    bool worker_router_post(const std::string &path, const std::string &body) const;
    std::string worker_registration_body() const;
    std::string worker_heartbeat_body() const;
    mutable std::atomic<bool> worker_vision_seen_{false};
    std::string worker_state_json() const;
    void start_worker_tunnel(int port);
    void stop_worker_tunnel();
    void worker_tunnel_loop();
    void worker_tunnel_handle_message(
        const std::string &message,
        const std::function<bool(const std::string &)> &send_text,
        const LlmInferenceStream::Interrupt &register_interrupt,
        const std::function<std::optional<std::pair<uint64_t, std::string>>()> &realtime_input = {},
        const std::function<uint64_t()> &realtime_read_ack = {}
    );
    void accept_loop();
    void handle_client(int client_fd);
    void handle_router_tunnel_websocket(int client_fd, const std::string &worker_id);
    void handle_tts_websocket(int client_fd,const std::string &initial);
    void handle_realtime_websocket(int client_fd, const std::string &query,
        const std::string &initial, const std::string &websocket_key);
};

// Initialize the worker's text and auxiliary models before opening listeners.
std::unique_ptr<NativeServer> create_native_server(ezl_env_config config,
    std::unique_ptr<InferenceBackend> backend, std::shared_ptr<ResourceManager> resources = {},
    SttPoolFactory stt_factory = {}, TtsPoolFactory tts_factory = {},
    std::optional<bool> precache_done = {});

std::string json_escape(const std::string &value);
std::string extract_json_string(const std::string &json, const std::string &key, const std::string &fallback);
bool extract_json_bool(const std::string &json, const std::string &key, bool fallback);

} // namespace ezlocalai

#endif
