#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

namespace ezlocalai {
struct SttError : std::runtime_error {
    std::string category;
    bool runtime_family = false;
    SttError(std::string category,std::string message,bool runtime_family = false) : std::runtime_error(std::move(message)), category(std::move(category)),runtime_family(runtime_family) {}
};
struct SttInferenceResult { std::string segments = "[]", info = "null"; };
struct SttBackend {
    virtual ~SttBackend() = default;
    std::function<SttInferenceResult(const std::string &,const std::string &)> transcribe;
};
// Native model and system boundaries are injectable so the wrapper's ordering,
// fallback and cleanup behavior can be compared with unchanged STT.py methods.
struct SttRuntimeOperations {
    bool batched_available = true;
    std::function<bool()> cuda_available, cudnn_available;
    std::function<double()> cuda_total_bytes, cuda_allocated_bytes;
    // The probe result stays alive through empty_cache, as the source's local
    // convolution output retains its autograd graph until the probe returns.
    std::function<std::shared_ptr<void>()> test_cudnn;
    std::function<void()> collect, empty_cache, synchronize, ipc_collect;
    std::function<int64_t()> cpu_count;
    std::function<std::shared_ptr<SttBackend>(const std::string &,const std::string &,const std::string &,const std::string &,int64_t)> load;
    std::function<std::shared_ptr<SttBackend>(const std::shared_ptr<SttBackend> &)> batch;
    // Both may be omitted to use native PortAudio initialization/termination.
    // Custom audio boundaries must supply both callbacks.
    std::function<std::shared_ptr<void>()> open_audio;
    std::function<void(const std::shared_ptr<void> &)> terminate_audio;
    std::function<std::string()> uuid_hex, working_directory;
    std::function<void(const std::string &,const std::string &)> write;
    std::function<bool(const std::string &)> exists;
    std::function<void(const std::string &)> remove;
    std::function<std::string(const std::string &,const std::string &,const std::string &,const std::string &)> diarize;
};
bool stt_gpu_error(const std::string &message);
int64_t stt_cpu_threads(const std::function<int64_t()> &cpu_count);
class SttRuntime {
public:
    SttRuntime(std::string model,SttRuntimeOperations operations);
    // JSON keyword values preserve the source wrapper's raw input types.
    // Returns a JSON string or the source-shaped text/segments/language object.
    std::string transcribe(const std::string &request);
    std::string transcribe_bytes(const std::string &request,const std::string &audio);
    void close();
    const std::string &device() const { return device_; }
    const std::string &compute_type() const { return compute_; }
    const std::string &model_name() const { return model_; }
    bool has_model() const { return bool(whisper_); }
    bool has_batch() const { return bool(batched_); }
    bool has_audio() const { return bool(audio_); }
private:
    std::string transcribe_impl(const std::string &request,const std::string *audio);
    std::string model_,device_,compute_;
    SttRuntimeOperations operations_;
    std::shared_ptr<SttBackend> whisper_,batched_;
    std::shared_ptr<void> audio_;
    bool whisper_present_ = false, batched_present_ = false;
    double available_vram();
    bool cudnn_works();
    void create_batch();
};
}
