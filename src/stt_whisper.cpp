#include "ezlocalai/stt_whisper.hpp"
#include "ezlocalai/whisper_model.hpp"
#ifdef EZLOCALAI_STT_TORCH
#include "ezlocalai/stt_torch.hpp"
#endif
#include <nlohmann/json.hpp>
#include <algorithm>
#include <filesystem>
#include <map>
#include <mutex>

namespace ezlocalai {
namespace {
using Json = nlohmann::json;
bool truth(const Json &value) {
    if (value.is_null()) return false; if (value.is_boolean()) return value.get<bool>();
    if (value.is_number()) return value.get<double>() != 0; if (value.is_string()) return !value.get_ref<const std::string &>().empty(); return !value.empty();
}
double number(const Json &value) { return value.is_boolean() ? double(value.get<bool>()) : value.get<double>(); }
int64_t integer(const Json &value) { return value.is_boolean() ? int64_t(value.get<bool>()) : value.get<int64_t>(); }
WhisperTranscriptionOptions settings(const Json &input,bool batched) {
    WhisperTranscriptionOptions out; out.vad_filter = input.contains("vad_filter") ? truth(input.at("vad_filter")) : batched; out.without_timestamps = batched;
    if (input.contains("task")) out.task = input.at("task").get<std::string>();
    if (input.contains("language") && !input.at("language").is_null()) out.language = input.at("language").get<std::string>();
    if (input.contains("initial_prompt") && !input.at("initial_prompt").is_null()) {
        const auto &prompt = input.at("initial_prompt"); if (prompt.is_string()) out.initial_prompt = prompt.get<std::string>(); else out.initial_prompt = prompt.get<std::vector<uint32_t>>();
    }
    if (input.contains("temperature")) {
        const auto &value = input.at("temperature"); out.temperatures.clear();
        if (value.is_array()) for (const auto &item : value) out.temperatures.push_back(number(item)); else out.temperatures.push_back(number(value));
    }
    if (input.contains("beam_size")) out.beam_size = integer(input.at("beam_size"));
    if (input.contains("condition_on_previous_text")) out.condition_on_previous_text = truth(input.at("condition_on_previous_text"));
    if (input.contains("word_timestamps")) out.word_timestamps = truth(input.at("word_timestamps"));
    return out;
}
SttInferenceResult result(const WhisperTranscription &transcription) {
    Json segments = Json::array();
    for (const auto &segment : transcription.segments) {
        Json words;
        if (segment.words) { words = Json::array(); for (const auto &word : *segment.words) words.push_back({{"word",word.word},{"start",word.start},{"end",word.end}}); }
        segments.push_back({{"text",segment.text},{"start",segment.start},{"end",segment.end},{"words",words}});
    }
    return {segments.dump(),Json({{"language",transcription.language}}).dump()};
}
std::shared_ptr<WhisperVadModel> vad_model(WhisperVadProfile profile,const std::string &directory) {
    static std::mutex mutex; static std::map<std::pair<int,std::string>,std::shared_ptr<WhisperVadModel>> models;
    std::lock_guard<std::mutex> lock(mutex); auto &model = models[{static_cast<int>(profile),directory}]; if (!model) model = std::make_shared<WhisperVadModel>(profile,directory); return model;
}
struct State {
    WhisperNativeEngine engine;
    WhisperEngineOperations operations;
    WhisperTokenizer tokenizer;
    SttWhisperConfiguration configuration;
    State(const std::string &directory,const std::string &device,const std::string &compute,int64_t threads,SttWhisperConfiguration config)
        : engine(directory,device,compute,threads),operations(engine.operations()),tokenizer(resolve_whisper_tokenizer(directory,operations.multilingual),operations.multilingual),configuration(std::move(config)) {
        operations.feature_configuration = whisper_feature_configuration(whisper_feature_kwargs(directory));
    }
    std::shared_ptr<WhisperVadModel> vad() const {
        return vad_model(configuration.profile == WhisperTranscriptionProfile::faster_whisper_1_0_3 ? WhisperVadProfile::faster_whisper_1_0_3 : WhisperVadProfile::faster_whisper_1_2_0,configuration.vad_directory);
    }
    WhisperTranscriptionOptions options(const Json &kwargs,bool batch) const {
        auto out = settings(kwargs,batch);
        if (kwargs.contains("vad_parameters") && !kwargs.at("vad_parameters").is_null()) {
            auto vad = whisper_vad_options(configuration.profile == WhisperTranscriptionProfile::faster_whisper_1_0_3 ? WhisperVadProfile::faster_whisper_1_0_3 : WhisperVadProfile::faster_whisper_1_2_0);
            const auto &values = kwargs.at("vad_parameters");
            if (values.contains("min_silence_duration_ms")) vad.min_silence_duration_ms = number(values.at("min_silence_duration_ms"));
            if (values.contains("speech_pad_ms")) vad.speech_pad_ms = number(values.at("speech_pad_ms")); out.vad_options = vad;
        }
        return out;
    }
};
std::vector<float> decode(const std::string &path,int sampling_rate) {
    try { return whisper_decode_audio(path,sampling_rate); }
    catch (const std::exception &error) {
        std::string message = error.what(); auto lower = message; std::transform(lower.begin(),lower.end(),lower.begin(),[](unsigned char c) { return std::tolower(c); });
        if (lower.find("invalid data found when processing input") != std::string::npos) throw SttError("InvalidDataError",message);
        if (message == "No audio stream") throw SttError("IndexError","tuple index out of range"); throw;
    }
}
struct NativeBackend : SttBackend { std::shared_ptr<State> state; };
}
SttRuntimeOperations stt_whisper_operations(SttWhisperConfiguration configuration,SttRuntimeOperations platform) {
#ifdef EZLOCALAI_STT_TORCH
    platform = stt_torch_operations(std::move(platform));
#endif
    if (!configuration.resolve_model) configuration.resolve_model = [profile = configuration.profile](const auto &name,const auto &root) { return resolve_whisper_model(name,root,profile); };
    platform.batched_available = configuration.profile == WhisperTranscriptionProfile::faster_whisper_1_2_0;
    platform.load = [configuration](const std::string &model,const std::string &root,const std::string &device,const std::string &compute,int64_t threads) {
        auto backend = std::make_shared<NativeBackend>(); backend->state = std::make_shared<State>(configuration.resolve_model(model,root),device,compute,threads,configuration);
        backend->transcribe = [state = backend->state](const std::string &path,const std::string &arguments) {
            const auto kwargs = Json::parse(arguments);
            if (kwargs.contains("batch_size")) throw SttError("TypeError","WhisperModel.transcribe() got an unexpected keyword argument 'batch_size'");
            const auto audio = decode(path,state->operations.feature_configuration->sampling_rate); const auto options = state->options(kwargs,false);
            return result(whisper_transcribe(audio,state->tokenizer,options,state->operations,state->configuration.profile,
                [state](const auto &input,const auto &vad) { return state->vad()->filter(input,vad); }));
        };
        return backend;
    };
    platform.batch = [](const std::shared_ptr<SttBackend> &model) {
        const auto native = std::dynamic_pointer_cast<NativeBackend>(model); if (!native) throw std::invalid_argument("Invalid native STT model handle");
        auto backend = std::make_shared<SttBackend>(); const auto state = native->state; auto pipeline = std::make_shared<WhisperBatchedPipeline>(state->operations);
        backend->transcribe = [state,pipeline](const std::string &path,const std::string &arguments) {
            const auto kwargs = Json::parse(arguments); const auto audio = decode(path,state->operations.feature_configuration->sampling_rate); WhisperBatchedOptions options; options.transcription = state->options(kwargs,true);
            if (kwargs.contains("batch_size")) options.batch_size = integer(kwargs.at("batch_size"));
            return result(pipeline->transcribe(audio,state->tokenizer,options,[state](const auto &input,const auto &vad) { return state->vad()->speech_timestamps(input,vad); }));
        };
        return backend;
    };
    return platform;
}
SttPoolOperations stt_whisper_pool_operations(SttWhisperConfiguration configuration,
    std::shared_ptr<ResourceManager> resources,SttRuntimeOperations platform) {
    const auto runtime = stt_whisper_operations(std::move(configuration),std::move(platform));
    SttPoolOperations pool;
    pool.load = [runtime](const std::string &name) {
        const auto instance = std::make_shared<SttRuntime>(name,runtime);
        auto handle = std::make_shared<SttPoolModel>();
        handle->device = [instance] { return instance->device(); };
        handle->close = [instance] { instance->close(); };
        handle->transcribe = [instance](const std::string &request) { return instance->transcribe(request); };
        handle->transcribe_bytes = [instance](const std::string &request,const std::string &audio) { return instance->transcribe_bytes(request,audio); };
        return handle;
    };
    pool.collect = runtime.collect; pool.cuda_available = runtime.cuda_available;
    pool.synchronize = runtime.synchronize; pool.empty_cache = runtime.empty_cache; pool.ipc_collect = runtime.ipc_collect;
    return stt_pool_resources(std::move(pool),std::move(resources));
}
}
