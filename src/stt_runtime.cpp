#include "ezlocalai/stt_runtime.hpp"
#include "ezlocalai/stt_diarization.hpp"
#include "ezlocalai/stt_audio.hpp"
#include "ezlocalai/stt_input.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/llm_response.hpp"
#include "ezlocalai/llm_startup.hpp"
#include <nlohmann/json.hpp>
#include <openssl/rand.h>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <thread>
#include <unistd.h>

namespace ezlocalai {
namespace {
using Json = nlohmann::ordered_json;
std::string lower(std::string value) { for (auto &c : value) if (c >= 'A' && c <= 'Z') c += 'a' - 'A'; return value; }
bool truth(const Json &value) {
    if (value.is_null()) return false; if (value.is_boolean()) return value.get<bool>();
    if (value.is_number()) return value.get<double>() != 0; if (value.is_string()) return !value.get_ref<const std::string &>().empty(); return !value.empty();
}
std::string type(const Json &value) {
    if (value.is_null()) return "NoneType"; if (value.is_boolean()) return "bool"; if (value.is_number_integer()) return "int";
    if (value.is_number()) return "float"; if (value.is_string()) return "str"; if (value.is_array()) return "list"; return "dict";
}
std::string string(const Json &value) {
    if (value.is_string()) return value.get<std::string>(); if (value.is_null()) return "None";
    if (value.is_boolean()) return value.get<bool>() ? "True" : "False"; return value.dump();
}
SttError current_error() {
    try { throw; }
    catch (const SttError &error) { return error; }
    catch (const LlmResponseError &error) { return {error.category,error.message}; }
    catch (const std::invalid_argument &error) { return {"ValueError",error.what()}; }
    catch (const std::out_of_range &error) { return {"IndexError",error.what()}; }
    catch (const std::exception &error) { return {"RuntimeError",error.what()}; }
}
bool runtime_error(const SttError &error) {
    const auto &type = error.category;
    return error.runtime_family || type == "RuntimeError" || type == "OutOfMemoryError" || type == "OSError" || type == "FileNotFoundError" || type == "PermissionError" || type == "IsADirectoryError";
}
std::string stripped(const Json &value) {
    if (!value.is_string()) throw SttError("AttributeError","'" + type(value) + "' object has no attribute 'strip'");
    return strip_config_value(value.get<std::string>());
}
const Json &attribute(const Json &value,const char *key) {
    if (!value.contains(key)) throw SttError("AttributeError",std::string("'types.SimpleNamespace' object has no attribute '") + key + "'");
    return value.at(key);
}
void file_error(int error,const std::string &path,bool with_path = true) {
    const auto category = error == ENOENT ? "FileNotFoundError" : error == EACCES ? "PermissionError" : error == EISDIR ? "IsADirectoryError" : error == ENOTDIR ? "NotADirectoryError" : "OSError";
    throw SttError(category,"[Errno " + std::to_string(error) + "] " + std::strerror(error) + (with_path ? ": " + python_string_repr(path) : ""));
}
void defaults(SttRuntimeOperations &ops) {
    if (!ops.diarize) ops.diarize = stt_diarize;
    ops = stt_portaudio_operations(std::move(ops));
    if (!ops.cuda_available) ops.cuda_available = [] { return false; };
    if (!ops.cudnn_available) ops.cudnn_available = [] { return false; };
    if (!ops.cuda_total_bytes) ops.cuda_total_bytes = [] { return 0.0; };
    if (!ops.cuda_allocated_bytes) ops.cuda_allocated_bytes = [] { return 0.0; };
    if (!ops.test_cudnn) ops.test_cudnn = [] { return std::shared_ptr<void>{}; };
    for (auto *action : {&ops.collect,&ops.empty_cache,&ops.synchronize,&ops.ipc_collect}) if (!*action) *action = [] {};
    if (!ops.cpu_count) ops.cpu_count = [] { const auto count = std::thread::hardware_concurrency(); if (!count) throw SttError("NotImplementedError","cannot determine number of cpus"); return count; };
    if (!ops.uuid_hex) ops.uuid_hex = [] {
        unsigned char bytes[16]; if (RAND_bytes(bytes,sizeof(bytes)) != 1) throw SttError("RuntimeError","Cannot generate STT audio filename");
        bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128; std::string value; const char *digits = "0123456789abcdef";
        for (auto byte : bytes) { value += digits[byte >> 4]; value += digits[byte & 15]; } return value;
    };
    if (!ops.working_directory) ops.working_directory = [] { return std::filesystem::current_path().string(); };
    if (!ops.write) ops.write = [](const std::string &path,const std::string &bytes) {
        if (path.find('\0') != std::string::npos) throw SttError("ValueError","embedded null byte");
        auto *file = std::fopen(path.c_str(),"wb"); if (!file) file_error(errno,path);
        const auto written = std::fwrite(bytes.data(),1,bytes.size(),file); const auto write_error = errno;
        const auto closed = std::fclose(file); const auto close_error = errno;
        if (written != bytes.size()) file_error(write_error ? write_error : EIO,path,false);
        if (closed) file_error(close_error ? close_error : EIO,path,false);
    };
    if (!ops.exists) ops.exists = [](const std::string &path) { std::error_code ignored; return std::filesystem::exists(path,ignored); };
    if (!ops.remove) ops.remove = [](const std::string &path) { if (::unlink(path.c_str())) file_error(errno,path); };
    if (!ops.load) throw std::invalid_argument("Native STT model loader is not configured");
    if (ops.batched_available && !ops.batch) throw std::invalid_argument("Native STT batch factory is not configured");
    if (!ops.open_audio) throw std::invalid_argument("Native STT audio initializer is not configured");
    if (!ops.terminate_audio) throw std::invalid_argument("Native STT audio termination is not configured");
}
bool wants_words(Json value) {
    if (!truth(value)) value = Json::array();
    if (value.is_string()) value = Json::array({value});
    if (!value.is_array() && !value.is_object()) throw SttError("TypeError","'" + type(value) + "' object is not iterable");
    if (value.is_object()) { for (const auto &item : value.items()) if (lower(item.key()) == "word") return true; }
    else for (const auto &item : value) if (item.is_string() && lower(item.get<std::string>()) == "word") return true;
    return false;
}
void check_backend(const std::shared_ptr<SttBackend> &backend,bool present,const char *name) {
    if (!present) throw SttError("AttributeError",std::string("'STT' object has no attribute '") + name + "'");
    if (!backend) throw SttError("AttributeError","'NoneType' object has no attribute 'transcribe'");
}
SttInferenceResult infer(const std::shared_ptr<SttBackend> &backend,const std::string &path,const Json &kwargs) {
    return backend->transcribe(path,kwargs.dump());
}
}
bool stt_gpu_error(const std::string &message) {
    const auto text = lower(message); for (const auto *part : {"out of memory","cuda","cudnn","libcudnn","invalid handle"}) if (text.find(part) != std::string::npos) return true; return false;
}
int64_t stt_cpu_threads(const std::function<int64_t()> &cpu_count) {
    try { return std::max<int64_t>(2,std::min<int64_t>(8,cpu_count() / 2)); } catch (...) { return 4; }
}
double SttRuntime::available_vram() {
    if (operations_.cuda_available()) try { return (operations_.cuda_total_bytes() - operations_.cuda_allocated_bytes()) / (1024 * 1024); } catch (...) { return 0; }
    return 0;
}
bool SttRuntime::cudnn_works() {
    if (!operations_.cuda_available()) return false;
    try { if (!operations_.cudnn_available()) return false; const auto output = operations_.test_cudnn(); operations_.empty_cache(); return true; } catch (...) { return false; }
}
void SttRuntime::create_batch() { batched_ = operations_.batched_available ? operations_.batch(whisper_) : whisper_; batched_present_ = true; }
SttRuntime::SttRuntime(std::string model,SttRuntimeOperations operations) : model_(std::move(model)),operations_(std::move(operations)) {
    defaults(operations_); const auto memory = available_vram(); const auto cudnn = cudnn_works();
    if (operations_.cuda_available() && memory >= 1000 && cudnn) { device_ = "cuda"; compute_ = "float16"; }
    else { device_ = "cpu"; compute_ = "int8"; operations_.cuda_available(); }
    const auto threads = device_ == "cpu" ? stt_cpu_threads(operations_.cpu_count) : 0;
    try { whisper_ = operations_.load(model_,"models",device_,compute_,threads); whisper_present_ = true; create_batch(); }
    catch (...) {
        const auto error = current_error(); if (!runtime_error(error) || !stt_gpu_error(error.what()) || device_ != "cuda") throw error;
        operations_.collect(); if (operations_.cuda_available()) operations_.empty_cache();
        device_ = "cpu"; compute_ = "int8"; whisper_ = operations_.load(model_,"models",device_,compute_,stt_cpu_threads(operations_.cpu_count)); whisper_present_ = true; create_batch();
    }
    audio_ = operations_.open_audio();
}
void SttRuntime::close() {
    auto batch = std::move(batched_); auto model = std::move(whisper_); auto audio = std::move(audio_);
    whisper_present_ = batched_present_ = true;
    if (audio) try { operations_.terminate_audio(audio); } catch (...) {}
    batch.reset(); model.reset(); audio.reset(); operations_.collect();
    if (operations_.cuda_available()) {
        try { operations_.synchronize(); } catch (...) {}
        operations_.empty_cache(); try { operations_.ipc_collect(); } catch (...) {}
    }
}
std::string SttRuntime::transcribe(const std::string &request) {
    return transcribe_impl(request,nullptr);
}
std::string SttRuntime::transcribe_bytes(const std::string &request,const std::string &audio) {
    return transcribe_impl(request,&audio);
}
std::string SttRuntime::transcribe_impl(const std::string &request,const std::string *audio) {
    const auto input = Json::parse(request);
    std::string extension, bytes;
    try { extension = stt_audio_extension(input.value("audio_format",Json("wav")).dump()); } catch (...) { throw current_error(); }
    const auto filename = operations_.uuid_hex() + "." + extension;
    const auto path = (std::filesystem::path(operations_.working_directory())/"outputs"/filename).string();
    try { bytes = audio ? stt_base64_decode_bytes(*audio) : stt_base64_decode(input.value("base64_audio",Json()).dump()); } catch (...) { throw current_error(); }
    operations_.write(path,bytes); if (!operations_.exists(path)) throw SttError("RuntimeError","Failed to load audio.");
    const auto language = input.value("language",Json()), prompt = input.value("prompt",Json()), temperature = input.value("temperature",Json(0.0));
    const auto beam = input.value("beam_size",Json(1)), condition = input.value("condition_on_previous_text",Json(true));
    const std::string task = truth(input.value("translate",Json(false))) ? "translate" : "transcribe";
    bool words = false; SttInferenceResult inferred;
    const auto standard = [&] {
        check_backend(whisper_,whisper_present_,"w");
        return infer(whisper_,path,{{"task",task},{"vad_filter",true},{"vad_parameters",{{"min_silence_duration_ms",500},{"speech_pad_ms",200}}},
            {"initial_prompt",prompt},{"language",language},{"temperature",temperature},{"beam_size",beam},{"condition_on_previous_text",condition},{"word_timestamps",words}});
    };
    try {
        words = wants_words(input.value("timestamp_granularities",Json()));
        if (truth(input.value("use_batched",Json(false)))) {
            check_backend(batched_,batched_present_,"batched_model");
            if (!temperature.is_number() && !temperature.is_boolean()) throw SttError("TypeError","'>' not supported between instances of '" + type(temperature) + "' and 'int'");
            const auto positive = temperature.is_boolean() ? temperature.get<bool>() : temperature.get<double>() > 0;
            inferred = infer(batched_,path,{{"task",task},{"language",language},{"initial_prompt",prompt},{"beam_size",beam},
                {"temperature",positive ? temperature : Json::array({0.0})},{"batch_size",input.value("batch_size",Json(8))},{"word_timestamps",words},
                {"vad_parameters",{{"min_silence_duration_ms",500},{"speech_pad_ms",200}}}});
        } else inferred = standard();
    } catch (...) {
        const auto error = current_error();
        if (runtime_error(error)) {
            if (stt_gpu_error(error.what()) && device_ == "cuda") {
                operations_.collect(); if (operations_.cuda_available()) operations_.empty_cache();
                whisper_.reset(); whisper_present_ = false; batched_.reset(); batched_present_ = false; operations_.collect(); if (operations_.cuda_available()) operations_.empty_cache();
                device_ = "cpu"; compute_ = "int8"; whisper_ = operations_.load(model_,"models",device_,compute_,stt_cpu_threads(operations_.cpu_count)); whisper_present_ = true; create_batch();
                inferred = standard();
            } else { operations_.remove(path); throw error; }
        } else {
            if (operations_.exists(path)) operations_.remove(path);
            if (error.category == "InvalidDataError" || lower(error.what()).find("invalid data found when processing input") != std::string::npos)
                throw SttError("ValueError","The uploaded file does not contain valid audio data");
            throw error;
        }
    }
    std::string text; Json segments = Json::array();
    for (const auto &segment : Json::parse(inferred.segments)) {
        const auto &part = attribute(segment,"text");
        if (!part.is_string()) throw SttError("TypeError","can only concatenate str (not \"" + type(part) + "\") to str");
        text += part.get<std::string>(); Json entry{{"id",segments.size()},{"start",attribute(segment,"start")},{"end",attribute(segment,"end")},{"text",stripped(part)}};
        const auto word_data = segment.value("words",Json());
        if (truth(word_data)) {
            if (!word_data.is_array()) throw SttError("TypeError","'" + type(word_data) + "' object is not iterable");
            entry["words"] = Json::array();
            for (const auto &word : word_data) { const auto value = stripped(word.value("word",Json(""))); if (!value.empty()) entry["words"].push_back({{"word",value},{"start",word.value("start",Json())},{"end",word.value("end",Json())}}); }
        }
        segments.push_back(std::move(entry));
    }
    if (truth(input.value("enable_diarization",Json(false))) && !segments.empty()) {
        const auto session = input.value("session_id",Json());
        if (truth(session) && !session.is_string() && !session.is_array()) throw SttError("TypeError","'" + type(session) + "' object is not subscriptable");
        if (!operations_.diarize) throw SttError("RuntimeError","Native STT diarization is not configured");
        segments = Json::parse(operations_.diarize(path,segments.dump(),input.value("num_speakers",Json()).dump(),session.dump()));
        text.clear(); Json current;
        for (const auto &segment : segments) {
            const auto speaker = segment.value("speaker",Json("SPEAKER_00"));
            if (speaker != current) { current = speaker; text += "\n[" + string(speaker) + "]: "; }
            text += segment.at("text").get<std::string>() + " ";
        }
        text = strip_config_value(text);
    }
    operations_.remove(path);
    if (truth(input.value("return_segments",Json(false)))) {
        const auto info = Json::parse(inferred.info); Json detected;
        if (!info.is_null() && info.is_object() && info.contains("language")) detected = info.at("language"); else if (truth(language)) detected = language;
        return Json({{"text",strip_config_value(text)},{"segments",segments},{"language",detected}}).dump();
    }
    return Json(text).dump();
}
}
