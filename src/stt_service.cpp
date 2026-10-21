#include "ezlocalai/stt_service.hpp"
#include "ezlocalai/stt_runtime.hpp"
#include "ezlocalai/resource_manager.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <iomanip>
#include <sstream>
namespace ezlocalai {
namespace {
thread_local uint64_t current_key = 0;
std::atomic<uint64_t> next_key{1};
bool value_error(const SttError &error) { return error.category == "ValueError" || error.category == "JSONDecodeError" || error.category == "UnicodeDecodeError" || error.category == "UnicodeEncodeError" || error.category == "UnicodeTranslateError"; }
std::string timestamp(double value,bool vtt) {
    if (!std::isfinite(value)) throw std::invalid_argument("Cannot format non-finite subtitle timestamp");
    const auto modulo = [](double x,double y) { auto value = std::fmod(x,y); if (value < 0) value += y; return value; };
    const auto padded = [](int64_t value,int width) { const auto raw = std::to_string(value); const auto zeros = std::string(std::max(0,width - static_cast<int>(raw.size())),'0'); return value < 0 ? "-" + zeros + raw.substr(1) : zeros + raw; };
    return padded(static_cast<int64_t>(std::floor(value / 3600)),2) + ':' + padded(static_cast<int64_t>(std::floor(modulo(value,3600) / 60)),2) + ':'
        + padded(static_cast<int64_t>(modulo(value,60)),2) + (vtt ? '.' : ',') + padded(static_cast<int64_t>((value - std::trunc(value)) * 1000),3);
}
}
SttService::SttService(std::string model,int64_t capacity,SttPoolOperations pool,VoiceSlotOperations voice,
    std::shared_ptr<VoicePermitQueue> shared,std::shared_ptr<ResourceManager> resources,std::function<bool()> voice_mode)
    : resources_(std::move(resources)),voice_server_mode_(std::move(voice_mode)) {
    pool.handoff_active = [this] { return handoffs_.load() != 0; };
    pool_ = std::make_unique<SttPool>(std::move(model),capacity,std::move(pool));
    const auto mark = voice.mark_handoff;
    voice.mark_handoff = [this,mark](const auto &service,bool active) { if (mark) mark(service,active); if (active) ++handoffs_; else { auto count = handoffs_.load(); while (count && !handoffs_.compare_exchange_weak(count,count - 1)) {} } };
    voice.destroy = [this](const auto &,bool async,bool force) { pool_->destroy(current_key,async,force); };
    guard_ = std::make_unique<VoiceSlotGuard>("stt",capacity,std::move(shared),std::move(voice));
}
SttService::~SttService() { try { pool_->destroy(0,false,true); } catch (...) {} }
void SttService::warm() { pool_->warm(); }
bool SttService::unload_idle() { return pool_->unload_idle(); }
void SttService::cancel() { cancellation_.cancel(); }
SttPoolSnapshot SttService::snapshot() const { return pool_->snapshot(); }
std::string SttService::transcribe(const std::string &request,bool bad_request,uint64_t key,SttServiceUse use) {
    return transcribe_impl(request,bad_request,key,use,nullptr);
}
std::string SttService::transcribe_bytes(const std::string &request,const std::string &audio,uint64_t key) {
    return transcribe_impl(request,false,key,SttServiceUse::TtsPipeline,&audio);
}
std::string SttService::transcribe_impl(const std::string &request,bool bad_request,uint64_t key,SttServiceUse use,const std::string *audio) {
    if (!key) key = next_key.fetch_add(1);
    struct Scope { uint64_t previous = current_key; explicit Scope(uint64_t value) { current_key = value; } ~Scope() { current_key = previous; } } scope(key);
    guard_->enter(key,cancellation_); std::string result; std::exception_ptr failure;
    try {
        const auto model = pool_->get(key); std::exception_ptr inference_failure;
        if (use != SttServiceUse::Endpoint) resources_->mark_model_in_use(ModelType::Stt,true);
        try { if(!model)throw SttError("AttributeError","'NoneType' object has no attribute 'transcribe_audio'");result = audio ? model->transcribe_bytes(request,*audio) : model->transcribe(request); }
        catch (const SttError &error) { if (bad_request && value_error(error)) inference_failure = std::make_exception_ptr(SttRequestError(400,error.what())); else inference_failure = std::current_exception(); }
        catch (...) { inference_failure = std::current_exception(); }
        resources_->mark_model_in_use(ModelType::Stt,false);
        if(use==SttServiceUse::TtsPipeline){
            if(inference_failure)std::rethrow_exception(inference_failure);
            if(!voice_server_mode_())pool_->destroy(key);
        }
        if (use == SttServiceUse::LlmVideo || (use == SttServiceUse::LlmAudio && !inference_failure)
            || (use == SttServiceUse::Endpoint && !voice_server_mode_())) pool_->destroy(key);
        if (inference_failure) std::rethrow_exception(inference_failure);
    } catch (...) { failure = std::current_exception(); }
    guard_->exit(key); if (failure) std::rethrow_exception(failure); return result;
}
std::string stt_segments_subtitles(const std::string &encoded,bool vtt) {
    const auto segments = nlohmann::json::parse(encoded); std::vector<std::string> lines;
    if (vtt) lines = {"WEBVTT",""}; size_t index = 0;
    for (const auto &segment : segments) {
        if (!vtt) lines.push_back(std::to_string(++index));
        lines.push_back(timestamp(segment.at("start").get<double>(),vtt) + " --> " + timestamp(segment.at("end").get<double>(),vtt));
        std::string text = segment.at("text");
        if (segment.contains("speaker")) { const auto speaker = segment.at("speaker").is_string() ? segment.at("speaker").get<std::string>() : segment.at("speaker").dump(); text = vtt ? "<v " + speaker + ">" + text + "</v>" : "[" + speaker + "]: " + text; }
        lines.push_back(text); lines.push_back("");
    }
    std::string out; for (size_t i = 0; i < lines.size(); ++i) { if (i) out += '\n'; out += lines[i]; } return out;
}
}
