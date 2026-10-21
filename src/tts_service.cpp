#include "ezlocalai/tts_service.hpp"
#include "ezlocalai/tts_websocket.hpp"
#include "ezlocalai/configuration.hpp"
#include "ezlocalai/llm_slots.hpp"
#include "ezlocalai/llm_response.hpp"
#include "ezlocalai/stt_input.hpp"
#include <nlohmann/json.hpp>
#include <atomic>
#include <cstdlib>
namespace ezlocalai {
namespace {
thread_local uint64_t current_key=0;
std::atomic<uint64_t> next_key{1};
struct KeyScope {uint64_t previous=current_key;explicit KeyScope(uint64_t key){current_key=key;}~KeyScope(){current_key=previous;}};
uint64_t identity(uint64_t key){return key?key:next_key.fetch_add(1);}
bool base_exception(const std::exception_ptr &failure){try{std::rethrow_exception(failure);}catch(const VoiceCancelled &){return true;}catch(const TtsError &e){return e.category=="CancelledError"||e.category=="GeneratorExit"||e.category=="KeyboardInterrupt"||e.category=="SystemExit";}catch(...){return false;}}
std::string message(const std::exception_ptr &failure){try{std::rethrow_exception(failure);}catch(const std::exception &e){return e.what();}catch(const LlmResponseError &e){return e.message;}catch(...){return "";}}
template<class Body,class Cleanup> void finally(Body body,Cleanup cleanup){std::exception_ptr failure;try{body();}catch(...){failure=std::current_exception();}cleanup();if(failure)std::rethrow_exception(failure);}
std::string decode(const std::string &encoded){try{return stt_base64_decode(nlohmann::json(encoded).dump());}catch(const LlmResponseError &e){throw TtsError(e.category,e.message);}}
}
bool tts_should_try_gtts(const std::string &message,const std::string &configured){
    const auto enabled=llm_unicode_lower(strip_config_value(configured));if(enabled!="1"&&enabled!="true"&&enabled!="yes"&&enabled!="on")return false;
    const auto lower=llm_unicode_lower(message);for(const auto fragment:{"qwen_tts_allow_cpu_fallback","needs cuda vram","cuda is not available","gpu load failed","gpu generation failed"})if(lower.find(fragment)!=std::string::npos)return true;return false;
}
struct TtsService::Impl {
    struct Stream;
    struct WebSocketLease;
    TtsRequestOperations request;
    std::atomic<unsigned> handoffs{0};
    VoiceCancellation cancellation;
    std::unique_ptr<TtsPool> pool;
    std::unique_ptr<VoiceSlotGuard> guard;
    ~Impl(){try{if(pool)pool->destroy(0,false,true);}catch(...){}}
};
TtsService::TtsService(int64_t capacity,TtsPoolOperations pool,VoiceSlotOperations voice,std::shared_ptr<VoicePermitQueue> shared,std::shared_ptr<ResourceManager> resources,TtsRequestOperations request):impl_(std::make_shared<Impl>()){
    if(!request.fallback_generate)throw std::invalid_argument("TTS service requires a fallback transport callback");
    if(!request.environment)request.environment=[](const auto &key)->std::optional<std::string>{const auto *value=std::getenv(key.c_str());return value?std::optional<std::string>(value):std::nullopt;};
    if(!request.voice_server_mode)request.voice_server_mode=[]{return false;};
    if(!request.mark_in_use)request.mark_in_use=[resources](bool active){resources->mark_model_in_use(ModelType::Tts,active);};
    impl_->request=std::move(request);const auto state=impl_.get();pool.handoff_active=[state]{return state->handoffs.load()!=0;};impl_->pool=std::make_unique<TtsPool>(capacity,std::move(pool));
    const auto mark=voice.mark_handoff;voice.mark_handoff=[state,mark](const auto &service,bool active){if(mark)mark(service,active);if(active)++state->handoffs;else{auto count=state->handoffs.load();while(count&&!state->handoffs.compare_exchange_weak(count,count-1)){}}};
    voice.destroy=[state](const auto &,bool async,bool force){state->pool->destroy(current_key,async,force);};
    impl_->guard=std::make_unique<VoiceSlotGuard>("tts",capacity,std::move(shared),std::move(voice));
}
TtsService::~TtsService()=default;
void TtsService::warm(){impl_->pool->warm();}
void TtsService::warm_cache(){impl_->pool->warm_cache();}
void TtsService::unload(uint64_t key){KeyScope scope(key);impl_->pool->destroy(key,false,true);}
bool TtsService::unload_idle(){return impl_->pool->unload_idle();}
void TtsService::cancel(){impl_->cancellation.cancel();}
TtsPoolSnapshot TtsService::snapshot()const{return impl_->pool->snapshot();}
bool TtsService::handoff_active()const{return impl_->handoffs.load()!=0;}
VoicePermitSnapshot TtsService::permits()const{return impl_->guard->permits();}
std::vector<std::pair<uint64_t,std::vector<VoiceSlotHandle>>> TtsService::slot_stacks()const{return impl_->guard->stacks();}
std::string TtsService::generate(const std::string &request,uint64_t key){
    const auto state=impl_;key=identity(key);KeyScope scope(key);state->guard->enter(key,state->cancellation);TtsPoolHandle model;std::string result;
    finally([&]{finally([&]{
        try{model=state->pool->get(key);if(!model)throw TtsError("AttributeError","'NoneType' object has no attribute 'generate'");result=decode(model->generate(request));}
        catch(...){const auto failure=std::current_exception();if(base_exception(failure))std::rethrow_exception(failure);if(tts_should_try_gtts(message(failure),state->request.environment("QWEN_TTS_ENABLE_GTTS_FALLBACK").value_or("true")))result=state->request.fallback_generate(request);else std::rethrow_exception(failure);}
    },[&]{state->request.mark_in_use(false);if(model&&!state->request.voice_server_mode())state->pool->destroy(key);});},[&]{state->guard->exit(key);});return result;
}
std::string TtsService::generate_pipeline(const std::string &request,uint64_t key){
    const auto state=impl_;key=identity(key);KeyScope scope(key);state->guard->enter(key,state->cancellation);std::string result;
    finally([&]{
        auto model=state->pool->get(key);state->request.mark_in_use(true);
        finally([&]{if(!model)throw TtsError("AttributeError","'NoneType' object has no attribute 'generate'");result=model->generate(request);},[&]{state->request.mark_in_use(false);});
        if(!state->request.voice_server_mode())state->pool->destroy(key);
    },[&]{state->guard->exit(key);});return result;
}
std::string TtsService::generate_local(const std::string &request,const VoiceCancellation &cancellation,uint64_t key){
    const auto state=impl_;key=identity(key);KeyScope scope(key);
    LinkedVoiceCancellation linked(state->cancellation,cancellation);
    state->guard->enter(key,linked.token());TtsPoolHandle model;std::string result;
    finally([&]{finally([&]{
        if(state->cancellation.requested()||cancellation.requested())throw VoiceCancelled("");
        model=state->pool->get(key);if(!model)throw TtsError("AttributeError","'NoneType' object has no attribute 'generate'");
        if(linked.token().requested())throw VoiceCancelled("");
        result=decode(model->generate(request));
        if(linked.token().requested())throw VoiceCancelled("");
    },[&]{state->request.mark_in_use(false);if(model)state->pool->destroy(key);});},[&]{state->guard->exit(key);});return result;
}
struct TtsService::Impl::Stream final:TtsServiceStream {
    std::shared_ptr<Impl> state;TtsPoolHandle model;VoiceSlotHandle slot;std::string request;uint64_t owner,consumer=0;bool started=false,finished=false;std::atomic<bool> cancelled{false};std::shared_ptr<CttsByteStream> inner;
    Stream(std::shared_ptr<Impl> state,TtsPoolHandle model,VoiceSlotHandle slot,std::string request,uint64_t owner):state(std::move(state)),model(std::move(model)),slot(std::move(slot)),request(std::move(request)),owner(owner){}
    ~Stream()override{try{close();}catch(...){}}
    void consumer_key(uint64_t key)override{if(key)consumer=key;else if(!consumer)consumer=identity(0);}
    void cleanup(){
        // A failed explicit mark prevents entering the source's nested finally.
        state->request.mark_in_use(false);
        finally([&]{if(!slot->snapshot().handoff)state->pool->destroy(consumer,false);},[&]{state->guard->release(slot);});
    }
    void finish(std::exception_ptr failure={}){
        finished=true;
        auto current=std::atomic_exchange(&inner,std::shared_ptr<CttsByteStream>{});
        try{if(current)current->close();}catch(...){failure=std::current_exception();}
        try{cleanup();}catch(...){failure=std::current_exception();}
        model.reset();slot.reset();state.reset();request.clear();if(failure)std::rethrow_exception(failure);
    }
    std::optional<std::string> next()override{
        if(finished)return {};if(!consumer)consumer=identity(0);KeyScope scope(consumer);
        if(cancelled&&!started){finished=true;model.reset();slot.reset();state.reset();throw TtsError("CancelledError","");}
        std::optional<std::string> result;std::exception_ptr failure;
        try{
            if(!started){started=true;state->pool->transfer(owner,consumer);if(!model)throw TtsError("AttributeError","'NoneType' object has no attribute 'generate_stream'");std::atomic_store(&inner,std::shared_ptr<CttsByteStream>(model->stream(request)));}
            if(cancelled)throw TtsError("CancelledError","");const auto current=std::atomic_load(&inner);result=current->next();
        }catch(...){failure=std::current_exception();}
        if(failure||!result)finish(failure);return result;
    }
    void close()override{
        if(finished)return;if(!started){finished=true;model.reset();slot.reset();state.reset();return;}
        KeyScope scope(consumer);finish();
    }
    void request_cancel()override{cancelled=true;if(const auto current=std::atomic_load(&inner))current->request_cancel();}
};
std::unique_ptr<TtsServiceStream> TtsService::stream(const std::string &request,uint64_t owner){
    const auto state=impl_;owner=identity(owner);KeyScope scope(owner);auto slot=state->guard->acquire(state->cancellation);TtsPoolHandle model;
    try{model=state->pool->get(owner);}catch(...){const auto failure=std::current_exception();if(!base_exception(failure))state->guard->release(slot);std::rethrow_exception(failure);}
    return std::make_unique<Impl::Stream>(state,std::move(model),std::move(slot),request,owner);
}
struct TtsService::Impl::WebSocketLease final:TtsWebSocketLease {
    std::shared_ptr<Impl> state;TtsPoolHandle model;VoiceSlotHandle slot;uint64_t key;bool finished=false;
    WebSocketLease(std::shared_ptr<Impl> state,TtsPoolHandle model,VoiceSlotHandle slot,uint64_t key):state(std::move(state)),model(std::move(model)),slot(std::move(slot)),key(key){}
    ~WebSocketLease()override{try{close();}catch(...){}}
    std::unique_ptr<CttsByteStream> stream(const std::string &request)override{KeyScope scope(key);if(!model)throw TtsError("AttributeError","'NoneType' object has no attribute 'generate_stream'");return model->stream(request);}
    void close()override{
        if(finished)return;finished=true;KeyScope scope(key);
        finally([&]{if(model&&!slot->snapshot().handoff)state->pool->destroy(key,false);},[&]{state->guard->release(slot);});
    }
};
std::unique_ptr<TtsWebSocketLease> TtsService::websocket_lease(const VoiceCancellation &cancellation,uint64_t key){
    const auto state=impl_;key=identity(key);KeyScope scope(key);
    if(state->cancellation.requested())throw VoiceCancelled("");
    auto slot=state->guard->acquire(cancellation);TtsPoolHandle model;
    try{model=state->pool->get(key);}catch(...){const auto failure=std::current_exception();state->guard->release(slot);std::rethrow_exception(failure);}
    return std::make_unique<Impl::WebSocketLease>(state,std::move(model),std::move(slot),key);
}
}
