#include "ezlocalai/ctts_stream_bridge.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <future>
#include <mutex>
#include <thread>
namespace ezlocalai {
struct CttsStreamBridge::Impl {
    struct Message { enum Kind {audio,done,error} kind; std::optional<TtsAudio> value; std::exception_ptr failure; };
    std::shared_ptr<CttsBackend> model;
    std::string request;
    std::atomic<bool> cancelled{false};
    std::mutex mutex;
    std::condition_variable available,space;
    std::deque<Message> queue;
    std::thread producer;
    std::exception_ptr producer_failure;
    bool started=false,complete=false,closed=false;
    Impl(std::shared_ptr<CttsBackend> value,std::string encoded):model(std::move(value)),request(std::move(encoded)) {}
    bool send(Message message) {
        std::unique_lock<std::mutex> lock(mutex);
        while(!cancelled.load()) {
            if(queue.size()<2) {queue.push_back(std::move(message));available.notify_one();return true;}
            space.wait_for(lock,std::chrono::milliseconds(100));
        }
        return false;
    }
    void produce(std::shared_ptr<CttsBackend> producer_model) noexcept {
        try {
            // Iterator creation is outside the source producer's try/finally.
            if(!producer_model) throw TtsError("AttributeError","'NoneType' object has no attribute 'generate_voice_clone_stream'");
            auto iterator=producer_model->stream(request);
            std::exception_ptr failure;
            try {
                for(;;) {
                    auto value=iterator->next();
                    if(!value) {send({Message::done,{},{}});break;}
                    if(cancelled.load() || !send({Message::audio,std::move(value),{}})) break;
                }
            } catch(const TtsError &e) {
                if(e.category=="GeneratorExit" || e.category=="KeyboardInterrupt" || e.category=="SystemExit" || e.category=="CancelledError") failure=std::current_exception();
                else if(!cancelled.load()) send({Message::error,{},std::current_exception()});
            } catch(const std::exception &) {
                if(!cancelled.load()) send({Message::error,{},std::current_exception()});
            } catch(...) { failure=std::current_exception(); }
            iterator->close();
            if(failure) std::rethrow_exception(failure);
        } catch(...) { producer_failure=std::current_exception(); }
    }
    void finish() {
        if(closed) return;
        closed=true;cancelled.store(true);available.notify_all();
        if(!started) {model.reset();return;}
        std::exception_ptr close_failure;
        if(!complete) try {std::async(std::launch::async,[&]{if(!model) throw TtsError("AttributeError","'NoneType' object has no attribute 'close'");model->close();}).get();} catch(...) {close_failure=std::current_exception();}
        model.reset();
        if(close_failure) {if(producer.joinable()) producer.detach();std::rethrow_exception(close_failure);}
        if(producer.joinable()) producer.join();
        if(producer_failure) std::rethrow_exception(producer_failure);
    }
};
CttsStreamBridge::CttsStreamBridge(std::shared_ptr<CttsBackend> model,std::string request):impl_(std::make_shared<Impl>(std::move(model),std::move(request))) {}
CttsStreamBridge::~CttsStreamBridge() {try {close();}catch(...) {}}
void CttsStreamBridge::request_cancel() {impl_->cancelled.store(true);impl_->available.notify_all();}
void CttsStreamBridge::close() {impl_->finish();}
std::optional<TtsAudio> CttsStreamBridge::next() {
    if(impl_->closed) return std::nullopt;
    if(!impl_->started) {const auto state=impl_;const auto model=state->model;impl_->producer=std::thread([state,model]{state->produce(model);});impl_->started=true;}
    std::unique_lock<std::mutex> lock(impl_->mutex);
    impl_->available.wait(lock,[&]{return !impl_->queue.empty() || impl_->cancelled.load();});
    if(impl_->cancelled.load()) {lock.unlock();impl_->finish();throw TtsError("CancelledError","");}
    auto message=std::move(impl_->queue.front());impl_->queue.pop_front();impl_->space.notify_one();lock.unlock();
    if(message.kind==Impl::Message::audio) return std::move(message.value);
    impl_->complete=message.kind==Impl::Message::done;impl_->finish();
    if(message.failure) std::rethrow_exception(message.failure);
    return std::nullopt;
}
}
