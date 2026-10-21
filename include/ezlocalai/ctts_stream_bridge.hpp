#pragma once
#include "ezlocalai/ctts_runtime.hpp"
namespace ezlocalai {
// Lazy, capacity-two producer bridge. next()/close() have a single consumer;
// request_cancel() is safe from another thread and interrupts a waiting next().
// Teardown joins the producer unless model.close itself throws, matching the
// source's exception precedence and still-running producer lifetime.
class CttsStreamBridge {
public:
    CttsStreamBridge(std::shared_ptr<CttsBackend> model,std::string request_json);
    ~CttsStreamBridge();
    CttsStreamBridge(const CttsStreamBridge &)=delete;
    CttsStreamBridge &operator=(const CttsStreamBridge &)=delete;
    std::optional<TtsAudio> next();
    void request_cancel();
    void close();
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};
}
