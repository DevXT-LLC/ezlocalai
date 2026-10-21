#pragma once
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
namespace ezlocalai {
struct TtsError : std::runtime_error { std::string category; bool runtime_family=false; TtsError(std::string category,std::string message,bool runtime_family=false) : std::runtime_error(std::move(message)),category(std::move(category)),runtime_family(runtime_family) {} };
struct LlamaTtsPlatform {
    std::map<std::string,std::string> environment;
    std::function<bool(const std::string &)> is_file, executable;
    std::function<std::string(const std::string &)> absolute;
    std::function<std::string()> home;
    std::function<std::string(const std::string &,const std::string &,const std::string &)> download;
};
struct LlamaTtsLaunch { double timeout = 300; std::string device; std::vector<std::string> arguments; std::map<std::string,std::string> environment; };
LlamaTtsPlatform native_llama_tts_platform();
std::string tts_python_integer(const std::string &json);
std::string tts_python_string(const std::string &json);
std::string resolve_tts_model_id(const std::string &configured);
LlamaTtsLaunch llama_tts_launch(const std::string &device_json,const std::string &generation_json,LlamaTtsPlatform platform);
std::string llama_tts_language(const std::string &language_json,const std::string &text_json);
// Decimal rate preserves the source wrapper's unbounded Python int conversion.
struct TtsAudio { std::vector<float> samples; std::string rate; int channels = 1; };
struct TtsStreamIterator {
    virtual ~TtsStreamIterator()=default;
    virtual std::optional<TtsAudio> next()=0;
    virtual void close()=0;
};
class LlamaTts {
public:
    explicit LlamaTts(LlamaTtsLaunch launch);
    ~LlamaTts();
    LlamaTts(const LlamaTts &) = delete;
    LlamaTts &operator=(const LlamaTts &) = delete;
    TtsAudio generate(const std::string &request_json);
    // Returning false closes an abandoned source generator and its worker.
    void stream(const std::string &request_json,const std::function<bool(const TtsAudio &)> &consume);
    // Lazy iterator. Advance and close on one producer thread; model.close()
    // may interrupt that thread while it waits for the native worker.
    std::unique_ptr<TtsStreamIterator> stream_iterator(const std::string &request_json);
    void close();
    bool alive() const;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};
}
