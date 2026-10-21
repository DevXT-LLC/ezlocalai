#include "ezlocalai/stt_audio.hpp"
#include <dlfcn.h>
#include <map>
#include <mutex>

namespace ezlocalai {
namespace {
struct PortAudio {
    void *library = nullptr;
    int (*initialize)() = nullptr;
    int (*terminate)() = nullptr;
    const char *(*error_text)(int) = nullptr;

    explicit PortAudio(const std::string &path) {
        library = dlopen(path.c_str(),RTLD_NOW | RTLD_LOCAL);
        if (!library) throw SttError("RuntimeError",std::string("Cannot load PortAudio: ") + dlerror());
        try {
            initialize = symbol<int (*)()>("Pa_Initialize");
            terminate = symbol<int (*)()>("Pa_Terminate");
            error_text = symbol<const char *(*)(int)>("Pa_GetErrorText");
        } catch (...) { dlclose(library); throw; }
    }
    template<class T> T symbol(const char *name) {
        dlerror(); auto value = dlsym(library,name); const auto *error = dlerror();
        if (error) throw SttError("RuntimeError",std::string("Cannot resolve PortAudio symbol: ") + error);
        return reinterpret_cast<T>(value);
    }
    ~PortAudio() { dlclose(library); }
};
struct Audio { std::shared_ptr<PortAudio> api; };
std::shared_ptr<PortAudio> portaudio(const std::string &path) {
    // Python keeps the imported extension and its PortAudio library resident.
    // Retain the native library too, including when an unclosed handle is lost.
    static std::mutex mutex;
    static std::map<std::string,std::shared_ptr<PortAudio>> libraries;
    std::lock_guard<std::mutex> lock(mutex);
    auto &api = libraries[path]; if (!api) api = std::make_shared<PortAudio>(path);
    return api;
}
}
SttRuntimeOperations stt_portaudio_operations(SttRuntimeOperations operations,const std::string &library) {
    // Preserve supplied boundaries, but do not pair a user-defined handle with
    // the native termination callback (or vice versa).
    if (operations.open_audio || operations.terminate_audio) return operations;
    auto api = portaudio(library);
    operations.open_audio = [api]() -> std::shared_ptr<void> {
        auto audio = std::make_shared<Audio>(); audio->api = api;
        const auto error = api->initialize();
        if (error) {
            api->terminate();
            throw SttError("OSError","[Errno " + std::to_string(error) + "] " + api->error_text(error));
        }
        return audio;
    };
    operations.terminate_audio = [api](const std::shared_ptr<void> &) { api->terminate(); };
    return operations;
}
}
