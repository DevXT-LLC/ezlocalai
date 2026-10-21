#include "ezlocalai/stt_audio.hpp"
#include "whisper_test_data.hpp"
#include <dlfcn.h>
#include <map>
#include <iostream>
#include <nlohmann/json.hpp>
using namespace whisper_test;
using Json = nlohmann::json;
template<class T> T symbol(void *library,const char *name) {
    auto result = reinterpret_cast<T>(dlsym(library,name)); expect(result != nullptr,std::string("symbol ") + name); return result;
}
int main(int argc,char **argv) { try {
    expect(argc == 3,"fixture directory and PortAudio fixture library or --live");
    const auto directory = fs::path(argv[1]);
    if (std::string(argv[2]) == "--live") {
        auto operations = ezlocalai::stt_portaudio_operations();
        void *library = dlopen("libportaudio.so.2",RTLD_NOW | RTLD_LOCAL); expect(library != nullptr,"load real PortAudio");
        const auto devices = symbol<int (*)()>(library,"Pa_GetDeviceCount"); Json states = Json::array();
        const auto snapshot = [&] { const auto count = devices(); states.push_back({{"initialized",count >= 0},{"error",count < 0 ? Json(count) : Json()}}); };
        snapshot(); auto first = operations.open_audio(); snapshot(); auto second = operations.open_audio(); snapshot();
        operations.terminate_audio(first); snapshot(); operations.terminate_audio(second); snapshot();
        expect(states == Json::parse(read(directory/"stt_audio_live.json")),"real PortAudio reference-counted lifetime matches PyAudio");
        dlclose(library); std::cout << "Real PortAudio lifetime matches unchanged PyAudio: " << states.dump() << '\n'; return 0;
    }
    const std::string path = argv[2];
    void *library = dlopen(path.c_str(),RTLD_NOW | RTLD_LOCAL); expect(library != nullptr,"load PortAudio fixture");
    const auto configure = symbol<void (*)(int,int,const char *)>(library,"portaudio_fixture_configure");
    const auto events = symbol<const char *(*)()>(library,"portaudio_fixture_events");
    std::ifstream input(directory/"stt_audio_cases.jsonl"); expect(bool(input),"read audio fixtures"); std::string line; size_t total = 0;
    expect(bool(std::getline(input,line)),"audio metadata"); const auto meta = Json::parse(line);
    expect(meta.at("pyaudio") == "0.2.14" && meta.at("archive_sha256") == "78dfff3879b4994d1f4fc6485646a57755c6ee3c19647a491f790a0895bd2f87","pinned PyAudio package");
    expect(meta.at("init_c_sha256") == "afa4b513f087fa8da55aa2d92f931e43536df0253f18bc39001c05bdd4df3065" && meta.at("init_py_sha256") == "ed5e998cd26b6b9c2329c22892599253ce1498532b166db70d3d5db56b715c65","unchanged source identities");
    while (std::getline(input,line)) {
        const auto row = Json::parse(line); configure(row.at("initialize"),row.at("terminate"),row.at("text").get<std::string>().c_str());
        auto operations = ezlocalai::stt_portaudio_operations({},path); std::map<std::string,std::shared_ptr<void>> handles; Json errors = Json::array();
        for (size_t i = 0; i < row.at("actions").size(); ++i) {
            const auto action = row.at("actions").at(i).get<std::string>(); const auto separator = action.find(':');
            const auto name = action.substr(0,separator), identity = action.substr(separator + 1);
            try {
                if (name == "open") handles[identity] = operations.open_audio();
                else if (name == "terminate") operations.terminate_audio(handles.at(identity));
                else if (name == "drop") handles.erase(identity);
                else throw std::runtime_error("unknown fixture action");
            } catch (const ezlocalai::SttError &error) { errors.push_back({{"index",i},{"category",error.category},{"message",error.what()}}); }
        }
        handles.clear();
        expect(errors == row.at("errors"),"audio case " + std::to_string(total) + " errors: " + errors.dump());
        expect(std::string(events()) == row.at("events"),"audio case " + std::to_string(total) + " call order: " + events());
        ++total;
    }
    expect(total == 120,"all audio cases");
    // The STT wrapper must terminate each handle only once, even when close is
    // repeated. Its existing source-policy tests cover backend call ordering.
    configure(0,-9986,"");
    auto operations = ezlocalai::stt_portaudio_operations({},path); operations.batched_available = false;
    operations.load = [](const auto &,const auto &,const auto &,const auto &,auto) { return std::make_shared<ezlocalai::SttBackend>(); };
    { ezlocalai::SttRuntime runtime("fixture",operations); runtime.close(); runtime.close(); }
    expect(std::string(events()) == "initialize,terminate","wrapper performs one explicit termination");
    configure(0,0,"");
    { ezlocalai::SttRuntime runtime("fixture",operations); }
    expect(std::string(events()) == "initialize","wrapper destruction preserves source's explicit-close requirement");
    // Unclosed instances must not unload/reinitialize the library implicitly.
    operations = {}; configure(0,0,"");
    { auto transient = ezlocalai::stt_portaudio_operations({},path); transient.open_audio(); }
    expect(std::string(events()) == "initialize","native audio lifetime does not introduce termination");
    dlclose(library); std::cout << total << " unchanged PyAudio cases and STT audio lifetime checks pass\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
