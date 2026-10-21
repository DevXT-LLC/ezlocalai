#include "ezlocalai/llama_tts.hpp"
#include "whisper_transcription_test.hpp"
#include <future>
#include <thread>
#include <regex>
#include <signal.h>
using namespace transcription_test;
Json normalized_error(const std::string &category,const std::string &message) { return {{"category",category},{"message",std::regex_replace(message,std::regex("/tmp/ezlocalai-tts-[^/]+/speech.wav"),"<OUTPUT>")}}; }
Json audio(const TtsAudio &a) { return {{"samples",a.samples},{"channels",a.channels},{"rate",a.rate}}; }
int main(int argc,char **argv) { try {
    struct RestoreSignal { struct sigaction previous{}; RestoreSignal() { struct sigaction ignored{}; ignored.sa_handler=SIG_IGN; sigemptyset(&ignored.sa_mask); sigaction(SIGPIPE,&ignored,&previous); } ~RestoreSignal() { sigaction(SIGPIPE,&previous,nullptr); } } restore_signal;
    expect(argc==4,"source, process fixtures, worker fixture"); const auto cases=records(argv[2]); expect(hash(read(fs::path(argv[1])/"ezlocalai/LlamaTTS.py"))==cases[0].at("source_sha256"),"unchanged TTS source"); expect(hash(read(fs::path(__FILE__).parent_path()/"tts_process_fixture.cpp"))==cases[0].at("fixture_sha256"),"unchanged C++ process fixture"); size_t operations=0,failures=0;
    for(size_t i=1;i<cases.size();++i) { const auto &row=cases[i]; Temporary temp; const auto trace=temp.path/"trace",config_file=temp.path/"config"; auto config=row.at("config"); config["trace"]=trace.string(); if(config.contains("actions")) for(auto &action:config["actions"]) if(action.value("wait_file","")=="<GATE>") action["wait_file"]=(temp.path/"gate").string(); std::ofstream(config_file)<<config.dump(); for(const auto *file:{"model","mmproj","speaker"}) std::ofstream(temp.path/file).close();
        auto platform=native_llama_tts_platform(); const auto parent=platform.environment; auto &env=platform.environment; env["EZLOCALAI_TTS_FIXTURE"]=config_file.string(); env["QWEN_TTS_BIN"]=argv[3]; env["QWEN_TTS_MODEL_FILE"]=(temp.path/"model").string(); env["QWEN_TTS_MMPROJ_FILE"]=(temp.path/"mmproj").string(); env["QWEN_TTS_TIMEOUT"]=row.at("timeout").dump(); env["QWEN_TTS_DISABLE_CUDA_GRAPHS"]="true"; env["CUDA_VISIBLE_DEVICES"]="7,3"; env["HIP_VISIBLE_DEVICES"]="original"; env["GGML_CUDA_DISABLE_GRAPHS"]="inherited"; env.erase("QWEN_TTS_THREADS"); env.erase("QWEN_TTS_CONTEXT_SIZE");
        Json actual={{"operations",Json::array()}}; std::unique_ptr<LlamaTts> model;
        try { model=std::make_unique<LlamaTts>(llama_tts_launch(row.at("device").dump(),"{\"seed\":23}",std::move(platform))); actual["startup"]="ready"; }
        catch(const TtsError &e) { actual["startup"]=normalized_error(e.category,e.what()); }
        if(model) for(const auto &operation:row.at("ops")) {
            ++operations; Json out={{"chunks",Json::array()}}; const auto mode=operation.at("mode").get<std::string>(); auto request=operation.value("request",Json::object()); if(request.value("ref_audio",Json())=="<SPEAKER>") request["ref_audio"]=(temp.path/"speaker").string();
            auto invoke=[&] { try {
                if(mode=="close") model->close();
                else if(mode=="generate" || mode=="close_generate") out["chunks"].push_back(audio(model->generate(request.dump())));
                else model->stream(request.dump(),[&](const auto &chunk) { out["chunks"].push_back(audio(chunk)); if(mode=="callback_error") throw TtsError("RuntimeError","consumer failed"); return mode!="abort"; });
            } catch(const TtsError &e) { out["error"]=normalized_error(e.category,e.what()); } catch(const std::exception &e) { out["error"]=normalized_error("C++ exception",e.what()); } };
            if(mode=="parallel") {
                std::promise<void> started; auto start=started.get_future();
                auto run=[&](int index) { if(index) started.set_value(); auto req=request; req["text"]=request["text"].get<std::string>()+std::to_string(index); Json chunks=Json::array(); if(operation.at(index?"second":"first")=="generate") chunks.push_back(audio(model->generate(req.dump()))); else model->stream(req.dump(),[&](const auto &chunk) { chunks.push_back(audio(chunk)); return true; }); return chunks; };
                const auto before=records(trace).size(); auto first=std::async(std::launch::async,run,0); const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2); while(records(trace).size()==before) { expect(std::chrono::steady_clock::now()<deadline,"first request reached worker"); std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
                auto second=std::async(std::launch::async,run,1); expect(start.wait_for(std::chrono::seconds(1))==std::future_status::ready,"second caller started"); std::this_thread::sleep_for(std::chrono::milliseconds(30)); const bool serialized=records(trace).size()==before+1; std::ofstream(temp.path/"gate").close(); auto a=first.get(),b=second.get(); expect(serialized,"buffered/stream calls serialize on one worker"); out["parallel"]=Json::array({a,b});
            }
            else if(mode.rfind("close_",0)==0) { const auto before=records(trace).size(); auto task=std::async(std::launch::async,invoke); const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2); while(records(trace).size()==before) { expect(std::chrono::steady_clock::now()<deadline,"worker received cancellation request"); std::this_thread::sleep_for(std::chrono::milliseconds(5)); } const auto start=std::chrono::steady_clock::now(); model->close(); expect(task.wait_for(std::chrono::seconds(1))==std::future_status::ready,"cancel waiting generation promptly"); task.get(); expect(std::chrono::steady_clock::now()-start<std::chrono::seconds(1),"close does not wait on inference lock"); }
            else invoke();
            out["alive"]=model->alive(); actual["operations"].push_back(std::move(out));
        }
        if(model) { model->close(); model->close(); expect(!model->alive(),"idempotent close"); }
        auto events=records(trace); const int pid=events.at(0).at("pid"); expect(kill(pid,0)==-1 && errno==ESRCH,"worker reaped after close or startup failure"); events[0].erase("pid"); auto &args=events[0]["arguments"]; args[0]="<WORKER>"; for(size_t j=1;j+1<args.size();++j) { if(args[j]=="-m") args[j+1]="<MODEL>"; if(args[j]=="-mm") args[j+1]="<MMPROJ>"; } actual["events"]=events;
        expect(native_llama_tts_platform().environment==parent,"parent environment unchanged");
        struct sigaction current{}; sigaction(SIGPIPE,nullptr,&current); expect(current.sa_handler==SIG_IGN,"parent SIGPIPE disposition unchanged");
        try { compare_json(actual,row.at("expected"),row.at("name")); } catch(const std::exception &e) { ++failures; std::cerr<<e.what()<<'\n'; }
    }
    expect(cases.size()-1==cases[0].at("cases"),"all process scenarios"); expect(!failures,std::to_string(failures)+" process scenarios differ"); std::cout<<cases.size()-1<<" source process scenarios, "<<operations<<" operations match; workers reaped and parent environment unchanged\n"; return 0;
} catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; } }
