#include "ezlocalai/stt_runtime.hpp"
#include "whisper_transcription_test.hpp"
using namespace transcription_test;
struct Environment {
    Json scenario, events = Json::array(); std::map<std::string,std::string> files;
    std::map<int,std::weak_ptr<SttBackend>> models; std::map<SttBackend *,int> ids; std::map<std::string,size_t> counters;
    int model_id = 0, uuid_id = 0; size_t inferred = 0;
    void event(const std::string &name,Json fields = Json::object()) { fields["event"] = name; events.push_back(std::move(fields)); }
    void fail(const Json &value) { if (!value.is_null()) throw SttError(value.at(0),value.at(1)); }
    void scripted(const std::string &name) { const auto i = counters[name]++; const auto errors = scenario.value(name + "_errors",Json::array()); if (i < errors.size()) fail(errors[i]); }
    void action(const std::string &name) { event(name); scripted(name); }
    SttInferenceResult infer(const std::string &kind,int id,const std::string &path,const std::string &kwargs) {
        event("infer",{{"kind",kind},{"model",id},{"path",path},{"kwargs",Json::parse(kwargs)}});
        const auto i = inferred++; const auto failures = scenario.value("infer_errors",Json::array()); if (i < failures.size()) fail(failures[i]);
        return {scenario.value("segments",Json::array()).dump(),scenario.value("info",Json{{"language","en"}}).dump()};
    }
    SttRuntimeOperations operations() {
        SttRuntimeOperations out; out.batched_available = scenario.value("batched_available",true);
        out.cuda_available = [this] { event("cuda"); return scenario.value("cuda",false); };
        out.cuda_total_bytes = [this] { action("total"); return scenario.value("total_mb",4096.0) * 1024 * 1024; };
        out.cuda_allocated_bytes = [this] { action("allocated"); return scenario.value("allocated_mb",0.0) * 1024 * 1024; };
        out.cudnn_available = [this] { action("cudnn"); return scenario.value("cudnn",true); };
        out.test_cudnn = [this] { action("test_cudnn"); return std::shared_ptr<void>{}; };
        out.empty_cache = [this] { action("empty_cache"); };
        out.synchronize = [this] { action("synchronize"); };
        out.ipc_collect = [this] { action("ipc_collect"); };
        out.cpu_count = [this] { action("cpu_count"); return scenario.value("cores",int64_t{16}); };
        out.collect = [this] { Json alive = Json::array(); for (const auto &item : models) if (!item.second.expired()) alive.push_back(item.first); event("collect",{{"alive",alive}}); };
        out.load = [this](const std::string &model,const std::string &root,const std::string &device,const std::string &compute,int64_t threads) {
            event("load",{{"model",model},{"root",root},{"device",device},{"compute",compute},{"threads",threads}}); scripted("load");
            const auto id = ++model_id; auto backend = std::make_shared<SttBackend>(); models[id] = backend; ids[backend.get()] = id;
            backend->transcribe = [this,id](const auto &path,const auto &kwargs) { return infer("standard",id,path,kwargs); }; return backend;
        };
        out.batch = [this](const std::shared_ptr<SttBackend> &model) {
            const auto id = ids.at(model.get()); event("batch",{{"model",id}}); scripted("batch"); auto backend = std::make_shared<SttBackend>();
            backend->transcribe = [this,id,model](const auto &path,const auto &kwargs) { (void)model; return infer("batch",id,path,kwargs); }; return backend;
        };
        out.open_audio = [this] { action("audio"); return std::make_shared<int>(1); };
        out.terminate_audio = [this](const std::shared_ptr<void> &) { action("terminate"); };
        out.uuid_hex = [this] { std::ostringstream value; value << std::hex << std::setfill('0') << std::setw(32) << ++uuid_id; event("uuid",{{"value",value.str()}}); return value.str(); };
        out.working_directory = [this] { event("cwd"); return "/work"; };
        out.write = [this](const std::string &path,const std::string &bytes) {
            event("open",{{"path",path},{"mode","wb"}}); fail(scenario.value("write_error",Json())); std::ostringstream hex;
            for (unsigned char byte : bytes) hex << std::hex << std::setfill('0') << std::setw(2) << unsigned(byte);
            event("write",{{"path",path},{"hex",hex.str()}}); files[path] = bytes;
        };
        out.exists = [this](const std::string &path) { event("exists",{{"path",path}}); return !scenario.value("missing_after_write",false) && files.count(path); };
        out.remove = [this](const std::string &path) { event("remove",{{"path",path}}); fail(scenario.value("remove_error",Json())); files.erase(path); };
        out.diarize = [this](const std::string &path,const std::string &segments,const std::string &speakers,const std::string &session) {
            auto data = Json::parse(segments); event("diarize",{{"path",path},{"segments",data},{"speakers",Json::parse(speakers)},{"session",Json::parse(session)}}); fail(scenario.value("diarize_error",Json()));
            for (size_t i = 0; i < data.size(); ++i) data[i]["speaker"] = i % 3 ? "SPEAKER_01" : "SPEAKER_00"; return data.dump();
        };
        return out;
    }
};
Json error(const std::exception &exception) {
    const auto *typed = dynamic_cast<const SttError *>(&exception);
    return {{"category",typed ? typed->category : "RuntimeError"},{"message",exception.what()}};
}
size_t filesystem_cases(const fs::path &fixture,const Json &source_hash) {
    const auto cases = records(fixture); expect(cases.size() == 10,"complete real-filesystem STT cases");
    expect(cases[0].at("source_sha256") == source_hash,"filesystem STT source identity");
    for (size_t i = 1; i < cases.size(); ++i) {
        const auto &row = cases[i]; const auto setup = row.at("setup").get<std::string>(); Temporary temporary;
        const auto outputs = temporary.path/"outputs", target = outputs/(std::string(32,'0') + ".wav");
        if (setup == "outputs_file") std::ofstream(outputs) << "file";
        else if (setup != "missing_outputs") fs::create_directory(outputs);
        if (setup == "target_directory") fs::create_directory(target);
        if (setup == "full_device") fs::create_symlink("/dev/full",target);
        if (setup == "dangling_link") fs::create_symlink("missing/target",target);
        SttRuntimeOperations operations; operations.batched_available = false;
        operations.working_directory = [&] { return temporary.path.string(); };
        operations.uuid_hex = [] { return std::string(32,'0'); };
        operations.open_audio = [] { return std::make_shared<int>(1); };
        operations.terminate_audio = [](const auto &) {};
        operations.load = [&](const auto &,const auto &,const auto &,const auto &,int64_t) {
            auto backend = std::make_shared<SttBackend>();
            backend->transcribe = [&](const std::string &path,const auto &) {
                if (setup == "replace_with_directory") { fs::remove(path); fs::create_directory(path); }
                return SttInferenceResult{R"([{"text":" text ","start":0.0,"end":1.0,"words":null}])",R"({"language":"en"})"};
            }; return backend;
        };
        SttRuntime runtime("test-model",std::move(operations)); Json outcome;
        try { outcome["value"] = Json::parse(runtime.transcribe(row.at("request").dump())); }
        catch (const std::exception &exception) {
            outcome["error"] = error(exception); auto message = outcome["error"]["message"].get<std::string>();
            const auto at = message.find(temporary.path.string()); if (at != std::string::npos) message.replace(at,temporary.path.string().size(),"/work");
            outcome["error"]["message"] = message;
        }
        std::error_code ignored; const auto status = fs::symlink_status(target,ignored);
        outcome["target_kind"] = fs::is_symlink(status) ? "link" : fs::is_directory(status) ? "directory" : fs::is_regular_file(status) ? "file" : "missing";
        auto expected = row; expected.erase("setup"); expected.erase("request");
        compare_json(outcome,expected,"filesystem STT " + setup); runtime.close();
    }
    return cases.size() - 1;
}
int main(int argc,char **argv) { try {
    expect(argc == 3,"reference root and STT fixture"); auto cases = records(argv[2]); expect(cases.size() == 801,"complete STT wrapper cases");
    expect(hash(read(fs::path(argv[1])/"ezlocalai/STT.py")) == cases[0].at("source_sha256"),"unchanged STT source identity");
    const auto byte_cases=records(fs::path(argv[2]).parent_path()/"stt_runtime_bytes_cases.jsonl");expect(byte_cases[0].at("source_sha256")==cases[0].at("source_sha256")&&byte_cases.size()-1==byte_cases[0].at("cases"),"complete raw-byte source capture");cases.insert(cases.end(),byte_cases.begin()+1,byte_cases.end());
    size_t actions = 0;
    for (size_t row_index = 1; row_index < cases.size(); ++row_index) {
        const auto &row = cases[row_index]; const auto label = "STT case " + std::to_string(row_index - 1); Environment environment; environment.scenario = row.at("scenario");
        std::unique_ptr<SttRuntime> runtime; Json outcomes = Json::array();
        try { runtime = std::make_unique<SttRuntime>("test-model",environment.operations()); }
        catch (const std::exception &exception) { outcomes.push_back({{"operation","init"},{"error",error(exception)}}); }
        if (runtime) for (const auto &action : row.at("actions")) {
            Json outcome{{"operation",action.at("operation")}};
            try { if (action["operation"] == "close") { runtime->close(); outcome["value"] = nullptr; } else outcome["value"] = Json::parse(action.contains("audio_hex") ? runtime->transcribe_bytes(action.at("request").dump(),unhex(action.at("audio_hex"))) : runtime->transcribe(action.at("request").dump())); }
            catch (const std::exception &exception) { outcome["error"] = error(exception); }
            outcome["state"] = {{"device",runtime->device()},{"compute",runtime->compute_type()},{"model",runtime->has_model()},{"batch",runtime->has_batch()},{"audio",runtime->has_audio()}};
            outcome["files"] = Json::array(); for (const auto &file : environment.files) outcome["files"].push_back(file.first); outcomes.push_back(std::move(outcome)); ++actions;
        }
        compare_json(outcomes,row.at("outcomes"),label + ".outcomes"); compare_json(environment.events,row.at("events"),label + ".events");
    }
    const auto io_cases = filesystem_cases(fs::path(argv[2]).parent_path()/"stt_runtime_io_cases.jsonl",cases[0].at("source_sha256"));
    std::cout << cases.size() - 1 << " STT wrapper cases and " << actions << " actions match source output, exceptions, state, cleanup and dependency-call order; " << io_cases << " real-filesystem cases match\n"; return 0;
} catch (const std::exception &exception) { std::cerr << exception.what() << '\n'; return 1; } }
