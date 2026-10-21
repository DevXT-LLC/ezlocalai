#include "ezlocalai/ctts_runtime.hpp"
#include "whisper_transcription_test.hpp"
#include <map>
#include <future>
using namespace transcription_test;
void raise_error(const Json &value) { if(value.is_object() && value.contains("error")) { const auto &e=value.at("error"); throw TtsError(e.at("category"),e.at("message")); } }
struct Model : CttsBackend { int id=0; bool closed=false; };
int main(int argc,char **argv) { try {
    expect(argc==3,"source and CTTS lifecycle fixture"); const auto rows=records(argv[2]); expect(hash(read(fs::path(argv[1])/"ezlocalai/CTTS.py"))==rows[0].at("source_sha256"),"pinned CTTS source"); expect(hash(read(fs::path(argv[1])/"ezlocalai/LlamaTTS.py"))==rows[0].at("llama_sha256"),"pinned TTS model policy"); size_t errors=0,operations=0;
    for(size_t i=1;i<rows.size();++i) { const auto &row=rows[i],&hardware=row.at("hardware"); Json environment=row.at("env"),events=Json::array(); std::map<std::string,size_t> counters; std::vector<std::shared_ptr<Model>> models; std::unique_ptr<CttsRuntime> owner; std::promise<void> started,released; auto starting=started.get_future(); auto releasing=released.get_future().share();
        const auto event=[&](const std::string &name,Json args=Json::array()) { Json entry=Json::array({name}); for(const auto &arg:args) entry.push_back(arg); events.push_back(std::move(entry)); const auto n=counters[name]++; const auto values=row.at("failures").value(name,Json::array()); if(n<values.size()) raise_error(values[n]); };
        CttsOperations ops; ops.environment=[&](const std::string &key)->std::optional<std::string> { events.push_back(Json::array({"env",key})); return environment.contains(key)?std::optional<std::string>(environment[key].get<std::string>()):std::nullopt; };
        ops.cuda_available=[&] { event("available"); auto value=hardware.value("available",Json(false)); if(value.is_array()) value=value.at(std::min(counters["available"]-1,value.size()-1)); raise_error(value); return value.get<bool>(); };
        ops.memory_info=[&](const auto &index) { event("memory",Json::array({index})); raise_error(hardware.value("memory",Json())); return std::pair<double,double>{hardware.value("free_mb",8000.0)*1048576,16000.0*1048576}; };
        ops.total_bytes=[&](const auto &index) { event("total",Json::array({index})); raise_error(hardware.value("total",Json())); return hardware.value("total_mb",16000.0)*1048576; };
        ops.allocated_bytes=[&](const auto &index) { event("allocated",Json::array({index})); raise_error(hardware.value("allocated",Json())); return hardware.value("allocated_mb",1000.0)*1048576; };
        ops.collect=[&] { event("collect"); }; ops.empty_cache=[&] { event("empty_cache"); }; ops.synchronize=[&] { event("synchronize"); }; ops.ipc_collect=[&] { event("ipc_collect"); };
        ops.inference_mode=[&] { event("enter"); return [&](const std::string &category) { event("exit",Json::array({category})); }; };
        ops.load=[&](const auto &device,const auto &kwargs) { event("load",Json::array({device,Json::parse(kwargs)})); const auto n=counters["load"]-1; if(n<row.at("loads").size()) raise_error(row["loads"][n]); auto model=std::make_shared<Model>(); model->id=models.size()+1; model->generate_name="ctts_fixture.Model.generate_voice_clone"; auto *raw=model.get();
            model->close=[&,raw] { event("close",Json::array({raw->id})); raw->closed=true; };
            model->generate=[&,raw](const std::string &request) { event("generate",Json::array({raw->id,Json::parse(request)})); const auto n=counters["generate"]-1; const auto action=n<row.at("generations").size()?row["generations"][n]:Json::object(); if(action.value("wait",false)) { started.set_value(); releasing.wait(); } if(action.value("close_owner",false)) owner->close(); raise_error(action); CttsGenerated result; if(!action.value("empty",false)) result.audio.push_back({{0.f,.25f,-.5f},"",1}); result.rate_json=action.value("rate_json",action.value("rate",Json(24000)).dump()); return result; };
            models.push_back(model); return model;
        };
        const auto state=[&] { const auto current=owner->backend(); Json all=Json::array(); for(const auto &model:models) all.push_back({{"id",model->id},{"closed",model->closed}}); return Json{{"settings",Json::parse(owner->settings().json())},{"device",owner->device()},{"closed",owner->closed()},{"model",current?Json(std::static_pointer_cast<Model>(current)->id):Json()},{"models",all}}; };
        Json actual={{"operations",Json::array()}};
        try { owner=std::make_unique<CttsRuntime>(row.at("device").dump(),std::move(ops)); actual["startup"]=state(); } catch(const TtsError &e) { actual["startup"]={{"error",{{"category",e.category},{"message",e.what()}}}}; }
        if(owner) for(const auto &operation:row.at("operations")) { ++operations; Json item=Json::object(); const auto kind=operation.at("kind").get<std::string>();
            try { if(kind=="concurrent") {
                auto task=std::async(std::launch::async,[&] { try { const auto audio=owner->generate_sample(operation.at("request").dump()); item["value"]={{"samples",audio.samples},{"rate",audio.rate},{"channels",audio.channels}}; } catch(const TtsError &e) { item["error"]={{"category",e.category},{"message",e.what()}}; } });
                const auto ready=starting.wait_for(std::chrono::seconds(2)); if(ready!=std::future_status::ready) released.set_value(); expect(ready==std::future_status::ready,"generation reached cross-thread gate"); const auto begin=std::chrono::steady_clock::now(); owner->close(); const auto elapsed=std::chrono::steady_clock::now()-begin; released.set_value(); task.get(); expect(elapsed<std::chrono::seconds(1),"close does not block on generation");
            } else if(kind=="warmup") item["value"]=Json::parse(owner->warmup_generation_kwargs()); else if(kind=="generate") { const auto audio=owner->generate_sample(operation.at("request").dump()); item["value"]={{"samples",audio.samples},{"rate",audio.rate},{"channels",audio.channels}}; } else if(kind=="release") owner->release_generation_memory(); else if(kind=="reload") owner->reload_on_cpu(); else if(kind=="close") owner->close(); else if(kind=="env") environment.update(operation.at("values")); else throw std::runtime_error("unknown operation"); }
            catch(const TtsError &e) { item["error"]={{"category",e.category},{"message",e.what()}}; } catch(const std::exception &e) { item["error"]={{"category","C++ exception"},{"message",e.what()}}; }
            item["state"]=state(); actual["operations"].push_back(std::move(item));
        }
        actual["events"]=events;
        try { compare_json(actual,row.at("expected"),std::to_string(i)+" "+row.at("name").get<std::string>()); } catch(const std::exception &e) { if(errors++<25) std::cerr<<e.what()<<'\n'; }
    }
    expect(rows.size()-1==rows[0].at("cases") && operations==rows[0].at("operations"),"complete CTTS lifecycle comparisons"); expect(!errors,std::to_string(errors)+" lifecycle scenarios differ"); std::cout<<rows.size()-1<<" unchanged CTTS constructor/model lifecycle scenarios and "<<operations<<" operations match\n"; return 0;
} catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; } }
