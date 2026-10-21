#include "ezlocalai/stt_torch.hpp"
#include "whisper_transcription_test.hpp"
#include <ATen/Context.h>
#include <ATen/CPUGeneratorImpl.h>
#include <torch/types.h>
#include "../src/torch_gpu_runtime.hpp"
using namespace transcription_test;
Json flags() {
    auto &c = at::globalContext();
    return Json::array({c.userEnabledCuDNN(),c.benchmarkCuDNN(),c.benchmarkLimitCuDNN(),c.deterministicCuDNN(),c.allowTF32CuDNN(),c.float32Precision("cuda","all")});
}
void flags(const Json &value) {
    auto &c = at::globalContext(); c.setUserEnabledCuDNN(value[0]); c.setBenchmarkCuDNN(value[1]); c.setBenchmarkLimitCuDNN(value[2]);
    c.setDeterministicCuDNN(value[3]); c.setAllowTF32CuDNN(value[4]); c.setFloat32Precision("cuda","all",value[5]);
}
std::string rng(const at::Generator &generator) {
    auto state = generator.get_state(); return hash(std::string(static_cast<const char *>(state.data_ptr()),state.numel()));
}
Json snapshot() {
    const auto stats = ezlocalai::torch_gpu::allocator::getDeviceStats(0);
    return {{"allocated",stats.allocated_bytes[0].current},{"reserved",stats.reserved_bytes[0].current},{"flags",flags()},
        {"cpu_rng",rng(at::detail::getDefaultCPUGenerator())},{"cuda_rng",rng(at::globalContext().defaultGenerator(at::Device(at::kCUDA,0)))}};
}
struct Backend : SttBackend { torch::Tensor tensor; std::shared_ptr<SttBackend> retained; };
int main(int argc,char **argv) { try {
    expect(argc == 3,"fixtures and --gpu/--unavailable"); const auto root = fs::path(argv[1]);
    auto native = stt_torch_operations();
    if (std::string(argv[2]) == "--unavailable") {
#ifdef EZLOCALAI_TORCH_HIP
        const auto expected = Json::parse(read(root/"stt_torch_hip_unavailable.json")); Json actual;
#else
        const auto expected = Json::parse(read(root/"stt_torch_unavailable.json")); Json actual;
#endif
        actual["cuda"] = native.cuda_available(); actual["cudnn"] = native.cudnn_available(); actual["allocated"] = native.cuda_allocated_bytes();
        native.empty_cache();
        for (const auto &item : std::vector<std::pair<std::string,std::function<void()>>>{{"total",[&] { native.cuda_total_bytes(); }},{"synchronize",native.synchronize},{"ipc_collect",native.ipc_collect}}) {
            try { item.second(); actual[item.first] = nullptr; }
            catch (const SttError &error) { actual[item.first] = {{"category",error.category},{"message",error.what()}}; }
        }
        compare_json(actual,expected,"hidden GPU native operations"); std::cout << "Hidden GPU operations match source\n"; return 0;
    }
    expect(native.cuda_available() && native.cudnn_available(),"CUDA and cuDNN available for explicitly enabled GPU tests");
    const auto rows = records(root/"stt_torch_cases.jsonl"); const auto &metadata = rows.at(0);
    expect(metadata.at("torch") == "2.9.1+cu128" && metadata.at("cudnn") == at::Context::versionRuntimeCuDNN(),"pinned Torch/cuDNN versions: compiled=" + std::to_string(at::Context::versionCuDNN()) + " runtime=" + std::to_string(at::Context::versionRuntimeCuDNN()));
    expect(metadata.at("source_sha256") == "d0dd8340c243d7ff9ba84f4e3babe3c41b0b0ce7c7dddf2565eb886ba3f21fe2","source identity");
    expect(native.cuda_total_bytes() == metadata.at("total").get<double>(),"matching GPU memory capacity");
    expect(rows.size() == 19,"all GPU cases");
    for (size_t index = 1; index < rows.size(); ++index) {
        const auto &row = rows[index]; native.empty_cache(); at::manual_seed(metadata.at("seed").get<int64_t>()); flags(row.at("flags"));
        torch::Tensor held; const auto count = row.at("held").get<int64_t>(); if (count) held = torch::empty({count},torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        auto operations = native; Json events = Json::array();
        const auto event = [&](const std::string &name,Json fields = Json::object()) { fields["event"] = name; fields["state"] = snapshot(); events.push_back(fields); };
        operations.empty_cache = [&] { event("empty_before"); native.empty_cache(); event("empty_after"); };
        operations.collect = [&] { event("collect"); };
        operations.load = [&](const auto &name,const auto &cache,const auto &device,const auto &compute,int64_t threads) {
            event("load",{{"model",name},{"root",cache},{"device",device},{"compute",compute},{"threads",threads}});
            if (row.at("mode") == "fallback" && device == "cuda") throw SttError("RuntimeError","CUDA fixture constructor failure");
            auto backend = std::make_shared<Backend>(); backend->tensor = torch::empty({4097},torch::TensorOptions().dtype(torch::kUInt8).device(torch::Device(device))); return backend;
        };
        operations.batch = [&](const auto &model) { event("batch"); auto backend = std::make_shared<Backend>(); backend->retained = model; return backend; };
        operations.open_audio = [&] { event("audio"); return std::make_shared<int>(1); };
        operations.terminate_audio = [&](const auto &) { event("terminate"); };
        const auto label = "GPU case " + std::to_string(index - 1); compare_json(snapshot(),row.at("before"),label + ".before");
        expect((operations.cuda_total_bytes() - operations.cuda_allocated_bytes()) / (1024 * 1024) == row.at("vram"),label + " available memory");
        Json value;
        if (row.at("mode") == "probe") {
            auto output = operations.test_cudnn(); operations.empty_cache(); output.reset(); value = true;
        } else {
            SttRuntime runtime("fixture",operations); value = {{"device",runtime.device()},{"compute",runtime.compute_type()}};
            event("loaded"); runtime.close(); event("closed"); runtime.close(); event("closed_twice");
        }
        compare_json(value,row.at("value"),label + ".value"); compare_json(events,row.at("events"),label + ".events");
        compare_json(snapshot(),row.at("after"),label + ".after"); held = torch::Tensor(); native.empty_cache(); std::cout << label << " matches\n";
    }
    std::cout << "18 real GPU initialization/probe/fallback cases match source memory, RNG, flags and cleanup\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
