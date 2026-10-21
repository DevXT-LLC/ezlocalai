#include "ezlocalai/video_runtime.hpp"
#include "ezlocalai/ctts_torch.hpp"
#include "ezlocalai/stt_torch.hpp"
#include <ATen/Context.h>
#include <c10/core/InferenceMode.h>
#include "torch_gpu_runtime.hpp"
namespace ezlocalai {
namespace {
template<class F> auto translated(F action) { return [action](auto &&...args)->decltype(auto) { try { return action(std::forward<decltype(args)>(args)...); } catch(const TtsError &) { throw; } catch(const SttError &e) { throw TtsError(e.category,e.what(),e.runtime_family); } catch(const c10::OutOfMemoryError &e) { throw TtsError("OutOfMemoryError",e.what_without_backtrace()); } catch(const c10::Error &e) { throw TtsError("RuntimeError",e.what_without_backtrace()); } }; }
void initialize() {
#ifdef EZLOCALAI_TORCH_GPU
    at::globalContext().lazyInitDevice(at::kCUDA);
#else
    throw TtsError("AssertionError","Torch not compiled with CUDA enabled");
#endif
}
int index(const std::string &value) {
    long long parsed=-1; try { parsed=std::stoll(value); } catch(...) { throw TtsError("AssertionError","Invalid device id"); }
#ifdef EZLOCALAI_TORCH_GPU
    if(parsed<0 || parsed>=torch_gpu::device_count() || parsed>127) throw TtsError("AssertionError","Invalid device id");
#else
    throw TtsError("AssertionError","Torch not compiled with CUDA enabled");
#endif
    return static_cast<int>(parsed);
}
}
ResourceProbes torch_resource_probes(ResourceProbes probes) {
    const auto operations = ctts_torch_operations();
    probes.allocator_stats = [operations] {
        return collect_cuda_allocator_stats({operations.cuda_available,
            [] {
#ifdef EZLOCALAI_TORCH_GPU
                return static_cast<int>(torch_gpu::device_count());
#else
                return 0;
#endif
            },
            // Same free/total as torch.cuda.mem_get_info, from NVML: the torch
            // query would create a CUDA context (~260 MiB) in an idle worker.
            [](int i) { const auto memory = cuda_memory_info(i); return std::pair<double, double>{static_cast<double>(memory.free_bytes), static_cast<double>(memory.total_bytes)}; },
            [operations](int i) { return operations.allocated_bytes(std::to_string(i)); },
            [](int i) -> double {
#ifdef EZLOCALAI_TORCH_GPU
                if (torch_gpu::allocator::get()->initialized())
                    return torch_gpu::allocator::getDeviceStats(i).reserved_bytes[static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE)].current;
#endif
                return 0;
            },
            [](int i) -> double {
#ifdef EZLOCALAI_TORCH_GPU
                if (torch_gpu::allocator::get()->initialized())
                    return torch_gpu::allocator::getDeviceStats(i).allocated_bytes[static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE)].peak;
#endif
                return 0;
            }
        });
    };
    probes.release_unused_cuda_memory = [operations] {
        if (operations.cuda_available()) operations.empty_cache();
    };
    return probes;
}
CttsOperations ctts_torch_operations(CttsOperations ops) {
    const auto stt=stt_torch_operations(); if(!ops.cuda_available) ops.cuda_available=translated(stt.cuda_available); if(!ops.collect) ops.collect=translated(stt.collect); if(!ops.empty_cache) ops.empty_cache=translated(stt.empty_cache); if(!ops.synchronize) ops.synchronize=translated(stt.synchronize); if(!ops.ipc_collect) ops.ipc_collect=translated(stt.ipc_collect);
    if(!ops.memory_info) ops.memory_info=translated([](const std::string &device) { initialize(); const auto selected=index(device); size_t free=0,total=0;
#ifdef EZLOCALAI_TORCH_GPU
        torch_gpu::memory_info(static_cast<c10::DeviceIndex>(selected), &free, &total);
#endif
        return std::pair<double,double>{static_cast<double>(free),static_cast<double>(total)};
    });
    if(!ops.total_bytes) ops.total_bytes=translated([](const std::string &device)->double { initialize(); const auto selected=index(device);
#ifdef EZLOCALAI_TORCH_GPU
        return torch_gpu::device_properties(selected)->totalGlobalMem;
#else
        return 0;
#endif
    });
    if(!ops.allocated_bytes) ops.allocated_bytes=translated([](const std::string &device)->double {
#ifdef EZLOCALAI_TORCH_GPU
        if(torch_gpu::allocator::get()->initialized()) return torch_gpu::allocator::getDeviceStats(index(device)).allocated_bytes[static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE)].current;
#endif
        return 0;
    });
    if(!ops.inference_mode) ops.inference_mode=[] { auto guard=std::make_shared<c10::InferenceMode>(true); return [guard=std::move(guard)](const std::string &) mutable { guard.reset(); }; };
    return ops;
}
}
