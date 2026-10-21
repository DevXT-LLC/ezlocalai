#include "ezlocalai/stt_torch.hpp"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <torch/nn/modules/conv.h>
#include <torch/types.h>
#include "torch_gpu_runtime.hpp"
#ifdef EZLOCALAI_TORCH_GPU
#include <torch/csrc/CudaIPCTypes.h>
#endif

namespace ezlocalai {
namespace {
template<class F> auto translated(F action) {
    return [action]() -> decltype(action()) {
        try { return action(); }
        catch (const c10::OutOfMemoryError &error) { throw SttError("OutOfMemoryError",error.what_without_backtrace()); }
        catch (const c10::Error &error) { throw SttError("RuntimeError",error.what_without_backtrace()); }
    };
}
struct CudnnFlags {
    at::Context &context = at::globalContext();
    bool enabled = context.userEnabledCuDNN(), benchmark = context.benchmarkCuDNN();
    int limit = context.benchmarkLimitCuDNN();
    bool deterministic = context.deterministicCuDNN(), tf32 = context.allowTF32CuDNN();
    std::string precision = context.float32Precision("cuda","all");
    void set(bool e,bool b,int l,bool d,bool t,const std::string &p) {
        context.setUserEnabledCuDNN(e); context.setBenchmarkCuDNN(b); context.setBenchmarkLimitCuDNN(l);
        context.setDeterministicCuDNN(d); context.setAllowTF32CuDNN(t); context.setFloat32Precision("cuda","all",p);
    }
    CudnnFlags() { set(true,false,10,false,true,"none"); }
    void restore() { set(enabled,benchmark,limit,deterministic,tf32,precision); }
};
void initialize_cuda() {
#ifdef EZLOCALAI_TORCH_GPU
    at::globalContext().lazyInitDevice(at::kCUDA);
#else
    throw SttError("AssertionError","Torch not compiled with CUDA enabled");
#endif
}
}
SttRuntimeOperations stt_torch_operations(SttRuntimeOperations operations) {
    if (!operations.cuda_available) operations.cuda_available = translated([] { return torch::cuda::is_available(); });
    if (!operations.cudnn_available) operations.cudnn_available = [] {
#ifdef EZLOCALAI_TORCH_HIP
        // torch/csrc/Module.cpp sets Python's _has_cudnn for USE_ROCM too.
        // ATen's hasCuDNN() alone returns false in this same distribution.
        return true;
#else
        return at::Context::hasCuDNN();
#endif
    };
    if (!operations.cuda_total_bytes) operations.cuda_total_bytes = translated([]() -> double {
        initialize_cuda();
#ifdef EZLOCALAI_TORCH_GPU
        return torch_gpu::device_properties(0)->totalGlobalMem;
#else
        return 0;
#endif
    });
    if (!operations.cuda_allocated_bytes) operations.cuda_allocated_bytes = translated([]() -> double {
#ifdef EZLOCALAI_TORCH_GPU
        if (torch_gpu::allocator::get()->initialized())
            return torch_gpu::allocator::getDeviceStats(0).allocated_bytes[static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE)].current;
#endif
        return 0;
    });
    if (!operations.test_cudnn) operations.test_cudnn = translated([]() -> std::shared_ptr<void> {
        auto input = torch::randn({1,1,8,8},torch::TensorOptions().device(torch::kCUDA));
        torch::nn::Conv2d conv(torch::nn::Conv2dOptions(1,1,3).padding(1)); conv->to(torch::kCUDA);
        torch::Tensor output;
        CudnnFlags flags;
        try { output = conv->forward(input); }
        catch (...) { flags.restore(); throw; }
        flags.restore();
        return std::make_shared<torch::Tensor>(std::move(output));
    });
    if (!operations.collect) operations.collect = [] {}; // Native shared owners release on scope exit.
    if (!operations.empty_cache) operations.empty_cache = translated([] {
#ifdef EZLOCALAI_TORCH_GPU
        if (torch_gpu::allocator::get()->initialized()) torch_gpu::allocator::emptyCache();
#endif
    });
    if (!operations.synchronize) operations.synchronize = translated([] {
        initialize_cuda();
#ifdef EZLOCALAI_TORCH_GPU
        torch_gpu::synchronize();
#endif
    });
    if (!operations.ipc_collect) operations.ipc_collect = translated([] {
        initialize_cuda();
#ifdef EZLOCALAI_TORCH_GPU
        torch::CudaIPCCollect();
#endif
    });
    return operations;
}
}
