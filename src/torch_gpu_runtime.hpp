#pragma once

#if defined(EZLOCALAI_TORCH_CUDA) || defined(EZLOCALAI_TORCH_HIP)
#define EZLOCALAI_TORCH_GPU 1

#ifdef EZLOCALAI_TORCH_HIP
#include <ATen/hip/HIPContextLight.h>
#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPFunctions.h>
#else
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#endif

namespace ezlocalai::torch_gpu {
#ifdef EZLOCALAI_TORCH_HIP
// PyTorch's public CUDA tensors use this allocator and guard on ROCm.
namespace allocator = c10::hip::HIPCachingAllocatorMasqueradingAsCUDA;
inline auto device_count() { return c10::hip::device_count(); }
inline void synchronize() { c10::hip::device_synchronize(); }
inline void memory_info(c10::DeviceIndex device, size_t *free, size_t *total) {
    const c10::hip::HIPGuardMasqueradingAsCUDA guard(device);
    C10_HIP_CHECK(hipMemGetInfo(free, total));
}
#else
namespace allocator = c10::cuda::CUDACachingAllocator;
inline auto device_count() { return c10::cuda::device_count(); }
inline void synchronize() { c10::cuda::device_synchronize(); }
inline void memory_info(c10::DeviceIndex device, size_t *free, size_t *total) {
    const c10::cuda::CUDAGuard guard(device);
    C10_CUDA_CHECK(cudaMemGetInfo(free, total));
}
#endif
inline auto device_properties(c10::DeviceIndex device) {
    return at::cuda::getDeviceProperties(device);
}
}
#endif
