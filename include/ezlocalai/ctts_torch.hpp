#pragma once
#include "ezlocalai/ctts_runtime.hpp"
#include "ezlocalai/ctts_audio.hpp"
#include "ezlocalai/resource_manager.hpp"
namespace ezlocalai {
// Optional LibTorch 2.9.1 hardware/memory/inference context for CTTS.
CttsOperations ctts_torch_operations(CttsOperations operations={});
// Attach the current process's LibTorch allocator diagnostics and cleanup.
ResourceProbes torch_resource_probes(ResourceProbes probes = native_resource_probes());
// torchaudio 2.9.1 Resample defaults: float64 Hann kernel cached as float32.
CttsAudioArray ctts_torch_resample(const CttsAudioArray &audio,int original_rate,int target_rate);
}
