#pragma once
#include "ezlocalai/stt_runtime.hpp"
namespace ezlocalai {
// Native LibTorch implementation of the Torch operations used by STT.py.
// Supplied callbacks take precedence. Requires the optional LibTorch target.
SttRuntimeOperations stt_torch_operations(SttRuntimeOperations operations = {});
}
