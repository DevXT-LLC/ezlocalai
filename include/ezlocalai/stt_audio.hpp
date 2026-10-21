#pragma once
#include "ezlocalai/stt_runtime.hpp"

namespace ezlocalai {
// Implements the PyAudio 0.2.14 initialization/termination operations used by
// STT. No stream is opened. Keep the handle until explicit termination; dropping
// it alone does not terminate PortAudio, matching PyAudio's object lifetime.
SttRuntimeOperations stt_portaudio_operations(SttRuntimeOperations operations = {},
                                            const std::string &library = "libportaudio.so.2");
}
