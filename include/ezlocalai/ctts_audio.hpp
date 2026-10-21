#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>
#include <utility>
namespace ezlocalai {
// Float32 array storage with NumPy-style element strides. Empty strides mean
// row-major contiguous storage; an empty shape denotes a scalar.
struct CttsAudioArray {
    std::vector<float> samples;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> strides;
    ptrdiff_t offset=0;
};
using CttsResampler=std::function<CttsAudioArray(const CttsAudioArray &,int,int)>;
CttsAudioArray ctts_contiguous_audio(const CttsAudioArray &audio);
CttsAudioArray ctts_mono_float32(const CttsAudioArray &audio);
// soundfile.read(dtype="float32"), then NumPy's channel mean, before any
// resampling. The returned array is one-dimensional and retains source rate.
std::pair<CttsAudioArray, int> soundfile_mono_float32(const std::string &bytes);
std::string ctts_pcm16_bytes(const CttsAudioArray &audio);
// Matches CTTS's caught conversion failure: empty bytes for array-to-WAV,
// nullopt for bytes-to-tensor or a failure before tensor-to-WAV delegation.
std::string ctts_wav_bytes(const CttsAudioArray &audio,int64_t sample_rate);
std::optional<CttsAudioArray> ctts_bytes_to_tensor(const std::string &bytes,int64_t sample_rate,const CttsResampler &resample={});
std::optional<std::string> ctts_tensor_to_bytes(const CttsAudioArray &tensor,int64_t sample_rate);
}
