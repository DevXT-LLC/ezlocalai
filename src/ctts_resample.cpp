// Port of torchaudio's default sinc resampling; see licenses/torchaudio.txt.
#include "ezlocalai/ctts_torch.hpp"
#include <ATen/ATen.h>
#include <cmath>
#include <cstring>
#include <numeric>
namespace ezlocalai {
CttsAudioArray ctts_torch_resample(const CttsAudioArray &input,int original_rate,int target_rate) {
    if(original_rate<=0 || target_rate<=0) throw TtsError("ValueError","Sample rate must be positive");
    const auto audio=ctts_contiguous_audio(input); if(original_rate==target_rate) return audio;
    const int divisor=std::gcd(original_rate,target_rate),original=original_rate/divisor,target=target_rate/divisor;
    const double base=std::min(original,target)*0.99,pi=std::acos(-1.0); const int64_t width=static_cast<int64_t>(std::ceil(6.0*original/base));
    const auto index=at::arange(-width,width+original,at::TensorOptions().dtype(at::kDouble)).view({1,1,-1})/original;
    auto time=at::arange(0,-target,-1,at::TensorOptions().dtype(at::kLong)).to(at::kFloat).view({-1,1,1})/target+index;
    time*=base; time.clamp_(-6,6); const auto window=at::cos(time*pi/6/2).pow(2); time*=pi;
    auto kernel=at::where(time==0,at::ones({},time.options()),time.sin()/time); kernel*=window*(base/original); kernel=kernel.to(at::kFloat);
    std::vector<int64_t> shape(audio.shape.begin(),audio.shape.end()); auto waveform=at::from_blob(const_cast<float *>(audio.samples.data()),shape,at::TensorOptions().dtype(at::kFloat)).clone();
    const auto length=shape.back(); waveform=waveform.view({-1,length}); const auto channels=waveform.size(0); waveform=at::constant_pad_nd(waveform,{width,width+original},0);
    auto output=at::conv1d(waveform.unsqueeze(1),kernel,std::nullopt,at::IntArrayRef{original},at::IntArrayRef{0},at::IntArrayRef{1},1).transpose(1,2).reshape({channels,-1});
    const auto target_length=static_cast<int64_t>(std::ceil(static_cast<float>(double(target)*length/original))); output=output.slice(-1,0,target_length);
    shape.back()=output.size(-1); output=output.reshape(shape).contiguous(); CttsAudioArray result; result.shape.assign(shape.begin(),shape.end()); result.samples.resize(output.numel()); if(!result.samples.empty()) std::memcpy(result.samples.data(),output.const_data_ptr<float>(),result.samples.size()*sizeof(float)); return result;
}
}
