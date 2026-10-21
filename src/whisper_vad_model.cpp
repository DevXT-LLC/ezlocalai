// Native execution of faster-whisper's Silero VAD preprocessing and models.
#include "ezlocalai/whisper_vad.hpp"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <filesystem>
#include <stdexcept>

namespace ezlocalai {
struct WhisperVadModel::Impl {
    WhisperVadProfile profile;
    Ort::Env environment{ORT_LOGGING_LEVEL_FATAL,"ezlocalai-vad"};
    Ort::SessionOptions options;
    Ort::Session encoder{nullptr}, decoder{nullptr};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    Impl(WhisperVadProfile selected,const std::string &directory) : profile(selected) {
        options.SetInterOpNumThreads(1); options.SetIntraOpNumThreads(1); options.SetLogSeverityLevel(4);
        const auto root = std::filesystem::path(directory);
        if (profile == WhisperVadProfile::faster_whisper_1_2_0) {
            options.DisableCpuMemArena();
            encoder = Ort::Session(environment,(root/"silero_encoder_v5.onnx").c_str(),options);
            decoder = Ort::Session(environment,(root/"silero_decoder_v5.onnx").c_str(),options);
        } else decoder = Ort::Session(environment,(root/"silero_vad.onnx").c_str(),options);
    }
    Ort::Value tensor(float *data,size_t count,const std::vector<int64_t> &shape) const {
        return Ort::Value::CreateTensor<float>(memory,data,count,shape.data(),shape.size());
    }
    static const float *output(const Ort::Value &value,size_t size) {
        if (!value.IsTensor() || value.GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            value.GetTensorTypeAndShapeInfo().GetElementCount() != size) throw std::runtime_error("Unexpected Silero VAD output shape");
        return value.GetTensorData<float>();
    }
    std::vector<float> modern(const std::vector<float> &audio) {
        const size_t windows = audio.size() / 512 + 1;
        std::vector<float> padded(windows * 512);
        std::copy(audio.begin(),audio.end(),padded.begin());
        // NumPy's context slice is a view: context[:, -1] = 0 also zeros
        // the final audio window's last 64 samples before concatenation.
        std::fill(padded.end() - 64,padded.end(),0);
        std::vector<float> encoded(windows * 128);
        const char *input_names[] = {"input"};
        auto encoder_output_name = encoder.GetOutputNameAllocated(0,Ort::AllocatorWithDefaultOptions());
        const char *output_names[] = {encoder_output_name.get()};
        for (size_t begin = 0; begin < windows; begin += 10000) {
            const auto count = std::min<size_t>(10000,windows - begin);
            std::vector<float> batch(count * 576);
            for (size_t i = 0; i < count; ++i) {
                const auto position = (begin + i) * 512;
                if (position) std::copy_n(padded.data() + position - 64,64,batch.data() + i * 576);
                std::copy_n(padded.data() + position,512,batch.data() + i * 576 + 64);
            }
            auto input = tensor(batch.data(),batch.size(),{static_cast<int64_t>(count),576});
            auto result = encoder.Run(Ort::RunOptions{nullptr},input_names,&input,1,output_names,1);
            std::copy_n(output(result[0],count * 128),count * 128,encoded.data() + begin * 128);
        }
        std::array<float,256> state{};
        std::vector<float> probabilities; probabilities.reserve(windows);
        const char *decoder_inputs[] = {"input","state"};
        auto probability_name = decoder.GetOutputNameAllocated(0,Ort::AllocatorWithDefaultOptions());
        auto state_name = decoder.GetOutputNameAllocated(1,Ort::AllocatorWithDefaultOptions());
        const char *decoder_outputs[] = {probability_name.get(),state_name.get()};
        for (size_t i = 0; i < windows; ++i) {
            std::array<Ort::Value,2> inputs{tensor(encoded.data() + i * 128,128,{1,128}),tensor(state.data(),state.size(),{2,1,128})};
            auto result = decoder.Run(Ort::RunOptions{nullptr},decoder_inputs,inputs.data(),inputs.size(),decoder_outputs,2);
            probabilities.push_back(output(result[0],1)[0]);
            std::copy_n(output(result[1],state.size()),state.size(),state.data());
        }
        return probabilities;
    }
    std::vector<float> legacy(const std::vector<float> &audio) {
        std::array<float,256> state{};
        std::array<float,576> input{};
        std::array<float,64> context{};
        std::vector<float> probabilities; probabilities.reserve((audio.size() + 511) / 512);
        const char *input_names[] = {"input","state","sr"};
        auto probability_name = decoder.GetOutputNameAllocated(0,Ort::AllocatorWithDefaultOptions());
        auto state_name = decoder.GetOutputNameAllocated(1,Ort::AllocatorWithDefaultOptions());
        const char *output_names[] = {probability_name.get(),state_name.get()};
        int64_t rate = 16000;
        for (size_t begin = 0; begin < audio.size(); begin += 512) {
            input.fill(0); std::copy(context.begin(),context.end(),input.begin());
            std::copy_n(audio.data() + begin,std::min<size_t>(512,audio.size() - begin),input.data() + 64);
            std::array<Ort::Value,3> inputs{tensor(input.data(),input.size(),{1,576}),tensor(state.data(),state.size(),{2,1,128}),
                Ort::Value::CreateTensor<int64_t>(memory,&rate,1,nullptr,0)};
            auto result = decoder.Run(Ort::RunOptions{nullptr},input_names,inputs.data(),inputs.size(),output_names,2);
            probabilities.push_back(output(result[0],1)[0]);
            std::copy_n(output(result[1],state.size()),state.size(),state.data());
            std::copy(input.end() - 64,input.end(),context.begin());
        }
        return probabilities;
    }
};
WhisperVadModel::WhisperVadModel(WhisperVadProfile profile,const std::string &directory)
    : impl_(std::make_unique<Impl>(profile,directory)) {}
WhisperVadModel::~WhisperVadModel() = default;
WhisperVadModel::WhisperVadModel(WhisperVadModel &&) noexcept = default;
WhisperVadModel &WhisperVadModel::operator=(WhisperVadModel &&) noexcept = default;
std::vector<float> WhisperVadModel::probabilities(const std::vector<float> &audio) const {
    return impl_->profile == WhisperVadProfile::faster_whisper_1_2_0 ? impl_->modern(audio) : impl_->legacy(audio);
}
std::vector<WhisperSpeechChunk> WhisperVadModel::speech_timestamps(const std::vector<float> &audio,const WhisperVadOptions &options) const {
    return whisper_speech_timestamps(probabilities(audio),audio.size(),options,impl_->profile);
}
WhisperVadAudio WhisperVadModel::filter(const std::vector<float> &audio,const WhisperVadOptions &options) const {
    WhisperVadAudio result;
    result.chunks = speech_timestamps(audio,options);
    if (impl_->profile == WhisperVadProfile::faster_whisper_1_0_3) result.audio = whisper_collect_chunks_legacy(audio,result.chunks);
    else {
        auto collected = whisper_collect_chunks(audio,result.chunks);
        for (auto &chunk : collected) result.audio.insert(result.audio.end(),chunk.audio.begin(),chunk.audio.end());
    }
    return result;
}
}
