// Feature extraction follows faster-whisper 1.0.3/1.2.0 (MIT). Complex magnitude
// follows NumPy's SIMD arithmetic (BSD). See licenses/faster-whisper.txt and NumPy.txt.
#include "ezlocalai/whisper_features.hpp"
#include <opencv2/core.hpp>
#include <cblas.h>
#include <dlfcn.h>
#ifndef EZLOCALAI_OPENBLAS_LIBRARY
#define EZLOCALAI_OPENBLAS_LIBRARY "libopenblas.so.0"
#endif
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libswresample/swresample.h>
}
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

namespace ezlocalai {
namespace {
decltype(&cblas_sgemm) whisper_sgemm() {
    // NumPy/faster-whisper uses OpenBLAS, while Torch uses its bundled MKL.
    // A global OpenBLAS dependency interposes Torch's BLAS symbols and changes
    // MFCCs (and potentially other models). Bind this kernel privately instead.
    static const auto function = [] {
        int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_DEEPBIND
        flags |= RTLD_DEEPBIND;
#endif
        void *library = dlopen(EZLOCALAI_OPENBLAS_LIBRARY, flags);
        if (!library) throw std::runtime_error("Cannot load Whisper OpenBLAS library");
        void *symbol = dlsym(library, "cblas_sgemm");
        if (!symbol) { dlclose(library); throw std::runtime_error("Missing Whisper OpenBLAS SGEMM"); }
        return reinterpret_cast<decltype(&cblas_sgemm)>(symbol);
    }();
    return function;
}
void av_check(int code, const char *stage) {
    if (code >= 0) return;
    char error[AV_ERROR_MAX_STRING_SIZE]; av_strerror(code,error,sizeof(error));
    throw std::runtime_error(std::string(stage)+": "+error);
}
struct AudioDecoder {
    AVFormatContext *format = nullptr;
    AVCodecContext *codec = nullptr;
    AVFrame *frame = av_frame_alloc();
    AVPacket *packet = av_packet_alloc();
    SwrContext *resampler = nullptr;
    ~AudioDecoder() { swr_free(&resampler); av_packet_free(&packet); av_frame_free(&frame); avcodec_free_context(&codec); avformat_close_input(&format); }
};
size_t reflected(int64_t position, size_t length) {
    if (length == 1) return 0;
    const auto period = static_cast<int64_t>(2 * (length - 1));
    auto index = position % period; if (index < 0) index += period;
    return static_cast<size_t>(index < static_cast<int64_t>(length) ? index : period - index);
}
float numpy_complex_absolute(float real,float imaginary) {
    real=std::abs(real);imaginary=std::abs(imaginary);
    if(std::isinf(real)||std::isinf(imaginary))return std::numeric_limits<float>::infinity();
    if(std::isnan(real)||std::isnan(imaginary))return std::numeric_limits<float>::quiet_NaN();
    const auto larger=std::max(real,imaginary),smaller=std::min(real,imaginary);
    const auto ratio=larger==0?0.0f:smaller/larger;
#if defined(__x86_64__) || defined(__i386__)
    const bool fused=__builtin_cpu_supports("fma");
#elif defined(__aarch64__)
    const bool fused=true;
#else
    const bool fused=false;
#endif
    return std::sqrt(fused?std::fma(ratio,ratio,1.0f):ratio*ratio+1.0f)*larger;
}
}
std::vector<float> whisper_decode_audio(const std::string &path, int sample_rate) {
    if (sample_rate <= 0) throw std::invalid_argument("Audio sample rate must be positive");
    AudioDecoder decoder;
    if (!decoder.frame || !decoder.packet) throw std::bad_alloc();
    av_check(avformat_open_input(&decoder.format,path.c_str(),nullptr,nullptr),"Open audio");
    av_check(avformat_find_stream_info(decoder.format,nullptr),"Read audio streams");
    int stream = -1;
    for (unsigned i = 0; i < decoder.format->nb_streams; ++i) if (decoder.format->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) { stream = i; break; }
    if (stream < 0) throw std::runtime_error("No audio stream");
    const auto parameters = decoder.format->streams[stream]->codecpar;
    const auto codec = avcodec_find_decoder(parameters->codec_id);
    if (!codec) throw std::runtime_error("Unsupported audio codec");
    decoder.codec = avcodec_alloc_context3(codec); if (!decoder.codec) throw std::bad_alloc();
    av_check(avcodec_parameters_to_context(decoder.codec,parameters),"Audio codec parameters");
    av_check(avcodec_open2(decoder.codec,codec,nullptr),"Open audio decoder");
    std::vector<float> audio;
    const auto resample = [&](AVFrame *frame) {
        if (!decoder.resampler) {
            if (!frame) return;
            AVChannelLayout mono = AV_CHANNEL_LAYOUT_MONO;
            av_check(swr_alloc_set_opts2(&decoder.resampler,&mono,AV_SAMPLE_FMT_S16,sample_rate,
                &frame->ch_layout,static_cast<AVSampleFormat>(frame->format),frame->sample_rate,0,nullptr),"Create audio resampler");
            av_check(swr_init(decoder.resampler),"Initialize audio resampler");
        }
        const auto capacity = swr_get_out_samples(decoder.resampler,frame ? frame->nb_samples : 0);
        av_check(capacity,"Audio resample capacity");
        std::vector<int16_t> samples(std::max(1,capacity));
        auto *output = reinterpret_cast<uint8_t *>(samples.data());
        const auto count = swr_convert(decoder.resampler,&output,capacity,frame ? const_cast<const uint8_t **>(frame->extended_data) : nullptr,frame ? frame->nb_samples : 0);
        av_check(count,"Resample audio");
        for (int i = 0; i < count; ++i) audio.push_back(samples[i] / 32768.0f);
    };
    const auto receive = [&] {
        for (;;) {
            const auto code = avcodec_receive_frame(decoder.codec,decoder.frame);
            if (code == AVERROR(EAGAIN) || code == AVERROR_EOF) return;
            if (code == AVERROR_INVALIDDATA) continue;
            av_check(code,"Decode audio frame"); resample(decoder.frame); av_frame_unref(decoder.frame);
        }
    };
    for (;;) {
        const auto code = av_read_frame(decoder.format,decoder.packet);
        if (code == AVERROR_EOF) break;
        if (code == AVERROR_INVALIDDATA) continue;
        av_check(code,"Read audio packet");
        if (decoder.packet->stream_index == stream) {
            const auto sent = avcodec_send_packet(decoder.codec,decoder.packet);
            if (sent != AVERROR_INVALIDDATA) { av_check(sent,"Send audio packet"); receive(); }
        }
        av_packet_unref(decoder.packet);
    }
    av_check(avcodec_send_packet(decoder.codec,nullptr),"Flush audio decoder"); receive(); resample(nullptr);
    return audio;
}

WhisperFeatures whisper_log_mel(const std::vector<float> &audio, size_t bands, size_t padding,
    int sample_rate, size_t n_fft, size_t hop, WhisperFeatureProfile profile) {
    const bool legacy = profile == WhisperFeatureProfile::faster_whisper_1_0_3;
    if (legacy && n_fft % 2) throw std::invalid_argument("FFT size must greater or equal the frame size");
    if (!bands || !n_fft || !hop || sample_rate <= 0 || bands > INT32_MAX || n_fft > INT32_MAX)
        throw std::invalid_argument("Invalid Whisper feature dimensions");
    if (padding > SIZE_MAX - audio.size()) throw std::length_error("Whisper audio too large");
    const auto length = audio.size() + padding;
    if (!length) throw std::invalid_argument("Cannot reflect-pad empty audio");
    const auto total = length + 2 * (n_fft / 2);
    if (total < n_fft) throw std::invalid_argument("Whisper FFT exceeds input length");
    const auto frames = (total - n_fft) / hop; // Drop the final centered STFT column.
    if (!frames || frames > INT32_MAX) throw std::invalid_argument("Empty or oversized Whisper spectrogram");
    const auto bins = n_fft / 2 + 1;
    std::vector<double> window(n_fft);
    std::vector<float> magnitudes(bins * frames), filters(bands * bins);
    const double pi = std::acos(-1.0);
    for (size_t i = 0; i < n_fft; ++i) {
        const auto value = 0.5 + 0.5 * std::cos(pi * (2.0 * i - n_fft) / n_fft);
        window[i] = legacy ? value : static_cast<double>(static_cast<float>(value));
    }
    cv::Mat input(1,n_fft,CV_64F), transformed;
    for (size_t frame = 0; frame < frames; ++frame) {
        for (size_t i = 0; i < n_fft; ++i) {
            const auto index = reflected(static_cast<int64_t>(frame * hop + i) - static_cast<int64_t>(n_fft / 2),length);
            const auto sample = index < audio.size() ? audio[index] : 0.0f;
            input.at<double>(0,i) = legacy ? sample * window[i] : static_cast<double>(sample * static_cast<float>(window[i]));
        }
        cv::dft(input,transformed,cv::DFT_COMPLEX_OUTPUT);
        for (size_t bin = 0; bin < bins; ++bin) {
            const auto value = transformed.at<cv::Vec2d>(0,bin);
            const auto magnitude = numpy_complex_absolute(static_cast<float>(value[0]),static_cast<float>(value[1]));
            magnitudes[bin * frames + frame] = magnitude * magnitude;
        }
    }
    std::vector<double> frequencies(bands + 2);
    const auto mel_step = 45.245640471924965 / (bands + 1), log_step = std::log(6.4) / 27;
    for (size_t i = 0; i < frequencies.size(); ++i) {
        const auto mel = i * mel_step;
        frequencies[i] = mel >= 15 ? 1000 * std::exp(log_step * (mel - 15)) : (200.0 / 3) * mel;
    }
    for (size_t band = 0; band < bands; ++band) for (size_t bin = 0; bin < bins; ++bin) {
        const auto frequency = bin * (1.0 / (n_fft * (1.0 / sample_rate)));
        const auto lower = (frequency - frequencies[band]) / (frequencies[band + 1] - frequencies[band]);
        const auto upper = (frequencies[band + 2] - frequency) / (frequencies[band + 2] - frequencies[band + 1]);
        auto weight = std::max(0.0,std::min(lower,upper));
        if (legacy) weight = static_cast<float>(weight);
        filters[band * bins + bin] = static_cast<float>(weight * (2 / (frequencies[band + 2] - frequencies[band])));
    }
    WhisperFeatures result{bands,frames,std::vector<float>(bands * frames)};
    whisper_sgemm()(CblasRowMajor,CblasNoTrans,CblasNoTrans,bands,frames,bins,1,filters.data(),bins,magnitudes.data(),frames,0,result.values.data(),frames);
    float maximum = -std::numeric_limits<float>::infinity();
    for (auto &value : result.values) { value = std::log10(std::max(value,1e-10f)); maximum = std::max(maximum,value); }
    for (auto &value : result.values) value = (std::max(value,maximum - 8) + 4) / 4;
    return result;
}
WhisperFeatures whisper_log_mel_legacy(const std::vector<float> &audio,size_t bands,bool padding,size_t chunk_seconds) {
    return whisper_log_mel(audio,bands,padding ? chunk_seconds * 16000 : 0,16000,400,160,WhisperFeatureProfile::faster_whisper_1_0_3);
}
WhisperFeatures whisper_feature_window(const WhisperFeatures &input,size_t start,size_t count,size_t padded_frames) {
    if (input.values.size() != input.bands * input.frames || start > input.frames) throw std::invalid_argument("Invalid Whisper feature window");
    WhisperFeatures output{input.bands,padded_frames,std::vector<float>(input.bands * padded_frames)};
    const auto copied = std::min({count,padded_frames,input.frames - start});
    for (size_t band = 0; band < input.bands; ++band) std::copy_n(input.values.data() + band * input.frames + start,copied,output.values.data() + band * padded_frames);
    return output;
}
}
