#include "ezlocalai/ctts_audio.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#ifndef EZLOCALAI_SNDFILE_LIBRARY
#define EZLOCALAI_SNDFILE_LIBRARY "libsndfile.so.1"
#endif
namespace ezlocalai {
namespace {
size_t count(const std::vector<size_t> &shape) {
    size_t n=1; for(auto dim:shape) { if(dim && n>size_t(PTRDIFF_MAX)/dim) throw std::overflow_error("Audio array too large"); n*=dim; } return n;
}
CttsAudioArray layout(const CttsAudioArray &input) {
    auto out=input; if(out.strides.empty()) { out.strides.resize(out.shape.size()); ptrdiff_t stride=1; for(size_t i=out.shape.size();i--;) { out.strides[i]=stride; if(out.shape[i] && size_t(stride)>size_t(PTRDIFF_MAX)/out.shape[i]) throw std::overflow_error("Audio stride overflow"); stride*=out.shape[i]; } }
    if(out.strides.size()!=out.shape.size()) throw std::invalid_argument("Audio shape/stride mismatch"); count(out.shape); return out;
}
float sample(const CttsAudioArray &audio,ptrdiff_t position) { if(position<0 || size_t(position)>=audio.samples.size()) throw std::out_of_range("Audio view exceeds storage"); return audio.samples[position]; }
float add(float a,float b) { return a+b; }
// NumPy 1.26.4 pairwise reduction order (loops_utils.h.src); see licenses/NumPy.txt.
float pairwise(const CttsAudioArray &audio,ptrdiff_t offset,size_t n,ptrdiff_t stride) {
    if(n<8) { float sum=-0.0f; for(size_t i=0;i<n;++i) sum=add(sum,sample(audio,offset+ptrdiff_t(i)*stride)); return sum; }
    if(n<=128) { float sums[8]; for(size_t i=0;i<8;++i) sums[i]=sample(audio,offset+ptrdiff_t(i)*stride); size_t i=8; for(;i<n-n%8;i+=8) for(size_t j=0;j<8;++j) sums[j]=add(sums[j],sample(audio,offset+ptrdiff_t(i+j)*stride)); float sum=add(add(add(sums[0],sums[1]),add(sums[2],sums[3])),add(add(sums[4],sums[5]),add(sums[6],sums[7]))); for(;i<n;++i) sum=add(sum,sample(audio,offset+ptrdiff_t(i)*stride)); return sum; }
    size_t half=n/2; half-=half%8; return add(pairwise(audio,offset,half,stride),pairwise(audio,offset+ptrdiff_t(half)*stride,n-half,stride));
}
struct Info { int64_t frames; int samplerate,channels,format,sections,seekable; };
struct VirtualIo { int64_t (*length)(void *); int64_t (*seek)(int64_t,int,void *); int64_t (*read)(void *,int64_t,void *); int64_t (*write)(const void *,int64_t,void *); int64_t (*tell)(void *); };
struct Buffer {
    std::string bytes; int64_t position=0; bool failed=false;
    static int64_t length(void *p) { return static_cast<Buffer *>(p)->bytes.size(); }
    static int64_t tell(void *p) { return static_cast<Buffer *>(p)->position; }
    static int64_t seek(int64_t offset,int whence,void *p) { auto &b=*static_cast<Buffer *>(p); const int64_t base=whence==0?0:whence==1?b.position:whence==2?int64_t(b.bytes.size()):-1; if(base<0 || (offset>0 && base>INT64_MAX-offset) || offset < -base) return -1; b.position=base+offset; return b.position; }
    static int64_t read(void *target,int64_t n,void *p) { auto &b=*static_cast<Buffer *>(p); if(n<=0 || uint64_t(b.position)>=b.bytes.size()) return 0; n=std::min(n,int64_t(b.bytes.size())-b.position); std::memcpy(target,b.bytes.data()+b.position,n); b.position+=n; return n; }
    static int64_t write(const void *source,int64_t n,void *p) { auto &b=*static_cast<Buffer *>(p); if(n<0 || b.position>INT64_MAX-n) { b.failed=true; return 0; } try { if(uint64_t(b.position+n)>b.bytes.size()) b.bytes.resize(b.position+n); if(n) std::memcpy(b.bytes.data()+b.position,source,n); b.position+=n; return n; } catch(...) { b.failed=true; return 0; } }
};
struct Sndfile {
    void *library=dlopen(EZLOCALAI_SNDFILE_LIBRARY,RTLD_NOW|RTLD_LOCAL);
    using Open=void *(*)(VirtualIo *,int,Info *,void *); using Read=int64_t (*)(void *,double *,int64_t); using Write=int64_t (*)(void *,const float *,int64_t); using Close=int (*)(void *); using Command=int (*)(void *,int,void *,int); using Error=int (*)(void *);
    using ReadFloat=int64_t (*)(void *,float *,int64_t);
    Open open; Read read; Write write; Close close; Command command; Error error;
    Sndfile() { if(!library) throw std::runtime_error("Cannot load libsndfile.so.1"); open=reinterpret_cast<Open>(dlsym(library,"sf_open_virtual")); read=reinterpret_cast<Read>(dlsym(library,"sf_readf_double")); write=reinterpret_cast<Write>(dlsym(library,"sf_writef_float")); close=reinterpret_cast<Close>(dlsym(library,"sf_close")); command=reinterpret_cast<Command>(dlsym(library,"sf_command")); error=reinterpret_cast<Error>(dlsym(library,"sf_error")); if(!open || !read || !write || !close || !command || !error) throw std::runtime_error("Incomplete libsndfile ABI"); }
};
Sndfile &sndfile() { static Sndfile api; return api; }
struct File { Sndfile &api; void *handle; ~File() { if(handle) api.close(handle); } };
VirtualIo virtual_io{Buffer::length,Buffer::seek,Buffer::read,Buffer::write,Buffer::tell};
int rate(int64_t value) { if(value<INT32_MIN || value>INT32_MAX) throw std::overflow_error("Sample rate outside C int range"); return static_cast<int>(value); }
}
CttsAudioArray ctts_contiguous_audio(const CttsAudioArray &input) {
    const auto audio=layout(input); CttsAudioArray out; out.shape=audio.shape.empty()?std::vector<size_t>{1}:audio.shape; const auto n=count(audio.shape); out.samples.reserve(n);
    for(size_t i=0;i<n;++i) { size_t remainder=i; ptrdiff_t position=audio.offset; for(size_t d=audio.shape.size();d--;) { position+=ptrdiff_t(remainder%audio.shape[d])*audio.strides[d]; remainder/=audio.shape[d]; } out.samples.push_back(sample(audio,position)); } return out;
}
CttsAudioArray ctts_mono_float32(const CttsAudioArray &input) {
    auto audio=layout(input); if(audio.shape.size()!=2) return ctts_contiguous_audio(audio);
    if(audio.shape[0]<=2 && audio.shape[1]>audio.shape[0]) { std::swap(audio.shape[0],audio.shape[1]); std::swap(audio.strides[0],audio.strides[1]); }
    const auto frames=audio.shape[0],channels=audio.shape[1]; CttsAudioArray out; out.shape={frames}; out.samples.resize(frames);
    for(size_t frame=0;frame<frames;++frame) { const auto offset=audio.offset+ptrdiff_t(frame)*audio.strides[0]; float sum=0.0f;
        if(std::abs(audio.strides[1])<=std::abs(audio.strides[0])) sum=add(sum,pairwise(audio,offset,channels,audio.strides[1]));
        else for(size_t channel=0;channel<channels;++channel) sum=add(sum,sample(audio,offset+ptrdiff_t(channel)*audio.strides[1]));
        out.samples[frame]=sum/static_cast<float>(channels);
    } return out;
}
std::pair<CttsAudioArray, int> soundfile_mono_float32(const std::string &bytes) {
    Buffer buffer{bytes}; Info info{}; auto &api = sndfile();
    File file{api, api.open(&virtual_io, 0x10, &info, &buffer)};
    if (!file.handle || info.frames < 0 || info.channels < 1) throw std::runtime_error("Cannot read audio");
    const auto read_float = reinterpret_cast<Sndfile::ReadFloat>(dlsym(api.library, "sf_readf_float"));
    if (!read_float) throw std::runtime_error("Incomplete libsndfile float32 ABI");
    CttsAudioArray interleaved;
    interleaved.samples.resize(count({size_t(info.frames), size_t(info.channels)}));
    const auto frames = read_float(file.handle, interleaved.samples.data(), info.frames);
    if (frames < 0 || api.error(file.handle) || buffer.failed) throw std::runtime_error("Cannot decode audio");
    interleaved.samples.resize(size_t(frames) * info.channels);
    if (info.channels == 1) { interleaved.shape = {size_t(frames)}; return {std::move(interleaved), info.samplerate}; }
    CttsAudioArray mono; mono.shape = {size_t(frames)}; mono.samples.resize(frames);
    for (size_t frame = 0; frame < size_t(frames); ++frame)
        mono.samples[frame] = add(0.0f, pairwise(interleaved, frame * info.channels, info.channels, 1)) / static_cast<float>(info.channels);
    return {std::move(mono), info.samplerate};
}
std::string ctts_pcm16_bytes(const CttsAudioArray &input) {
    const auto audio=ctts_mono_float32(input); std::string bytes; bytes.reserve(audio.samples.size()*2);
    for(auto value:audio.samples) { if(std::isnan(value)) value=0; value=std::max(-1.0f,std::min(1.0f,value)); const auto sample=static_cast<int16_t>(value*32767.0f); bytes+=static_cast<char>(uint16_t(sample)&255); bytes+=static_cast<char>(uint16_t(sample)>>8); } return bytes;
}
std::string ctts_wav_bytes(const CttsAudioArray &input,int64_t sample_rate) {
    try { const auto audio=ctts_mono_float32(input); if(audio.shape.size()!=1) return {}; Buffer buffer; Info info{}; info.samplerate=rate(sample_rate); info.channels=1; info.format=0x010000|0x0002; auto &api=sndfile(); File file{api,api.open(&virtual_io,0x20,&info,&buffer)}; if(!file.handle) return {}; api.command(file.handle,4288,nullptr,1); const auto n=api.write(file.handle,audio.samples.data(),audio.samples.size()); const auto error=api.error(file.handle); const auto closed=api.close(file.handle); file.handle=nullptr; if(n!=int64_t(audio.samples.size()) || error || closed || buffer.failed) return {}; return std::move(buffer.bytes); } catch(const std::exception &) { return {}; }
}
std::optional<CttsAudioArray> ctts_bytes_to_tensor(const std::string &bytes,int64_t sample_rate,const CttsResampler &resample) {
    try { Buffer buffer{bytes}; Info info{}; auto &api=sndfile(); File file{api,api.open(&virtual_io,0x10,&info,&buffer)}; if(!file.handle || info.frames<0 || info.channels<1) return {}; const auto n=count({size_t(info.frames),size_t(info.channels)}); std::vector<double> values(n); const auto frames=api.read(file.handle,values.data(),info.frames); if(frames<0 || api.error(file.handle) || buffer.failed) return {}; CttsAudioArray out; out.shape={size_t(info.channels),size_t(frames)}; out.samples.resize(size_t(frames)*info.channels);
        for(size_t frame=0;frame<size_t(frames);++frame) for(int channel=0;channel<info.channels;++channel) out.samples[size_t(channel)*frames+frame]=static_cast<float>(values[frame*info.channels+channel]);
        if(info.samplerate!=sample_rate) { if(!resample) return {}; out=resample(out,info.samplerate,rate(sample_rate)); } return out;
    } catch(const std::exception &) { return {}; }
}
std::optional<std::string> ctts_tensor_to_bytes(const CttsAudioArray &input,int64_t sample_rate) {
    try { auto tensor=layout(input); if(tensor.shape.empty()) return {}; if(tensor.shape.size()==1) { tensor.shape.insert(tensor.shape.begin(),1); tensor.strides.insert(tensor.strides.begin(),ptrdiff_t(tensor.shape[1])*tensor.strides[0]); } if(tensor.shape[0]<=2) { std::reverse(tensor.shape.begin(),tensor.shape.end()); std::reverse(tensor.strides.begin(),tensor.strides.end()); } return ctts_wav_bytes(tensor,sample_rate); } catch(const std::exception &) { return {}; }
}
}
