#include "ezlocalai/stt_diarization.hpp"
#include "ezlocalai/ctts_audio.hpp"
#include "ezlocalai/whisper_features.hpp"
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <openssl/rand.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <dlfcn.h>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>

namespace ezlocalai {
namespace {
using Json = nlohmann::json;
// libsoxr's public stream API defaults to float32, HQ and one CPU thread.
// https://github.com/chirlu/soxr/blob/master/src/soxr.h
struct Soxr {
    using Create = void *(*)(double,double,unsigned,const char **,const void *,const void *,const void *);
    using Process = const char *(*)(void *,const void *,size_t,size_t *,void *,size_t,size_t *);
    using Delete = void (*)(void *);
    void *library = dlopen("libsoxr.so.0", RTLD_NOW|RTLD_LOCAL);
    Create create; Process process; Delete destroy;
    Soxr() {
        if (!library) throw std::runtime_error("Diarization requires libsoxr");
        create = reinterpret_cast<Create>(dlsym(library,"soxr_create"));
        process = reinterpret_cast<Process>(dlsym(library,"soxr_process"));
        destroy = reinterpret_cast<Delete>(dlsym(library,"soxr_delete"));
        if (!create || !process || !destroy) throw std::runtime_error("Incomplete libsoxr API");
    }
};
std::vector<float> resample(const std::vector<float> &input, int rate) {
    if (rate == 16000 || input.empty()) return input;
    if (rate <= 0) throw std::invalid_argument("Invalid audio rate");
    static Soxr api; const char *error = nullptr;
    std::unique_ptr<void,Soxr::Delete> state(api.create(rate,16000,1,&error,nullptr,nullptr,nullptr),api.destroy);
    if (error || !state) throw std::runtime_error(error ? error : "Cannot create audio resampler");
    const size_t expected = std::ceil(input.size()*16000.0/rate);
    std::vector<float> result(expected+1); size_t read = 0, written = 0, count = 0;
    error = api.process(state.get(),input.data(),input.size(),&read,result.data(),result.size(),&written);
    if (error || read != input.size()) throw std::runtime_error("Cannot resample diarization audio");
    error = api.process(state.get(),nullptr,0,nullptr,result.data()+written,result.size()-written,&count);
    if (error) throw std::runtime_error(error);
    result.resize(written+count); result.resize(expected,0.0f); // librosa fix_length uses ceil.
    return result;
}
struct Merge { int left, right, count; double distance; };
// Nearest-neighbor chain, stable distance ordering and tree traversal follow
// SciPy's BSD-licensed hierarchy implementation; see licenses/SciPy.txt.
std::vector<Merge> hierarchy(const std::vector<SttSpeakerFeature> &features) {
    const size_t n = features.size();
    if (n > size_t(INT32_MAX)/2 || n > std::numeric_limits<size_t>::max()/n) throw std::length_error("Too many diarization segments");
    std::vector<double> distance(n*(n-1)/2);
    const auto at = [&](int a,int b) -> double & {
        if (a>b) std::swap(a,b);
        return distance[n*size_t(a)-size_t(a)*(a+1)/2+size_t(b-a-1)];
    };
    for (size_t a=0;a<n;++a) for (size_t b=0;b<a;++b) {
        double dot=0, aa=0, bb=0;
        for (size_t j=0;j<40;++j) { dot+=double(features[a][j])*features[b][j]; aa+=double(features[a][j])*features[a][j]; bb+=double(features[b][j])*features[b][j]; }
        const auto value=1-dot/std::sqrt(aa*bb);
        if (!std::isfinite(value)) throw std::invalid_argument("Diarization features contain nonfinite distances");
        at(a,b)=at(b,a)=std::clamp(value,0.0,2.0);
    }
    std::vector<int> size(n,1), chain; std::vector<Merge> merges;
    for (size_t step=0;step+1<n;++step) {
        if (chain.empty()) { size_t i=0;while (!size[i])++i;chain.push_back(i); }
        int x=0,y=0;double closest=0;
        for (;;) {
            x=chain.back(); closest=std::numeric_limits<double>::infinity();
            if (chain.size()>1) { y=chain[chain.size()-2];closest=at(x,y); }
            for (size_t i=0;i<n;++i) if (size[i] && int(i)!=x && at(x,i)<closest) { closest=at(x,i);y=i; }
            if (chain.size()>1 && y==chain[chain.size()-2]) break;
            chain.push_back(y);
        }
        chain.resize(chain.size()-2); if (x>y) std::swap(x,y);
        const auto nx=size[x],ny=size[y]; merges.push_back({x,y,nx+ny,closest}); size[x]=0;size[y]=nx+ny;
        for (size_t i=0;i<n;++i) if (size[i] && int(i)!=y) at(i,y)=at(y,i)=(nx*at(i,x)+ny*at(i,y))/(nx+ny);
    }
    std::stable_sort(merges.begin(),merges.end(),[](const auto &a,const auto &b) {return a.distance<b.distance;});
    std::vector<int> parent(2*n-1);std::iota(parent.begin(),parent.end(),0);
    const auto find = [&](int x) { while (parent[x]!=x)x=parent[x];return x; };
    for (size_t i=0;i<merges.size();++i) {
        const auto x=find(merges[i].left),y=find(merges[i].right);
        merges[i].left=std::min(x,y);merges[i].right=std::max(x,y);parent[x]=parent[y]=n+i;
    }
    return merges;
}
bool truth(const Json &value) { if(value.is_null())return false;if(value.is_boolean())return value.get<bool>();if(value.is_number())return value.get<double>()!=0;return !value.empty(); }
std::string random_id() {
    unsigned char bytes[6];if(RAND_bytes(bytes,sizeof(bytes))!=1)throw std::runtime_error("Cannot create speaker ID");
    static const char hex[]="0123456789abcdef";std::string result;for(auto byte:bytes){result+=hex[byte>>4];result+=hex[byte&15];}return result;
}
}
std::vector<float> stt_diarization_audio(const std::string &path) {
    std::ifstream input(path,std::ios::binary);if(!input)throw std::runtime_error("Cannot open diarization audio");
    const std::string bytes{std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>()};
    auto [audio,rate]=soundfile_mono_float32(bytes);
    for (auto sample : audio.samples) if (!std::isfinite(sample)) throw std::invalid_argument("Nonfinite diarization audio");
    return resample(audio.samples,rate);
}
SttSpeakerFeature stt_diarization_feature(const std::vector<float> &audio) {
    if(audio.empty())throw std::invalid_argument("Empty diarization segment");
    for (auto sample : audio) if (!std::isfinite(sample)) throw std::invalid_argument("Nonfinite diarization audio");
    constexpr size_t fft=2048,bands=128,hop=512,bins=1025;
    const size_t frames=1+audio.size()/hop;const double pi=std::acos(-1.0);
    std::vector<float> power(bins*frames),filters(bands*bins),mel(bands*frames);
    cv::Mat input(1,fft,CV_64F),output;
    for(size_t frame=0;frame<frames;++frame){
        for(size_t j=0;j<fft;++j){const auto index=int64_t(frame*hop+j)-int64_t(fft/2);input.at<double>(0,j)=(index>=0&&size_t(index)<audio.size()?audio[index]:0.0f)*(.5-.5*std::cos(2*pi*j/fft));}
        cv::dft(input,output,cv::DFT_COMPLEX_OUTPUT);
        for(size_t j=0;j<bins;++j){const auto value=output.at<cv::Vec2d>(0,j);const float magnitude=std::hypot(float(value[0]),float(value[1]));power[j*frames+frame]=magnitude*magnitude;}
    }
    std::array<double,bands+2> frequency{};const double maximum=15+std::log(8.0)/(std::log(6.4)/27);
    for(size_t i=0;i<frequency.size();++i){const double m=maximum*i/(bands+1);frequency[i]=m<15?m*(200.0/3):1000*std::exp((m-15)*(std::log(6.4)/27));}
    for(size_t band=0;band<bands;++band)for(size_t bin=0;bin<bins;++bin){const auto hz=bin*16000.0/fft;const float triangle=std::max(0.0,std::min((hz-frequency[band])/(frequency[band+1]-frequency[band]),(frequency[band+2]-hz)/(frequency[band+2]-frequency[band+1])));filters[band*bins+bin]=triangle*(2.0/(frequency[band+2]-frequency[band]));}
    float maximum_db=-std::numeric_limits<float>::infinity();
    for(size_t band=0;band<bands;++band)for(size_t frame=0;frame<frames;++frame){float value=0;for(size_t bin=0;bin<bins;++bin)value+=filters[band*bins+bin]*power[bin*frames+frame];const float db=10.0f*std::log10(std::max(1e-10f,value));mel[band*frames+frame]=db;maximum_db=std::max(maximum_db,db);}
    for(auto &value:mel)value=std::max(value,maximum_db-80.0f);
    SttSpeakerFeature result{};std::vector<float> coefficients(frames);
    for(size_t c=0;c<20;++c){
        for(size_t frame=0;frame<frames;++frame){double value=0;for(size_t band=0;band<bands;++band)value+=mel[band*frames+frame]*std::cos(pi/bands*(band+.5)*c);coefficients[frame]=value*std::sqrt((c?2.0:1.0)/bands);}
        float mean=0;for(auto value:coefficients)mean+=value;mean/=float(frames);result[c]=mean;
        float variance=0;for(auto value:coefficients){const float delta=value-mean;variance+=delta*delta;}result[c+20]=std::sqrt(variance/float(frames));
    }
    return result;
}
std::vector<int> stt_diarization_labels(const std::vector<SttSpeakerFeature> &features,std::optional<int> speakers) {
    const int n=features.size();if(n<2)return std::vector<int>(n,1);const auto tree=hierarchy(features);
    int wanted=speakers.value_or(0);
    if(wanted<=0){
        wanted=1;const int depth=std::min(3,n-1);
        for(int i=n-2;i>=0;--i){
            double sum=0,squares=0;int count=0;
            const auto visit=[&](auto &&self,int node,int level)->void{const auto &m=tree[node];if(level+1<depth){if(m.left>=n)self(self,m.left-n,level+1);if(m.right>=n)self(self,m.right-n,level+1);}sum+=m.distance;squares+=m.distance*m.distance;++count;};visit(visit,i,0);
            const auto variance=count<2?(squares-sum*sum)/count:(squares-sum*sum/count)/(count-1);
            const auto inconsistency=variance>0?(tree[i].distance-sum/count)/std::sqrt(variance):0;
            if(inconsistency>1)++wanted;else break;
        }
    }
    std::vector<int> labels(n);if(wanted>=n){std::iota(labels.begin(),labels.end(),1);return labels;}
    // Average-linkage heights are monotonic. Choose the smallest cut producing
    // at most the requested number of clusters, including tied merge heights.
    const double cutoff=tree[n-wanted-1].distance;
    int next=0;
    const auto visit=[&](auto &&self,int node,int label)->void{
        if(node<n){labels[node]=label?label:++next;return;}
        const auto &m=tree[node-n];if(!label&&m.distance<=cutoff)label=++next;
        // SciPy assigns internal children before singleton children.
        if(m.left>=n)self(self,m.left,label);if(m.right>=n)self(self,m.right,label);
        if(m.left<n)self(self,m.left,label);if(m.right<n)self(self,m.right,label);
    };visit(visit,2*n-2,0);return labels;
}
struct SttDiarizer::State {
    struct Print { std::string id;SttSpeakerFeature centroid;int64_t count; };
    std::map<std::string,std::vector<Print>> prints;std::map<std::string,double> timestamps;
    std::mutex mutex;std::function<double()> now;std::function<std::string()> uuid;double ttl;
    std::string match(const std::string &session,const SttSpeakerFeature &feature,int count){
        const auto time=now();for(auto it=timestamps.begin();it!=timestamps.end();)if(time-it->second>ttl){prints.erase(it->first);it=timestamps.erase(it);}else ++it;
        timestamps[session]=time;auto &values=prints[session];double best=-1;size_t selected=values.size();
        for(size_t i=0;i<values.size();++i){float dot=0,a=0,b=0;for(size_t j=0;j<40;++j){dot+=feature[j]*values[i].centroid[j];a+=feature[j]*feature[j];b+=values[i].centroid[j]*values[i].centroid[j];}const float norm=std::sqrt(a)*std::sqrt(b);if(norm>0&&dot/norm>best){best=dot/norm;selected=i;}}
        if(best<.7){values.push_back({uuid(),feature,count});return values.back().id;}
        auto &value=values[selected];for(size_t j=0;j<40;++j)value.centroid[j]=(value.centroid[j]*float(value.count)+feature[j]*float(count))/float(value.count+count);value.count+=count;return value.id;
    }
};
SttDiarizer::SttDiarizer(std::function<double()> now,std::function<std::string()> uuid,double ttl):state_(std::make_unique<State>()){
    state_->now=now?std::move(now):[]{return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();};state_->uuid=uuid?std::move(uuid):random_id;state_->ttl=ttl;
}
SttDiarizer::~SttDiarizer()=default;
std::string SttDiarizer::diarize(const std::string &path,const std::string &raw,const std::string &speaker_json,const std::string &session_json){
    auto segments=Json::parse(raw);const auto requested=Json::parse(speaker_json),session=Json::parse(session_json);
    const auto fallback=[&]{for(auto &segment:segments)segment["speaker"]="SPEAKER_00";return segments.dump();};
    std::vector<float> audio;try{audio=stt_diarization_audio(path);}catch(const std::exception &){return fallback();}
    std::vector<SttSpeakerFeature> features;std::vector<size_t> indices;
    const auto index=[&](double seconds){auto value=int64_t(seconds*16000);if(value<0)value+=audio.size();return size_t(std::clamp<int64_t>(value,0,audio.size()));};
    for(size_t i=0;i<segments.size();++i){const auto begin=index(segments[i].at("start").get<double>()),end=index(segments[i].at("end").get<double>());if(end<begin||end-begin<1600)continue;try{features.push_back(stt_diarization_feature({audio.begin()+begin,audio.begin()+end}));indices.push_back(i);}catch(const std::exception &){}}
    if(features.empty())return fallback();
    const bool persistent=truth(session);if(features.size()==1){if(!persistent)return fallback();std::lock_guard<std::mutex> lock(state_->mutex);const auto id=state_->match(session.get<std::string>(),features[0],1);for(auto &segment:segments)segment["speaker"]="SPK_"+id;return segments.dump();}
    const auto labels=stt_diarization_labels(features,truth(requested)?std::optional<int>(requested.is_boolean()?int(requested.get<bool>()):requested.get<int>()):std::nullopt);
    std::map<int,std::string> names;
    if(persistent){std::lock_guard<std::mutex> lock(state_->mutex);std::map<int,std::pair<SttSpeakerFeature,int>> clusters;
        for(size_t i=0;i<features.size();++i){auto &entry=clusters[labels[i]];for(size_t j=0;j<40;++j)entry.first[j]+=features[i][j];++entry.second;}
        for(auto &[label,entry]:clusters){for(auto &value:entry.first)value/=float(entry.second);names[label]="SPK_"+state_->match(session.get<std::string>(),entry.first,entry.second);}
    }else for(auto label:labels)names[label]="SPEAKER_"+(label<=10?std::string("0"):std::string())+std::to_string(std::max(0,label-1));
    for(size_t i=0;i<indices.size();++i)segments[indices[i]]["speaker"]=names[labels[i]];
    for(auto &segment:segments)if(!segment.contains("speaker"))segment["speaker"]=persistent?names[1]:"SPEAKER_00";
    return segments.dump();
}
std::string stt_diarize(const std::string &path,const std::string &segments,const std::string &speakers,const std::string &session){static SttDiarizer diarizer;return diarizer.diarize(path,segments,speakers,session);}
}
