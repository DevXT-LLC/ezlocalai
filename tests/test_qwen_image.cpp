#include "ezlocalai/qwen_image.hpp"
#include "ezlocalai/image_processing.hpp"
#include "ezlocalai/stt_input.hpp"
#include "ezlocalai/process.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <iomanip>
#include <sstream>
#include <future>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <poll.h>
#include <png.h>

namespace fs = std::filesystem;
using json = nlohmann::json;
std::string read(const fs::path &path) { std::ifstream in(path, std::ios::binary); return {std::istreambuf_iterator<char>(in), {}}; }
void write(const fs::path &path, const std::string &value) { fs::create_directories(path.parent_path()); std::ofstream out(path, std::ios::binary); out << value; }
void expect(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
std::string hex(const unsigned char *data, size_t size) { std::ostringstream out; for (size_t i=0;i<size;++i) out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(data[i]);return out.str(); }
std::string sha(const std::string &s) { unsigned char digest[32]; SHA256(reinterpret_cast<const unsigned char *>(s.data()),s.size(),digest);return hex(digest,32); }
std::string normalize(std::string s) {
    const std::string models = std::getenv("SDCPP_MODELS_DIR"), binary = std::getenv("SDCPP_BIN");
    if (s == binary) return "SDCLI";
    if (s.rfind(models,0)==0) s.replace(0,models.size(),"MODELS");
    if(s.find("/qwen-image-")!=std::string::npos) s="TMP/"+fs::path(s).filename().string();
    return s;
}
int engine(int argc,char **argv) {
    const auto test=json::parse(read(std::getenv("QWEN_FIXTURE_CASE")));
    json captured={{"argv",json::array()},{"reference_pixels",json::array()},{"paths",json::array()}};
    std::string output;
    for(int i=0;i<argc;++i) {
        captured["argv"].push_back(normalize(argv[i]));
        if(std::string(argv[i])=="-o") output=argv[i+1];
        if(std::string(argv[i])=="-r") {
            const auto img=ezlocalai::decode_rgb_image(read(argv[i+1]));
            captured["reference_pixels"].push_back({{"size",{img.width,img.height}},{"pixels",hex(img.pixels.data(),img.pixels.size())}});
            captured["paths"].push_back(argv[i+1]);
        }
    }
    captured["paths"].push_back(output);
    write(std::getenv("QWEN_FIXTURE_CAPTURE"),captured.dump());
    if (test.contains("delay_ms")) std::this_thread::sleep_for(std::chrono::milliseconds(test.at("delay_ms").get<int>()));
    const auto mode=test.value("mode","ok");
    if(mode!="missing") write(output,ezlocalai::stt_base64_decode_bytes(test.at("engine_output")));
    if(mode=="exit")return 7;
    if(mode=="signal") { raise(SIGTERM); return 99; }
    return 0;
}
class TrackedBackend : public ezlocalai::InferenceBackend {
public:
    std::atomic<bool> loaded{true}; std::atomic<int> unloads{0}, restores{0};
    std::string chat(const std::string &) override { return "{\"choices\":[]}"; }
    std::string completion(const std::string &body) override { return chat(body); }
    std::string embedding(const std::string &) override { return "{}"; }
    std::string loaded_models_json() const override { return loaded ? "{\"llm\":{\"name\":\"fixture\"}}" : "{}"; }
    std::function<void(bool)> unload_llms_for_media() override { ++unloads; loaded = false; return [this](bool reload) { ++restores; loaded = reload; }; }
};
class ImagePeer {
public:
    std::string url, posted_path, posted_body, image_bytes;
    ImagePeer() {
        fd=socket(AF_INET,SOCK_STREAM|SOCK_CLOEXEC,0);expect(fd>=0,"image peer socket");
        sockaddr_in address{};address.sin_family=AF_INET;address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
        expect(bind(fd,reinterpret_cast<sockaddr *>(&address),sizeof(address))==0,"image peer bind");
        socklen_t size=sizeof(address);getsockname(fd,reinterpret_cast<sockaddr *>(&address),&size);
        url="http://127.0.0.1:"+std::to_string(ntohs(address.sin_port));expect(listen(fd,8)==0,"image peer listen");
    }
    void start() { worker=std::thread([this] {
        while(!stopped) {
            pollfd poller{fd,POLLIN,0};if(poll(&poller,1,100)<=0)continue;
            const int peer=accept4(fd,nullptr,nullptr,SOCK_CLOEXEC);if(peer<0)continue;
            timeval timeout{3,0};setsockopt(peer,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
            std::string raw;char buffer[4096];size_t header;
            while((header=raw.find("\r\n\r\n"))==std::string::npos){const auto n=recv(peer,buffer,sizeof(buffer),0);if(n<=0)break;raw.append(buffer,n);}
            std::istringstream first(raw);std::string method,path;first>>method>>path;
            const auto length_pos=raw.find("Content-Length:");const size_t length=length_pos==std::string::npos?0:std::stoull(raw.substr(length_pos+15));
            if(header!=std::string::npos)while(raw.size()<header+4+length){const auto n=recv(peer,buffer,sizeof(buffer),0);if(n<=0)break;raw.append(buffer,n);}
            std::string body=path=="/ref"?image_bytes:R"({"free_vram_gb":24,"alive_workers":1,"data":[{"url":"http://peer/outputs/edit.png"}]})";
            if(method=="POST"){posted_path=path;posted_body=header==std::string::npos?"":raw.substr(header+4);}
            const auto response="HTTP/1.1 200 OK\r\nContent-Type: "+std::string(path=="/ref"?"image/png":"application/json")+"\r\nContent-Length: "+std::to_string(body.size())+"\r\nConnection: close\r\n\r\n"+body;
            send(peer,response.data(),response.size(),MSG_NOSIGNAL);shutdown(peer,SHUT_RDWR);close(peer);
        }
    }); }
    void stop(){stopped=true;shutdown(fd,SHUT_RDWR);if(worker.joinable())worker.join();}
    ~ImagePeer(){stop();close(fd);}
private:
    int fd;std::atomic<bool> stopped{false};std::thread worker;
};
void api_cases(const json &cases,const json &fixture,const fs::path &executable,const fs::path &temp) {
    for (const auto &entry : ezlocalai::environment_defaults) unsetenv(entry.first.data());
    setenv("EZLOCALAI_DOTENV_PATH","",1);setenv("DEFAULT_MODEL","none",1);setenv("ROUTER_DRAIN_FILE","",1);
    setenv("TTS_ENABLED","false",1);setenv("STT_ENABLED","false",1);setenv("EMBEDDING_ENABLED","false",1);
    setenv("IMAGE_ENABLED","true",1);setenv("IMG_MODEL","qwen-image-2.1",1);setenv("EZLOCALAI_URL","",1);
    setenv("SDCPP_MODELS_DIR",(temp/"models").c_str(),1);setenv("SDCPP_BIN","/missing/sd-cli",1);
    write(temp/"devices","");
    ezl_env_config config{};ezl_load_config(&config);
    ezlocalai::NativeServer unavailable(config,std::make_unique<ezlocalai::EchoInferenceBackend>("model"));
    for(const auto &test:cases.at("cases")) {
        const auto response=unavailable.handle_request("POST","/v1/images/edits","",test.at("body").dump());
        if(test.contains("error")) expect(response.status==422 && json::parse(response.body)==test.at("error"),"API "+test.at("name").get<std::string>()+": "+response.body+" expected "+test.at("error").dump());
        else expect(response.status==503,"valid image edit failed schema validation: "+response.body);
    }
    setenv("SDCPP_BIN",executable.c_str(),1);
    auto sample=fixture.at("cases")[0];write(temp/"case.json",sample.dump());
    ezlocalai::NativeServer server(config,std::make_unique<ezlocalai::EchoInferenceBackend>("model"));
    auto response=server.handle_request("POST","/v1/images/generations","",R"({"prompt":"test","size":"32x64","response_format":"b64_json","n":2})");
    expect(response.status==200,"local image API: "+response.body);
    const auto data=json::parse(response.body).at("data");expect(data.size()==2&&data[0].at("b64_json")==sample.at("engine_output"),"image API outputs/count");
    auto resources=json::parse(server.handle_request("GET","/v1/resources","","").body);
    expect(resources.at("loaded_models").contains("img")&&!resources.at("loaded_models").at("img").at("in_use").get<bool>(),"image resource residency");
    const auto ref=fixture.at("cases")[2].at("kwargs").at("images");
    response=server.handle_request("POST","/v1/images/edits","",json{{"prompt","edit"},{"images",ref},{"strength",0.3},{"size","32x64"},{"response_format","b64_json"}}.dump());
    expect(response.status==200,"images-only edit API: "+response.body);
    const auto captured=json::parse(read(temp/"capture.json"));
    expect(captured.at("reference_pixels").size()==2,"additional images forwarded to engine");
    expect(std::find(captured.at("argv").begin(),captured.at("argv").end(),"--init-img")==captured.at("argv").end(),"additional images do not become init image");
    setenv("EZLOCALAI_URL","http://local/",1);
    response=server.handle_request("POST","/v1/images/generations","",R"({"prompt":"test","size":"32x64"})");
    const auto url=json::parse(response.body).at("data")[0].at("url").get<std::string>();
    expect(url.rfind("http://local/outputs/",0)==0&&fs::is_regular_file(url.substr(13)),"published image URL/file");
    expect(ezlocalai::decode_rgb_image(read(url.substr(13))).width==32,"published PNG");
    const auto rgba=fixture.at("cases").back();expect(rgba.at("mode")=="rgba","source RGBA fixture");write(temp/"case.json",rgba.dump());
    response=server.handle_request("POST","/v1/images/generations","",R"({"prompt":"rgba","size":"32x64"})");
    const auto rgba_path=json::parse(response.body).at("data")[0].at("url").get<std::string>().substr(13);
    const auto rgba_bytes=read(rgba_path);expect(rgba_bytes.size()>25&&rgba_bytes[25]==PNG_COLOR_TYPE_RGBA,"URL output preserves RGBA format");
    png_image decoded{};decoded.version=PNG_IMAGE_VERSION;
    expect(png_image_begin_read_from_memory(&decoded,rgba_bytes.data(),rgba_bytes.size()),"decode published RGBA PNG");
    decoded.format=PNG_FORMAT_RGBA;std::vector<unsigned char> pixels(PNG_IMAGE_SIZE(decoded));
    expect(png_image_finish_read(&decoded,nullptr,pixels.data(),0,nullptr),"read published RGBA pixels");png_image_free(&decoded);
    for(size_t i=3;i<pixels.size();i+=4)expect(pixels[i]==127,"URL output retains generated alpha values");
    setenv("DEFAULT_MODEL","fixture",1);setenv("IMAGE_UNLOAD_LLM_DURING_GENERATION","true",1);ezl_load_config(&config);
    auto tracked=std::make_unique<TrackedBackend>();auto *backend=tracked.get();
    ezlocalai::NativeServer handoff(config,std::move(tracked));
    sample["delay_ms"]=300;write(temp/"case.json",sample.dump());fs::remove(temp/"capture.json");
    auto request=std::async(std::launch::async,[&]{return handoff.handle_request("POST","/v1/images/generations","",R"({"prompt":"test","size":"32x64"})");});
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
    while(!fs::exists(temp/"capture.json")&&std::chrono::steady_clock::now()<deadline)std::this_thread::sleep_for(std::chrono::milliseconds(5));
    resources=json::parse(handoff.handle_request("GET","/v1/resources","","").body);
    expect(resources.at("slots").at("cap_slots").at("text").at("capacity")==0&&resources.at("loaded_models").at("img").at("in_use")==true,"image handoff admission/resource usage");
    expect(resources.at("loaded_models").at("img").at("active_requests")==2,"source image resource acquisition count");
    expect(request.get().status==200&&backend->unloads==1&&backend->restores==1&&backend->loaded,"image restores prior LLM");
    expect(!json::parse(handoff.handle_request("GET","/v1/resources","","").body).at("loaded_models").contains("img"),"handoff destroys image wrapper");
    sample["mode"]="exit";sample.erase("delay_ms");write(temp/"case.json",sample.dump());
    response=handoff.handle_request("POST","/v1/images/edits","",json{{"prompt","edit"},{"images",ref},{"size","32x64"}}.dump());
    expect(response.status==503&&backend->unloads==2&&backend->restores==2&&backend->loaded,"failed edit restores prior LLM");
    for(const char *target:{"IMAGE_SERVER","FALLBACK_SERVER","ROUTER_URL"}) {
        // Each peer has its own ephemeral port, so availability caches and the
        // real router cannot affect this transport test.
        ImagePeer peer;peer.start();
        unsetenv("IMAGE_SERVER");unsetenv("FALLBACK_SERVER");unsetenv("ROUTER_URL");
        setenv(target,peer.url.c_str(),1);setenv("IMAGE_ENABLED","false",1);
        ezl_load_config(&config);ezlocalai::NativeServer forward(config,std::make_unique<ezlocalai::EchoInferenceBackend>("model"));
        response=forward.handle_request("POST","/v1/images/edits","",json{{"prompt","edit"},{"images",ref},{"strength",0.3},{"size","32x64"},{"n",2}}.dump());
        peer.stop();
        expect(response.status==200,std::string(target)+": forwarding status "+response.body);
        expect(peer.posted_path=="/v1/images/edits",std::string(target)+": edit endpoint");
        const auto payload=json::parse(peer.posted_body);
        expect(payload.at("images")==ref&&payload.at("strength")==0.3&&payload.at("n")==2&&!payload.contains("image"),std::string(target)+": complete multi-reference payload");
    }
    unsetenv("IMAGE_SERVER");unsetenv("FALLBACK_SERVER");unsetenv("ROUTER_URL");
    sample.erase("mode");write(temp/"case.json",sample.dump());
    ImagePeer peer;peer.image_bytes=ezlocalai::stt_base64_decode_bytes(sample.at("engine_output"));peer.start();
    ezlocalai::QwenImage url_model("cpu");ezlocalai::QwenImageSettings settings;settings.size="32x64";settings.image=peer.url+"/ref";
    expect(url_model.generate("url edit",settings)==sample.at("engine_output"),"HTTP reference image download");peer.stop();
    sample["delay_ms"]=250;write(temp/"case.json",sample.dump());settings.image.reset();
    ezlocalai::QwenImage second_model("cpu");
    const auto started=std::chrono::steady_clock::now();
    auto first=std::async(std::launch::async,[&]{return url_model.generate("one",settings);});
    auto second=std::async(std::launch::async,[&]{return second_model.generate("two",settings);});
    expect(first.get()==sample.at("engine_output")&&second.get()==sample.at("engine_output"),"concurrent image outputs");
    expect(std::chrono::steady_clock::now()-started>=std::chrono::milliseconds(500),"file lock serializes separate image instances");
    std::cout<<cases.at("cases").size()<<" Python edit schemas and local HTTP/handoff checks passed\n";
}
int main(int argc,char **argv) {
    try {
        if(argc>1 && std::string(argv[1])=="--diffusion-model")return engine(argc,argv);
        expect(argc==3,"usage: test_qwen_image reference_root fixture");
        if(!std::getenv("QWEN_FIXTURE_REEXEC")) {
            setenv("QWEN_FIXTURE_REEXEC","1",1);
            setenv("LD_LIBRARY_PATH",TEST_CUDA_DRIVER_DIRECTORY,1);
            return ezlocalai::run_process({argv[0],argv[1],argv[2]}).status;
        }
        const auto fixture=json::parse(read(argv[2]));
        const auto api_fixture=json::parse(read(fs::path(argv[2]).parent_path()/"image_edit_api_cases.json"));
        const auto policy_fixture=json::parse(read(fs::path(argv[2]).parent_path()/"image_policy_cases.json"));
        expect(sha(read(fs::path(argv[1])/"app.py"))==api_fixture.at("source_sha256"),"Python app source changed; recapture fixtures");
        expect(sha(read(fs::path(argv[1])/"Pipes.py"))==policy_fixture.at("source_sha256"),"Python Pipes source changed; recapture fixtures");
        for(const auto &test:policy_fixture.at("cases")) {
            for(const char *key:{"IMAGE_UNLOAD_LLM_DURING_GENERATION","MEDIA_KEEP_MODELS_LOADED","IMAGE_SERVER","IMAGE_MODEL_MIN_FREE_GB"})unsetenv(key);
            for(const auto &[key,value]:test.at("env").items())setenv(key.c_str(),value.get<std::string>().c_str(),1);
            const auto secondary=test.at("secondary").is_null()?std::nullopt:std::optional<int>(test.at("secondary").get<int>());
            expect(ezlocalai::image_should_unload_llm(test.at("resident"),test.at("loaded"),secondary,test.at("free"))==test.at("unload"),"image GPU handoff policy: "+test.dump());
        }
        for(const char *key:{"IMAGE_UNLOAD_LLM_DURING_GENERATION","MEDIA_KEEP_MODELS_LOADED","IMAGE_SERVER","IMAGE_MODEL_MIN_FREE_GB"})unsetenv(key);
        expect(sha(read(fs::path(argv[1])/"ezlocalai/IMG.py"))==fixture.at("source_sha256"),"Python IMG source changed; recapture fixtures");
        const auto executable=fs::canonical(argv[0]);
        auto temp=(fs::temp_directory_path()/"qwen-test-XXXXXX").string();expect(mkdtemp(temp.data()),"temporary directory");
        const auto previous=fs::current_path();fs::current_path(temp);
        struct Cleanup { fs::path path,previous; ~Cleanup(){fs::current_path(previous);std::error_code ignored;fs::remove_all(path,ignored);} } cleanup{temp,previous};
        const auto models=fs::path(temp)/"models";
        for(const auto &a:ezlocalai::qwen_image_assets())write(models/a.filename,"fixture");
        setenv("SDCPP_BIN",executable.c_str(),1);setenv("SDCPP_MODELS_DIR",models.c_str(),1);
        setenv("QWEN_FIXTURE_CASE",(fs::path(temp)/"case.json").c_str(),1);
        setenv("QWEN_FIXTURE_CAPTURE",(fs::path(temp)/"capture.json").c_str(),1);
        setenv("EZLOCALAI_TEST_CUDA_DEVICES",(fs::path(temp)/"devices").c_str(),1);
        unsetenv("CUDA_VISIBLE_DEVICES");unsetenv("EZLOCALAI_TEST_CUDA_FAIL");
        size_t count=0;
        for(const auto &test:fixture.at("cases")) {
            write(fs::path(temp)/"case.json",test.dump());fs::remove(fs::path(temp)/"capture.json");
            const uint64_t free=test.value("free_gb",0)*1024ULL*1024*1024;
            write(fs::path(temp)/"devices","GPU0\t8 6 25769803776 "+std::to_string(free)+"\nGPU1\t8 6 25769803776 "+std::to_string(free)+"\n");
            ezlocalai::QwenImage image(test.value("device","cpu"));expect(image.available(),"model available");
            if(test.value("mode","")=="missing_mmproj")fs::remove(models/ezlocalai::qwen_image_assets()[3].filename);
            const auto &kw=test.at("kwargs");ezlocalai::QwenImageSettings s;s.size=kw.value("size","32x64");
            s.strength=kw.value("strength",0.75);s.steps=kw.value("num_inference_steps",20);s.guidance=kw.value("guidance_scale",6.0);s.negative_prompt=kw.value("negative_prompt","");
            if(kw.contains("image"))s.image=kw.at("image").get<std::string>();
            if(kw.contains("images"))s.images=kw.at("images").get<std::vector<std::string>>();
            std::string result,error;
            try { result=image.generate(kw.value("prompt","test"),s); }catch(const std::exception &e){error=e.what();}
            const auto name=test.at("name").get<std::string>();
            expect(error==test.value("error",""),name+": error "+error+" expected "+test.value("error",""));
            expect(result==test.value("result",""),name+": base64 result differs");
            if(test.contains("argv")) {
                const auto actual=json::parse(read(fs::path(temp)/"capture.json"));
                expect(actual.at("argv")==test.at("argv"),name+": argv "+actual.at("argv").dump()+" expected "+test.at("argv").dump());
                expect(actual.at("reference_pixels")==test.at("reference_pixels"),name+": reference pixels/order");
                for(const auto &p:actual.at("paths"))expect(!fs::exists(p.get<std::string>()),name+": temporary files survived");
            }else expect(!fs::exists(fs::path(temp)/"capture.json"),name+": engine launched on invalid input");
            write(models/ezlocalai::qwen_image_assets()[3].filename,"fixture");++count;
        }
        setenv("SDCPP_BIN","/nonexistent/qwen-engine",1);ezlocalai::QwenImage missing;
        expect(!missing.available(),"missing engine unavailable");
        try {missing.generate("test",{});throw std::logic_error("missing engine accepted");}
        catch(const std::runtime_error &e){expect(std::string(e.what())=="Qwen-Image-2.1 is unavailable (binary or models missing)","missing engine diagnostic");}
        api_cases(api_fixture,fixture,executable,temp);
        std::cout<<count<<" Python Qwen image cases passed; process failures, references, GPU offload, cleanup verified\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
