#include "whisper_transcription_test.hpp"
#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/process.hpp"
#include "fixture_ports.hpp"
#include <sys/wait.h>
#include <openssl/evp.h>
using namespace transcription_test;
int main(int argc,char **argv) { try {
    expect(argc == 5,"server binary, fixtures, English model and modern VAD directory"); const auto binary = fs::absolute(argv[1]),fixtures = fs::absolute(argv[2]),model = fs::absolute(argv[3]),vad = fs::absolute(argv[4]);
    Temporary temporary; struct Restore { fs::path path = fs::current_path(); ~Restore() { fs::current_path(path); } } restore; fs::current_path(temporary.path);
    for (const auto &entry : environment_defaults) setenv(std::string(entry.first).c_str(),std::string(entry.second).c_str(),1);
    for (const auto &pair : std::vector<std::pair<std::string,std::string>>{{"EZLOCALAI_NATIVE_BACKEND","echo"},{"EZLOCALAI_API_KEY",""},{"DEFAULT_MODEL","none"},{"CUDA_VISIBLE_DEVICES",""},{"WHISPER_MODEL",model.string()},{"WHISPER_VAD_DIRECTORY",vad.string()},{"LAZY_LOAD_VOICE","true"},{"STT_UNLOAD_LLM_DURING_GENERATION","false"},{"WHISPER_CPP_BIN","/nonexistent-whisper-cli"},{"WHISPER_CPP_CLI","/nonexistent-whisper-cli"},{"OPENBLAS_NUM_THREADS","1"}}) setenv(pair.first.c_str(),pair.second.c_str(),1);
    const auto port = fixture_ports(); const auto port_text = std::to_string(port),url = "http://127.0.0.1:" + port_text; const auto log = temporary.path/"server.log";
    const pid_t pid = fork(); expect(pid >= 0,"fork native server");
    if (!pid) { const int fd = open(log.c_str(),O_CREAT | O_WRONLY | O_TRUNC,0600); dup2(fd,STDOUT_FILENO); dup2(fd,STDERR_FILENO); close(fd); execl(binary.c_str(),binary.c_str(),"--host","127.0.0.1","--port",port_text.c_str(),nullptr); _exit(127); }
    struct Child { pid_t pid; ~Child() { if (pid > 0) { kill(pid,SIGKILL); while (waitpid(pid,nullptr,0) < 0 && errno == EINTR) {} } } } child{pid};
    bool ready = false; const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
        const auto response = run_process({"curl","-sS","--max-time","1",url + "/health"},true,true);
        if (!response.status && response.output.find("healthy") != std::string::npos) { ready = true; break; }
        int status; if (waitpid(pid,&status,WNOHANG) == pid) { child.pid = -1; throw std::runtime_error("native server exited: " + read(log)); }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    expect(ready,"native server readiness: " + read(log));
    const auto body_path = temporary.path/"response.json";
    const auto actual = run_process({"curl","-sS","--max-time","120","-o",body_path.string(),"-w","%{http_code}","-F","file=@" + (fixtures/"stt_runtime_assets/speech.wav").string() + ";type=audio/wav","-F","language=en",url + "/v1/audio/transcriptions"},true);
    expect(!actual.status && actual.output == "200","native process transcription: " + actual.output + " " + read(body_path));
    const auto expected = records(fixtures/"stt_api_model_modern.jsonl")[1].at("outcomes")[0].at("response").get<std::string>(); compare_json(Json::parse(read(body_path)),Json::parse(expected),"actual TCP model transcription");
    const auto invalid = run_process({"curl","-sS","--max-time","120","-o",body_path.string(),"-w","%{http_code}","-F","file=@" + (fixtures/"stt_runtime_assets/invalid.bin").string() + ";type=audio/wav",url + "/v1/audio/transcriptions"},true);
    expect(!invalid.status && invalid.output == "400","native process invalid audio"); compare_json(Json::parse(read(body_path)),Json{{"detail","The uploaded file does not contain valid audio data"}},"actual TCP error response");
    const auto subtitles = run_process({"curl","-sS","--max-time","120","-o",body_path.string(),"-w","%{http_code}","-F","file=@" + (fixtures/"stt_runtime_assets/speech.wav").string() + ";type=audio/wav","-F","source_language=en","-F","format=json","-F","beam_size=3",url + "/v1/audio/subtitles"},true);
    expect(!subtitles.status && subtitles.output == "200","native process subtitles: " + subtitles.output + " " + read(body_path));
    const auto subtitle_expected = records(fixtures/"subtitles_api_model_modern.jsonl")[1].at("outcomes")[0].at("response").get<std::string>(); compare_json(Json::parse(read(body_path)),Json::parse(subtitle_expected),"actual TCP model subtitles");
    const auto speech = read(fixtures/"stt_runtime_assets/speech.wav"); std::string encoded(4*((speech.size()+2)/3)+1,'\0');
    const auto size = EVP_EncodeBlock(reinterpret_cast<unsigned char *>(encoded.data()),reinterpret_cast<const unsigned char *>(speech.data()),speech.size()); expect(size >= 0,"encode TCP LLM audio"); encoded.resize(size);
    const auto request_path = temporary.path/"audio-request.json";
    std::ofstream(request_path) << Json{{"messages",Json::array({Json{{"role","user"},{"content",encoded}}})},{"audio_format","wav"}}.dump();
    const auto llm_audio = run_process({"curl","-sS","--max-time","120","-o",body_path.string(),"-w","%{http_code}","-H","Content-Type: application/json","--data-binary","@"+request_path.string(),url+"/v1/chat/completions"},true);
    expect(!llm_audio.status && llm_audio.output == "200","native process LLM audio: " + llm_audio.output + " " + read(body_path));
    const auto transcript = records(fixtures/"stt_pipeline_model_modern.jsonl")[1].at("outcomes")[0].at("value").get<std::string>();
    compare_json(Json::parse(read(body_path)).at("choices")[0].at("message").at("content"),"Native ezlocalai echo backend received: "+transcript,"actual TCP native audio reaches LLM");
    kill(pid,SIGTERM); const auto stopped = std::chrono::steady_clock::now() + std::chrono::seconds(20); int status = 0; bool exited = false;
    while (std::chrono::steady_clock::now() < stopped) { if (waitpid(pid,&status,WNOHANG) == pid) { exited = true; child.pid = -1; break; } std::this_thread::sleep_for(std::chrono::milliseconds(20)); }
    expect(exited && WIFEXITED(status) && WEXITSTATUS(status) == 0,"native server clean shutdown"); expect(fs::is_empty(temporary.path/"outputs"),"native server audio cleanup");
    std::cout << "compiled server serves real-model transcription, subtitles, LLM audio and source HTTP 400 over TCP without whisper.cpp\n"; return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
