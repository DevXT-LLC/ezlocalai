#include "ezlocalai/environment_defaults.hpp"
#include "ezlocalai/native_server.hpp"
#include "ezlocalai/stt_runtime.hpp"
#include "fixture_ports.hpp"
#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <sstream>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

namespace {
using Json = nlohmann::json;
void check(bool value, const std::string &message) { if (!value) throw std::runtime_error(message); }
struct Socket {
    int fd;
    ~Socket() { if (fd >= 0) close(fd); }
};
void send_all(int fd, const std::string &bytes) {
    size_t at = 0;
    while (at < bytes.size()) {
        const auto count = send(fd,bytes.data() + at,std::min<size_t>(32768,bytes.size() - at),MSG_NOSIGNAL);
        if (count < 0 && errno == EINTR) continue;
        check(count > 0,"TCP send");
        at += static_cast<size_t>(count);
    }
}
struct Response { int status; std::string body,headers; };
Response socket_request(int port, const std::string &wire_request, bool finish_request = true) {
    Socket socket{::socket(AF_INET,SOCK_STREAM,0)};
    check(socket.fd >= 0,"TCP socket");
    timeval timeout{10,0};
    setsockopt(socket.fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
    setsockopt(socket.fd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK); address.sin_port = htons(port);
    check(connect(socket.fd,reinterpret_cast<sockaddr *>(&address),sizeof(address)) == 0,"TCP connect");
    send_all(socket.fd,wire_request);
    if (finish_request) shutdown(socket.fd,SHUT_WR);
    std::string wire;
    char buffer[4096];
    for (;;) {
        const auto count = recv(socket.fd,buffer,sizeof(buffer),0);
        if (count < 0 && errno == EINTR) continue;
        if (count < 0 && errno == ECONNRESET && !wire.empty()) break;
        check(count >= 0,"TCP receive");
        if (!count) break;
        wire.append(buffer,static_cast<size_t>(count));
    }
    const auto end = wire.find("\r\n\r\n"),space = wire.find(' ');
    check(end != std::string::npos && space != std::string::npos,"HTTP response framing: " + wire);
    return {std::stoi(wire.substr(space + 1,3)),wire.substr(end + 4),wire.substr(0,end)};
}
Response request(int port, const std::string &framing, const std::string &body = "",
    const std::string &path = "/v1/audio/transcriptions") {
    return socket_request(port,"POST " + path + " HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nAuthorization: Bearer upload-key\r\nContent-Type: multipart/form-data; boundary=caption-upload\r\n" + framing + "\r\n" + body);
}
std::string frame(unsigned char opcode,const std::string &payload) {
    check(payload.size() < 126,"short fixture frame");
    std::string bytes{char(128 | opcode),char(128 | payload.size())};
    const std::string mask = "abcd"; bytes += mask;
    for (size_t i = 0; i < payload.size(); ++i) bytes += char(payload[i] ^ mask[i % mask.size()]);
    return bytes;
}
std::string chunk(const std::string &body) {
    std::string encoded;
    for (size_t at = 0; at < body.size(); at += 32749) {
        const auto size = std::min<size_t>(32749,body.size() - at);
        std::ostringstream header; header << std::hex << size << ";fixture=yes\r\n";
        encoded += header.str(); encoded.append(body,at,size); encoded += "\r\n";
    }
    return encoded + "0\r\nX-Fixture: complete\r\n\r\n";
}
std::string wav(size_t size) {
    check(size >= 44 && size % 2 == 0,"PCM fixture size");
    std::string bytes(size,'\0');
    bytes.replace(0,4,"RIFF"); bytes.replace(8,8,"WAVEfmt "); bytes.replace(36,4,"data");
    const auto little = [&](size_t at,uint32_t value,size_t width) { for (size_t i = 0; i < width; ++i) bytes[at + i] = char(value >> (8 * i)); };
    little(4,size - 8,4); little(16,16,4); little(20,1,2); little(22,1,2);
    little(24,16000,4); little(28,32000,4); little(32,2,2); little(34,16,2); little(40,size - 44,4);
    return bytes;
}
std::string multipart(const std::string &audio, const std::string &format) {
    std::string body = "--caption-upload\r\nContent-Disposition: form-data; name=\"file\"; filename=\"caption.wav\"\r\nContent-Type: audio/wav\r\n\r\n" + audio + "\r\n";
    for (const auto &field : std::vector<std::pair<std::string,std::string>>{{"model","XT"},{"response_format",format},{"timestamps","true"}})
        body += "--caption-upload\r\nContent-Disposition: form-data; name=\"" + field.first + "\"\r\n\r\n" + field.second + "\r\n";
    return body + "--caption-upload--\r\n";
}
}

int main() { try {
    for (const auto &entry : ezlocalai::environment_defaults) setenv(std::string(entry.first).c_str(),std::string(entry.second).c_str(),1);
    for (const auto &entry : std::vector<std::pair<std::string,std::string>>{
        {"EZLOCALAI_API_KEY","upload-key"},{"ROUTER_MODE","false"},{"ROUTER_URL",""},{"FALLBACK_SERVER",""},
        {"DEFAULT_MODEL","none"},{"WHISPER_MODEL","fixture-whisper"},{"STT_ENABLED","true"},{"VOICE_SERVER","true"},
        {"LAZY_LOAD_VOICE","true"},{"STT_UNLOAD_LLM_DURING_GENERATION","false"},{"TTS_ENABLED","false"},
        {"EMBEDDING_ENABLED","false"},{"IMAGE_ENABLED","false"},{"VIDEO_ENABLED","false"},{"MUSIC_ENABLED","false"}})
        setenv(entry.first.c_str(),entry.second.c_str(),1);
    unsetenv("EZLOCALAI_MAX_REQUEST_BODY_BYTES");
    unsetenv("EZLOCALAI_MAX_IN_FLIGHT_REQUEST_BODY_BYTES");
    const auto audio = wav(17 * 1024 * 1024);
    std::atomic<size_t> calls{0};
    auto probes = ezlocalai::native_resource_probes();
    probes.gpu_count = [] { return 0; }; probes.total_per_gpu = [] { return std::vector<double>{}; };
    probes.free_per_gpu = [] { return std::vector<double>{}; }; probes.system_ram = [] { return 16.; };
    auto resources = std::make_shared<ezlocalai::ResourceManager>(probes);
    ezl_env_config config{}; ezl_load_config(&config);
    ezlocalai::NativeServer server(config,std::make_unique<ezlocalai::EchoInferenceBackend>("none"),resources,[&](auto manager) {
        auto operations = ezlocalai::stt_pool_resources({},manager);
        operations.load = [&](const auto &name) {
            check(name == "fixture-whisper","worker retains configured STT model");
            auto model = std::make_shared<ezlocalai::SttPoolModel>();
            model->device = [] { return "cpu"; }; model->close = [] {};
            model->transcribe = [&](const auto &encoded) {
                const auto input = Json::parse(encoded);
                const auto base64 = input.at("base64_audio").template get<std::string>();
                std::string decoded(base64.size(),'\0');
                auto size = EVP_DecodeBlock(reinterpret_cast<unsigned char *>(decoded.data()),reinterpret_cast<const unsigned char *>(base64.data()),base64.size());
                check(size >= 0,"decode uploaded audio");
                if (base64.back() == '=') --size; if (base64[base64.size() - 2] == '=') --size;
                decoded.resize(size);
                check(decoded == audio,"complete multipart WAV is byte-identical at STT inference");
                check(input.at("return_segments") == true && input.at("audio_format") == "audio/wav","caption STT options");
                ++calls;
                return std::string(R"({"text":"caption","language":"en","segments":[{"id":0,"start":0.1,"end":1.25,"text":"caption"}]})");
            };
            return model;
        };
        return operations;
    });
    const auto port = fixture_ports(); std::string error;
    check(server.start("127.0.0.1",port,&error),"server start: " + error);
    size_t checks = 0;
    for (const auto &format : {"srt","vtt","verbose_json"}) {
        const auto body = multipart(audio,format);
        for (const bool chunked : {false,true}) {
            const auto response = request(port,chunked ? "Transfer-Encoding: chunked\r\n" : "Content-Length: " + std::to_string(body.size()) + "\r\n",chunked ? chunk(body) : body);
            check(response.status == 200,"large upload HTTP status: " + std::to_string(response.status) + " " + response.body);
            if (std::string(format) == "verbose_json") {
                const auto value = Json::parse(response.body);
                check(value.at("segments")[0].at("start") == 0.1 && value.at("segments")[0].at("end") == 1.25,"timed JSON segments");
            } else {
                const auto expected = std::string(format) == "srt" ? "1\n00:00:00,100 --> 00:00:01,250\ncaption\n" : "WEBVTT\n\n00:00:00.100 --> 00:00:01.250\ncaption\n";
                check(response.body == expected,"multiline subtitle response");
            }
            ++checks;
        }
    }
    check(calls == 6,"exactly one STT inference per valid upload");
    const auto rejected = [&](const std::string &framing,const std::string &body,int status,const std::string &detail) {
        const auto response = request(port,framing,body);
        check(response.status == status,"framing error status: " + std::to_string(response.status) + " " + response.body);
        check(Json::parse(response.body).at("detail").template get<std::string>().find(detail) != std::string::npos,"explicit transport error: " + response.body);
        check(calls == 6,"invalid framing did not reach inference"); ++checks;
    };
    for (const auto &length : {"-1","+1","1x","18446744073709551616",""})
        rejected("Content-Length: " + std::string(length) + "\r\n","",400,"Content-Length");
    rejected("Content-Length: 10\r\n","short",400,"Incomplete HTTP request body");
    rejected("Content-Length: 0\r\nContent-Length: 0\r\n","",400,"Duplicate");
    for (const auto &name : {"Content-Length "," Content-Length","\tContent-Length","Content[Length"})
        rejected("Content-Length: 0\r\n" + std::string(name) + ": 2\r\n","",400,"header name");
    rejected("Content-Length: 0\r\nTransfer-Encoding : chunked\r\n","",400,"header name");
    rejected("Transfer-Encoding: chunked\r\nTransfer-Encoding\t: chunked\r\n","",400,"header name");
    rejected("Content-Length: 0\r\nTransfer-Encoding: chunked\r\n","",400,"Transfer-Encoding");
    rejected("Transfer-Encoding: chunked\r\nTransfer-Encoding: chunked\r\n","",400,"Duplicate");
    for (const auto &coding : {"gzip","gzip, chunked","chunked, gzip","xchunked"})
        rejected("Transfer-Encoding: " + std::string(coding) + "\r\n","",400,"Transfer-Encoding");
    for (const auto &encoded : {"g\r\n","1\r\nxXX","1\nx\r\n","-1\r\n","10000000000000000\r\n"})
        rejected("Transfer-Encoding: chunked\r\n",encoded,400,"Malformed chunked");
    for (const auto &encoded : {"1\r\nx\r\n","0\r\nX-Incomplete: yes\r\n","3\r\nx"})
        rejected("Transfer-Encoding: chunked\r\n",encoded,400,"Incomplete chunked");
    rejected("Transfer-Encoding: chunked\r\n",std::string(65537,'a'),400,"Malformed chunked");
    rejected("Content-Length: 536870912\r\n","",400,"Incomplete HTTP request body");
    rejected("Content-Length: 536870913\r\n","",413,"EZLOCALAI_MAX_REQUEST_BODY_BYTES");
    rejected("Transfer-Encoding: chunked\r\n","20000000\r\n",400,"Incomplete chunked");
    rejected("Transfer-Encoding: chunked\r\n","20000001\r\n",413,"EZLOCALAI_MAX_REQUEST_BODY_BYTES");
    const auto body = multipart(audio,"srt");
    setenv("EZLOCALAI_MAX_REQUEST_BODY_BYTES",std::to_string(body.size()).c_str(),1);
    check(request(port,"Content-Length: " + std::to_string(body.size()) + "\r\n",body).status == 200,"exact fixed-length cap accepted");
    check(request(port,"Transfer-Encoding: chunked\r\n",chunk(body)).status == 200,"exact decoded chunked cap accepted");
    check(calls == 8,"exact-cap requests reached inference");
    setenv("EZLOCALAI_MAX_REQUEST_BODY_BYTES",std::to_string(body.size() - 1).c_str(),1);
    const auto before_limit = calls.load();
    check(request(port,"Content-Length: " + std::to_string(body.size()) + "\r\n").status == 413,"fixed-length cap plus one rejected before receiving payload");
    std::ostringstream over; over << std::hex << body.size() << "\r\n";
    check(request(port,"Transfer-Encoding: chunked\r\n",over.str()).status == 413,"declared chunk cap plus one rejected before receiving payload");
    check(calls == before_limit,"oversize requests did not reach inference");
    setenv("EZLOCALAI_MAX_REQUEST_BODY_BYTES","3",1);
    check(request(port,"Transfer-Encoding: chunked\r\n","2\r\nxx\r\n2\r\n").status == 413,"cumulative chunk cap enforced");
    unsetenv("EZLOCALAI_MAX_REQUEST_BODY_BYTES");
    const auto recovery = request(port,"Content-Length: " + std::to_string(body.size()) + "\r\n",body);
    check(recovery.status == 200 && calls == 9,"server recovers without leaked STT leases after framing errors");
    const auto headers_only = [&](const std::string &method,const std::string &path,const std::string &authorization,int status) {
        const auto response = socket_request(port,method + " " + path + " HTTP/1.1\r\nHost: localhost\r\nContent-Length: 536870913\r\n" + authorization + "\r\n",false);
        check(response.status == status,"reject protected upload before receiving body: " + std::to_string(response.status));
    };
    headers_only("POST","/v1/audio/transcriptions","",401);
    headers_only("POST","/v1/audio/transcriptions","Authorization: Bearer wrong\r\n",401);
    headers_only("GET","/v1/audio/transcriptions","",405);
    headers_only("POST","/missing-upload","",404);
    for (const bool cookie : {false,true}) {
        for (const auto &authorization : {"","Authorization: Bearer wrong\r\n"}) {
            const auto response = socket_request(port,"POST /v1/audio/transcriptions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 536870913\r\nOrigin: https://school.example\r\n"
                + std::string(cookie ? "Cookie: sid=1\r\n" : "") + authorization + "\r\n",false);
            check(response.status == 401 && Json::parse(response.body).at("detail") == "Invalid API Key","early auth rejection body");
            check(response.headers.find(cookie ? "Access-Control-Allow-Origin: https://school.example" : "Access-Control-Allow-Origin: *") != std::string::npos,"early auth rejection origin: " + response.headers);
            check(response.headers.find("Access-Control-Allow-Credentials: true") != std::string::npos,"early auth rejection credentials");
        }
    }
    const auto head_error = socket_request(port,"HEAD /v1/audio/transcriptions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 536870913\r\nOrigin: https://school.example\r\nCookie: sid=1\r\n\r\n",false);
    check(head_error.status == 405 && head_error.body.empty() && head_error.headers.find("Access-Control-Allow-Origin: https://school.example") != std::string::npos,"early HEAD method rejection retains CORS and suppresses body");
    const auto unauthorized_chunks = socket_request(port,"POST /v1/audio/transcriptions HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n",false);
    check(unauthorized_chunks.status == 401,"unauthorized chunked upload rejected before receiving first chunk");
    const auto public_read = socket_request(port,"GET /health HTTP/1.1\r\nHost: localhost\r\nContent-Length: 536870912\r\n\r\n",false);
    check(public_read.status == 200 && Json::parse(public_read.body).at("status") == "healthy","public health responds without receiving or reserving a declared body");
    check(request(port,"Content-Length: " + std::to_string(body.size()) + "\r\n",body).status == 200 && calls == 10,"public read cannot starve authenticated upload admission");
    setenv("EZLOCALAI_MAX_IN_FLIGHT_REQUEST_BODY_BYTES","2",1);
    Socket held{::socket(AF_INET,SOCK_STREAM,0)};
    check(held.fd >= 0,"held upload socket");
    timeval held_timeout{10,0}; setsockopt(held.fd,SOL_SOCKET,SO_RCVTIMEO,&held_timeout,sizeof(held_timeout));
    sockaddr_in held_address{}; held_address.sin_family = AF_INET; held_address.sin_addr.s_addr = htonl(INADDR_LOOPBACK); held_address.sin_port = htons(port);
    check(connect(held.fd,reinterpret_cast<sockaddr *>(&held_address),sizeof(held_address)) == 0,"held upload connect");
    send_all(held.fd,"POST /v1/audio/transcriptions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer upload-key\r\nContent-Length: 2\r\nExpect: 100-continue\r\n\r\n");
    std::string acknowledgement; char received[4096];
    while (acknowledgement.find("\r\n\r\n") == std::string::npos) {
        const auto size = recv(held.fd,received,sizeof(received),0); check(size > 0,"upload admission acknowledgement"); acknowledgement.append(received,size);
    }
    check(acknowledgement == "HTTP/1.1 100 Continue\r\n\r\n","body reserved before Continue");
    const auto busy = request(port,"Content-Length: 1\r\n");
    check(busy.status == 503 && busy.headers.find("Retry-After: 1") != std::string::npos,"aggregate fixed-length budget rejects concurrent upload with retry hint");
    check(request(port,"Transfer-Encoding: chunked\r\n","1\r\nx\r\n0\r\n\r\n").status == 503,"aggregate chunked budget enforced");
    headers_only("POST","/v1/audio/transcriptions","",401);
    send_all(held.fd,"xx"); shutdown(held.fd,SHUT_WR);
    std::string held_response;
    for (;;) { const auto size = recv(held.fd,received,sizeof(received),0); check(size >= 0,"held response"); if (!size) break; held_response.append(received,size); }
    check(held_response.find("422 Unprocessable Entity") != std::string::npos,"complete admitted request reaches multipart validation");
    unsetenv("EZLOCALAI_MAX_IN_FLIGHT_REQUEST_BODY_BYTES");
    check(request(port,"Content-Length: " + std::to_string(body.size()) + "\r\n",body).status == 200 && calls == 11,"aggregate reservation released after request completion");
    setenv("TTS_ENABLED","true",1);
    setenv("EZLOCALAI_MAX_REQUEST_BODY_BYTES","1",1);
    const auto early_frames = frame(9,"early") + frame(8,std::string("\x03\xe8",2));
    const auto websocket = socket_request(port,"GET /v1/audio/speech/ws HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer upload-key\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n" + early_frames);
    check(websocket.status == 101,"WebSocket upgrade with coalesced early frames");
    check(websocket.body == std::string("\x8a\x05",2) + "early" + std::string("\x88\x02\x03\xe8",4),"early ping and close frames preserved outside HTTP body limit");
    unsetenv("EZLOCALAI_MAX_REQUEST_BODY_BYTES");
    server.stop();
    std::cout << checks << " real TCP framing/large STT checks plus exact/over-limit boundaries and recovery passed\n";
    return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; } }
