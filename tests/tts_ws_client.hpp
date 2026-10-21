#pragma once
#include "whisper_transcription_test.hpp"
#include <arpa/inet.h>
#include <sys/socket.h>
namespace tts_ws_test {
using namespace transcription_test;
struct Client {
    int fd=-1;std::string pending;
    explicit Client(int port){
        fd=socket(AF_INET,SOCK_STREAM|SOCK_CLOEXEC,0);expect(fd>=0,"WebSocket socket");timeval timeout{60,0};setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
        sockaddr_in address{};address.sin_family=AF_INET;address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);address.sin_port=htons(port);expect(!connect(fd,reinterpret_cast<sockaddr *>(&address),sizeof(address)),"WebSocket connect");
        send("GET /v1/audio/speech/ws HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n");
        while(pending.find("\r\n\r\n")==std::string::npos)receive();expect(pending.rfind("HTTP/1.1 101",0)==0,"WebSocket upgrade: "+pending);pending.erase(0,pending.find("\r\n\r\n")+4);
    }
    ~Client(){disconnect();}
    void disconnect(){if(fd>=0){shutdown(fd,SHUT_RDWR);close(fd);fd=-1;}}
    void send(const std::string &bytes){for(size_t at=0;at<bytes.size();){const auto n=::send(fd,bytes.data()+at,bytes.size()-at,MSG_NOSIGNAL);expect(n>0,"WebSocket send");at+=n;}}
    void message(const std::string &text){expect(text.size()<126,"short fixture message");std::string wire;wire+=char(129);wire+=char(128|text.size());wire+="abcd";for(size_t i=0;i<text.size();++i)wire+=char(text[i]^"abcd"[i%4]);send(wire);}
    void receive(){char bytes[65536];ssize_t n;do{n=recv(fd,bytes,sizeof(bytes),0);}while(n<0&&errno==EINTR);expect(n>0,"WebSocket receive");pending.append(bytes,n);}
    std::pair<unsigned,std::string> frame(){
        while(pending.size()<2)receive();const unsigned opcode=static_cast<unsigned char>(pending[0])&15;expect(!(static_cast<unsigned char>(pending[1])&128),"server frame unmasked");uint64_t size=static_cast<unsigned char>(pending[1])&127;size_t at=2;
        if(size>=126){const size_t width=size==126?2:8;while(pending.size()<at+width)receive();size=0;for(size_t i=0;i<width;++i)size=(size<<8)|static_cast<unsigned char>(pending[at++]);}
        expect(size<32*1024*1024,"bounded fixture frame");while(pending.size()<at+size)receive();auto value=pending.substr(at,size);pending.erase(0,at+size);return {opcode,std::move(value)};
    }
};
}
