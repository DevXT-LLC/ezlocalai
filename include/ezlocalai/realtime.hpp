#pragma once
#include "ezlocalai/tunnel_connection.hpp"
#include <functional>
#include <string>

namespace ezlocalai {
class VoiceConversion;
// Relay already-upgraded RFC6455 connections with bounded backpressure in
// both directions. Masks, fragmented messages, close frames and audio bytes
// remain intact; the model server performs protocol/event validation.
void relay_realtime_websocket(int client, TunnelWebSocketConnection &engine,
    std::string from_client, std::string from_engine,
    const std::function<bool()> &stopped, double idle_seconds = 600);
void relay_realtime_voice(int client, TunnelWebSocketConnection &engine,
    std::string from_client, std::string from_engine, VoiceConversion &converter,
    const std::string &voice, const std::function<bool()> &stopped,
    double idle_seconds = 600);
}
