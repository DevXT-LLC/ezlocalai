"""Text-in/audio-out sessions over the worker's existing framed PCM protocol."""

import asyncio
import re
import struct
from contextlib import aclosing


async def segment_events(stream):
    """Validate one HTTP-style segment; omit its end marker when concatenating."""
    buffer = bytearray()
    header = False
    remaining = None
    ended = False
    async for chunk in stream:
        if ended and chunk:
            raise ValueError("Unexpected bytes after TTS segment end")
        buffer.extend(chunk)
        if not header:
            if len(buffer) < 8:
                continue
            value = bytes(buffer[:8])
            del buffer[:8]
            if struct.unpack("<IHH", value) != (24000, 16, 1):
                raise ValueError("Unsupported TTS audio format")
            yield "header", value
            header = True
        while buffer and not ended:
            if remaining is None:
                if len(buffer) < 4:
                    break
                remaining = struct.unpack_from("<I", buffer)[0]
                del buffer[:4]
                if remaining == 0:
                    ended = True
                    if buffer:
                        raise ValueError("Unexpected bytes after TTS segment end")
                    break
                if remaining > 16 * 1024 * 1024 or remaining % 2:
                    raise ValueError("Invalid TTS PCM frame size")
                yield "audio", struct.pack("<I", remaining)
            take = min(remaining, len(buffer))
            if take:
                yield "audio", bytes(buffer[:take])
                del buffer[:take]
                remaining -= take
            if remaining == 0:
                remaining = None
    if not header or not ended or buffer:
        raise ValueError("Truncated TTS segment")


async def stream_tts_session(websocket, pipe, idle_timeout=30):
    """Receive text during synthesis; lease the shared slot only for ready text.

    Four queued segments and a 4096-character input buffer bound pending work.
    Explicit flush submits short segments; otherwise complete sentences are sent
    after 50 characters, or a word boundary after 350 characters.
    """
    pending = asyncio.Queue(maxsize=4)

    async def receive():
        buffer = ""
        voice, language = "default", "en"
        while True:
            data = await asyncio.wait_for(websocket.receive_json(), idle_timeout)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            text = data.get("text", "")
            if not isinstance(text, str) or len(buffer) + len(text) > 4096:
                raise ValueError("Text buffer exceeds 4096 characters")
            new_voice = data.get("voice", voice)
            new_language = data.get("language", language)
            if not isinstance(new_voice, str) or not isinstance(new_language, str):
                raise ValueError("Voice and language must be strings")
            if buffer.strip() and (new_voice, new_language) != (voice, language):
                raise ValueError(
                    "Flush buffered text before changing voice or language"
                )
            voice, language = new_voice, new_language
            buffer += text
            boundary = 0
            if data.get("done") or data.get("flush"):
                boundary = len(buffer)
            elif len(buffer) >= 50:
                ends = list(re.finditer(r"[.!?。！？](?:\s+|$)", buffer))
                if ends:
                    boundary = ends[-1].end()
                elif len(buffer) >= 350:
                    boundary = buffer.rfind(" ", 0, 351)
                    if boundary <= 0:
                        boundary = 350
            if boundary:
                text, buffer = buffer[:boundary].strip(), buffer[boundary:]
                if text:
                    await pending.put((text, voice, language))
            if data.get("done"):
                await pending.put(None)
                return

    async def send(data):
        await asyncio.wait_for(websocket.send_bytes(data), idle_timeout)

    async def synthesize():
        header_sent = False
        while True:
            item = await pending.get()
            if item is None:
                break
            # Do not reserve/unload the LLM while an idle client composes text.
            lease = await pipe._tts_lock.acquire()
            model = None
            try:
                model = pipe._get_tts()
                while item is not None:
                    text, voice, language = item
                    async with aclosing(
                        model.generate_stream(text, voice, language)
                    ) as stream:
                        async for kind, payload in segment_events(stream):
                            if kind == "header":
                                if not header_sent:
                                    await send(payload)
                                    header_sent = True
                            else:
                                await send(payload)
                    try:
                        item = pending.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                else:
                    break
            finally:
                try:
                    if model is not None and not lease.get("handoff"):
                        pipe._destroy_tts(async_cleanup=False)
                finally:
                    await pipe._tts_lock.release(lease)
        if not header_sent:
            await send(struct.pack("<IHH", 24000, 16, 1))
        await send(struct.pack("<I", 0))
        await send(b"")  # Legacy WebSocket end-of-session notification.

    tasks = [asyncio.create_task(receive()), asyncio.create_task(synthesize())]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
