import asyncio
import os
import threading
import time
from collections import OrderedDict

import edge_tts

import timing

DEFAULT_VOICE = "ur-PK-UzmaNeural"  # female Urdu (Pakistan). Alternative: ur-PK-AsadNeural (male)

# Speaking speed. Slightly slower than edge-tts's default so patient names come out clearly
# (testers found names hard to catch). "+0%" restores the original speed; "-15%" is slower still.
TTS_RATE = os.environ.get("TTS_RATE", "-10%").strip() or "+0%"

# Finished mp3s for text that's been spoken before (fixed phrases, and template replies that
# repeat, e.g. the same patient name during a demo). edge-tts has no persistent connection to
# reuse — every call opens a fresh websocket to Microsoft — so skipping the call entirely is the
# only way to remove that setup time. Set TTS_CACHE_SIZE=0 to turn the cache off.
TTS_CACHE_SIZE = int(os.environ.get("TTS_CACHE_SIZE", "128"))
_cache: "OrderedDict" = OrderedDict()
_cache_lock = threading.Lock()


async def _synthesize_async(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice, rate=TTS_RATE)
    chunks = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.extend(chunk["data"])
    return bytes(chunks)


def synthesize(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """Text -> mp3 bytes. Safe to call from a background thread (used by session.py)."""
    key = (voice, TTS_RATE, text)
    if TTS_CACHE_SIZE > 0:
        with _cache_lock:
            audio = _cache.get(key)
            if audio is not None:
                _cache.move_to_end(key)
        if audio is not None:
            timing.log("tts", 0.0, chars=len(text), bytes=len(audio), cache="hit")
            return audio
    t0 = time.perf_counter()
    audio = asyncio.run(_synthesize_async(text, voice))
    timing.log("tts", time.perf_counter() - t0, chars=len(text), bytes=len(audio), cache="miss")
    if TTS_CACHE_SIZE > 0 and audio:
        with _cache_lock:
            _cache[key] = audio
            _cache.move_to_end(key)
            while len(_cache) > TTS_CACHE_SIZE:
                _cache.popitem(last=False)
    return audio


# Streaming: the first piece is sent as soon as there's enough for the browser to start
# playing; later pieces are batched so a reply isn't split into hundreds of tiny messages.
STREAM_FIRST_CHUNK_BYTES = 4 * 1024
STREAM_CHUNK_BYTES = 16 * 1024


def synthesize_stream(text: str, on_chunk, voice: str = DEFAULT_VOICE) -> bytes:
    """Like synthesize(), but hands mp3 pieces to on_chunk(bytes) while edge-tts is still
    producing them, so playback can start before the whole reply exists. on_chunk is called
    from this thread and may block (it sends over the WebSocket). Returns the full mp3, which
    is also cached, so a cached phrase is handed over as one piece."""
    key = (voice, TTS_RATE, text)
    if TTS_CACHE_SIZE > 0:
        with _cache_lock:
            audio = _cache.get(key)
            if audio is not None:
                _cache.move_to_end(key)
        if audio is not None:
            timing.log("tts.stream", 0.0, chars=len(text), bytes=len(audio), cache="hit")
            on_chunk(audio)
            return audio

    t0 = time.perf_counter()
    first_chunk_at = None

    async def run():
        nonlocal first_chunk_at
        pending, total = bytearray(), bytearray()
        async for chunk in edge_tts.Communicate(text, voice, rate=TTS_RATE).stream():
            if chunk["type"] != "audio":
                continue
            pending.extend(chunk["data"])
            total.extend(chunk["data"])
            limit = STREAM_CHUNK_BYTES if first_chunk_at else STREAM_FIRST_CHUNK_BYTES
            if len(pending) >= limit:
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter() - t0
                on_chunk(bytes(pending))
                pending.clear()
        if pending:
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter() - t0
            on_chunk(bytes(pending))
        return bytes(total)

    audio = asyncio.run(run())
    timing.log("tts.stream", time.perf_counter() - t0, chars=len(text), bytes=len(audio), cache="miss",
               first_chunk=f"{first_chunk_at:.2f}s" if first_chunk_at is not None else "n/a")
    if TTS_CACHE_SIZE > 0 and audio:
        with _cache_lock:
            _cache[key] = audio
            _cache.move_to_end(key)
            while len(_cache) > TTS_CACHE_SIZE:
                _cache.popitem(last=False)
    return audio


def prewarm(phrases, voice: str = DEFAULT_VOICE):
    """Synthesize fixed phrases into the cache in the background, so the first time one is
    needed it's already there. Failures are ignored (the phrase just gets made on demand)."""
    if TTS_CACHE_SIZE <= 0:
        return

    def work():
        for text in phrases:
            try:
                synthesize(text, voice)
            except Exception as e:  # network hiccup at startup: not worth crashing over
                print(f"[tts] prewarm skipped ({e.__class__.__name__}): {text[:30]}")

    threading.Thread(target=work, daemon=True).start()
