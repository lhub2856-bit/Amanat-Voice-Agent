import asyncio
import os
import queue
import re
import secrets
import threading
import time
import traceback
from collections import deque

from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TerminationEvent,
    TurnEvent,
)
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

import agents
import db
import timing
import web_tts
from session import ClientGone, Session

load_dotenv()

SAMPLE_RATE = 16000


def _env_int(name):
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


# AssemblyAI end-of-turn settings. Left unset, AssemblyAI's own defaults apply (unchanged
# behaviour). Only set these after the [timing] "stt.end_of_turn" lines show a long silence
# wait. Lower = faster replies, but a pause mid-sentence is more likely to cut her off.
STT_MIN_TURN_SILENCE_MS = _env_int("STT_MIN_TURN_SILENCE_MS")
STT_MAX_TURN_SILENCE_MS = _env_int("STT_MAX_TURN_SILENCE_MS")

# Name recognition boost (opt-in): send recently seen patient names to AssemblyAI as
# "keyterms", so the recogniser favours those spellings instead of garbling a name until it's
# repeated. AssemblyAI documents keyterms for its streaming models; check once that whisper-rt
# accepts it (look for "[session] STT error" in the logs) before relying on it in a demo.
STT_KEYTERMS = os.environ.get("STT_KEYTERMS", "0").strip().lower() in ("1", "true", "yes", "on")
MAX_KEYTERMS = 100  # AssemblyAI's per-session limit


def _patient_keyterms():
    try:
        names = [p["name"].strip() for p in db.recent_patients(days=90, limit=MAX_KEYTERMS)]
    except Exception:
        return []
    seen, terms = set(), []
    for name in names:
        if name and len(name) <= 50 and name.lower() not in seen:
            seen.add(name.lower())
            terms.append(name)
    return terms[:MAX_KEYTERMS]


# Server-side switch for streamed replies (the browser also has to ask for them). 0 = off.
STREAM_AUDIO = os.environ.get("STREAM_AUDIO", "1").strip().lower() not in ("0", "false", "no", "off")

# Fixed phrases (confirmations/questions with no name in them) are synthesized once at startup,
# so the first time one is needed it plays without a TTS round trip.
if agents.REPLY_TEMPLATES and os.environ.get("TTS_PREWARM", "1").strip() not in ("0", "false", "no", "off"):
    web_tts.prewarm(agents.FIXED_PHRASES)

# Same list as stt_stream.py — Whisper-family models can hallucinate these from
# silence/background noise regardless of the language actually being spoken.
HALLUCINATION_PATTERNS = (
    "thanks for watching", "thank you for watching", "please subscribe", "subtitles by",
    "subtitles created by", "amara.org", "dimatorzok", "продолжение следует",
    "субтитры сделал", "obrigado", "ça va", "gracias por ver", "merci d'avoir regard",
    "i'm proud to be relevant", "here we go", "so...", "or something", "my point is",
    # Seen in the Railway logs (27 Sep): more "thanks for watching" variants from silence.
    "ご視聴", "시청해", "terima kasih kerana menonton", "mulțumim pentru vizionare",
    "see you next time", "be right back", "次回予告",
)

# Turns that consist ONLY of one of these (ignoring case and punctuation) are dropped. In the
# Railway logs they arrived every few seconds during silence; each one cost a full Gemini call,
# queued ahead of her real sentences, and counted toward the hourly quota guard, which then
# shut the pipeline off for real visits. Deliberately excludes yes/no/okay-style words: those can
# be genuine answers to "same patient or a new one?".
FILLER_TURNS = {
    "thank you", "thank you very much", "thanks", "you", "gracias", "obrigado", "obrigada",
    "tchau", "adiós", "adios", "ciao", "ah ciao", "bye", "nice", "namely",
}
# Scripts no one speaks on this app (Cyrillic, Japanese kana, CJK, Hangul). Urdu, Punjabi,
# English, Roman Urdu and Hindi (Devanagari) all pass.
_FOREIGN_SCRIPT = re.compile(r"[Ѐ-ӿ぀-ヿ一-鿿가-힯]")


def _looks_hallucinated(text: str) -> bool:
    low = text.lower()
    if any(p in low for p in HALLUCINATION_PATTERNS):
        return True
    if _FOREIGN_SCRIPT.search(text):
        return True
    bare = re.sub(r"[\s.,!?¡¿…'\"-]+", " ", low).strip()
    return bare in FILLER_TURNS


app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---------------------------------------------------------------- abuse guard
# The WebSocket URL is publicly visible in the frontend's source (there's no way around
# that for a public demo page). This doesn't add real access control — it's a circuit
# breaker so a stranger hammering the endpoint can't fully drain the free-tier Gemini
# quota right before an actual demo. Raise the number via an env var if it's ever too low.
MAX_PIPELINE_CALLS_PER_HOUR = int(os.environ.get("MAX_PIPELINE_CALLS_PER_HOUR", "150"))
_pipeline_call_times: "deque" = deque()


def _pipeline_quota_ok() -> bool:
    now = time.time()
    while _pipeline_call_times and now - _pipeline_call_times[0] > 3600:
        _pipeline_call_times.popleft()
    return len(_pipeline_call_times) < MAX_PIPELINE_CALLS_PER_HOUR


def _record_pipeline_call():
    _pipeline_call_times.append(time.time())

# ---------------------------------------------------------------- admin auth
security = HTTPBasic()


def require_admin(credentials: HTTPBasicCredentials = Depends(security)):
    user_ok = secrets.compare_digest(credentials.username, os.environ.get("ADMIN_USER", "admin"))
    pass_ok = secrets.compare_digest(credentials.password, os.environ.get("ADMIN_PASS", "changeme"))
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True


@app.get("/api/visits")
def api_visits(_auth: bool = Depends(require_admin)):
    return {"stats": db.get_stats(), "visits": db.get_visits()}


@app.get("/api/analytics")
def api_analytics(_auth: bool = Depends(require_admin)):
    return db.get_analytics()


# A browser's own visit history for the public demo page. The device id is a random value the
# page generates and keeps in its storage, so a browser only ever gets back what it recorded
# itself. Everyone's visits stay behind the admin password.
_DEVICE_ID = re.compile(r"^[A-Za-z0-9-]{16,64}$")


def _valid_device(value):
    return value if isinstance(value, str) and _DEVICE_ID.match(value) else None


@app.get("/api/my-visits")
def api_my_visits(device: str = ""):
    device_id = _valid_device(device)
    if device_id is None:
        raise HTTPException(status_code=400, detail="missing or invalid device id")
    return {"visits": db.get_device_visits(device_id)}


@app.get("/admin", response_class=HTMLResponse)
def admin_page(_auth: bool = Depends(require_admin)):
    # Deliberately NOT in static/ — anything there is served unauthenticated at its own path.
    return open("private/admin.html", encoding="utf-8").read()


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    # A page that can play mp3 in pieces connects with ?stream=1 and gets each reply streamed
    # while it's being made. Any other page (or STREAM_AUDIO=0 here) gets complete mp3s as before.
    stream_audio = STREAM_AUDIO and websocket.query_params.get("stream") == "1"
    session = Session(loop, websocket, stream_audio=stream_audio,
                      device_id=_valid_device(websocket.query_params.get("device")))

    def send_json_threadsafe(payload: dict):
        """For messages that don't belong to Session (partial transcripts, connection-level
        status) but still need to reach the browser from this background STT thread.
        Raises ClientGone if the browser has already disconnected."""
        session._send(websocket.send_json(payload))

    audio_queue: "queue.Queue" = queue.Queue()
    handled_turns = set()
    last_partial = {"text": None, "at": None}  # when the transcript last changed (for [timing])

    def audio_generator():
        """Feeds the AssemblyAI client from whatever the browser sends over the WebSocket.
        A None sentinel (put on disconnect) ends the stream cleanly."""
        while True:
            chunk = audio_queue.get()
            if chunk is None:
                return
            yield chunk

    def on_begin(client, event: BeginEvent):
        print(f"[session] started: {event.id}")

    def on_turn(client, event: TurnEvent):
        text = (event.transcript or "").strip()
        if session.closed:
            return
        if not event.end_of_turn:
            # Live partial transcript, shown on screen as she's still speaking.
            if text and text != last_partial["text"]:
                last_partial["text"], last_partial["at"] = text, time.perf_counter()
            if text:
                try:
                    send_json_threadsafe({"type": "partial", "text": text})
                except ClientGone:
                    pass
            return
        order = getattr(event, "turn_order", None)
        if order is not None:
            if order in handled_turns:
                return
            handled_turns.add(order)
        # How long AssemblyAI waited after the words stopped changing before ending the turn.
        # Approximate (partials arrive in batches), but a consistently large value here means
        # the end-of-turn silence settings, not our pipeline, are adding the wait.
        silence = time.perf_counter() - last_partial["at"] if last_partial["at"] else None
        last_partial["text"], last_partial["at"] = None, None
        if not text:
            return
        if _looks_hallucinated(text):
            print(f"[HEARD] {text}  (looks like a hallucinated filler phrase - ignored)")
            return
        print(f"[HEARD] {text}")
        timing.turn_start()
        timing.log(
            "stt.end_of_turn",
            silence_after_last_partial=f"{silence:.2f}s" if silence is not None else "n/a",
            formatted=getattr(event, "turn_is_formatted", None),
            language=getattr(event, "language_code", None),
            chars=len(text),
        )
        started = time.time()
        try:
            send_json_threadsafe({"type": "final", "text": text})
            if not _pipeline_quota_ok():
                print("[session] hourly quota guard triggered — skipping pipeline call")
                send_json_threadsafe({"type": "quota_exceeded"})
                return
            _record_pipeline_call()
            session.process_sync(text)  # blocking: fine, this thread's only job is this session
            print(f"[session] replied in {time.time() - started:.1f}s")
        except ClientGone:
            print(f"[session] browser left after {time.time() - started:.1f}s, before the reply was sent")
        except Exception:
            print("[session] pipeline error:")
            traceback.print_exc()
        finally:
            timing.turn_end()

    def on_terminated(client, event: TerminationEvent):
        print(f"[session] done: {event.audio_duration_seconds}s processed")

    def on_error(client, error: StreamingError):
        print(f"[session] STT error: {error}")

    def run_stt():
        client = StreamingClient(
            StreamingClientOptions(api_key=os.environ["ASSEMBLYAI_API_KEY"])
        )
        client.on(StreamingEvents.Begin, on_begin)
        client.on(StreamingEvents.Turn, on_turn)
        client.on(StreamingEvents.Termination, on_terminated)
        client.on(StreamingEvents.Error, on_error)
        stt_options = {}
        if STT_MIN_TURN_SILENCE_MS is not None:
            stt_options["min_turn_silence"] = STT_MIN_TURN_SILENCE_MS
        if STT_MAX_TURN_SILENCE_MS is not None:
            stt_options["max_turn_silence"] = STT_MAX_TURN_SILENCE_MS
        if STT_KEYTERMS:
            terms = _patient_keyterms()
            if terms:
                stt_options["keyterms_prompt"] = terms
                print(f"[session] boosting {len(terms)} patient names in speech recognition")
        client.connect(
            StreamingParameters(
                sample_rate=SAMPLE_RATE, speech_model="whisper-rt", format_turns=True, **stt_options
            )
        )
        try:
            client.stream(audio_generator())
        finally:
            client.disconnect(terminate=True)

    stt_thread = threading.Thread(target=run_stt, daemon=True)
    stt_thread.start()

    try:
        while True:
            message = await websocket.receive_bytes()
            audio_queue.put(message)
    except WebSocketDisconnect:
        print("[session] client disconnected")
    finally:
        session.closed = True  # stops any in-flight pipeline at its next send
        audio_queue.put(None)  # let the STT thread exit its generator cleanly


# Serves static/index.html, app.js, recorder-worklet.js at the root path.
# Must be mounted last — routes above take priority over static files.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
