import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from agents import (
    AMANAT_PERSONA,
    REPLY_TEMPLATES,
    TEMPLATE_SAME_OR_NEW,
    TEMPLATE_URGENT_UPDATED,
    call_llm,
    parse_json,
    reply_agent,
    run_agents,
    template_recorded,
    use_urdu_name,
    verification_agent,
)
from web_tts import synthesize, synthesize_stream
import care
import db
import timing

MAX_FOLLOWUPS = 2          # how many times Amanat asks a question about the same visit
PENDING_TIMEOUT = 120       # seconds; a question awaiting an answer expires after this
                            # (raised from 90 — was too tight against the pipeline's current latency)
CONTINUATION_WINDOW = 45    # seconds; a visit just recorded can still receive one more sentence
                            # (raised from 20 — a slow reply could outlast the old window entirely,
                            # making a genuine continuation look like a brand-new visit. Revisit
                            # once the pipeline's per-step latency is fixed — this is a safety
                            # margin, not the real solution to a slow round trip.)
CLARIFY_TIMEOUT = 60        # seconds to wait for an answer to "same patient, or a new one?"

# On a possible continuation, analyse "earlier sentence + new sentence" at the same time as the
# new sentence alone, instead of one after the other. Same number of Gemini calls as before;
# the merged one is simply thrown away if the new sentence turns out to be a different patient.
# Set PARALLEL_CONTINUATION=0 to go back to running them one after the other.
PARALLEL_CONTINUATION = os.environ.get("PARALLEL_CONTINUATION", "1").strip().lower() not in ("0", "false", "no", "off")

# Worker threads for work that can overlap within one turn (TTS while the visit is saved, the
# merged continuation analysis, a follow-up question's audio). Shared by all sessions.
_pool = ThreadPoolExecutor(max_workers=int(os.environ.get("PIPELINE_WORKERS", "8")), thread_name_prefix="pipeline")


def _in_background(fn, *args, **kwargs):
    """Start fn on a worker thread, keeping this turn's [timing] context. Returns a Future."""
    return _pool.submit(timing.run_in_turn(fn, *args, **kwargs))


class ClientGone(Exception):
    """The browser disconnected mid-pipeline. Raised so the pipeline stops right away
    instead of spending more Gemini/TTS calls on a reply nobody will hear."""


def _merge_extracted(new: dict, prior: dict) -> dict:
    """Fill gaps in a fresh extraction using a known-good extraction from earlier in the same
    conversation. Needed because a continuation is built by concatenating raw transcript text
    and re-running extraction on the WHOLE thing (see process_sync) — that re-extraction can
    silently drop a detail (a name, a symptom) that was caught correctly the first time,
    especially on quieter or mumbled audio. This backfills from what's already confirmed,
    it never overwrites something the new extraction did find."""
    merged = dict(new)
    if not merged.get("patient_name") and prior.get("patient_name"):
        merged["patient_name"] = prior["patient_name"]
    prior_symptoms = prior.get("symptoms") or []
    new_symptoms = merged.get("symptoms") or []
    for s in prior_symptoms:
        if s not in new_symptoms:
            new_symptoms.append(s)
    merged["symptoms"] = new_symptoms
    prior_vitals = prior.get("vitals") or {}
    new_vitals = merged.get("vitals") or {}
    for key in ("temp", "bp"):
        if not new_vitals.get(key) and prior_vitals.get(key):
            new_vitals[key] = prior_vitals[key]
    merged["vitals"] = new_vitals
    if merged.get("visit_type", "general") == "general" and prior.get("visit_type"):
        merged["visit_type"] = prior["visit_type"]
    details = dict(prior.get("details") or {})
    details.update(merged.get("details") or {})
    merged["details"] = details
    return merged


class Session:
    """One connected browser = one Session. Mirrors orchestrator.py's process() exactly,
    except output goes over a WebSocket instead of print()/local speaker playback.
    process_sync() is called from the background STT thread (see server.py) — that's fine,
    since every call inside it (Gemini, edge-tts) is a normal blocking call."""

    def __init__(self, loop: asyncio.AbstractEventLoop, websocket, stream_audio: bool = False,
                 device_id=None):
        self.loop = loop
        self.websocket = websocket
        # True when the browser asked for replies in pieces (see server.py). Otherwise every
        # reply goes out as one complete mp3, exactly as before.
        self.stream_audio = stream_audio
        # Random id the demo page keeps in its browser storage; visits are saved with it so the
        # page can show "your recorded visits" again after a reload (see db.get_device_visits).
        self.device_id = device_id
        self._pending = None
        self._recent = None
        self._clarify = None
        self._last_patient_id = None
        self.closed = False    # set by server.py when the browser disconnects

    def _status(self, step: str):
        """Tell the browser which pipeline step is running: understanding / replying / listening."""
        if not self.closed:
            self._send(self.websocket.send_json({"type": "status", "step": step}))

    # ---- output helpers: schedule the actual send onto the asyncio loop and wait for it ----
    def _send(self, coro):
        """Run one websocket send from this background thread. Any failure means the browser
        is gone, so it's turned into ClientGone to end the pipeline quietly."""
        if self.closed:
            coro.close()
            raise ClientGone()
        try:
            asyncio.run_coroutine_threadsafe(coro, self.loop).result()
        except Exception:
            self.closed = True
            raise ClientGone()

    def _send_json(self, payload: dict):
        self._send(self.websocket.send_text(json.dumps(payload, ensure_ascii=False)))

    @staticmethod
    def _tts(text: str) -> bytes:
        return synthesize(text) if text else b""

    def _send_audio(self, audio_bytes: bytes):
        """Send one finished spoken reply. The BROWSER decides when to unmute the mic again
        (based on actual playback duration there), so no timing coordination happens here.
        A streaming browser gets it as a one-piece stream (audio_start, bytes, audio_end)."""
        if not audio_bytes:
            return
        with timing.timed("send.audio", bytes=len(audio_bytes)):
            if self.stream_audio:
                self._send_json({"type": "audio_start"})
                self._send(self.websocket.send_bytes(audio_bytes))
                self._send_json({"type": "audio_end"})
            else:
                self._send(self.websocket.send_bytes(audio_bytes))
        timing.audio_sent()

    def _stream_speech(self, text: str):
        """Make and send a reply's audio. Streaming browser: audio_start, then mp3 pieces as
        edge-tts produces them (playback starts on the first one), then audio_end.
        Otherwise: the complete mp3 in one message, as before."""
        if not text:
            return
        if not self.stream_audio:
            self._send_audio(self._tts(text))
            return
        self._send_json({"type": "audio_start"})
        sent = 0

        def on_chunk(chunk):
            nonlocal sent
            self._send(self.websocket.send_bytes(chunk))
            sent += 1
            if sent == 1:
                timing.audio_sent()

        try:
            synthesize_stream(text, on_chunk)
        finally:
            # Always close the stream, even if TTS failed halfway, so the browser moves on
            # instead of waiting for pieces that will never come.
            if not self.closed:
                self._send_json({"type": "audio_end"})
        timing.log("send.audio_stream", chunks=sent)

    def _prepare_speech(self, text_fn):
        """Start working on a reply in the background: its text (may call Gemini) and, for a
        non-streaming browser, its whole mp3. Hand the result to _deliver_speech()."""
        if self.stream_audio:
            return _in_background(text_fn)  # audio is streamed at delivery time instead
        return _in_background(lambda: self._tts(text_fn()))

    def _deliver_speech(self, prepared):
        if self.stream_audio:
            self._stream_speech(prepared.result())
        else:
            self._send_audio(prepared.result())

    def _speak(self, text: str):
        """Synthesize and send audio bytes."""
        if not text:
            return
        if self.closed:
            raise ClientGone()
        self._status("replying")
        self._stream_speech(text)

    def _reply(self, result, extracted, triage, name_urdu=None):
        """The spoken confirmation. For a routine visit of a patient not seen before, a fixed
        sentence (no Gemini call, cacheable audio — see agents.REPLY_TEMPLATES). Otherwise the
        reply combined_agent already wrote (saves a whole Gemini round trip), falling back to a
        fresh reply_agent call if there wasn't one, or if merging with an earlier sentence
        changed the patient's name since it was written. For a child's visit, the vaccine
        reminder is added after it."""
        reply = None
        if not triage["escalate"] and not result.get("history"):
            reply = template_recorded(extracted.get("patient_name"), name_urdu)
        if not reply:
            if result.get("reply") and extracted.get("patient_name") == result["extracted"].get("patient_name"):
                reply = result["reply"]
            else:
                reply = reply_agent(extracted, triage, name_urdu)
            # If Gemini still copied the name in Latin/Devanagari letters, swap in Urdu script.
            reply = use_urdu_name(reply, extracted.get("patient_name"), name_urdu)
        vaccines, _ = self._vaccines(extracted)
        return f"{reply} {vaccines}".strip()

    @staticmethod
    def _vaccines(extracted):
        """(Urdu sentence, English note) about vaccines due for a child's visit, else ("", "")."""
        if extracted.get("visit_type") != "child":
            return "", ""
        return care.vaccine_reminder((extracted.get("details") or {}).get("child_age_weeks"))

    def _emit(self, result, extracted, triage, visit_id):
        """Save the visit (updating it if this conversation already saved it) and show it
        in the browser. Returns the visit id."""
        with timing.timed("db.save_visit"):
            visit_id, patient_id = db.save_visit(
                extracted, triage, patient_id=result.get("patient_id"), visit_id=visit_id,
                device_id=self.device_id,
            )
        self._last_patient_id = patient_id  # kept with the visit in _remember()
        history = result.get("history") or []
        self._send_json({
            "type": "record",
            "extracted": extracted,
            "triage": triage,
            "visit_type": extracted.get("visit_type", "general"),
            "visit_number": len(history) + 1 if patient_id else None,
            "last_seen": care.days_ago(history[0]["recorded_at"]) if history else None,
            "vaccine_note": self._vaccines(extracted)[1],
        })
        return visit_id

    def _emit_then_say(self, result, extracted, triage, visit_id, prepared):
        """Save + show the visit while the reply is being prepared, then send the audio.
        The record message still reaches the browser before the audio, as before."""
        visit_id = self._emit(result, extracted, triage, visit_id)
        self._status("replying")
        self._deliver_speech(prepared)
        return visit_id

    def _record_and_reply(self, result, extracted, triage, visit_id, name_urdu):
        prepared = self._prepare_speech(lambda: self._reply(result, extracted, triage, name_urdu))
        return self._emit_then_say(result, extracted, triage, visit_id, prepared)

    def _remember(self, text, name, extracted, visit_id, name_urdu):
        self._recent = {"text": text, "name": name, "extracted": extracted, "visit_id": visit_id,
                        "name_urdu": name_urdu, "patient_id": self._last_patient_id, "time": time.time()}

    @staticmethod
    def _run_agents(text, visit_id, kind):
        with timing.timed("pipeline.run_agents", kind=kind):
            return run_agents(text, current_visit_id=visit_id)

    # ---- same-patient disambiguation (identical to orchestrator.py) ----
    def _same_patient(self, known_name, candidate_name):
        if not known_name or not candidate_name:
            return True
        a, b = known_name.strip().lower(), candidate_name.strip().lower()
        return a == b or a in b or b in a

    def _ask_same_or_new(self, recent_name):
        if REPLY_TEMPLATES:
            return TEMPLATE_SAME_OR_NEW
        name_clause = f" ({recent_name})" if recent_name else ""
        return call_llm(
            AMANAT_PERSONA,
            "Ask ONE short, polite question in Urdu script: is what she is about to say about the "
            f"same patient as before{name_clause}, or a different, new patient? "
            "Reply with only the question.",
            purpose="same_or_new_question",
        )

    def _classify_same_or_new(self, recent_name, answer_text):
        prompt = """You are reading a Lady Health Worker's answer to the question "is this the same
patient as before, or a new patient?" (asked in Urdu). Her answer may be Urdu script, Roman
Urdu, Punjabi, or English, and may be short.
Return ONLY JSON: {"same": true | false | null}   (use null only if her answer truly does not
indicate either way)."""
        user = f"Previous patient's name (may be unknown): {recent_name or '(unknown)'}\nHer answer: {answer_text}"
        data = parse_json(call_llm(prompt, user, json_mode=True, purpose="same_or_new_classify"), {})
        return data.get("same") if isinstance(data, dict) else None

    def process_sync(self, transcript: str):
        """Handle one finished sentence from the Lady Health Worker. Identical control flow
        to orchestrator.py's process() — see that file's comments for the reasoning."""
        transcript = (transcript or "").strip()
        if not transcript:
            return
        self._status("understanding")
        try:
            self._process_sync_inner(transcript)
        finally:
            if not self.closed:
                self._status("listening")

    def _process_sync_inner(self, transcript: str):

        count, already_escalated = 0, False
        prior_extracted = None  # known-good extraction from earlier in this conversation, if any
        prior_name_urdu = None  # that extraction's patient name in Urdu script, if known
        visit_id = None         # this visit's saved row, once there is one (updated, never duplicated)
        prefetched = None       # run_agents() result for `text` computed early (a dict or a Future)

        clarify, self._clarify = self._clarify, None
        if clarify and time.time() - clarify["time"] <= CLARIFY_TIMEOUT:
            same = self._classify_same_or_new(clarify["recent_name"], transcript)
            if same is True:
                text = f"{clarify['recent_text']} {clarify['held']} {transcript}"
                prior_extracted = clarify.get("recent_extracted")
                prior_name_urdu = clarify.get("recent_name_urdu")
                visit_id = clarify.get("recent_visit_id")
            elif same is False:
                text = f"{clarify['held']} {transcript}"
            else:
                text = f"{clarify['recent_text']} {clarify['held']} {transcript}"
                prior_extracted = clarify.get("recent_extracted")
                prior_name_urdu = clarify.get("recent_name_urdu")
                visit_id = clarify.get("recent_visit_id")

        else:
            pending, self._pending = self._pending, None
            if pending and time.time() - pending["time"] > PENDING_TIMEOUT:
                pending = None

            if pending:
                text = f"{pending['text']} {transcript}"
                count, already_escalated = pending["count"], pending["escalated"]
                prior_extracted = pending.get("extracted")
                prior_name_urdu = pending.get("name_urdu")
                visit_id = pending.get("visit_id")

            else:
                recent = self._recent
                if recent and time.time() - recent["time"] > CONTINUATION_WINDOW:
                    recent = None
                    self._recent = None

                if recent:
                    merged_text = f"{recent['text']} {transcript}"
                    merged_future = None
                    if PARALLEL_CONTINUATION:
                        merged_future = _in_background(
                            self._run_agents, merged_text, recent.get("visit_id"), "continuation_merged"
                        )
                    # current_visit_id=None: identical to how a brand-new visit is analysed, so
                    # this result can be reused as-is if it turns out to be a different patient
                    # (it used to be run a second time with exactly that input).
                    solo = self._run_agents(transcript, None, "continuation_solo")
                    if solo["outcome"] == "skip":
                        if merged_future:
                            merged_future.cancel()
                        self._send_json({"type": "skip"})
                        return
                    candidate_name = solo["extracted"].get("patient_name")
                    if candidate_name:
                        # The STT writes one name in different scripts from turn to turn
                        # ("Buxira" / "بشرا" in the Railway logs), so a spelling mismatch alone
                        # isn't proof of a new patient: also accept Gemini matching both to the
                        # same known patient (it compares across scripts).
                        same_by_id = (
                            solo.get("patient_id") is not None
                            and solo.get("patient_id") == recent.get("patient_id")
                        )
                        if same_by_id or self._same_patient(recent["name"], candidate_name):
                            text = merged_text
                            prior_extracted = recent.get("extracted")
                            prior_name_urdu = recent.get("name_urdu")
                            visit_id = recent.get("visit_id")
                            prefetched = merged_future
                        else:
                            if merged_future:
                                merged_future.cancel()
                            self._recent = None
                            text = transcript
                            prefetched = solo
                    else:
                        if merged_future:
                            merged_future.cancel()
                        self._speak(self._ask_same_or_new(recent["name"]))
                        self._clarify = {
                            "recent_text": recent["text"],
                            "recent_name": recent["name"],
                            "recent_extracted": recent.get("extracted"),
                            "recent_name_urdu": recent.get("name_urdu"),
                            "recent_visit_id": recent.get("visit_id"),
                            "held": transcript,
                            "time": time.time(),
                        }
                        return
                else:
                    text = transcript

        if isinstance(prefetched, dict):
            result = prefetched
        elif prefetched is not None:
            result = prefetched.result()
        else:
            result = self._run_agents(text, visit_id, "main")
        outcome = result["outcome"]

        if outcome == "skip":
            self._send_json({"type": "skip"})
            return

        extracted = result["extracted"]
        if prior_extracted:
            extracted = _merge_extracted(extracted, prior_extracted)
        triage = result["triage"]
        name = extracted.get("patient_name")
        # Urdu-script spelling of the name for fixed replies: from this analysis if it's the one
        # that supplied the name, otherwise from the earlier sentence the name was merged from.
        if name == result["extracted"].get("patient_name"):
            name_urdu = result.get("name_urdu")
        else:
            name_urdu = prior_name_urdu

        if outcome == "record":
            visit_id = self._record_and_reply(result, extracted, triage, visit_id, name_urdu)
            self._remember(text, name, extracted, visit_id, name_urdu)
            return

        if outcome == "escalate":
            missing = result["verification"]["missing"]
            if already_escalated and not missing:
                confirm = TEMPLATE_URGENT_UPDATED if REPLY_TEMPLATES else None
                prepared = self._prepare_speech(lambda: confirm or call_llm(
                    AMANAT_PERSONA,
                    "Confirm briefly, in Urdu script, that the patient's details have been "
                    "added to the urgent case.",
                    purpose="urgent_update_confirm",
                ))
                visit_id = self._emit_then_say(result, extracted, triage, visit_id, prepared)
                self._remember(text, name, extracted, visit_id, name_urdu)
                return

            # Prepare both spoken parts at once: the danger reply (unless already said on an
            # earlier turn) and, if something is missing, the follow-up question. They're still
            # sent in the same order as before — danger first, question after. The question's
            # audio is made in full meanwhile, so it's ready the moment the reply is sent.
            reply_audio = None
            if not already_escalated:
                reply_audio = self._prepare_speech(lambda: self._reply(result, extracted, triage, name_urdu))
            question_job = None
            if missing:
                def make_question():
                    # Re-check against the merged extraction: an earlier sentence may have filled the gap.
                    question = verification_agent(
                        extracted, result["verification"].get("follow_up_question", "")
                    ).get("follow_up_question")
                    audio = self._tts(question) if question and count < MAX_FOLLOWUPS else b""
                    return question, audio
                question_job = _in_background(make_question)

            visit_id = self._emit(result, extracted, triage, visit_id)
            if reply_audio is not None:
                self._status("replying")
                self._deliver_speech(reply_audio)
            if question_job is None:
                self._remember(text, name, extracted, visit_id, name_urdu)
                return
            question, question_audio = question_job.result()
            if question and count < MAX_FOLLOWUPS:
                self._status("replying")
                self._send_audio(question_audio)
                self._pending = {"text": text, "count": count + 1, "escalated": True, "extracted": extracted,
                                 "name_urdu": name_urdu, "visit_id": visit_id, "time": time.time()}
            else:
                self._remember(text, name, extracted, visit_id, name_urdu)
            return

        if outcome == "ask":
            if count < MAX_FOLLOWUPS:
                self._speak(result["verification"]["follow_up_question"])
                self._pending = {"text": text, "count": count + 1, "escalated": False, "extracted": extracted,
                                 "name_urdu": name_urdu, "visit_id": visit_id, "time": time.time()}
            else:
                visit_id = self._record_and_reply(result, extracted, triage, visit_id, name_urdu)
                self._remember(text, name, extracted, visit_id, name_urdu)
