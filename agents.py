import json
import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

import care
import db
import timing

load_dotenv()

# Free-tier quotas are counted PER MODEL, so we keep a list of models and fall back to the
# next one when a model has used up its daily quota (or does not exist for your account).
# Put the model IDs you see on https://aistudio.google.com/rate-limit in .env like this:
#   GEMINI_MODELS=model-one,model-two,model-three
# Models with the highest free "requests per day" are best for development, so list them first.
DEFAULT_MODELS = (
    "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.8-flash,"
    "gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash"
)
MODELS = [
    m.strip()
    for m in os.environ.get("GEMINI_MODELS", DEFAULT_MODELS).split(",")
    if m.strip()
]

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

AMANAT_PERSONA = """You are Amanat, a calm, respectful voice assistant supporting a Lady Health Worker (LHW)
during home visits. You speak in simple Urdu written in Urdu script, formal enough to show respect
(use "aap"), warm but brief -- the LHW is busy and often has her hands full. Never sound robotic or
overly cheerful. When escalating a case, sound steady and reassuring, not alarming. When asking a
follow-up question, keep it short and specific. Keep every reply under two short sentences.
Write EVERY word in Urdu script (Nastaliq), with no Latin/English letters anywhere in your reply,
even for medical or technical terms -- write those in Urdu script too (e.g. "بی پی" for BP,
"ٹمپریچر" for temperature), never spell them out in English letters. This matters because your
reply is read aloud by a text-to-speech voice, and any English-letter word in the middle of an
Urdu sentence gets mispronounced."""

# DRAFT red-flag list. STILL NOT clinically reviewed — get sign-off before relying on this.
# Original 5 lines: from the team's earlier draft.
# Two lines below added from WHO IMCI's general danger signs and a Pakistan-specific LHW
# training gap (pre-eclampsia/eclampsia) identified when comparing against WHO's checklist
# and published LHW-program research — see the team's hackathon notes for sources.
# This is still a draft: get a clinician or LHW supervisor to review before the final submission.
RED_FLAGS = """- Very high fever (about 103 F / 39.5 C or higher)
- Difficulty breathing, fast or labored breathing, blue or bluish lips
- Severe or heavy bleeding (including heavy bleeding after delivery)
- Unconscious, unresponsive, fainting, or convulsions/seizures
- Low blood pressure (about 90/60 or lower) together with dizziness or weakness
- Baby or patient unable to drink or breastfeed at all (WHO IMCI general danger sign)
- High blood pressure during pregnancy together with severe headache, swelling, or
  vision changes (possible pre-eclampsia/eclampsia -- a leading cause of maternal
  death in Pakistan, and an area LHW supervisors have flagged as under-trained)"""

VALID_INTENTS = {"new_visit", "follow_up", "emergency", "off_topic"}


# ---------------------------------------------------------------- errors
class QuotaExhausted(Exception):
    """Every configured model has used its free daily quota (or is not available)."""


class NetworkDown(Exception):
    """The internet connection to Google failed."""


# ---------------------------------------------------------------- helpers
_dead_models = set()  # models to skip for the rest of this run
_last_model = None
# Gemini 3.x models "think" before answering by default, which added 10-50s per call here.
# Our tasks (classify, extract, one short Urdu sentence) don't need it, so ask for the minimum.
# If a model rejects the setting, it's remembered here and that model is called without it.
_no_thinking_config = set()
_THINKING = types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)

# Latency guards. Without a timeout the SDK waits forever on a stuck request (its default is
# no timeout, and it does no retries of its own), so one hung call used to stall the whole reply.
# Railway logs (27 Sep): healthy calls took 0.4-2.2s, but some single calls took 10.7-18.9s.
# Past GEMINI_TIMEOUT_S it's quicker to ask the next model than to keep waiting.
GEMINI_TIMEOUT_S = float(os.environ.get("GEMINI_TIMEOUT_S", "8"))
# A model that just timed out, was overloaded (503) or hit its per-minute limit is skipped for
# this long, so every following turn doesn't pay the same failover delay again. It isn't
# blacklisted: once the cooldown passes it's tried first again. 0 turns this off.
MODEL_COOLDOWN_S = float(os.environ.get("MODEL_COOLDOWN_S", "60"))
_cooldown_until = {}


def _network_error(msg):
    text = msg.lower()
    return any(s in text for s in ("getaddrinfo", "connecterror", "connection", "timed out", "timeout"))


def _timed_out(error, msg):
    return (
        "timeout" in type(error).__name__.lower()
        or "timed out" in msg.lower()
        or "DEADLINE_EXCEEDED" in msg
    )


def _model_order():
    """Models to try, in the configured order, with any that are cooling down moved to the
    end (still tried as a last resort, never dropped)."""
    now = time.time()
    live = [m for m in MODELS if m not in _dead_models]
    ready = [m for m in live if _cooldown_until.get(m, 0) <= now]
    return ready + [m for m in live if m not in ready]


def _cool_down(model):
    if MODEL_COOLDOWN_S > 0:
        _cooldown_until[model] = time.time() + MODEL_COOLDOWN_S


def call_llm(system_prompt, user_text, json_mode=False, purpose="llm"):
    """Send one request to Gemini. purpose is only a label for the [timing] logs.
    - timeout / overloaded -> move on to the next model quickly (and let this one cool down)
    - per-minute limit     -> next model if there is one, else wait a little and retry
    - per-day limit        -> give up on that model, try the next one in MODELS
    - model not found      -> skip it
    - no internet          -> retry a couple of times, then raise NetworkDown"""
    global _last_model

    def make_config(model):
        return types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
            response_mime_type="application/json" if json_mode else "text/plain",
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=None if model in _no_thinking_config else _THINKING,
            http_options=types.HttpOptions(timeout=int(GEMINI_TIMEOUT_S * 1000)),
        )

    prompt_chars = len(system_prompt or "") + len(user_text or "")
    order = _model_order()
    for index, model in enumerate(order):
        has_next = index < len(order) - 1
        busy_tries = 0
        net_tries = 0
        timeout_tries = 0
        while True:
            started = time.perf_counter()
            try:
                resp = client.models.generate_content(
                    model=model, contents=user_text, config=make_config(model)
                )
                elapsed = time.perf_counter() - started
                if model != _last_model:
                    print(f"  [using model: {model}]")
                    _last_model = model
                print(f"  [llm {elapsed:.1f}s]")
                text = (resp.text or "").strip()
                timing.log("llm", elapsed, purpose=purpose, model=model, result="ok",
                           prompt_chars=prompt_chars, output_chars=len(text))
                return text
            except Exception as e:
                elapsed = time.perf_counter() - started
                msg = str(e)

                def failed(result):
                    timing.log("llm", elapsed, purpose=purpose, model=model, result=result)

                if "INVALID_ARGUMENT" in msg and "thinking" in msg.lower() and model not in _no_thinking_config:
                    failed("thinking_rejected")
                    print(f"  ({model} rejected the thinking setting - calling it without)")
                    _no_thinking_config.add(model)
                    continue
                if "RESOURCE_EXHAUSTED" in msg:
                    if "PerDay" in msg:
                        failed("daily_quota")
                        print(f"  (daily free quota used up for {model} - trying next model)")
                        _dead_models.add(model)
                        break
                    failed("per_minute_limit")
                    if has_next:
                        # Quotas are per model, so another model can answer right now instead
                        # of waiting 10-30s for this one's minute to roll over.
                        print(f"  (per-minute limit on {model} - trying next model)")
                        _cool_down(model)
                        break
                    busy_tries += 1
                    if busy_tries > 3:
                        raise
                    wait = 10 * busy_tries
                    print(f"  (per-minute limit on {model}, waiting {wait}s...)")
                    time.sleep(wait)
                    continue
                if "NOT_FOUND" in msg:
                    failed("not_found")
                    print(f"  (model {model} not available for your account - skipping)")
                    _dead_models.add(model)
                    break
                if "UNAVAILABLE" in msg or "503" in msg:
                    failed("unavailable")
                    busy_tries += 1
                    # Temporary overload, not exhausted quota - don't blacklist the model.
                    # With another model to try: one quick retry, then fail over (was 3s + 6s
                    # of sleeping first). Without one: a few short retries on this model.
                    if busy_tries > (1 if has_next else 3):
                        print(f"  ({model} still unavailable - trying next model)")
                        _cool_down(model)
                        break
                    time.sleep(1 if has_next else 2 * busy_tries)
                    continue
                if _timed_out(e, msg):
                    failed("timeout")
                    timeout_tries += 1
                    _cool_down(model)
                    if has_next:
                        print(f"  ({model} took longer than {GEMINI_TIMEOUT_S:.0f}s - trying next model)")
                        break
                    if timeout_tries > 1:
                        raise NetworkDown(f"{model} timed out twice after {GEMINI_TIMEOUT_S:.0f}s: {msg}")
                    continue  # last model: one more try before giving up
                if _network_error(msg):
                    failed("network")
                    net_tries += 1
                    if net_tries > 2:
                        raise NetworkDown(msg)
                    print("  (no connection to Google, retrying shortly...)")
                    time.sleep(1.5 * net_tries)
                    continue
                failed("error")
                raise
    raise QuotaExhausted(
        "All models are out of free quota for today (or unavailable): " + ", ".join(MODELS)
    )


def parse_json(text, fallback):
    """Turn the model's text into a Python object, tolerating extra text around the JSON."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return fallback


# ---------------------------------------------------------------- extraction shape
EMPTY_EXTRACTION = {
    "patient_name": None,
    "symptoms": [],
    "vitals": {"temp": None, "bp": None},
    "notes": "",
    "visit_type": "general",
    "details": {},
}


def normalise_extraction(data):
    """Make sure the extraction always has the exact shape the backend expects."""
    out = json.loads(json.dumps(EMPTY_EXTRACTION))
    if not isinstance(data, dict):
        return out
    name = data.get("patient_name")
    out["patient_name"] = name.strip() if isinstance(name, str) and name.strip() else None
    symptoms = data.get("symptoms")
    if isinstance(symptoms, list):
        out["symptoms"] = [str(s).strip() for s in symptoms if str(s).strip()]
    vitals = data.get("vitals")
    if isinstance(vitals, dict):
        for key in ("temp", "bp"):
            value = vitals.get(key)
            out["vitals"][key] = str(value).strip() if value not in (None, "") else None
    notes = data.get("notes")
    out["notes"] = notes.strip() if isinstance(notes, str) else ""
    visit_type = data.get("visit_type")
    out["visit_type"] = visit_type if visit_type in care.VISIT_TYPES else "general"
    out["details"] = care.normalise_details(data.get("details"))
    return out


EXTRACTION_RULES = f"""- patient_name: the name exactly as spoken, e.g. "Fatima" or "Ahmed's baby". If she corrects
  herself, use the corrected version. If no name is given (only "a house", "she", "bhabi",
  "her father"), use null.
- symptoms: short English terms, e.g. ["fever", "cough"]. Empty list if none mentioned.
- vitals.temp: the temperature as stated. If a number is given, write it as stated ("104 F"). If
  only a description is given with no number ("high", "normal", "mild fever"), write that
  description instead. If temperature is not mentioned at all, use null.
- vitals.bp: blood pressure as stated. If a number is given, write it as "systolic/diastolic"
  ("130/85"). If only a description is given with no number ("low", "high", "normal"), write that
  description instead ("low"). If blood pressure is not mentioned at all, use null.
  NEVER return null just because there was no number -- a word like "low" is still useful information.
- notes: short English note ONLY for clinical details that do not fit the other fields (weight,
  feeding, medicine given, how the patient looks, duration). NEVER write that a visit happened or
  that something was "checked" (e.g. do not write "Visited Rukhsana's house" or "Checked
  temperature"), and never repeat symptoms or vitals already in the other fields. If there is no
  clinical detail beyond the other fields, notes must be an empty string "".
- visit_type: one of
{care.VISIT_TYPE_RULES}
- details: only these keys, and only when actually said (leave a key out otherwise):
{care.DETAIL_RULES}
If information is unclear or garbled, use null rather than guessing. Never invent details."""

INTENT_RULES = """- "new_visit": she describes visiting a patient / patient details (even if incomplete)
- "follow_up": she describes a repeat visit or progress of an earlier case
- "emergency": she says it is an emergency or urgent help is needed
- "off_topic": no patient or visit information at all (weather, travel, small talk)
If the text is garbled but mentions symptoms or a visit, do NOT use off_topic."""

TRIAGE_RULES = f"""Escalate if ANY of these red flags is present:
{RED_FLAGS}
Do NOT escalate ordinary mild illness (mild fever, cough, headache, stomach ache, weakness that
does not come with a red flag). If a red flag is clearly described, always escalate."""


# ---------------------------------------------------------------- the separate agents (guide structure)
def router_agent(transcript):
    prompt = f"""You classify a Lady Health Worker's spoken visit note.
The text may be Urdu script, Roman Urdu, Punjabi, English, or a mix, and may be noisy.
Intents:
{INTENT_RULES}
Return ONLY JSON: {{"intent": "new_visit" | "follow_up" | "emergency" | "off_topic"}}"""
    result = parse_json(call_llm(prompt, transcript, json_mode=True, purpose="router"), {})
    if not isinstance(result, dict) or result.get("intent") not in VALID_INTENTS:
        return {"intent": "new_visit"}  # safe default: never silently drop a visit
    return result


def extraction_agent(transcript):
    prompt = f"""Extract patient visit details from this Lady Health Worker transcript
(Urdu script / Roman Urdu / Punjabi / English mixed).
Rules:
{EXTRACTION_RULES}
Return ONLY JSON: {{"patient_name": str|null, "symptoms": [str], "vitals": {{"temp": str|null, "bp": str|null}}, "notes": str, "visit_type": str, "details": {{}}}}"""
    data = parse_json(call_llm(prompt, transcript, json_mode=True, purpose="extraction"), None)
    return normalise_extraction(data)


def triage_agent(transcript, extracted, intent="new_visit"):
    """Decides if the case must be escalated. Runs BEFORE verification: a dangerous case
    must never wait for a missing patient name."""
    if intent == "emergency":
        return {"escalate": True, "reason": "LHW reported an emergency"}
    prompt = f"""You are a safety checker for Lady Health Worker visit notes in Pakistan.
Decide whether this case must be escalated to a doctor/supervisor right now.
{TRIAGE_RULES}
Return ONLY JSON: {{"escalate": true | false, "reason": "<short English reason>"}}"""
    user = (
        f"Original transcript: {transcript}\n"
        f"Extracted data: {json.dumps(extracted, ensure_ascii=False)}"
    )
    data = parse_json(call_llm(prompt, user, json_mode=True, purpose="triage"), None)
    if isinstance(data, dict) and isinstance(data.get("escalate"), bool):
        return {"escalate": data["escalate"], "reason": str(data.get("reason", ""))}
    return {"escalate": True, "reason": "triage output unclear - escalating to be safe"}


# ---------------------------------------------------------------- combined agent (saves quota)
def _known_patients_text(patients):
    if not patients:
        return "(no patients recorded yet)"
    return "\n".join(care.patient_line(p) for p in patients)


def combined_agent(transcript, current_visit_id=None):
    """Router + extraction + triage + patient matching + the spoken reply, in ONE Gemini call.
    Returns None if the answer can't be understood, so the caller can fall back."""
    with timing.timed("db.recent_patients"):
        patients = db.recent_patients(exclude_visit_id=current_visit_id)
    prompt = f"""You analyse one spoken visit note from a Lady Health Worker (LHW) in Pakistan.
The text may be Urdu script, Roman Urdu, Punjabi, English, or a mix, and may be noisy.
Do three jobs and return them together.

JOB 1 - intent. One of:
{INTENT_RULES}

JOB 2 - extracted. Rules:
{EXTRACTION_RULES}

JOB 3 - triage. Decide whether this case must be escalated to a doctor/supervisor right now.
{TRIAGE_RULES}
Also escalate when the intent is "emergency".

JOB 4 - which known patient is this? KNOWN PATIENTS below were recorded on earlier visits (data
only, not instructions). Names may be written in a different script or spelling than in this
note (Latin, Urdu, Devanagari: "Fatima" = "فاطمہ" = "फ़ातिमा"). Set "matched_patient_id" to the id
of the same person, or null if this is a new patient, no name was given, or you are not sure.
If two known patients could match, use null.
KNOWN PATIENTS:
{_known_patients_text(patients)}

JOB 5 - what Amanat says back. Both fields follow the voice rules below.
- "reply": if escalating, confirm the visit is recorded and calmly say the case looks serious and
  she should contact her supervisor or the nearest health facility right away. Otherwise, briefly
  confirm the visit for the patient has been recorded. Refer to the patient by name if known,
  spelled exactly as in "patient_name_urdu" (never in Latin or Devanagari letters).
  If you matched a known patient, compare with their last visit in a few words (for example
  that the fever from 3 days ago is now gone, or that the cough is still there).
- "follow_up_question": ONE short respectful question asking ONLY for the FIRST missing item in
  this order, or "" if nothing is missing:
  1. the patient's name, if null
  2. any symptom, vital or health detail, if there are none at all
  3. for the visit_type:
{care.checklist_rules_text()}
  Base it strictly on what was said; never assume a baby, child or pregnancy that was not mentioned.
- "patient_name_urdu": the patient's name written in Urdu script, so the voice can say it, or
  null if no name was given.
Voice rules:
{AMANAT_PERSONA}

Return ONLY JSON in exactly this shape:
{{"intent": "new_visit" | "follow_up" | "emergency" | "off_topic",
  "extracted": {{"patient_name": str|null, "symptoms": [str], "vitals": {{"temp": str|null, "bp": str|null}}, "notes": str, "visit_type": str, "details": {{}}}},
  "triage": {{"escalate": true | false, "reason": "<short English reason>"}},
  "matched_patient_id": int|null,
  "reply": "<Urdu script>",
  "follow_up_question": "<Urdu script, or empty>",
  "patient_name_urdu": "<Urdu script>" | null}}"""
    data = parse_json(call_llm(prompt, transcript, json_mode=True, purpose="combined"), None)
    if not isinstance(data, dict) or data.get("intent") not in VALID_INTENTS:
        return None
    extracted = normalise_extraction(data.get("extracted"))
    triage_raw = data.get("triage") if isinstance(data.get("triage"), dict) else {}
    escalate = triage_raw.get("escalate") is True or data["intent"] == "emergency"
    reason = str(triage_raw.get("reason", "")) or ("LHW reported an emergency" if escalate else "")
    def text_field(key):
        value = data.get(key)
        return value.strip() if isinstance(value, str) else ""

    known_ids = {p["id"] for p in patients}
    matched = data.get("matched_patient_id")
    if not isinstance(matched, int) or isinstance(matched, bool) or matched not in known_ids:
        matched = None

    return {
        "intent": data["intent"],
        "extracted": extracted,
        "triage": {"escalate": escalate, "reason": reason},
        "reply": text_field("reply"),
        "follow_up_question": text_field("follow_up_question"),
        "name_urdu": text_field("patient_name_urdu") or None,
        "matched_patient_id": matched,
    }


# ---------------------------------------------------------------- verification
def find_missing(extracted):
    missing = []
    if not extracted.get("patient_name"):
        missing.append("patient name")
    vitals = extracted.get("vitals") or {}
    has_details = (
        extracted.get("symptoms") or any(vitals.values()) or extracted.get("notes")
        or extracted.get("details")
    )
    if not has_details:
        missing.append("symptoms or health details")
    missing.extend(care.find_care_gaps(extracted))
    return missing


def verification_agent(extracted, prewritten_question=""):
    missing = find_missing(extracted)
    if not missing:
        return {"ok": True, "missing": []}
    if prewritten_question:
        return {"ok": False, "missing": missing, "follow_up_question": prewritten_question}
    if REPLY_TEMPLATES and missing[0] in QUESTION_TEMPLATES:
        # Same rule the combined prompt uses (ask only for the FIRST missing item), without
        # another Gemini round trip.
        return {"ok": False, "missing": missing, "follow_up_question": QUESTION_TEMPLATES[missing[0]]}
    known = json.dumps(extracted, ensure_ascii=False)
    question = call_llm(
        AMANAT_PERSONA,
        "Ask ONE short, respectful follow-up question in Urdu script to get ONLY the missing "
        f"information: {', '.join(missing)}.\n"
        f"What is already known so far: {known}\n"
        "Base the question strictly on what is known above. Do NOT assume or mention a baby, "
        "child, pregnancy, or any other detail that was not already stated - if nothing else is "
        "known, keep the question completely general. Reply with only the question.",
        purpose="follow_up_question",
    )
    return {"ok": False, "missing": missing, "follow_up_question": question}


# ---------------------------------------------------------------- fixed replies (no Gemini call)
# Routine confirmations and questions don't need an LLM to write them. Fixed Urdu sentences
# save a Gemini round trip where one was being made, and make the audio cacheable (web_tts).
# The LLM still writes: every escalation, and the confirmation for a patient seen before (it
# compares with their last visit). Set REPLY_TEMPLATES=0 to go back to LLM-written text
# everywhere. Have a native speaker on the team check these sentences.
REPLY_TEMPLATES = os.environ.get("REPLY_TEMPLATES", "1").strip().lower() not in ("0", "false", "no", "off")

TEMPLATE_RECORDED_NAMED = "{name} کا وزٹ ریکارڈ ہو گیا ہے۔"
TEMPLATE_RECORDED = "وزٹ ریکارڈ ہو گیا ہے۔"
TEMPLATE_URGENT_UPDATED = "مریض کی تفصیلات فوری کیس میں شامل کر دی گئی ہیں۔"
TEMPLATE_SAME_OR_NEW = "کیا یہ اسی مریض کے بارے میں ہے، یا کسی نئے مریض کے بارے میں؟"

# Follow-up questions, keyed by the English description find_missing() returns.
_CARE_QUESTIONS_BY_KEY = {
    "pregnancy_month": "وہ حمل کے کون سے مہینے میں ہیں؟",
    "bp": "ان کا بلڈ پریشر کتنا ہے؟",
    "days_since_delivery": "ان کی ڈیلیوری کو کتنے دن ہوئے ہیں؟",
    "bleeding": "کیا انہیں زیادہ خون آ رہا ہے؟",
    "child_age_weeks": "بچے کی عمر کتنی ہے؟",
    "fp_method": "وہ خاندانی منصوبہ بندی کا کون سا طریقہ استعمال کر رہی ہیں؟",
}
QUESTION_TEMPLATES = {
    "patient name": "مریض کا نام کیا ہے؟",
    "symptoms or health details": "مریض کو کیا تکلیف ہے؟",
    care.PRE_ECLAMPSIA_CHECK: "کیا انہیں شدید سر درد، چہرے یا ہاتھوں پر سوجن، یا دھندلا نظر آنے کی شکایت ہے؟",
}
for _items in care.CHECKLISTS.values():
    for _key, _desc in _items:
        if _key in _CARE_QUESTIONS_BY_KEY:
            QUESTION_TEMPLATES[_desc] = _CARE_QUESTIONS_BY_KEY[_key]

# Phrases with no name in them: synthesized once at startup so they play instantly.
FIXED_PHRASES = [TEMPLATE_RECORDED, TEMPLATE_URGENT_UPDATED, TEMPLATE_SAME_OR_NEW] + sorted(set(QUESTION_TEMPLATES.values()))

_URDU_NAME = re.compile(r"^[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿\s]+$")


def speakable_name(*candidates):
    """First candidate written purely in Urdu/Arabic script, or None. A Latin or Devanagari
    name (the STT sometimes writes those) would be mispronounced by the Urdu voice, so it
    never goes into a template; the LLM-written reply is used instead."""
    for name in candidates:
        if isinstance(name, str) and name.strip() and _URDU_NAME.match(name.strip()):
            return name.strip()
    return None


def template_recorded(patient_name, name_urdu=None):
    """Fixed "visit recorded" sentence, or None if it can't be said properly (a name that
    isn't available in Urdu script), so the caller falls back to LLM-written text."""
    if not REPLY_TEMPLATES:
        return None
    if not patient_name:
        return TEMPLATE_RECORDED
    name = speakable_name(name_urdu, patient_name)
    return TEMPLATE_RECORDED_NAMED.format(name=name) if name else None


# ---------------------------------------------------------------- reply
def use_urdu_name(text, name, name_urdu):
    """Swap a name the STT wrote in Latin/Devanagari letters for its Urdu-script spelling, so
    the Urdu voice pronounces it properly. No-op when there's nothing safe to swap in."""
    urdu = speakable_name(name_urdu)
    if not text or not name or not urdu or name == urdu:
        return text
    return text.replace(name, urdu)


def reply_agent(extracted, triage, name_urdu=None):
    # Give Gemini the Urdu-script spelling when known, so the spoken name matches it.
    name = speakable_name(name_urdu) or extracted.get("patient_name") or "the patient (name not known yet)"
    if triage["escalate"]:
        task = (
            f"Confirm that the visit for {name} is recorded, and calmly say this case looks "
            "serious and she should contact her supervisor or the nearest health facility "
            f"right away. Reason: {triage['reason']}."
        )
    else:
        task = f"Confirm briefly that the visit for {name} has been recorded."
    return call_llm(
        AMANAT_PERSONA, task + " Reply with only the spoken sentences, in Urdu script.",
        purpose="reply",
    )


# ---------------------------------------------------------------- text-only pipeline
def run_agents(transcript, combined=True, current_visit_id=None):
    """Runs the agents on TEXT (no audio).
    combined=True  -> 1 Gemini call for router+extraction+triage (saves free quota)
    combined=False -> 3 separate calls, like the original guide
    outcome is one of: "skip", "escalate", "ask", "record"."""
    analysis = combined_agent(transcript, current_visit_id) if combined else None

    if analysis is None:  # combined answer unusable (or combined=False): use separate agents
        # 3 sequential Gemini calls instead of 1 - shows up as a very slow turn.
        timing.log("pipeline.fallback_to_separate_agents")
        intent = router_agent(transcript)["intent"]
        if intent == "off_topic":
            return {"intent": intent, "outcome": "skip"}
        extracted = extraction_agent(transcript)
        triage = triage_agent(transcript, extracted, intent)
        analysis = {"intent": intent, "extracted": extracted, "triage": triage}

    intent = analysis["intent"]
    extracted = analysis["extracted"]
    triage = analysis["triage"]

    if intent == "off_topic":
        return {"intent": intent, "outcome": "skip"}

    # Known patient: look at their past visits. A symptom that keeps coming back visit after
    # visit is a reason to escalate even when each visit alone looks mild.
    patient_id = analysis.get("matched_patient_id")
    history = []
    if patient_id:
        with timing.timed("db.patient_history"):
            history = db.patient_history(patient_id, exclude_visit_id=current_visit_id)
    if history and not triage["escalate"]:
        persistent = care.persistent_symptoms(extracted.get("symptoms"), history)
        if persistent:
            triage = {
                "escalate": True,
                "reason": f"{', '.join(persistent)} reported on {care.PERSISTENT_VISITS} visits in a row - not improving",
            }
            analysis["reply"] = ""  # the pre-written reply didn't know about the escalation

    if triage["escalate"]:
        missing = find_missing(extracted)
        verification = {
            "ok": not missing,
            "missing": missing,
            "follow_up_question": analysis.get("follow_up_question", "") if missing else "",
        }
        outcome = "escalate"
    else:
        verification = verification_agent(extracted, analysis.get("follow_up_question", ""))
        outcome = "record" if verification["ok"] else "ask"

    return {
        "intent": intent,
        "extracted": extracted,
        "triage": triage,
        "verification": verification,
        "outcome": outcome,
        "reply": analysis.get("reply", ""),  # pre-written by combined_agent; "" if unavailable
        "name_urdu": analysis.get("name_urdu"),
        "patient_id": patient_id,
        "history": history,
    }
