# Amanat Health — Voice Assistant for Lady Health Workers

*Built for the AssemblyAI Voice Agent Hackathon (lablab.ai)*

**🔴 Live demo:** [maryumakram16.github.io/amanat-voice-agent](https://maryumakram16.github.io/amanat-voice-agent/)

Amanat Health lets a Lady Health Worker (LHW) record a home-visit report by speaking
instead of writing it down. She talks the way she would to a colleague; the assistant
listens, pulls out the patient's details, checks for danger signs, saves the visit, and
answers out loud in Urdu.

- **Speaks naturally** — Urdu, Punjabi, English, or any mix of the three, in the same sentence
- **Listens in real time** and pulls out the patient's name and health details
- **Checks every visit for danger signs** and escalates serious cases immediately
- **Replies out loud**, in Urdu, with a short spoken confirmation that starts playing within moments
- **Never guesses** — asks one short follow-up question when something important is missing
- **Remembers patients** — recognises a returning patient, compares with the last visit, and
  flags a symptom that keeps coming back
- **Knows the visit type** — pregnancy, after delivery, child, family planning — and asks for
  what that visit needs (for example the month of pregnancy, or a child's age for vaccines)

---

## Contents

1. [The problem](#the-problem)
2. [The solution](#the-solution)
3. [What a live exchange looks like](#what-a-live-exchange-looks-like)
4. [How it works — the pipeline](#how-it-works--the-pipeline)
5. [Conversation logic](#conversation-logic)
6. [Speed: how the reply is kept fast](#speed-how-the-reply-is-kept-fast)
7. [Patient names: recognition and pronunciation](#patient-names-recognition-and-pronunciation)
8. [Danger-sign list](#danger-sign-list--status-draft-pending-clinical-review)
9. [Repository structure](#repository-structure)
10. [Tech stack](#tech-stack)
11. [Running it locally](#running-it-locally)
12. [Environment variables](#environment-variables)
13. [Deploying](#deploying)
14. [Monitoring and troubleshooting](#monitoring-and-troubleshooting)
15. [WebSocket message reference](#websocket-message-reference)
16. [Known limitations](#known-limitations)
17. [Team](#team)

---

## The problem

Pakistan's Lady Health Worker Programme is one of the largest community health systems in
the world: roughly 90,000+ LHWs, each responsible for about a thousand people, reaching close
to 60% of the country's population — mostly rural, mostly underserved.

Every one of those visits is still recorded the same way it has been for decades: on paper,
standing up, often in poor light, sometimes hours after the visit when details have already
blurred. That creates three concrete problems:

1. **Nothing is structured.** A paper note can't be searched, aggregated, or used to spot a
   pattern across visits.
2. **Nothing is reviewed in real time.** A dangerous symptom written on paper sits there until
   someone reads it later — there's no moment where it gets flagged.
3. **Typing isn't realistic for the job.** An LHW's hands are often full — with a child, a
   stethoscope, or her own notebook and pen. A form-based digital app doesn't solve the
   problem; it just moves the same friction onto a screen.

## The solution

Amanat Health replaces the notebook with a voice conversation. The LHW speaks the visit
exactly as she would say it out loud to a colleague. Amanat Health:

- **Listens** in real time, handling Urdu, Punjabi, and English mixed freely in the same
  sentence — because that's how people actually speak, not how a form expects them to.
- **Extracts** the patient's name, symptoms, vitals, visit type, and visit-specific details —
  without guessing. If something essential is missing, it asks one short, specific follow-up
  question instead of filing an incomplete record.
- **Screens every visit** against a danger-sign list. A serious case is flagged immediately —
  before anything else happens in that turn — with an alert tone and a red banner.
- **Replies out loud**, in Urdu, with a short calm confirmation, so the LHW knows the visit is
  recorded without breaking her stride or looking at a screen.
- **Remembers the conversation.** A follow-up sentence about the same patient is merged into
  one record, not filed as a second, disconnected entry — and if she starts describing someone
  without saying who, the assistant asks directly.
- **Keeps the record.** Every visit is saved (on a persistent volume) and shown on the admin
  dashboard; the demo page also lists the visits recorded from that browser after a reload.
- **Remembers patients across visits.** A returning patient is linked to their earlier visits,
  even when the name comes out in a different script; the reply mentions what changed, and a
  symptom reported on three visits in a row is escalated.
- **Reminds about child vaccines** due now or coming up (Pakistan EPI schedule, draft).

Everything is written to a shared, live dashboard — the read side a supervisor would use.

---

## What a live exchange looks like

**LHW:** *"Fatima ke ghar gayi thi, usko bukhar hai aur khaansi bhi, do din se."* —
"Went to Fatima's house — she has fever and a cough, for two days."

**Amanat:** *"فاطمہ کا وزٹ ریکارڈ ہو گیا ہے۔"* — "Fatima's visit has been recorded."

If a danger sign is present instead — high fever with difficulty breathing, for example —
Amanat Health still confirms the record, but calmly tells her to contact her supervisor or
the nearest health facility right away. The visit is flagged on the dashboard immediately,
and the page plays a distinct alert tone with a red banner. If the patient's name was
missing, the follow-up question comes straight after the warning — danger first, questions
after.

---

## How it works — the pipeline

```
Browser (mic)
   │  raw audio from an AudioWorklet, downsampled to 16 kHz PCM16
   ▼
WebSocket  ─────────────────────────────────────────────────────────────────┐
   │                                                                         │
   ▼                                                                         │
AssemblyAI real-time STT (whisper-rt model)                                  │
   │  partial transcripts stream back live (shown on screen as she talks)    │
   │  finished turns are filtered for Whisper "hallucinations"               │
   │  (e.g. "Thank you.", "Thanks for watching", text in unrelated scripts)  │
   ▼                                                                         │
Conversation state machine (session.py)                                      │
   │  is this a new visit, an answer to a question, or a continuation?       │
   ▼                                                                         │
ONE combined Gemini call (agents.py)                                         │
   │  intent + extraction + triage + known-patient match                     │
   │  + the spoken reply + the follow-up question, as one JSON answer        │
   ▼                                                                         │
Rules in plain Python (agents.py, care.py)                                   │
   │  missing-information check, visit-type checklist, symptom-history       │
   │  escalation, vaccine reminder                                           │
   ▼                                                                         │
Save the visit (SQLite)  ──in parallel──  prepare the reply                  │
   │                                        fixed Urdu sentence for routine  │
   │                                        cases, Gemini-written otherwise  │
   ▼                                                                         │
edge-tts (ur-PK-UzmaNeural) → mp3 streamed to the browser in pieces  ────────┘
   │                                       playback starts on the first piece
   ▼
Admin dashboard — live table, charts, and an instant escalation alert
```

### Why these specific technical choices

- **AssemblyAI's whisper-rt streaming model, not the flagship real-time model**
  - The flagship real-time model natively handles only 18 languages — Urdu and Punjabi aren't among them
  - whisper-rt covers 99+ languages, including both
  - Without this choice, the project couldn't work in the languages its actual users speak

- **Router, extraction, triage, patient matching and the reply combined into one Gemini call**
  - Fits inside a free API tier's rate limits, and each extra sequential call adds latency
  - Gemini 3.x "thinking" is set to the minimum: these tasks don't need it, and it added seconds per call
  - If the combined answer can't be parsed, the code falls back to separate router / extraction / triage calls

- **Verification never guesses**
  - The extraction prompt returns `null` for anything unclear instead of inventing a plausible value
  - The follow-up-question flow exists specifically to keep a wrong guess out of a health record

- **Structured conversation state, not just concatenated text**
  - The structured extraction is carried forward between turns
  - A backfill step (`_merge_extracted`) fills any gap in a fresh extraction using what's
    already confirmed — it never overwrites a new, correct value

- **Rules that must be reviewable live in plain Python, not in prompts**
  - Visit-type checklists, the vaccine schedule and the "same symptom 3 visits in a row" rule
    are in `care.py`, so a clinician can read and check them

---

## Conversation logic

`session.py` keeps a small amount of state per connected browser and decides what each new
sentence means:

| Situation | What happens |
|---|---|
| New visit, nothing missing | Visit saved, short Urdu confirmation spoken |
| New visit, something missing (name, symptoms, visit-type item) | One follow-up question; her next sentence is treated as the answer (up to 2 questions per visit) |
| Danger sign | Visit saved and flagged, calm warning spoken; missing details asked for straight after |
| Another sentence within 45 s about the **same** patient | Merged into the same visit record (updated, never duplicated) |
| Another sentence naming a **different** patient | Treated as a new visit |
| Another sentence with **no name** | Asks: *"is this the same patient, or a new one?"* and routes her answer |
| Off-topic sentence | Nothing recorded (`skip`) |

"Same patient" is decided by name — and, because the speech recogniser can write one name in
different scripts from turn to turn (for example *"Buxira"* then *"بشرا"*), also by Gemini
matching both sentences to the same known patient.

---

## Speed: how the reply is kept fast

The time she waits = the speech recogniser deciding she has finished + Gemini + text-to-speech
+ network. What the code does about each part:

| Measure | Effect | Switch |
|---|---|---|
| One combined Gemini call per turn (reply included) | No separate "write the reply" call in the normal case | — |
| Fixed Urdu sentences for routine confirmations and questions | No Gemini call for them, and their audio can be cached; Gemini still writes every escalation and replies for returning patients | `REPLY_TEMPLATES=0` |
| Audio cache + fixed phrases pre-made at startup | Repeated phrases play with no TTS wait | `TTS_CACHE_SIZE`, `TTS_PREWARM` |
| **Streaming audio** | The browser starts playing the first piece while the rest is still being made | `STREAM_AUDIO=0` |
| Continuations analysed in parallel | The two analyses of a follow-up sentence run at the same time instead of one after the other | `PARALLEL_CONTINUATION=0` |
| Visit saved while the reply is prepared | Database write and reply preparation overlap | — |
| Gemini timeout + fast failover + cooldown | A stuck or overloaded model is abandoned after 8 s, a per-minute limit switches straight to the next model, and a failing model is skipped for 60 s | `GEMINI_TIMEOUT_S`, `MODEL_COOLDOWN_S` |
| Hallucination filter | Junk transcripts from silence ("Thank you.", "ご視聴ありがとうございました", …) no longer cost a Gemini call, delay real sentences, or use up the hourly quota guard | — |

### How streaming works

The browser asks for streamed replies by connecting with `?stream=1` (only when it supports
MediaSource playback of mp3). For each reply the server sends a JSON `audio_start`, then the
mp3 in binary pieces as edge-tts produces them, then a JSON `audio_end`. The page appends the
pieces to a MediaSource buffer and plays from the first one. Browsers without MediaSource
(for example older iPhones), the simple test page in `static/`, and any page that doesn't ask
all keep receiving each reply as one complete mp3 — so the frontend and backend can be
deployed in either order.

---

## Patient names: recognition and pronunciation

Names are the hardest part of the conversation: the recogniser may write a name in Latin,
Urdu or Devanagari letters, and the Urdu voice mispronounces a name written in Latin letters.

**Recognition**
- Continuations match the patient by Gemini's cross-script patient match, not only by
  spelling, so repeating the name in another script no longer starts a new visit.
- Optional: `STT_KEYTERMS=1` sends recently seen patient names to AssemblyAI as "keyterms"
  so the recogniser favours those spellings. AssemblyAI documents keyterms for its streaming
  models; test it once with whisper-rt (watch for `[session] STT error` in the logs) before
  relying on it.

**Pronunciation**
- Gemini also returns the name written in Urdu script (`patient_name_urdu`); every spoken
  reply uses that spelling, and a reply that still contains the Latin/Devanagari spelling has
  it swapped for the Urdu one before synthesis.
- The voice speaks slightly slower than edge-tts's default (`TTS_RATE=-10%`) so names come out
  clearly. Use `+0%` for the original speed.

---

## Danger-sign list — status: draft, pending clinical review

The red-flag list lives in `agents.py` (`RED_FLAGS`). It combines an initial team draft with
two additions checked against WHO's IMCI (Integrated Management of Childhood Illness) general
danger signs and a documented gap in Pakistan's LHW training around recognising
pre-eclampsia/eclampsia:

- Very high fever (≈103°F / 39.5°C or higher)
- Difficulty breathing, fast/labored breathing, blue or bluish lips
- Severe or heavy bleeding (including postpartum hemorrhage)
- Unconscious, unresponsive, fainting, or convulsions/seizures
- Low blood pressure (≈90/60 or lower) together with dizziness or weakness
- Unable to drink or breastfeed at all (a core WHO IMCI danger sign)
- High blood pressure in pregnancy with severe headache, swelling, or vision changes
  (possible pre-eclampsia/eclampsia)

In addition, a symptom reported on **three visits in a row** for the same patient is escalated
as "not improving".

**This list has not been reviewed by anyone with clinical training.** It's an intentionally
honest limitation, not an oversight — a wrong threshold here has real consequences, and that
call shouldn't be made by an LLM or by non-clinicians alone. The same applies to the visit
checklists and vaccine schedule in `care.py`.

---

## Repository structure

```
amanat-voice-agent/
├── server.py              # FastAPI: WebSocket (audio in, AssemblyAI, replies out), admin API, hallucination filter
├── session.py             # Per-connection conversation state machine (continuity, follow-ups, merge, streaming)
├── agents.py              # Gemini calls: combined agent, fallbacks, verification, reply, fixed Urdu replies, failover
├── care.py                # Visit types, checklists, vaccine schedule, symptom-history rules
├── db.py                  # SQLite persistence: patients, visits, analytics for the dashboard
├── web_tts.py             # Urdu text-to-speech (edge-tts): complete or streamed mp3, audio cache
├── timing.py              # Optional [timing] latency logs (TIMING_LOGS=1)
├── requirements.txt
├── Procfile               # Railway start command
├── .env.example           # Environment variables (placeholders only)
├── DEPLOY.md              # Step-by-step Railway + GitHub Pages deployment guide
├── static/                # Minimal testing interface (served by the backend itself)
│   ├── index.html
│   ├── app.js
│   └── recorder-worklet.js
├── private/
│   └── admin.html         # Password-protected dashboard (visits table, charts, escalation alerts)
└── docs/
    ├── index.html         # The public demo page (GitHub Pages)
    └── og-image.png       # Link-preview image
```

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| Speech-to-text | AssemblyAI real-time STT (`whisper-rt`) | Real-time model covering Urdu and Punjabi |
| Language understanding | Google Gemini (`gemini-3.5-flash-lite`, with fallback models) | Free-tier friendly, fast enough for a conversational loop |
| Text-to-speech | `edge-tts` (`ur-PK-UzmaNeural`), streamed | Free, native Urdu-Pakistan neural voice |
| Backend | FastAPI + WebSockets on Railway | Real-time bidirectional audio/text |
| Storage | SQLite on a Railway volume | Zero setup, enough for a hackathon's data |
| Frontend | Vanilla JS, Web Audio API (AudioWorklet), MediaSource, on GitHub Pages | No framework or build step |
| Admin dashboard | Chart.js (CDN) | Lightweight charts, no build step |

---

## Running it locally

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your own API keys
uvicorn server:app --reload
```

Open `http://localhost:8000` in Chrome (it needs real browser APIs — microphone and
AudioWorklet — so it can't be tested with curl). To test the main demo page locally,
temporarily point `BACKEND_WS_URL` in `docs/index.html` at `ws://localhost:8000/ws`, and
change it back before deploying. Add `TIMING_LOGS=1` to `.env` to see where each turn's time
goes.

## Environment variables

Only the first four are required. Everything else has a safe default.

| Variable | Default | Purpose |
|---|---|---|
| `ASSEMBLYAI_API_KEY` | — (required) | Real-time speech-to-text |
| `GEMINI_API_KEY` | — (required) | Extraction, triage, verification, replies |
| `ADMIN_USER` / `ADMIN_PASS` | — (required) | Protect `/admin` — the dashboard shows patient data |
| `GEMINI_MODELS` | built-in list | Comma-separated models, tried in order. **List at least two** so failover has somewhere to go |
| `DB_PATH` | `/data/amanat.db` if a volume is mounted | Where the SQLite file lives |
| `MAX_PIPELINE_CALLS_PER_HOUR` | `150` | Circuit breaker against strangers draining the Gemini quota |
| `TIMING_LOGS` | off | `1` prints `[timing]` latency lines for every turn |
| `GEMINI_TIMEOUT_S` | `8` | Per-request Gemini timeout before trying the next model |
| `MODEL_COOLDOWN_S` | `60` | How long a timed-out / overloaded model is skipped (`0` = off) |
| `REPLY_TEMPLATES` | on | `0` = Gemini writes every reply instead of fixed Urdu sentences |
| `PARALLEL_CONTINUATION` | on | `0` = run the two continuation analyses one after the other |
| `STREAM_AUDIO` | on | `0` = always send each reply as one complete mp3 |
| `TTS_RATE` | `-10%` | Voice speed; slower makes names clearer, `+0%` is edge-tts's default |
| `TTS_CACHE_SIZE` / `TTS_PREWARM` | `128` / on | Reply-audio cache size (`0` = off); pre-make fixed phrases at startup |
| `STT_KEYTERMS` | off | `1` = boost recently seen patient names in speech recognition (test first) |
| `STT_MIN_TURN_SILENCE_MS` / `STT_MAX_TURN_SILENCE_MS` | AssemblyAI defaults | End-of-turn silence; lower = faster replies but more risk of cutting her off mid-sentence |
| `PIPELINE_WORKERS` | `8` | Worker threads for overlapping steps |

## Deploying

The backend runs on Railway (auto-deploys from `main`); the demo page is served by GitHub
Pages from `docs/`. Full steps, including the WebSocket URL wiring and the persistent volume
for the database, are in [`DEPLOY.md`](./DEPLOY.md).

Streaming is opt-in per browser, so the backend and the demo page can be deployed in either
order without breaking the live demo.

---

## Monitoring and troubleshooting

Set `TIMING_LOGS=1` on Railway, then filter the logs on `[timing]`. Each turn is numbered and
every line shows the time since her sentence was finalised:

```
[timing] turn=3 +0.00s stt.end_of_turn silence_after_last_partial=0.91s formatted=False language=ur chars=42
[timing] turn=3 +1.21s llm 1.20s purpose=combined model=gemini-3.5-flash-lite result=ok prompt_chars=8123
[timing] turn=3 +1.23s db.save_visit 0.01s
[timing] turn=3 +1.55s turn.first_audio_sent 1.55s
[timing] turn=3 +2.10s tts.stream 0.86s cache=miss first_chunk=0.31s
[timing] turn=3 +2.12s turn.total 2.12s audio_replies=1
```

| What you see | What it means |
|---|---|
| `turn.first_audio_sent` | The wait she actually hears — the number to keep low |
| `llm … result=timeout / unavailable / per_minute_limit` | Provider trouble; failover should move to the next model |
| Two or more `llm … purpose=combined` in one turn | A continuation (normal), or the fallback path |
| `pipeline.fallback_to_separate_agents` | Gemini's combined answer couldn't be parsed — should be rare |
| Large `silence_after_last_partial` every turn | The end-of-turn wait is long; consider `STT_MAX_TURN_SILENCE_MS` |
| `[HEARD] … (looks like a hallucinated filler phrase - ignored)` | Junk transcript from silence, dropped before any Gemini call |
| `hourly quota guard triggered` | `MAX_PIPELINE_CALLS_PER_HOUR` reached; real sentences get no reply until the hour rolls over |

---

## WebSocket message reference

**Browser → server:** binary 16 kHz PCM16 microphone audio. Connect to `/ws`. Optional query
parameters: `stream=1` to receive streamed replies, and `device=<random id>` to tag the visits
with this browser (see *HTTP endpoints* below).

**Server → browser (JSON):**

| `type` | Meaning |
|---|---|
| `status` | Pipeline step: `understanding` / `replying` / `listening` |
| `partial` | Live transcript while she's speaking |
| `final` | Her finished sentence |
| `record` | A saved visit: extracted details, triage, visit type, visit number, last seen, vaccine note |
| `skip` | Off-topic — nothing recorded |
| `quota_exceeded` | Hourly quota guard reached |
| `audio_start` / `audio_end` | Start / end of a streamed reply (streaming clients only) |

**Server → browser (binary):** the spoken reply — a complete mp3, or, between `audio_start`
and `audio_end`, one piece of a streamed mp3.

### HTTP endpoints

| Endpoint | Access | Returns |
|---|---|---|
| `GET /admin` | admin password | The dashboard page |
| `GET /api/visits` | admin password | All visits + totals (the dashboard polls it every 5 s) |
| `GET /api/analytics` | admin password | Visits per day, top symptoms, escalation rate |
| `GET /api/my-visits?device=<id>` | public | Only the visits recorded from that browser, so the demo page can show them again after a reload |

---

## Known limitations

- **Red-flag thresholds, checklists and the vaccine schedule are drafts** — they need sign-off
  from someone with medical training before this could be trusted beyond a hackathon demo.
- **The speech recogniser sometimes writes Urdu in Hindi (Devanagari) script** or garbles a
  name; Gemini understands both scripts, and name matching works across scripts, but a badly
  garbled name may still need repeating.
- **Whisper "hallucinates" text from silence or background noise.** The filter drops the
  common cases; test in a quiet room with the microphone close.
- **Streaming playback needs MediaSource.** Older iPhones get complete replies instead —
  slightly later, but otherwise identical.
- **Visits are only kept if Railway has a persistent volume mounted at `/data`.** Without it,
  every redeploy or restart empties the admin dashboard. The startup log says which case
  you're in (`[db] using /data/amanat.db` vs `[db] WARNING: … NOT on a persistent volume`);
  setup steps are in `DEPLOY.md`.
- **This is a hackathon prototype handling patient-shaped data.** No authentication beyond the
  admin password, no encryption at rest beyond Railway's defaults, and no consent flow. Test
  with fictional names, never real patients.

---

## Team

Built for the AssemblyAI Voice Agent Hackathon on lablab.ai.

| Member | Role |
|---|---|
| **Maryum Akram** | Backend & frontend |
| **Sehrish Riaz** | Voice agent |
| **Annum Nisar** | Voice agent |
| **Shabana** | Testing |
| **Hina** | Demo video & slides |
