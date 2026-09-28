# Deploying Amanat Health as a web app

This folder turns your existing `agents.py` pipeline into a browser-based voice agent:
browser mic -> WebSocket -> this backend -> AssemblyAI -> Gemini agents -> edge-tts ->
WebSocket -> browser speaker. `agents.py` itself is untouched from your original build.

## 1. Test locally first

```
pip install -r requirements.txt
cp .env.example .env   # fill in your real keys
uvicorn server:app --reload
```

Open `http://localhost:8000` in Chrome (needs a real browser, not curl — it uses the
microphone and AudioWorklet APIs). Click Start, allow mic access, and speak a test
sentence from your `amanat_voice_tests.xlsx` sheet. You should hear Amanat Health reply.

Note: `static/app.js` currently points `BACKEND_WS_URL` at a placeholder Railway URL.
For local testing, temporarily change it to `ws://localhost:8000/ws` — just remember to
change it back before deploying.

## 2. Deploy the backend to Railway

1. Push this whole `amanat-web/` folder to a GitHub repo (a fresh repo, or a subfolder
   of your existing one — if it's a subfolder, set Railway's "Root Directory" to it).
2. In Railway: **New Project -> Deploy from GitHub repo** -> select the repo.
3. Railway auto-detects Python and reads `Procfile` for the start command. Nothing else
   to configure there.
4. Go to the service's **Variables** tab and add:
   - `ASSEMBLYAI_API_KEY`
   - `GEMINI_API_KEY`
   - `GEMINI_MODELS` (optional, same value as your local `.env`)
   - `ADMIN_USER` and `ADMIN_PASS` — your own choice, protects the `/admin` dashboard
5. Go to **Settings -> Networking -> Generate Domain**. This gives you a public URL like
   `https://amanat-production.up.railway.app`. Your WebSocket URL is the same thing with
   `wss://` instead of `https://` and `/ws` on the end:
   `wss://amanat-production.up.railway.app/ws`

## 3.5. The admin dashboard

Visit `https://your-app.up.railway.app/admin` (your browser will prompt for the
username/password you set above). It shows every visit recorded during testing or a
demo — total count, how many were escalated, and a table with each patient, symptoms,
vitals, and status. It polls `/api/visits` every 5 seconds, so leave it open on a second
screen during a live demo to show it updating in real time.

**Important:** this page shows real patient names and health details entered during
testing. Use fictional names for any demo or recorded video — never real patients' data,
even during internal testing.

### Keep visits after redeploys and restarts (persistent volume)

Visits are stored in a SQLite file. Railway **wipes the container's own disk on every
redeploy and restart**, so without a volume the admin dashboard is empty again after each
deploy. To keep them:

1. Open your Railway project and select the backend service.
2. Add a volume to it (right-click the service on the project canvas → **Attach Volume**, or
   **+ New → Volume** and connect it to the service).
3. Set the volume's **mount path** to `/data`.
4. Optional but recommended: in **Variables**, add `DB_PATH=/data/amanat.db`.
5. Redeploy once. The app notices `/data` and stores `amanat.db` there from now on.

Check it worked: the deploy logs print one of these at startup.

```
[db] using /data/amanat.db (12 visits stored)                         <- good, visits persist
[db] WARNING: using /app/amanat.db, which is NOT on a persistent volume  <- no volume yet
```

Visits recorded before the volume existed can't be recovered, only the ones after it.

### Visits on the demo page after a reload

The public demo page (`docs/index.html`) now lists the visits recorded **from that browser**
when it loads, so a reload no longer empties it. Each browser keeps a random id in its local
storage and only ever gets its own visits back from `/api/my-visits`; all visits together are
only visible on the password-protected `/admin` page. A private/incognito window starts with
a fresh id, and clearing site data forgets it. This also needs the volume above: if the
database was wiped, there's nothing to show.

## 3. Point the frontend at your backend, then deploy to GitHub Pages

1. In `static/app.js`, replace the placeholder:
   ```js
   const BACKEND_WS_URL = "wss://amanat-production.up.railway.app/ws";
   ```
2. Commit and push that change.
3. In your GitHub repo: **Settings -> Pages -> Source**, pick the branch and the
   `/static` folder (or copy `static/`'s contents to a `docs/` folder at the repo root
   first, since GitHub Pages doesn't let you serve from an arbitrary subfolder on all
   plans — `docs/` is the safest choice).
4. Your live demo URL will be `https://yourusername.github.io/reponame`.

## 4. Before the actual demo in front of judges

- Test the full deployed round trip (GitHub Pages page -> Railway backend) at least
  once on the same day you present, since Gemini's free-tier daily quota is shared
  across all your testing that day.
- Test in a quiet room with the mic close, per the known limitation in your README
  about speech-model hallucination on background noise.
- Have a backup: keep `orchestrator.py` (the local text-input version) runnable in a
  terminal as a fallback if venue wifi causes WebSocket issues during the live demo.

## What's reused vs. new

| File | Status |
|---|---|
| `agents.py` | Unchanged from your original build |
| `web_tts.py` | New — same edge-tts call, but returns mp3 bytes instead of playing locally |
| `session.py` | New — same conversation state machine as your `orchestrator.py`, sending over a WebSocket instead of printing/speaking locally |
| `server.py` | New — FastAPI WebSocket endpoint, replaces `stt_stream.py`'s direct PyAudio mic capture |
| `static/*` | New — browser mic capture (AudioWorklet) and playback |

Your original `orchestrator.py`, `stt_stream.py`, and `tts.py` still work locally exactly
as before — this is a parallel web version, not a replacement.
