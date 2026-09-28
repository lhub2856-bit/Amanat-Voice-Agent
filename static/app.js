// Real backend URL — update if you ever redeploy Railway to a new domain.
const BACKEND_WS_URL = "wss://web-production-11fd82.up.railway.app/ws";

const ECHO_TAIL_MS = 600; // matches ECHO_TAIL_SECONDS in the original tts.py

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");

let ws, audioCtx, workletNode, mediaStream;
let muted = false; // true while Amanat's reply is playing, so we don't stream her own voice back

// One reused <audio> element, unlocked during the Start click (a real user gesture) so
// iOS Safari allows later programmatic play() calls triggered from WebSocket messages.
// A queue stops two replies from ever overlapping.
const player = new Audio();
let audioQueue = [];
let isPlayingAudio = false;

function unlockAudioForSession() {
  player.src = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=";
  player.play().catch(() => {});
}

function enqueueReply(blob) {
  audioQueue.push(blob);
  playNextQueued();
}

function playNextQueued() {
  if (isPlayingAudio || audioQueue.length === 0) return;
  isPlayingAudio = true;
  muted = true;
  const blob = audioQueue.shift();
  const url = URL.createObjectURL(blob);
  player.src = url;
  player.onended = () => {
    URL.revokeObjectURL(url);
    isPlayingAudio = false;
    setTimeout(() => {
      muted = false;
      playNextQueued();
    }, ECHO_TAIL_MS);
  };
  player.play().catch((err) => console.error("[amanat] playback blocked:", err));
}

function log(text) {
  const div = document.createElement("div");
  div.textContent = text;
  logEl.prepend(div);
}

async function start() {
  unlockAudioForSession(); // must run synchronously here, before any await
  statusEl.textContent = "Connecting...";
  ws = new WebSocket(BACKEND_WS_URL);
  ws.binaryType = "arraybuffer";

  ws.onopen = async () => {
    statusEl.textContent = "Connected — listening...";
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      statusEl.textContent = "Microphone permission denied.";
      return;
    }
    audioCtx = new AudioContext();
    await audioCtx.audioWorklet.addModule("recorder-worklet.js");
    const source = audioCtx.createMediaStreamSource(mediaStream);
    workletNode = new AudioWorkletNode(audioCtx, "recorder-processor", {
      processorOptions: { targetSampleRate: 16000 },
    });
    workletNode.port.onmessage = (event) => {
      if (!muted && ws.readyState === WebSocket.OPEN) {
        ws.send(event.data);
      }
    };
    source.connect(workletNode);
  };

  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      const msg = JSON.parse(event.data);
      if (msg.type === "status") {
        log(`[pipeline] ${msg.step}`);
      } else if (msg.type === "partial") {
        log(`(hearing) ${msg.text}`);
      } else if (msg.type === "final") {
        log(`(heard) ${msg.text}`);
      } else if (msg.type === "record") {
        const escalated = msg.triage.escalate ? " — ESCALATED (supervisor notified, simulated)" : "";
        log(`Recorded: ${msg.extracted.patient_name || "(no name given)"}${escalated}`);
        log(JSON.stringify(msg, null, 2));
      } else if (msg.type === "skip") {
        log("(off-topic — nothing recorded)");
      }
      return;
    }
    // Binary message = Amanat's spoken reply (mp3 bytes)
    const blob = new Blob([event.data], { type: "audio/mpeg" });
    enqueueReply(blob);
  };

  ws.onclose = () => {
    statusEl.textContent = "Disconnected";
    startBtn.disabled = false;
    stopBtn.disabled = true;
  };

  ws.onerror = () => {
    statusEl.textContent = "Connection error — check the backend URL.";
  };

  startBtn.disabled = true;
  stopBtn.disabled = false;
}

function stop() {
  if (workletNode) workletNode.disconnect();
  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
  if (audioCtx) audioCtx.close();
  if (ws) ws.close();
  statusEl.textContent = "Stopped";
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);
