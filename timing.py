"""Cheap latency logging for the voice pipeline.

Off by default. Set TIMING_LOGS=1 (Railway: service -> Variables) to turn it on. Every line
starts with "[timing]" so the Railway log search can filter them, e.g.:

  [timing] turn=3 +0.00s stt.end_of_turn silence_after_last_partial=0.91s formatted=False
  [timing] turn=3 +1.84s llm 1.83s purpose=combined model=gemini-3.5-flash-lite ok prompt_chars=8123
  [timing] turn=3 +2.61s tts 0.74s chars=42 bytes=19584 cache=miss
  [timing] turn=3 +2.63s send.audio 0.01s bytes=19584
  [timing] turn=3 +2.63s turn.first_audio_sent 2.63s
  [timing] turn=3 +2.63s turn.total 2.63s audio_replies=1

"+N.NNs" is the time since that turn's final transcript arrived from AssemblyAI.
"""
import itertools
import os
import threading
import time

ENABLED = os.environ.get("TIMING_LOGS", "0").strip().lower() in ("1", "true", "yes", "on")

_turn_ids = itertools.count(1)
_local = threading.local()  # each browser session's pipeline runs on its own STT thread


def turn_start():
    """Mark the moment a final transcript arrived. Returns the turn id."""
    _local.turn = next(_turn_ids)
    _local.start = time.perf_counter()
    _local.audio_sent = 0
    return _local.turn


def _prefix():
    turn = getattr(_local, "turn", None)
    if turn is None:
        return "[timing]"
    return f"[timing] turn={turn} +{time.perf_counter() - _local.start:.2f}s"


def log(stage, seconds=None, **fields):
    if not ENABLED:
        return
    parts = [_prefix(), stage]
    if seconds is not None:
        parts.append(f"{seconds:.2f}s")
    parts.extend(f"{k}={v}" if v is not True else k for k, v in fields.items() if v is not None)
    print(" ".join(parts), flush=True)


class timed:
    """with timing.timed("db.save_visit"): ...   -> logs how long the block took."""

    def __init__(self, stage, **fields):
        self.stage, self.fields = stage, fields

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.fields["error"] = exc_type.__name__
        log(self.stage, time.perf_counter() - self.t0, **self.fields)
        return False


def audio_sent():
    """Call right after an audio reply reached the browser socket."""
    if getattr(_local, "turn", None) is None:
        return
    _local.audio_sent += 1
    if _local.audio_sent == 1:
        log("turn.first_audio_sent", time.perf_counter() - _local.start)


def turn_end():
    """Call when the pipeline is done with this turn (whatever the outcome)."""
    if getattr(_local, "turn", None) is None:
        return
    log("turn.total", time.perf_counter() - _local.start, audio_replies=_local.audio_sent)
    _local.turn = None


def run_in_turn(fn, *args, **kwargs):
    """Wrap fn so that, when run on a worker thread, its timing lines still carry this
    turn's id and clock. Use for work handed to a thread pool mid-turn."""
    turn = getattr(_local, "turn", None)
    start = getattr(_local, "start", None)

    def wrapped():
        if turn is not None:
            _local.turn, _local.start, _local.audio_sent = turn, start, 0
        try:
            return fn(*args, **kwargs)
        finally:
            _local.turn = None

    return wrapped
