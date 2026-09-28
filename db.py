import json
import sqlite3
import time
from pathlib import Path

# Railway wipes the container's own filesystem on every redeploy. If a persistent
# volume is mounted at /data (see DEPLOY.md), use it so visit history survives
# deploys; otherwise fall back to a local file next to this script for local dev.
_default_path = "/data/amanat.db" if Path("/data").exists() else str(Path(__file__).parent / "amanat.db")
import os
DB_PATH = Path(os.environ.get("DB_PATH", _default_path))


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at REAL NOT NULL,
            patient_name TEXT,
            symptoms TEXT,
            vitals_temp TEXT,
            vitals_bp TEXT,
            notes TEXT,
            escalate INTEGER NOT NULL,
            reason TEXT
        )
    """)
    # A patient is someone seen on one or more visits. Names arrive in whatever script the
    # speech-to-text used (Latin, Urdu, even Devanagari), so the LLM does the matching (see
    # agents.combined_agent) and visits are linked to a patient by id, not by name text.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)
    # Columns added after the first release. The volume on Railway keeps the old table,
    # so add them in place instead of recreating it (that would delete visit history).
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(visits)")}
    for column, kind in (("patient_id", "INTEGER"), ("visit_type", "TEXT"), ("details", "TEXT"),
                         ("device_id", "TEXT")):
        if column not in existing:
            conn.execute(f"ALTER TABLE visits ADD COLUMN {column} {kind}")
    _backfill_patients(conn)
    conn.commit()
    total = conn.execute("SELECT COUNT(*) AS c FROM visits").fetchone()["c"]
    conn.close()
    # Printed at every start so the Railway logs show whether visits will survive a redeploy.
    if str(DB_PATH).startswith("/data/") or os.environ.get("DB_PATH"):
        print(f"[db] using {DB_PATH} ({total} visits stored)")
    else:
        print(f"[db] WARNING: using {DB_PATH}, which is NOT on a persistent volume - all "
              f"{total} visits will be lost on the next redeploy or restart. See DEPLOY.md.")


def _backfill_patients(conn):
    """Give visits recorded before the patients table existed a patient, grouping by
    exact (case-insensitive) name. Cross-script duplicates stay separate; new visits get
    linked properly by the LLM from here on."""
    rows = conn.execute(
        "SELECT id, patient_name FROM visits WHERE patient_id IS NULL AND patient_name IS NOT NULL "
        "ORDER BY recorded_at"
    ).fetchall()
    by_name = {}
    for row in rows:
        key = row["patient_name"].strip().lower()
        if key not in by_name:
            cur = conn.execute(
                "INSERT INTO patients (name, created_at) VALUES (?, ?)", (row["patient_name"].strip(), time.time())
            )
            by_name[key] = cur.lastrowid
        conn.execute("UPDATE visits SET patient_id = ? WHERE id = ?", (by_name[key], row["id"]))


def save_visit(extracted: dict, triage: dict, patient_id=None, visit_id=None, device_id=None):
    """Save one visit, or update visit_id when the LHW is still adding to the same visit
    (a follow-up answer or an extra sentence), so one visit never becomes several rows.
    patient_id is an existing patient the LLM matched this visit to (validated here);
    otherwise the visit keeps its earlier patient, or a new patient is created when a
    name is known. device_id is the browser that recorded it (see get_device_visits).
    Returns (visit_id, patient_id)."""
    conn = _connect()
    name = extracted.get("patient_name")
    if patient_id is not None:
        if not conn.execute("SELECT 1 FROM patients WHERE id = ?", (patient_id,)).fetchone():
            patient_id = None  # the model invented an id; don't link to it
    if visit_id is not None:
        row = conn.execute("SELECT patient_id FROM visits WHERE id = ?", (visit_id,)).fetchone()
        if row is None:
            visit_id = None
        elif patient_id is None:
            patient_id = row["patient_id"]
    if patient_id is None and name:
        patient_id = conn.execute(
            "INSERT INTO patients (name, created_at) VALUES (?, ?)", (name, time.time())
        ).lastrowid
    values = (
        name,
        json.dumps(extracted.get("symptoms", []), ensure_ascii=False),
        (extracted.get("vitals") or {}).get("temp"),
        (extracted.get("vitals") or {}).get("bp"),
        extracted.get("notes", ""),
        1 if triage.get("escalate") else 0,
        triage.get("reason", ""),
        patient_id,
        extracted.get("visit_type") or "general",
        json.dumps(extracted.get("details") or {}, ensure_ascii=False),
    )
    if visit_id is not None:
        conn.execute(
            """UPDATE visits SET patient_name = ?, symptoms = ?, vitals_temp = ?, vitals_bp = ?,
               notes = ?, escalate = ?, reason = ?, patient_id = ?, visit_type = ?, details = ?,
               device_id = COALESCE(device_id, ?)
               WHERE id = ?""",
            values + (device_id, visit_id),
        )
    else:
        visit_id = conn.execute(
            """INSERT INTO visits
               (patient_name, symptoms, vitals_temp, vitals_bp, notes, escalate, reason,
                patient_id, visit_type, details, recorded_at, device_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            values + (time.time(), device_id),
        ).lastrowid
    conn.commit()
    conn.close()
    return visit_id, patient_id


def _visit_summary(row) -> dict:
    try:
        symptoms = json.loads(row["symptoms"] or "[]")
    except (json.JSONDecodeError, TypeError):
        symptoms = []
    try:
        details = json.loads(row["details"] or "{}")
    except (json.JSONDecodeError, TypeError):
        details = {}
    return {
        "recorded_at": row["recorded_at"],
        "visit_type": row["visit_type"] or "general",
        "symptoms": symptoms,
        "temp": row["vitals_temp"],
        "bp": row["vitals_bp"],
        "notes": row["notes"] or "",
        "details": details,
        "escalated": bool(row["escalate"]),
    }


def recent_patients(days: int = 60, limit: int = 40, exclude_visit_id=None):
    """Patients seen recently, newest first, each with their latest visit and visit count.
    Handed to the LLM so it can tell whether a spoken name is someone already known.
    exclude_visit_id leaves out the visit currently being recorded."""
    since = time.time() - days * 86400
    exclude = -1 if exclude_visit_id is None else exclude_visit_id
    conn = _connect()
    rows = conn.execute(
        """SELECT p.id AS patient_id, p.name, v.*, counts.n AS visit_count
           FROM patients p
           JOIN (SELECT patient_id, MAX(recorded_at) AS last_at, COUNT(*) AS n
                 FROM visits WHERE patient_id IS NOT NULL AND id != ? GROUP BY patient_id) counts
             ON counts.patient_id = p.id
           JOIN visits v ON v.patient_id = p.id AND v.recorded_at = counts.last_at AND v.id != ?
           WHERE counts.last_at >= ?
           ORDER BY counts.last_at DESC
           LIMIT ?""",
        (exclude, exclude, since, limit),
    ).fetchall()
    conn.close()
    return [
        {"id": r["patient_id"], "name": r["name"], "visit_count": r["visit_count"], "last": _visit_summary(r)}
        for r in rows
    ]


def patient_history(patient_id, limit: int = 5, exclude_visit_id=None):
    """A patient's most recent visits, newest first (leaving out the visit being recorded)."""
    exclude = -1 if exclude_visit_id is None else exclude_visit_id
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM visits WHERE patient_id = ? AND id != ? ORDER BY recorded_at DESC LIMIT ?",
        (patient_id, exclude, limit),
    ).fetchall()
    conn.close()
    return [_visit_summary(r) for r in rows]


def get_device_visits(device_id: str, limit: int = 30):
    """Visits recorded from one browser, oldest first, so the demo page can show them again
    after a reload. Only this browser knows its random device_id, so it only ever sees its
    own visits - never everyone's (that's the password-protected admin page)."""
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM visits WHERE device_id = ? ORDER BY recorded_at DESC LIMIT ?", (device_id, limit)
    ).fetchall()
    conn.close()
    visits = []
    for r in reversed(rows):
        summary = _visit_summary(r)
        visits.append({
            "id": r["id"],
            "recorded_at": summary["recorded_at"],
            "patient_name": r["patient_name"],
            "symptoms": summary["symptoms"],
            "temp": summary["temp"],
            "bp": summary["bp"],
            "visit_type": summary["visit_type"],
            "escalated": summary["escalated"],
            "reason": r["reason"] or "",
        })
    return visits


def get_visits(limit: int = 200):
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM visits ORDER BY recorded_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    conn = _connect()
    total = conn.execute("SELECT COUNT(*) AS c FROM visits").fetchone()["c"]
    escalated = conn.execute(
        "SELECT COUNT(*) AS c FROM visits WHERE escalate = 1"
    ).fetchone()["c"]
    conn.close()
    return {"total": total, "escalated": escalated}


def get_analytics():
    """Aggregates for the admin dashboard's charts. Symptoms are stored as a JSON list
    per row, so counting how often each one occurs is done in Python rather than SQL."""
    conn = _connect()
    rows = conn.execute("SELECT recorded_at, symptoms, escalate FROM visits").fetchall()
    conn.close()

    per_day = {}
    symptom_counts = {}
    escalated = 0
    for row in rows:
        day = time.strftime("%Y-%m-%d", time.localtime(row["recorded_at"]))
        per_day[day] = per_day.get(day, 0) + 1
        if row["escalate"]:
            escalated += 1
        try:
            symptoms = json.loads(row["symptoms"] or "[]")
        except (json.JSONDecodeError, TypeError):
            symptoms = []
        for s in symptoms:
            key = s.strip().lower()
            if key:
                symptom_counts[key] = symptom_counts.get(key, 0) + 1

    visits_per_day = [{"day": d, "count": c} for d, c in sorted(per_day.items())]
    top_symptoms = sorted(symptom_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
    total = len(rows)
    escalation_rate = round((escalated / total) * 100, 1) if total else 0.0

    return {
        "visits_per_day": visits_per_day,
        "top_symptoms": [{"symptom": s, "count": c} for s, c in top_symptoms],
        "escalation_rate": escalation_rate,
        "total": total,
        "escalated": escalated,
    }


init_db()
