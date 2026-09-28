"""Visit-type knowledge for Amanat: what an LHW should capture on each kind of visit, the
child vaccine schedule, and simple checks across a patient's past visits.

Kept as plain Python (not prompt text) so the rules are easy to read, test, and get reviewed.
DRAFT, like RED_FLAGS in agents.py: have an LHW supervisor or clinician check these before
relying on them.
"""
import re
import time

VISIT_TYPES = ("general", "pregnancy", "postnatal", "child", "family_planning")

VISIT_TYPE_RULES = """- "pregnancy": the patient is currently pregnant
- "postnatal": the mother within about 6 weeks after delivery
- "child": a baby or child under 5 is the patient (newborn checks, vaccines, feeding, growth)
- "family_planning": contraception / birth spacing is the main topic
- "general": anything else"""

# Type-specific details. Every key is optional: null when not said. Numbers are plain numbers.
DETAIL_FIELDS = {
    "pregnancy_month": "number, month of pregnancy (1-9)",
    "headache": "true/false if she says whether there is a severe headache, else null",
    "swelling": "true/false for swelling of face/hands/feet, else null",
    "vision_changes": "true/false for blurred vision / seeing spots, else null",
    "bleeding": "true/false for vaginal bleeding (pregnancy or after delivery), else null",
    "days_since_delivery": "number of days since delivery",
    "child_age_weeks": "number, the child's age converted to weeks (e.g. 2 months -> 8.7, 1 year -> 52)",
    "weight": "weight as stated, e.g. \"6 kg\"",
    "feeding": "short English note on breastfeeding/feeding, e.g. \"breastfeeding well\"",
    "vaccines_given": "list of vaccines she says were given on this visit, short English names",
    "fp_method": "family planning method, short English, e.g. \"pills\", \"injection\", \"IUD\"",
}

DETAIL_RULES = "\n".join(f'  - {key}: {desc}' for key, desc in DETAIL_FIELDS.items())

# What to ask for when it's missing, per visit type, in priority order.
# (key, English description used in prompts)
CHECKLISTS = {
    "pregnancy": [
        ("pregnancy_month", "which month of pregnancy she is in"),
        ("bp", "her blood pressure"),
    ],
    "postnatal": [
        ("days_since_delivery", "how many days ago she delivered"),
        ("bleeding", "whether she has heavy bleeding"),
    ],
    "child": [
        ("child_age_weeks", "the child's age"),
    ],
    "family_planning": [
        ("fp_method", "which family planning method she is using"),
    ],
    "general": [],
}

PRE_ECLAMPSIA_CHECK = "whether she has a severe headache, swelling of face or hands, or blurred vision"


def checklist_rules_text():
    """The same checklist, written out for the LLM prompt so its follow-up question asks for
    the same thing find_care_gaps() will flag."""
    lines = []
    for visit_type, items in CHECKLISTS.items():
        if items:
            lines.append(f'  - {visit_type}: ' + "; ".join(desc for _, desc in items))
    lines.append(f"  - pregnancy with high blood pressure (140/90 or more, or described as high): {PRE_ECLAMPSIA_CHECK}")
    return "\n".join(lines)


def normalise_details(data):
    """Keep only known keys, with sane types."""
    out = {}
    if not isinstance(data, dict):
        return out
    for key in DETAIL_FIELDS:
        value = data.get(key)
        if value in (None, "", []):
            continue
        if key in ("pregnancy_month", "days_since_delivery", "child_age_weeks"):
            try:
                value = round(float(value), 1)
            except (TypeError, ValueError):
                continue
            if value < 0:
                continue
        elif key in ("headache", "swelling", "vision_changes", "bleeding"):
            if not isinstance(value, bool):
                continue
        elif key == "vaccines_given":
            if not isinstance(value, list):
                continue
            value = [str(v).strip() for v in value if str(v).strip()]
            if not value:
                continue
        else:
            value = str(value).strip()
        out[key] = value
    return out


def bp_is_high(bp):
    if not bp:
        return False
    match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", str(bp))
    if match:
        systolic, diastolic = int(match.group(1)), int(match.group(2))
        return systolic >= 140 or diastolic >= 90
    return "high" in str(bp).lower()


def find_care_gaps(extracted):
    """Visit-type items still missing, highest priority first (English descriptions)."""
    visit_type = extracted.get("visit_type") or "general"
    details = extracted.get("details") or {}
    vitals = extracted.get("vitals") or {}
    gaps = []
    for key, desc in CHECKLISTS.get(visit_type, []):
        present = vitals.get("bp") if key == "bp" else details.get(key) is not None
        if not present:
            gaps.append(desc)
    if visit_type == "pregnancy" and bp_is_high(vitals.get("bp")):
        if all(details.get(k) is None for k in ("headache", "swelling", "vision_changes")):
            gaps.append(PRE_ECLAMPSIA_CHECK)
    return gaps


# ---------------------------------------------------------------- child vaccines
# Pakistan EPI routine schedule (DRAFT - verify against the current EPI card before relying on it).
# (age in weeks, Urdu label for the age, vaccines in Urdu script so the TTS voice says them right)
EPI_SCHEDULE = [
    (0, "پیدائش", "بی سی جی، پولیو کے قطرے اور ہیپاٹائٹس بی"),
    (6, "چھ ہفتے", "پینٹا، نمونیا، روٹا اور پولیو کے قطرے"),
    (10, "دس ہفتے", "پینٹا، نمونیا، روٹا اور پولیو کے قطرے"),
    (14, "چودہ ہفتے", "پینٹا، نمونیا، پولیو کے قطرے اور آئی پی وی"),
    (39, "نو مہینے", "خسرہ روبیلا، ٹائیفائیڈ اور آئی پی وی"),
    (65, "پندرہ مہینے", "خسرہ روبیلا"),
]
DUE_WINDOW_WEEKS = 4   # a dose is "due now" for this long after its scheduled age
UPCOMING_WEEKS = 2     # mention the next dose when it's this close


def vaccine_reminder(age_weeks):
    """One Urdu sentence about which vaccines are due now or coming up, or "" if none.
    Returns (sentence, english_summary) so the UI and admin page can show it too."""
    if age_weeks is None:
        return "", ""
    due = [m for m in EPI_SCHEDULE if m[0] <= age_weeks < m[0] + DUE_WINDOW_WEEKS]
    if due:
        weeks, label, vaccines = due[-1]
        return (
            f"یاد رکھیں، {label} پر بچے کو {vaccines} لگنے ہیں، اگر ابھی تک نہیں لگے تو ضرور لگوائیں۔",
            f"{weeks}-week vaccines due",
        )
    upcoming = [m for m in EPI_SCHEDULE if 0 < m[0] - age_weeks <= UPCOMING_WEEKS]
    if upcoming:
        weeks, label, vaccines = upcoming[0]
        return (
            f"اگلے ہفتوں میں، {label} پر بچے کو {vaccines} لگنے ہیں۔",
            f"{weeks}-week vaccines coming up",
        )
    return "", ""


# ---------------------------------------------------------------- history checks
PERSISTENT_VISITS = 3  # same symptom on this many visits in a row -> escalate


def persistent_symptoms(current_symptoms, history):
    """Symptoms reported on this visit AND each of the previous (PERSISTENT_VISITS - 1)
    visits. history is newest first, from db.patient_history()."""
    previous = history[: PERSISTENT_VISITS - 1]
    if len(previous) < PERSISTENT_VISITS - 1:
        return []
    current = {s.strip().lower() for s in current_symptoms or []}
    for visit in previous:
        current &= {s.strip().lower() for s in visit.get("symptoms") or []}
    return sorted(current)


def days_ago(timestamp):
    days = int((time.time() - timestamp) // 86400)
    if days <= 0:
        return "today"
    return "1 day ago" if days == 1 else f"{days} days ago"


def patient_line(patient):
    """One compact line describing a known patient, for the LLM prompt."""
    last = patient["last"]
    parts = [f'id={patient["id"]}', f'name: {patient["name"]}', f'{patient["visit_count"]} visit(s)',
             f'last seen {days_ago(last["recorded_at"])}', f'type: {last["visit_type"]}']
    if last["symptoms"]:
        parts.append("symptoms: " + ", ".join(last["symptoms"]))
    vitals = ", ".join(v for v in (last["temp"] and f'temp {last["temp"]}', last["bp"] and f'bp {last["bp"]}') if v)
    if vitals:
        parts.append(vitals)
    if last["details"]:
        parts.append("details: " + ", ".join(f"{k}={v}" for k, v in last["details"].items()))
    if last["escalated"]:
        parts.append("was escalated")
    return " | ".join(parts)
