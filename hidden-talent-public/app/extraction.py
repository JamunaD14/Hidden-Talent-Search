"""
Model calls shared by the app and by prepare_matching.py:
  - extract_job:             job ad (Swedish) -> English requirements + conditions
  - extract_candidate_prefs: candidate's own text -> stated work conditions
  - explain_matches:         short, evidence-based explanations for a recruiter
All calls use structured output, so the model always returns valid JSON.
"""

import json
import re

import numpy as np

from common import CHAT_DEPLOYMENT, EMBEDDING_DEPLOYMENT

SCHEDULE = ["daytime", "evenings", "nights", "weekends", "shifts", "flexible"]
HOURS = ["full-time", "part-time", "either", "not stated"]


def _strings():
    return {"type": "array", "items": {"type": "string"}}


# --- Job requirements -------------------------------------------------------
JOB_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "title_en", "summary_en", "must_have", "nice_to_have", "hours", "schedule",
        "sales_targets", "training_provided", "languages_required", "other_requirements",
    ],
    "properties": {
        "title_en": {"type": "string"},
        "summary_en": {"type": "string"},
        "must_have": _strings(),
        "nice_to_have": _strings(),
        "hours": {"type": "string", "enum": HOURS},
        "schedule": {"type": "array", "items": {"type": "string", "enum": SCHEDULE}},
        "sales_targets": {"type": "boolean"},
        "training_provided": {"type": "boolean"},
        "languages_required": _strings(),
        "other_requirements": _strings(),
    },
}

JOB_PROMPT = """You read a job ad (usually in Swedish) and describe what the employer needs, in English.

Rules:
- title_en: the job title in English.
- summary_en: 2 sentences in English: the role and its main tasks.
- must_have: 3-8 skills or experiences the ad requires, as short standard English skill terms (1-4 words), e.g. "customer service", "cash handling", "stock management", "Python". No brackets or slashes.
- nice_to_have: up to 6 skills the ad lists as a plus (e.g. "meriterande").
- Turn personality words into skills only when the ad describes a task (e.g. "you help colleagues" -> "teamwork"). Do not list adjectives like "positive" on their own.
- hours: "full-time" for heltid/100 %, "part-time" for deltid or under 100 % (e.g. 75 %), "either" if both are possible, otherwise "not stated".
- schedule: every one that applies: daytime, evenings, nights, weekends, shifts, flexible. Empty if not stated.
- sales_targets: true only if the role involves active selling or sales targets.
- training_provided: true if the ad says new staff get training or an introduction.
- languages_required: languages the ad requires (e.g. "Swedish", "English").
- other_requirements: formal requirements such as "driving licence B" or "background check". Empty if none.
- Use only what the ad says."""


# --- Candidate's stated work conditions ------------------------------------
CANDIDATE_PREFS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "hours", "prefers_daytime", "wants_predictable_schedule", "avoids_sales_targets",
        "avoids_weekends", "wants_training", "open_to_sector_change",
    ],
    "properties": {
        "hours": {"type": "string", "enum": HOURS},
        "prefers_daytime": {"type": "boolean"},
        "wants_predictable_schedule": {"type": "boolean"},
        "avoids_sales_targets": {"type": "boolean"},
        "avoids_weekends": {"type": "boolean"},
        "wants_training": {"type": "boolean"},
        "open_to_sector_change": {"type": "boolean"},
    },
}

CANDIDATE_PREFS_PROMPT = """You read a jobseeker's own description and record only the work conditions they state. Anything not stated is false or "not stated".

- hours: "part-time" if they ask for or are open to part-time only, "full-time" if they ask for full-time, "either" if both, otherwise "not stated".
- prefers_daytime: they ask for daytime work.
- wants_predictable_schedule: they ask for a predictable or regular schedule.
- avoids_sales_targets: they say they do not want sales targets or sales-driven work.
- avoids_weekends: they say they cannot or do not want to work weekends.
- wants_training: they ask for training, an introduction or time to learn the role.
- open_to_sector_change: they say they are open to changing sector or industry."""


# --- Explanations -----------------------------------------------------------
EXPLAIN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["explanations"],
    "properties": {
        "explanations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["candidate_id", "why", "watch_out", "interview_question"],
                "properties": {
                    "candidate_id": {"type": "string"},
                    "why": {"type": "string"},
                    "watch_out": {"type": "string"},
                    "interview_question": {"type": "string"},
                },
            },
        }
    },
}

EXPLAIN_PROMPT = """You help a recruiter see why each candidate could fit a job, even when the candidate uses different words than the job ad.

For each candidate write:
- why: 1-2 plain English sentences linking the candidate's own experience to the job's needs. Base it on the evidence quotes given. Never invent experience.
- watch_out: one short sentence on the most important gap or preference conflict, or "None" if there is none.
- interview_question: one question the recruiter could ask to verify the most important skill.

Never mention or guess names, age, gender, nationality, ethnicity, religion, health or family situation. Refer to each person as "the candidate"."""


# --- Calls --------------------------------------------------------------------
def call_structured(client, system, user, name, schema, max_tokens=4000):
    response = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": name, "strict": True, "schema": schema},
        },
        reasoning_effort="low",
        max_completion_tokens=max_tokens,
    )
    return json.loads(response.choices[0].message.content)


def extract_job(client, title, text):
    return call_structured(client, JOB_PROMPT, f"Title: {title}\n\n{text}",
                           "job_requirements", JOB_SCHEMA)


def extract_candidate_prefs(client, background_text):
    return call_structured(client, CANDIDATE_PREFS_PROMPT, background_text,
                           "candidate_conditions", CANDIDATE_PREFS_SCHEMA)


def explain_matches(client, job, candidates):
    """job: dict with title/summary/requirements; candidates: list of dicts with
    candidate_id, summary, matched skills, gaps and notes. Returns {id: explanation}."""
    payload = json.dumps({"job": job, "candidates": candidates}, ensure_ascii=False)
    result = call_structured(client, EXPLAIN_PROMPT, payload, "match_explanations",
                             EXPLAIN_SCHEMA, max_tokens=8000)
    return {item["candidate_id"]: item for item in result["explanations"]}


def embed_texts(client, texts, batch_size=100):
    """Embeds texts; returns unit-length vectors (dot product = cosine)."""
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = [t if t.strip() else "-" for t in texts[start:start + batch_size]]
        response = client.embeddings.create(model=EMBEDDING_DEPLOYMENT, input=batch)
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
    matrix = np.array(vectors, dtype=np.float32).reshape(len(vectors), -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


# --- Text helpers -------------------------------------------------------------
def clean_term(term):
    """'Helpdesk / Service Desk Agent (junior)' -> ['Helpdesk', 'Service Desk Agent']"""
    term = re.sub(r"\([^)]*\)", " ", term)
    parts = re.split(r"/|\s[-–—]\s", term)
    cleaned = [re.sub(r"\s+", " ", p).strip(" ,.;:-") for p in parts]
    return [p for p in cleaned if len(p) >= 2]
