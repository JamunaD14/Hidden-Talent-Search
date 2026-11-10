"""
Reading a jobseeker's story into a structured profile.

Used by Step 1 (enrich_candidates.py) for the candidate pool, and by the app's
jobseeker view for a story someone types in. Also holds the two helpers that
turn a profile into what the matching uses: one text for the profile's vector,
and the list of terms (skills, roles, Swedish words) needs are compared with.

The text passed in must already be anonymised (app/privacy.py).
"""

import json
import re

from app.extraction import clean_term
from common import CHAT_DEPLOYMENT

# --- What the model must return (enforced with structured output) ----------
PROFILE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "summary", "skills", "possible_roles", "search_terms_sv",
        "experience_settings", "years_experience", "stated_gaps",
        "work_preferences", "languages",
    ],
    "properties": {
        "summary": {"type": "string"},
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "kind", "evidence"],
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": ["hard", "soft"]},
                    "evidence": {"type": "string"},
                },
            },
        },
        "possible_roles": {"type": "array", "items": {"type": "string"}},
        "search_terms_sv": {"type": "array", "items": {"type": "string"}},
        "experience_settings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["setting", "kind"],
                "properties": {
                    "setting": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["paid", "volunteer", "education", "personal", "unknown"],
                    },
                },
            },
        },
        "years_experience": {"type": ["number", "null"]},
        "stated_gaps": {"type": "array", "items": {"type": "string"}},
        "work_preferences": {
            "type": "object",
            "additionalProperties": False,
            "required": ["wanted", "avoid", "open_to", "hours"],
            "properties": {
                "wanted": {"type": "array", "items": {"type": "string"}},
                "avoid": {"type": "array", "items": {"type": "string"}},
                "open_to": {"type": "array", "items": {"type": "string"}},
                "hours": {
                    "type": "string",
                    "enum": ["full-time", "part-time", "either", "not stated"],
                },
            },
        },
        "languages": {"type": "array", "items": {"type": "string"}},
    },
}

SYSTEM_PROMPT = """You turn a jobseeker's own description of their experience into a profile that employers can search.

Rules:
- Use only what the text supports. Never invent experience, years, languages or places.
- Translate everyday wording into the terms recruiters search for. Example: "help visitors use self-service machines" -> "technical support", "user guidance".
- skills: "name" is a short, standard skill term recruiters use (1-4 words), with no brackets, slashes or context; put the context in the evidence instead. Good: "incident documentation". Bad: "customer support (volunteer repair intake)". Every skill needs an evidence quote copied word for word from the text (a short phrase). Mark each as "hard" (technical/practical) or "soft" (interpersonal).
- possible_roles: 3-6 English job titles a recruiter might search for this person, including adjacent roles they could grow into.
- search_terms_sv: 5-10 Swedish words or job titles a Swedish employer would use in a job ad for these skills (e.g. "kundtjänst", "kundsupport", "felsökning").
- experience_settings: where the experience comes from, and whether it was paid, volunteer, education or personal.
- years_experience: a number only if the text states it, otherwise null.
- stated_gaps: skills, tools or experience the person says they lack (e.g. "commercial ticketing system"). Empty if none are stated.
- work_preferences: roles, tasks or conditions the person says they want or want to avoid (e.g. "daytime work", "sales targets"); "open_to" only for things the text explicitly says they are open to or willing to do (e.g. "changing sectors", "initial training", "part-time while learning"), otherwise empty; and working hours if stated.
- languages: only languages the text says the person speaks.
- Never include or guess names, age, gender, nationality, ethnicity, religion, health or family situation in any field, even if the text mentions them. Placeholders such as [name] or [age] mean a detail was removed: ignore them.
- summary: 2 sentences of neutral, professional English starting with "The candidate", with no personal details."""


def normalise(text):
    return re.sub(r"\s+", " ", text.lower()).strip(' ."\'')


def extract_profile(client, text):
    """Anonymised free text -> profile dict. Each skill gets evidence_verified:
    whether its quote really appears in the text."""
    response = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "candidate_profile", "strict": True, "schema": PROFILE_SCHEMA},
        },
        reasoning_effort="low",
        max_completion_tokens=4000,
    )
    profile = json.loads(response.choices[0].message.content)
    source = normalise(text)
    for skill in profile["skills"]:
        skill["evidence_verified"] = normalise(skill["evidence"]) in source
    return profile


def profile_text(profile):
    """One text per candidate, in employer vocabulary, used for the vector."""
    return " ".join([
        profile["summary"],
        "Skills: " + ", ".join(s["name"] for s in profile["skills"]) + ".",
        "Possible roles: " + ", ".join(profile["possible_roles"]) + ".",
        "Svenska sökord: " + ", ".join(profile["search_terms_sv"]) + ".",
    ])


def candidate_terms(profile):
    """Every term a candidate can be matched on, with where it came from."""
    terms, seen = [], set()

    def add(raw, source, evidence=None, verified=None):
        for part in clean_term(raw):
            if part.lower() not in seen:
                seen.add(part.lower())
                terms.append({"term": part, "source": source,
                              "evidence": evidence, "verified": verified})

    for skill in profile["skills"]:
        add(skill["name"], "skill", skill["evidence"], skill.get("evidence_verified"))
    for role in profile["possible_roles"]:
        add(role, "role")
    for term in profile["search_terms_sv"]:
        add(term, "swedish")
    return terms
