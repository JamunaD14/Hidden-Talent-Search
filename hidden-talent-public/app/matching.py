"""
The matching engine, stage 2: scores candidates against one job (recruiter
view), or jobs against one candidate (jobseeker view).

Stage 1 (in main.py) is Azure AI Search hybrid search (keywords + vectors,
with filters), which narrows the pool. This file then re-ranks with criteria
a recruiter can see and adjust.

Scores are computed in code (not by the model), so they are repeatable,
testable and explainable. Four criteria, each 0-1:

  skills       How many of the job's needs the candidate covers, compared by
               meaning (embeddings), so "help visitors use self-service
               machines" can cover "kundsupport". Must-haves count double.
  semantic     Overall similarity of the job and the candidate profile,
               relative to the others being compared.
  preferences  Fit with what the candidate said they want (hours, daytime,
               no sales targets...). Starts at 1, conflicts lower it.
  evidence     How many matched skills are backed by a quote from the
               candidate's own words (3 or more = full score).

Total = weighted average x 100. Weights can be changed per search.
"""

import json
import os

import numpy as np

from app.extraction import clean_term

DEFAULT_WEIGHTS = {"skills": 50, "semantic": 25, "preferences": 15, "evidence": 10}

# Similarity thresholds for text-embedding-3-small (cosine similarity)
LOW, HIGH = 0.40, 0.70   # similarity below LOW = no credit, above HIGH = full credit
MATCHED = 0.55           # shown as a match in the skills bridge
MISSING = 0.45           # a must-have below this counts as a gap

# Taxonomy concept types used as job needs ("occupation-field" is too broad)
JOB_CONCEPT_TYPES = {"skill", "occupation-name", "ssyk-level-4"}
NEED_ORDER = {"must": 0, "nice": 1, "taxonomy": 2}


def normalise_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        norm = np.linalg.norm(matrix)
        return matrix / norm if norm else matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def job_profile_text(job):
    """English text for a job's vector (same language as candidate profiles)."""
    req = job.get("requirements") or {}
    if not req:
        return f"{job.get('title', '')}. {job.get('text', '')[:2000]}"
    return (f"{req['title_en']}. {req['summary_en']} "
            f"Needs: {', '.join(req['must_have'])}. "
            f"Nice to have: {', '.join(req['nice_to_have'])}.")


def job_requirements(job):
    """List of (term, weight, kind) the job needs. Must-haves weigh 2."""
    req = job.get("requirements") or {}
    items = [(t, 2.0, "must") for t in req.get("must_have", [])]
    items += [(t, 1.0, "nice") for t in req.get("nice_to_have", [])]
    items += [(t, 1.0, "taxonomy") for t in job.get("taxonomy_labels", [])]
    seen, unique = set(), []
    for term, weight, kind in items:
        key = term.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append((term.strip(), weight, kind))
    return unique


def match_label(total):
    """The Match score in words, for people who don't want a number."""
    if total >= 70:
        return "Strong match"
    if total >= 55:
        return "Good match"
    if total >= 40:
        return "Possible match"
    return "Weak match"


def candidate_summary(candidate):
    """A readable, anonymous headline for a candidate: no names, ever."""
    profile = candidate["profile"]
    settings = profile.get("experience_settings") or []
    roles, seen = [], set()
    for role in profile.get("possible_roles", []):
        for part in clean_term(role):
            if part.lower() not in seen:
                seen.add(part.lower())
                roles.append(part)
    number = candidate["id"].rsplit("-", 1)[-1]
    return {
        "candidate_id": candidate["id"],
        "alias": f"Candidate {number}",
        "roles": roles[:3],
        "experience": settings[0]["setting"] if settings else None,
        "experience_kind": settings[0]["kind"] if settings else None,
    }


def job_card(job):
    """The facts a jobseeker needs about a job ad."""
    req = job.get("requirements") or {}
    return {
        "job_id": job["id"], "title": job["title"], "title_en": req.get("title_en"),
        "municipality": job.get("municipality"), "sector": job.get("sector"),
        "source_url": job.get("source_url"), "hours": req.get("hours"),
        "schedule": req.get("schedule", []), "sales_targets": req.get("sales_targets"),
        "training_provided": req.get("training_provided"),
    }


class Store:
    """Everything the matching needs, loaded once from the app_data folder."""

    def __init__(self, folder):
        def load_json(name):
            with open(os.path.join(folder, name), encoding="utf-8") as fh:
                return json.load(fh)

        self.jobs_list = load_json("jobs.json")
        self.jobs = {job["id"]: job for job in self.jobs_list}
        self.job_vectors = normalise_rows(np.load(os.path.join(folder, "job_vectors.npy")))
        self.job_row = {job["id"]: i for i, job in enumerate(self.jobs_list)}

        self.candidates = load_json("candidates.json")
        self.candidate_by_id = {c["id"]: c for c in self.candidates}
        self.candidate_vectors = normalise_rows(np.load(os.path.join(folder, "candidate_vectors.npy")))
        self.candidate_row = {c["id"]: i for i, c in enumerate(self.candidates)}

        terms = load_json("terms.json")
        self.term_vectors = normalise_rows(np.load(os.path.join(folder, "term_vectors.npy")))
        self.term_row = {t.lower(): i for i, t in enumerate(terms)}
        self.extra_vectors = {}   # terms embedded while the app runs
        self._job_needs = {}      # job id -> (requirements, their vectors)

        for candidate in self.candidates:
            candidate["_rows"] = np.array(
                [self.term_row[t["term"].lower()] for t in candidate["terms"]], dtype=int)

    def job_vector(self, job_id):
        return self.job_vectors[self.job_row[job_id]]

    def candidate_vector(self, candidate_id):
        return self.candidate_vectors[self.candidate_row[candidate_id]]

    def candidate_term_vectors(self, candidate):
        return self.term_vectors[candidate["_rows"]]

    def vectors_for(self, terms, embed):
        """Vectors for terms; embeds (and remembers) any the store hasn't seen."""
        unknown = [t for t in terms
                   if t.lower() not in self.term_row and t.lower() not in self.extra_vectors]
        if unknown:
            for term, vector in zip(unknown, embed(unknown)):
                self.extra_vectors[term.lower()] = vector
        rows = [self.term_vectors[self.term_row[t.lower()]] if t.lower() in self.term_row
                else self.extra_vectors[t.lower()] for t in terms]
        return normalise_rows(np.array(rows)) if rows else np.zeros((0, self.term_vectors.shape[1]))

    def needs_for(self, job, embed):
        """(requirements, requirement vectors) for a job; cached for stored jobs."""
        if job["id"] in self.jobs and job["id"] in self._job_needs:
            return self._job_needs[job["id"]]
        requirements = job_requirements(job)
        result = (requirements, self.vectors_for([r[0] for r in requirements], embed))
        if job["id"] in self.jobs:
            self._job_needs[job["id"]] = result
        return result


def preference_fit(prefs, req):
    """Compares what the candidate said they want with the job's conditions."""
    if not prefs or not req:
        return 1.0, [], False
    score, notes, conflict = 1.0, [], False
    schedule = set(req.get("schedule", []))

    if prefs.get("avoids_sales_targets") and req.get("sales_targets"):
        score -= 0.6
        conflict = True
        notes.append({"type": "warn", "text": "Wants to avoid sales targets; this role involves sales targets"})

    unsocial = schedule & {"evenings", "nights", "shifts"}
    if prefs.get("prefers_daytime") and unsocial:
        score -= 0.3
        notes.append({"type": "warn", "text": "Prefers daytime work; role includes " + ", ".join(sorted(unsocial))})
    elif prefs.get("prefers_daytime") and "daytime" in schedule:
        notes.append({"type": "ok", "text": "Daytime work, as the candidate prefers"})

    if prefs.get("avoids_weekends") and "weekends" in schedule:
        score -= 0.3
        notes.append({"type": "warn", "text": "Wants to avoid weekends; role includes weekend work"})

    if prefs.get("wants_predictable_schedule") and "shifts" in schedule:
        score -= 0.2
        notes.append({"type": "warn", "text": "Wants a predictable schedule; role is shift-based"})

    wanted, offered = prefs.get("hours"), req.get("hours")
    if wanted == "part-time" and offered == "full-time":
        score -= 0.4
        notes.append({"type": "warn", "text": "Wants part-time; role is full-time"})
    elif wanted == "full-time" and offered == "part-time":
        score -= 0.2
        notes.append({"type": "warn", "text": "Wants full-time; role is part-time"})
    elif wanted in ("part-time", "full-time") and (wanted == offered or offered == "either"):
        notes.append({"type": "ok", "text": f"Hours fit: candidate wants {wanted}"})

    if prefs.get("wants_training") and req.get("training_provided"):
        notes.append({"type": "ok", "text": "Employer offers training, which the candidate asked for"})

    return max(0.0, score), notes, conflict


def match_needs(requirements, requirement_vectors, terms, term_vectors):
    """Compares each job need with the candidate's terms by meaning.
    Returns (skills score 0-1, matched needs, missing must-haves, skills with evidence)."""
    if not len(requirements) or not len(term_vectors):
        return 0.0, [], [], set()
    need_weights = np.array([w for _, w, _ in requirements], dtype=np.float32)
    similarity = requirement_vectors @ term_vectors.T     # needs x candidate terms
    best_term = similarity.argmax(axis=1)
    best_sim = similarity.max(axis=1)
    credit = np.clip((best_sim - LOW) / (HIGH - LOW), 0, 1)
    skills = float((need_weights * credit).sum() / need_weights.sum())

    matched, gaps, evidence_skills = [], [], set()
    for n, (need, _, kind) in enumerate(requirements):
        term = terms[int(best_term[n])]
        sim = float(best_sim[n])
        if sim >= MATCHED:
            matched.append({
                "need": need, "need_kind": kind,
                "candidate_term": term["term"], "source": term["source"],
                "evidence": term.get("evidence"), "similarity": round(sim, 2),
            })
            if term["source"] == "skill" and term.get("verified"):
                evidence_skills.add(term["term"].lower())
        elif kind == "must" and sim < MISSING:
            gaps.append(need)
    matched.sort(key=lambda m: (NEED_ORDER[m["need_kind"]], -m["similarity"]))
    return skills, matched, gaps, evidence_skills


def clean_weights(weights):
    return {k: max(0.0, float((weights or {}).get(k, 0))) for k in DEFAULT_WEIGHTS}


def weighted_total(scores, weights):
    weight_sum = sum(weights.values()) or 1.0
    return sum(weights[k] * scores[k] for k in weights) / weight_sum * 100


def relative(values):
    """Rescales to 0-1 so the best gets 1 and the worst 0."""
    low, high = float(values.min()), float(values.max())
    return (values - low) / (high - low) if high > low else np.ones_like(values)


def score_candidates(store, job, job_vector, requirements, requirement_vectors,
                     weights, baseline_ids=None, exclude_conflicts=False, candidate_ids=None):
    """Scores candidates for one job (all, or only candidate_ids, e.g. the ones
    Azure AI Search retrieved). Returns a list sorted best first."""
    weights = clean_weights(weights)
    semantic = relative(store.candidate_vectors @ normalise_rows(job_vector))

    results = []
    for i, candidate in enumerate(store.candidates):
        if candidate_ids is not None and candidate["id"] not in candidate_ids:
            continue
        skills, matched, gaps, evidence_skills = match_needs(
            requirements, requirement_vectors, candidate["terms"], store.term_vectors[candidate["_rows"]])
        prefs_score, notes, conflict = preference_fit(candidate.get("prefs"), job.get("requirements"))
        if exclude_conflicts and conflict:
            continue

        scores = {
            "skills": round(skills, 3),
            "semantic": round(float(semantic[i]), 3),
            "preferences": round(prefs_score, 3),
            "evidence": round(min(1.0, len(evidence_skills) / 3), 3),
        }
        total = weighted_total(scores, weights)
        results.append(dict(
            candidate_summary(candidate),
            total=round(total, 1),
            label=match_label(total),
            scores=scores,
            hidden_talent=baseline_ids is not None and candidate["id"] not in baseline_ids,
            found_by_keyword_search=baseline_ids is not None and candidate["id"] in baseline_ids,
            matched=matched,
            gaps=gaps,
            stated_gaps=candidate["profile"].get("stated_gaps", []),
            preference_notes=notes,
            preference_conflict=conflict,
            summary=candidate["profile"]["summary"],
            skills=[s["name"] for s in candidate["profile"]["skills"]],
            possible_roles=candidate["profile"]["possible_roles"],
            background_text=candidate["background_text"],
        ))

    results.sort(key=lambda r: r["total"], reverse=True)
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    return results


def score_jobs_for_candidate(store, candidate, candidate_vector, term_vectors, embed,
                             weights=None, top_k=5):
    """The jobseeker view: scores every job ad for one candidate, with the same
    four criteria. 'Overall meaning' is relative to the other job ads."""
    weights = clean_weights(weights or DEFAULT_WEIGHTS)
    semantic = relative(store.job_vectors @ normalise_rows(candidate_vector))

    results = []
    for i, job in enumerate(store.jobs_list):
        if not job.get("requirements"):
            continue
        requirements, requirement_vectors = store.needs_for(job, embed)
        skills, matched, gaps, evidence_skills = match_needs(
            requirements, requirement_vectors, candidate["terms"], term_vectors)
        prefs_score, notes, conflict = preference_fit(candidate.get("prefs"), job["requirements"])
        scores = {
            "skills": round(skills, 3),
            "semantic": round(float(semantic[i]), 3),
            "preferences": round(prefs_score, 3),
            "evidence": round(min(1.0, len(evidence_skills) / 3), 3),
        }
        total = weighted_total(scores, weights)
        results.append(dict(
            job_card(job), total=round(total, 1), label=match_label(total), scores=scores,
            matched=matched, gaps=gaps, preference_notes=notes, preference_conflict=conflict,
        ))
    results.sort(key=lambda r: r["total"], reverse=True)
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    return results[:top_k]
