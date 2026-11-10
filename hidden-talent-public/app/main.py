"""
Hidden Talent Search: the web API and the web page.

Run locally (from the project folder):
  uvicorn app.main:app --reload --port 8000
  then open http://localhost:8000

API (all JSON, so another front end can use it too):
  GET  /api/health                 status and data counts
  GET  /api/jobs?q=lager&limit=50  find job ads (AI Search)
  GET  /api/jobs/{job_id}          one job ad with its extracted needs
  POST /api/match                  rank candidates for a job ad or a free-text need:
                                   stage 1 Azure AI Search hybrid (keywords + vectors,
                                   optional hours filter), stage 2 re-rank (matching.py)
  POST /api/explain                short explanations for chosen candidates
  GET  /api/matches/{job_id}       the last saved ranking for a job (Cosmos DB)

  Jobseeker view:
  GET  /api/candidates?ids=a,b     anonymous headlines (and texts) of sample candidates
  GET  /api/candidates/{id}/jobs   the job ads a pool candidate fits best
  POST /api/jobseeker              {"text": "..."}: anonymise, read, and find the best-fit jobs

  Shortlist:
  POST /api/shortlist              save a recruiter's shortlist for a job (Cosmos DB)
  GET  /api/shortlist/{job_id}     read it back
"""

import hashlib
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.extraction import embed_texts, explain_matches, extract_candidate_prefs, extract_job
from app.matching import (
    DEFAULT_WEIGHTS, Store, candidate_summary, job_profile_text, score_candidates,
    score_jobs_for_candidate,
)
from app.privacy import anonymise
from app.profile import candidate_terms, extract_profile, profile_text
from common import cosmos_container, openai_client, search_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for noisy in ("azure", "httpx", "openai"):  # keep the terminal readable
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("hidden-talent")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("APP_DATA", os.path.join(os.path.dirname(APP_DIR), "app_data"))
CANDIDATE_INDEX = os.getenv("CANDIDATE_INDEX", "candidates")

store = Store(DATA_DIR)
client = openai_client()
query_cache = {}      # free-text searches already analysed: hash -> (job, vector)
jobseeker_cache = {}  # jobseeker stories already read: hash -> response

app = FastAPI(title="Hidden Talent Search")
app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")
log.info("Loaded %d jobs and %d candidates from %s",
         len(store.jobs), len(store.candidates), DATA_DIR)


def embed(texts):
    return embed_texts(client, texts)


# --- Helpers ----------------------------------------------------------------
def resolve_job(job_id=None, query_text=None):
    """Returns (job, job_vector) for a job ad id or a free-text description."""
    if job_id:
        job = store.jobs.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} not found")
        return job, store.job_vector(job_id)
    if query_text and query_text.strip():
        key = hashlib.sha256(query_text.strip().encode()).hexdigest()
        if key not in query_cache:
            requirements = extract_job(client, "Recruiter's description", query_text)
            job = {"id": "free-text", "title": requirements["title_en"], "text": query_text,
                   "taxonomy_labels": [], "requirements": requirements,
                   "sector": None, "municipality": None, "source_url": None}
            query_cache[key] = (job, embed([job_profile_text(job)])[0])
        return query_cache[key]
    raise HTTPException(400, "Give a job_id or a query_text")


RETRIEVE_K = int(os.getenv("RETRIEVE_K", "50"))
HOURS_FILTERS = {  # OData filters on the candidates index (field 'hours')
    "part-time": "hours ne 'full-time'",   # candidates who did not ask for full-time only
    "full-time": "hours ne 'part-time'",   # candidates who did not ask for part-time only
}


def search_text_for(job):
    req = job.get("requirements") or {}
    words = [req.get("title_en") or job["title"]] + req.get("must_have", []) + job.get("taxonomy_labels", [])
    return re.sub(r"[^\w\s]", " ", " ".join(words))


def retrieve_candidates(job, job_vector, hours_filter=None):
    """Stage 1: Azure AI Search hybrid search (keywords + vectors) on the
    enriched candidate profiles, with an optional filter. Narrows the pool,
    so the same design works for thousands of candidates."""
    from azure.search.documents.models import VectorizedQuery

    vector_query = VectorizedQuery(vector=[float(x) for x in job_vector],
                                   k_nearest_neighbors=RETRIEVE_K, fields="vector")
    results = search_client(CANDIDATE_INDEX).search(
        search_text=search_text_for(job), vector_queries=[vector_query],
        filter=HOURS_FILTERS.get(hours_filter), select=["id"], top=RETRIEVE_K)
    return [r["id"] for r in results]


def local_hours_filter(hours_filter):
    """Same filter as HOURS_FILTERS, applied in code (fallback)."""
    blocked = {"part-time": "full-time", "full-time": "part-time"}.get(hours_filter)
    if not blocked:
        return None
    return {c["id"] for c in store.candidates
            if c["profile"]["work_preferences"]["hours"] != blocked}


def keyword_baseline(job, k, hours_filter=None):
    """What a recruiter's keyword search finds: the ad's English title plus its
    required skills (any word may match), against the candidates' own words,
    ranked by keyword relevance. English gives keyword search its best chance,
    since the candidates write in English. Returns (top-k ids, title, error)."""
    req = job.get("requirements") or {}
    title = req.get("title_en") or job["title"]
    query = re.sub(r"[^\w\s]", " ", " ".join([title] + req.get("must_have", [])))
    try:
        results = search_client(CANDIDATE_INDEX).search(
            search_text=query, search_fields=["background_text"], search_mode="any",
            filter=HOURS_FILTERS.get(hours_filter), select=["id"], top=k)
        return [r["id"] for r in results], title, None
    except Exception as error:
        log.warning("Keyword baseline failed: %s", error)
        return None, title, str(error)[:200]


def clean_weights(weights):
    merged = dict(DEFAULT_WEIGHTS)
    for key, value in (weights or {}).items():
        if key in merged:
            merged[key] = max(0.0, float(value))
    return merged


def run_match(job, job_vector, weights, exclude_conflicts, top_k=10,
              with_baseline=True, use_retrieval=True, hours_filter=None):
    """Stage 1 (AI Search hybrid retrieval + filter), then stage 2 (re-rank)."""
    requirements, requirement_vectors = store.needs_for(job, embed)

    retrieval = {"method": "all candidates", "retrieved": None, "error": None}
    candidate_ids = local_hours_filter(hours_filter)
    if use_retrieval:
        try:
            ids = retrieve_candidates(job, job_vector, hours_filter)
            candidate_ids = set(ids)
            retrieval = {"method": "Azure AI Search hybrid", "retrieved": len(ids), "error": None}
        except Exception as error:
            log.warning("Hybrid retrieval failed, scoring all candidates: %s", error)
            retrieval["error"] = str(error)[:200]

    if with_baseline:
        baseline_ids, title, error = keyword_baseline(job, top_k, hours_filter)
    else:
        baseline_ids, title, error = None, None, "not requested"
    results = score_candidates(store, job, job_vector, requirements, requirement_vectors,
                               weights, None if baseline_ids is None else set(baseline_ids),
                               exclude_conflicts, candidate_ids)
    baseline = {"query": title, "top_k": top_k, "error": error, "ids": baseline_ids}
    return results, baseline, retrieval


def job_summary(job):
    return {key: job.get(key) for key in
            ["id", "title", "sector", "municipality", "source_url", "taxonomy_labels", "requirements"]}


def explanation_input(job, results):
    req = job.get("requirements") or {}
    job_info = {"title": req.get("title_en") or job["title"], "summary": req.get("summary_en"),
                "must_have": req.get("must_have"), "nice_to_have": req.get("nice_to_have")}
    candidates = [{
        "candidate_id": r["candidate_id"],
        "summary": r["summary"],
        "candidate_own_words": r["background_text"],
        "matched": [{"need": m["need"], "candidate_skill": m["candidate_term"],
                     "evidence": m["evidence"]} for m in r["matched"][:8]],
        "missing_must_haves": r["gaps"],
        "stated_gaps": r["stated_gaps"],
        "preference_notes": [n["text"] for n in r["preference_notes"]],
    } for r in results]
    return job_info, candidates


def save_match(job, weights, baseline, results, retrieval=None, hours_filter=None):
    """Stores the ranking in Cosmos DB 'matches' (runs after the response)."""
    try:
        cosmos_container("matches").upsert_item({
            "id": job["id"], "partition_key": job["id"], "job_id": job["id"],
            "job_title": job["title"], "created_at": datetime.now(timezone.utc).isoformat(),
            "weights": weights, "baseline": baseline, "retrieval": retrieval,
            "hours_filter": hours_filter,
            "results": [{key: r[key] for key in
                         ["rank", "candidate_id", "total", "scores", "hidden_talent"]}
                        for r in results],
        })
        log.info("Saved ranking for %s to Cosmos DB", job["id"])
    except Exception as error:
        log.warning("Could not save to Cosmos DB: %s", error)


# --- Request bodies ---------------------------------------------------------
class MatchRequest(BaseModel):
    job_id: Optional[str] = None
    query_text: Optional[str] = None
    weights: Optional[dict] = None
    top_k: int = Field(10, ge=1, le=100)
    exclude_conflicts: bool = False
    hours_filter: Optional[str] = None  # "part-time" or "full-time"
    explain: bool = False
    save: bool = True


class ExplainRequest(BaseModel):
    job_id: Optional[str] = None
    query_text: Optional[str] = None
    candidate_ids: list[str] = Field(..., min_length=1, max_length=10)


# --- Endpoints --------------------------------------------------------------
@app.get("/")
def home():
    return FileResponse(os.path.join(APP_DIR, "static", "index.html"))


@app.get("/api/health")
def health():
    with_req = sum(1 for job in store.jobs_list if job.get("requirements"))
    return {"status": "ok", "jobs": len(store.jobs), "jobs_with_requirements": with_req,
            "candidates": len(store.candidates), "default_weights": DEFAULT_WEIGHTS}


@app.get("/api/jobs")
def list_jobs(q: str = "", limit: int = 50):
    limit = max(1, min(limit, 200))
    ids = None
    if q.strip():
        try:
            results = search_client("jobs").search(search_text=q, select=["id"], top=limit)
            ids = [r["id"] for r in results]
        except Exception as error:  # fall back to a simple local search
            log.warning("Job search failed, using local search: %s", error)
            needle = q.lower()
            ids = [job["id"] for job in store.jobs_list
                   if needle in f"{job['title']} {job.get('municipality')} {job.get('sector')} "
                                 f"{(job.get('requirements') or {}).get('title_en', '')}".lower()]
    jobs = [store.jobs[i] for i in ids if i in store.jobs] if ids is not None else store.jobs_list
    return [{"id": job["id"], "title": job["title"],
             "title_en": (job.get("requirements") or {}).get("title_en"),
             "municipality": job.get("municipality"), "sector": job.get("sector")}
            for job in jobs[:limit]]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = store.jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return dict(job_summary(job), text=job["text"])


@app.post("/api/match")
def match(request: MatchRequest, background: BackgroundTasks):
    started = time.time()
    job, job_vector = resolve_job(request.job_id, request.query_text)
    weights = clean_weights(request.weights)
    results, baseline, retrieval = run_match(job, job_vector, weights, request.exclude_conflicts,
                                             request.top_k, hours_filter=request.hours_filter)
    top = results[: request.top_k]

    if request.explain and top:
        job_info, candidates = explanation_input(job, top[:5])
        explanations = explain_matches(client, job_info, candidates)
        for result in top:
            result["explanation"] = explanations.get(result["candidate_id"])

    hidden_in_top = sum(r["hidden_talent"] for r in top) if baseline["ids"] is not None else None
    our_rank = {r["candidate_id"]: r["rank"] for r in results}
    baseline_results = [
        dict(candidate_summary(store.candidate_by_id[cid]), keyword_rank=n, our_rank=our_rank.get(cid))
        for n, cid in enumerate(baseline["ids"] or [], start=1) if cid in store.candidate_by_id
    ]
    response = {
        "job": job_summary(job), "weights": weights, "baseline": baseline,
        "baseline_results": baseline_results,
        "retrieval": retrieval, "hours_filter": request.hours_filter,
        "stats": {"candidates_total": len(store.candidates),
                  "removed_by_filter": (len(store.candidates) - len(local_hours_filter(request.hours_filter))
                                        if request.hours_filter in HOURS_FILTERS else None),
                  "candidates_scored": len(results), "top_k": len(top),
                  "hidden_talent_in_top_k": hidden_in_top,
                  "seconds": round(time.time() - started, 2)},
        "results": top,
    }
    if request.save and request.job_id:
        background.add_task(save_match, job, weights, baseline, top, retrieval, request.hours_filter)
    log.info("match job=%s top=%d hidden=%s took=%.2fs",
             job["id"], len(top), hidden_in_top, time.time() - started)
    return response


@app.post("/api/explain")
def explain(request: ExplainRequest):
    job, job_vector = resolve_job(request.job_id, request.query_text)
    results, _, _ = run_match(job, job_vector, DEFAULT_WEIGHTS, False,
                              with_baseline=False, use_retrieval=False)
    chosen = [r for r in results if r["candidate_id"] in set(request.candidate_ids)]
    if not chosen:
        raise HTTPException(404, "None of those candidates were found")
    job_info, candidates = explanation_input(job, chosen)
    return explain_matches(client, job_info, candidates)


@app.get("/api/matches/{job_id}")
def saved_match(job_id: str):
    try:
        return cosmos_container("matches").read_item(item=job_id, partition_key=job_id)
    except Exception as error:
        raise HTTPException(404, f"No saved ranking for {job_id}: {str(error)[:120]}")


# --- Jobseeker view -----------------------------------------------------------
def profile_view(profile):
    """The parts of a profile a jobseeker sees: their words in employer terms."""
    return {
        "skills": [{"name": s["name"], "evidence": s["evidence"],
                    "verified": s.get("evidence_verified")} for s in profile["skills"]],
        "possible_roles": profile["possible_roles"],
        "search_terms_sv": profile["search_terms_sv"],
        "stated_gaps": profile.get("stated_gaps", []),
    }


@app.get("/api/candidates")
def list_candidates(ids: str = "", limit: int = 100):
    wanted = [i.strip() for i in ids.split(",") if i.strip()] or [c["id"] for c in store.candidates]
    found = [store.candidate_by_id[i] for i in wanted if i in store.candidate_by_id][: max(1, min(limit, 100))]
    return [dict(candidate_summary(c), background_text=c["background_text"]) for c in found]


@app.get("/api/candidates/{candidate_id}/jobs")
def candidate_jobs(candidate_id: str, top_k: int = 5):
    candidate = store.candidate_by_id.get(candidate_id)
    if not candidate:
        raise HTTPException(404, f"Candidate {candidate_id} not found")
    jobs = score_jobs_for_candidate(store, candidate, store.candidate_vector(candidate_id),
                                    store.candidate_term_vectors(candidate), embed,
                                    top_k=max(1, min(top_k, 20)))
    return {"candidate": candidate_summary(candidate), "background_text": candidate["background_text"],
            "removed_details": [], "profile": profile_view(candidate["profile"]), "jobs": jobs}


class JobseekerRequest(BaseModel):
    text: str = Field(..., min_length=40, max_length=3000)
    top_k: int = Field(5, ge=1, le=20)


@app.post("/api/jobseeker")
def jobseeker(request: JobseekerRequest):
    """A story typed in by a jobseeker: anonymised first, then read by the
    model exactly like the candidate pool (Step 1), then matched to every ad."""
    started = time.time()
    clean, removed = anonymise(request.text)
    if len(clean) < 40:
        raise HTTPException(400, "Tell us a little more about what you have done: a few sentences is enough.")
    key = hashlib.sha256(f"{clean}|{request.top_k}".encode()).hexdigest()
    if key in jobseeker_cache:
        return jobseeker_cache[key]

    with ThreadPoolExecutor(max_workers=2) as pool:
        profile_job = pool.submit(extract_profile, client, clean)
        prefs_job = pool.submit(extract_candidate_prefs, client, clean)
        profile, prefs = profile_job.result(), prefs_job.result()

    terms = candidate_terms(profile)
    term_vectors = store.vectors_for([t["term"] for t in terms], embed)
    vector = embed([profile_text(profile)])[0]
    person = {"id": "candidate-you", "profile": profile, "prefs": prefs, "terms": terms,
              "background_text": clean}
    jobs = score_jobs_for_candidate(store, person, vector, term_vectors, embed, top_k=request.top_k)
    response = {"background_text": clean, "removed_details": removed,
                "profile": profile_view(profile), "jobs": jobs,
                "seconds": round(time.time() - started, 1)}
    jobseeker_cache[key] = response
    log.info("jobseeker story read: %d skills, top job %s, took %.1fs",
             len(profile["skills"]), jobs[0]["job_id"] if jobs else None, time.time() - started)
    return response


# --- Shortlist -------------------------------------------------------------------
class ShortlistRequest(BaseModel):
    job_id: str = Field(..., min_length=1, max_length=120)
    job_title: Optional[str] = None
    candidates: list[dict] = Field(default_factory=list, max_length=50)


@app.post("/api/shortlist")
def save_shortlist(request: ShortlistRequest):
    item = {
        "id": f"shortlist-{request.job_id}", "partition_key": request.job_id, "kind": "shortlist",
        "job_id": request.job_id, "job_title": request.job_title,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "candidates": [{k: c.get(k) for k in ("candidate_id", "alias", "label", "total", "hidden_talent")}
                       for c in request.candidates],
    }
    try:
        cosmos_container("matches").upsert_item(item)
        return {"saved": True, "count": len(item["candidates"]), "updated_at": item["updated_at"]}
    except Exception as error:
        log.warning("Could not save shortlist: %s", error)
        return {"saved": False, "error": str(error)[:200]}


@app.get("/api/shortlist/{job_id}")
def get_shortlist(job_id: str):
    try:
        return cosmos_container("matches").read_item(item=f"shortlist-{job_id}", partition_key=job_id)
    except Exception:
        return {"job_id": job_id, "candidates": []}
