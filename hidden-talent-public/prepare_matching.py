"""
Step 3 of the pipeline: prepare everything the app needs, so searches are fast.

What it does:
  1. Reads the 200 job ads from AI Search and extracts each job's needs in
     English (must-haves, nice-to-haves) and its conditions (hours, schedule,
     sales targets, training).
  2. Reads each candidate's own text for stated work conditions (part-time,
     daytime, no sales targets, wants training...).
  3. Embeds all job needs and all candidate skills, roles and Swedish terms.
  4. Saves everything in the app_data/ folder, which the app loads at start.

Input : AI Search index 'jobs'
        candidates_ready.json, candidate_vectors.npy, taxonomy_meta.json (Step 2)
Output: app_data/  (jobs, candidates, vectors, plus caches of the model results)

Run (from the project folder):
  python3 prepare_matching.py --limit 5    # test: 5 jobs + 5 candidates
  python3 prepare_matching.py              # everything (re-uses earlier results)
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from app.extraction import clean_term, embed_texts, extract_candidate_prefs, extract_job
from app.matching import JOB_CONCEPT_TYPES, job_profile_text
from app.profile import candidate_terms
from common import EMBEDDING_DIMENSIONS, openai_client, search_client

DATA_DIR = "app_data"
JOB_REQ_CACHE = os.path.join(DATA_DIR, "job_requirements_cache.json")
PREFS_CACHE = os.path.join(DATA_DIR, "candidate_prefs_cache.json")


def load_json(path, default=None):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)


def run_model_calls(name, items, call, cache, cache_path, workers):
    """Runs `call` for each item in parallel; stores results in cache by id."""
    if not items:
        print(f"{name}: all done already ({len(cache)} cached).")
        return
    print(f"{name}: processing {len(items)}...")
    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                cache[item["id"]] = future.result()
                done += 1
                save_json(cache_path, cache)
            except Exception as error:
                failed += 1
                print(f"  [FAIL] {item['id']}: {type(error).__name__}: {str(error)[:200]}")
            if (done + failed) % 20 == 0:
                print(f"  {done + failed}/{len(items)}")
    print(f"{name}: {done} done, {failed} failed.")


def load_jobs():
    fields = ["id", "title", "text", "sector", "municipality", "source_url", "taxonomy_ids", "vector"]
    jobs = list(search_client("jobs").search(search_text="*", top=1000, select=fields))
    print(f"Loaded {len(jobs)} job ads from AI Search.")
    return [dict(job) for job in jobs]


def clean_list(terms):
    out, seen = [], set()
    for term in terms:
        for part in clean_term(term):
            if part.lower() not in seen:
                seen.add(part.lower())
                out.append(part)
    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare data for the matching app")
    parser.add_argument("--limit", type=int, help="only process this many new jobs and candidates")
    parser.add_argument("--workers", type=int, default=8, help="parallel model calls")
    args = parser.parse_args()

    for needed in ["candidates_ready.json", "candidate_vectors.npy", "taxonomy_meta.json"]:
        if not os.path.exists(needed):
            raise SystemExit(f"{needed} not found: run build_index.py (Step 2) first, in this folder.")
    os.makedirs(DATA_DIR, exist_ok=True)

    client = openai_client()
    jobs = load_jobs()
    taxonomy = {c["id"]: c for c in load_json("taxonomy_meta.json")}
    candidates = load_json("candidates_ready.json")

    # 1 + 2. Model extraction (cached, so re-running only does what is missing)
    job_reqs = load_json(JOB_REQ_CACHE, {})
    todo = [j for j in jobs if j["id"] not in job_reqs][: args.limit]
    run_model_calls("Job requirements", todo,
                    lambda job: extract_job(client, job["title"], job["text"]),
                    job_reqs, JOB_REQ_CACHE, args.workers)

    prefs = load_json(PREFS_CACHE, {})
    todo = [c for c in candidates if c["id"] not in prefs][: args.limit]
    run_model_calls("Candidate conditions", todo,
                    lambda c: extract_candidate_prefs(client, c["background_text"]),
                    prefs, PREFS_CACHE, args.workers)

    # 3. Build the app's job list
    app_jobs = []
    for job in jobs:
        requirements = job_reqs.get(job["id"])
        if requirements:
            requirements = dict(requirements,
                                must_have=clean_list(requirements["must_have"]),
                                nice_to_have=clean_list(requirements["nice_to_have"]))
        labels = [taxonomy[cid]["label"] for cid in (job.get("taxonomy_ids") or [])
                  if cid in taxonomy and taxonomy[cid]["type"] in JOB_CONCEPT_TYPES]
        app_jobs.append({
            "id": job["id"], "title": job["title"], "text": job["text"],
            "sector": job.get("sector"), "municipality": job.get("municipality"),
            "source_url": job.get("source_url"),
            "taxonomy_ids": job.get("taxonomy_ids") or [], "taxonomy_labels": labels,
            "requirements": requirements,
        })

    # Job vectors: English profile text where we have requirements, else the
    # index's own vector of the Swedish ad (or the ad text, if not readable).
    job_vectors = np.zeros((len(jobs), EMBEDDING_DIMENSIONS), dtype=np.float32)
    to_embed = []
    for i, job in enumerate(jobs):
        if app_jobs[i]["requirements"] or not job.get("vector"):
            to_embed.append(i)
        else:
            job_vectors[i] = job["vector"]
    if to_embed:
        print(f"Embedding {len(to_embed)} job profiles...")
        embedded = embed_texts(client, [job_profile_text(app_jobs[i]) for i in to_embed])
        for row, i in enumerate(to_embed):
            job_vectors[i] = embedded[row]

    # 4. Build the app's candidate list
    app_candidates = []
    for doc in candidates:
        profile = json.loads(doc["profile_json"])
        app_candidates.append({
            "id": doc["id"], "background_text": doc["background_text"],
            "profile": profile, "prefs": prefs.get(doc["id"]),
            "terms": candidate_terms(profile),
        })

    # 5. Embed every distinct term (candidate terms + job needs) once
    all_terms, seen = [], set()
    for term in ([t["term"] for c in app_candidates for t in c["terms"]]
                 + [t for j in app_jobs if j["requirements"] for t in j["requirements"]["must_have"]]
                 + [t for j in app_jobs if j["requirements"] for t in j["requirements"]["nice_to_have"]]
                 + [t for j in app_jobs for t in j["taxonomy_labels"]]):
        if term.lower() not in seen:
            seen.add(term.lower())
            all_terms.append(term)
    print(f"Embedding {len(all_terms)} distinct terms...")
    term_vectors = embed_texts(client, all_terms)

    # 6. Save
    save_json(os.path.join(DATA_DIR, "jobs.json"), app_jobs)
    np.save(os.path.join(DATA_DIR, "job_vectors.npy"), job_vectors)
    save_json(os.path.join(DATA_DIR, "candidates.json"), app_candidates)
    shutil.copyfile("candidate_vectors.npy", os.path.join(DATA_DIR, "candidate_vectors.npy"))
    save_json(os.path.join(DATA_DIR, "terms.json"), all_terms)
    np.save(os.path.join(DATA_DIR, "term_vectors.npy"), term_vectors)

    # Summary
    reqs = [j["requirements"] for j in app_jobs if j["requirements"]]
    known = [p for p in prefs.values()]
    print(f"\nSaved app_data/: {len(app_jobs)} jobs ({len(reqs)} with requirements), "
          f"{len(app_candidates)} candidates, {len(all_terms)} terms.")
    if reqs:
        print(f"Jobs with sales targets: {sum(r['sales_targets'] for r in reqs)}, "
              f"part-time: {sum(r['hours'] == 'part-time' for r in reqs)}, "
              f"offer training: {sum(r['training_provided'] for r in reqs)}")
        example = next(j for j in app_jobs if j["requirements"])
        print(f"\nExample job: {example['title']}")
        print(json.dumps(example["requirements"], indent=2, ensure_ascii=False))
    if known:
        print(f"\nCandidates who avoid sales targets: {sum(p['avoids_sales_targets'] for p in known)}, "
              f"prefer daytime: {sum(p['prefers_daytime'] for p in known)}, "
              f"want part-time: {sum(p['hours'] == 'part-time' for p in known)}, "
              f"want training: {sum(p['wants_training'] for p in known)}")


if __name__ == "__main__":
    main()
