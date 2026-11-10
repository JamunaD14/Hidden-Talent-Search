"""
Step 5: test whether the matching works, and get the numbers for the pitch.

Tests:
  1. Hidden talent across all job ads: how many of our top 10 a keyword search misses
  2. Same search twice gives the same ranking
  3. The Azure AI Search step keeps the best candidates (retrieval check)
  4. Relevance: an independent judge (a separate model call that sees only the
     original job ad and the candidate's own words, not our scores or profiles)
     grades the top 5 of: our tool, keyword search, skills only, meaning only
  5. Fairness: a name, age and family situation are added to a candidate's text.
     Sent straight to the model, they show the risk; after the anonymiser
     (app/privacy.py, used by the real pipeline) the score should move no more
     than normal model variation, and no details should reach the profile

Run (from the project folder, after prepare_matching.py):
  python3 evaluate.py                             # everything, about 5-10 minutes
  python3 evaluate.py --judge-jobs 0 --fairness 0 # quick tests only, no model calls
  python3 evaluate.py --workers 1                 # if you get rate-limit (429) errors

Output: eval_results.json      all numbers
        eval_judgements.csv    every judged pair; fill in 'team_grade' (0, 1 or 2)
                               for some rows and re-run to check the judge agrees
                               with your team. Your grades are kept on re-runs.
"""

import argparse
import csv
import json
import os
import random
import statistics
import types
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from app import main as appmain  # loads app_data/ and the Azure clients
from app.extraction import call_structured, extract_candidate_prefs
from app.matching import DEFAULT_WEIGHTS, job_requirements, score_candidates
from app.privacy import anonymise
from build_index import profile_text
from enrich_candidates import extract_profile
from prepare_matching import candidate_terms

store, client, embed = appmain.store, appmain.client, appmain.embed
RESULTS_FILE = "eval_results.json"
JUDGE_CSV = "eval_judgements.csv"
JUDGE_CACHE = "eval_judge_cache.json"

SYSTEMS = {
    "Our tool (all criteria)": None,  # default weights
    "Skills only": {"skills": 100, "semantic": 0, "preferences": 0, "evidence": 0},
    "Meaning only": {"skills": 0, "semantic": 100, "preferences": 0, "evidence": 0},
}


def load_json(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return default


def pct(x):
    return f"{100 * x:.0f}%"


# --- Tests 1-3: rank every job ----------------------------------------------
def rank_job(job, weights=None):
    vector = store.job_vector(job["id"])
    weights = weights or DEFAULT_WEIGHTS
    app_results, baseline, retrieval = appmain.run_match(job, vector, weights, False, 10)
    full_results, _, _ = appmain.run_match(job, vector, weights, False, 10,
                                           with_baseline=False, use_retrieval=False)
    return {
        "app": [r["candidate_id"] for r in app_results],
        "full": [r["candidate_id"] for r in full_results],
        "full_scores": {r["candidate_id"]: r["total"] for r in full_results},
        "keyword": baseline["ids"],
        "retrieval": retrieval,
    }


def test_all_jobs(workers):
    print(f"\n[1-3] Ranking all {len(store.jobs_list)} job ads...")
    ranked = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(rank_job, job): job["id"] for job in store.jobs_list}
        for n, future in enumerate(as_completed(futures), start=1):
            ranked[futures[future]] = future.result()
            if n % 50 == 0:
                print(f"  {n}/{len(futures)}")

    hidden, recall = [], []
    for r in ranked.values():
        if r["keyword"] is not None:
            keyword_top = set(r["keyword"][:10])
            hidden.append(sum(c not in keyword_top for c in r["app"][:10]))
        if r["retrieval"]["retrieved"] is not None and r["full"]:
            best = set(r["full"][:10])
            recall.append(len(set(r["app"][:10]) & best) / len(best))

    out = {"jobs": len(ranked)}
    if hidden:
        out.update({
            "hidden_in_top10_avg": round(statistics.mean(hidden), 1),
            "hidden_in_top10_median": statistics.median(hidden),
            "jobs_with_5plus_hidden": sum(h >= 5 for h in hidden),
        })
        print(f"  Hidden talent: on average {out['hidden_in_top10_avg']} of our top 10 are not in "
              f"a keyword search's top 10 (median {out['hidden_in_top10_median']}; "
              f"{out['jobs_with_5plus_hidden']} of {len(hidden)} jobs have 5 or more).")
    else:
        print("  Keyword search unavailable: no hidden-talent numbers.")
    if recall:
        out["retrieval_recall_at10"] = round(statistics.mean(recall), 3)
        print(f"  Retrieval check: the AI Search step keeps {pct(out['retrieval_recall_at10'])} of the "
              f"candidates that ranking everyone would put in the top 10.")
    else:
        print("  AI Search hybrid step unavailable: all candidates were ranked directly.")

    # Test 2: same search twice
    same = all(rank_job(store.jobs[job_id])["app"][:10] == ranked[job_id]["app"][:10]
               for job_id in list(ranked)[:5])
    out["deterministic"] = same
    print(f"  Same search twice, same ranking: {'yes' if same else 'NO, check the logs'}")
    return ranked, out


# --- Test 4: independent relevance judge -------------------------------------
JUDGE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["grades"],
    "properties": {"grades": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["candidate_id", "grade", "reason"],
        "properties": {
            "candidate_id": {"type": "string"},
            "grade": {"type": "integer", "enum": [0, 1, 2]},
            "reason": {"type": "string"},
        }}}},
}

JUDGE_PROMPT = """You are an experienced recruiter in Sweden. For one job ad, grade how well each candidate could do the job.

Use only the job ad and the candidate's own words.
- 2 = strong fit: they have done most of the core tasks of this job, in any setting (paid work, volunteering, studies and hobbies all count).
- 1 = possible fit: they have done some core tasks or closely related ones, and could learn the rest.
- 0 = poor fit: little overlap with the core tasks.
If the candidate says they do not want something central to the job (for example sales targets, weekends or full-time), lower the grade by one.
Ignore how well they write. Never consider names, age, gender or background.
reason: one short sentence."""


def judge(job, candidate_ids):
    candidates = [c for c in store.candidates if c["id"] in candidate_ids]
    random.Random(job["id"]).shuffle(candidates)  # no order hints
    payload = {"job_title": job["title"], "job_ad": job["text"][:3500],
               "candidates": [{"candidate_id": c["id"], "text": c["background_text"]}
                              for c in candidates]}
    result = call_structured(client, JUDGE_PROMPT, json.dumps(payload, ensure_ascii=False),
                             "relevance_grades", JUDGE_SCHEMA, max_tokens=8000)
    return {g["candidate_id"]: {"grade": g["grade"], "reason": g["reason"]}
            for g in result["grades"] if g["candidate_id"] in candidate_ids}


def ndcg(grades, ideal_grades, k=5):
    def dcg(values):
        return sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(values[:k]))
    best = dcg(sorted(ideal_grades, reverse=True))
    return dcg(grades) / best if best else 0.0


def test_relevance(ranked, n_jobs, workers):
    jobs = random.Random(42).sample(store.jobs_list, min(n_jobs, len(store.jobs_list)))
    print(f"\n[4] Relevance check on {len(jobs)} job ads (independent judge)...")

    rankings = {}
    for job in jobs:
        lists = {}
        for name, weights in SYSTEMS.items():
            lists[name] = ranked[job["id"]]["app"] if weights is None else rank_job(job, weights)["app"]
        if ranked[job["id"]]["keyword"] is not None:
            lists["Keyword search"] = ranked[job["id"]]["keyword"]
        rankings[job["id"]] = lists

    cache = load_json(JUDGE_CACHE, {})
    todo = []
    for job in jobs:
        wanted = {c for lst in rankings[job["id"]].values() for c in lst[:5]}
        missing = wanted - set(cache.get(job["id"], {}))
        if missing:
            todo.append((job, missing))
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(judge, job, ids): job for job, ids in todo}
        for future in as_completed(futures):
            job = futures[future]
            try:
                cache.setdefault(job["id"], {}).update(future.result())
                with open(JUDGE_CACHE, "w", encoding="utf-8") as fh:
                    json.dump(cache, fh, ensure_ascii=False)
            except Exception as error:
                failed += 1
                print(f"  [FAIL] {job['id']}: {type(error).__name__}: {str(error)[:200]}")
    if failed:
        print(f"  {failed} jobs failed; re-run to fill them in (results so far are kept).")

    # Metrics per system
    per_system = {name: {"good": [], "strong": [], "avg": [], "ndcg": []}
                  for name in list(SYSTEMS) + ["Keyword search"]}
    hidden_good = []
    for job in jobs:
        grades = cache.get(job["id"], {})
        if not grades:
            continue
        lists = rankings[job["id"]]
        keyword_top10 = set(ranked[job["id"]]["keyword"] or [])
        for name, lst in lists.items():
            g = [grades[c]["grade"] for c in lst[:5] if c in grades]
            if len(g) < min(5, len(lst)):
                continue  # judge skipped a candidate: leave this job out for this system
            # Always out of 5 slots: a list with fewer than 5 candidates gets no credit
            # for the empty slots (standard precision@5).
            m = per_system[name]
            m["good"].append(sum(x >= 1 for x in g) / 5)
            m["strong"].append(sum(x == 2 for x in g) / 5)
            m["avg"].append(sum(g) / 5)
            m["ndcg"].append(ndcg(g, [v["grade"] for v in grades.values()]))
        if ranked[job["id"]]["keyword"] is not None:
            hidden_good += [grades[c]["grade"] >= 1 for c in lists["Our tool (all criteria)"][:5]
                            if c not in keyword_top10 and c in grades]

    table = {}
    print(f"\n  {'Ranking':<26} {'Good fit in top 5':>18} {'Strong fit':>11} {'Avg grade (0-2)':>16} {'nDCG@5':>8}")
    for name, m in per_system.items():
        if not m["good"]:
            continue
        table[name] = {"good_fit_top5": round(float(np.mean(m["good"])), 3),
                       "strong_fit_top5": round(float(np.mean(m["strong"])), 3),
                       "avg_grade": round(float(np.mean(m["avg"])), 2),
                       "ndcg_at5": round(float(np.mean(m["ndcg"])), 3), "jobs": len(m["good"])}
        t = table[name]
        print(f"  {name:<26} {pct(t['good_fit_top5']):>18} {pct(t['strong_fit_top5']):>11} "
              f"{t['avg_grade']:>16} {t['ndcg_at5']:>8}")
    out = {"jobs_judged": len([j for j in jobs if cache.get(j["id"])]), "systems": table}
    if hidden_good:
        out["hidden_talent_good_fit"] = round(float(np.mean(hidden_good)), 3)
        out["hidden_talent_judged"] = len(hidden_good)
        print(f"\n  Of the {len(hidden_good)} hidden-talent candidates in our top 5s, "
              f"{pct(out['hidden_talent_good_fit'])} were graded a good fit by the judge.")

    write_judgement_csv(jobs, rankings, cache, ranked)
    out["team_agreement"] = team_agreement()
    return out


def write_judgement_csv(jobs, rankings, cache, ranked):
    team = {}
    if os.path.exists(JUDGE_CSV):
        with open(JUDGE_CSV, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("team_grade", "").strip():
                    team[(row["job_id"], row["candidate_id"])] = row["team_grade"].strip()
    candidates = {c["id"]: c for c in store.candidates}
    fields = ["job_id", "job_title", "candidate_id", "judge_grade", "team_grade", "judge_reason",
              "hidden_talent", "rank_our_tool", "rank_keyword", "rank_skills_only",
              "rank_meaning_only", "candidate_text", "job_ad_start"]
    with open(JUDGE_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            lists = rankings[job["id"]]
            keyword_top10 = set(ranked[job["id"]]["keyword"] or [])

            def rank_in(name, cid):
                lst = lists.get(name) or []
                return lst.index(cid) + 1 if cid in lst else ""
            for cid, g in sorted(cache.get(job["id"], {}).items()):
                writer.writerow({
                    "job_id": job["id"], "job_title": job["title"], "candidate_id": cid,
                    "judge_grade": g["grade"], "team_grade": team.get((job["id"], cid), ""),
                    "judge_reason": g["reason"],
                    "hidden_talent": cid in lists["Our tool (all criteria)"][:10] and cid not in keyword_top10,
                    "rank_our_tool": rank_in("Our tool (all criteria)", cid),
                    "rank_keyword": rank_in("Keyword search", cid),
                    "rank_skills_only": rank_in("Skills only", cid),
                    "rank_meaning_only": rank_in("Meaning only", cid),
                    "candidate_text": candidates[cid]["background_text"],
                    "job_ad_start": job["text"][:300].replace("\n", " "),
                })
    print(f"  Wrote {JUDGE_CSV}: fill in 'team_grade' for some rows to check the judge.")


def team_agreement():
    if not os.path.exists(JUDGE_CSV):
        return None
    with open(JUDGE_CSV, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("team_grade", "").strip() in {"0", "1", "2"}]
    if not rows:
        return None
    exact = np.mean([int(r["team_grade"]) == int(r["judge_grade"]) for r in rows])
    close = np.mean([abs(int(r["team_grade"]) - int(r["judge_grade"])) <= 1 for r in rows])
    print(f"  Team vs judge on {len(rows)} pairs: same grade {pct(exact)}, within one grade {pct(close)}.")
    return {"pairs": len(rows), "exact": round(float(exact), 3), "within_one": round(float(close), 3)}


# --- Test 5: fairness ---------------------------------------------------------
IDENTITIES = [
    ("Anna Lindqvist", "26", "My name is Anna Lindqvist and I am 26 years old."),
    ("Mohammed Hassan", "58", "My name is Mohammed Hassan and I am 58 years old."),
    ("Amina Yusuf", "41", "My name is Amina Yusuf, I am 41 and a mother of two."),
]


def build_variant(variant_id, text):
    profile = extract_profile(client, {"background_text": text})
    prefs = extract_candidate_prefs(client, text)
    terms = candidate_terms(profile)
    vector = embed([profile_text(profile)])[0]
    term_vectors = embed([t["term"] for t in terms]) if terms else np.zeros((0, vector.shape[0]))
    return {"id": variant_id, "background_text": text, "profile": profile, "prefs": prefs,
            "terms": terms, "vector": vector, "term_vectors": term_vectors}


def score_with_variants(job, variants):
    """Scores the variants as if they were in the pool, next to the 100 real candidates."""
    term_matrix = np.vstack([store.term_vectors] + [v["term_vectors"] for v in variants])
    offset, extra = len(store.term_vectors), []
    for v in variants:
        rows = np.arange(offset, offset + len(v["terms"]), dtype=int)
        offset += len(v["terms"])
        extra.append({"id": v["id"], "background_text": v["background_text"], "profile": v["profile"],
                      "prefs": v["prefs"], "terms": v["terms"], "_rows": rows})
    pool = types.SimpleNamespace(
        candidates=list(store.candidates) + extra,
        candidate_vectors=np.vstack([store.candidate_vectors] + [v["vector"][None, :] for v in variants]),
        term_vectors=term_matrix)
    requirements = job_requirements(job)
    requirement_vectors = store.vectors_for([r[0] for r in requirements], embed)
    results = score_candidates(pool, job, store.job_vector(job["id"]), requirements,
                               requirement_vectors, DEFAULT_WEIGHTS)
    real_scores = [r["total"] for r in results if not r["candidate_id"].startswith("variant:")]
    return {r["candidate_id"]: (r["total"], 1 + sum(s > r["total"] for s in real_scores))
            for r in results if r["candidate_id"].startswith("variant:")}


def find_leaks(profile, name, age):
    blob = json.dumps(profile, ensure_ascii=False).lower()
    return [w for w in name.lower().split() + [age, "years old", "mother"] if w in blob]


def test_fairness(ranked, n_candidates, workers):
    pool_ids = [c["id"] for c in store.candidates]
    step = max(1, len(pool_ids) // n_candidates)
    chosen = pool_ids[::step][:n_candidates]
    print(f"\n[5] Fairness check on {len(chosen)} candidates ({', '.join(chosen)})...")
    print("    Each candidate's text goes through the full pipeline 9 times: 3 times unchanged\n"
          "    (normal variation), 3 times with a name/age/family sentence sent straight to the\n"
          "    model, and 3 times with that sentence after the anonymiser (how the pipeline works).")

    by_id = {c["id"]: c for c in store.candidates}
    tasks = []
    for cid in chosen:
        best_job = max(ranked, key=lambda j: ranked[j]["full_scores"].get(cid, -1))
        text = by_id[cid]["background_text"]
        texts = [(f"control {i}", text) for i in (1, 2, 3)]
        for name, _, intro in IDENTITIES:
            raw = f"{intro} {text}"
            texts.append((f"raw {name}", raw))
            texts.append((f"anon {name}", anonymise(raw)[0]))
        tasks.append((cid, best_job, texts))

    built = {}
    jobs_to_run = [(cid, label, text) for cid, _, texts in tasks for label, text in texts]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(build_variant, f"variant:{cid}:{label}", text): (cid, label)
                   for cid, label, text in jobs_to_run}
        for future in as_completed(futures):
            key = futures[future]
            try:
                built[key] = future.result()
            except Exception as error:
                print(f"  [FAIL] {key}: {type(error).__name__}: {str(error)[:200]}")

    rows, leaks = [], {"raw": [], "anon": []}
    for cid, job_id, texts in tasks:
        variants = [built[(cid, label)] for label, _ in texts if (cid, label) in built]
        if len(variants) < len(texts):
            print(f"  {cid}: skipped (some model calls failed; re-run)")
            continue
        scores = score_with_variants(store.jobs[job_id], variants)
        original = ranked[job_id]["full_scores"][cid]
        controls = [scores[f"variant:{cid}:control {i}"][0] for i in (1, 2, 3)]
        raw = {n: scores[f"variant:{cid}:raw {n}"][0] for n, _, _ in IDENTITIES}
        anon = {n: scores[f"variant:{cid}:anon {n}"][0] for n, _, _ in IDENTITIES}
        centre = statistics.mean([original] + controls)
        for mode in ("raw", "anon"):
            for name, age, _ in IDENTITIES:
                found = find_leaks(built[(cid, f"{mode} {name}")]["profile"], name, age)
                if found:
                    leaks[mode].append({"candidate": cid, "identity": name, "found_in_profile": found})
        rows.append({
            "candidate": cid, "job_id": job_id, "stored_score": original, "control_scores": controls,
            "noise": round(max(abs(x - centre) for x in [original] + controls), 1),
            "raw_scores": raw, "raw_shift": round(max(abs(x - centre) for x in raw.values()), 1),
            "anon_scores": anon, "anon_shift": round(max(abs(x - centre) for x in anon.values()), 1),
        })

    if not rows:
        return {"candidates": []}
    noise = max(r["noise"] for r in rows)  # normal variation, pooled over all candidates
    print(f"\n  {'Candidate':<14} {'Job':<30} {'Unchanged x3':>15} {'Sent raw':>15} {'Anonymised':>15}  Verdict")
    for r in rows:
        r["verdict"] = "OK" if r["anon_shift"] <= noise + 2 else "CHECK"
        fmt = lambda xs: " / ".join(f"{x:.0f}" for x in xs)
        print(f"  {r['candidate']:<14} {store.jobs[r['job_id']]['title'][:29]:<30} "
              f"{fmt(r['control_scores']):>15} {fmt(r['raw_scores'].values()):>15} "
              f"{fmt(r['anon_scores'].values()):>15}  {r['verdict']}")

    summary = {"candidates": rows, "max_model_noise": noise,
               "max_raw_shift": max(r["raw_shift"] for r in rows),
               "max_anon_shift": max(r["anon_shift"] for r in rows),
               "leaks_raw": len(leaks["raw"]), "leaks_anonymised": leaks["anon"]}
    print(f"\n  Normal variation (same text, run again): up to {noise:.1f} points.")
    print(f"  Name/age/family sent straight to the model: score moved up to {summary['max_raw_shift']:.1f} points; "
          f"details copied into {len(leaks['raw'])} of {len(rows) * len(IDENTITIES)} profiles.")
    print(f"  After the anonymiser: score moved up to {summary['max_anon_shift']:.1f} points; "
          f"details copied into {len(leaks['anon'])} profiles.")
    return summary


# --- Main ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate the Hidden Talent matching")
    parser.add_argument("--judge-jobs", type=int, default=15, help="job ads for the relevance check (0 = skip)")
    parser.add_argument("--fairness", type=int, default=4, help="candidates for the fairness check (0 = skip)")
    parser.add_argument("--workers", type=int, default=2, help="parallel model calls")
    parser.add_argument("--search-workers", type=int, default=4, help="parallel AI Search calls")
    args = parser.parse_args()

    results = {}
    verified = [s.get("evidence_verified", False) for c in store.candidates for s in c["profile"]["skills"]]
    results["evidence_verified"] = round(float(np.mean(verified)), 4) if verified else None
    print(f"[0] Skills backed by a word-for-word quote from the candidate: "
          f"{sum(verified)}/{len(verified)} ({pct(results['evidence_verified'] or 0)})")

    ranked, results["all_jobs"] = test_all_jobs(args.search_workers)
    if args.judge_jobs > 0:
        results["relevance"] = test_relevance(ranked, args.judge_jobs, args.workers)
    if args.fairness > 0:
        results["fairness"] = test_fairness(ranked, args.fairness, args.workers)

    with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    # Numbers for the pitch
    print("\n=== Numbers for your pitch ===")
    a = results["all_jobs"]
    if "hidden_in_top10_avg" in a:
        print(f"- Across {a['jobs']} real Swedish job ads, on average {a['hidden_in_top10_avg']} of our top 10 "
              f"candidates are not in a keyword search's top 10.")
    rel = results.get("relevance", {}).get("systems", {})
    ours, keyword = rel.get("Our tool (all criteria)"), rel.get("Keyword search")
    if ours and keyword:
        print(f"- An independent check rated {pct(ours['good_fit_top5'])} of our top 5 a good fit, "
              f"against {pct(keyword['good_fit_top5'])} for keyword search.")
    if "hidden_talent_good_fit" in results.get("relevance", {}):
        print(f"- {pct(results['relevance']['hidden_talent_good_fit'])} of the hidden candidates we surface "
              f"were rated a good fit.")
    if results.get("fairness", {}).get("candidates"):
        f = results["fairness"]
        print(f"- Sent straight to the model, a name, age or family detail moved a score by up to "
              f"{f['max_raw_shift']} points. Our anonymiser removes them first: then at most "
              f"{f['max_anon_shift']} points (normal variation: {f['max_model_noise']}).")
    if "retrieval_recall_at10" in a:
        print(f"- The Azure AI Search step keeps {pct(a['retrieval_recall_at10'])} of the best candidates "
              f"while passing on only {appmain.RETRIEVE_K} of {len(store.candidates)} for detailed ranking.")
    print(f"- {pct(results['evidence_verified'] or 0)} of extracted skills are backed by the candidate's own words.")
    print(f"- Same search, same ranking: {'yes' if a['deterministic'] else 'no'}.")
    print(f"\nSaved {RESULTS_FILE}.")


if __name__ == "__main__":
    main()
