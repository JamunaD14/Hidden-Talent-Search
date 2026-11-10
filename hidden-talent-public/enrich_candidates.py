"""
Step 1 of the pipeline: turn each candidate's free-text story into a
structured profile that employers can search.

Input : Cosmos DB  hidden-talent / candidates  (field: background_text)
Output: enriched_candidates.json  (one entry per candidate, in this folder)

Run (Cloud Shell or laptop, with common.py in the same folder):
  python3 enrich_candidates.py --limit 3     # test on 3 candidates first
  python3 enrich_candidates.py               # then all candidates

Re-running skips candidates already in the output file, so you can stop and
continue at any time. The original Cosmos DB data is never changed.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.privacy import anonymise
from app.profile import PROFILE_SCHEMA, SYSTEM_PROMPT  # noqa: F401 (shared with the app)
from app.profile import extract_profile as read_profile
from common import cosmos_container, openai_client

OUTPUT_FILE = "enriched_candidates.json"

# --- Steps ------------------------------------------------------------------
def load_candidates():
    """Reads candidates and removes personal details before any model sees them.
    From here on, every step (profiles, index, app) uses the cleaned text."""
    container = cosmos_container("candidates")
    query = "SELECT c.id, c.language, c.background_text FROM c"
    candidates = list(container.query_items(query, enable_cross_partition_query=True))
    for candidate in candidates:
        candidate["background_text"], candidate["removed_details"] = anonymise(candidate["background_text"])
    cleaned = sum(bool(c["removed_details"]) for c in candidates)
    print(f"Anonymiser removed personal details from {cleaned} of {len(candidates)} candidates.")
    return candidates


def extract_profile(client, candidate):
    """Profile for one candidate dict (its background_text is already anonymised)."""
    return read_profile(client, candidate["background_text"])


def load_existing():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, encoding="utf-8") as fh:
            return {item["id"]: item for item in json.load(fh)}
    return {}


def save(results):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(list(results.values()), fh, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Enrich candidate profiles")
    parser.add_argument("--limit", type=int, help="only process this many new candidates")
    parser.add_argument("--workers", type=int, default=5, help="parallel model calls")
    args = parser.parse_args()

    results = load_existing()
    candidates = load_candidates()
    todo = [c for c in candidates if c["id"] not in results]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(candidates)} candidates in Cosmos DB, {len(results)} already done, "
          f"processing {len(todo)} now.\n")

    client = openai_client()
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_profile, client, c): c for c in todo}
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                profile = future.result()
            except Exception as error:
                failed += 1
                print(f"[FAIL] {candidate['id']}: {type(error).__name__}: {str(error)[:300]}")
                continue
            results[candidate["id"]] = {
                "id": candidate["id"],
                "language": candidate.get("language"),
                "background_text": candidate["background_text"],  # anonymised
                "removed_details": candidate["removed_details"],
                "profile": profile,
            }
            save(results)  # save after each one so no work is lost
            roles = ", ".join(profile["possible_roles"][:3])
            print(f"[OK]   {candidate['id']}: {len(profile['skills'])} skills | {roles}")

    # Summary + evidence-quality number for your evaluation
    skills = [s for item in results.values() for s in item["profile"]["skills"]]
    verified = sum(s["evidence_verified"] for s in skills)
    print(f"\nDone. {len(results)} profiles in {OUTPUT_FILE}, {failed} failed.")
    if skills:
        print(f"Evidence quotes found word for word in the text: "
              f"{verified}/{len(skills)} ({100 * verified / len(skills):.0f}%)")

    if args.limit and results:
        print("\nExample profile:")
        example = next(iter(results.values()))
        print(json.dumps(example["profile"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
