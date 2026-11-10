"""
Step 2 of the pipeline: make the candidates searchable.

What it does:
  1. Downloads the 1,248 JobTech taxonomy concepts (once, then cached).
  2. Maps each candidate's skills, roles and Swedish search terms to taxonomy
     concepts, so candidates and job ads share one "skills language".
  3. Embeds each candidate profile (1536 dimensions).
  4. Creates the AI Search index 'candidates' and uploads all candidates.
  5. Saves local copies the app and the evaluation can use.

Input : enriched_candidates.json   (from Step 1: enrich_candidates.py)
Output: AI Search index 'candidates'
        candidates_ready.json, candidate_vectors.npy   (local copy)
        taxonomy_meta.json, taxonomy_vectors.npy       (taxonomy cache)

Run (with common.py in the same folder):
  python3 build_index.py               # normal run
  python3 build_index.py --recreate    # delete and rebuild the index
  python3 build_index.py --threshold 0.6   # stricter taxonomy mapping
"""

import argparse
import json
import os
import re
from collections import Counter

import numpy as np

from app.profile import profile_text
from common import (
    EMBEDDING_DEPLOYMENT, EMBEDDING_DIMENSIONS, SEARCH_ENDPOINT,
    credential, openai_client, search_client,
)

ENRICHED_FILE = "enriched_candidates.json"
TAXONOMY_META = "taxonomy_meta.json"
TAXONOMY_VECTORS = "taxonomy_vectors.npy"
CANDIDATES_FILE = "candidates_ready.json"
CANDIDATE_VECTORS = "candidate_vectors.npy"
INDEX_NAME = os.getenv("CANDIDATE_INDEX", "candidates")


# --- Embeddings -------------------------------------------------------------
def normalise_rows(matrix):
    """Scale every vector to length 1, so a dot product = cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def embed(client, texts, batch_size=100):
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(model=EMBEDDING_DEPLOYMENT, input=batch)
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
    return normalise_rows(np.array(vectors, dtype=np.float32))


# --- 1. Taxonomy ------------------------------------------------------------
def load_taxonomy(client):
    if os.path.exists(TAXONOMY_META) and os.path.exists(TAXONOMY_VECTORS):
        with open(TAXONOMY_META, encoding="utf-8") as fh:
            meta = json.load(fh)
        print(f"Taxonomy: {len(meta)} concepts (from local cache)")
        return meta, np.load(TAXONOMY_VECTORS)

    print("Downloading taxonomy concepts from AI Search...")
    taxonomy = search_client("taxonomy")
    expected = taxonomy.get_document_count()
    docs, skip = {}, 0
    while True:
        page = list(taxonomy.search(
            search_text="*", top=1000, skip=skip,
            select=["id", "preferred_label", "type", "alternative_labels", "vector"],
        ))
        for doc in page:
            docs[doc["id"]] = doc
        if len(page) < 1000:
            break
        skip += 1000
    if len(docs) < expected:
        print(f"  Warning: got {len(docs)} of {expected} concepts.")

    meta = [{
        "id": d["id"],
        "label": d["preferred_label"],
        "type": d.get("type"),
        "alternative_labels": d.get("alternative_labels") or [],
    } for d in docs.values()]

    raw_vectors = [d.get("vector") for d in docs.values()]
    if any(v is None for v in raw_vectors):
        print("  Taxonomy vectors not readable: embedding the labels instead...")
        texts = [m["label"] + (" (" + ", ".join(m["alternative_labels"]) + ")"
                               if m["alternative_labels"] else "") for m in meta]
        vectors = embed(client, texts)
    else:
        vectors = normalise_rows(np.array(raw_vectors, dtype=np.float32))

    with open(TAXONOMY_META, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False)
    np.save(TAXONOMY_VECTORS, vectors)
    print(f"Taxonomy: {len(meta)} concepts, types: {dict(Counter(m['type'] for m in meta))}")
    return meta, vectors


# --- 2. Map candidate terms to taxonomy concepts ---------------------------
def clean_term(term):
    """Split and tidy names like 'Helpdesk / Service Desk Agent (junior)'
    into short terms: ['Helpdesk', 'Service Desk Agent']."""
    term = re.sub(r"\([^)]*\)", " ", term)           # drop "(entry-level)" etc.
    parts = re.split(r"/|\s[-–—]\s", term)            # split on "/" and " - "
    cleaned = [re.sub(r"\s+", " ", p).strip(" ,.;:-") for p in parts]
    return [p for p in cleaned if len(p) >= 2]


def candidate_terms(profile):
    """The words we try to translate into taxonomy concepts."""
    raw = [s["name"] for s in profile["skills"]]
    raw += profile["possible_roles"] + profile["search_terms_sv"]
    seen, unique = set(), []
    for term in (part for item in raw for part in clean_term(item)):
        key = term.lower()
        if key not in seen:
            seen.add(key)
            unique.append(term)
    return unique


def map_to_taxonomy(terms, term_vectors, tax_meta, tax_vectors, threshold):
    if not terms:
        return []
    similarities = term_vectors @ tax_vectors.T
    mapping = []
    for i, term in enumerate(terms):
        best = int(np.argmax(similarities[i]))
        score = float(similarities[i, best])
        if score >= threshold:
            concept = tax_meta[best]
            mapping.append({
                "term": term,
                "concept_id": concept["id"],
                "concept_label": concept["label"],
                "concept_type": concept["type"],
                "score": round(score, 3),
            })
    return mapping


# --- 3. Search index --------------------------------------------------------
def index_definition():
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration, SearchableField, SearchField,
        SearchFieldDataType as T, SearchIndex, SimpleField,
        VectorSearch, VectorSearchProfile,
    )

    fields = [
        SimpleField(name="id", type=T.String, key=True, filterable=True),
        SearchableField(name="summary", analyzer_name="en.lucene"),
        SearchableField(name="background_text", analyzer_name="en.lucene"),
        SearchableField(name="skills", collection=True, filterable=True,
                        facetable=True, analyzer_name="en.lucene"),
        SearchableField(name="possible_roles", collection=True, analyzer_name="en.lucene"),
        SearchableField(name="search_terms_sv", collection=True, analyzer_name="sv.lucene"),
        SearchableField(name="taxonomy_labels", collection=True, analyzer_name="sv.lucene"),
        SimpleField(name="taxonomy_ids", type=T.Collection(T.String), filterable=True),
        SimpleField(name="hours", type=T.String, filterable=True, facetable=True),
        SearchableField(name="wanted", collection=True),
        SearchableField(name="avoid", collection=True),
        SearchableField(name="open_to", collection=True),
        SearchableField(name="stated_gaps", collection=True),
        SimpleField(name="years_experience", type=T.Double, filterable=True),
        SimpleField(name="profile_json", type=T.String),       # full profile + evidence
        SimpleField(name="taxonomy_mapping", type=T.String),   # term -> concept, for the UI
        SearchField(
            name="vector",
            type=T.Collection(T.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="vector-profile",
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[VectorSearchProfile(name="vector-profile", algorithm_configuration_name="hnsw")],
    )
    return SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)


def upload_to_search(docs, vectors, recreate):
    from azure.core.exceptions import HttpResponseError
    from azure.search.documents.indexes import SearchIndexClient

    index_client = SearchIndexClient(SEARCH_ENDPOINT, credential)
    try:
        if recreate and INDEX_NAME in list(index_client.list_index_names()):
            index_client.delete_index(INDEX_NAME)
            print(f"Deleted old index '{INDEX_NAME}'.")
        index_client.create_or_update_index(index_definition())
        print(f"Index '{INDEX_NAME}' is ready.")
    except HttpResponseError as error:
        print(f"\nCould not create the index: {error.status_code} {error.message[:300]}")
        if error.status_code == 403:
            print("Your account may not create indexes. Ask the organisers for the "
                  "'Search Service Contributor' role, or keep going with the local files.")
        elif error.status_code == 400:
            print("The index exists with different fields. Re-run with --recreate.")
        return False

    client = search_client(INDEX_NAME)
    upload_docs = [dict(doc, vector=vector.tolist()) for doc, vector in zip(docs, vectors)]
    failed = []
    for start in range(0, len(upload_docs), 50):
        results = client.merge_or_upload_documents(upload_docs[start:start + 50])
        failed += [r.key for r in results if not r.succeeded]
    print(f"Uploaded {len(upload_docs) - len(failed)} candidates to '{INDEX_NAME}'"
          + (f", failed: {failed}" if failed else "."))
    return not failed


# --- Diagnostics: do candidate concepts appear in job ads? -----------------
def job_overlap(docs):
    jobs = list(search_client("jobs").search(search_text="*", top=1000,
                                             select=["id", "taxonomy_ids"]))
    job_concepts = {cid for job in jobs for cid in (job.get("taxonomy_ids") or [])}
    candidate_concepts = {cid for doc in docs for cid in doc["taxonomy_ids"]}
    shared = candidate_concepts & job_concepts
    print(f"\nJob ads use {len(job_concepts)} different concepts; candidates map to "
          f"{len(candidate_concepts)}; {len(shared)} appear on both sides.")
    if candidate_concepts and len(shared) / len(candidate_concepts) < 0.3:
        print("Few exact matches: the app should compare concepts by meaning "
              "(vector similarity), not only by identical IDs.")


# --- Main -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build the candidates search index")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="minimum similarity to map a term to a taxonomy concept")
    parser.add_argument("--recreate", action="store_true",
                        help="delete and rebuild the index")
    args = parser.parse_args()

    if not os.path.exists(ENRICHED_FILE):
        raise SystemExit(f"{ENRICHED_FILE} not found: run enrich_candidates.py first.")
    with open(ENRICHED_FILE, encoding="utf-8") as fh:
        candidates = json.load(fh)
    print(f"Loaded {len(candidates)} enriched candidates.")

    client = openai_client()
    tax_meta, tax_vectors = load_taxonomy(client)

    # Embed every distinct term once, then map each candidate's terms.
    all_terms = sorted({t for c in candidates for t in candidate_terms(c["profile"])})
    print(f"Embedding {len(all_terms)} distinct skill and role terms...")
    term_vectors = dict(zip(all_terms, embed(client, all_terms))) if all_terms else {}

    print("Embedding candidate profiles...")
    candidate_vectors = embed(client, [profile_text(c["profile"]) for c in candidates])

    docs = []
    for candidate in candidates:
        profile = candidate["profile"]
        terms = candidate_terms(profile)
        vectors = np.array([term_vectors[t] for t in terms]) if terms else np.zeros((0, EMBEDDING_DIMENSIONS))
        mapping = map_to_taxonomy(terms, vectors, tax_meta, tax_vectors, args.threshold)
        prefs = profile["work_preferences"]
        docs.append({
            "id": candidate["id"],
            "summary": profile["summary"],
            "background_text": candidate["background_text"],
            "skills": [s["name"] for s in profile["skills"]],
            "possible_roles": profile["possible_roles"],
            "search_terms_sv": profile["search_terms_sv"],
            "taxonomy_ids": sorted({m["concept_id"] for m in mapping}),
            "taxonomy_labels": sorted({m["concept_label"] for m in mapping}),
            "hours": prefs["hours"],
            "wanted": prefs["wanted"],
            "avoid": prefs["avoid"],
            "open_to": prefs["open_to"],
            "stated_gaps": profile["stated_gaps"],
            "years_experience": profile["years_experience"],
            "profile_json": json.dumps(profile, ensure_ascii=False),
            "taxonomy_mapping": json.dumps(mapping, ensure_ascii=False),
        })

    # Local copies (also the fallback if the index can't be created)
    with open(CANDIDATES_FILE, "w", encoding="utf-8") as fh:
        json.dump(docs, fh, indent=2, ensure_ascii=False)
    np.save(CANDIDATE_VECTORS, candidate_vectors)
    print(f"Saved {CANDIDATES_FILE} and {CANDIDATE_VECTORS}.")

    mapped = [len(d["taxonomy_ids"]) for d in docs]
    print(f"Taxonomy concepts per candidate: average {np.mean(mapped):.1f}, "
          f"min {min(mapped)}, max {max(mapped)} (threshold {args.threshold})")
    example = json.loads(docs[0]["taxonomy_mapping"])
    print(f"\nExample mapping for {docs[0]['id']}:")
    for m in example[:10]:
        print(f"  {m['term']:<35} -> {m['concept_label']} [{m['concept_type']}] ({m['score']})")

    upload_to_search(docs, candidate_vectors, args.recreate)
    job_overlap(docs)


if __name__ == "__main__":
    main()
