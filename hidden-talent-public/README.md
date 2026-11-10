# Hidden Talent Search

Finds candidates employers would miss with a keyword search, by translating
everyday experience into the skills language of job ads.

## Project layout

| File / folder | What it is | Status |
| --- | --- | --- |
| `common.py` | Shared settings and Azure sign-in | Ready |
| `smoke_test.py` | Step 0: checks access to every Azure resource | Ready |
| `enrich_candidates.py` | Step 1: free-text CV -> structured profile | Ready |
| `build_index.py` | Step 2: maps candidates to the taxonomy, creates and fills the `candidates` search index | Ready |
| `app/` | Step 3: web UI + API (runs in Azure Container Apps) | Later |
| `evaluate.py` | Step 4: ranking quality and fairness tests | Later |

## One-time setup (laptop, Linux)

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
az login
az account set --subscription "<team subscription>"
```

Every new terminal: `source .venv/bin/activate` first.

## Run the steps in order

```
python3 smoke_test.py                    # 0. all checks should say OK
python3 enrich_candidates.py --limit 3   # 1a. test on 3 candidates
python3 enrich_candidates.py             # 1b. all candidates
python3 build_index.py                   # 2. taxonomy mapping + candidates index
```

## What the data looks like

| Source | Key fields |
| --- | --- |
| Cosmos `candidates` (100) | `id`, `background_text` (English free text), partition key `/partition_key` |
| Search `jobs` (200) | `id`, `title`, `text` (Swedish), `sector`, `municipality`, `taxonomy_ids`, `vector` |
| Search `taxonomy` (1,248) | `id`, `preferred_label` (Swedish), `type`, `alternative_labels`, `vector` |
| Cosmos `matches` | empty; the app writes results here, partition key `/partition_key` |

## Azure resources (team t01)

| Resource | Name |
| --- | --- |
| Foundry project | `aispire-t01-ai-alvmb4wnkyzua` / `hidden-talent` (deployments: `chat`, `embedding`) |
| AI Search | `aispire-t01-search-alvmb4wnkyzua` (indexes: `jobs`, `taxonomy`) |
| Cosmos DB | `aispire-t01-cosmos-alvmb4wnkyzua` (database `hidden-talent`: `candidates`, `matches`) |
| Container Apps | environment `aispire-t01-env` |
| Identity / logs | `aispire-t01-app` / `aispire-t01-logs` |

No API keys: sign in with `az login` locally; the app uses the managed identity in Azure.
