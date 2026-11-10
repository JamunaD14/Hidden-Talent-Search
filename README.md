# Hidden-Talent-Search

An AI-powered tool on Azure that helps employers find the candidates a keyword search misses, by
translating people's everyday experience into the skills language of job ads.

Built in two days at the **AI:spire Hackathon** (Accenture and Microsoft), November 2026.

## The problem

Thousands of jobseekers have the right competencies, but they describe them in their own words:
"I organised the food bank's weekly deliveries" instead of "logistics coordination". Employers search
for job-ad words, so these people never show up.

## What it does

**For recruiters**
- Pick a real Swedish job ad (or describe who you need) and get a ranked list of candidates
- See the job's needs next to each candidate's own words, so you can check every match yourself
- Compare our top 10 with a keyword search's top 10: everyone the keyword search missed is marked
- Change the focus (skills, overall fit, the candidate's wishes), build a shortlist and export it as CSV
- Get a short explanation and an interview question for each candidate

**For jobseekers**
- Paste your story in your own words and see your skills in employer words, the Swedish search terms
  employers use, and the jobs you fit best, with what you already have and what to grow into

![The jobseeker view](docs/jobseeker.png)

## How it works

1. **Anonymise:** names, age, family, gender, origin, contact and ID details are removed before any AI
   model sees a candidate's text (`app/privacy.py`)
2. **Read profiles:** a model in Microsoft Foundry turns each story into skills, each backed by a quote
   from the candidate's own words (structured output; quotes are checked word for word)
3. **Map to the labour market's words:** skills are matched to the JobTech taxonomy, and job ads are read
   into must-haves, nice-to-haves and conditions (hours, schedule, sales targets)
4. **Find:** Azure AI Search hybrid search (keywords + vectors, plus filters) picks the 50 most promising
   candidates for a job
5. **Rank:** our code scores each one on skills (50%), overall fit (25%), their stated wishes (15%) and
   evidence (10%). Scores are computed in code, not by the model, so the same search always gives
   the same ranking
6. **Explain and store:** the top matches get an explanation and an interview question, and rankings and
   shortlists are saved in Cosmos DB

## Results

From our evaluation: an independent AI judge saw only the original job ad and each candidate's own words,
and graded the top 5 of each ranking on 40 random job ads (100 fictional candidates).

| | Hidden Talent Search | Keyword search |
| --- | --- | --- |
| Top 5 rated a strong fit | **40%** | 21.5% |
| Top 5 rated a good fit | **77.5%** | 61.5% |

- 83% of the hidden candidates we surfaced (people the keyword search missed) were rated good fits
- In 144 of 200 job ads, half our shortlist or more were people a keyword search missed

These numbers come from an AI judge, not from recruiters, so treat them as a first signal.

## Fairness

Our first fairness test found candidates' names leaking into 10 of 12 extracted profiles. We fixed it in
two layers (an anonymiser before any model call, and a rule in the model instructions) and tested
again: none of the 12 kept a personal detail. `evaluate.py` re-runs this test.

## Tech stack

| Part | Technology |
| --- | --- |
| AI models | Microsoft Foundry: gpt-5-mini (reading and explaining), text-embedding-3-small (1536 dimensions) |
| Search | Azure AI Search: hybrid keyword + vector search, filters |
| Storage | Azure Cosmos DB |
| Hosting | Azure Container Apps (Docker), managed identity, no API keys |
| App | Python, FastAPI, plain HTML/CSS/JavaScript |

## Project layout

| File | What it is |
| --- | --- |
| `common.py` | Settings and Azure sign-in |
| `smoke_test.py` | Step 0: checks access to every Azure resource |
| `enrich_candidates.py` | Step 1: anonymises each story and reads it into a profile |
| `build_index.py` | Step 2: maps profiles to the JobTech taxonomy and fills the search index |
| `prepare_matching.py` | Step 3: reads job needs and candidate wishes, embeds everything |
| `app/main.py` | Step 4: the web API and web page |
| `evaluate.py` | Step 5: evaluation (hidden talent, retrieval, AI judge, fairness) |
| `app/matching.py` | The scoring |
| `app/extraction.py`, `app/profile.py` | Model calls |
| `app/privacy.py` | The anonymiser |
| `app/static/index.html` | The web page |

## Running it yourself

The hackathon's data and Azure resources are **not included** in this repository. To run it you need:

- A Microsoft Foundry resource with a chat deployment and an embedding deployment
- Azure AI Search with a `jobs` index (job ads) and a `taxonomy` index (JobTech taxonomy concepts)
- Azure Cosmos DB with a `candidates` container (`id`, `language`, `background_text`) and a `matches` container

Point the code at your resources with environment variables, then run the steps in order:

```
export FOUNDRY_ENDPOINT=https://<your-foundry>.openai.azure.com/
export SEARCH_ENDPOINT=https://<your-search>.search.windows.net
export COSMOS_ENDPOINT=https://<your-cosmos>.documents.azure.com:443/

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
az login

python3 smoke_test.py          # 0. access checks
python3 enrich_candidates.py   # 1. profiles
python3 build_index.py         # 2. taxonomy + search index
python3 prepare_matching.py    # 3. job needs + vectors
uvicorn app.main:app --port 8000   # 4. open http://localhost:8000
python3 evaluate.py            # 5. evaluation
```

Free job ads are available from Arbetsförmedlingen's [JobTech APIs](https://jobtechdev.se).

## Team

Built by Jamuna, Jagriti, Yali, Dinesh and Ahmed at the AI:spire Hackathon. Thanks to Accenture and Microsoft for organising it.
