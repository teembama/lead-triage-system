# Lead Triage System

Turns a raw export of inbound leads into a ranked call list, with a recommendation and a plain-language explanation for every lead.

<!-- SCREENSHOT + DEMO VIDEO: added in the second pass -->

**Stack:** Python · Streamlit · pandas · Gemini or Claude (structured outputs) · **Live demo:** available on request

## The problem

An agency was reviewing hundreds of inbound leads a month by hand to decide who to call. The export is messy: mixed date formats, budgets and headcounts written every possible way, test rows, newsletter sign-ups and duplicates. The useful signal (is this person actually buying, and how soon?) is buried in free-text notes. Meanwhile good leads wait while someone works through the list.

## Results

On the 520-row lead export used to build it:

- **509 leads scoreable and 11 set aside, none deleted.** The 11 are 5 newsletter sign-ups, 2 test rows, a leaked header row, an empty row, a row with no notes and one self-described non-lead.
- **24 rows flagged as possible duplicates**, for a person to decide.
- **Every lead gets exactly one recommendation** (contact now, nurture, disqualify or review), a score out of 100, a factor-by-factor breakdown and a written explanation.
- **803 offline tests** (`pytest -q`) cover cleaning, the extraction contract, scoring, routing, resilience, the target profile and the interface. A further 128 tests call the real model API and run only on request (`pytest -m live`). On the current Streamlit release, 2 of the 803 fail (see Limitations).

## What it does

1. **Upload** the lead export (CSV). It's checked before anything runs.
2. **Set the target profile:** the industry and budget you're after (both required) and, optionally, a location.
3. **Process.** Each lead is cleaned, its notes are read by the model, and it's scored and routed.
4. **Work the queue.** Leads are ordered by recommended action (contact now, then nurture, review and disqualify) and by score within each action. Any row opens that lead's full assessment: the score breakdown, the evidence behind each factor and the explanation.
5. **Export** the ranked queue as a CSV, or the selected lead alone as a plain-text report.

| Outcome | Meaning | Score shown? |
| --- | --- | --- |
| **Contact now** | A prospect worth calling today: a total of 75 or more, with buying intent of at least 35 out of 50 | Yes |
| **Nurture** | A genuine prospect who isn't ready yet | Yes |
| **Disqualify** | Not a prospect: the notes state a non-buying purpose (benchmarking, job seeking, research, a vendor pitch…), or say they aren't buying | Yes |
| **Review** | Unclear or contradictory signals: a person decides | **No:** a score would invite ranking a lead the system can't judge |

## How it works

```
CSV upload
  → cleaning        normalise every field, set junk rows aside, flag possible duplicates
  → extraction      one model call per lead (concurrent, cached), validated JSON:
                    a category plus a quoted evidence snippet for each factor
  → routing gate    stated purpose and buyer status, read before any score
  → scoring         company fit /50 + buying intent /50, from fixed lookup tables
  → explanation     templated from the scored breakdown, no model call
  → ranking         by action, then score, then urgency
  → Streamlit UI, CSV export, per-lead report
```

The design rests on a strict three-way split:

| Layer | Responsibility | Never does |
| --- | --- | --- |
| **Model** | Reads the free-text notes and returns a category and a short evidence quote per factor | Arithmetic, or writing the justification |
| **Python: scoring** | Turns each category into points with dictionary lookups | Re-interpreting free text |
| **Python: explanation** | Builds the summary from the scored breakdown | Calling the model |

So identical extractions always produce identical points and identical explanations. The only run-to-run variation is the model's reading of genuinely ambiguous notes, never the maths.

| Part | Where | What it does |
| --- | --- | --- |
| App | `app.py`, `ui.py`, `target_profile.py` | The Streamlit interface, the lead detail view and report, and the target profile |
| Cleaning | `data_cleaning.py` | Parsers for dates, headcounts, budgets and emails; junk-row and duplicate detection |
| Extraction | `llm_extraction.py`, `prompts.py`, `schema.py`, `providers/` | Model calls, the JSON contract, rate limiting, retries and caching |
| Scoring and routing | `scoring.py`, `recommendation.py` | Points, the routing gate, ranking and the review bucket |
| Pipeline | `pipeline.py` | Runs the stages and builds the CSV export |
| Sample data | `sample_data/` | Small fixtures, including an 18-row sample chosen to cover every value shape in the full export |

## Design decisions

- **The score sets priority, never whether someone is a prospect.** That's decided from the evidence first. A genuine buyer with no budget yet is early, not uninterested: they become *nurture* and sort lower, but they're never disqualified for a low number. In the other direction, no score rescues a stated non-buying purpose. A competitor with a perfect company fit who says they're benchmarking your pricing is disqualified, because the gate is read before the total.
- **Purpose and description are different questions.** Being a competitor, an agency or a recruiter describes the company, and on its own it disqualifies nobody. What disqualifies someone is a stated purpose (benchmarking, job seeking, a vendor pitch…). When the notes say both "we'll benchmark your pricing" and that a purchase is under way (budget approved, committed), a person decides.
- **Review leads show no score.** Their score fields are left empty rather than set to zero, because a zero would sort and threshold like a real result.
- **Cleaning never destroys data.** Every raw value is kept next to its cleaned version, junk rows are set aside rather than deleted, and duplicates are flagged rather than removed. An unreadable value becomes "unknown" with a reason, never a guessed number or a silent zero.
- **Dates are parsed by pattern, not by trial and error.** Trying formats one after another would read "6/1/24" as the year 24 and silently mis-date 91 rows. Ambiguous day-month dates are resolved from the evidence in the file and flagged for audit.
- **Headcounts and budgets are ranges, not points.** "50+" is stored as "at least 50" and "35–55" as a range, rather than inventing a midpoint the source never gave. Scoring decides how a range maps onto each rubric, once, with the uncertainty still visible.
- **Broken emails are repaired, not dropped.** Some genuine prospects wrote `name[at]domain.com`. Address problems are flagged, never grounds for exclusion.
- **Built for real API limits.** A shared rate limiter keeps the run within the provider's requests-per-minute, a rate-limit response pauses every worker at once, and transport retries and "fix your JSON" retries are counted separately. Extractions are cached on the exact prompt, schema version, profile, provider and model, so a repeated note costs one call and a schema change can never be served a stale answer.

## Limitations

- **The model's reading is the one judgement call.** Scoring and routing are deterministic, but each lead's factors come from the model's reading of its notes. Genuinely ambiguous notes can be read differently between runs, and those leads are meant to land in review.
- **Dependencies aren't pinned.** `requirements.txt` sets only minimum versions. On Streamlit 1.65, 2 interface tests fail (`tests/test_app_flow.py`), because they simulate a row click through Streamlit internals that changed. The app's own logic tests pass.
- **Throughput depends on your API tier.** The default of 15 requests a minute matches Gemini's free tier, so a full 520-row export takes a while on it.
- **One model provider is the reference.** Gemini is the default and the provider the system was verified against. Claude is kept working as an alternative through the same contract.

## Run it locally

**Prerequisites:** Python 3.11 and a Gemini API key (or an Anthropic API key).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt          # Windows
# source .venv/bin/activate && pip install -r requirements-dev.txt   # macOS/Linux

cp .streamlit/secrets.toml.example .streamlit/secrets.toml          # then add your key
```

`.streamlit/secrets.toml` is gitignored; never commit it. Settings:

- `LLM_PROVIDER`: `gemini` (default) or `anthropic`.
- `GEMINI_API_KEY` or `ANTHROPIC_API_KEY`: only the active provider's key is needed.
- Optional: `GEMINI_MODEL` (default `gemini-3.5-flash-lite`) or `ANTHROPIC_MODEL` (default `claude-opus-5`).
- Optional: `LLM_RPM`, requests per minute (default 15). It also sizes the worker pool.

Run the app and the tests:

```bash
.venv/Scripts/python -m streamlit run app.py
.venv/Scripts/python -m pytest -q             # offline tests; no API calls, no cost
.venv/Scripts/python -m pytest -m live        # real API calls; costs credits
```

On Windows, if installing fails with a long-path error, create the virtual environment at a shorter path.

## Deploy

Streamlit Community Cloud finds `app.py` at the repo root. Set the API key and provider in the app's **Secrets** panel.

## Data

The source CSV is never modified. `sample_data/leads_sample.csv` is an 18-row subset drawn unchanged from the full export, chosen to cover all 45 distinct value shapes and categories (every date format, every headcount and budget notation, every source, every exclusion reason and each lead category). It's the fast fixture for trying changes without processing 520 rows.
