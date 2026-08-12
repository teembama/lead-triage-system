# Lead Triage

Scores inbound leads from a messy CSV and ranks them by who is worth calling today.

## Architecture

A strict three-way split, which is the point of the design:

| Layer | Responsibility | Never does |
|---|---|---|
| **LLM** | Reads unstructured `notes`, returns a categorical `level` + a short `evidence` snippet per factor | Arithmetic, prose justification |
| **Python — scoring** | Dict lookups turning each `level` into points | Re-interpreting free text |
| **Python — explanation** | Templated summary built from the scored breakdown | Calling the LLM |

Identical extractions therefore always produce identical points and identical
explanation text. The only source of run-to-run variance is the LLM's reading of
genuinely ambiguous notes — never the maths.

```
CSV upload
  -> data_cleaning      normalise, split junk into excluded_rows.csv, flag duplicates
  -> llm_extraction     one call per lead (concurrent + cached), validated JSON
  -> qualification      exclusion gate: stated purpose, then buyer status
  -> scoring            fit /50 + intent /50 + per-factor breakdown + explanation
  -> recommendation     prospect first, then priority; ranking
  -> Streamlit UI + ranked CSV export
```

## Build status

All stages are implemented.

| Stage | Module | State |
|---|---|---|
| 1 | scaffolding | done |
| 2 | `data_cleaning.py` | done — interval-preserving parsers |
| 3 | `qualification.py` | folded into `recommendation.recommend()` — the gate is a routing branch, not a separate pass |
| 4 | `schema.py`, `prompts.py`, `providers/`, `llm_extraction.py` | done — live-verified |
| 5 | `scoring.py` | done — deterministic, no LLM |
| 6 | `recommendation.py` | done — exclusion gate, routing, ranking, REVIEW bucket |
| 7 | `app.py`, `ui.py`, `target_profile.py` | done — four sections, target profile, clickable queue rows, lead detail |
| 8 | `pipeline.export_csv`, `ui.lead_report` | done — folded into the modules that own the data and the presentation; no separate `export.py` |

**Interface.** Upload and validate a CSV, set the target profile (industry and
budget required, location optional), process, then work the ranked queue: any
row is clickable and opens its full assessment in `04 // LEAD DETAIL`. Two
downloads — the ranked queue as CSV (`pipeline.export_csv`), and the selected
lead alone as a plain-text report (`ui.lead_report`).

**Tests.** 668 offline tests pass (`pytest -q`), covering cleaning, the
extraction contract, scoring, routing, resilience, the target profile and the
interface. A further 128 live assertions hit the API and are deselected by
default (`pytest -m live`); the last full run was 117 passed, 10 skipped — the
skips are gated leads inside a sweep that only applies to genuine buyers, and
are covered by the exclusion tests instead.

**Extraction.** `providers/` is a thin abstraction over two interchangeable
back-ends selected by `LLM_PROVIDER`: **Gemini is the default and the one the
system is verified against** (`gemini-3.5-flash-lite`, overridable with
`GEMINI_MODEL`), with Anthropic (`claude-opus-5`, `ANTHROPIC_MODEL`) kept
working as an alternate. Both are driven through structured outputs against the
JSON Schema generated from `schema.py`, so the contract is identical either way.
Throughput is quota-shaped rather than fixed: a shared token-bucket limiter
holds the run to `LLM_RPM` (default 15), the thread pool is sized from that
budget rather than hardcoded, a provider's own `retry_after` is honoured and a
429 pauses every worker at once. Retries are split by kind — up to 4 transport
attempts with exponential backoff and jitter, and 2 corrective attempts when a
response misses the contract. Extractions are cached content-addressed on the
prompt, schema version, profile, provider and model, so a repeated note costs
one call and a version bump can never be served a stale answer.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt   # Windows
# source .venv/bin/activate && pip install -r requirements-dev.txt   # macOS/Linux

cp .streamlit/secrets.toml.example .streamlit/secrets.toml    # add your API key
```

Run the app: `.venv/Scripts/python -m streamlit run app.py`
Run the tests: `.venv/Scripts/python -m pytest tests/ -q`

Stages 2, 5 and 6 are pure functions and their tests cost nothing to run. The
Stage 4 extraction test hits the API and is kept separate.

## Routing outcomes

**There are exactly four routing outcomes.** Every lead receives exactly one:

| Outcome | Meaning | Score published? |
|---|---|---|
| `CONTACT_NOW` | a prospect worth calling today — total ≥ 75 | yes |
| `NURTURE` | a prospect who is not ready yet — total below 75 | yes |
| `DISQUALIFY` | **not a prospect**: a stated non-buying purpose, or `is_potential_buyer = "no"` | yes |
| `REVIEW` | purpose unresolved, or two signals contradicting each other — a human decides | **no — suppressed** |

**The score sets priority, never prospect-hood.** Whether a lead is a prospect is
decided from the extracted evidence alone; the score then decides whether to call
them today and where they rank. So a genuine buyer is never disqualified by a low
number: "no budget yet, no timeline, still reading around" describes someone who
is early, not someone who is not buying. Those leads are `NURTURE`, and they sort
to the bottom of the queue where a weak opportunity belongs.

Conversely, no score rescues a stated non-buying purpose. A competitor with a
perfect 50/50 company fit who says they are comparing your pricing to benchmark
their own offering is `DISQUALIFY` at a total of 56, because the gate is read
before the total.

**`Total` is a metric, not a routing outcome.** The UI shows five tiles —
`Total Processed` plus the four routes — and the four route counts sum to
`Total Processed`. No lead is ever routed to "Total"; it is a count of leads
seen. `summarise_routes()` returns five keys for this reason, and the extra key
is `total_processed`, not a fifth bucket.

Evaluation order (architecture §8/§9). Evidence is read first, and a score can
neither overturn it nor stand in for it:

```
non_buyer_signal unrecognised           → REVIEW        (contract drift)
non_buyer_signal ≠ "none" and buyer=yes → REVIEW        (the two contradict)
non_buyer_signal ≠ "none"               → DISQUALIFY    (score never read)
buyer == "no"                           → DISQUALIFY    (score never read)
buyer == "ambiguous"                    → REVIEW        (score never read)
buyer not in {yes, no, ambiguous}       → REVIEW        (contract drift)
buyer == "yes":
    total ≥ 75                          → CONTACT_NOW
    otherwise                           → NURTURE
```

`non_buyer_signal` is what the writer *said they came for* — benchmarking,
job seeking, academic research, a media enquiry, a vendor pitch, an investor
introduction, a free resource, or spam. It is deliberately a different question
from `buyer_type`: being a competitor, an agency or a recruiter is a description
of the company, not a statement of purpose, and on its own it disqualifies
nobody. Where the two signals disagree — the model calls someone a buyer *and*
records a non-buying purpose — that is insufficient information rather than
grounds for deletion, so the lead goes to `REVIEW`.

REVIEW leads have `company_fit_score`, `buying_intent_score` and `total_score`
blanked to `None` (not `0` — a zero would sort and threshold like a real
result). They are excluded from `rank_leads()`, since §10 ranks on a total they
deliberately do not have, and returned by `review_queue()` instead.

## Cleaning behaviour

Non-destructive by design. Every raw value survives as `<column>_raw` alongside
its derived counterpart, junk rows are routed to a separate excluded frame
rather than deleted, and duplicates are flagged for a human instead of being
auto-removed. Nothing is guessed: an unparseable value becomes `None` with a
status string, never a fabricated number and never a silent zero.

Against the 520-row assessment file: **509 scoreable, 11 excluded**
(5 newsletter signups, 2 test rows, 1 leaked header, 1 empty row, 1 row with no
notes, 1 self-described non-lead), 24 rows flagged as possible duplicates.

## Deviations from the implementation plan

Each of these was a defect in the spec rather than a preference, so they are
recorded here rather than fixed silently.

1. **Header-leak rule (§2).** The plan drops rows where `lead_id == "lead_id"`.
   No such row exists — the leaked header is column-shifted
   (`lead_id="header"`, `created="lead_id"`, …), so that rule matches nothing.
   Detection is structural instead: a row with three or more fields equal to
   their own column name is a header. Two further junk rows the plan did not
   mention (one entirely blank, one with no notes) are also excluded.

2. **Date parsing is regex-dispatched, not a `strptime` cascade (§2).**
   `strptime("6/1/24", "%m/%d/%Y")` succeeds and returns year 24, so a
   try-each-format cascade would silently mis-date all 91 rows of that shape.
   The dataset also contains a sixth format the plan omits: unpadded ISO
   (`2024-6-7`, 69 rows).

3. **`DD-MM-YYYY` resolved day-first, on evidence.** 38 rows of that shape have
   a first component > 12 and none have a second component > 12. The 33
   individually-ambiguous rows carry `created_ambiguous=True` for audit.
   Low stakes regardless: `created` feeds no score.

4. **Broken emails are repaired and kept (§2).** The plan lists "broken email"
   under row-level junk, but 11 of those rows use `name[at]domain.com` and are
   genuine prospects; they are rewritten to `@` and retained. The one true
   non-lead is identified by its *notes*, not its address. Address validity is
   flagged, never grounds for exclusion.

5. **Duplicate matching ignores placeholder addresses.** `weird-email-no-domain`
   occurs 9 times and blank addresses 8 times; matching on those would
   manufacture large bogus duplicate clusters from unrelated rows.

6. **Parsers return `None` + a status, not `int | "unknown"` (§16).** The
   union-with-magic-string signature forces every caller to type-check before
   doing arithmetic. `None` plus `status` also distinguishes a confirmed zero
   budget from an unrecorded one — different facts, only one of which is
   evidence.

7. **Titles keep their original casing (§2).** Title-casing mangles the
   acronyms that dominate the column (`CEO` → `Ceo`).

8. **§7 vs §12 conflict.** The §7 table annotates the *high* tier of
   industry-fit and company-size as "(2pt)", which are the *low*-tier values.
   §12's lookup tables are treated as authoritative.

9. **Modules live at the repo root**, not in a nested `lead_triage/` package
   (§14) — the repository is already the project, and Streamlit Community Cloud
   finds a root-level `app.py` with no extra configuration.

10. **`python-dateutil` dropped** from the dependency list; the parser uses only
    the standard library.

11. **Headcount and budget are intervals, not points.** `50+` is stored as
    `low=50, high=None, status="floor"` and `35-55` as `low=35, high=55` —
    §16's midpoint and `int | "unknown"` signature both assert a precision the
    source never gave, and cannot express "at least 50" at all. The cleaning
    layer now says what the data says; mapping that onto the Company Size /10
    and Budget /10 rubrics is scoring.py's decision, made once, with the
    uncertainty still visible.

12. **The LLM reports notes-stated headcount and budget as intervals, not as a
    `level` (§11).** Judging `23 employees` against the 10–70 band is the size
    rubric — a scoring decision, which by the architecture's own rule cannot
    live in the LLM. The model returns `{stated, low, high, kind, evidence}` in
    the same vocabulary `Measure` uses, so scoring.py reconciles the notes value
    and the CRM value under the §2 precedence rule and applies the band once.
    Factors that are genuine text interpretation (industry fit, seniority,
    urgency, …) still return a `level`. `budget_in_notes` is an addition — §2's
    precedence rule covers `monthly_budget`, but §11 provides no field for it.

13. **The intent floor and the `< 45` DISQUALIFY band were removed (§9).** Both
    asked the score to answer a question it cannot answer — whether someone is a
    prospect at all — and both did so by deleting genuine early-stage buyers.
    "I am researching tools and gathering ideas. No project approved yet." is a
    real enquiry with every intent factor at its floor; §9 disqualified it.

    Nothing is lost by dropping the floor. §9 introduced it so that fit alone
    could not carry a lead to `CONTACT_NOW`, and that guarantee is now arithmetic
    rather than a rule: company fit is clamped at 50, so a lead below the floor
    tops out at 64 and cannot reach 75 by any combination of factors.
    `test_the_intent_floor_is_subsumed_by_the_contact_now_line` asserts it
    directly. `INTENT_FLOOR` and `NURTURE_MIN_TOTAL` survive in the source as
    documented non-routing constants so the deviation stays visible.

    45 therefore stops being a routing boundary: every lead that clears the gates
    and scores below 75 is `NURTURE`. §9 anticipated the thresholds moving —
    *"the cutoff should move, not the underlying logic"* — and this is the
    opposite, so it is recorded here rather than treated as tuning. It restores
    §3's own disposition table, which sends only non-buyer categories to "hard
    disqualify" and routes no-budget freelancers to "scoring (low fit, **not
    gated**)".

14. **`non_buyer_signal` is an addition to the §11 contract.** §8 asks for a hard
    pre-scoring gate on explicit non-buyers but provides only
    `is_potential_buyer`, which conflates "here for another reason" with "cannot
    tell". A separate enum for the *stated purpose* keeps the two apart, which is
    what lets a peer agency buying for its own team stay a buyer while a peer
    agency benchmarking your pricing does not.

Still open: the §11 extraction enums must be reconciled with the §12 lookup keys
(`within_1_month` vs `near_term`) via a shared `schema.py`.

## Data

The source CSV is never modified. `sample_data/leads_sample.csv` is an 18-row
subset drawn verbatim from it, chosen by set-cover to exercise all 45 distinct
value shapes and categories (every date format, every employee and budget
notation, every source, every exclusion reason, and each lead category) — it is
the fast fixture for iterating without processing 520 rows.

## Deployment

Streamlit Community Cloud, with the API key set as a secret. `.streamlit/secrets.toml`
is gitignored and must never be committed.
