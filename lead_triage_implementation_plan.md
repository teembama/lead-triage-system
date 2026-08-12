# Lead Triage System — Implementation Plan
*Prepared for: AI Automation Academy Assessment — Task 1*
*Prepared for handoff to: Claude Code*
*Do not implement from this document directly — this is the spec, not the build.*

---

## 1. Dataset Findings

- **520 rows** (519 valid + 1 header row leaked into the data as a data row), 11 columns:
  `lead_id, created, name, email, company, employees, website, title, source, monthly_budget, notes`
- **Missing values**: `monthly_budget` 38% missing, `employees` 25%, `website` 21%, `title` 13%, `created` 5%, `name` 4%, `email`/`company` ~1-2%. `notes` is present on all but 2 rows.
- **Duplicate/junk rows**: at least one row's own notes text says `"(duplicate submission)"`; at least one row is a newsletter signup that doesn't belong; one row is a broken/garbage email (`'call me'`); a handful of rows are literally `test`/`asdf` values in `title`/`source`; the CSV header itself appears once as a data row.
- **Date formats** (`created`): mixed — `MM/DD/YYYY`, `YYYY-MM-DD`, `Mon D YYYY`, `DD-MM-YYYY`, `M/D/YY`. Ambiguous without normalization (e.g. `04-06-2024` could be April 6 or June 4).
- **Employees formats**: plain integers, ranges (`35-55`), plus-notation (`51+`), approximations (`~61`), and junk (`asdf`).
- **Budget formats**: `$6k/mo`, `5,000/mo`, `6k-8k`, `$8,500`, plain numbers, `0`, and non-numeric placeholders (`TBD`, `depends`, `budget`).
- **Titles**: 23 distinct values, mostly clean (`Founder`, `CEO`, `Owner`, `COO`, `VP Growth`, `Head of Ops`, etc.), plus junk (`asdf`, `test`) and a header-leak (`title`).
- **Source values**: `webform` (185), `referral` (127), `linkedin` (77), `event` (64), `cold reply` (61), plus 3 junk (`test`, header-leak).

**No modifications were made to the source file** — all analysis above was read-only.

---

## 2. Data-Cleaning Plan

Cleaning must be **non-destructive and reusable**: it should produce a normalized column *alongside* the raw value, never overwrite/guess at missing data, and flag anything ambiguous rather than silently resolving it.

| Field | Rule |
|---|---|
| `lead_id` | Trim; if blank, generate a synthetic ID and flag `id_missing=true` |
| `created` | Parse using a cascade of known formats (`%m/%d/%Y`, `%Y-%m-%d`, `%b %d %Y`, `%d-%m-%Y`, `%m/%d/%y`); normalize to ISO `YYYY-MM-DD`; if unparseable, keep raw and set `created_clean = None` |
| `name`, `email`, `company` | Trim whitespace, lowercase email for matching (keep display case separately) |
| `employees` | Parse ranges → midpoint; `N+` → treat as the stated floor; `~N` → N; plain int → int; unparseable/blank → `"unknown"` (never assume small) |
| `monthly_budget` | Strip `$`, `,`, normalize `k`→×1000, strip `/mo`; ranges → midpoint; `TBD`/`depends`/`budget`/blank → `"unknown"` |
| `title` | Trim, title-case for display; map to a controlled seniority tier (see §7) |
| `source` | Lowercase, trim; map `test`/header-leak/blank → `"unknown"` |
| `notes` | Trim only — never alter content, this is the LLM's input |
| **Row-level junk filter** | Drop rows where `lead_id` is literally `"lead_id"` (header leak), or `source`/`title` is `"test"`/`"asdf"`, or notes indicate the row isn't a real lead (newsletter signup, broken email) — route these to a separate `excluded_rows.csv`, don't silently delete |
| **Duplicates** | Flag (don't auto-remove) rows where `email` + `company` match another row, or where notes contain `"duplicate"` — human reviews, since the assessment says preserve information rather than destructively change it |

**Data-precedence rule (notes vs. structured fields):** the `employees` and `monthly_budget` structured fields are sometimes blank or stale while the notes state a specific current number (e.g. lead 1137's `employees` field is `9` but its notes say *"23 people"*; lead 1337's `employees` field is blank but its notes say *"43 people"*). When the LLM extraction (§11) surfaces a specific, credible numeric value stated directly in the notes, **that value takes precedence over a conflicting or missing structured field** for scoring purposes — the notes are closer to a live conversation and more likely current than a CRM field that may not have been updated. The structured field is still retained and displayed for reference, and any conflict is recorded (not silently overwritten) so the discrepancy is visible in the lead detail view, not just resolved invisibly.

Output of this stage: a cleaned DataFrame with both `_raw` and `_clean` columns for anything transformed, plus an `excluded_rows.csv` for junk that shouldn't be scored at all.

---

## 3. Lead-Type Analysis

Categories found, with real counts from the 519 notes (categories can overlap since one note can hit multiple patterns):

| Category | Count | Disposition |
|---|---|---|
| Genuine business prospect (agency/company describing a real pain point) | ~270 | → scoring |
| Explicit non-buyer statement | 48 | → hard disqualify |
| No-budget freelancer/solo operator | 41 | → scoring (low fit, not gated) |
| Student/researcher, no commercial intent | 33 | → hard disqualify |
| Spam/scam | 31 | → hard disqualify |
| VC intro (not a direct buyer) | 11 | → hard disqualify |
| Recruiter | 9 | → hard disqualify |
| Competitor doing recon | 8 | → hard disqualify |
| Journalist/media | 5 | → hard disqualify |
| Data junk (not a real lead at all) | 6 | → excluded pre-scoring, not even "disqualified" |

**Strongest recurring buyer template** (100 of 519 rows): *"We're a [agency type], N people. [specific pain] is eating our week. Want it automated end to end. Budget approved, [urgency phrase]."* These are the clearest `CONTACT_NOW` candidates and should score high on both fit and intent independently — a good internal consistency check for the scoring model.

**ICP-adjacent edge cases** (real budget, outside the core marketing/growth-agency ICP): small local business, car dealership, SaaS company, ecom brand. These are legitimate buyers — they should pass the buyer gate but score lower on *industry fit* specifically, not get gated out. Gating them would hide a real prospect (e.g. *"SaaS company... Budgeted, serious"*).

---

## 4. Intent-Signal Analysis

| Intent level | Recurring phrases |
|---|---|
| **High** | `"Budget approved"`, `"ASAP"`, `"ready to pilot in the next 2 weeks"`, `"keen to move fast"`, `"priority for the quarter"`, `"I make the call here"` / `"Decision is mine"` |
| **Medium** | `"comparing a few options"`, `"budget not locked yet"`, `"have some budget"`, `"decision in about a month"`, `"would need to loop in the team"` |
| **Low** | `"not totally sure what we need yet"`, `"curious"`, `"exploring"`, `"maybe later"`, `"price sensitive"` (dampener, not standalone) |
| **Non-buying** | any of the hard-disqualify patterns in §3 |

`"Price sensitive"` recurs often enough as an explicit modifier on otherwise-positive notes that it should reduce *purchasing readiness* specifically, rather than being its own scored factor.

No notes in this dataset explicitly ask for pricing or request a demo — those signal types exist in your brief's examples but don't appear in this data; the LLM extraction schema should still support them for future exports, just expect them to be rare/absent here.

---

## 5. Company-Fit Analysis

- **Industry fit**: cleanly extractable — buyer-template notes self-identify agency type (influencer marketing, media buying, appointment-setting, cold email, B2B, content, GTM, growth, performance marketing). This should matter most (15 pts) since it's the clearest, most reliable signal in the dataset.
- **Company size**: buyer-template agencies cluster in the **10–70 employee** range. Very small (1-3, "very early startup") and unknown should not automatically be penalized to zero — but per your earlier instruction to treat missing data conservatively, "unknown" gets a low-but-not-zero score, clearly labeled as *missing* rather than *confirmed small*.
- **Seniority**: title field is clean and maps to 3 tiers well (see §7).
- **Budget capacity**: once normalized to monthly USD, ranges roughly $500–$18k/mo among real prospects — wide enough to differentiate meaningfully.
- **Source**: see dedicated analysis below — should stay minor.

---

## 6. Source Analysis

| Source | Count | Disqualify rate | "Budget approved" mention rate |
|---|---|---|---|
| webform | 185 | 23% | 18% |
| referral | 127 | 29% | 20% |
| linkedin | 77 | 23% | 23% |
| event | 64 | 17% | 20% |
| cold reply | 61 | 21% | 13% |

**Finding**: the data does not support "referral is automatically stronger than webform" — referral actually has the highest junk rate. Event is the cleanest channel. Cold reply is the only source that's meaningfully weaker on positive-intent language.

**Recommendation**: keep source at low weight as you specified (your instinct to cap it low was right), but don't hard-code a referral-favoring hierarchy. Use: `event / referral / linkedin / webform` cluster together (roughly equal), `cold reply` scores slightly lower. This makes the small weight (5 pts) defensible from evidence rather than assumption — exactly the caution your brief asked for.

---

## 7. Final Scoring Model

### Company Fit — /50

| Factor | Max | Low | Medium | High |
|---|---|---|---|---|
| Industry/business fit | 15 | Unrelated to agency's offering (car dealership, local retail) | Adjacent (SaaS, ecom brand w/ real budget) | Core ICP — marketing/growth/ops agency (2pt) |
| Company size fit | 10 | Unknown, or far outside 10-70 range | 1-9 or 70+ employees | 10-70 employees (2pt) — **uses notes-stated figure over structured field on conflict, per data-precedence rule in §2** |
| Decision-maker seniority | 10 | Consultant/Freelancer/Developer, or unknown | VP/Head of | Founder/CEO/Owner/COO/Partner/Managing Director |
| Budget capacity | 10 | <$1k/mo or unknown (2 pts) | $1k–<$4k/mo (4 pts) | $4k–<$8k/mo (7 pts) · ≥$8k/mo (10 pts) — **four-tier scale**, see §12 |
| Lead source | 5 | cold reply | webform | event/referral/linkedin (clustered, not strongly differentiated per §6) |

### Buying Intent — /50

| Factor | Max | Low | Medium | High |
|---|---|---|---|---|
| Urgency | 15 | none stated | "comparing options" | "ASAP"/"ready to pilot"/"keen to move fast" |
| Pain severity | 12 | no problem named | vague interest | "[X] is eating our week" — specific, named pain |
| Purchasing readiness | 10 | unknown/no budget | budget not locked / "have some budget" | "Budget approved" (–2pt if "price sensitive" also present) |
| Timeline | 8 | none stated | "decision in about a month" | "this month"/"within 2 weeks" |
| Buying stage | 5 | "not totally sure what we need yet" | "comparing a few options" | "want it automated end to end" (committed) |

**Weights confirmed against the data**: your proposed 50/50 split holds up — the buyer-template rows score high on both dimensions independently, so a balanced model won't systematically favor one signal type over the other.

---

## 8. Buyer / Disqualification Rules

**Hard gate (pre-scoring, `is_potential_buyer = false` → always `DISQUALIFY`)**:
job seeker, student/researcher w/ no commercial intent, journalist, recruiter, VC intro, competitor recon, spam, and structural data-junk (newsletter signup, broken email, header-leak/test rows — these are excluded before even reaching the gate).

**NOT hard-gated — passes gate, scored low instead**:
no-budget freelancers/solo operators, ICP-adjacent businesses (car dealership, local business, SaaS, ecom brand). These are real commercial inquiries; gating them would hide legitimate — if weak or off-ICP — opportunities.

**Ambiguous → flagged for human review, not auto-resolved either direction**:
notes that are genuinely unclear about intent (e.g. *"Fellow agency owner here, mostly researching the market"* — competitor or soft prospect?). The LLM extraction should be allowed to output `is_potential_buyer: "ambiguous"` alongside a reasoning string, and these should surface distinctly in the UI rather than being silently bucketed.

---

## 9. Recommendation Thresholds

- `is_potential_buyer = false` → **DISQUALIFY**, overriding any score.
- `is_potential_buyer = true` and `buying_intent_score < 15/50` → **DISQUALIFY** even if fit is high. This directly answers your concern about a lead scoring well on fit alone with no real intent — a hard floor on intent, not just a blended total, prevents that.
- Total ≥ 75/100 (with intent floor satisfied) → **CONTACT_NOW**
- Total 45–74 → **NURTURE**
- Total < 45 (among legitimate buyers) → **DISQUALIFY**

These are starting thresholds — treat them as adjustable after seeing the real score distribution from a first run; if e.g. `CONTACT_NOW` ends up with only 3 leads or 200 leads, the cutoff should move, not the underlying logic.

---

## 10. Ranking

**Primary sort**: total score (descending).
**Tiebreak 1**: buying intent score (descending) — reflects your stated principle that intent matters more than fit alone at the margin.
**Tiebreak 2**: urgency sub-signal (high > medium > low).

This ordering means two leads with identical totals but different intent/fit splits will surface the more sales-ready one first, which matches the practical goal (who do I call today).

---

## 11. LLM Extraction Schema

The LLM returns **structured signals only** — a `level` plus one short evidence snippet per factor. It does not compute points, and it does not write the final explanation; both of those are Python's job (§12). This keeps the LLM's output small, fast, and strictly about interpretation, not scoring.

```json
{
  "is_potential_buyer": true,
  "buyer_type": "business_prospect",
  "disqualification_evidence": null,

  "industry_fit": { "level": "high", "evidence": "Full-service marketing agency" },
  "company_size_signal": { "level": "high", "evidence": "23 employees", "source": "notes" },
  "seniority": { "level": "decision_maker", "evidence": "I make the call here" },

  "urgency": { "level": "high", "evidence": "Ready to pilot in 2 weeks" },
  "pain_severity": { "level": "high", "evidence": "Eating our week" },
  "purchasing_readiness": { "level": "approved", "evidence": "Budget approved" },
  "timeline": { "level": "within_1_month", "evidence": "Next 2 weeks" },
  "buying_stage": { "level": "committed", "evidence": "Want it automated end to end" },
  "price_sensitive_flag": false
}
```

`level` values are fixed enums (not free text) so the deterministic scorer can map them directly to points. `evidence` is a short quoted/paraphrased snippet (a few words, not a sentence) — enough to point back at the source phrase, not a written justification. `source` on `company_size_signal` records whether the notes or the structured field was used, per the §2 precedence rule. Note there is no `reasoning`/`explanation` field here — that's generated downstream in Python (§12), not by the LLM.

---

## 12. Deterministic Scoring Specification

Straight lookup tables — no LLM involvement past extraction:

```
industry_fit:        high=15, medium=8,  low=2
company_size_signal:  high=10, medium=6,  low=3,  unknown=2   # uses notes value over structured field on conflict (§2 precedence rule)
seniority:            decision_maker=10, high_influence=6, low_influence=3, unknown=2
budget_capacity:      >=8000/mo=10, 4000-7999=7, 1000-3999=4, <1000/unknown=2   # four-tier scale, replaces prior single cutoff
source (from clean data, not LLM): event/referral/linkedin=5, webform=3, cold_reply=1, unknown=2

urgency:              high=15, medium=8, low=2
pain_severity:        high=12, medium=6, low=2
purchasing_readiness: approved=10, partial=5, unknown=1  (−2 if price_sensitive_flag)
timeline:             near_term=8, medium_term=4, none=0
buying_stage:         committed=5, evaluating=3, exploring=1
```

`company_fit_score = sum of the 5 fit factors` (max 50)
`buying_intent_score = sum of the 5 intent factors` (max 50, floor 0)
`total_score = company_fit_score + buying_intent_score`

The scorer's output retains a **per-factor breakdown** built by Python, not the LLM: `{"industry_fit": {"level": "high", "points": 15, "evidence": "Full-service marketing agency"}, ...}` — the `level` and `evidence` come straight from the §11 extraction unchanged; `points` is looked up here.

**Final explanation generation is also Python's job, not the LLM's.** A separate function, `generate_explanation(breakdown, recommendation) -> str`, builds a short human-readable summary from the breakdown deterministically — e.g. templated as *"Total 98/100 → CONTACT_NOW. Strong industry fit (full-service marketing agency), high urgency (ready to pilot in 2 weeks), budget approved."* This keeps explanations fully reproducible: the same breakdown always produces the same explanation text, with no LLM call involved at this step.

All point math and explanation text lives in plain Python — no randomness, fully unit-testable, and independent of which LLM model produced the extraction.

---

## 13. Streamlit UI Specification

Single-page app, four sections in order:

1. **Upload & mode selector** — file uploader for CSV, plus a **Processing mode** radio: `Demo (first N rows)` with an `st.number_input` for N (default 20) vs `Full CSV`. "Process Leads" button. Demo mode lets you iterate on prompts/thresholds without burning API calls or waiting on all 520 rows every run; Full mode is what you'd run once before submitting.
2. **Summary metrics row** — 4 `st.metric` widgets: Total Processed, Contact Now, Nurture, Disqualified.
3. **Ranked table** — `st.dataframe` sorted by total_score desc, columns: rank, name, company, total_score, recommendation, fit_score, intent_score. Filterable by recommendation via a `st.selectbox`.
4. **Lead detail drill-down** — `st.selectbox` to pick a lead by name/company, then display its full breakdown: every scored factor with its `level`, its `points`, and its short `evidence` snippet (e.g. `Urgency: high (15 pts) — "ready to pilot in the next 2 weeks"`), plus the Python-generated overall explanation string and, where relevant, the `company_size_signal.source` conflict note. Every number on the page traces back to a specific phrase in the original notes, not just asserted.
5. **Download button** — `st.download_button` for the full ranked CSV.

No auth, no database, no multi-page nav — matches your "short assessment app" scope.

---

## 14. Project Structure

```
lead_triage/
├── app.py                 # Streamlit entrypoint — orchestrates the pipeline, renders UI
├── data_cleaning.py        # §2 — normalization functions, junk-row filtering
├── qualification.py        # §8 — buyer-gate logic (post-LLM-extraction routing)
├── llm_extraction.py       # §11 — prompt template + API call + schema validation
├── scoring.py               # §12 — deterministic point lookups + totals
├── recommendation.py        # §9-10 — thresholds, ranking
├── prompts.py               # LLM system/user prompt text, kept separate for iteration
├── requirements.txt
├── README.md
└── sample_data/
    └── leads_sample.csv     # small sanitized sample for quick testing
```

This mirrors your suggested structure with one addition (`recommendation.py` split out from `scoring.py`) since thresholding/ranking is a distinct concern from point math and will likely need tuning independently.

---

## 15. Hosting Recommendation

**Streamlit Community Cloud** — free, connects directly to a GitHub repo, deploys on push, gives a public URL with zero server management. This is the simplest option that satisfies "a short hosted app we can actually run." Store the LLM API key as a Streamlit secret (`st.secrets`), never hardcoded.

Alternative if GitHub isn't available: Streamlit can also run locally and be tunneled temporarily, but Community Cloud is strictly better for a submission link since it stays up without your machine running.

---

## 16. Claude Code Implementation Plan

**Stage 1 — Project setup**
Files: `requirements.txt`, `README.md`, folder structure per §14.
Dependencies: `streamlit`, `pandas`, `anthropic` (or chosen LLM SDK), `python-dateutil`.
Output: runnable skeleton, empty `app.py` that renders "Upload a CSV" placeholder.

**Stage 2 — Data cleaning (`data_cleaning.py`)**
Functions: `parse_date(raw) -> str|None`, `parse_employees(raw) -> int|"unknown"`, `parse_budget(raw) -> float|"unknown"`, `clean_lead_row(row) -> dict`, `clean_dataframe(df) -> (clean_df, excluded_df)`.
Input: raw pandas DataFrame from uploaded CSV.
Output: cleaned DataFrame + separate excluded-rows DataFrame.
Edge cases: unparseable dates, ranges with unusual separators, header row leaked as data, duplicate detection.
Tests: unit test each parser against every raw format variant found in §1 (table-driven test using the actual value samples documented above).

**Stage 3 — Buyer qualification integration (`qualification.py`)**
Function: `apply_buyer_gate(lead_with_llm_output) -> dict` — reads `is_potential_buyer`/`buyer_type` from the LLM extraction and short-circuits scoring when false.
Input: cleaned lead + LLM extraction JSON.
Output: lead dict annotated with gate decision.
Edge case: `is_potential_buyer = "ambiguous"` — routed to a review bucket, not scored, not auto-disqualified.

**Stage 4 — LLM extraction (`llm_extraction.py`, `prompts.py`)**
Function: `extract_signals(lead_notes: str, lead_context: dict) -> dict` matching the schema in §11 — structured `{level, evidence}` per factor, **no points, no free-text explanation**. Retry/validation: reject and retry once if required enum fields are missing or out of the allowed value set, or if a scored factor's `evidence` is empty.
Input: cleaned lead's notes + title/company for context, including the structured `employees`/`monthly_budget` values so the model can apply the §2 precedence rule and set `company_size_signal.source`.
Output: validated JSON matching §11 — the prompt should explicitly instruct the model to keep each `evidence` value short (a few words, not a sentence) and to never include a score or a written justification, since Python owns both.
Dependencies: LLM API key from `st.secrets`.
Edge cases: API failure (fallback: mark as `"unknown"` across fields, flag for manual review, don't crash the batch), malformed JSON response (retry once, then flag), response missing evidence for a factor (retry once, then fall back to `evidence: "not specified"` rather than blocking).
Tests: run against ~15 hand-picked notes spanning every category in §3, assert expected `is_potential_buyer`, rough signal levels, and that every scored factor has a non-empty, short evidence string.

**Stage 5 — Deterministic scoring (`scoring.py`)**
Functions:
- `score_lead(extraction: dict, clean_lead: dict) -> dict` implementing the lookup tables in §12, producing `{company_fit_score, buying_intent_score, total_score, breakdown: {factor: {level, points, evidence}, ...}}`.
- `generate_explanation(breakdown: dict, recommendation: str) -> str` — builds the human-readable summary from the breakdown via a fixed template, entirely in Python, no LLM call.
Input: LLM extraction + cleaned source/employees/budget fields (fit's `source` and `budget_capacity`/`company_size_signal` numeric parts come from cleaned data, except where the §2 precedence rule directs the notes value to be used).
Output: scored lead dict including the breakdown and the generated explanation string.
Tests: table-driven — feed every combination of `level` values, assert exact point totals match §12; assert the budget four-tier boundaries ($999→2pts, $1000→4pts, $3999→4pts, $4000→7pts, $7999→7pts, $8000→10pts); assert `evidence` passes through unmodified into the breakdown; assert `generate_explanation` output is identical for two calls with the same breakdown (determinism check).

**Stage 6 — Recommendation & ranking (`recommendation.py`)**
Function: `recommend(scored_lead) -> str` applying §9 thresholds and the intent floor; `rank_leads(list[scored_lead]) -> sorted list` applying §10 sort order.
Tests: verify the intent-floor override actually blocks a high-fit/low-intent lead from `CONTACT_NOW`.

**Stage 7 — Streamlit UI (`app.py`)**
Wires stages 2-6 into the interface described in §13, including the Demo/Full processing-mode selector.
Should show a progress indicator while LLM extraction runs (this is the slow step — likely several seconds per lead).
Demo mode (default, small N like 20 rows) is for iterating on prompts/thresholds cheaply and quickly during development; Full mode processes the entire uploaded CSV and is what gets run once before submission. Both modes reuse the exact same pipeline functions — the only difference is how many cleaned rows get passed to the LLM extraction step.

**Stage 8 — CSV export**
Function: `export_ranked_csv(ranked_leads) -> bytes` for `st.download_button`, including both original and derived columns, per-factor levels/points, a flattened `evidence` column per factor, and the Python-generated explanation string — so the output is self-explanatory, including *why* each score was awarded, without re-opening this document.

**Stage 9 — Testing**
Unit tests for stages 2, 5, 6 (pure functions, no API calls, fast).
Integration test for stage 4 against a small fixed sample with expected outputs, run manually/sparingly (costs API calls).
Manual end-to-end test: run the full 520-row file through the deployed app once before submission, sanity-check the `CONTACT_NOW` count is plausible (not 0, not 400+).

**Stage 10 — Deployment**
Push to GitHub, connect repo to Streamlit Community Cloud, set the LLM API key as a secret, verify the public URL loads and processes a test CSV successfully before submitting.

---

## 17. Architecture Summary

The system is a strict three-way split: **the LLM only classifies** (reads unstructured notes, returns a `level` + a short `evidence` snippet per factor — never a number, never a written justification); **Python scores** (dict lookups converting each `level` into points, per §12); **Python also explains** (a templated function turns the scored breakdown into the human-readable summary shown in the UI and CSV). Nothing downstream of the LLM call re-interprets free text, and nothing upstream of the LLM call does arithmetic — each layer has exactly one job, which is what makes the system debuggable and explainable to a non-technical reviewer. Re-running the same lead twice could get slightly different LLM `level` classifications for genuinely ambiguous notes, but identical classifications will always produce identical points and identical explanation text — the only source of variability is interpretation, never arithmetic.

Data flows: CSV → clean (Python) → LLM extraction (per lead) → buyer gate (Python, reads LLM's `is_potential_buyer`) → scoring (Python, reads LLM's categorical fields) → recommendation + ranking (Python) → Streamlit table + CSV download. Nothing downstream of the LLM call re-interprets free text — it's categorical values in, points out, every time.
