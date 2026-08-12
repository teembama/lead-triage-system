"""Stage 4 live check - runs the five specified leads and prints the report.

    python run_stage4_live.py

Five leads only. This is an extraction-quality check, not a batch run; the full
dataset is deliberately not processed at this stage.

Prints the full structured extraction per lead plus provider, model, validation
result, retry count and errors. It prints no scores, no ranking and no
recommendation, because none of that exists yet and none of it is Stage 4's job.
"""

from __future__ import annotations

import json
import os
import sys

import llm_extraction
import providers
import schema
from data_cleaning import clean_dataframe, read_leads_csv

SOURCE_CSV = "Cohort 3 Assessment — Task 1 Leads (messy).csv"

# The lead set is imported from the live test suite rather than restated here.
# These two were separate hardcoded lists once, and they silently diverged: a
# counter-example added to the test suite was absent from this script, so a
# clean run here reported success without ever exercising it. One source of
# truth removes that failure mode.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
from test_extraction_live import CASES  # noqa: E402

LEAD_IDS = list(CASES)
WHY = {lead_id: case.get("note", "") for lead_id, case in CASES.items()}


def main() -> int:
    try:
        provider = providers.get_provider()
    except providers.ProviderConfigError as exc:
        print(f"SKIPPED: {exc}")
        return 0
    if hasattr(provider, "is_configured") and not provider.is_configured():
        print(f"SKIPPED: no API key configured for provider {provider.name!r}.")
        print("Set GEMINI_API_KEY, or add it to .streamlit/secrets.toml (gitignored).")
        return 0

    if not os.path.exists(SOURCE_CSV):
        print(f"SKIPPED: {SOURCE_CSV} not found.")
        return 0

    clean_df, _ = clean_dataframe(read_leads_csv(SOURCE_CSV))
    by_id = {row["lead_id"]: row for row in clean_df.to_dict("records")}
    missing = [lid for lid in LEAD_IDS if lid not in by_id]
    if missing:
        print(f"SKIPPED: leads not in CSV: {missing}")
        return 0

    leads = [by_id[lid] for lid in LEAD_IDS]
    print(f"Provider: {provider.name}   Model: {provider.model}   Leads: {len(leads)}\n")

    results = llm_extraction.extract_batch(
        leads, provider,
        cache=llm_extraction.ExtractionCache(path=None),
        max_workers=3,
    )

    for lead, result in zip(leads, results):
        problems = schema.validate_extraction(result.extraction)
        expected = CASES.get(result.lead_id, {}).get("buyer")
        actual = result.extraction.get("is_potential_buyer")
        verdict = "" if expected is None else (
            "  [MATCHES EXPECTATION]" if actual in expected
            else f"  [!! MISMATCH - expected one of {sorted(expected)} !!]"
        )
        print("=" * 78)
        print(f"LEAD {result.lead_id}  ({WHY.get(result.lead_id, '')})")
        print("=" * 78)
        print(f"  buyer status    : {actual!r}{verdict}")
        print(f"  provider        : {result.provider}")
        print(f"  model           : {result.model}")
        print(f"  status          : {result.status}")
        print(f"  attempts        : {result.attempts}   (retries: {result.attempts - 1})")
        print(f"  validation      : {'PASS' if not problems else 'FAIL ' + str(problems)}")
        print(f"  error           : {result.error or 'none'}")
        print(f"  tokens          : in={result.input_tokens} out={result.output_tokens}")
        print(f"  CRM employees   : {lead.get('employees_display')!r}"
              f"   CRM budget: {lead.get('budget_display')!r}")
        print(f"  notes           : {(lead.get('notes_clean') or '')[:150]}")
        print("\n  --- structured extraction ---")
        print(_indent(json.dumps(result.extraction, indent=2)))
        print("\n  --- evidence snippets ---")
        for factor, snippet in schema.evidence_fields(result.extraction).items():
            print(f"    {factor:26} {snippet!r}")
        print()

    summary = llm_extraction.summarise_batch(results)
    print("=" * 78)
    print("RUN SUMMARY (extraction outcomes only - no scoring exists yet)")
    print("=" * 78)
    for key in ("total", "ok", "recovered", "failed", "api_calls",
                "buyer_yes", "buyer_no", "buyer_ambiguous",
                "input_tokens", "output_tokens"):
        print(f"  {key:18} {summary[key]}")
    return 0


def _indent(text: str, pad: str = "    ") -> str:
    return "\n".join(pad + line for line in text.splitlines())


if __name__ == "__main__":
    sys.exit(main())
