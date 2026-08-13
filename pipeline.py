"""Orchestration: an uploaded file in, presentation-ready rows out.

Separate from app.py so the whole upload-to-results path can be run in tests
without a page or a browser, and so the interface file contains flow rather
than plumbing.

Nothing here decides anything about a lead. Cleaning, extraction, scoring and
routing are called unchanged; this module only sequences them and assembles the
fields the interface needs.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import pandas as pd

from data_cleaning import clean_dataframe
from llm_extraction import ExtractionCache, extract_batch
from providers import get_provider
from recommendation import rank_leads, route_lead, summarise_routes
from scoring import explanation_parts, score_lead

ROUTE_ORDER = ("CONTACT_NOW", "NURTURE", "DISQUALIFY", "REVIEW")

# The single user-facing sentence for a lead the system could not assess.
# Kept here rather than in ui.py so the interface and the CSV export cannot
# drift apart, and so no code path can substitute vendor text for it.
ASSESSMENT_FAILED_REVIEW_REASON = (
    "This lead could not be assessed automatically and has been set aside "
    "for manual review."
)


def run_pipeline(
    raw_df: pd.DataFrame,
    limit: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    provider: Any = None,
    cache: Optional[ExtractionCache] = None,
    rpm: Optional[float] = None,
    profile: Any = None,
    sleep: Optional[Callable[[float], None]] = None,
) -> dict[str, Any]:
    """Clean, classify, score and route an uploaded lead file.

    `provider` and `cache` are injectable so tests can drive the full path
    deterministically; in the app both are left to their configured defaults.

    Every lead is returned in one ranked queue, ordered by route and then by
    score within the route. Review leads take a rank like any other row; what
    they still do not carry is a score.
    """
    clean_df, excluded_df = clean_dataframe(raw_df)
    leads = clean_df.to_dict("records")
    if limit:
        leads = leads[: int(limit)]

    provider = provider or get_provider()
    cache = cache if cache is not None else ExtractionCache()
    # `rpm` is a passthrough so a caller can override the configured request
    # budget - the app leaves it to LLM_RPM; tests disable it with rpm=0.
    extra = {"sleep": sleep} if sleep is not None else {}
    results = extract_batch(leads, provider, cache=cache, rpm=rpm, profile=profile,
                            progress_callback=progress, **extra)

    routed: list[dict[str, Any]] = []
    for lead, result in zip(leads, results):
        scores = score_lead(result.extraction, lead, profile)
        record = route_lead(result.extraction, scores, lead)
        # The per-lead failure signal was previously computed and then dropped
        # here, leaving the interface unable to tell a provider outage from a
        # genuinely ambiguous lead. Both still route to REVIEW - the route set
        # is unchanged - but they are now distinguishable downstream.
        failed = result.status == "failed"
        record.update(
            breakdown=scores["breakdown"],
            title=lead.get("title_clean"),
            source=lead.get("source_clean"),
            notes=lead.get("notes_clean"),
            employees_display=lead.get("employees_display"),
            budget_display=lead.get("budget_display"),
            assessment_failed=failed,
            failure_code=result.failure_code if failed else None,
            explanation_parts=(
                [] if record["scores_suppressed"]
                else explanation_parts(scores["breakdown"], record["recommendation"])
            ),
        )
        if failed:
            # Clean, non-technical copy. `result.error` holds the vendor detail
            # and stays in the logs.
            record["review_reason"] = ASSESSMENT_FAILED_REVIEW_REASON
        routed.append(record)

    # One queue, ordered by route then score. REVIEW leads used to be appended
    # unranked after the sorted rows; they are now ranked in place, between
    # NURTURE and DISQUALIFY.
    return {
        "rows": rank_leads(routed),
        "summary": summarise_routes(routed),
        "excluded": len(excluded_df),
        "failed": sum(1 for r in results if r.status == "failed"),
    }


def export_csv(rows: list[dict[str, Any]]) -> bytes:
    """Flatten results for download - one row per lead, evidence included.

    A review lead's per-factor points are omitted rather than blanked: publishing
    them would let anyone add them back into the total the review state exists
    to withhold.
    """
    out = []
    for row in rows:
        suppressed = bool(row.get("scores_suppressed"))
        flat: dict[str, Any] = {
            "rank": row.get("rank"),
            "lead_id": row.get("lead_id"),
            "name": row.get("name"),
            "company": row.get("company"),
            "recommendation": row.get("recommendation"),
            "fit_score": row.get("company_fit_score"),
            "intent_score": row.get("buying_intent_score"),
            "total_score": row.get("total_score"),
            "needs_human_review": row.get("needs_human_review"),
            # Boolean only. The failure code and vendor text stay internal.
            "assessment_failed": bool(row.get("assessment_failed")),
            "review_reason": row.get("review_reason"),
            "explanation": " ".join(row.get("explanation_parts") or []),
        }
        for key, entry in (row.get("breakdown") or {}).items():
            flat[f"{key}_level"] = entry.get("level")
            if not suppressed:
                flat[f"{key}_points"] = entry.get("points")
            flat[f"{key}_evidence"] = entry.get("evidence")
        out.append(flat)
    return pd.DataFrame(out).to_csv(index=False).encode("utf-8")
