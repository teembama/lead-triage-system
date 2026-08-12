"""Stage 5 - deterministic scoring.

No LLM call happens here, and none may ever be added. This module converts the
categorical levels produced in Stage 4 into points via fixed lookup tables, and
builds the human-readable explanation from those points with a fixed template.
The same extraction always produces the same score and the same sentence.

Boundaries this module respects:

* It never re-reads or re-interprets evidence text. `level` and `evidence` come
  from the extraction unchanged; only `points` is computed here.
* It does not decide CONTACT_NOW / NURTURE / DISQUALIFY, does not apply
  thresholds, and does not rank. `generate_explanation` is *given* the
  recommendation as an argument - it never derives one.
* It does not gate leads. A non-buyer scores like anything else; whether that
  score is used is a later decision.

Two of the ten factors are derived here rather than classified by the model,
because bucketing a number into a band is arithmetic, not interpretation:
`company_size_signal` from headcount and `budget_capacity` from budget. Both
consume the intervals Stage 4 preserved, and neither invents a bound.
"""

from __future__ import annotations

import math
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Lookup tables (plan §12) - the only place points are defined
# --------------------------------------------------------------------------- #

# Industry fit carries 20 and company size 5, a reweighting of §12's 15/10.
# Company fit still totals exactly 50 (20 + 5 + 10 + 10 + 5), so the /50 cap and
# the routing thresholds are untouched. Within each factor the lower tiers are
# the §12 shape scaled by the same ratio and rounded to the nearest whole point
# (industry x4/3: 8->11, 2->3; size x1/2: 6->3, 3->2, 2->1), which keeps every
# level strictly ordered and stops `medium` size from outscoring its own ceiling.
INDUSTRY_FIT_POINTS = {"high": 20, "medium": 11, "low": 3, "unknown": 3}
COMPANY_SIZE_POINTS = {"high": 5, "medium": 3, "low": 2, "unknown": 1}
SENIORITY_POINTS = {"decision_maker": 10, "high_influence": 6, "low_influence": 3, "unknown": 2}
SOURCE_POINTS = {
    "event": 5, "referral": 5, "linkedin": 5,
    "webform": 3,
    "cold_reply": 1,
    "unknown": 2,
}

URGENCY_POINTS = {"high": 15, "medium": 8, "low": 2}
PAIN_SEVERITY_POINTS = {"high": 12, "medium": 6, "low": 2}
PURCHASING_READINESS_POINTS = {"approved": 10, "partial": 5, "unknown": 1}
TIMELINE_POINTS = {"near_term": 8, "medium_term": 4, "none": 0}
BUYING_STAGE_POINTS = {"committed": 5, "evaluating": 3, "exploring": 1}

PRICE_SENSITIVE_PENALTY = 2

COMPANY_FIT_FACTORS = (
    "industry_fit", "company_size_signal", "seniority", "budget_capacity", "source",
)
BUYING_INTENT_FACTORS = (
    "urgency", "pain_severity", "purchasing_readiness", "timeline", "buying_stage",
)
FACTOR_ORDER = COMPANY_FIT_FACTORS + BUYING_INTENT_FACTORS

MAX_COMPANY_FIT = 50
MAX_BUYING_INTENT = 50
MAX_TOTAL = 100

# --------------------------------------------------------------------------- #
# Bands for the two derived factors
# --------------------------------------------------------------------------- #

# Headcount bands (plan §7: high = 10-70, medium = 1-9 or 70+).
#
# §7's "low" tier reads "Unknown, or far outside 10-70 range", but its "medium"
# tier already claims *everything* outside 10-70 (1-9 or 70+), which leaves
# `low` unreachable while §12 still prices it separately. The boundary used for "far
# outside" is 201+, taken from the size categories described alongside this
# stage's brief (1-9 small, 10-70 core fit, 71-200 larger but viable, 201+
# enterprise). Flagged rather than assumed - see the Stage 5 notes.
EMPLOYEE_BANDS: tuple[tuple[int, Optional[int], str], ...] = (
    (0, 9, "medium"),
    (10, 70, "high"),
    (71, 200, "medium"),
    (201, None, "low"),
)

# Budget bands in monthly USD (plan §12, four-tier scale).
BUDGET_BANDS: tuple[tuple[int, Optional[int], int], ...] = (
    (0, 999, 2),
    (1000, 3999, 4),
    (4000, 7999, 7),
    (8000, None, 10),
)


def _band_value(bands, number: float):
    for low, high, value in bands:
        if number >= low and (high is None or number <= high):
            return value
    return bands[-1][2]


def _bands_touched(bands, low: float, high: Optional[float]) -> list:
    """Every band value an interval overlaps. `high=None` means unbounded above."""
    touched = []
    for band_low, band_high, value in bands:
        if high is not None and high < band_low:
            continue
        if band_high is not None and low > band_high:
            continue
        if value not in touched:
            touched.append(value)
    return touched


# --------------------------------------------------------------------------- #
# Data precedence (plan §2) - resolve, but never hide, a conflict
# --------------------------------------------------------------------------- #

def _number(value: Any) -> Optional[float]:
    """A usable number, or None.

    NaN is rejected explicitly. A cleaned row that went through a pandas float
    column carries `float('nan')` where the source was blank, and `nan` passes
    an `isinstance(..., float)` check - so without this guard an empty CRM
    column reads as a real figure and manufactures a phantom notes-vs-CRM
    conflict.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _measure_from_notes(block: Any) -> Optional[dict[str, Any]]:
    """The notes-stated interval, or None when the notes said nothing.

    "Credible" is read narrowly: `stated` is true and a usable lower bound is
    present. A blank or malformed block is treated as silence, never as a zero.
    """
    if not isinstance(block, dict) or not block.get("stated"):
        return None
    low = _number(block.get("low"))
    if low is None:
        return None
    return {
        "low": low,
        "high": _number(block.get("high")),
        "kind": block.get("kind") or "exact",
        "evidence": block.get("evidence") or "not specified",
    }


def _measure_from_clean(lead: dict[str, Any], prefix: str) -> Optional[dict[str, Any]]:
    """The cleaned CRM interval, or None when the column held nothing usable."""
    low = _number(lead.get(f"{prefix}_low"))
    if low is None:
        return None
    display = lead.get(f"{prefix}_display")
    return {
        "low": low,
        "high": _number(lead.get(f"{prefix}_high")),
        "kind": lead.get(f"{prefix}_status") or "exact",
        "evidence": display if display and display != "unknown" else _plain(low),
    }


def _plain(number: float) -> str:
    return str(int(number)) if float(number).is_integer() else str(number)


def resolve_measure(
    extraction: dict[str, Any],
    clean_lead: dict[str, Any],
    notes_field: str,
    clean_prefix: str,
) -> dict[str, Any]:
    """Apply the §2 precedence rule to one numeric fact.

    The notes win when they state a figure, because a note is closer to a live
    conversation than a CRM column that may never have been updated. The
    structured value is *retained alongside* and any disagreement is recorded,
    so the conflict stays visible in the breakdown instead of being silently
    overwritten.
    """
    notes = _measure_from_notes(extraction.get(notes_field))
    structured = _measure_from_clean(clean_lead, clean_prefix)

    if notes is not None:
        chosen, source = notes, "notes"
    elif structured is not None:
        chosen, source = structured, "structured"
    else:
        chosen, source = None, "unknown"

    conflict = (
        notes is not None
        and structured is not None
        and (notes["low"], notes["high"]) != (structured["low"], structured["high"])
    )

    return {
        "value": chosen,
        "source": source,
        "conflict": conflict,
        "notes_value": notes,
        "structured_value": structured,
    }


def _describe(measure: Optional[dict[str, Any]]) -> str:
    if measure is None:
        return "unknown"
    low, high, kind = measure["low"], measure["high"], measure["kind"]
    if kind == "floor" or high is None:
        return f"{_plain(low)}+"
    if kind == "range" and high != low:
        return f"{_plain(low)}-{_plain(high)}"
    if kind == "approx":
        return f"~{_plain(low)}"
    return _plain(low)


# --------------------------------------------------------------------------- #
# The two derived factors
# --------------------------------------------------------------------------- #

def score_company_size(resolution: dict[str, Any]) -> dict[str, Any]:
    """Map a resolved headcount interval onto the §7 size bands.

    An interval is scored on its **lower bound**, which is the only figure the
    source actually asserts - `51+` is scored as 51, never as an invented 51-100.
    When the interval straddles more than one band the result is flagged
    `band_uncertain`, so a reviewer can see that the band was decided by the
    stated floor rather than by a known figure.
    """
    measure = resolution["value"]
    if measure is None:
        return {
            "level": "unknown", "points": COMPANY_SIZE_POINTS["unknown"],
            "evidence": "not specified", "band_uncertain": False,
        }

    level = _band_value(EMPLOYEE_BANDS, measure["low"])
    touched = _bands_touched(EMPLOYEE_BANDS, measure["low"], measure["high"])
    return {
        "level": level,
        "points": COMPANY_SIZE_POINTS[level],
        "evidence": measure["evidence"],
        "band_uncertain": len(touched) > 1,
    }


# Budget scored against the operator's target range instead of fixed dollar
# bands. Same point values as §12 so the /50 cap and the routing thresholds are
# untouched - only the question changes, from "is this a big budget" to "is this
# the budget we asked for".
BUDGET_TARGET_POINTS = {
    "on_target": 10,     # overlaps the requested range
    "near_target": 7,    # just outside it, either side
    "unknown": 4,        # nothing stated - absence of evidence, not a mismatch
    "off_target": 2,     # clearly outside the range
}

# How far outside the range still counts as "near".
BUDGET_NEAR_BELOW = 0.5      # down to half the minimum
BUDGET_NEAR_ABOVE = 2.0      # up to twice the maximum

# Location is a modifier, not a sixth factor: adding one would push company fit
# past 50 and silently move the 75/45 thresholds. Small magnitudes keep an
# optional criterion from overwhelming the four factors that are always present.
LOCATION_MATCH_BONUS = 2
LOCATION_MISMATCH_PENALTY = -3


def score_budget_against_target(
    resolution: dict[str, Any], profile: Any
) -> dict[str, Any]:
    """Compare the lead's stated budget interval to the target range.

    Unknown scores above off_target on purpose. A lead that never mentioned
    money is not the same as one that told you it has a tenth of your minimum,
    and scoring them identically would punish silence as if it were a bad
    answer.
    """
    measure = resolution["value"]
    if measure is None:
        return {
            "level": "unknown", "points": BUDGET_TARGET_POINTS["unknown"],
            "evidence": "not specified", "band_uncertain": False,
            "target_range": profile.budget_label(),
        }

    low = measure["low"]
    high = measure["high"] if measure["high"] is not None else float("inf")
    target_low, target_high = profile.budget_min, profile.budget_max

    if low <= target_high and high >= target_low:
        level = "on_target"
    elif high < target_low:
        level = "near_target" if high >= target_low * BUDGET_NEAR_BELOW else "off_target"
    else:
        level = "near_target" if low <= target_high * BUDGET_NEAR_ABOVE else "off_target"

    return {
        "level": level,
        "points": BUDGET_TARGET_POINTS[level],
        "evidence": measure["evidence"],
        "band_uncertain": measure["high"] is None,
        "target_range": profile.budget_label(),
    }


def score_location(extraction: dict[str, Any], profile: Any) -> dict[str, Any]:
    """The optional location criterion, as a bounded adjustment to company fit.

    Returns zero adjustment when no location was configured, and zero when the
    lead never stated one - an unstated location is unknown, and must never
    become an accidental disqualification.
    """
    if profile is None or not profile.has_location:
        return {"level": "not_configured", "adjustment": 0, "evidence": "not specified"}

    block = extraction.get("location_in_notes")
    stated = block.get("value") if isinstance(block, dict) and block.get("stated") else None
    evidence = (block or {}).get("evidence") or "not specified"

    verdict = profile.matches_location(stated)
    if verdict is None:
        return {"level": "unknown", "adjustment": 0, "evidence": evidence,
                "target_location": profile.location}
    if verdict:
        return {"level": "match", "adjustment": LOCATION_MATCH_BONUS,
                "evidence": evidence, "stated_location": stated,
                "target_location": profile.location}
    return {"level": "mismatch", "adjustment": LOCATION_MISMATCH_PENALTY,
            "evidence": evidence, "stated_location": stated,
            "target_location": profile.location}


def score_budget_capacity(resolution: dict[str, Any]) -> dict[str, Any]:
    """Map a resolved budget interval onto the §12 four-tier scale.

    Scored on the lower bound for the same reason as headcount. `<$1k/mo` and
    `unknown` both score 2 per §12, but the level string keeps them
    distinguishable: a recorded zero is evidence, a blank column is not.
    """
    measure = resolution["value"]
    if measure is None:
        return {
            "level": "unknown", "points": BUDGET_BANDS[0][2],
            "evidence": "not specified", "band_uncertain": False,
        }

    points = _band_value(BUDGET_BANDS, measure["low"])
    touched = _bands_touched(BUDGET_BANDS, measure["low"], measure["high"])
    return {
        "level": _budget_level_name(points),
        "points": points,
        "evidence": measure["evidence"],
        "band_uncertain": len(touched) > 1,
    }


def _budget_level_name(points: int) -> str:
    return {10: "high", 7: "medium", 4: "low", 2: "minimal"}[points]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _level_entry(extraction: dict[str, Any], factor: str, table: dict[str, int]) -> dict[str, Any]:
    """Look up one LLM-classified factor. Level and evidence pass through as-is."""
    block = extraction.get(factor) or {}
    level = block.get("level")
    evidence = block.get("evidence", "not specified")
    if level not in table:
        # An unrecognised level scores the table's minimum rather than crashing
        # the run; the level is preserved verbatim so the anomaly stays visible.
        return {
            "level": level, "points": min(table.values()),
            "evidence": evidence, "unrecognised_level": True,
        }
    return {"level": level, "points": table[level], "evidence": evidence}


def score_lead(
    extraction: dict[str, Any],
    clean_lead: dict[str, Any],
    profile: Any = None,
) -> dict[str, Any]:
    """Convert one extraction into points. Pure, deterministic, no I/O.

    `profile` is a `TargetProfile` or None. It changes what two fit factors
    mean - budget is compared to the operator's range, and location becomes a
    bounded adjustment - but never the point ceilings, the intent factors, or
    anything routing reads. With `profile=None` the result is identical to
    before the feature existed.
    """
    breakdown: dict[str, dict[str, Any]] = {}

    # --- company fit ------------------------------------------------------- #
    breakdown["industry_fit"] = _level_entry(extraction, "industry_fit", INDUSTRY_FIT_POINTS)
    if profile is not None:
        breakdown["industry_fit"]["target_industry"] = profile.industry

    size_resolution = resolve_measure(
        extraction, clean_lead, "employee_count_in_notes", "employees",
    )
    size_entry = score_company_size(size_resolution)
    size_entry.update({
        "source": size_resolution["source"],
        "conflict": size_resolution["conflict"],
        "notes_value": _describe(size_resolution["notes_value"]),
        "structured_value": _describe(size_resolution["structured_value"]),
    })
    breakdown["company_size_signal"] = size_entry

    breakdown["seniority"] = _level_entry(extraction, "seniority", SENIORITY_POINTS)

    budget_resolution = resolve_measure(
        extraction, clean_lead, "budget_in_notes", "budget",
    )
    # With a profile the question is "is this the budget we asked for"; without
    # one it stays the original absolute-size question, so every pre-existing
    # score is reproduced exactly.
    budget_entry = (
        score_budget_against_target(budget_resolution, profile)
        if profile is not None
        else score_budget_capacity(budget_resolution)
    )
    budget_entry.update({
        "source": budget_resolution["source"],
        "conflict": budget_resolution["conflict"],
        "notes_value": _describe(budget_resolution["notes_value"]),
        "structured_value": _describe(budget_resolution["structured_value"]),
    })
    breakdown["budget_capacity"] = budget_entry

    # Source comes from the cleaned CSV column, never from the LLM (§12).
    source_level = clean_lead.get("source_clean") or "unknown"
    if source_level not in SOURCE_POINTS:
        source_level = "unknown"
    breakdown["source"] = {
        "level": source_level,
        "points": SOURCE_POINTS[source_level],
        "evidence": clean_lead.get("source_raw") or source_level,
    }

    # --- buying intent ----------------------------------------------------- #
    breakdown["urgency"] = _level_entry(extraction, "urgency", URGENCY_POINTS)
    breakdown["pain_severity"] = _level_entry(extraction, "pain_severity", PAIN_SEVERITY_POINTS)

    readiness = _level_entry(extraction, "purchasing_readiness", PURCHASING_READINESS_POINTS)
    price_sensitive = bool(extraction.get("price_sensitive_flag"))
    if price_sensitive:
        # §12 applies the penalty to this factor specifically; the documented
        # floor is on the intent subtotal, not on the individual factor.
        readiness["points"] -= PRICE_SENSITIVE_PENALTY
        readiness["price_sensitive_penalty"] = -PRICE_SENSITIVE_PENALTY
    breakdown["purchasing_readiness"] = readiness

    breakdown["timeline"] = _level_entry(extraction, "timeline", TIMELINE_POINTS)
    breakdown["buying_stage"] = _level_entry(extraction, "buying_stage", BUYING_STAGE_POINTS)

    location_entry = score_location(extraction, profile)
    breakdown["location"] = location_entry

    raw_fit = sum(breakdown[f]["points"] for f in COMPANY_FIT_FACTORS)
    # Clamped to the documented ceiling: an optional criterion must not be able
    # to push company fit past 50 and move the 75/45 routing thresholds.
    company_fit_score = max(0, min(MAX_COMPANY_FIT, raw_fit + location_entry["adjustment"]))
    buying_intent_score = max(0, sum(breakdown[f]["points"] for f in BUYING_INTENT_FACTORS))

    return {
        "company_fit_score": company_fit_score,
        "buying_intent_score": buying_intent_score,
        "total_score": company_fit_score + buying_intent_score,
        "breakdown": {
            **{factor: breakdown[factor] for factor in FACTOR_ORDER},
            "location": breakdown["location"],
        },
    }


# --------------------------------------------------------------------------- #
# Explanation - templated, never generated by a model
# --------------------------------------------------------------------------- #

# (factor, level) -> phrase. Missing pairs fall back to a generic phrase, so a
# new level can never make the explanation raise.
_PHRASES: dict[tuple[str, str], str] = {
    ("industry_fit", "high"): "strong industry fit",
    ("industry_fit", "medium"): "adjacent industry fit",
    ("industry_fit", "low"): "weak industry fit",
    ("industry_fit", "unknown"): "industry fit unclear",
    ("company_size_signal", "high"): "company size in range",
    ("company_size_signal", "medium"): "company size outside the core range",
    ("company_size_signal", "low"): "company size well outside the core range",
    ("company_size_signal", "unknown"): "company size unknown",
    ("seniority", "decision_maker"): "decision maker",
    ("seniority", "high_influence"): "senior influencer",
    ("seniority", "low_influence"): "limited authority",
    ("seniority", "unknown"): "seniority unknown",
    ("budget_capacity", "high"): "strong budget",
    ("budget_capacity", "medium"): "moderate budget",
    ("budget_capacity", "low"): "small budget",
    ("budget_capacity", "minimal"): "little or no budget",
    ("budget_capacity", "unknown"): "budget not stated",
    ("budget_capacity", "on_target"): "budget in target range",
    ("budget_capacity", "near_target"): "budget near target range",
    ("budget_capacity", "off_target"): "budget outside target range",
    ("location", "match"): "in target location",
    ("location", "mismatch"): "outside target location",
    ("source", "event"): "from an event",
    ("source", "referral"): "referred",
    ("source", "linkedin"): "from LinkedIn",
    ("source", "webform"): "from the web form",
    ("source", "cold_reply"): "a cold reply",
    ("source", "unknown"): "source unknown",
    ("urgency", "high"): "high urgency",
    ("urgency", "medium"): "moderate urgency",
    ("urgency", "low"): "no urgency stated",
    ("pain_severity", "high"): "specific pain named",
    ("pain_severity", "medium"): "vague interest",
    ("pain_severity", "low"): "no problem named",
    ("purchasing_readiness", "approved"): "budget approved",
    ("purchasing_readiness", "partial"): "budget not locked",
    ("purchasing_readiness", "unknown"): "no budget signal",
    ("timeline", "near_term"): "near-term timeline",
    ("timeline", "medium_term"): "timeline about a month out",
    ("timeline", "none"): "no timeline stated",
    ("buying_stage", "committed"): "knows what they want",
    ("buying_stage", "evaluating"): "comparing options",
    ("buying_stage", "exploring"): "still exploring",
}

# A factor is a "strength" at these levels and a "drag" at these others.
_STRONG_LEVELS = {
    "industry_fit": {"high"},
    "company_size_signal": {"high"},
    "seniority": {"decision_maker"},
    "budget_capacity": {"high", "medium", "on_target"},
    "location": {"match"},
    "urgency": {"high"},
    "pain_severity": {"high"},
    "purchasing_readiness": {"approved"},
    "timeline": {"near_term"},
    "buying_stage": {"committed"},
}
_WEAK_LEVELS = {
    "industry_fit": {"low", "unknown"},
    "seniority": {"low_influence", "unknown"},
    "budget_capacity": {"minimal", "off_target"},
    "location": {"mismatch"},
    "urgency": {"low"},
    "pain_severity": {"low"},
    "purchasing_readiness": {"unknown"},
    "timeline": {"none"},
    "buying_stage": {"exploring"},
}

MAX_STRENGTHS = 3
MAX_DRAGS = 2
_EVIDENCE_MAX = 60


def explanation_parts(breakdown: dict[str, Any], recommendation: str) -> list[str]:
    """The explanation as discrete paragraphs, verdict first.

    `generate_explanation` joins these into one string. The UI renders them as
    separate paragraphs, which is why the parts are exposed rather than being
    recovered by splitting the joined text.
    """
    total = sum(
        entry.get("points", 0)
        for entry in breakdown.values()
        if isinstance(entry, dict)
    )
    ordered = _ordered_factors(breakdown)

    strengths = [
        _phrase_with_evidence(factor, breakdown[factor])
        for factor in ordered
        if breakdown[factor].get("level") in _STRONG_LEVELS.get(factor, set())
    ][:MAX_STRENGTHS]

    drags = [
        _phrase(factor, breakdown[factor])
        for factor in reversed(ordered)
        if breakdown[factor].get("level") in _WEAK_LEVELS.get(factor, set())
    ][:MAX_DRAGS]

    parts = [f"Total {total}/{MAX_TOTAL} → {recommendation}."]

    reasoning = []
    if strengths:
        reasoning.append(_sentence_case(", ".join(strengths)) + ".")
    if drags:
        reasoning.append("Held back by " + ", ".join(drags) + ".")
    if not strengths and not drags:
        reasoning.append("No standout signals either way.")
    location_line = _location_phrase(breakdown)
    if location_line:
        reasoning.append(location_line)
    parts.append(" ".join(reasoning))

    conflict = _conflict_note(breakdown)
    if conflict:
        parts.append(conflict)

    return parts


def _location_phrase(breakdown: dict[str, Any]) -> str:
    entry = breakdown.get("location")
    if not isinstance(entry, dict) or entry.get("level") not in {"match", "mismatch"}:
        return ""
    stated = entry.get("stated_location") or "location stated"
    target = entry.get("target_location")
    if entry["level"] == "match":
        return f"Located in the target area ({stated})."
    return f"Located outside the target area ({stated}; target is {target})."


def generate_explanation(breakdown: dict[str, Any], recommendation: str) -> str:
    """Build the summary sentence from a scored breakdown.

    Fully deterministic: the same breakdown and recommendation always produce
    byte-identical text. Ordering is by points descending with the canonical
    factor order as the tiebreak, so no dictionary iteration order can leak in.
    `recommendation` is supplied by the caller - this function never decides it.
    """
    return " ".join(explanation_parts(breakdown, recommendation))


def _ordered_factors(breakdown: dict[str, Any]) -> list[str]:
    present = [f for f in FACTOR_ORDER if isinstance(breakdown.get(f), dict)]
    return sorted(
        present,
        key=lambda f: (-breakdown[f].get("points", 0), FACTOR_ORDER.index(f)),
    )


def _phrase(factor: str, entry: dict[str, Any]) -> str:
    level = entry.get("level")
    text = _PHRASES.get((factor, level), f"{factor.replace('_', ' ')} {level}")
    if factor == "industry_fit" and level in {"low", "medium"} and entry.get("target_industry"):
        text = f"{text} (target profile is {entry['target_industry']})"
    return text


def _phrase_with_evidence(factor: str, entry: dict[str, Any]) -> str:
    text = _phrase(factor, entry)
    evidence = (entry.get("evidence") or "").strip()
    if evidence and evidence.lower() != "not specified":
        if len(evidence) > _EVIDENCE_MAX:
            evidence = evidence[:_EVIDENCE_MAX].rstrip() + "…"
        return f"{text} ({evidence})"
    return text


def _conflict_note(breakdown: dict[str, Any]) -> str:
    """Surface a notes-vs-source disagreement instead of hiding the resolution.

    Worded for the reader, not the system: "the original lead data" rather than
    "the CRM field", since this string is shown in the interface and exported in
    the CSV, where internal vocabulary means nothing to the person reading it.
    """
    notes = []
    for factor, label in (("company_size_signal", "Headcount"), ("budget_capacity", "Budget")):
        entry = breakdown.get(factor)
        if isinstance(entry, dict) and entry.get("conflict"):
            notes.append(
                f"{label} scored from the notes ({entry.get('notes_value')}); "
                f"the original lead data says {entry.get('structured_value')}."
            )
    return " ".join(notes)


def _sentence_case(text: str) -> str:
    return text[0].upper() + text[1:] if text else text
