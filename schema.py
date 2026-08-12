"""Single source of truth for the LLM extraction contract.

Every factor's allowed values are declared exactly once here. The JSON Schema
sent to the API is generated from these declarations, the prompt text is built
from them, and scoring.py (Stage 5) looks up points against the same constants.
That makes the plan's §11-vs-§12 enum drift - §11's example emits
`timeline: "within_1_month"` while §12's lookup table is keyed on `near_term` -
structurally impossible to reintroduce: a level the scorer cannot price is a
level the model was never offered.

Division of labour encoded here:

* Factors interpreted from unstructured text return a `level` (a semantic
  classification, which is the model's job).
* Numeric facts stated in the notes - headcount, budget - are returned as
  intervals in the same vocabulary `data_cleaning.Measure` uses, NOT as a
  rubric band. Bucketing a number is scoring, and scoring lives in Python.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

# Bump when the contract changes; participates in the extraction cache key so a
# schema edit can never be served a stale response shaped for the old contract.
SCHEMA_VERSION = "1.3"

# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #

BUYER_STATUS = ("yes", "no", "ambiguous")
"""Plan §8 needs three states. §11's example types this as a boolean, which
cannot express `ambiguous`; a string enum can."""

BUYER_TYPES = (
    "business_prospect",
    "freelancer_solo",
    "icp_adjacent_business",
    "job_seeker",
    "student_researcher",
    "journalist",
    "recruiter",
    "vc_intro",
    "competitor",
    "spam",
    "other",
)

NON_BUYER_SIGNALS = (
    "none",
    "competitive_research",
    "job_seeking",
    "academic_research",
    "media_enquiry",
    "recruiting",
    "vendor_pitch",
    "investor_intro",
    "free_resource_only",
    "spam",
)
"""What the writer STATES they came here to do, when that is not buying.

Deliberately a different question from `buyer_type`. `buyer_type` asks what kind
of person or company this is; this asks what they said their purpose was. The
distinction is what lets a peer agency buying for its own delivery team stay a
buyer (`none`) while a peer agency comparing prices to benchmark its own pricing
does not (`competitive_research`).

`none` is the answer for every ordinary enquiry, including a weak one. Absence
of budget, timeline, urgency or size is not a stated purpose, and neither is
being in a related industry."""

INDUSTRY_FIT_LEVELS = ("high", "medium", "low", "unknown")
SENIORITY_LEVELS = ("decision_maker", "high_influence", "low_influence", "unknown")
URGENCY_LEVELS = ("high", "medium", "low")
PAIN_SEVERITY_LEVELS = ("high", "medium", "low")
PURCHASING_READINESS_LEVELS = ("approved", "partial", "unknown")
TIMELINE_LEVELS = ("near_term", "medium_term", "none")
BUYING_STAGE_LEVELS = ("committed", "evaluating", "exploring")

MEASURE_KINDS = ("exact", "range", "floor", "approx", "missing")
"""Deliberately mirrors `data_cleaning.Measure.status`. A note saying "50+
people" and a CRM cell reading `50+` therefore arrive at the later scoring layer
in the same shape, and neither one invents an upper bound.

`missing` (not `none`) so the mirror is literal: the same word means the same
thing on both sides of the pipeline."""

# Factors the model classifies into a level. Ordered fit-first, then intent,
# matching the §7 scoring tables.
LEVEL_FACTORS: dict[str, tuple[str, ...]] = {
    "industry_fit": INDUSTRY_FIT_LEVELS,
    "seniority": SENIORITY_LEVELS,
    "urgency": URGENCY_LEVELS,
    "pain_severity": PAIN_SEVERITY_LEVELS,
    "purchasing_readiness": PURCHASING_READINESS_LEVELS,
    "timeline": TIMELINE_LEVELS,
    "buying_stage": BUYING_STAGE_LEVELS,
}

# Factors the model reports as an observed interval rather than a level.
# These are FACTS extracted from the notes. They are deliberately not bucketed,
# labelled or scored here - the later Python layer needs the raw bounds to make
# that call, and a pre-bucketed label would throw away the precision it needs.
MEASURE_FACTORS = ("employee_count_in_notes", "budget_in_notes")

# Extracted so the optional location criterion has something to compare against.
# Not a measure: it is a place name, not an interval.
LOCATION_FACTOR = "location_in_notes"

EVIDENCE_MAX_CHARS = 80
"""Evidence is a pointer back at the source phrase, not a justification. Not
enforced as a hard rejection - an over-long snippet is still usable - but
measured, so prompt drift is visible."""


# --------------------------------------------------------------------------- #
# JSON Schema generation
# --------------------------------------------------------------------------- #

def _level_object(levels: Iterable[str], description: str) -> dict[str, Any]:
    return {
        "type": "object",
        "description": description,
        "properties": {
            "level": {"type": "string", "enum": list(levels)},
            "evidence": {
                "type": "string",
                "description": (
                    "A few words quoted or closely paraphrased from the notes that "
                    "justify this level. Not a sentence, not an explanation. Use "
                    "'not specified' when the notes say nothing on this factor."
                ),
            },
        },
        "required": ["level", "evidence"],
        "additionalProperties": False,
    }


def _measure_object(description: str, unit_hint: str) -> dict[str, Any]:
    return {
        "type": "object",
        "description": description,
        "properties": {
            "stated": {
                "type": "boolean",
                "description": "True only if the notes state a specific figure.",
            },
            "low": {
                "anyOf": [{"type": "number"}, {"type": "null"}],
                "description": f"Lower bound {unit_hint}. Null when not stated.",
            },
            "high": {
                "anyOf": [{"type": "number"}, {"type": "null"}],
                "description": (
                    f"Upper bound {unit_hint}. Equal to `low` for an exact figure. "
                    "Null for an open-ended 'N+' floor - never invent a ceiling. "
                    "Null when not stated."
                ),
            },
            "kind": {
                "type": "string",
                "enum": list(MEASURE_KINDS),
                "description": (
                    "exact: one figure. range: two bounds, both kept. floor: 'N+' or "
                    "'at least N', high must be null. approx: 'around N'. "
                    "missing: not stated in the notes."
                ),
            },
            "evidence": {"type": "string"},
        },
        "required": ["stated", "low", "high", "kind", "evidence"],
        "additionalProperties": False,
    }


def _location_object() -> dict[str, Any]:
    return {
        "type": "object",
        "description": (
            "Where the lead's business is based, as stated IN THE NOTES or clearly "
            "implied by them (for example 'Nigerian agency' -> Nigeria). Do not "
            "guess from a company name, an email domain or a person's name. If the "
            "notes do not say, set stated=false - an unstated location is unknown, "
            "not a mismatch."
        ),
        "properties": {
            "stated": {"type": "boolean"},
            "value": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "Country, region or city as stated. Null when not stated.",
            },
            "evidence": {"type": "string"},
        },
        "required": ["stated", "value", "evidence"],
        "additionalProperties": False,
    }


def build_json_schema(profile: Any = None) -> dict[str, Any]:
    """The `output_config.format` schema for the extraction call.

    `profile` is a `TargetProfile` or None. When given, the industry-fit tiers
    are described relative to the operator's target industry instead of the
    hardcoded agency ICP this schema used to carry - which is what made the
    system a demonstration of one business rather than a reusable tool. The
    field names, enums and shapes are identical either way, so the contract
    downstream is unchanged.
    """
    properties: dict[str, Any] = {
        "is_potential_buyer": {
            "type": "string",
            "enum": list(BUYER_STATUS),
            "description": (
                "'no' only for a hard-disqualify category. 'ambiguous' when the notes "
                "genuinely do not settle it. Weak or off-ICP buyers are still 'yes'."
            ),
        },
        "buyer_type": {"type": "string", "enum": list(BUYER_TYPES)},
        "non_buyer_signal": {
            "type": "string",
            "enum": list(NON_BUYER_SIGNALS),
            "description": (
                "What the writer STATES they are here to do, when that purpose is "
                "not buying. This is about their stated purpose, NOT about what "
                "kind of person or company they are. "
                "competitive_research: comparing your pricing or approach to "
                "benchmark their own offering. job_seeking: looking for work. "
                "academic_research: a dissertation, thesis or study. "
                "media_enquiry: writing an article or seeking a comment. "
                "recruiting: hiring, or asking you to fill a role. "
                "vendor_pitch: selling you their own product or service. "
                "investor_intro: offering an introduction or investment. "
                "free_resource_only: after a free guide, template or sample and "
                "nothing more. spam: junk or automated. "
                "Use 'none' for every genuine commercial enquiry, including weak "
                "ones. Weak intent, no budget, no timeline, still exploring, a "
                "small company, or working in the same industry as you are NOT "
                "non-buying purposes - they are 'none'."
            ),
        },
        "disqualification_evidence": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": (
                "Short phrase from the notes supporting a 'no' or a non_buyer_signal "
                "other than 'none'. Quote the notes; never invent it. Null otherwise."
            ),
        },
        "ambiguity_reason": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": (
                "One short clause naming what is unresolved. Required when "
                "is_potential_buyer is 'ambiguous'; null otherwise."
            ),
        },
    }

    if profile is not None:
        industry_description = (
            f"How closely the lead's own business matches the target industry: "
            f"{profile.industry}. "
            "high: the lead is in that industry, or a direct sub-speciality of it. "
            "medium: an adjacent or overlapping industry that plausibly has the "
            "same need. low: a clearly different industry. unknown: the notes do "
            "not say what business they are in. "
            "Judge from what the notes actually describe - a passing mention of a "
            "related word is not the same as being in that industry."
        )
    else:
        industry_description = (
            "How commercially relevant the lead's business is. "
            "high: core target customer. medium: adjacent commercial buyer. "
            "low: unrelated. unknown: the notes do not say."
        )
    properties["industry_fit"] = _level_object(INDUSTRY_FIT_LEVELS, industry_description)
    properties["employee_count_in_notes"] = _measure_object(
        "Headcount as stated IN THE NOTES ONLY. The CRM field is shown to you as "
        "context and the two often disagree; do not reconcile them and do not copy "
        "the CRM value here. Report only what the notes assert. Which source wins "
        "is decided later, outside this step.",
        "in employees",
    )
    properties["budget_in_notes"] = _measure_object(
        "Monthly budget in USD as stated IN THE NOTES ONLY. Same rule as headcount: "
        "do not copy or reconcile the CRM field. 'no budget' / 'tiny budget' stated "
        "as zero is stated=true with low=0 and high=0.",
        "in USD per month",
    )
    properties["seniority"] = _level_object(
        SENIORITY_LEVELS,
        "decision_maker: founder/CEO/owner/COO/partner/MD, or the notes say they "
        "decide. high_influence: VP or head of. low_influence: consultant, "
        "freelancer, developer, individual contributor. unknown: no signal.",
    )
    properties["urgency"] = _level_object(
        URGENCY_LEVELS,
        "high: ASAP, ready to pilot, moving fast, priority this quarter. "
        "medium: comparing options, actively looking. low: no urgency stated.",
    )
    properties["pain_severity"] = _level_object(
        PAIN_SEVERITY_LEVELS,
        "high: a specific named problem with stated cost, e.g. 'eating our week'. "
        "medium: a vague or general interest. low: no problem named.",
    )
    properties["purchasing_readiness"] = _level_object(
        PURCHASING_READINESS_LEVELS,
        "approved: budget approved or signed off. partial: some budget, not locked. "
        "unknown: no budget signal, or explicitly none.",
    )
    properties["timeline"] = _level_object(
        TIMELINE_LEVELS,
        "near_term: this month or within a few weeks. medium_term: roughly a month "
        "out or longer, or a decision cycle is described. none: no timeline stated.",
    )
    properties["buying_stage"] = _level_object(
        BUYING_STAGE_LEVELS,
        "committed: knows what they want, e.g. 'want it automated end to end'. "
        "evaluating: comparing a few options. exploring: unsure what they need yet.",
    )
    properties[LOCATION_FACTOR] = _location_object()
    properties["price_sensitive_flag"] = {
        "type": "boolean",
        "description": "True only if the notes explicitly signal price sensitivity.",
    }

    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

class ExtractionValidationError(ValueError):
    """Raised when a response does not satisfy the contract above."""


def validate_extraction(data: Any) -> list[str]:
    """Return a list of contract violations. Empty list means valid.

    Structured outputs already guarantee the shape, so this is a belt-and-braces
    check that also covers the semantic rules a JSON Schema cannot express (a
    `floor` measure must not carry an upper bound; an `ambiguous` verdict must
    say what is unresolved).
    """
    problems: list[str] = []
    if not isinstance(data, dict):
        return ["response is not a JSON object"]

    status = data.get("is_potential_buyer")
    if status not in BUYER_STATUS:
        problems.append(f"is_potential_buyer={status!r} not in {BUYER_STATUS}")
    if data.get("buyer_type") not in BUYER_TYPES:
        problems.append(f"buyer_type={data.get('buyer_type')!r} not recognised")
    if status == "ambiguous" and not _nonempty(data.get("ambiguity_reason")):
        problems.append("is_potential_buyer='ambiguous' requires ambiguity_reason")

    signal = data.get("non_buyer_signal")
    if signal not in NON_BUYER_SIGNALS:
        problems.append(f"non_buyer_signal={signal!r} not in {NON_BUYER_SIGNALS}")
    elif signal != "none" and not _nonempty(data.get("disqualification_evidence")):
        # A stated purpose the notes cannot be quoted for is an inference, and an
        # inference must not be able to disqualify anyone.
        problems.append(
            f"non_buyer_signal={signal!r} requires disqualification_evidence"
        )

    for factor, levels in LEVEL_FACTORS.items():
        block = data.get(factor)
        if not isinstance(block, dict):
            problems.append(f"{factor}: missing or not an object")
            continue
        if block.get("level") not in levels:
            problems.append(f"{factor}.level={block.get('level')!r} not in {levels}")
        if not _nonempty(block.get("evidence")):
            problems.append(f"{factor}.evidence is empty")

    for factor in MEASURE_FACTORS:
        problems.extend(_validate_measure(factor, data.get(factor)))
    problems.extend(_validate_location(data.get(LOCATION_FACTOR)))

    if not isinstance(data.get("price_sensitive_flag"), bool):
        problems.append("price_sensitive_flag must be a boolean")

    return problems


def _validate_location(block: Any) -> list[str]:
    """A location is unknown or stated - never both, never neither."""
    if not isinstance(block, dict):
        return [f"{LOCATION_FACTOR}: missing or not an object"]
    problems: list[str] = []
    stated, value = block.get("stated"), block.get("value")
    if not isinstance(stated, bool):
        problems.append(f"{LOCATION_FACTOR}.stated must be a boolean")
    if stated and not _nonempty(value):
        problems.append(f"{LOCATION_FACTOR}: stated=true requires a value")
    if stated is False and _nonempty(value):
        problems.append(f"{LOCATION_FACTOR}: stated=false must leave value null")
    if not _nonempty(block.get("evidence")):
        problems.append(f"{LOCATION_FACTOR}.evidence is empty")
    return problems


def _validate_measure(factor: str, block: Any) -> list[str]:
    if not isinstance(block, dict):
        return [f"{factor}: missing or not an object"]

    problems: list[str] = []
    kind, stated = block.get("kind"), block.get("stated")
    low, high = block.get("low"), block.get("high")

    if kind not in MEASURE_KINDS:
        problems.append(f"{factor}.kind={kind!r} not in {MEASURE_KINDS}")
    if not isinstance(stated, bool):
        problems.append(f"{factor}.stated must be a boolean")

    if kind == "missing" or stated is False:
        if low is not None or high is not None:
            problems.append(f"{factor}: nothing stated, so low/high must both be null")
        return problems

    if low is None:
        problems.append(f"{factor}: kind={kind!r} requires a lower bound")
    if kind == "floor" and high is not None:
        # The whole point of a floor: the source gives no ceiling, so neither do we.
        problems.append(f"{factor}: kind='floor' must leave high null (got {high!r})")
    if kind in {"exact", "approx"} and low is not None and high != low:
        problems.append(f"{factor}: kind={kind!r} requires high == low")
    if kind == "range":
        if high is None:
            problems.append(f"{factor}: kind='range' requires an upper bound")
        elif low is not None and high < low:
            problems.append(f"{factor}: range bounds are inverted")
        elif low is not None and high == low:
            # A range whose bounds are equal is a midpoint wearing a range's
            # label - exactly the collapse the interval design exists to prevent.
            problems.append(
                f"{factor}: kind='range' with low == high looks like a collapsed "
                f"midpoint; report the two stated bounds, or use kind='exact'"
            )
    return problems


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


ASSESSMENT_FAILED_REASON = "This lead could not be assessed automatically."
"""The only reason string a failed extraction may carry.

Deliberately takes no argument from the caller. The previous signature accepted
a `reason` and wrote it straight into `ambiguity_reason`, which is how a
stringified Gemini 429 - HTTP code, quota metric, retry timing and all - ended
up rendered to users under "Why this needs a person". The technical detail
belongs in `ExtractionResult.error` and the logs, never in the record.
"""


def unknown_extraction() -> dict[str, Any]:
    """A fully-unknown record, used when the API call cannot be completed.

    Per plan §16 Stage 4 the batch must never crash on one bad lead. This keeps
    the row in the pipeline, scoring at the floor, flagged for manual review -
    rather than silently vanishing from the ranked output.

    `is_potential_buyer` stays "ambiguous" so routing is unchanged (§8/§9 send
    it to REVIEW). What distinguishes a failure from genuine ambiguity is
    `ExtractionResult.status == "failed"` plus `failure_code`, carried through
    the pipeline onto the routed record - not a different route.

    `non_buyer_signal` is "none" for the same reason: never having read the
    notes is not evidence of a non-buying purpose, and a lead the system failed
    to assess must not be excluded on the strength of that failure.
    """
    record: dict[str, Any] = {
        "is_potential_buyer": "ambiguous",
        "buyer_type": "other",
        "non_buyer_signal": "none",
        "disqualification_evidence": None,
        "ambiguity_reason": ASSESSMENT_FAILED_REASON,
        "price_sensitive_flag": False,
    }
    fallbacks = {
        "industry_fit": "unknown",
        "seniority": "unknown",
        "urgency": "low",
        "pain_severity": "low",
        "purchasing_readiness": "unknown",
        "timeline": "none",
        "buying_stage": "exploring",
    }
    for factor, level in fallbacks.items():
        record[factor] = {"level": level, "evidence": "not specified"}
    for factor in MEASURE_FACTORS:
        record[factor] = {
            "stated": False, "low": None, "high": None,
            "kind": "missing", "evidence": "not specified",
        }
    record[LOCATION_FACTOR] = {"stated": False, "value": None, "evidence": "not specified"}
    return record


def evidence_fields(data: dict[str, Any]) -> dict[str, Optional[str]]:
    """Flatten every factor's evidence snippet, for the CSV export and the UI."""
    out: dict[str, Optional[str]] = {}
    for factor in list(LEVEL_FACTORS) + list(MEASURE_FACTORS) + [LOCATION_FACTOR]:
        block = data.get(factor)
        out[factor] = block.get("evidence") if isinstance(block, dict) else None
    return out
