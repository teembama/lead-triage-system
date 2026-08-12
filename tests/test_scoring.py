"""Stage 5 unit tests - lookup tables, precedence, boundaries, determinism.

Pure functions only: no API key, no network, no model. Every point value in the
architecture's §12 tables is asserted explicitly, so a silent edit to a lookup
table fails a named test rather than quietly shifting every lead's score.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scoring  # noqa: E402
from scoring import generate_explanation, score_lead  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def extraction(**overrides):
    """A mid-scoring extraction; individual tests override one factor at a time."""
    payload = {
        "is_potential_buyer": "yes",
        "buyer_type": "business_prospect",
        "disqualification_evidence": None,
        "ambiguity_reason": None,
        "industry_fit": {"level": "medium", "evidence": "SaaS company"},
        "employee_count_in_notes": {
            "stated": False, "low": None, "high": None,
            "kind": "missing", "evidence": "not specified",
        },
        "budget_in_notes": {
            "stated": False, "low": None, "high": None,
            "kind": "missing", "evidence": "not specified",
        },
        "location_in_notes": {"stated": False, "value": None, "evidence": "not specified"},
        "seniority": {"level": "high_influence", "evidence": "VP Growth"},
        "urgency": {"level": "medium", "evidence": "comparing options"},
        "pain_severity": {"level": "medium", "evidence": "vague interest"},
        "purchasing_readiness": {"level": "partial", "evidence": "budget not locked"},
        "timeline": {"level": "medium_term", "evidence": "about a month"},
        "buying_stage": {"level": "evaluating", "evidence": "comparing a few options"},
        "price_sensitive_flag": False,
    }
    payload.update(overrides)
    return payload


def clean_lead(**overrides):
    lead = {
        "lead_id": "L-1",
        "source_clean": "webform",
        "source_raw": "webform",
        "employees_low": None, "employees_high": None,
        "employees_status": "missing", "employees_display": "unknown",
        "budget_low": None, "budget_high": None,
        "budget_status": "missing", "budget_display": "unknown",
    }
    lead.update(overrides)
    return lead


def points_for(factor, result):
    return result["breakdown"][factor]["points"]


# --------------------------------------------------------------------------- #
# Every level in every lookup table
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("level,expected", [("high", 20), ("medium", 11), ("low", 3)])
def test_industry_fit_table(level, expected):
    result = score_lead(extraction(industry_fit={"level": level, "evidence": "x"}), clean_lead())
    assert points_for("industry_fit", result) == expected


@pytest.mark.parametrize(
    "level,expected",
    [("decision_maker", 10), ("high_influence", 6), ("low_influence", 3), ("unknown", 2)],
)
def test_seniority_table(level, expected):
    result = score_lead(extraction(seniority={"level": level, "evidence": "x"}), clean_lead())
    assert points_for("seniority", result) == expected


@pytest.mark.parametrize(
    "source,expected",
    [("event", 5), ("referral", 5), ("linkedin", 5), ("webform", 3),
     ("cold_reply", 1), ("unknown", 2)],
)
def test_source_table(source, expected):
    result = score_lead(extraction(), clean_lead(source_clean=source))
    assert points_for("source", result) == expected


def test_source_comes_from_clean_data_not_the_llm():
    """§12 is explicit that source is read from the CSV, never classified."""
    result = score_lead(extraction(source={"level": "event", "evidence": "x"}),
                        clean_lead(source_clean="cold_reply"))
    assert points_for("source", result) == 1


@pytest.mark.parametrize("level,expected", [("high", 15), ("medium", 8), ("low", 2)])
def test_urgency_table(level, expected):
    result = score_lead(extraction(urgency={"level": level, "evidence": "x"}), clean_lead())
    assert points_for("urgency", result) == expected


@pytest.mark.parametrize("level,expected", [("high", 12), ("medium", 6), ("low", 2)])
def test_pain_severity_table(level, expected):
    result = score_lead(extraction(pain_severity={"level": level, "evidence": "x"}), clean_lead())
    assert points_for("pain_severity", result) == expected


@pytest.mark.parametrize("level,expected", [("approved", 10), ("partial", 5), ("unknown", 1)])
def test_purchasing_readiness_table(level, expected):
    result = score_lead(
        extraction(purchasing_readiness={"level": level, "evidence": "x"}), clean_lead(),
    )
    assert points_for("purchasing_readiness", result) == expected


@pytest.mark.parametrize("level,expected", [("near_term", 8), ("medium_term", 4), ("none", 0)])
def test_timeline_table(level, expected):
    result = score_lead(extraction(timeline={"level": level, "evidence": "x"}), clean_lead())
    assert points_for("timeline", result) == expected


@pytest.mark.parametrize("level,expected", [("committed", 5), ("evaluating", 3), ("exploring", 1)])
def test_buying_stage_table(level, expected):
    result = score_lead(extraction(buying_stage={"level": level, "evidence": "x"}), clean_lead())
    assert points_for("buying_stage", result) == expected


@pytest.mark.parametrize(
    "level,expected", [("high", 5), ("medium", 3), ("low", 2), ("unknown", 1)],
)
def test_company_size_table_values(level, expected):
    assert scoring.COMPANY_SIZE_POINTS[level] == expected


# --------------------------------------------------------------------------- #
# Company size bands
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "headcount,level,expected",
    [
        (1, "medium", 3), (9, "medium", 3),
        (10, "high", 5), (26, "high", 5), (70, "high", 5),
        (71, "medium", 3), (200, "medium", 3),
        (201, "low", 2), (5000, "low", 2),
    ],
)
def test_company_size_bands(headcount, level, expected):
    result = score_lead(
        extraction(employee_count_in_notes={
            "stated": True, "low": headcount, "high": headcount,
            "kind": "exact", "evidence": f"{headcount} people",
        }),
        clean_lead(),
    )
    entry = result["breakdown"]["company_size_signal"]
    assert (entry["level"], entry["points"]) == (level, expected)


def test_company_size_band_boundaries_exactly():
    """The 9/10 and 70/71 edges decide 3 vs 5 points."""
    def band(n):
        return score_lead(
            extraction(employee_count_in_notes={
                "stated": True, "low": n, "high": n, "kind": "exact", "evidence": "x",
            }),
            clean_lead(),
        )["breakdown"]["company_size_signal"]["points"]

    assert (band(9), band(10)) == (3, 5)
    assert (band(70), band(71)) == (5, 3)
    assert (band(200), band(201)) == (3, 2)


def test_unknown_headcount_scores_low_but_not_zero():
    """Plan §5: missing data is low-but-not-zero, and labelled missing rather
    than treated as confirmed small."""
    result = score_lead(extraction(), clean_lead())
    entry = result["breakdown"]["company_size_signal"]
    assert (entry["level"], entry["points"]) == ("unknown", 1)


def test_floor_headcount_is_scored_on_its_stated_lower_bound():
    """`51+` asserts 51 and nothing more; scoring must not invent a ceiling."""
    result = score_lead(
        extraction(employee_count_in_notes={
            "stated": True, "low": 51, "high": None, "kind": "floor",
            "evidence": "51+ people",
        }),
        clean_lead(),
    )
    entry = result["breakdown"]["company_size_signal"]
    assert (entry["level"], entry["points"]) == ("high", 5)
    assert entry["band_uncertain"] is True   # 51+ also reaches the 71-200 band


def test_range_within_one_band_is_not_flagged_uncertain():
    result = score_lead(
        extraction(employee_count_in_notes={
            "stated": True, "low": 35, "high": 55, "kind": "range",
            "evidence": "35-55 people",
        }),
        clean_lead(),
    )
    entry = result["breakdown"]["company_size_signal"]
    assert (entry["level"], entry["band_uncertain"]) == ("high", False)


def test_range_straddling_bands_is_flagged_uncertain():
    result = score_lead(
        extraction(employee_count_in_notes={
            "stated": True, "low": 60, "high": 90, "kind": "range", "evidence": "60-90",
        }),
        clean_lead(),
    )
    entry = result["breakdown"]["company_size_signal"]
    assert (entry["level"], entry["band_uncertain"]) == ("high", True)


# --------------------------------------------------------------------------- #
# Budget boundaries (explicitly required)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "amount,expected",
    [(999, 2), (1000, 4), (3999, 4), (4000, 7), (7999, 7), (8000, 10)],
)
def test_budget_boundaries(amount, expected):
    result = score_lead(
        extraction(),
        clean_lead(budget_low=amount, budget_high=amount,
                   budget_status="exact", budget_display=str(amount)),
    )
    assert points_for("budget_capacity", result) == expected


@pytest.mark.parametrize("amount,expected", [(0, 2), (500, 2), (12000, 10), (100000, 10)])
def test_budget_outside_the_boundary_cases(amount, expected):
    result = score_lead(
        extraction(),
        clean_lead(budget_low=amount, budget_high=amount,
                   budget_status="exact", budget_display=str(amount)),
    )
    assert points_for("budget_capacity", result) == expected


def test_unknown_budget_scores_two():
    result = score_lead(extraction(), clean_lead())
    entry = result["breakdown"]["budget_capacity"]
    assert (entry["level"], entry["points"]) == ("unknown", 2)


def test_recorded_zero_budget_is_distinguishable_from_unknown():
    """Both score 2 per §12, but 'no budget' is evidence and blank is not."""
    zero = score_lead(extraction(), clean_lead(
        budget_low=0, budget_high=0, budget_status="zero", budget_display="0",
    ))["breakdown"]["budget_capacity"]
    unknown = score_lead(extraction(), clean_lead())["breakdown"]["budget_capacity"]
    assert zero["points"] == unknown["points"] == 2
    assert zero["level"] != unknown["level"]


def test_budget_range_is_scored_on_its_lower_bound():
    """`6k-8k` scores the 4000-7999 tier, not the 8000+ tier it merely touches."""
    result = score_lead(
        extraction(),
        clean_lead(budget_low=6000, budget_high=8000,
                   budget_status="range", budget_display="6000-8000"),
    )
    entry = result["breakdown"]["budget_capacity"]
    assert entry["points"] == 7
    assert entry["band_uncertain"] is True


# --------------------------------------------------------------------------- #
# Data precedence (plan §2)
# --------------------------------------------------------------------------- #

def test_notes_headcount_takes_precedence_over_a_conflicting_crm_field():
    """The NimbleMedia case: CRM says 9 (medium), notes say 23 (high)."""
    result = score_lead(
        extraction(employee_count_in_notes={
            "stated": True, "low": 23, "high": 23, "kind": "exact", "evidence": "23 people",
        }),
        clean_lead(employees_low=9, employees_high=9,
                   employees_status="exact", employees_display="9"),
    )
    entry = result["breakdown"]["company_size_signal"]
    assert entry["points"] == 5
    assert entry["source"] == "notes"


def test_the_conflict_stays_visible_rather_than_being_overwritten():
    result = score_lead(
        extraction(employee_count_in_notes={
            "stated": True, "low": 23, "high": 23, "kind": "exact", "evidence": "23 people",
        }),
        clean_lead(employees_low=9, employees_high=9,
                   employees_status="exact", employees_display="9"),
    )
    entry = result["breakdown"]["company_size_signal"]
    assert entry["conflict"] is True
    assert entry["notes_value"] == "23"
    assert entry["structured_value"] == "9"     # the CRM value survives


def test_notes_fill_a_blank_crm_field():
    result = score_lead(
        extraction(employee_count_in_notes={
            "stated": True, "low": 26, "high": 26, "kind": "exact", "evidence": "26 people",
        }),
        clean_lead(),
    )
    entry = result["breakdown"]["company_size_signal"]
    assert (entry["source"], entry["points"]) == ("notes", 5)
    assert entry["conflict"] is False           # nothing to disagree with


def test_crm_field_is_used_when_the_notes_say_nothing():
    result = score_lead(
        extraction(),
        clean_lead(employees_low=40, employees_high=40,
                   employees_status="exact", employees_display="40"),
    )
    entry = result["breakdown"]["company_size_signal"]
    assert (entry["source"], entry["points"]) == ("structured", 5)


def test_agreeing_sources_are_not_reported_as_a_conflict():
    result = score_lead(
        extraction(employee_count_in_notes={
            "stated": True, "low": 40, "high": 40, "kind": "exact", "evidence": "40 people",
        }),
        clean_lead(employees_low=40, employees_high=40,
                   employees_status="exact", employees_display="40"),
    )
    assert result["breakdown"]["company_size_signal"]["conflict"] is False


def test_notes_budget_takes_precedence_over_the_crm_field():
    result = score_lead(
        extraction(budget_in_notes={
            "stated": True, "low": 9000, "high": 9000, "kind": "exact",
            "evidence": "$9k/mo approved",
        }),
        clean_lead(budget_low=2000, budget_high=2000,
                   budget_status="exact", budget_display="2000"),
    )
    entry = result["breakdown"]["budget_capacity"]
    assert (entry["points"], entry["source"], entry["conflict"]) == (10, "notes", True)


def test_nan_from_a_pandas_column_is_not_read_as_a_real_figure():
    """Regression: a blank CRM column arrives as float('nan') after a DataFrame
    round-trip, and `isinstance(nan, float)` is True. Without an explicit guard
    that reads as a real value and manufactures a phantom conflict."""
    nan = float("nan")
    result = score_lead(
        extraction(employee_count_in_notes={
            "stated": True, "low": 26, "high": 26, "kind": "exact", "evidence": "26 people",
        }),
        clean_lead(employees_low=nan, employees_high=nan,
                   employees_status="missing", employees_display="unknown"),
    )
    entry = result["breakdown"]["company_size_signal"]
    assert entry["conflict"] is False, "a blank CRM column is not a conflicting value"
    assert entry["structured_value"] == "unknown"
    assert (entry["source"], entry["points"]) == ("notes", 5)


def test_nan_alone_scores_as_unknown_not_as_zero():
    nan = float("nan")
    result = score_lead(extraction(), clean_lead(budget_low=nan, budget_high=nan))
    entry = result["breakdown"]["budget_capacity"]
    assert (entry["level"], entry["points"]) == ("unknown", 2)


def test_real_dataframe_rows_round_trip_without_phantom_conflicts():
    """Guards the integration path the unit fixtures cannot: the actual cleaned
    frame, with its pandas dtypes, must not produce spurious conflicts."""
    import os

    import pandas as pd

    from data_cleaning import clean_dataframe, read_leads_csv

    csv = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Cohort 3 Assessment — Task 1 Leads (messy).csv",
    )
    if not os.path.exists(csv):
        pytest.skip("assessment CSV not present")

    clean_df, _ = clean_dataframe(read_leads_csv(csv))
    rows = clean_df.to_dict("records")[:40]
    for row in rows:
        if pd.isna(row["employees_low"]):
            result = score_lead(extraction(), row)
            entry = result["breakdown"]["company_size_signal"]
            assert entry["conflict"] is False
            assert entry["level"] == "unknown"


def test_a_notes_block_with_stated_false_is_not_treated_as_zero():
    """Silence must never be read as a confirmed zero budget."""
    result = score_lead(
        extraction(budget_in_notes={
            "stated": False, "low": None, "high": None,
            "kind": "missing", "evidence": "not specified",
        }),
        clean_lead(budget_low=8000, budget_high=8000,
                   budget_status="exact", budget_display="8000"),
    )
    entry = result["breakdown"]["budget_capacity"]
    assert (entry["source"], entry["points"]) == ("structured", 10)


# --------------------------------------------------------------------------- #
# Price-sensitivity penalty
# --------------------------------------------------------------------------- #

def test_price_sensitive_penalty_applies_to_purchasing_readiness():
    base = score_lead(
        extraction(purchasing_readiness={"level": "approved", "evidence": "Budget approved"}),
        clean_lead(),
    )
    penalised = score_lead(
        extraction(
            purchasing_readiness={"level": "approved", "evidence": "Budget approved"},
            price_sensitive_flag=True,
        ),
        clean_lead(),
    )
    assert points_for("purchasing_readiness", base) == 10
    assert points_for("purchasing_readiness", penalised) == 8
    assert penalised["buying_intent_score"] == base["buying_intent_score"] - 2


def test_price_sensitive_penalty_is_recorded_in_the_breakdown():
    result = score_lead(
        extraction(price_sensitive_flag=True), clean_lead(),
    )
    assert result["breakdown"]["purchasing_readiness"]["price_sensitive_penalty"] == -2


def test_no_penalty_when_the_flag_is_false():
    result = score_lead(extraction(price_sensitive_flag=False), clean_lead())
    assert "price_sensitive_penalty" not in result["breakdown"]["purchasing_readiness"]


def test_penalty_applies_to_every_readiness_level():
    for level, base_points in (("approved", 10), ("partial", 5), ("unknown", 1)):
        result = score_lead(
            extraction(
                purchasing_readiness={"level": level, "evidence": "x"},
                price_sensitive_flag=True,
            ),
            clean_lead(),
        )
        assert points_for("purchasing_readiness", result) == base_points - 2


def test_intent_subtotal_is_floored_at_zero():
    """§12 floors the intent subtotal, not the individual factor."""
    result = score_lead(
        extraction(
            urgency={"level": "low", "evidence": "x"},
            pain_severity={"level": "low", "evidence": "x"},
            purchasing_readiness={"level": "unknown", "evidence": "x"},
            timeline={"level": "none", "evidence": "x"},
            buying_stage={"level": "exploring", "evidence": "x"},
            price_sensitive_flag=True,
        ),
        clean_lead(),
    )
    assert result["buying_intent_score"] >= 0


# --------------------------------------------------------------------------- #
# Totals
# --------------------------------------------------------------------------- #

def test_maximum_company_fit_is_fifty():
    result = score_lead(
        extraction(
            industry_fit={"level": "high", "evidence": "marketing agency"},
            employee_count_in_notes={
                "stated": True, "low": 26, "high": 26, "kind": "exact", "evidence": "26 people",
            },
            seniority={"level": "decision_maker", "evidence": "I make the call"},
        ),
        clean_lead(source_clean="referral", budget_low=12000, budget_high=12000,
                   budget_status="exact", budget_display="12000"),
    )
    assert result["company_fit_score"] == 50 == scoring.MAX_COMPANY_FIT


def test_maximum_buying_intent_is_fifty():
    result = score_lead(
        extraction(
            urgency={"level": "high", "evidence": "ASAP"},
            pain_severity={"level": "high", "evidence": "eating our week"},
            purchasing_readiness={"level": "approved", "evidence": "Budget approved"},
            timeline={"level": "near_term", "evidence": "2 weeks"},
            buying_stage={"level": "committed", "evidence": "end to end"},
        ),
        clean_lead(),
    )
    assert result["buying_intent_score"] == 50 == scoring.MAX_BUYING_INTENT


def test_maximum_total_is_one_hundred():
    result = score_lead(
        extraction(
            industry_fit={"level": "high", "evidence": "marketing agency"},
            employee_count_in_notes={
                "stated": True, "low": 26, "high": 26, "kind": "exact", "evidence": "26 people",
            },
            seniority={"level": "decision_maker", "evidence": "I make the call"},
            urgency={"level": "high", "evidence": "ASAP"},
            pain_severity={"level": "high", "evidence": "eating our week"},
            purchasing_readiness={"level": "approved", "evidence": "Budget approved"},
            timeline={"level": "near_term", "evidence": "2 weeks"},
            buying_stage={"level": "committed", "evidence": "end to end"},
        ),
        clean_lead(source_clean="event", budget_low=12000, budget_high=12000,
                   budget_status="exact", budget_display="12000"),
    )
    assert (result["company_fit_score"], result["buying_intent_score"]) == (50, 50)
    assert result["total_score"] == 100 == scoring.MAX_TOTAL


def test_minimum_scores_are_not_negative():
    result = score_lead(
        extraction(
            industry_fit={"level": "low", "evidence": "x"},
            seniority={"level": "low_influence", "evidence": "x"},
            urgency={"level": "low", "evidence": "x"},
            pain_severity={"level": "low", "evidence": "x"},
            purchasing_readiness={"level": "unknown", "evidence": "x"},
            timeline={"level": "none", "evidence": "x"},
            buying_stage={"level": "exploring", "evidence": "x"},
        ),
        clean_lead(source_clean="cold_reply"),
    )
    assert result["company_fit_score"] >= 0
    assert result["buying_intent_score"] >= 0
    assert result["total_score"] == result["company_fit_score"] + result["buying_intent_score"]


def test_total_is_always_the_sum_of_the_two_subtotals():
    result = score_lead(extraction(), clean_lead())
    assert result["total_score"] == result["company_fit_score"] + result["buying_intent_score"]


def test_subtotals_are_the_sum_of_their_own_factors():
    result = score_lead(extraction(), clean_lead())
    fit = sum(result["breakdown"][f]["points"] for f in scoring.COMPANY_FIT_FACTORS)
    intent = sum(result["breakdown"][f]["points"] for f in scoring.BUYING_INTENT_FACTORS)
    assert result["company_fit_score"] == fit
    assert result["buying_intent_score"] == intent


def test_worked_example_totals_exactly():
    """A full hand-computed lead, so an arithmetic slip anywhere is caught.

    fit    : 20 industry + 5 size + 10 seniority + 10 budget + 5 source = 50
    intent : 15 urgency + 12 pain + 10 readiness + 8 timeline + 5 stage  = 50
    """
    result = score_lead(
        extraction(
            industry_fit={"level": "high", "evidence": "full-service marketing agency"},
            employee_count_in_notes={
                "stated": True, "low": 23, "high": 23, "kind": "exact", "evidence": "23 people",
            },
            seniority={"level": "decision_maker", "evidence": "I make the call here"},
            urgency={"level": "high", "evidence": "ready to pilot in 2 weeks"},
            pain_severity={"level": "high", "evidence": "eating our week"},
            purchasing_readiness={"level": "approved", "evidence": "Budget approved"},
            timeline={"level": "near_term", "evidence": "next 2 weeks"},
            buying_stage={"level": "committed", "evidence": "automated end to end"},
        ),
        clean_lead(source_clean="referral", budget_low=18000, budget_high=18000,
                   budget_status="exact", budget_display="18000"),
    )
    assert result == {
        "company_fit_score": 50,
        "buying_intent_score": 50,
        "total_score": 100,
        "breakdown": result["breakdown"],
    }


# --------------------------------------------------------------------------- #
# Evidence passthrough
# --------------------------------------------------------------------------- #

def test_evidence_is_copied_unchanged_from_the_extraction():
    payload = extraction(
        industry_fit={"level": "high", "evidence": "full-service marketing agency"},
        urgency={"level": "high", "evidence": "ready to pilot in the next 2 weeks"},
    )
    result = score_lead(payload, clean_lead())
    assert result["breakdown"]["industry_fit"]["evidence"] == "full-service marketing agency"
    assert result["breakdown"]["urgency"]["evidence"] == "ready to pilot in the next 2 weeks"


def test_every_llm_factor_preserves_its_level_and_evidence_verbatim():
    payload = extraction()
    result = score_lead(payload, clean_lead())
    for factor in ("industry_fit", "seniority", "urgency", "pain_severity",
                   "purchasing_readiness", "timeline", "buying_stage"):
        assert result["breakdown"][factor]["level"] == payload[factor]["level"]
        assert result["breakdown"][factor]["evidence"] == payload[factor]["evidence"]


def test_measure_evidence_passes_through_when_the_notes_win():
    result = score_lead(
        extraction(employee_count_in_notes={
            "stated": True, "low": 23, "high": 23, "kind": "exact", "evidence": "23 people",
        }),
        clean_lead(),
    )
    assert result["breakdown"]["company_size_signal"]["evidence"] == "23 people"


def test_scoring_does_not_mutate_its_inputs():
    payload, lead = extraction(), clean_lead()
    import copy

    before_payload, before_lead = copy.deepcopy(payload), copy.deepcopy(lead)
    score_lead(payload, lead)
    assert payload == before_payload
    assert lead == before_lead


def test_breakdown_contains_every_factor_in_canonical_order():
    """Ten scored factors in order, then the location modifier.

    Location is deliberately outside FACTOR_ORDER: it adjusts the fit subtotal
    rather than contributing its own points, because a sixth scored factor
    would push company fit past 50 and move the routing thresholds.
    """
    result = score_lead(extraction(), clean_lead())
    breakdown = result["breakdown"]
    assert list(breakdown)[:10] == list(scoring.FACTOR_ORDER)
    assert len(breakdown) == 11
    assert "location" in breakdown
    assert "location" not in scoring.FACTOR_ORDER
    assert "points" not in breakdown["location"]      # a modifier, not a factor


def test_unrecognised_level_scores_the_minimum_and_stays_visible():
    """A contract drift must not crash a run, nor silently look like a real level."""
    result = score_lead(extraction(urgency={"level": "extreme", "evidence": "x"}), clean_lead())
    entry = result["breakdown"]["urgency"]
    assert entry["points"] == 2
    assert entry["level"] == "extreme"
    assert entry["unrecognised_level"] is True


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

def test_score_lead_is_deterministic():
    payload, lead = extraction(), clean_lead()
    assert score_lead(payload, lead) == score_lead(payload, lead)


def test_generate_explanation_is_deterministic():
    """The same breakdown must always produce byte-identical text."""
    breakdown = score_lead(extraction(), clean_lead())["breakdown"]
    first = generate_explanation(breakdown, "NURTURE")
    second = generate_explanation(breakdown, "NURTURE")
    assert first == second


def test_generate_explanation_is_deterministic_across_key_order():
    """Dictionary iteration order must not leak into the output."""
    breakdown = score_lead(extraction(), clean_lead())["breakdown"]
    shuffled = {k: breakdown[k] for k in reversed(list(breakdown))}
    assert generate_explanation(breakdown, "NURTURE") == generate_explanation(shuffled, "NURTURE")


def test_explanation_leads_with_the_total_and_recommendation():
    result = score_lead(
        extraction(
            industry_fit={"level": "high", "evidence": "full-service marketing agency"},
            urgency={"level": "high", "evidence": "ready to pilot in 2 weeks"},
            purchasing_readiness={"level": "approved", "evidence": "Budget approved"},
        ),
        clean_lead(),
    )
    text = generate_explanation(result["breakdown"], "CONTACT_NOW")
    assert text.startswith(f"Total {result['total_score']}/100 → CONTACT_NOW.")


def test_explanation_quotes_evidence_for_its_strengths():
    result = score_lead(
        extraction(
            industry_fit={"level": "high", "evidence": "full-service marketing agency"},
            urgency={"level": "high", "evidence": "ready to pilot in 2 weeks"},
        ),
        clean_lead(),
    )
    text = generate_explanation(result["breakdown"], "CONTACT_NOW")
    assert "full-service marketing agency" in text
    assert "ready to pilot in 2 weeks" in text


def test_explanation_uses_the_recommendation_it_is_given():
    """Stage 5 never derives a recommendation; it renders the one passed in."""
    breakdown = score_lead(extraction(), clean_lead())["breakdown"]
    for recommendation in ("CONTACT_NOW", "NURTURE", "DISQUALIFY", "REVIEW"):
        assert recommendation in generate_explanation(breakdown, recommendation)


def test_explanation_reports_a_notes_vs_crm_conflict():
    result = score_lead(
        extraction(employee_count_in_notes={
            "stated": True, "low": 23, "high": 23, "kind": "exact", "evidence": "23 people",
        }),
        clean_lead(employees_low=9, employees_high=9,
                   employees_status="exact", employees_display="9"),
    )
    text = generate_explanation(result["breakdown"], "CONTACT_NOW")
    assert "notes" in text.lower() and "23" in text and "9" in text


def test_explanation_mentions_weaknesses_for_a_poor_lead():
    result = score_lead(
        extraction(
            industry_fit={"level": "low", "evidence": "car dealership"},
            urgency={"level": "low", "evidence": "not specified"},
            pain_severity={"level": "low", "evidence": "not specified"},
            purchasing_readiness={"level": "unknown", "evidence": "not specified"},
            timeline={"level": "none", "evidence": "not specified"},
            buying_stage={"level": "exploring", "evidence": "just looking"},
        ),
        clean_lead(source_clean="cold_reply"),
    )
    text = generate_explanation(result["breakdown"], "DISQUALIFY")
    assert "Held back by" in text


def _runtime_string_constants(module) -> list[str]:
    """Every string literal the module actually evaluates, docstrings excluded.

    Scanning raw source would match the module docstring, which legitimately
    explains that Stage 5 does *not* own the recommendation labels.
    """
    import ast

    tree = ast.parse(open(module.__file__, encoding="utf-8").read())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    docstrings.add(id(body[0].value))
    return [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_explanation_never_invents_a_recommendation_of_its_own():
    """No threshold logic and no recommendation labels may leak into Stage 5."""
    literals = " ".join(_runtime_string_constants(scoring))
    for banned in ("CONTACT_NOW", "NURTURE", "DISQUALIFY"):
        assert banned not in literals, f"{banned} is a runtime literal in scoring.py"


def test_scoring_module_imports_nothing_that_could_call_a_model():
    import ast

    tree = ast.parse(open(scoring.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for banned in ("llm_extraction", "providers", "anthropic", "google", "genai",
                   "requests", "httpx", "urllib"):
        assert banned not in imported, f"scoring.py must not import {banned}"
