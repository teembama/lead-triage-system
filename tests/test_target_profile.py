"""Target Profile: validation, scoring effect, and the ranking it must be able
to change.

The feature is only real if the same CSV produces different assessments under
different profiles, so the centrepiece here is a two-profile run over one
dataset. Everything else guards the rules that keep it from becoming a filter:
missing information is unknown rather than a mismatch, location is inert when
unset, and a perfect profile match still cannot manufacture buying intent.

No API key and no network - the provider is a fake that answers according to the
profile it is given, the way a model would.
"""

import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline  # noqa: E402
import prompts  # noqa: E402
import schema  # noqa: E402
import scoring  # noqa: E402
import ui  # noqa: E402
from data_cleaning import CANONICAL_COLUMNS  # noqa: E402
from llm_extraction import ExtractionCache  # noqa: E402
from providers import ProviderResponse  # noqa: E402
from scoring import score_lead  # noqa: E402
from target_profile import ANY_INDUSTRY, TargetProfile, TargetProfileError  # noqa: E402

SAAS_NG = TargetProfile.create("SaaS", 10_000, 50_000, "Nigeria")
HEALTH_UK = TargetProfile.create("Healthcare", 50_000, 100_000, "United Kingdom")
NO_PLACE = TargetProfile.create("SaaS", 10_000, 50_000)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def measure(stated, low=None, high=None, kind="missing", ev="not specified"):
    return {"stated": stated, "low": low, "high": high, "kind": kind, "evidence": ev}


def place(stated, value=None, ev="not specified"):
    return {"stated": stated, "value": value, "evidence": ev}


def extraction(industry="high", budget=None, location=None, **overrides):
    payload = {
        "is_potential_buyer": "yes",
        "buyer_type": "business_prospect",
        "non_buyer_signal": "none",
        "disqualification_evidence": None,
        "ambiguity_reason": None,
        "industry_fit": {"level": industry, "evidence": "what they do"},
        "employee_count_in_notes": measure(False),
        "budget_in_notes": budget or measure(False),
        "location_in_notes": location or place(False),
        "seniority": {"level": "decision_maker", "evidence": "founder"},
        "urgency": {"level": "high", "evidence": "ASAP"},
        "pain_severity": {"level": "high", "evidence": "eating our week"},
        "purchasing_readiness": {"level": "approved", "evidence": "Budget approved"},
        "timeline": {"level": "near_term", "evidence": "2 weeks"},
        "buying_stage": {"level": "committed", "evidence": "end to end"},
        "price_sensitive_flag": False,
    }
    payload.update(overrides)
    return payload


def lead(**overrides):
    row = {
        "lead_id": "L-1", "source_clean": "event", "source_raw": "event",
        "title_clean": "Founder", "company_clean": "Acme",
        "employees_low": None, "employees_high": None,
        "employees_status": "missing", "employees_display": "unknown",
        "budget_low": None, "budget_high": None,
        "budget_status": "missing", "budget_display": "unknown",
        "notes_clean": "notes",
    }
    row.update(overrides)
    return row


def budget_points(ext, profile, lead_row=None):
    result = score_lead(ext, lead_row or lead(), profile)
    return result["breakdown"]["budget_capacity"]


def csv_row(lead_id, notes, **extra):
    row = {c: "" for c in CANONICAL_COLUMNS}
    row.update(lead_id=lead_id, created="2024-06-07", name="X", email=f"{lead_id}@x.co",
               company="Co", title="Founder", source="event", notes=notes)
    row.update(extra)
    return row


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_industry_and_budget_are_required():
    with pytest.raises(TargetProfileError):
        TargetProfile.create("", 1_000, 5_000)
    with pytest.raises(TargetProfileError):
        TargetProfile.create("SaaS", "not a number", 5_000)


def test_budget_bounds_must_be_ordered():
    with pytest.raises(TargetProfileError, match="must not be larger"):
        TargetProfile.create("SaaS", 50_000, 10_000)


def test_budget_accepts_operator_shorthand():
    assert TargetProfile.create("SaaS", "$10,000", "50k").budget_min == 10_000
    assert TargetProfile.create("SaaS", "10k", "1.5m").budget_max == 1_500_000


def test_location_is_optional():
    assert NO_PLACE.has_location is False
    assert NO_PLACE.location is None
    assert SAAS_NG.has_location is True


def test_validation_messages_are_safe_to_show():
    """Operator-facing copy: no field names, no exception types."""
    for bad in [("", 1, 2), ("SaaS", "x", 2), ("SaaS", 9, 8)]:
        try:
            TargetProfile.create(*bad)
        except TargetProfileError as exc:
            text = str(exc)
            for leak in ("Traceback", "ValueError", "budget_min", "self.", "None"):
                assert leak not in text


def test_company_size_is_not_a_profile_criterion():
    fields = set(TargetProfile.__dataclass_fields__)
    assert fields == {"industries", "budget_min", "budget_max", "location"}
    assert not any("size" in f or "employee" in f or "headcount" in f for f in fields)


# --------------------------------------------------------------------------- #
# The profile actually reaches the model  (requirement 12)
# --------------------------------------------------------------------------- #

def test_profile_is_in_the_prompt_not_just_the_ui():
    message = prompts.build_user_message(lead(), SAAS_NG)
    assert "<target_profile>" in message
    assert "SaaS" in message
    assert "$10,000 - $50,000" in message
    assert "Nigeria" in message


def test_prompt_omits_location_entirely_when_unset():
    message = prompts.build_user_message(lead(), NO_PLACE)
    assert "Target location" not in message


def test_no_profile_leaves_the_prompt_untouched():
    assert "<target_profile>" not in prompts.build_user_message(lead())


def test_schema_describes_industry_against_the_target():
    described = schema.build_json_schema(SAAS_NG)["properties"]["industry_fit"]["description"]
    assert "SaaS" in described
    # The hardcoded demo ICP is gone - that was the reusability blocker.
    assert "marketing / growth / ops / lead-gen agency" not in described


def test_changing_the_profile_changes_the_cache_key():
    """Two profiles are two different questions about the same notes."""
    a = ExtractionCache.key(lead(), "gemini", "m", SAAS_NG)
    b = ExtractionCache.key(lead(), "gemini", "m", HEALTH_UK)
    none = ExtractionCache.key(lead(), "gemini", "m", None)
    assert len({a, b, none}) == 3


def test_pipeline_forwards_the_profile_to_extraction():
    seen = {}

    class Spy:
        name, model = "t", "t"

        def extract(self, _system, user_message, json_schema):
            seen["message"] = user_message
            seen["schema"] = json_schema
            return ProviderResponse(text=json.dumps(extraction()))

    frame = pd.DataFrame([csv_row("L-1", "A SaaS company in Lagos with a $25k budget.")]).astype(str)
    pipeline.run_pipeline(frame, provider=Spy(), cache=ExtractionCache(path=None),
                          rpm=0, profile=SAAS_NG)
    assert "SaaS" in seen["message"]
    assert "Nigeria" in seen["message"]
    assert "SaaS" in seen["schema"]["properties"]["industry_fit"]["description"]


# --------------------------------------------------------------------------- #
# Industry  (requirements 6, 7)
# --------------------------------------------------------------------------- #

def test_matching_industry_improves_fit():
    high = score_lead(extraction(industry="high"), lead(), SAAS_NG)
    low = score_lead(extraction(industry="low"), lead(), SAAS_NG)
    assert high["breakdown"]["industry_fit"]["points"] == 20
    assert low["breakdown"]["industry_fit"]["points"] == 3
    assert high["company_fit_score"] > low["company_fit_score"]


def test_mismatching_industry_reduces_fit():
    assert (score_lead(extraction(industry="low"), lead(), SAAS_NG)["company_fit_score"]
            < score_lead(extraction(industry="medium"), lead(), SAAS_NG)["company_fit_score"])


def test_weak_industry_explanation_names_the_target():
    """"Limited industry fit (restaurant; target profile is SaaS)."""
    result = score_lead(
        extraction(industry="low", **{"industry_fit": {"level": "low", "evidence": "restaurant"}}),
        lead(), SAAS_NG,
    )
    text = scoring.generate_explanation(result["breakdown"], "DISQUALIFY")
    assert "target profile is SaaS" in text


# --------------------------------------------------------------------------- #
# Budget  (requirements 3, 4, 5)
# --------------------------------------------------------------------------- #

def test_budget_inside_the_target_range_scores_highest():
    entry = budget_points(extraction(budget=measure(True, 25_000, 25_000, "exact", "$25k")), SAAS_NG)
    assert entry["level"] == "on_target"
    assert entry["points"] == 10


def test_budget_far_below_the_target_scores_lowest():
    entry = budget_points(extraction(budget=measure(True, 1_500, 1_500, "exact", "$1,500")), SAAS_NG)
    assert entry["level"] == "off_target"
    assert entry["points"] == 2


def test_budget_just_below_the_target_is_near_not_off():
    entry = budget_points(extraction(budget=measure(True, 7_000, 7_000, "exact", "$7k")), SAAS_NG)
    assert entry["level"] == "near_target"
    assert entry["points"] == 7


def test_budget_far_above_the_target_is_also_weaker():
    entry = budget_points(extraction(budget=measure(True, 500_000, 500_000, "exact", "$500k")), SAAS_NG)
    assert entry["level"] == "off_target"


def test_missing_budget_is_unknown_not_a_mismatch():
    """Requirement 3: silence must not be scored as if it were a bad answer."""
    unknown = budget_points(extraction(budget=measure(False)), SAAS_NG)
    mismatch = budget_points(extraction(budget=measure(True, 100, 100, "exact", "$100")), SAAS_NG)
    assert unknown["level"] == "unknown"
    assert unknown["points"] > mismatch["points"], (
        "a lead that never mentioned money must not score like one that cannot afford you"
    )


def test_a_budget_range_overlapping_the_target_is_on_target():
    entry = budget_points(
        extraction(budget=measure(True, 8_000, 12_000, "range", "$8-12k")), SAAS_NG)
    assert entry["level"] == "on_target"


def test_the_same_budget_scores_differently_under_different_profiles():
    """$25k is on target for 10-50k and below target for 50-100k."""
    ext = extraction(budget=measure(True, 25_000, 25_000, "exact", "$25k"))
    assert budget_points(ext, SAAS_NG)["level"] == "on_target"
    assert budget_points(ext, HEALTH_UK)["level"] == "near_target"


def test_budget_scoring_is_unchanged_without_a_profile():
    """The absolute §12 bands still apply when no profile is configured."""
    row = lead(budget_low=9_000, budget_high=9_000, budget_status="exact", budget_display="9000")
    assert score_lead(extraction(), row)["breakdown"]["budget_capacity"]["points"] == 10


# --------------------------------------------------------------------------- #
# Location  (requirements 2, 8, 9)
# --------------------------------------------------------------------------- #

def test_matching_location_improves_fit():
    match = score_lead(extraction(location=place(True, "Nigeria", "Lagos-based")), lead(), SAAS_NG)
    unknown = score_lead(extraction(location=place(False)), lead(), SAAS_NG)
    assert match["breakdown"]["location"]["level"] == "match"
    assert match["company_fit_score"] > unknown["company_fit_score"]


def test_mismatching_location_reduces_fit():
    miss = score_lead(extraction(location=place(True, "Germany", "Berlin office")), lead(), SAAS_NG)
    unknown = score_lead(extraction(location=place(False)), lead(), SAAS_NG)
    assert miss["breakdown"]["location"]["level"] == "mismatch"
    assert miss["company_fit_score"] < unknown["company_fit_score"]


def test_missing_lead_location_does_not_fail_the_criterion():
    """Requirement 9: never an accidental disqualification."""
    entry = score_lead(extraction(location=place(False)), lead(), SAAS_NG)["breakdown"]["location"]
    assert entry["level"] == "unknown"
    assert entry["adjustment"] == 0


def test_location_has_no_effect_when_not_configured():
    """Requirement 2: a blank location must change nothing at all."""
    stated = extraction(location=place(True, "Germany", "Berlin"))
    silent = extraction(location=place(False))
    assert (score_lead(stated, lead(), NO_PLACE)["company_fit_score"]
            == score_lead(silent, lead(), NO_PLACE)["company_fit_score"])
    assert score_lead(stated, lead(), NO_PLACE)["breakdown"]["location"]["level"] == "not_configured"


def test_location_recognises_demonyms_and_cities():
    for stated in ("Nigerian agency", "based in Lagos", "Nigeria"):
        entry = score_lead(extraction(location=place(True, stated, stated)),
                           lead(), SAAS_NG)["breakdown"]["location"]
        assert entry["level"] == "match", stated


def test_location_cannot_disqualify_on_its_own():
    """A mismatch is a small adjustment, never a gate."""
    miss = score_lead(extraction(location=place(True, "Germany", "Berlin")), lead(), SAAS_NG)
    assert miss["company_fit_score"] > 0
    assert miss["breakdown"]["location"]["adjustment"] >= scoring.LOCATION_MISMATCH_PENALTY


def test_fit_stays_within_its_ceiling():
    """Location is a modifier, so it must not push company fit past 50."""
    best = score_lead(
        extraction(industry="high",
                   budget=measure(True, 25_000, 25_000, "exact", "$25k"),
                   location=place(True, "Nigeria", "Lagos"),
                   employee_count_in_notes=measure(True, 30, 30, "exact", "30 people")),
        lead(source_clean="event"), SAAS_NG,
    )
    assert best["company_fit_score"] <= scoring.MAX_COMPANY_FIT
    assert best["total_score"] <= scoring.MAX_TOTAL


# --------------------------------------------------------------------------- #
# Fit and intent stay independent  (requirements 10, 11)
# --------------------------------------------------------------------------- #

def test_perfect_fit_with_weak_intent_is_not_contact_now():
    """Requirement 10 - a strong ICP match must not manufacture urgency.

    The intent floor used to enforce this by disqualifying the lead. It is now
    enforced by the 75 line alone: fit is clamped at 50, so weak intent cannot
    reach CONTACT_NOW however perfect the fit. The lead is nurtured, not
    deleted - it is a genuine buyer who is simply not ready.
    """
    from recommendation import recommend

    result = score_lead(
        extraction(industry="high",
                   budget=measure(True, 25_000, 25_000, "exact", "$25k"),
                   location=place(True, "Nigeria", "Lagos"),
                   urgency={"level": "low", "evidence": "not specified"},
                   pain_severity={"level": "low", "evidence": "not specified"},
                   purchasing_readiness={"level": "unknown", "evidence": "not specified"},
                   timeline={"level": "none", "evidence": "not specified"},
                   buying_stage={"level": "exploring", "evidence": "just looking"}),
        lead(), SAAS_NG,
    )
    assert result["buying_intent_score"] < 15
    route = recommend({"is_potential_buyer": "yes", **result})
    assert route != "CONTACT_NOW", "a strong ICP match must not manufacture intent"
    assert route == "NURTURE", "a genuine buyer who is early is nurtured, not excluded"


def test_strong_intent_with_weak_fit_follows_the_existing_rules():
    """Requirement 11 - no special-casing, just the existing bands."""
    from recommendation import recommend

    result = score_lead(
        extraction(industry="low",
                   budget=measure(True, 200, 200, "exact", "$200"),
                   location=place(True, "Germany", "Berlin")),
        lead(source_clean="cold_reply"), SAAS_NG,
    )
    assert result["buying_intent_score"] >= 15
    expected = ("CONTACT_NOW" if result["total_score"] >= 75
                else "NURTURE" if result["total_score"] >= 45 else "DISQUALIFY")
    assert recommend({"is_potential_buyer": "yes", **result}) == expected


def test_the_profile_never_touches_intent():
    """Identical notes, opposite profiles - intent must be identical."""
    ext = extraction(industry="high", budget=measure(True, 25_000, 25_000, "exact", "$25k"))
    a = score_lead(ext, lead(), SAAS_NG)
    b = score_lead(ext, lead(), HEALTH_UK)
    assert a["buying_intent_score"] == b["buying_intent_score"]


def test_the_profile_does_not_change_the_buyer_gate():
    from recommendation import recommend

    for profile in (SAAS_NG, HEALTH_UK, NO_PLACE, None):
        result = score_lead(extraction(), lead(), profile)
        assert recommend({"is_potential_buyer": "no", **result}) == "DISQUALIFY"
        assert recommend({"is_potential_buyer": "ambiguous", **result}) == "REVIEW"


# --------------------------------------------------------------------------- #
# Several target industries at once
# --------------------------------------------------------------------------- #

BOTH = TargetProfile.create(["SaaS", "Healthcare"], 10_000, 50_000, "Nigeria")
ANY_PROFILE = TargetProfile.create(ANY_INDUSTRY, 10_000, 50_000, "Nigeria")

MATCH_POINTS = scoring.INDUSTRY_FIT_POINTS["high"]
MISS_POINTS = scoring.INDUSTRY_FIT_POINTS["low"]


def industry_points(profile, level="high"):
    return score_lead(extraction(industry=level), lead(), profile)["breakdown"]["industry_fit"]


# ---- A. one industry still behaves exactly as it did ---------------------- #

def test_one_industry_as_a_string_and_as_a_list_are_the_same_profile():
    assert TargetProfile.create("SaaS", 10_000, 50_000, "Nigeria") == SAAS_NG
    assert TargetProfile.create(["SaaS"], 10_000, 50_000, "Nigeria") == SAAS_NG
    assert TargetProfile.create(["SaaS"], 10_000, 50_000, "Nigeria").signature() \
        == SAAS_NG.signature()


def test_one_industry_asks_the_model_the_question_it_always_asked():
    """Byte-for-byte, so a single selection cannot drift on the model's side."""
    assert SAAS_NG.describe_for_model().splitlines()[0] == "Target industry: SaaS"
    described = schema.build_json_schema(SAAS_NG)["properties"]["industry_fit"]["description"]
    assert described.startswith(
        "How closely the lead's own business matches the target industry: SaaS."
    )
    assert "ANY ONE" not in described


@pytest.mark.parametrize("level,expected", [("high", 20), ("medium", 11), ("low", 3),
                                            ("unknown", 3)])
def test_one_industry_scores_exactly_as_before(level, expected):
    assert industry_points(SAAS_NG, level)["points"] == expected


# ---- B. several industries ------------------------------------------------ #

def test_the_model_is_told_a_lead_need_match_only_one():
    described = schema.build_json_schema(BOTH)["properties"]["industry_fit"]["description"]
    assert "ANY ONE" in described
    assert "SaaS, Healthcare" in described
    assert "matching more than one is not better than matching one" in described
    assert "Target industries" in BOTH.describe_for_model()


@pytest.mark.parametrize("level,expected", [("high", 20), ("medium", 11), ("low", 3)])
def test_several_industries_use_the_same_scoring_treatment(level, expected):
    """Matching one of two selected industries is worth exactly what matching
    the single selected industry is worth today."""
    assert industry_points(BOTH, level)["points"] == expected
    assert industry_points(BOTH, level)["points"] == industry_points(SAAS_NG, level)["points"]


def test_a_lead_matching_neither_gets_the_existing_non_match_treatment():
    assert industry_points(BOTH, "low")["points"] == MISS_POINTS


# ---- C. no double counting ------------------------------------------------ #

@pytest.mark.parametrize("count", [1, 2, 3, 6])
def test_more_selected_industries_never_add_points(count):
    names = ["SaaS", "Healthcare", "Logistics", "Retail", "Education", "Legal"][:count]
    profile = TargetProfile.create(names, 10_000, 50_000)
    entry = industry_points(profile, "high")
    assert entry["points"] == MATCH_POINTS
    assert entry["points"] <= scoring.INDUSTRY_FIT_POINTS["high"]


def test_the_fit_subtotal_still_caps_at_fifty_with_many_industries():
    profile = TargetProfile.create(
        ["SaaS", "Healthcare", "Logistics", "Retail"], 10_000, 50_000, "Nigeria")
    result = score_lead(
        extraction(industry="high",
                   budget=measure(True, 25_000, 25_000, "exact", "$25k"),
                   location=place(True, "Nigeria", "Lagos")),
        lead(), profile,
    )
    assert result["company_fit_score"] <= scoring.MAX_COMPANY_FIT
    assert result["breakdown"]["industry_fit"]["points"] == MATCH_POINTS


# ---- D. Any --------------------------------------------------------------- #

def test_any_is_stored_as_no_industries_at_all():
    assert ANY_PROFILE.industries == ()
    assert ANY_PROFILE.accepts_any_industry is True
    assert ANY_PROFILE.industry_label() == "Any"
    assert SAAS_NG.accepts_any_industry is False


@pytest.mark.parametrize("mixed", [
    ["Any", "SaaS"], ["SaaS", "Any"], ["any", "Healthcare"], ["SaaS", "ANY", "Healthcare"],
])
def test_any_cannot_coexist_with_a_named_industry(mixed):
    with pytest.raises(TargetProfileError):
        TargetProfile.create(mixed, 10_000, 50_000)


@pytest.mark.parametrize("empty", [[], "", None, ["", "  "]])
def test_an_empty_selection_is_rejected_rather_than_read_as_any(empty):
    """Choosing nothing is an unfinished form; choosing Any is a decision."""
    with pytest.raises(TargetProfileError):
        TargetProfile.create(empty, 10_000, 50_000)


@pytest.mark.parametrize("level", ["high", "medium", "low", "unknown"])
def test_any_never_penalises_a_lead_for_its_industry(level):
    entry = industry_points(ANY_PROFILE, level)
    assert entry["points"] == MATCH_POINTS
    assert entry["level"] == "any"


def test_any_cannot_separate_two_leads():
    """The point of 'not a constraint': industry stops discriminating."""
    scores = {
        level: score_lead(extraction(industry=level), lead(), ANY_PROFILE)["total_score"]
        for level in ("high", "medium", "low", "unknown")
    }
    assert len(set(scores.values())) == 1


def test_any_still_respects_the_fit_ceiling():
    result = score_lead(
        extraction(industry="low",
                   budget=measure(True, 25_000, 25_000, "exact", "$25k"),
                   location=place(True, "Nigeria", "Lagos")),
        lead(), ANY_PROFILE,
    )
    assert result["company_fit_score"] <= scoring.MAX_COMPANY_FIT


def test_any_tells_the_model_not_to_rank_industries():
    described = schema.build_json_schema(ANY_PROFILE)["properties"]["industry_fit"]["description"]
    assert "not a criterion" in described
    assert "Do not rank one industry above another." in described


# ---- explanations (requirement 7) ----------------------------------------- #

def test_a_match_names_the_industries_it_matched():
    result = score_lead(
        extraction(industry="high",
                   **{"industry_fit": {"level": "high", "evidence": "SaaS company"}}),
        lead(), BOTH,
    )
    text = scoring.generate_explanation(result["breakdown"], "CONTACT_NOW")
    assert "matches SaaS or Healthcare" in text


def test_a_miss_names_every_industry_it_missed():
    result = score_lead(
        extraction(industry="low",
                   **{"industry_fit": {"level": "low", "evidence": "restaurant"}}),
        lead(), BOTH,
    )
    text = scoring.generate_explanation(result["breakdown"], "DISQUALIFY")
    assert "target profile is SaaS or Healthcare" in text


def test_a_single_industry_explanation_is_worded_exactly_as_before():
    for level, expected in (("low", True), ("high", False)):
        result = score_lead(
            extraction(industry=level,
                       **{"industry_fit": {"level": level, "evidence": "restaurant"}}),
            lead(), SAAS_NG,
        )
        text = scoring.generate_explanation(result["breakdown"], "NURTURE")
        assert ("target profile is SaaS" in text) is expected
        assert "matches SaaS" not in text          # no annotation on a lone target


@pytest.mark.parametrize("level", ["high", "medium", "low", "unknown"])
def test_under_any_industry_is_neither_a_strength_nor_a_drag(level):
    """Requirement 7: the assessment must not claim industry told us anything.
    A factor that scores the same for everyone has no business appearing as a
    reason the lead was ranked where it was."""
    result = score_lead(extraction(industry=level), lead(), ANY_PROFILE)
    text = scoring.generate_explanation(result["breakdown"], "NURTURE")
    for phrase in ("strong industry fit", "adjacent industry fit",
                   "weak industry fit", "industry fit unclear", "target profile is"):
        assert phrase not in text, f"{level}: explanation still argues from industry"


def test_under_any_the_operator_sees_that_industry_was_unconstrained():
    result = score_lead(extraction(industry="low"), lead(), ANY_PROFILE)
    panel = ui.lead_detail({**result, "lead_id": "L-1", "recommendation": "NURTURE",
                            "scores_suppressed": False})
    assert ">any<" in panel                          # the factor table's level
    assert "Any" in ui.profile_summary(ANY_PROFILE)  # the yardstick shown above it


def test_no_internal_names_reach_the_operator():
    for profile in (BOTH, ANY_PROFILE):
        result = score_lead(extraction(industry="high"), lead(), profile)
        text = scoring.generate_explanation(result["breakdown"], "NURTURE")
        for banned in ("industries", "target_industries", "accepts_any_industry",
                       "industry_unconstrained", "industry_fit"):
            assert banned not in text
        assert "_" not in text


# ---- F. propagation into the pipeline ------------------------------------- #

def test_selected_industries_reach_both_the_model_and_the_scorer():
    seen = {}

    class Spy:
        name, model = "t", "t"

        def extract(self, _system, user_message, json_schema):
            seen["message"] = user_message
            seen["schema"] = json_schema
            return ProviderResponse(text=json.dumps(extraction(industry="high")))

    frame = pd.DataFrame([csv_row("L-1", "A SaaS company in Lagos with a $25k budget.")]).astype(str)
    out = pipeline.run_pipeline(frame, provider=Spy(), cache=ExtractionCache(path=None),
                                rpm=0, profile=BOTH)

    # ...into the prompt,
    assert "SaaS" in seen["message"] and "Healthcare" in seen["message"]
    # ...into the schema the model answers against,
    assert "Healthcare" in seen["schema"]["properties"]["industry_fit"]["description"]
    # ...and onto the scored breakdown the interface reads.
    entry = out["rows"][0]["breakdown"]["industry_fit"]
    assert entry["target_industries"] == ["SaaS", "Healthcare"]
    assert entry["points"] == MATCH_POINTS


def test_any_reaches_the_scorer_as_an_unconstrained_factor():
    class Always:
        name, model = "t", "t"

        def extract(self, *_a):
            return ProviderResponse(text=json.dumps(extraction(industry="low")))

    frame = pd.DataFrame([csv_row("L-1", "A restaurant in Lagos.")]).astype(str)
    out = pipeline.run_pipeline(frame, provider=Always(), cache=ExtractionCache(path=None),
                                rpm=0, profile=ANY_PROFILE)
    entry = out["rows"][0]["breakdown"]["industry_fit"]
    assert entry["level"] == "any"
    assert entry["points"] == MATCH_POINTS
    assert "target_industries" not in entry


def test_two_industry_sets_are_two_different_cache_keys():
    keys = {
        ExtractionCache.key(lead(), "gemini", "m", SAAS_NG),
        ExtractionCache.key(lead(), "gemini", "m", BOTH),
        ExtractionCache.key(lead(), "gemini", "m", ANY_PROFILE),
    }
    assert len(keys) == 3


def test_the_order_industries_were_picked_in_is_not_a_different_question():
    a = TargetProfile.create(["SaaS", "Healthcare"], 10_000, 50_000)
    b = TargetProfile.create(["Healthcare", "SaaS"], 10_000, 50_000)
    assert a.signature() == b.signature()


def test_the_same_industry_picked_twice_is_one_criterion():
    assert TargetProfile.create(["SaaS", "saas ", "SaaS"], 10_000, 50_000).industries \
        == ("SaaS",)


# --------------------------------------------------------------------------- #
# Ranking: the same dataset under two profiles  (the critical test)
# --------------------------------------------------------------------------- #

LEADS = [
    csv_row("SAAS-NG", "We're a SaaS company based in Lagos, Nigeria. Budget approved at "
                       "$25k/mo. Manual onboarding is eating our week. Want it automated "
                       "end to end, starting in 2 weeks. I make the call."),
    csv_row("HEALTH-UK", "We're a healthcare provider in London, UK. Budget signed off at "
                         "$75k/mo. Patient intake paperwork is eating our week. Want it "
                         "automated end to end within 2 weeks. I decide here."),
    csv_row("RESTAURANT", "UK restaurant, budget about $1,500. Looking at options."),
    # Sits between the two matched leads on every profile below, and matches
    # neither industry - so it is the hurdle a lead has to clear when its own
    # industry is added to the target. Without something in between, a fit
    # change can raise a score without ever being able to move a rank.
    csv_row("MIDCO", "We're a logistics operator in London, UK. Budget signed off at "
                     "$60k/mo. Manual dispatch paperwork is eating our week. Want it "
                     "automated end to end within 2 weeks. I decide here."),
]


class ProfileAwareProvider:
    """Answers the way a model would: industry fit and location read off the
    notes relative to whichever target profile appears in the prompt."""

    name, model = "profile-aware", "test"

    def extract(self, _system, user_message, _schema):
        notes = user_message.lower()
        # Read the target line the way a model would, so one industry or
        # several is answered from the same text the real prompt carries.
        target_line = next(
            (line.split(":", 1)[1] for line in notes.splitlines()
             if line.startswith("target industry:") or line.startswith("target industries")),
            "",
        )
        unconstrained = "no industry preference" in target_line
        target_saas = unconstrained or "saas" in target_line
        target_health = unconstrained or "healthcare" in target_line

        if "saas company" in notes:
            industry = "high" if target_saas else "low"
            loc, budget = ("Nigeria", 25_000)
        elif "healthcare provider" in notes:
            industry = "high" if target_health else "low"
            loc, budget = ("United Kingdom", 75_000)
        elif "logistics operator" in notes:
            # Adjacent to everything, in neither target list.
            industry, loc, budget = "medium", "United Kingdom", 60_000
        else:
            industry, loc, budget = "low", "United Kingdom", 1_500

        strong = budget > 5_000
        return ProviderResponse(text=json.dumps(extraction(
            industry=industry,
            budget=measure(True, budget, budget, "exact", f"${budget:,}"),
            location=place(True, loc, loc),
            urgency={"level": "high" if strong else "low", "evidence": "2 weeks"},
            pain_severity={"level": "high" if strong else "low", "evidence": "eating our week"},
            purchasing_readiness={"level": "approved" if strong else "unknown",
                                  "evidence": "Budget approved"},
            timeline={"level": "near_term" if strong else "none", "evidence": "2 weeks"},
            buying_stage={"level": "committed" if strong else "exploring",
                          "evidence": "end to end"},
        )))


def run_under(profile):
    frame = pd.DataFrame(LEADS).astype(str)
    return pipeline.run_pipeline(frame, provider=ProfileAwareProvider(),
                                 cache=ExtractionCache(path=None), rpm=0, profile=profile)


def test_the_same_dataset_ranks_differently_under_two_profiles():
    """The feature's whole point: change the profile, change the ranking."""
    saas = {r["lead_id"]: r for r in run_under(SAAS_NG)["rows"]}
    health = {r["lead_id"]: r for r in run_under(HEALTH_UK)["rows"]}

    # Each lead scores higher under the profile that matches it.
    assert saas["SAAS-NG"]["total_score"] > health["SAAS-NG"]["total_score"]
    assert health["HEALTH-UK"]["total_score"] > saas["HEALTH-UK"]["total_score"]

    # ...and the top of the queue swaps accordingly.
    top_saas = [r for r in run_under(SAAS_NG)["rows"] if r["rank"] == 1][0]
    top_health = [r for r in run_under(HEALTH_UK)["rows"] if r["rank"] == 1][0]
    assert top_saas["lead_id"] == "SAAS-NG"
    assert top_health["lead_id"] == "HEALTH-UK"


def test_the_poor_fit_lead_stays_poor_under_both_profiles():
    """Poor fit is a low score and a low rank, under either profile - not an
    exclusion. The restaurant is off-ICP and browsing, which makes it a weak
    prospect rather than a non-buyer."""
    for out in (run_under(SAAS_NG), run_under(HEALTH_UK)):
        rows = {r["lead_id"]: r for r in out["rows"]}
        row = rows["RESTAURANT"]
        assert row["recommendation"] == "NURTURE"
        assert row["total_score"] < 45
        assert row["rank"] == max(r["rank"] for r in out["rows"] if r["rank"])


def test_adding_an_industry_to_the_target_changes_fit_and_ranking():
    """E: the feature has to move the pipeline, not the UI state.

    One profile, one dataset, one edit - add SaaS alongside Healthcare - and the
    SaaS lead's industry fit, total and rank all move.
    """
    health_only = TargetProfile.create("Healthcare", 50_000, 100_000, "United Kingdom")
    plus_saas = TargetProfile.create(["Healthcare", "SaaS"], 50_000, 100_000,
                                     "United Kingdom")

    before = {r["lead_id"]: r for r in run_under(health_only)["rows"]}
    after = {r["lead_id"]: r for r in run_under(plus_saas)["rows"]}

    saas_before = before["SAAS-NG"]
    saas_after = after["SAAS-NG"]

    # Industry fit moves from the miss tier to the match tier...
    assert saas_before["breakdown"]["industry_fit"]["points"] == MISS_POINTS
    assert saas_after["breakdown"]["industry_fit"]["points"] == MATCH_POINTS
    # ...the total moves with it...
    assert saas_after["total_score"] > saas_before["total_score"]
    # ...and the lead climbs the queue.
    assert saas_after["rank"] < saas_before["rank"]

    # The lead that already matched is unaffected by the addition.
    assert after["HEALTH-UK"]["breakdown"]["industry_fit"]["points"] == MATCH_POINTS
    assert (after["HEALTH-UK"]["total_score"]
            == before["HEALTH-UK"]["total_score"])


def test_both_leads_match_when_both_industries_are_targeted():
    rows = {r["lead_id"]: r for r in run_under(BOTH)["rows"]}
    assert rows["SAAS-NG"]["breakdown"]["industry_fit"]["points"] == MATCH_POINTS
    assert rows["HEALTH-UK"]["breakdown"]["industry_fit"]["points"] == MATCH_POINTS
    assert rows["RESTAURANT"]["breakdown"]["industry_fit"]["points"] == MISS_POINTS


def test_under_any_industry_stops_separating_the_dataset():
    rows = {r["lead_id"]: r for r in run_under(ANY_PROFILE)["rows"]}
    points = {r["breakdown"]["industry_fit"]["points"] for r in rows.values()}
    assert points == {MATCH_POINTS}
    # The restaurant is still last - on the factors that are still criteria.
    assert rows["RESTAURANT"]["rank"] == max(r["rank"] for r in rows.values() if r["rank"])


def test_the_same_profile_is_deterministic():
    """Requirement 1: same CSV + same profile -> identical results."""
    first, second = run_under(SAAS_NG), run_under(SAAS_NG)
    assert [(r["lead_id"], r["rank"], r["total_score"]) for r in first["rows"]] == \
           [(r["lead_id"], r["rank"], r["total_score"]) for r in second["rows"]]
    assert first["summary"] == second["summary"]


def test_a_new_dataset_works_without_any_hardcoding():
    """A CSV unrelated to the demonstration leads, under an unrelated profile."""
    profile = TargetProfile.create("Logistics & Supply Chain", 2_000, 8_000, "Kenya")
    frame = pd.DataFrame([
        csv_row("FREIGHT", "Nairobi freight forwarder, 40 staff. Manual customs paperwork "
                           "is eating our week. Budget approved at $4k/mo, moving in 2 weeks."),
    ]).astype(str)

    class Generic:
        name, model = "t", "t"

        def extract(self, _s, user_message, _j):
            assert "Logistics & Supply Chain" in user_message
            assert "Kenya" in user_message
            return ProviderResponse(text=json.dumps(extraction(
                industry="high",
                budget=measure(True, 4_000, 4_000, "exact", "$4k/mo"),
                location=place(True, "Kenya", "Nairobi"))))

    out = pipeline.run_pipeline(frame, provider=Generic(), cache=ExtractionCache(path=None),
                                rpm=0, profile=profile)
    row = out["rows"][0]
    assert row["breakdown"]["budget_capacity"]["level"] == "on_target"
    assert row["breakdown"]["location"]["level"] == "match"
    assert row["recommendation"] in ("CONTACT_NOW", "NURTURE")


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #

def test_the_queue_shows_what_the_run_was_measured_against():
    html = ui.profile_summary(SAAS_NG)
    assert "SaaS" in html and "$10,000 - $50,000" in html and "Nigeria" in html


def test_an_unset_location_reads_as_any():
    assert "Any" in ui.profile_summary(NO_PLACE)


def test_the_detail_view_names_the_target_range():
    row = {r["lead_id"]: r for r in run_under(SAAS_NG)["rows"]}["SAAS-NG"]
    html = ui.lead_detail(row)
    assert "target $10,000 - $50,000" in html
    assert "In target location" in html


def test_no_implementation_detail_reaches_the_profile_surfaces():
    row = {r["lead_id"]: r for r in run_under(SAAS_NG)["rows"]}["SAAS-NG"]
    surfaces = ui.profile_summary(SAAS_NG) + ui.lead_detail(row) + " ".join(
        row.get("explanation_parts") or [])
    for leak in ("gemini", "claude", "prompt", "token", "rate limit", "CRM",
                 "provider", "api", "schema"):
        assert leak.lower() not in surfaces.lower(), f"{leak!r} surfaced to the user"
