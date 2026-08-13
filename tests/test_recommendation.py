"""Stage 6 unit tests - routing precedence, thresholds, boundaries, ranking.

Pure functions: no API key, no network, no model. Every threshold in §9 is
asserted at its exact boundary, so an off-by-one in a comparison operator fails
a named test rather than silently re-routing part of the dataset.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import recommendation  # noqa: E402
import schema  # noqa: E402
import scoring  # noqa: E402
from recommendation import (  # noqa: E402
    CONTACT_NOW,
    DISQUALIFY,
    NURTURE,
    REVIEW,
    rank_leads,
    recommend,
    review_queue,
    route_lead,
    summarise_routes,
)


def lead(buyer="yes", fit=25, intent=25, total=None, **extra):
    record = {
        "is_potential_buyer": buyer,
        "company_fit_score": fit,
        "buying_intent_score": intent,
        "total_score": fit + intent if total is None else total,
    }
    record.update(extra)
    return record


# --------------------------------------------------------------------------- #
# The six cases from the routing analysis
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "buyer,fit,intent,total,expected,case",
    [
        ("yes", 45, 50, 95, CONTACT_NOW, "A: yes + high"),
        ("yes", 30, 30, 60, NURTURE, "A/B: yes + medium"),
        ("yes", 20, 20, 40, NURTURE, "B: yes + low - early, not excluded"),
        ("yes", 50, 14, 64, NURTURE, "B: yes + weak intent - early, not excluded"),
        ("no", 50, 50, 100, DISQUALIFY, "C: no + high"),
        ("no", 5, 5, 10, DISQUALIFY, "D: no + low"),
        ("ambiguous", 48, 50, 98, REVIEW, "E: ambiguous + high"),
        ("ambiguous", 5, 2, 7, REVIEW, "F: ambiguous + low"),
    ],
)
def test_routing_matrix(buyer, fit, intent, total, expected, case):
    assert recommend(lead(buyer, fit, intent, total)) == expected, case


# --------------------------------------------------------------------------- #
# The stated-purpose exclusion gate
# --------------------------------------------------------------------------- #

EXCLUDING_SIGNALS = [s for s in schema.NON_BUYER_SIGNALS if s != "none"]


@pytest.mark.parametrize("total", [0, 44, 45, 74, 75, 100])
@pytest.mark.parametrize("signal", EXCLUDING_SIGNALS)
def test_a_stated_non_buying_purpose_disqualifies_at_every_total(signal, total):
    """The gate is independent of the score, so it is asserted on both sides of
    both thresholds rather than at one convenient total."""
    assert recommend(
        lead("no", fit=total // 2, intent=total - total // 2, total=total,
             non_buyer_signal=signal)
    ) == DISQUALIFY


@pytest.mark.parametrize("signal", EXCLUDING_SIGNALS)
def test_a_maximum_intent_score_cannot_rescue_a_stated_non_buyer(signal):
    assert recommend(
        lead("no", fit=50, intent=50, total=100, non_buyer_signal=signal)
    ) == DISQUALIFY


def test_the_gate_is_read_before_the_intent_floor():
    """Both would disqualify; the point is that the gate decides first, so the
    reason a lead was excluded is the one the notes actually gave."""
    excluded = lead("ambiguous", fit=50, intent=0, total=50,
                    non_buyer_signal="competitive_research")
    assert recommend(excluded) == DISQUALIFY      # not REVIEW, which ambiguous alone gives


def test_an_ambiguous_lead_with_a_stated_purpose_is_excluded_not_reviewed():
    """An explicit statement of purpose settles what the buyer gate could not."""
    assert recommend(lead("ambiguous", total=90,
                          non_buyer_signal="media_enquiry")) == DISQUALIFY


@pytest.mark.parametrize("signal", EXCLUDING_SIGNALS)
def test_a_buyer_with_a_stated_non_buying_purpose_goes_to_a_human(signal):
    """The two fields contradict each other. Discarding the lead on one of them
    would be a guess; a person decides instead."""
    assert recommend(lead("yes", fit=50, intent=50, total=100,
                          non_buyer_signal=signal)) == REVIEW
    assert recommend(lead("yes", fit=1, intent=1, total=2,
                          non_buyer_signal=signal)) == REVIEW


def test_a_contradiction_suppresses_the_score_like_any_review():
    routed = route_lead(
        {"is_potential_buyer": "yes", "non_buyer_signal": "vendor_pitch",
         "disqualification_evidence": "we'd like to pitch our platform"},
        {"company_fit_score": 40, "buying_intent_score": 45, "total_score": 85},
        {"lead_id": "L-9"},
    )
    assert routed["recommendation"] == REVIEW
    assert (routed["total_score"], routed["company_fit_score"]) == (None, None)
    assert routed["needs_human_review"] is True
    assert routed["review_reason"] == recommendation.CONTRADICTION_REVIEW_REASON


def test_a_genuine_ambiguity_keeps_its_own_reason():
    routed = route_lead(
        {"is_potential_buyer": "ambiguous", "non_buyer_signal": "none",
         "ambiguity_reason": "unclear whether they buy or refer"},
        {"company_fit_score": 20, "buying_intent_score": 20, "total_score": 40},
        {"lead_id": "L-9"},
    )
    assert routed["review_reason"] == "unclear whether they buy or refer"


def test_an_unrecognised_signal_goes_to_a_human():
    """Contract drift must neither exclude nor promote a lead."""
    assert recommend(lead("yes", total=95, non_buyer_signal="benchmarking")) == REVIEW
    assert recommend(lead("no", total=10, non_buyer_signal="benchmarking")) == REVIEW


@pytest.mark.parametrize("missing", [None, ""])
def test_a_record_without_the_field_routes_exactly_as_before(missing):
    """Nothing in the pipeline may treat an absent signal as an exclusion."""
    record = lead("yes", fit=40, intent=40, total=80)
    if missing is not None:
        record["non_buyer_signal"] = missing
    assert recommend(record) == CONTACT_NOW


# --------------------------------------------------------------------------- #
# The four routes, stated as the business means them
# --------------------------------------------------------------------------- #
#
# One test per sentence of the specification, written against the shapes real
# extractions produce rather than against the rule that implements them - so
# they still describe the intended behaviour if the implementation is rewritten.

def extraction_shape(buyer="yes", signal="none", **levels):
    base = {"urgency": "low", "pain_severity": "low", "purchasing_readiness": "unknown",
            "timeline": "none", "buying_stage": "exploring"}
    base.update(levels)
    return {
        "is_potential_buyer": buyer, "non_buyer_signal": signal,
        **{factor: {"level": level, "evidence": "x"} for factor, level in base.items()},
    }


def route_of(extraction, fit, intent):
    return recommend({
        "is_potential_buyer": extraction["is_potential_buyer"],
        "non_buyer_signal": extraction["non_buyer_signal"],
        "company_fit_score": fit, "buying_intent_score": intent,
        "total_score": fit + intent,
    })


def test_a_genuine_early_stage_buyer_is_nurtured():
    """"I am researching tools and gathering ideas. No project approved yet."
    Every intent factor at its floor, and still a prospect."""
    early = extraction_shape()          # low/low/unknown/none/exploring
    assert route_of(early, fit=28, intent=6) == NURTURE


def test_a_genuine_low_intent_buyer_is_nurtured():
    """No budget, no timeline, comparing options - weak, not disqualified."""
    low = extraction_shape(urgency="medium", pain_severity="medium",
                           buying_stage="evaluating")
    assert route_of(low, fit=23, intent=17) == NURTURE


@pytest.mark.parametrize("fit,intent", [(0, 0), (10, 4), (23, 8), (50, 14), (44, 30)])
def test_no_genuine_buyer_is_ever_disqualified_by_its_score(fit, intent):
    assert route_of(extraction_shape(), fit, intent) != DISQUALIFY


def test_explicit_competitor_benchmarking_is_disqualified_at_any_score():
    """The case this system exists to catch. A perfect company fit must not
    rescue it, at any total."""
    benchmarking = extraction_shape(buyer="no", signal="competitive_research")
    for fit, intent in ((50, 50), (50, 6), (25, 6), (0, 0)):
        assert route_of(benchmarking, fit, intent) == DISQUALIFY


def test_an_explicit_job_seeker_is_disqualified():
    seeker = extraction_shape(buyer="no", signal="job_seeking")
    assert route_of(seeker, fit=40, intent=40) == DISQUALIFY


def test_an_ambiguous_purpose_goes_to_review():
    """Unresolved purpose, not unresolved readiness - and no score resolves it."""
    unclear = extraction_shape(buyer="ambiguous")
    for fit, intent in ((50, 50), (40, 6), (0, 0)):
        assert route_of(unclear, fit, intent) == REVIEW


def test_a_strong_genuine_buyer_is_contacted_now():
    strong = extraction_shape(urgency="high", pain_severity="high",
                              purchasing_readiness="approved", timeline="near_term",
                              buying_stage="committed")
    assert route_of(strong, fit=46, intent=50) == CONTACT_NOW


def test_the_four_routes_partition_every_genuine_score():
    """Sweep the whole score space a prospect can occupy: two outcomes only,
    split at 75, and DISQUALIFY never among them."""
    seen = {route_of(extraction_shape(), fit, intent)
            for fit in range(0, 51) for intent in range(0, 51)}
    assert seen == {NURTURE, CONTACT_NOW}


# --------------------------------------------------------------------------- #
# The gate must not touch ordinary leads
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "buyer,fit,intent,total,expected,case",
    [
        ("yes", 45, 50, 95, CONTACT_NOW, "A: yes + high"),
        ("yes", 30, 30, 60, NURTURE, "A/B: yes + medium"),
        ("yes", 20, 20, 40, NURTURE, "B: yes + low - early, not excluded"),
        ("yes", 50, 14, 64, NURTURE, "B: yes + weak intent - early, not excluded"),
        ("no", 50, 50, 100, DISQUALIFY, "C: no + high"),
        ("ambiguous", 48, 50, 98, REVIEW, "E: ambiguous + high"),
    ],
)
def test_signal_none_leaves_every_existing_route_unchanged(
    buyer, fit, intent, total, expected, case,
):
    assert recommend(lead(buyer, fit, intent, total, non_buyer_signal="none")) == expected, case


def test_a_non_buyer_without_a_stated_purpose_still_disqualifies():
    """The original gate is untouched: `no` alone is still enough."""
    assert recommend(lead("no", fit=50, intent=50, total=100,
                          non_buyer_signal="none")) == DISQUALIFY


def test_a_weak_but_genuine_buyer_is_nurtured_not_excluded():
    """No budget, no timeline, low intent, total 46 - a poor prospect, not a
    non-buyer. This is the test that fails if the gate ever starts reading
    weakness as a stated purpose."""
    weak = lead("yes", fit=31, intent=15, total=46, non_buyer_signal="none")
    assert recommend(weak) == NURTURE


def test_the_gate_adds_no_fifth_outcome():
    seen = {
        recommend(lead(buyer, total=total, intent=intent, non_buyer_signal=signal))
        for buyer in ("yes", "no", "ambiguous")
        for signal in schema.NON_BUYER_SIGNALS
        for total in (0, 44, 45, 74, 75, 100)
        for intent in (0, 14, 15, 50)
    }
    assert seen <= set(recommendation.ROUTES)


# --------------------------------------------------------------------------- #
# Buyer-status precedence
# --------------------------------------------------------------------------- #

def test_a_perfect_score_cannot_rescue_a_non_buyer():
    """§9: DISQUALIFY 'overriding any score'."""
    assert recommend(lead("no", fit=50, intent=50, total=100)) == DISQUALIFY


def test_a_perfect_score_cannot_route_an_ambiguous_lead():
    """§8: ambiguous is 'not auto-resolved either direction'."""
    assert recommend(lead("ambiguous", fit=50, intent=50, total=100)) == REVIEW


def test_a_zero_score_cannot_downgrade_an_ambiguous_lead():
    assert recommend(lead("ambiguous", fit=0, intent=0, total=0)) == REVIEW


@pytest.mark.parametrize("total", [0, 44, 45, 74, 75, 100])
def test_non_buyer_routes_identically_at_every_total(total):
    assert recommend(lead("no", fit=total // 2, intent=total // 2, total=total)) == DISQUALIFY


@pytest.mark.parametrize("total", [0, 44, 45, 74, 75, 100])
def test_ambiguous_routes_identically_at_every_total(total):
    assert recommend(lead("ambiguous", fit=total // 2, intent=total // 2, total=total)) == REVIEW


def test_ambiguous_with_missing_scores_still_routes_to_review():
    """A REVIEW lead may legitimately carry no score at all."""
    assert recommend({"is_potential_buyer": "ambiguous"}) == REVIEW
    assert recommend({
        "is_potential_buyer": "ambiguous",
        "company_fit_score": None, "buying_intent_score": None, "total_score": None,
    }) == REVIEW


def test_non_buyer_with_missing_scores_still_disqualifies():
    assert recommend({"is_potential_buyer": "no"}) == DISQUALIFY


def test_unrecognised_buyer_value_goes_to_a_human():
    """Defensive, not an architecture rule: the schema permits only three
    values, so a fourth is a contract violation rather than a verdict."""
    for value in (None, "maybe", True, "", "YES"):
        assert recommend(lead(value, fit=50, intent=50, total=100)) == REVIEW


# --------------------------------------------------------------------------- #
# The intent floor, and what replaced it (§9)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "intent,expected", [(0, NURTURE), (13, NURTURE), (14, NURTURE), (15, NURTURE)],
)
def test_a_buyer_below_the_old_intent_floor_is_nurtured_not_deleted(intent, expected):
    """The floor used to disqualify here. Weak intent describes a lead who is
    early, not a lead who is not buying, so it is nurtured and ranked last."""
    assert recommend(lead("yes", fit=50, intent=intent, total=50 + intent)) == expected


def test_the_intent_floor_is_subsumed_by_the_contact_now_line():
    """§9 introduced the floor so fit alone could not reach CONTACT_NOW. That
    guarantee is now arithmetic rather than a rule: company fit is clamped at
    50, so anything under the floor tops out at 64. Removing the rule therefore
    removed no protection - this test is what holds that claim in place."""
    assert (scoring.MAX_COMPANY_FIT + recommendation.INTENT_FLOOR - 1
            < recommendation.CONTACT_NOW_MIN_TOTAL)
    for fit in range(0, scoring.MAX_COMPANY_FIT + 1):
        for intent in range(0, recommendation.INTENT_FLOOR):
            assert recommend(lead("yes", fit, intent, fit + intent)) != CONTACT_NOW


def test_intent_floor_applies_only_to_buyers():
    assert recommend(lead("ambiguous", fit=50, intent=0, total=50)) == REVIEW


# --------------------------------------------------------------------------- #
# Total-score bands (§9)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "total,expected",
    [
        # 45 is no longer a routing boundary: a lead that clears the gates is a
        # prospect at any score, so everything under 75 is NURTURE.
        (0, NURTURE), (44, NURTURE),
        (45, NURTURE), (46, NURTURE), (74, NURTURE),
        (75, CONTACT_NOW), (76, CONTACT_NOW), (100, CONTACT_NOW),
    ],
)
def test_total_band_boundaries(total, expected):
    """The 44/45 and 74/75 edges, asserted exactly."""
    assert recommend(lead("yes", fit=total - 25, intent=25, total=total)) == expected


def test_band_edges_are_inclusive_where_the_document_says_so():
    assert recommend(lead("yes", fit=20, intent=25, total=45)) == NURTURE       # "45-74"
    assert recommend(lead("yes", fit=49, intent=25, total=74)) == NURTURE
    assert recommend(lead("yes", fit=50, intent=25, total=75)) == CONTACT_NOW   # ">= 75"


def test_every_total_maps_to_exactly_one_route():
    for total in range(0, 101):
        route = recommend(lead("yes", fit=max(0, total - 25), intent=25, total=total))
        assert route in (CONTACT_NOW, NURTURE, DISQUALIFY)


# --------------------------------------------------------------------------- #
# Score suppression for REVIEW
# --------------------------------------------------------------------------- #

def extraction(buyer="ambiguous", **extra):
    record = {
        "is_potential_buyer": buyer,
        "buyer_type": "other",
        "ambiguity_reason": "researching the market; intent unclear",
        "disqualification_evidence": None,
    }
    record.update(extra)
    return record


def scores(fit=34, intent=6):
    return {
        "company_fit_score": fit,
        "buying_intent_score": intent,
        "total_score": fit + intent,
        "breakdown": {"urgency": {"level": "low", "points": 2, "evidence": "not specified"}},
    }


def test_review_lead_has_all_three_scores_blanked():
    routed = route_lead(extraction("ambiguous"), scores())
    assert routed["recommendation"] == REVIEW
    assert routed["company_fit_score"] is None
    assert routed["buying_intent_score"] is None
    assert routed["total_score"] is None
    assert routed["scores_suppressed"] is True


def test_suppressed_scores_are_blank_not_zero():
    """A zero would threshold and sort like a real result; None cannot."""
    routed = route_lead(extraction("ambiguous"), scores())
    assert routed["total_score"] is not 0  # noqa: F632 - identity is the point
    assert routed["total_score"] is None


def test_review_lead_keeps_its_breakdown_for_the_human():
    routed = route_lead(extraction("ambiguous"), scores())
    assert routed["breakdown"]["urgency"]["evidence"] == "not specified"


def test_review_lead_carries_the_ambiguity_reason():
    routed = route_lead(extraction("ambiguous"), scores())
    assert routed["needs_human_review"] is True
    assert "researching the market" in routed["review_reason"]


def test_routed_buyer_keeps_its_scores():
    routed = route_lead(extraction("yes"), scores(fit=45, intent=50))
    assert routed["recommendation"] == CONTACT_NOW
    assert (routed["company_fit_score"], routed["total_score"]) == (45, 95)
    assert routed["scores_suppressed"] is False
    assert routed["review_reason"] is None


def test_disqualified_lead_keeps_its_scores():
    """Only REVIEW suppresses; a DISQUALIFY is a scored, explained decision."""
    routed = route_lead(extraction("no"), scores(fit=50, intent=50))
    assert routed["recommendation"] == DISQUALIFY
    assert routed["total_score"] == 100


def test_review_note_states_no_total_and_no_verdict():
    routed = route_lead(extraction("ambiguous"), scores())
    note = recommendation.review_note(routed)
    assert "human review" in note
    assert "researching the market" in note.lower()   # the reason is sentence-cased
    for banned in ("/100", CONTACT_NOW, NURTURE, DISQUALIFY):
        assert banned not in note


def test_review_note_without_a_reason_still_reads_cleanly():
    routed = route_lead(extraction("ambiguous", ambiguity_reason=None), scores())
    assert recommendation.review_note(routed).endswith("whether this is a buyer.")


def test_route_lead_carries_identity_from_the_clean_lead():
    routed = route_lead(
        extraction("yes"), scores(),
        {"lead_id": "L-1", "name_clean": "Ada", "company_clean": "PipeGTM"},
    )
    assert (routed["lead_id"], routed["name"], routed["company"]) == ("L-1", "Ada", "PipeGTM")


# --------------------------------------------------------------------------- #
# Ranking (§10)
# --------------------------------------------------------------------------- #

def routed(lead_id, total, intent, urgency="medium", route=NURTURE):
    return {
        "lead_id": lead_id, "recommendation": route,
        "total_score": total, "buying_intent_score": intent,
        "breakdown": {"urgency": {"level": urgency, "points": 0, "evidence": "x"}},
    }


# --------------------------------------------------------------------------- #
# Route priority: the queue is worked in the order of the recommended action
# --------------------------------------------------------------------------- #

def review_row(lead_id, urgency="medium"):
    """A REVIEW lead as `route_lead` builds one: routed, and unscored."""
    return {
        "lead_id": lead_id, "recommendation": REVIEW,
        "total_score": None, "buying_intent_score": None, "company_fit_score": None,
        "scores_suppressed": True,
        "breakdown": {"urgency": {"level": urgency, "points": 0, "evidence": "x"}},
    }


def test_a_weaker_contact_now_outranks_a_stronger_nurture():
    ranked = rank_leads([routed("nu", 74, 40), routed("cn", 20, 5, route=CONTACT_NOW)])
    assert [x["lead_id"] for x in ranked] == ["cn", "nu"]


def test_a_nurture_lead_outranks_any_review_lead():
    ranked = rank_leads([review_row("rv"), routed("nu", 1, 0)])
    assert [x["lead_id"] for x in ranked] == ["nu", "rv"]


def test_a_review_lead_outranks_any_disqualify_lead():
    """Including a disqualified lead with a near-perfect score - the score
    cannot promote a lead past the action it was assigned."""
    ranked = rank_leads([routed("dq", 99, 50, route=DISQUALIFY), review_row("rv")])
    assert [x["lead_id"] for x in ranked] == ["rv", "dq"]


def test_the_queue_runs_contact_now_nurture_review_disqualify():
    """The full ordering, with each route's scores chosen so that a score-first
    sort would produce a visibly different answer."""
    batch = [
        routed("dq-high", 88, 44, route=DISQUALIFY),
        review_row("rv-b"),
        routed("nu-low", 30, 12),
        routed("cn-high", 98, 50, route=CONTACT_NOW),
        routed("dq-low", 12, 2, route=DISQUALIFY),
        review_row("rv-a"),
        routed("cn-low", 76, 30, route=CONTACT_NOW),
        routed("nu-high", 74, 35),
    ]
    assert [x["lead_id"] for x in rank_leads(batch)] == [
        "cn-high", "cn-low", "nu-high", "nu-low", "rv-a", "rv-b", "dq-high", "dq-low",
    ]


@pytest.mark.parametrize("route", [CONTACT_NOW, NURTURE, DISQUALIFY])
def test_score_still_orders_leads_within_a_route(route):
    batch = [routed("mid", 60, 30, route=route), routed("top", 90, 45, route=route),
             routed("low", 20, 10, route=route)]
    assert [x["lead_id"] for x in rank_leads(batch)] == ["top", "mid", "low"]


def test_review_leads_order_deterministically_without_a_score():
    """They tie at every score position, so the existing tiebreaks decide:
    urgency, then lead_id. No score is invented to break the tie."""
    batch = [review_row("z", urgency="low"), review_row("a", urgency="low"),
             review_row("m", urgency="high")]
    assert [x["lead_id"] for x in rank_leads(batch)] == ["m", "a", "z"]
    assert rank_leads(batch) == rank_leads(list(reversed(batch)))
    assert all(x["total_score"] is None for x in rank_leads(batch))


def test_ranking_changes_nothing_but_the_rank():
    """The guarantee that makes this a presentation change: every other field
    survives untouched, scores and route assignments included."""
    import copy

    batch = [routed("a", 40, 20), routed("b", 90, 45, route=CONTACT_NOW),
             routed("c", 88, 44, route=DISQUALIFY), review_row("r")]
    before = {row["lead_id"]: copy.deepcopy(row) for row in batch}

    for ranked_row in rank_leads(batch):
        original = before[ranked_row["lead_id"]]
        assert set(ranked_row) - set(original) == {"rank"}
        for field, value in original.items():
            assert ranked_row[field] == value, f"{ranked_row['lead_id']}.{field} changed"


def test_route_priority_covers_every_route_exactly_once():
    assert set(recommendation.ROUTE_PRIORITY) == set(recommendation.ROUTES)
    assert sorted(recommendation.ROUTE_PRIORITY.values()) == [0, 1, 2, 3]
    assert [route for route, _ in sorted(recommendation.ROUTE_PRIORITY.items(),
                                         key=lambda item: item[1])] == [
        CONTACT_NOW, NURTURE, REVIEW, DISQUALIFY]


def test_ranking_does_not_change_what_recommend_decides():
    """Ranking reads the route and never contributes to it: `recommend()` is
    asserted over the whole input space it accepts."""
    cases = [
        (buyer, signal, intent, total)
        for buyer in ("yes", "no", "ambiguous", "wat")
        for signal in schema.NON_BUYER_SIGNALS
        for intent in (0, 14, 15, 50)
        for total in (0, 44, 45, 74, 75, 100)
    ]
    routes = [
        recommend({"is_potential_buyer": buyer, "non_buyer_signal": signal,
                   "buying_intent_score": intent, "total_score": total})
        for buyer, signal, intent, total in cases
    ]
    assert len(routes) == len(cases)
    assert set(routes) <= set(recommendation.ROUTES)

    # And every one of those leads keeps the route it was given when ranked.
    batch = [
        {"lead_id": f"L-{i}", "recommendation": route, "total_score": total,
         "buying_intent_score": intent, "breakdown": {}}
        for i, (route, (_, _, intent, total)) in enumerate(zip(routes, cases))
    ]
    assigned = {row["lead_id"]: row["recommendation"] for row in batch}
    after = {row["lead_id"]: row["recommendation"] for row in rank_leads(batch)}
    assert after == assigned


def test_primary_sort_is_total_descending():
    ranked = rank_leads([routed("a", 40, 20), routed("b", 90, 45), routed("c", 60, 30)])
    assert [x["lead_id"] for x in ranked] == ["b", "c", "a"]


def test_tiebreak_one_is_buying_intent_descending():
    ranked = rank_leads([routed("a", 70, 20), routed("b", 70, 40)])
    assert [x["lead_id"] for x in ranked] == ["b", "a"]


def test_tiebreak_two_is_urgency():
    ranked = rank_leads([
        routed("a", 70, 30, urgency="low"),
        routed("b", 70, 30, urgency="high"),
        routed("c", 70, 30, urgency="medium"),
    ])
    assert [x["lead_id"] for x in ranked] == ["b", "c", "a"]


def test_ranking_is_total_and_reproducible():
    """Identical on all three §10 criteria still sorts deterministically."""
    batch = [routed("z", 70, 30), routed("a", 70, 30), routed("m", 70, 30)]
    assert [x["lead_id"] for x in rank_leads(batch)] == ["a", "m", "z"]
    assert rank_leads(batch) == rank_leads(list(reversed(batch)))


def test_rank_is_one_based_and_contiguous():
    ranked = rank_leads([routed("a", 40, 20), routed("b", 90, 45), routed("c", 60, 30)])
    assert [x["rank"] for x in ranked] == [1, 2, 3]


def test_review_leads_are_ranked_between_nurture_and_disqualify():
    """Previously excluded from the ranking on the grounds that §10 sorts on a
    total a REVIEW lead does not have. The route is known for every lead, so it
    can be placed without one - and it still publishes no score."""
    batch = [
        routed("a", 90, 45),
        {"lead_id": "r", "recommendation": REVIEW, "total_score": None,
         "buying_intent_score": None, "breakdown": {}},
        routed("b", 50, 25),
        routed("d", 88, 40, route=DISQUALIFY),
    ]
    ranked = rank_leads(batch)
    assert [x["lead_id"] for x in ranked] == ["a", "b", "r", "d"]
    assert [x["rank"] for x in ranked] == [1, 2, 3, 4]

    review_row = next(x for x in ranked if x["lead_id"] == "r")
    assert review_row["rank"] == 3
    assert review_row["total_score"] is None      # ranked, still unscored


def test_disqualified_leads_are_still_ranked():
    """§10 excludes nothing but the unscored; DISQUALIFY has a real total."""
    batch = [routed("a", 90, 45), routed("d", 20, 5, route=DISQUALIFY)]
    assert [x["lead_id"] for x in rank_leads(batch)] == ["a", "d"]


def test_review_queue_returns_only_review_leads_in_stable_order():
    batch = [
        routed("a", 90, 45),
        {"lead_id": "r2", "recommendation": REVIEW},
        {"lead_id": "r1", "recommendation": REVIEW},
    ]
    assert [x["lead_id"] for x in review_queue(batch)] == ["r1", "r2"]


def test_ranking_does_not_mutate_its_input():
    batch = [routed("a", 40, 20), routed("b", 90, 45)]
    import copy

    before = copy.deepcopy(batch)
    rank_leads(batch)
    assert batch == before


def test_empty_batch_ranks_to_empty():
    assert rank_leads([]) == []
    assert review_queue([]) == []


# --------------------------------------------------------------------------- #
# Summary tiles (§13 + the agreed fifth bucket)
# --------------------------------------------------------------------------- #

def test_summary_counts_all_five_tiles():
    batch = [
        routed("a", 90, 45, route=CONTACT_NOW),
        routed("b", 60, 30, route=NURTURE),
        routed("c", 20, 5, route=DISQUALIFY),
        {"lead_id": "r", "recommendation": REVIEW},
    ]
    summary = summarise_routes(batch)
    assert summary == {
        "total_processed": 4, CONTACT_NOW: 1, NURTURE: 1, DISQUALIFY: 1, REVIEW: 1,
    }


def test_summary_of_an_empty_batch_is_all_zero():
    summary = summarise_routes([])
    assert summary["total_processed"] == 0
    assert all(summary[route] == 0 for route in recommendation.ROUTES)


def test_there_are_exactly_four_routing_outcomes():
    assert len(recommendation.ROUTES) == 4
    assert set(recommendation.ROUTES) == {CONTACT_NOW, NURTURE, DISQUALIFY, REVIEW}


def test_total_processed_is_a_metric_not_a_route():
    """Five tiles, four routes: the extra key counts leads, it is not a bucket."""
    assert recommendation.TOTAL_PROCESSED_KEY not in recommendation.ROUTES
    batch = [
        routed("a", 90, 45, route=CONTACT_NOW),
        routed("b", 60, 30, route=NURTURE),
        routed("c", 20, 5, route=DISQUALIFY),
        {"lead_id": "r", "recommendation": REVIEW},
    ]
    summary = summarise_routes(batch)
    assert summary["total_processed"] == sum(summary[r] for r in recommendation.ROUTES)
    assert len(summary) == len(recommendation.ROUTES) + 1


def test_every_lead_receives_exactly_one_of_the_four_routes():
    for buyer in ("yes", "no", "ambiguous", "nonsense"):
        for total in (0, 44, 45, 74, 75, 100):
            route = recommend(lead(buyer, fit=total // 2, intent=total // 2, total=total))
            assert route in recommendation.ROUTES


# --------------------------------------------------------------------------- #
# Layer boundaries
# --------------------------------------------------------------------------- #

def test_recommendation_module_computes_no_points():
    """Stage 6 compares totals; it must never build one."""
    import ast

    tree = ast.parse(open(recommendation.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for banned in ("scoring", "llm_extraction", "providers", "anthropic", "google"):
        assert banned not in imported, f"recommendation.py must not import {banned}"


def test_recommend_is_deterministic():
    candidate = lead("yes", fit=40, intent=35, total=75)
    assert recommend(candidate) == recommend(candidate) == CONTACT_NOW


def test_recommend_does_not_mutate_its_input():
    candidate = lead("yes", fit=40, intent=35, total=75)
    import copy

    before = copy.deepcopy(candidate)
    recommend(candidate)
    assert candidate == before


# --------------------------------------------------------------------------- #
# End-to-end against the real Stage 5 scorer
# --------------------------------------------------------------------------- #

def test_the_five_stage4_leads_route_as_expected():
    """Guards the seam between Stage 5's output shape and Stage 6's input."""
    from scoring import score_lead

    def measure(stated, low=None, high=None, kind="missing", ev="not specified"):
        return {"stated": stated, "low": low, "high": high, "kind": kind, "evidence": ev}

    def build(buyer, industry, seniority, urgency, pain, readiness, timeline, stage,
              emp=None, price_sensitive=False, reason=None):
        return {
            "is_potential_buyer": buyer, "buyer_type": "business_prospect",
            "disqualification_evidence": None, "ambiguity_reason": reason,
            "industry_fit": {"level": industry, "evidence": "x"},
            "employee_count_in_notes": emp or measure(False),
            "budget_in_notes": measure(False),
            "seniority": {"level": seniority, "evidence": "x"},
            "urgency": {"level": urgency, "evidence": "x"},
            "pain_severity": {"level": pain, "evidence": "x"},
            "purchasing_readiness": {"level": readiness, "evidence": "x"},
            "timeline": {"level": timeline, "evidence": "x"},
            "buying_stage": {"level": stage, "evidence": "x"},
            "price_sensitive_flag": price_sensitive,
        }

    clean = {"source_clean": "event", "budget_low": 10000, "budget_high": 10000,
             "budget_status": "exact", "budget_display": "10000",
             "employees_low": None, "employees_high": None,
             "employees_status": "missing", "employees_display": "unknown"}

    strong = build("yes", "high", "decision_maker", "high", "high", "approved",
                   "near_term", "committed",
                   emp=measure(True, 40, 40, "exact", "40 people"))
    weak = build("yes", "medium", "decision_maker", "low", "medium", "unknown",
                 "none", "exploring", price_sensitive=True)
    unclear = build("ambiguous", "high", "decision_maker", "low", "low", "unknown",
                    "none", "exploring", reason="researching the market")

    routed_strong = route_lead(strong, score_lead(strong, clean), {"lead_id": "L-1009"})
    routed_weak = route_lead(weak, score_lead(weak, clean), {"lead_id": "L-1033"})
    routed_unclear = route_lead(unclear, score_lead(unclear, clean), {"lead_id": "L-1261"})

    assert routed_strong["recommendation"] == CONTACT_NOW
    assert routed_strong["total_score"] == 100
    # L-1033 is a solo consultant with a tiny budget: a weak prospect, and a
    # prospect all the same. It ranks last rather than being thrown away.
    assert routed_weak["recommendation"] == NURTURE
    assert routed_unclear["recommendation"] == REVIEW
    assert routed_unclear["total_score"] is None            # suppressed

    # One queue: CONTACT_NOW, then NURTURE, then the REVIEW lead last of these
    # three - ahead of any DISQUALIFY, of which this batch has none.
    ranked = rank_leads([routed_strong, routed_weak, routed_unclear])
    assert [x["lead_id"] for x in ranked] == ["L-1009", "L-1033", "L-1261"]
    # The review bucket is still available on its own for callers that want it.
    assert [x["lead_id"] for x in review_queue([routed_unclear])] == ["L-1261"]
