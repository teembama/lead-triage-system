"""Stage 6 - routing and ranking.

Turns a buyer status plus a score into one of four routes, and orders the
routable leads. Every rule here comes from §8, §9 and §10 of the architecture;
no additional routing rule is introduced.

## Order of evaluation (§8/§9)

The buyer gate is described in §8 as *pre-scoring*, and §9 states that
`is_potential_buyer = false` disqualifies "overriding any score". So buyer
status is read first and a score can never overturn it:

    1. a stated non-buying purpose AND stated purchase commitment -> REVIEW
    2. a stated non-buying purpose                    -> DISQUALIFY
    3. buyer == "no"        -> DISQUALIFY   (score never consulted)
    4. buyer == "ambiguous" -> REVIEW       (score never consulted)
    5. buyer == "yes":
         a. total >= 75 and intent >= 35 -> CONTACT_NOW
         b. otherwise                    -> NURTURE

Steps 1 and 2 are the exclusion gate. `non_buyer_signal` carries what the writer
said they came here to do; anything other than "none" is a purpose that is not
buying, and no score can outweigh it. It sits above the buyer status because it
is the more specific statement: "I am here to benchmark your pricing" settles
the question that `is_potential_buyer` was only estimating.

Step 1 is the safeguard for the one case where the two genuinely disagree: the
notes state a non-buying purpose *and* state that a purchase is under way.
That is insufficient information rather than grounds for deletion, so it goes
to a person. Being labelled a buyer is not itself enough - a competitor is
excluded on what the notes say, not on whether the extraction happened to call
them a prospect.

## Why the score no longer disqualifies

Steps 1-4 decide whether this is a prospect; step 5 decides how urgent one is.
That division is the whole point. The score measures how strong an opportunity
looks, and a weak opportunity is still an opportunity - "no budget yet, no
timeline, still reading around" describes a buyer who is early, not a person who
is not buying. Two earlier rules asked the score to answer the first question:
a hard floor on intent, and a DISQUALIFY band below 45. Both are gone.

Nothing is lost by removing the intent floor. §9 introduced it so that fit alone
could not carry a lead to CONTACT_NOW, and that guarantee is now arithmetic
rather than a rule: company fit is clamped at 50, so a lead under the floor tops
out at 64 and cannot reach 75 by any combination of factors. The floor's only
remaining effect was to delete early-stage buyers, which is the behaviour this
routing exists to avoid. `test_the_intent_floor_is_subsumed_by_the_contact_now_line`
holds the arithmetic in place.

The consequence is deliberate and worth stating plainly: DISQUALIFY now means
"not a prospect", and nothing else. A lead's score decides where it ranks and
whether it is contacted today - never whether it counts as a lead at all.

REVIEW is the fifth bucket for §8's "flagged for human review, not auto-resolved
either direction". Its scores are suppressed: an ambiguous lead is not scored
for routing, so publishing a total would invite exactly the auto-resolution §8
forbids. The factor breakdown is kept, because the levels and evidence are what
a human needs in order to make the call.

This module computes no points. It reads the totals Stage 5 produced and
compares them against thresholds.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from schema import NON_BUYER_SIGNALS

# --------------------------------------------------------------------------- #
# Routes and thresholds (plan §9)
# --------------------------------------------------------------------------- #

# There are exactly FOUR routing outcomes. Every lead receives exactly one of
# them, and there is no fifth.
CONTACT_NOW = "CONTACT_NOW"
NURTURE = "NURTURE"
DISQUALIFY = "DISQUALIFY"
REVIEW = "REVIEW"

ROUTES = (CONTACT_NOW, NURTURE, DISQUALIFY, REVIEW)

# The three routes that carry a published score. REVIEW is excluded because its
# scores are suppressed, not because it is a lesser kind of outcome.
SCORED_ROUTES = (CONTACT_NOW, NURTURE, DISQUALIFY)

# `total_processed` in summarise_routes() is a COUNT OF LEADS, not a route.
# The UI shows five tiles - one running total plus the four routes - and the
# four route counts sum to the total. Do not read the tile count as a bucket
# count; "Total" is a metric, and no lead is ever routed to it.
TOTAL_PROCESSED_KEY = "total_processed"

CONTACT_NOW_MIN_TOTAL = 75  # §9: total >= 75.

CONTACT_NOW_MIN_INTENT = 35
"""Buying intent a lead must show before it is worth contacting today.

Not a new threshold on the score - the 75 line is untouched - but a second
condition on the same decision, and it exists because 75 alone stopped meaning
what it used to. With no target industry set, `industry_fit` awards every lead
the full 20 points, so company fit sits near its ceiling for everyone and the
total clears 75 on roughly 27 points of intent. "Decision in about a month,
budget not locked, still exploring" scores about that, and it is not a lead to
telephone this afternoon.

35 is read off the data rather than picked: in the 509-lead run the leads in
question scored 25-34 and the genuine ones scored 50, so the gap between 34 and
50 is empty and the cut is not delicate.

Distinct from `INTENT_FLOOR` below, which was a retired *disqualification* rule.
Failing this one costs a lead its priority, never its place in the queue."""

# Kept as documentation of two §9 rules that no longer route anything, so the
# deviation stays visible instead of looking like an oversight.
INTENT_FLOOR = 15
"""§9's hard floor on intent. No longer disqualifies: the guarantee it was
written for - that fit alone cannot reach CONTACT_NOW - is now implied by
`MAX_COMPANY_FIT + (INTENT_FLOOR - 1) < CONTACT_NOW_MIN_TOTAL`, which a test
asserts directly."""

NURTURE_MIN_TOTAL = 45
"""§9's NURTURE/DISQUALIFY line. No longer a routing boundary: a lead that
clears the gates is a prospect, and a prospect is never disqualified by its
score. Every scored lead below CONTACT_NOW_MIN_TOTAL is NURTURE."""

# §10 tiebreak 2 - urgency, high > medium > low.
URGENCY_RANK = {"high": 3, "medium": 2, "low": 1}

# Levels that state a purchase is actually under way, rather than merely being
# considered. A lead benchmarking your pricing can describe interest, a problem,
# even a timeline for "a decision" - what it does not do is say the budget is
# signed off or that it has settled on what it wants.
#
# Used for one thing only: telling a competitor apart from a competitor who is
# also buying. Nothing here scores anything.
PURCHASE_COMMITMENT_LEVELS = {
    "purchasing_readiness": {"approved"},
    "buying_stage": {"committed"},
}

# The order the queue is worked in. The route is the recommended action, so it
# leads the sort and the score orders leads within each action rather than
# across all of them - otherwise a disqualified lead with a high fit score sits
# above a contactable one, which is not a queue anybody works top-down.
#
# This is a presentation decision and nothing more. `recommend()` has already
# chosen the route by the time any of this runs; ranking reads that choice and
# never contributes to it.
ROUTE_PRIORITY = {CONTACT_NOW: 0, NURTURE: 1, REVIEW: 2, DISQUALIFY: 3}


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #

def recommend(scored_lead: dict[str, Any]) -> str:
    """Route one lead. Reads the exclusion gate and buyer status first; a score
    never overrides either.

    `scored_lead` carries the extraction's `is_potential_buyer` and
    `non_buyer_signal` alongside the subtotals produced by Stage 5.
    """
    buyer = scored_lead.get("is_potential_buyer")
    signal = scored_lead.get("non_buyer_signal") or "none"

    if signal not in NON_BUYER_SIGNALS:
        # Contract drift. Neither excluding nor promoting a lead on a value the
        # system does not understand; the same treatment an unrecognised buyer
        # status gets below.
        return REVIEW

    if signal != "none":
        # The score is not read on either branch: an explicitly stated
        # non-buying purpose is settled before scoring is relevant.
        #
        # It used to be enough for the model to also call the writer a buyer to
        # divert them to a human, which meant a competitor was excluded only
        # when the extraction happened not to say `yes`. Being called a buyer is
        # not evidence of buying; saying the budget is approved is. So the
        # exception now needs the notes to state a purchase actually under way -
        # "budget approved, starting this month, and we will benchmark your
        # pricing against ours" is two facts that cannot both be acted on
        # automatically, and only that shape reaches a person.
        if buyer == "yes" and _states_purchase_commitment(scored_lead):
            return REVIEW
        return DISQUALIFY

    # §9: "overriding any score" - the score is not read on either branch below.
    if buyer == "no":
        return DISQUALIFY
    if buyer == "ambiguous":
        return REVIEW

    if buyer != "yes":
        # Not an architecture rule: the schema permits only yes/no/ambiguous, so
        # anything else is a contract violation. Routing it to a human is the
        # only option that neither discards a possible buyer nor fabricates a
        # verdict from a value the system does not understand.
        return REVIEW

    # A prospect from here. The score sets priority, not prospect-hood: it says
    # whether to call them today, and where they rank - never whether they count.
    #
    # Contacting today asks for two things: a strong lead overall, and buying
    # intent that is actually current. A high total earned mostly on fit
    # describes a good prospect who is not ready, and falling short here costs a
    # lead its priority and nothing else - it stays in the queue as NURTURE.
    total = _score(scored_lead, "total_score")
    intent = _score(scored_lead, "buying_intent_score")
    if total >= CONTACT_NOW_MIN_TOTAL and intent >= CONTACT_NOW_MIN_INTENT:
        return CONTACT_NOW
    return NURTURE


def _states_purchase_commitment(scored_lead: dict[str, Any]) -> bool:
    """Do the notes say a purchase is under way, not just contemplated?

    Read from the levels Stage 4 already extracted - no new field, no new
    weight, and nothing re-interpreted here. A lead whose breakdown is absent
    reads as "no", which is the safe direction: an explicit non-buying purpose
    then stands rather than being softened into a review.
    """
    breakdown = scored_lead.get("breakdown") or {}
    return any(
        (breakdown.get(factor) or {}).get("level") in levels
        for factor, levels in PURCHASE_COMMITMENT_LEVELS.items()
    )


def _score(lead: dict[str, Any], key: str) -> float:
    value = lead.get(key)
    return float(value) if isinstance(value, (int, float)) and value == value else 0.0


def route_lead(
    extraction: dict[str, Any],
    scores: dict[str, Any],
    clean_lead: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble the routed record, applying score suppression for REVIEW.

    Stage 5 scores every lead, including ambiguous ones. Routing is where that
    score is either published or withheld: a REVIEW lead's three score fields
    are blanked so nothing downstream can rank, threshold or otherwise
    auto-resolve it. The breakdown survives, since its levels and evidence are
    the material a reviewer actually needs.
    """
    clean_lead = clean_lead or {}
    candidate = {
        "is_potential_buyer": extraction.get("is_potential_buyer"),
        "non_buyer_signal": extraction.get("non_buyer_signal"),
        # Carried so the dual-intent exception can read the purchase signals.
        # Read only; routing never writes to or re-scores the breakdown.
        "breakdown": scores.get("breakdown", {}),
        "company_fit_score": scores.get("company_fit_score"),
        "buying_intent_score": scores.get("buying_intent_score"),
        "total_score": scores.get("total_score"),
    }
    route = recommend(candidate)
    suppressed = route == REVIEW

    return {
        "lead_id": clean_lead.get("lead_id") or extraction.get("lead_id"),
        "name": clean_lead.get("name_clean"),
        "company": clean_lead.get("company_clean"),
        "recommendation": route,
        "is_potential_buyer": extraction.get("is_potential_buyer"),
        "buyer_type": extraction.get("buyer_type"),
        # Carried so the interface can explain an exclusion in words. The value
        # itself is an internal token and is never shown.
        "non_buyer_signal": extraction.get("non_buyer_signal") or "none",
        # Blank, not zero: a suppressed score is an absent one, and a zero would
        # sort and threshold like a real result.
        "company_fit_score": None if suppressed else scores.get("company_fit_score"),
        "buying_intent_score": None if suppressed else scores.get("buying_intent_score"),
        "total_score": None if suppressed else scores.get("total_score"),
        "scores_suppressed": suppressed,
        "needs_human_review": suppressed,
        "review_reason": _review_reason(extraction) if suppressed else None,
        "disqualification_evidence": extraction.get("disqualification_evidence"),
        "breakdown": scores.get("breakdown", {}),
    }


CONTRADICTION_REVIEW_REASON = (
    "the lead gives a reason for getting in touch that is not buying, but also "
    "reads as a genuine enquiry"
)


def _review_reason(extraction: dict[str, Any]) -> Optional[str]:
    """Why a lead was held back, in the reviewer's language.

    A contradiction carries no `ambiguity_reason` - the model was not ambiguous,
    it gave two answers that cannot both be true - so it needs its own sentence
    rather than falling through to the generic one about unsettled notes.
    """
    stated = extraction.get("ambiguity_reason")
    if _nonempty(stated):
        return stated
    signal = extraction.get("non_buyer_signal") or "none"
    if signal != "none" and extraction.get("is_potential_buyer") == "yes":
        return CONTRADICTION_REVIEW_REASON
    return stated


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def review_note(routed_lead: dict[str, Any]) -> str:
    """The summary line for a REVIEW lead - states no total and no verdict."""
    reason = (routed_lead.get("review_reason") or "").strip()
    base = "Flagged for human review: the notes do not settle whether this is a buyer."
    return f"{base} {reason[0].upper() + reason[1:]}." if reason else base


# --------------------------------------------------------------------------- #
# Ranking (plan §10)
# --------------------------------------------------------------------------- #

def _sort_key(lead: dict[str, Any]) -> tuple:
    """Route first, then §10's criteria within it.

    A REVIEW lead reaches the score positions carrying nothing - its three score
    fields are deliberately blank - so every REVIEW lead ties there and the band
    is ordered by the tiebreaks below it. That is intended: suppressed scores
    stay suppressed, and ordering a band with no scores falls to urgency and
    then lead_id, which is deterministic without inventing a total.
    """
    urgency = ((lead.get("breakdown") or {}).get("urgency") or {}).get("level")
    return (
        # Primary: the recommended action, read from `recommend()`'s output.
        ROUTE_PRIORITY.get(lead.get("recommendation"), len(ROUTE_PRIORITY)),
        -_score(lead, "total_score"),                   # then: total desc
        -_score(lead, "buying_intent_score"),           # tiebreak 1: intent desc
        -URGENCY_RANK.get(urgency, 0),                  # tiebreak 2: urgency desc
        str(lead.get("lead_id") or ""),                 # stable, not an §10 rule
    )


def rank_leads(scored_leads: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order every lead by route, then by §10 within the route, and stamp a
    1-based `rank`.

    All four routes are ranked in one queue. REVIEW was previously excluded on
    the grounds that §10 sorts on a total it does not have - true of the score,
    but the route is known for every lead, so a REVIEW lead can be placed
    without one. It sits below NURTURE and above DISQUALIFY, which is where a
    reader looking for the next thing to do would expect to find it, and it
    still publishes no score.

    Nothing here decides anything. `recommendation` arrives already chosen and
    is read, never written; the only field this function adds is `rank`.

    The final tiebreak is `lead_id`, so the ordering is total rather than
    partial - two leads identical on all the criteria above still sort
    reproducibly instead of depending on input order.
    """
    ordered = sorted(scored_leads, key=_sort_key)
    return [{**lead, "rank": index} for index, lead in enumerate(ordered, start=1)]


def review_queue(scored_leads: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The REVIEW bucket on its own, ordered by lead_id.

    Kept for callers that want the review work in isolation. The main queue no
    longer needs it: `rank_leads` ranks all four routes together.
    """
    return sorted(
        (lead for lead in scored_leads if lead.get("recommendation") == REVIEW),
        key=lambda lead: str(lead.get("lead_id") or ""),
    )


def summarise_routes(scored_leads: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Counts for the UI metric tiles.

    Returns five keys but describes FOUR routes: `total_processed` is how many
    leads were seen, and the four route counts sum to it. `total_processed` is
    a metric, never a destination - no lead is ever routed to "Total".
    """
    leads = list(scored_leads)
    counts = {route: 0 for route in ROUTES}
    for lead in leads:
        route = lead.get("recommendation")
        if route in counts:
            counts[route] += 1
    return {TOTAL_PROCESSED_KEY: len(leads), **counts}
