"""Stage 4 live test - the five specified leads, real notes, real provider.

Marked `live` and deselected by default (see pytest.ini) because it costs API
calls. Run explicitly:

    python -m pytest tests/test_extraction_live.py -m live -v

Skips cleanly - never fails - when no key is configured.

Deliberately five leads, not the full dataset. Stage 4 is being validated for
extraction quality; there is no reason to spend quota on 520 rows to learn
whether the contract holds.

Every lead is loaded from the assessment CSV by ID so the notes are the real
ones. Expectations are loose where the notes are genuinely open to reading: the
point is to catch a broken gate, a keyword misfire or a mis-wired contract, not
to pin the model to one reading of an ambiguous enquiry.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm_extraction  # noqa: E402
from target_profile import TargetProfile  # noqa: E402
import providers  # noqa: E402
import schema  # noqa: E402
from data_cleaning import clean_dataframe, read_leads_csv  # noqa: E402

pytestmark = pytest.mark.live

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_CSV = os.path.join(ROOT, "Cohort 3 Assessment — Task 1 Leads (messy).csv")

# Location left unset on purpose: these cases test the buyer gate and industry
# fit, and an unset location is inert by design.
LIVE_PROFILE = TargetProfile.create("Marketing & Advertising", 4_000, 20_000)

# The five specified leads.
#   buyer        : allowed values of is_potential_buyer
#   buyer_type   : allowed values, where the category is clear
#   not_type     : values that would indicate a keyword misfire
#   expect       : factor -> allowed levels
#   emp_*/bud_*  : what the NOTES state (never the CRM field)
CASES: dict[str, dict] = {
    "L-1168": {
        "company": "PipeGTM",
        "buyer": {"yes"}, "buyer_type": {"business_prospect"},
        "expect": {"industry_fit": {"high"}, "urgency": {"high"},
                   "pain_severity": {"high"}, "purchasing_readiness": {"approved"},
                   "buying_stage": {"committed"}},
        "emp_low": 26, "emp_high": 26, "emp_kind": "exact",
        "note": "the dominant buyer template: agency, named pain, budget approved, ASAP",
    },
    "1137": {
        "company": "NimbleMedia",
        "buyer": {"yes"}, "buyer_type": {"business_prospect"},
        "expect": {"industry_fit": {"high"}, "seniority": {"decision_maker"},
                   "purchasing_readiness": {"approved"}, "timeline": {"near_term"}},
        "emp_low": 23, "emp_high": 23, "emp_kind": "exact",
        "note": "notes say 23 people, CRM field says 9 - both must survive as facts",
    },
    "L-1033": {
        "company": "EarlyStageCo",
        "buyer": {"yes"}, "buyer_type": {"freelancer_solo", "business_prospect"},
        "note": "solo consultant, tiny budget - a weak buyer, never gated out",
    },
    "L-1009": {
        "company": "Northforge",
        "buyer": {"yes"}, "buyer_type": {"business_prospect"},
        "not_type": {"recruiter"},
        "expect": {"industry_fit": {"high", "medium"}, "seniority": {"decision_maker"},
                   "purchasing_readiness": {"approved"}},
        "emp_low": 40, "emp_high": 40, "emp_kind": "exact",
        "note": "keyword trap: a recruitment MARKETING agency is a buyer, not a recruiter",
    },
    "L-1261": {
        "company": "CladwellWorks",
        "buyer": {"ambiguous"},
        "note": "'fellow agency owner, mostly researching the market' - genuinely unresolved",
    },
    # Counter-example, added with the L-1261 prompt fix: guards against
    # over-correcting. A peer agency with a STATED benchmarking purpose is a
    # competitor, and must not drift into `ambiguous` along with L-1261.
    "L-1144": {
        "company": None,
        "buyer": {"no", "ambiguous"}, "buyer_type": {"competitor", "other"},
        "signal": {"competitive_research"},
        "note": "'we do similar work - curious about your pricing for benchmarking'",
    },
}


@pytest.fixture(scope="module")
def leads_by_id():
    if not os.path.exists(SOURCE_CSV):
        pytest.skip("assessment CSV not present")
    clean_df, _ = clean_dataframe(read_leads_csv(SOURCE_CSV))
    records = {row["lead_id"]: row for row in clean_df.to_dict("records")}
    missing = [lid for lid in CASES if lid not in records]
    if missing:
        pytest.skip(f"leads not found in CSV: {missing}")
    return records


@pytest.fixture(scope="module")
def extractions(leads_by_id):
    try:
        provider = providers.get_provider()
    except providers.ProviderConfigError as exc:
        pytest.skip(f"provider not configured: {exc}")
    if hasattr(provider, "is_configured") and not provider.is_configured():
        pytest.skip(f"no API key configured for provider {provider.name!r}")

    leads = [leads_by_id[lid] for lid in CASES]
    # The app always assesses against a target profile, so these tests do too.
    # The assessment file is an agency's own inbound, hence this profile; without
    # one there is no yardstick and industry fit is necessarily vaguer.
    results = llm_extraction.extract_batch(
        leads, provider,
        cache=llm_extraction.ExtractionCache(path=None),
        profile=LIVE_PROFILE,
        max_workers=3,   # a handful of leads on a free tier: gentle by design
    )
    return dict(zip(CASES, results))


@pytest.mark.parametrize("lead_id", list(CASES))
def test_extraction_satisfies_the_contract(lead_id, extractions):
    result = extractions[lead_id]
    assert result.status != "failed", result.error
    assert schema.validate_extraction(result.extraction) == []


@pytest.mark.parametrize("lead_id", list(CASES))
def test_every_factor_carries_concise_evidence(lead_id, extractions):
    for factor, snippet in schema.evidence_fields(extractions[lead_id].extraction).items():
        assert snippet and snippet.strip(), f"{lead_id}: {factor} evidence is empty"
        assert len(snippet) <= schema.EVIDENCE_MAX_CHARS * 2, (
            f"{lead_id}: {factor} evidence reads as prose, not a pointer: {snippet!r}"
        )


@pytest.mark.parametrize("lead_id", list(CASES))
def test_buyer_gate_matches_expectation(lead_id, extractions):
    case = CASES[lead_id]
    actual = extractions[lead_id].extraction["is_potential_buyer"]
    assert actual in case["buyer"], (
        f"{lead_id} ({case['note']}): expected {case['buyer']}, got {actual!r}"
    )


@pytest.mark.parametrize("lead_id", [lid for lid, c in CASES.items() if "buyer_type" in c])
def test_buyer_type_matches_expectation(lead_id, extractions):
    case = CASES[lead_id]
    actual = extractions[lead_id].extraction["buyer_type"]
    assert actual in case["buyer_type"], (
        f"{lead_id} ({case['note']}): expected {case['buyer_type']}, got {actual!r}"
    )


@pytest.mark.parametrize("lead_id", [lid for lid, c in CASES.items() if "signal" in c])
def test_stated_purpose_matches_expectation(lead_id, extractions):
    case = CASES[lead_id]
    actual = extractions[lead_id].extraction["non_buyer_signal"]
    assert actual in case["signal"], (
        f"{lead_id} ({case['note']}): expected {case['signal']}, got {actual!r}"
    )


@pytest.mark.parametrize("lead_id", [lid for lid, c in CASES.items() if "signal" not in c])
def test_the_assessment_leads_state_no_non_buying_purpose(lead_id, extractions):
    """None of the five original leads says why-not-buying, including L-1261,
    which is unresolved rather than excluded."""
    assert extractions[lead_id].extraction["non_buyer_signal"] == "none"


@pytest.mark.parametrize("lead_id", [lid for lid, c in CASES.items() if "not_type" in c])
def test_keyword_trap_did_not_misfire(lead_id, extractions):
    """'recruitment marketing agency' must not be read as 'recruiter'."""
    case = CASES[lead_id]
    extraction = extractions[lead_id].extraction
    assert extraction["buyer_type"] not in case["not_type"], (
        f"{lead_id}: keyword misfire - classified as {extraction['buyer_type']!r}"
    )
    assert extraction["is_potential_buyer"] == "yes"


@pytest.mark.parametrize("lead_id", [lid for lid, c in CASES.items() if "expect" in c])
def test_signal_levels_match_expectation(lead_id, extractions):
    extraction = extractions[lead_id].extraction
    for factor, allowed in CASES[lead_id]["expect"].items():
        actual = extraction[factor]["level"]
        assert actual in allowed, (
            f"{lead_id}: {factor} expected one of {allowed}, got {actual!r} "
            f"(evidence: {extraction[factor]['evidence']!r})"
        )


@pytest.mark.parametrize("lead_id", [lid for lid, c in CASES.items() if "emp_low" in c])
def test_employee_count_is_read_from_the_notes_not_the_crm_field(lead_id, extractions):
    case = CASES[lead_id]
    block = extractions[lead_id].extraction["employee_count_in_notes"]
    assert block["stated"] is True, f"{lead_id}: notes state a headcount but stated=False"
    assert block["low"] == case["emp_low"], f"{lead_id}: {block!r}"
    assert block["high"] == case["emp_high"], f"{lead_id}: {block!r}"
    assert block["kind"] == case["emp_kind"], f"{lead_id}: {block!r}"


def test_crm_conflict_is_preserved_as_two_separate_facts(extractions, leads_by_id):
    """1137: the CRM says 9, the notes say 23. Stage 4 records both and resolves
    neither - precedence is a policy decision that lives elsewhere."""
    crm_value = leads_by_id["1137"]["employees_low"]
    notes_block = extractions["1137"].extraction["employee_count_in_notes"]
    assert crm_value == 9, "CRM field should still read 9 after cleaning"
    assert notes_block["low"] == 23, "notes-stated headcount should read 23"
    assert crm_value != notes_block["low"], "the conflict must survive extraction"


@pytest.mark.parametrize("lead_id", list(CASES))
def test_no_floor_ever_carries_an_invented_upper_bound(lead_id, extractions):
    for factor in schema.MEASURE_FACTORS:
        block = extractions[lead_id].extraction[factor]
        if block["kind"] == "floor":
            assert block["high"] is None, f"{lead_id}: {factor} invented a ceiling: {block!r}"


@pytest.mark.parametrize("lead_id", list(CASES))
def test_ambiguous_leads_explain_themselves(lead_id, extractions):
    extraction = extractions[lead_id].extraction
    if extraction["is_potential_buyer"] == "ambiguous":
        assert extraction["ambiguity_reason"], f"{lead_id}: ambiguous with no reason given"


# --------------------------------------------------------------------------- #
# The stated-purpose gate, on synthetic notes
# --------------------------------------------------------------------------- #
#
# Written rather than drawn from the CSV: the assessment file has no job seeker,
# journalist or vendor pitch in it, and the gate has to be exercised on the
# wording it exists to catch. `expect` is the required signal; `route` is what
# the whole pipeline must then do with it.

PURPOSE_CASES: dict[str, dict] = {
    "benchmarking": {
        "notes": "We do similar work and are comparing your pricing for benchmarking "
                 "our own offering.",
        "expect": {"competitive_research"}, "route": {"DISQUALIFY"},
        "note": "the reported case: a stated competitive purpose",
    },
    "vague research": {
        "notes": "Fellow agency owner here, mostly researching the market.",
        "expect": {"none"}, "route": {"REVIEW"},
        "buyer": {"ambiguous"},
        "note": "vague research language is NOT a stated non-buying purpose",
    },
    "peer agency buying": {
        "notes": "We run a recruitment marketing agency, 30 people, and want to "
                 "automate our own lead follow-up. Budget approved, moving this month.",
        "expect": {"none"}, "route": {"CONTACT_NOW", "NURTURE"},
        "buyer": {"yes"},
        "note": "a related industry with real intent to buy is a buyer",
    },
    "comparing in order to buy": {
        "notes": "I'm comparing vendors because we're ready to purchase this month. "
                 "35-person agency, budget signed off.",
        "expect": {"none"}, "route": {"CONTACT_NOW", "NURTURE"},
        "buyer": {"yes"},
        "note": "comparing suppliers in order to buy is buying behaviour",
    },
    # Early, unready, and entirely genuine. This is the distinction the buyer
    # gate previously got wrong: it read "no project approved yet" as an
    # unresolved *purpose* and sent a routine prospect to a human queue. The
    # purpose is plain; only the readiness is not.
    "early but genuine": {
        "notes": "I am researching tools and gathering ideas. No project approved yet.",
        "expect": {"none"}, "buyer": {"yes"},
        "route": {"NURTURE"},
        "note": "early is not unresolved and not a non-buyer: scored, and nurtured",
    },
    "vague curiosity": {   # L-1148, verbatim from the assessment file
        "notes": "Curious about automating stitching together reporting for 40+ "
                 "clients. Not totally sure what we need yet.",
        "expect": {"none"}, "buyer": {"yes"},
        "route": {"CONTACT_NOW", "NURTURE"},
        "note": "vague interest with no budget or timeline is still a buyer",
    },
    "same industry, genuine": {   # L-1205, verbatim
        "notes": "We're a SEO agency. Want to add an AI service line for clients. "
                 "Not sure who signs off internally.",
        "expect": {"none"}, "buyer": {"yes"},
        "route": {"CONTACT_NOW", "NURTURE"},
        "note": "working in the same industry is not a non-buying purpose",
    },
    "solo operator, tiny budget": {   # L-1033, verbatim
        "notes": "solo consultant wanting to automate my own outreach. tiny budget, "
                 "one-man shop.",
        "expect": {"none"}, "buyer": {"yes"}, "route": {"NURTURE"},
        "note": "a weak prospect is a prospect - it ranks last, it is not deleted",
    },
    "researching is their work": {   # L-1357, verbatim - the keyword trap
        "notes": "We're a social media agency, 35 people. Researching prospects and "
                 "drafting first-touch messages is eating our week. Want it automated "
                 "end to end. Budget approved, ready to move.",
        "expect": {"none"}, "buyer": {"yes"}, "route": {"CONTACT_NOW"},
        "note": "'researching' describes what they do all day, not why they wrote",
    },
    "investor intro": {   # L-1326, verbatim
        "notes": "VC here — wanting to intro you to a few portfolio companies. "
                 "Not a direct buyer.",
        "expect": {"investor_intro"}, "route": {"DISQUALIFY"},
    },
    "spam": {   # L-1029, verbatim
        "notes": "You have WON $1,000,000!!! Click here to claim.",
        "expect": {"spam"}, "route": {"DISQUALIFY"},
    },
    "job seeker": {
        "notes": "I'm a senior automation engineer looking for my next role. Are you "
                 "hiring? Happy to share my portfolio.",
        "expect": {"job_seeking"}, "route": {"DISQUALIFY"},
    },
    "academic": {
        "notes": "I'm a final-year student writing my dissertation on marketing "
                 "automation and would like to ask about your process.",
        "expect": {"academic_research"}, "route": {"DISQUALIFY"},
    },
    "journalist": {
        "notes": "I'm writing a piece for a trade publication on agency automation "
                 "and would love a comment on your pricing model.",
        "expect": {"media_enquiry"}, "route": {"DISQUALIFY"},
    },
    "vendor pitch": {
        "notes": "We build AI automation tooling and would love to pitch our platform "
                 "to you. Can we book a call?",
        "expect": {"vendor_pitch"}, "route": {"DISQUALIFY"},
    },
    "free resource": {
        "notes": "Do you have a free template or checklist for lead routing? Not "
                 "looking to buy anything, just after the resource.",
        "expect": {"free_resource_only"}, "route": {"DISQUALIFY"},
    },
    "dual intent": {
        "notes": "We run an agency like yours. We want to buy this for our own "
                 "delivery team - budget approved, starting this month - and we will "
                 "also use it to benchmark our own pricing.",
        # Either reading is defensible; what must not happen is a silent
        # DISQUALIFY of a lead that says it wants to buy.
        "expect": set(schema.NON_BUYER_SIGNALS),
        "route": {"REVIEW", "CONTACT_NOW", "NURTURE"},
        "note": "genuine dual intent must reach a person, not the bin",
    },
}

PURPOSE_LEAD = {
    "lead_id": "LIVE", "name_clean": "Probe", "company_clean": "Probe Co",
    "title_clean": "Owner", "source_clean": "webform",
    "employees_low": None, "employees_high": None,
    "employees_status": "missing", "employees_display": "unknown",
    "budget_low": None, "budget_high": None,
    "budget_status": "missing", "budget_display": "unknown",
}


@pytest.fixture(scope="module")
def purpose_results():
    try:
        provider = providers.get_provider()
    except providers.ProviderConfigError as exc:
        pytest.skip(f"provider not configured: {exc}")
    if hasattr(provider, "is_configured") and not provider.is_configured():
        pytest.skip(f"no API key configured for provider {provider.name!r}")

    leads = [dict(PURPOSE_LEAD, lead_id=f"LIVE-{i}", notes_clean=case["notes"])
             for i, case in enumerate(PURPOSE_CASES.values())]
    results = llm_extraction.extract_batch(
        leads, provider, cache=llm_extraction.ExtractionCache(path=None),
        profile=LIVE_PROFILE, max_workers=3,
    )
    return {label: (lead, result)
            for label, lead, result in zip(PURPOSE_CASES, leads, results)}


@pytest.mark.parametrize("label", list(PURPOSE_CASES))
def test_the_stated_purpose_is_read_correctly(label, purpose_results):
    case = PURPOSE_CASES[label]
    _, result = purpose_results[label]
    assert result.status != "failed", result.error
    signal = result.extraction["non_buyer_signal"]
    assert signal in case["expect"], (
        f"{label} ({case.get('note', '')}): expected {case['expect']}, got {signal!r}"
    )


@pytest.mark.parametrize("label", [l for l, c in PURPOSE_CASES.items() if "buyer" in c])
def test_the_buyer_gate_is_unchanged_by_the_new_field(label, purpose_results):
    case = PURPOSE_CASES[label]
    actual = purpose_results[label][1].extraction["is_potential_buyer"]
    assert actual in case["buyer"], f"{label}: expected {case['buyer']}, got {actual!r}"


@pytest.mark.parametrize("label", list(PURPOSE_CASES))
def test_an_excluded_lead_quotes_the_words_that_excluded_it(label, purpose_results):
    extraction = purpose_results[label][1].extraction
    if extraction["non_buyer_signal"] != "none":
        assert extraction["disqualification_evidence"], (
            f"{label}: excluded with nothing quoted from the notes"
        )


@pytest.mark.parametrize("label", list(PURPOSE_CASES))
def test_the_whole_pipeline_routes_the_stated_purpose(label, purpose_results):
    """Extraction through scoring to a route - the gate is only useful if the
    route at the end of it is the right one."""
    from recommendation import route_lead
    from scoring import score_lead

    case = PURPOSE_CASES[label]
    lead, result = purpose_results[label]
    scores = score_lead(result.extraction, lead, LIVE_PROFILE)
    routed = route_lead(result.extraction, scores, lead)
    assert routed["recommendation"] in case["route"], (
        f"{label} ({case.get('note', '')}): expected {case['route']}, got "
        f"{routed['recommendation']!r} "
        f"[buyer={result.extraction['is_potential_buyer']!r}, "
        f"signal={result.extraction['non_buyer_signal']!r}, "
        f"total={scores['total_score']}]"
    )


@pytest.mark.parametrize("label", list(PURPOSE_CASES))
def test_no_genuine_buyer_is_disqualified_on_real_notes(label, purpose_results):
    """The end of the chain: whatever the model read, a lead it calls a buyer
    with no stated non-buying purpose is never routed out of the pipeline."""
    from recommendation import route_lead
    from scoring import score_lead

    lead, result = purpose_results[label]
    extraction = result.extraction
    if extraction["non_buyer_signal"] != "none" or extraction["is_potential_buyer"] != "yes":
        pytest.skip("gated lead - covered by the exclusion tests")
    scores = score_lead(extraction, lead, LIVE_PROFILE)
    routed = route_lead(extraction, scores, lead)
    assert routed["recommendation"] != "DISQUALIFY", (
        f"{label}: genuine buyer disqualified at total {scores['total_score']}"
    )


def test_a_weak_but_genuine_enquiry_is_not_excluded(purpose_results):
    """Guards the failure mode this change could introduce: weakness read as a
    stated purpose. No budget, no timeline, still exploring - and still 'none'."""
    try:
        provider = providers.get_provider()
    except providers.ProviderConfigError as exc:
        pytest.skip(f"provider not configured: {exc}")

    lead = dict(PURPOSE_LEAD, notes_clean=(
        "Small agency, 6 people. Curious about what automation could do for us one "
        "day. No budget set aside and no timeline yet."
    ))
    result = llm_extraction.extract_batch(
        [lead], provider, cache=llm_extraction.ExtractionCache(path=None),
        profile=LIVE_PROFILE, max_workers=1,
    )[0]
    assert result.extraction["non_buyer_signal"] == "none"
    assert result.extraction["is_potential_buyer"] == "yes"


def test_no_scoring_vocabulary_appears_in_any_extraction(extractions):
    """Stage 4 output must contain no score, band, rank or recommendation."""
    import json as _json

    for lead_id, result in extractions.items():
        blob = _json.dumps(result.extraction).lower()
        for banned in ("contact_now", "nurture", "disqualify", "score", "points", "rank"):
            assert banned not in blob, f"{lead_id}: scoring vocabulary {banned!r} in output"
