"""Stage 7 flow tests - upload through to rendered detail, without a browser.

`run_pipeline` is a plain function over a DataFrame, so the whole
upload -> process -> summary -> queue -> select -> detail path is exercisable
here. The extraction step is stubbed with a fake provider: these tests are
about the interface's handling of results, not about the model.
"""

import json
import os
import re
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ui  # noqa: E402
from llm_extraction import ExtractionCache  # noqa: E402
from providers import ProviderResponse  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

RAW_ROWS = [
    # CONTACT_NOW - the buyer template
    dict(lead_id="L-1009", created="2024-06-02", name="Kemi A.", email="kemi@northforge.io",
         company="Northforge", employees="", website="northforge.io", title="Managing Partner",
         source="event", monthly_budget="10,000",
         notes="We're a recruitment marketing agency, 40 people. Manual lead routing so hot "
               "leads go stale is eating our week. Want it automated end to end. Budget "
               "approved, wants to move in 2 weeks. I make the call here."),
    # NURTURE
    dict(lead_id="L-1362", created="06/25/2024", name="Gbenga", email="g@performengine.agency",
         company="PerformEngine", employees="20", website="performengine.agency",
         title="Head of Ops", source="linkedin", monthly_budget="5,000/mo",
         notes="Curious about automating pacing ad budgets. Comparing a few options."),
    # NURTURE - a weak prospect, not a non-buyer
    dict(lead_id="L-1033", created="2024-06-08", name="Femi", email="femi@earlystage.co",
         company="EarlyStageCo", employees="1", website="", title="Consultant",
         source="linkedin", monthly_budget="0",
         notes="solo consultant wanting to automate my own outreach. tiny budget, one-man shop."),
    # DISQUALIFY - a stated non-buying purpose, which no score can outweigh
    dict(lead_id="L-1144", created="2024-06-09", name="Tunde", email="t@cladwellsend.io",
         company="CladwellSend", employees="30", website="cladwellsend.io", title="Founder",
         source="event", monthly_budget="9,000",
         notes="We do similar work — curious about your pricing for benchmarking."),
    # REVIEW
    dict(lead_id="L-1261", created="Jun 7 2024", name="Grace W.", email="grace@cladwell.ng",
         company="CladwellWorks", employees="", website="", title="Owner",
         source="referral", monthly_budget="",
         notes="Fellow agency owner here, mostly researching the market."),
    # excluded pre-scoring
    dict(lead_id="TESTROW", created="2024-06-06", name="Test User", email="test@test.com",
         company="Test", employees="", website="test.com", title="test", source="test",
         monthly_budget="", notes="QA test entry, please ignore."),
]


def raw_frame() -> pd.DataFrame:
    return pd.DataFrame(RAW_ROWS).astype(str)


def measure(stated, low=None, high=None, kind="missing", ev="not specified"):
    return {"stated": stated, "low": low, "high": high, "kind": kind, "evidence": ev}


def extraction_for(lead_id: str) -> dict:
    common = dict(non_buyer_signal="none",
                  disqualification_evidence=None, ambiguity_reason=None,
                  budget_in_notes=measure(False), price_sensitive_flag=False,
                  location_in_notes={'stated': False, 'value': None,
                                     'evidence': 'not specified'})
    if lead_id == "L-1009":
        return {**common, "is_potential_buyer": "yes", "buyer_type": "business_prospect",
                "industry_fit": {"level": "high", "evidence": "recruitment marketing agency"},
                "employee_count_in_notes": measure(True, 40, 40, "exact", "40 people"),
                "seniority": {"level": "decision_maker", "evidence": "I make the call here"},
                "urgency": {"level": "high", "evidence": "move in 2 weeks"},
                "pain_severity": {"level": "high", "evidence": "eating our week"},
                "purchasing_readiness": {"level": "approved", "evidence": "Budget approved"},
                "timeline": {"level": "near_term", "evidence": "2 weeks"},
                "buying_stage": {"level": "committed", "evidence": "automated end to end"}}
    if lead_id == "L-1362":
        return {**common, "is_potential_buyer": "yes", "buyer_type": "business_prospect",
                "industry_fit": {"level": "high", "evidence": "ad budget pacing"},
                # conflicts with the CSV's employees=20 -> exercises the conflict flag
                "employee_count_in_notes": measure(True, 34, 34, "exact", "34 people"),
                "seniority": {"level": "decision_maker", "evidence": "Head of Ops"},
                "urgency": {"level": "medium", "evidence": "comparing a few options"},
                "pain_severity": {"level": "medium", "evidence": "pacing ad budgets"},
                "purchasing_readiness": {"level": "unknown", "evidence": "not specified"},
                "timeline": {"level": "none", "evidence": "not specified"},
                "buying_stage": {"level": "evaluating", "evidence": "comparing a few options"}}
    if lead_id == "L-1033":
        return {**common, "is_potential_buyer": "yes", "buyer_type": "freelancer_solo",
                "industry_fit": {"level": "low", "evidence": "my own outreach"},
                "employee_count_in_notes": measure(True, 1, 1, "exact", "one-man shop"),
                "budget_in_notes": measure(True, 0, 0, "exact", "tiny budget"),
                "seniority": {"level": "decision_maker", "evidence": "solo consultant"},
                "urgency": {"level": "low", "evidence": "not specified"},
                "pain_severity": {"level": "medium", "evidence": "automate my own outreach"},
                "purchasing_readiness": {"level": "unknown", "evidence": "tiny budget"},
                "timeline": {"level": "none", "evidence": "not specified"},
                "buying_stage": {"level": "exploring", "evidence": "wanting to automate"},
                "price_sensitive_flag": True}
    if lead_id == "L-1144":
        # The only route to DISQUALIFY is now the gate, so the fixture needs a
        # lead that states a non-buying purpose. Its score is deliberately
        # respectable: it proves the gate outranks the number.
        return {**common, "is_potential_buyer": "no", "buyer_type": "competitor",
                "non_buyer_signal": "competitive_research",
                "disqualification_evidence": "comparing your pricing for benchmarking",
                "industry_fit": {"level": "high", "evidence": "We do similar work"},
                "employee_count_in_notes": measure(True, 30, 30, "exact", "30 people"),
                "seniority": {"level": "decision_maker", "evidence": "Founder"},
                "urgency": {"level": "low", "evidence": "not specified"},
                "pain_severity": {"level": "low", "evidence": "not specified"},
                "purchasing_readiness": {"level": "unknown", "evidence": "not specified"},
                "timeline": {"level": "none", "evidence": "not specified"},
                "buying_stage": {"level": "exploring", "evidence": "comparing your pricing"}}
    return {**common, "is_potential_buyer": "ambiguous", "buyer_type": "competitor",
            "ambiguity_reason": "agency owner researching the market, intent unclear",
            "industry_fit": {"level": "high", "evidence": "Fellow agency owner"},
            "employee_count_in_notes": measure(False),
            "seniority": {"level": "decision_maker", "evidence": "agency owner"},
            "urgency": {"level": "low", "evidence": "not specified"},
            "pain_severity": {"level": "low", "evidence": "not specified"},
            "purchasing_readiness": {"level": "unknown", "evidence": "not specified"},
            "timeline": {"level": "none", "evidence": "not specified"},
            "buying_stage": {"level": "exploring", "evidence": "researching the market"}}


class ScriptedProvider:
    """Returns the canned extraction for whichever lead is in the message."""

    name, model = "test-provider", "test-model"

    def extract(self, system_prompt, user_message, json_schema):
        for lead_id in ("L-1009", "L-1362", "L-1033", "L-1144", "L-1261"):
            marker = {"L-1009": "recruitment marketing", "L-1362": "pacing ad budgets",
                      "L-1033": "one-man shop", "L-1144": "for benchmarking",
                      "L-1261": "Fellow agency owner"}[lead_id]
            if marker.lower() in user_message.lower():
                return ProviderResponse(text=json.dumps(extraction_for(lead_id)))
        raise AssertionError(f"unexpected lead in prompt: {user_message[:120]}")


@pytest.fixture(scope="module")
def results():
    import pipeline

    return pipeline.run_pipeline(raw_frame(), provider=ScriptedProvider(),
                            cache=ExtractionCache(path=None), rpm=0)


@pytest.fixture(scope="module")
def by_id(results):
    return {r["lead_id"]: r for r in results["rows"]}


# --------------------------------------------------------------------------- #
# Upload -> process
# --------------------------------------------------------------------------- #

def test_uploaded_csv_runs_end_to_end(results):
    assert results["summary"]["total_processed"] == 5    # 6 rows, 1 excluded
    assert results["excluded"] == 1
    assert results["failed"] == 0


def test_every_route_is_represented(results):
    routes = {r["recommendation"] for r in results["rows"]}
    assert routes == {"CONTACT_NOW", "NURTURE", "DISQUALIFY", "REVIEW"}


def test_summary_counts_match_the_rows(results):
    s = results["summary"]
    assert s["total_processed"] == sum(
        s[r] for r in ("CONTACT_NOW", "NURTURE", "DISQUALIFY", "REVIEW")
    )


def test_pipeline_is_not_hardcoded_to_demonstration_leads():
    """A different file must produce different leads - nothing is baked in."""
    import pipeline

    other = pd.DataFrame([dict(
        lead_id="X-1", created="2024-06-01", name="Ada", email="ada@acme.io", company="Acme",
        employees="30", website="acme.io", title="Founder", source="event",
        monthly_budget="9k", notes="We're a growth agency, 30 people. Budget approved, ASAP.",
    )]).astype(str)

    class Always:
        name, model = "t", "t"

        def extract(self, *a):
            return ProviderResponse(text=json.dumps(extraction_for("L-1009")))

    out = pipeline.run_pipeline(other, provider=Always(),
                                cache=ExtractionCache(path=None), rpm=0)
    assert [r["lead_id"] for r in out["rows"]] == ["X-1"]
    assert out["rows"][0]["company"] == "Acme"


def test_sample_limit_processes_only_that_many():
    import pipeline

    class Always:
        name, model = "t", "t"

        def extract(self, *a):
            return ProviderResponse(text=json.dumps(extraction_for("L-1009")))

    out = pipeline.run_pipeline(raw_frame(), limit=2, provider=Always(),
                                cache=ExtractionCache(path=None), rpm=0)
    assert out["summary"]["total_processed"] == 2


def test_progress_callback_reports_completion():
    import pipeline

    seen = []

    class Always:
        name, model = "t", "t"

        def extract(self, *a):
            return ProviderResponse(text=json.dumps(extraction_for("L-1009")))

    pipeline.run_pipeline(raw_frame(), provider=Always(), cache=ExtractionCache(path=None),
                          rpm=0, progress=lambda d, t: seen.append((d, t)))
    assert seen and seen[-1][0] == seen[-1][1]


# --------------------------------------------------------------------------- #
# Queue
# --------------------------------------------------------------------------- #

def test_every_lead_is_ranked_including_review(results):
    rows = results["rows"]
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    assert all(r["rank"] is not None for r in rows)
    # Ranked, but a REVIEW lead still publishes no score.
    for row in rows:
        if row["recommendation"] == "REVIEW":
            assert row["total_score"] is None
            assert row["scores_suppressed"] is True


def test_the_queue_is_ordered_by_route_then_score(results):
    """The pipeline's own output, not just `rank_leads` in isolation."""
    from recommendation import ROUTE_PRIORITY

    rows = results["rows"]
    priorities = [ROUTE_PRIORITY[r["recommendation"]] for r in rows]
    assert priorities == sorted(priorities), [r["recommendation"] for r in rows]

    for route in ("CONTACT_NOW", "NURTURE", "DISQUALIFY"):
        totals = [r["total_score"] for r in rows if r["recommendation"] == route]
        assert totals == sorted(totals, reverse=True), route


def test_queue_table_renders_every_route_marker(results):
    html = ui.queue_table(results["rows"], selected_id=None)
    for mod in ("k-mk contact", "k-mk nurture", "k-mk disq", "k-mk review"):
        assert mod in html


def test_review_row_shows_dashes_not_zeroes(results, by_id):
    html = ui.queue_table([by_id["L-1261"]], selected_id=None)
    assert "&mdash;" in html
    assert ">0<" not in html


def test_selected_row_is_marked(results, by_id):
    html = ui.queue_table(results["rows"], selected_id="L-1009")
    assert '<tr class="sel" data-lead="L-1009"' in html
    assert html.count('class="sel"') == 1


def test_every_row_carries_the_id_that_makes_it_clickable(results):
    """The click handler reads `data-lead` off the row; without it a row is
    inert and the queue can only be driven from the dropdown."""
    html = ui.queue_table(results["rows"], selected_id=None)
    for row in results["rows"]:
        assert f'data-lead="{row["lead_id"]}"' in html
    assert html.count("data-lead=") == len(results["rows"])


def test_a_lead_id_cannot_break_out_of_the_click_attribute(by_id):
    """Lead ids come from an uploaded file, and land in an HTML attribute."""
    hostile = dict(by_id["L-1009"], lead_id='x" onclick="alert(1)')
    html = ui.queue_table([hostile], selected_id=None)
    assert 'onclick="alert(1)"' not in html
    assert "&quot;" in html


# --------------------------------------------------------------------------- #
# Lead detail - scored routes
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("lead_id,route", [
    ("L-1009", "CONTACT_NOW"), ("L-1362", "NURTURE"), ("L-1033", "NURTURE"),
    ("L-1144", "DISQUALIFY"),
])
def test_scored_leads_keep_scores_and_points(lead_id, route, by_id):
    row = by_id[lead_id]
    assert row["recommendation"] == route
    assert row["scores_suppressed"] is False
    assert isinstance(row["total_score"], int)
    html = ui.factor_table(row["breakdown"], row["company_fit_score"],
                           row["buying_intent_score"], show_points=True)
    assert 'class="fp"' in html
    assert "Company fit &mdash;" in html


@pytest.mark.parametrize("lead_id", ["L-1009", "L-1362", "L-1033"])
def test_explanation_is_multi_paragraph(lead_id, by_id):
    parts = by_id[lead_id]["explanation_parts"]
    assert len(parts) >= 2
    assert parts[0].startswith("Total ")
    html = ui.explanation_block(parts)
    assert html.count("<p") == len(parts)
    assert 'class="verdict"' in html


def test_conflict_is_surfaced_with_both_values(by_id):
    """L-1362: notes say 34, the uploaded file says 20."""
    row = by_id["L-1362"]
    entry = row["breakdown"]["company_size_signal"]
    assert entry["conflict"] is True
    html = ui.conflict_flag(row["breakdown"])
    assert "34" in html and "20" in html
    assert "Original lead data" in html


# --------------------------------------------------------------------------- #
# Lead detail - REVIEW
# --------------------------------------------------------------------------- #

def test_review_lead_has_no_published_scores(by_id):
    row = by_id["L-1261"]
    assert row["recommendation"] == "REVIEW"
    assert row["scores_suppressed"] is True
    assert row["company_fit_score"] is None
    assert row["buying_intent_score"] is None
    assert row["total_score"] is None


def test_review_signal_table_hides_points_entirely(by_id):
    row = by_id["L-1261"]
    html = ui.factor_table(row["breakdown"], None, None, show_points=False)
    assert 'class="fp"' not in html
    assert "Company signals" in html
    assert "/ 50" not in html


def test_review_lead_carries_its_reason(by_id):
    assert "researching the market" in by_id["L-1261"]["review_reason"]


def test_review_lead_produces_no_explanation_parts(by_id):
    """No total to state, so no verdict line is generated at all."""
    assert by_id["L-1261"]["explanation_parts"] == []


# --------------------------------------------------------------------------- #
# Copy discipline
# --------------------------------------------------------------------------- #

def test_interface_never_says_crm(results):
    """"CRM" is internal vocabulary and means nothing to an evaluator.

    Scans rendered copy only. The stylesheet is excluded deliberately: it
    carries base64 font payloads whose random alphabet contains "crm" by
    coincidence, which would make this a permanently red test about nothing.
    """
    surfaces = [ui.hero(), ui.masthead("<span>NO FILE LOADED</span>")]
    for row in results["rows"]:
        surfaces.append(ui.conflict_flag(row["breakdown"]))
        surfaces.append(ui.lead_detail(row))
        surfaces.append(" ".join(row["explanation_parts"]))
    assert "crm" not in " ".join(surfaces).lower()


def test_conflict_wording_names_the_uploaded_file_not_a_system(by_id):
    html = ui.conflict_flag(by_id["L-1362"]["breakdown"])
    assert "Original lead data" in html
    parts = by_id["L-1362"]["explanation_parts"]
    assert any("original lead data" in p.lower() for p in parts)


def test_interface_exposes_no_implementation_details():
    """The reader is evaluating leads, not the stack that scored them."""
    source = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "app.py"), encoding="utf-8").read()
    # Strip comments and docstrings; only rendered strings matter.
    import ast

    tree = ast.parse(source)
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docs.add(id(body[0].value))
    literals = " ".join(
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs
    ).lower()
    for banned in ("streamlit", "gemini", "anthropic", "python", "flash-lite", "provider",
                   "llm", "api key", "pipeline internals"):
        assert banned not in literals, f"{banned!r} is visible in the interface"


def test_hero_and_masthead_name_the_product_not_the_stack():
    combined = (ui.hero() + ui.masthead("<span>NO FILE LOADED</span>")).lower()
    for banned in ("streamlit", "gemini", "model", "provider", "python"):
        assert banned not in combined


def test_lead_data_is_escaped_into_the_markup():
    """Lead fields are user-supplied and are rendered as raw HTML."""
    row = {"lead_id": "X<script>", "company": "A&B", "name": None, "rank": 1,
           "source": "event", "title": "<b>", "company_fit_score": 1,
           "buying_intent_score": 2, "total_score": 3, "recommendation": "NURTURE"}
    html = ui.queue_table([row], selected_id=None)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "A&amp;B" in html


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #

def test_export_includes_every_lead_and_its_evidence(results):
    import pipeline

    csv = pipeline.export_csv(results["rows"]).decode("utf-8")
    for lead_id in ("L-1009", "L-1362", "L-1033", "L-1261"):
        assert lead_id in csv
    assert "industry_fit_evidence" in csv
    assert "explanation" in csv


def test_export_omits_points_for_review_leads(results, by_id):
    import pipeline

    csv = pipeline.export_csv([by_id["L-1261"]]).decode("utf-8")
    header = csv.splitlines()[0]
    assert "industry_fit_level" in header
    assert "industry_fit_points" not in header


# --------------------------------------------------------------------------- #
# Explaining an exclusion without naming the machinery
# --------------------------------------------------------------------------- #

def excluded_lead(signal="competitive_research", **overrides):
    row = {
        "lead_id": "L-77", "name": "Sam", "company": "Peerworks",
        "recommendation": "DISQUALIFY", "scores_suppressed": False,
        "non_buyer_signal": signal,
        "disqualification_evidence": "benchmarking our own offering",
        "company_fit_score": 30, "buying_intent_score": 26, "total_score": 56,
        "breakdown": {}, "explanation_parts": ["Total 56/100 → DISQUALIFY."],
        "notes": "We do similar work and are comparing your pricing for benchmarking "
                 "our own offering.",
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize("signal", [s for s in __import__("schema").NON_BUYER_SIGNALS
                                    if s != "none"])
def test_every_exclusion_reads_as_a_sentence_not_a_token(signal):
    text = ui.exclusion_reason(excluded_lead(signal))
    assert text.startswith("Disqualified because the lead stated that ")
    assert signal not in text
    assert "_" not in text


def test_the_exclusion_sentence_names_what_the_lead_said():
    lead = excluded_lead()
    assert ui.exclusion_reason(lead) == (
        "Disqualified because the lead stated that they were comparing our pricing "
        "to benchmark their own offering."
    )


def test_a_lead_with_no_stated_purpose_gets_no_exclusion_sentence():
    assert ui.exclusion_reason(excluded_lead(signal="none")) == ""
    assert ui.exclusion_reason({}) == ""


def test_a_held_contradiction_is_worded_as_undecided_not_disqualified():
    text = ui.exclusion_reason(excluded_lead(
        recommendation="REVIEW", scores_suppressed=True,
    ))
    assert "Disqualified" not in text
    assert "held for a decision" in text


def test_the_detail_panel_shows_the_reason_and_the_lead_s_own_words():
    html = ui.lead_detail(excluded_lead())
    assert "Basis for this decision" in html
    assert "comparing our pricing to benchmark their own offering" in html
    assert "benchmarking our own offering" in html          # the quote
    assert "competitive_research" not in html
    assert "non_buyer_signal" not in html


def test_the_downloaded_report_shows_the_reason_not_the_token():
    text = ui.lead_report(excluded_lead()).decode("utf-8")
    # The report wraps to 74 columns, so the sentence is checked as a reader
    # meets it rather than as one unbroken string.
    flowed = " ".join(text.split())
    assert "BASIS FOR THIS DECISION" in text
    assert "comparing our pricing to benchmark their own offering" in flowed
    assert "competitive_research" not in text
    assert "non_buyer_signal" not in text


def test_the_queue_export_does_not_publish_the_internal_token():
    """The CSV is read by people too."""
    import pipeline

    csv = pipeline.export_csv([excluded_lead()]).decode("utf-8")
    assert "competitive_research" not in csv
    assert "non_buyer_signal" not in csv


# --------------------------------------------------------------------------- #
# Single-lead detail report
# --------------------------------------------------------------------------- #

def report_for(row, profile=None) -> str:
    return ui.lead_report(row, profile).decode("utf-8")


def test_report_covers_the_selected_lead_score_route_and_evidence(by_id):
    text = report_for(by_id["L-1009"])
    assert "L-1009" in text and "Northforge" in text
    assert "Contact now" in text                      # route
    assert "/ 100" in text                            # total
    assert "Industry fit" in text and "Scoring factors".upper() in text
    assert "eating our week" in text                  # evidence, verbatim
    assert by_id["L-1009"]["notes"][:30] in text      # original notes


def test_report_contains_only_the_selected_lead(by_id):
    text = report_for(by_id["L-1009"])
    for other in ("L-1362", "L-1033", "L-1261", "PerformEngine", "CladwellWorks"):
        assert other not in text


def test_report_records_the_target_profile_it_was_assessed_against(by_id):
    from target_profile import TargetProfile

    profile = TargetProfile.create("Healthcare", 20_000, 100_000, "Nigeria")
    text = report_for(by_id["L-1009"], profile)
    assert "TARGET PROFILE" in text
    assert "Healthcare" in text
    assert profile.budget_label() in text
    assert "Nigeria" in text


def test_report_without_a_profile_omits_the_section_rather_than_inventing_one(by_id):
    text = report_for(by_id["L-1009"])
    assert "TARGET PROFILE" not in text


def test_report_withholds_the_score_of_a_review_lead(by_id):
    """The same suppression the screen applies: a review lead has no score, and
    printing its per-factor points would let a reader add one back."""
    text = report_for(by_id["L-1261"])
    assert "/ 100" not in text
    assert "withheld pending review" in text
    assert "pts" not in text
    assert "SIGNALS OBSERVED" in text


def test_report_never_carries_internal_or_technical_detail(by_id):
    for row in by_id.values():
        text = report_for(row).lower()
        for banned in ("traceback", "exception", "http", "api", "token", "quota",
                       "gemini", "claude", "anthropic", "streamlit", "python",
                       "crm", "failure_code", "prompt", "model"):
            assert banned not in text, f"{row['lead_id']}: {banned!r} reached the report"


def test_report_of_an_unassessable_lead_says_so_without_the_technical_reason():
    import pipeline

    row = {
        "lead_id": "L-9", "company": "Acme", "name": "Ada", "recommendation": "REVIEW",
        "scores_suppressed": True, "assessment_failed": True, "failure_code": "rate_limited",
        "review_reason": pipeline.ASSESSMENT_FAILED_REVIEW_REASON, "breakdown": {},
        "notes": "wants a quote",
    }
    text = report_for(row)
    assert "manual review" in text
    assert "rate_limited" not in text
    assert "SCORING FACTORS" not in text     # nothing was assessed to show


def test_report_filename_identifies_the_lead_and_nothing_else():
    assert ui.report_filename({"lead_id": "L-1009"}) == "lead-detail-L-1009.txt"
    assert ui.report_filename({"lead_id": "../../etc/passwd"}) == "lead-detail-------etc-passwd.txt"
    assert ui.report_filename({}) == "lead-detail-lead.txt"


def test_report_is_plain_readable_text_not_markup(by_id):
    text = report_for(by_id["L-1009"])
    assert "<" not in text and "&mdash;" not in text
    assert max(len(line) for line in text.splitlines()) <= 90


def test_the_widest_report_label_still_has_a_gap_after_it(by_id):
    """"Purchasing readiness" is exactly as wide as the label column was, so it
    ran straight into the level printed beside it."""
    text = report_for(by_id["L-1009"])
    line = next(l for l in text.splitlines() if "Purchasing readiness" in l)
    assert line.strip().startswith("Purchasing readiness  ")


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #

def test_stylesheet_embeds_the_three_typefaces():
    css = ui.stylesheet()
    for family in ("Geist Pixel", "Geist Mono", "'Geist'"):
        assert family in css
    assert css.count("@font-face") == 3
    assert "data:font/woff2;base64," in css


def test_visual_system_tokens_are_the_approved_ones():
    """The approved dark palette, asserted by value so a stray edit is caught."""
    css = ui.stylesheet()
    for token in (
        "#0C0C0B",   # canvas
        "#131312",   # surface / panels
        "#F5F5F3",   # primary text
        "#A8A29E",   # secondary text
        "#6B6660",   # muted text
        "#2A2A27",   # borders / rules
        "#FF5A16",   # accent
    ):
        assert token in css, f"{token} missing from the stylesheet"


def test_no_light_theme_tokens_remain():
    """Guards against a half-finished conversion leaving light values behind."""
    css = ui.stylesheet()
    body = css.split("/* ---- masthead ---- */")[0]
    for stale in ("#FFFFFF;", "#FAFAF9", "#0A0A0A", "#E64500", "#E4E1DC", "#57534E", "#CFCAC3"):
        assert stale not in body, f"light-theme value {stale} still in the token block"


def test_config_and_stylesheet_agree_on_the_palette():
    """config.toml themes the host's own widgets; ui.py themes ours. If they
    drift, the two halves of the page stop matching."""
    import tomllib
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    config = tomllib.loads((root / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    theme = config["theme"]
    assert theme["base"] == "dark"
    assert theme["backgroundColor"].upper() == "#0C0C0B"
    assert theme["secondaryBackgroundColor"].upper() == "#131312"
    assert theme["textColor"].upper() == "#F5F5F3"
    assert theme["primaryColor"].upper() == "#FF5A16"
    assert theme["borderColor"].upper() == "#2A2A27"


def test_no_new_hues_were_introduced():
    """Strictly a light -> dark conversion: no purple, pink or decorative depth."""
    css = ui.stylesheet().lower()
    for banned in ("purple", "pink", "violet", "magenta", "indigo",
                   "radial-gradient", "text-shadow", "filter:blur", "backdrop-filter"):
        assert banned not in css, f"{banned} appeared in the stylesheet"

    # The single gradient is the half-filled NURTURE marker, painted with --ink.
    assert css.count("linear-gradient") == 1

    # `box-shadow: inset` is used to draw the 2px selected-row rule; a drop
    # shadow would carry a blur radius, which is what this actually bans.
    shadows = re.findall(r"box-shadow:([^;}]+)", css)
    assert shadows, "expected the selected-row inset rule"
    for shadow in shadows:
        assert "inset" in shadow, f"non-inset shadow introduced: {shadow.strip()}"


def test_geometry_stays_square():
    css = ui.stylesheet()
    assert "border-radius:0" in css.replace(" ", "")
    assert "box-shadow:0 4px" not in css.replace(" ", "")


def test_section_numbering_uses_the_display_face():
    assert "var(--pixel)" in ui.stylesheet()
    assert re.search(r'<span class="n">01</span>', ui.section("01", "Source"))


# --------------------------------------------------------------------------- #
# End-to-end through the real app script
# --------------------------------------------------------------------------- #

@pytest.mark.live
def test_full_user_flow_through_the_real_app():
    """Upload -> process -> summary -> queue -> select -> detail.

    Runs app.py itself in Streamlit's own test runtime, so it covers the wiring
    the headless tests above cannot: widget state, the rerun after processing,
    and the fact that progress updates arrive from worker threads. Marked live
    because it assesses real leads.
    """
    from streamlit.testing.v1 import AppTest

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample = os.path.join(root, "sample_data", "demo_upload.csv")
    if not os.path.exists(sample):
        pytest.skip("sample_data/demo_upload.csv not present")

    at = AppTest.from_file(os.path.join(root, "app.py"), default_timeout=300)
    at.run()
    assert not at.exception

    at.file_uploader[0].set_value([("demo_upload.csv", open(sample, "rb").read(), "text/csv")])
    at.run()
    assert not at.exception
    assert [b.label for b in at.button] == ["Process leads"]

    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]

    results = at.session_state["results"]
    summary = results["summary"]
    assert summary["total_processed"] == 5      # 6 rows, 1 excluded pre-scoring
    assert results["excluded"] == 1
    assert summary["total_processed"] == sum(
        summary[r] for r in ("CONTACT_NOW", "NURTURE", "DISQUALIFY", "REVIEW")
    )
    assert {r["recommendation"] for r in results["rows"]} == {
        "CONTACT_NOW", "NURTURE", "DISQUALIFY", "REVIEW",
    }

    # Selecting a lead populates section 04. The lead picker is found by its
    # contents rather than its index: section 02's industry control shifts the
    # widget order, and a positional lookup silently grabs the wrong box.
    lead_select = next(
        sb for sb in at.selectbox
        if any("L-1261" in str(o) for o in sb.options)
    )
    review_option = next(o for o in lead_select.options if "L-1261" in o)
    lead_select.select(review_option).run()
    assert at.session_state["selected"] == "L-1261"

    rendered = " ".join(m.value for m in at.markdown)
    assert "Lead detail" in rendered or "LEAD DETAIL" in rendered.upper()

    # The review lead publishes no routing score at all: no score row, no
    # denominators, no points column - the section is absent, not blanked.
    detail = rendered.split(">04<")[-1]
    assert "/ 100" not in detail
    assert "/ 50" not in detail
    assert 'class="k-sc' not in detail
    assert 'class="fp"' not in detail
    # What a reviewer does get: the reason, the signals, and the raw notes.
    assert "Why this needs a person" in detail
    assert "Signals observed" in detail
    assert "researching the market" in detail


# --------------------------------------------------------------------------- #
# Target industries: the picker, driven through the real app script (offline)
# --------------------------------------------------------------------------- #

def _app_at_the_profile_step():
    """app.py with a file uploaded, so section 02 is on screen. No API calls:
    nothing is processed, only the profile controls are exercised."""
    from streamlit.testing.v1 import AppTest

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample = os.path.join(root, "sample_data", "demo_upload.csv")
    if not os.path.exists(sample):
        pytest.skip("sample_data/demo_upload.csv not present")

    at = AppTest.from_file(os.path.join(root, "app.py"), default_timeout=60)
    at.run()
    at.file_uploader[0].set_value([("demo_upload.csv", open(sample, "rb").read(), "text/csv")])
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def test_the_industry_picker_offers_any_alongside_the_suggestions():
    at = _app_at_the_profile_step()
    options = at.multiselect[0].options
    assert options[0] == "Any"
    assert "Healthcare" in options


def test_a_fresh_session_defaults_to_any_industry():
    """No industry restriction until the operator names one - a specific
    default silently scored every lead against a target nobody chose."""
    at = _app_at_the_profile_step()
    assert at.multiselect[0].value == ["Any"]
    assert at.session_state["target_industries"] == ["Any"]


def test_the_default_profile_places_no_industry_constraint():
    """The default reaches scoring as "no constraint", not as an industry."""
    from target_profile import TargetProfile

    at = _app_at_the_profile_step()
    profile = TargetProfile.create(at.session_state["target_industries"], 5_000, 50_000)
    assert profile.accepts_any_industry is True
    assert profile.industries == ()


def test_industries_can_be_added_and_removed_one_at_a_time():
    at = _app_at_the_profile_step()
    at.multiselect[0].set_value(["Marketing & Advertising", "Healthcare"]).run()
    assert at.session_state["target_industries"] == ["Marketing & Advertising", "Healthcare"]

    # Removing one leaves the other in place - the selection is not rebuilt.
    at.multiselect[0].set_value(["Healthcare"]).run()
    assert at.session_state["target_industries"] == ["Healthcare"]


def test_choosing_any_clears_the_specific_industries():
    at = _app_at_the_profile_step()
    at.multiselect[0].set_value(["Marketing & Advertising", "Healthcare"]).run()
    at.multiselect[0].set_value(["Marketing & Advertising", "Healthcare", "Any"]).run()
    assert at.session_state["target_industries"] == ["Any"]


def test_choosing_a_specific_industry_removes_any():
    at = _app_at_the_profile_step()
    assert at.session_state["target_industries"] == ["Any"]     # the default

    at.multiselect[0].set_value(["Any", "Healthcare"]).run()
    assert at.session_state["target_industries"] == ["Healthcare"]


def test_any_and_a_specific_industry_never_coexist():
    """Whatever order they arrive in, the state ['Any', X] does not survive."""
    at = _app_at_the_profile_step()
    for attempt in (["Any", "Healthcare"], ["Healthcare", "Any"],
                    ["Any", "Healthcare", "Real Estate"]):
        at.multiselect[0].set_value(attempt).run()
        picked = at.session_state["target_industries"]
        assert not ("Any" in picked and len(picked) > 1), f"{attempt} -> {picked}"


def test_the_picker_selection_becomes_the_profile_the_run_uses():
    """The widget is not decoration: what is picked is what gets assessed."""
    at = _app_at_the_profile_step()
    at.multiselect[0].set_value(["Healthcare", "Real Estate"]).run()

    from target_profile import TargetProfile

    built = TargetProfile.create(at.session_state["target_industries"], 5_000, 50_000)
    assert built.industries == ("Healthcare", "Real Estate")


# --------------------------------------------------------------------------- #
# Processing state: locked results, disabled start, working Cancel
# --------------------------------------------------------------------------- #

class _FakeRun:
    """A RunHandle at a chosen point in its life, without a real thread."""

    def __init__(self, done=0, total=10, finished=False):
        self.done, self.total = done, total
        self._finished = finished
        self.result = None
        self.unavailable = self.failed = False
        self.profile = None

    finished = property(lambda self: self._finished)
    fraction = property(lambda self: self.done / self.total if self.total else 0.0)


def _app_mid_run(results, done=3, total=10, **kw):
    """A file uploaded (so section 02 and its controls exist) and a run active."""
    at = _app_at_the_profile_step()
    at.session_state["results"] = results
    at.session_state["run"] = _FakeRun(done=done, total=total, **kw)
    at.run()
    return at


def _app_idle_with_results(results):
    """The same page with no run in flight - the harvested, unlocked state."""
    at = _app_at_the_profile_step()
    at.session_state["results"] = results
    at.run()
    return at


LOCK_MARKER = "st-key-k-results{"      # emitted only while locked


def test_the_start_button_is_disabled_while_a_run_is_active(results):
    at = _app_mid_run(results)
    process = next(b for b in at.button if b.label == "Process leads")
    assert process.disabled is True


def test_a_run_in_flight_offers_no_stop_control(results):
    """Cancellation was removed: a run started is a run completed, and no
    control on the page claims otherwise."""
    at = _app_mid_run(results)
    labels = [b.label.lower() for b in at.button]
    assert not [label for label in labels if "cancel" in label or "stop" in label]


def test_the_results_section_is_locked_while_processing(results):
    at = _app_mid_run(results)
    rendered = " ".join(m.value for m in at.markdown)
    assert LOCK_MARKER in rendered, "no lock styling emitted"
    assert "pointer-events:none" in rendered

    # ...and the controls inside it are disabled too, so keyboard focus cannot
    # reach what the pointer cannot.
    assert all(sb.disabled for sb in at.selectbox if sb.label in ("Route", "Open lead"))
    assert all(d.disabled for d in at.download_button)


@pytest.mark.parametrize("stage", ["idle", "running"])
def test_the_page_renders_each_section_exactly_once(results, stage):
    """One 03 // QUEUE, in order, whether or not a run is in flight."""
    import re

    at = _app_idle_with_results(results) if stage == "idle" else _app_mid_run(results)
    headers = []
    for block in at.markdown:
        headers += re.findall(
            r'<span class="n">(\d+)</span><span class="t">// ([^<]+)', block.value
        )
    numbers = [n for n, _ in headers]
    assert numbers == ["01", "02", "03", "04"], headers
    assert numbers.count("03") == 1


def test_the_results_stay_visible_while_processing(results):
    """Locked, not removed - the previous answer is still the best one until
    the new run replaces it."""
    at = _app_mid_run(results)
    rendered = " ".join(m.value for m in at.markdown)
    assert "k-table" in rendered, "the queue disappeared during processing"
    assert "L-1009" in rendered
    # Processing is announced...
    assert "Processing" in rendered
    assert ui.run_banner(3, 10) in rendered, "the run banner is not on screen"
    # ...and the results are genuinely locked, asserted against the mechanism
    # rather than the copy, so a reworded banner cannot make this pass while
    # the section stays interactive.
    assert LOCK_MARKER in rendered, "results were not locked"
    assert "pointer-events:none" in rendered


def test_the_banner_reports_actual_progress(results):
    at = _app_mid_run(results, done=137, total=509)
    rendered = " ".join(m.value for m in at.markdown)
    assert "137" in rendered and "509" in rendered


def test_completion_unlocks_the_interface(results):
    """Once harvested there is no handle, so nothing is dimmed or disabled."""
    at = _app_idle_with_results(results)     # run is None: the harvested state
    rendered = " ".join(m.value for m in at.markdown)
    assert LOCK_MARKER not in rendered
    assert not [b for b in at.button if "Cancel" in b.label]
    assert next(b for b in at.button if b.label == "Process leads").disabled is False
    assert all(not sb.disabled for sb in at.selectbox if sb.label in ("Route", "Open lead"))
    assert all(not d.disabled for d in at.download_button)


def test_a_first_run_with_no_previous_results_still_shows_progress():
    """Nothing to dim underneath, and nothing invented to fill the space."""
    at = _app_at_the_profile_step()
    at.session_state["run"] = _FakeRun(done=2, total=20)
    at.run()
    rendered = " ".join(m.value for m in at.markdown)
    assert "Processing leads" in rendered
    # `<table class="k-table">` only appears when a queue is actually drawn;
    # the bare class name is in the stylesheet on every page.
    assert '<table class="k-table">' not in rendered, "invented a queue"
    assert not at.exception


# --------------------------------------------------------------------------- #
# Row selection through the real app script (offline)
# --------------------------------------------------------------------------- #

def _app_with_results(results, profile=None):
    """app.py in Streamlit's own runtime, with a finished run already in state.

    Seeded rather than processed, so this covers the queue-to-detail wiring
    without spending an API call on leads the tests above already assessed.
    """
    from streamlit.testing.v1 import AppTest

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    at = AppTest.from_file(os.path.join(root, "app.py"), default_timeout=60)
    at.session_state["results"] = results
    at.session_state["profile"] = profile
    at.run()
    return at


def _click_row(at, lead_id):
    """Deliver a row click the way the browser does.

    The click arrives as a trigger value on the component's event channel;
    AppTest has no helper for that, so the widget state is assembled by hand and
    the script is rerun with it - the same path a real click takes.

    The trigger id is derived from the component's own widget key rather than
    searched for in session state: Streamlit only lists the event channel there
    once a value has been delivered on it, so discovery works on the second
    click and not the first.
    """
    base = next(
        key for key in at.session_state._state._new_widget_state
        if key.endswith("queue_rows")
    )
    trigger_id = f"$$STREAMLIT_INTERNAL_KEY_{base}__events"
    states = at._tree.get_widget_states()
    widget = states.widgets.add()
    widget.id = trigger_id
    widget.json_trigger_value = json.dumps([{"event": "lead", "value": lead_id}])
    return at._run(states)


def test_clicking_a_row_opens_that_lead(results):
    at = _app_with_results(results)
    assert at.session_state["selected"] is None

    at = _click_row(at, "L-1033")
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["selected"] == "L-1033"

    rendered = " ".join(m.value for m in at.markdown)
    assert '<tr class="sel" data-lead="L-1033"' in rendered
    detail = rendered.split('class="k-panel"')[-1]
    assert "EarlyStageCo" in detail and "Northforge" not in detail


def test_the_detail_download_appears_only_once_a_lead_is_selected(results):
    at = _app_with_results(results)
    assert [b.label for b in at.download_button] == ["Download results (CSV)"]

    at = _click_row(at, "L-1009")
    assert [b.label for b in at.download_button] == [
        "Download results (CSV)", "Download detail",
    ]


def test_the_queue_export_is_unchanged_by_the_detail_download(results, by_id):
    """The two downloads are separate: the queue keeps every lead, the detail
    keeps one."""
    import pipeline

    csv = pipeline.export_csv(results["rows"]).decode("utf-8")
    report = ui.lead_report(by_id["L-1009"]).decode("utf-8")
    assert csv.count("\n") >= len(results["rows"])
    assert "L-1261" in csv and "L-1261" not in report
