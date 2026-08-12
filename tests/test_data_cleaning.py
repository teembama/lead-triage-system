"""Stage 2 unit tests.

Table-driven against the value shapes actually present in the assessment CSV
(profiled read-only before implementation), not invented examples. Pure
functions only - no API calls, no network, fast enough to run on every edit.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_cleaning import (  # noqa: E402
    EXCL_EMPTY_ROW,
    EXCL_HEADER_LEAK,
    EXCL_NEWSLETTER,
    EXCL_NO_NOTES,
    EXCL_NOT_A_LEAD,
    EXCL_TEST_ROW,
    clean_dataframe,
    clean_lead_row,
    describe_measure,
    detect_exclusion,
    flag_duplicates,
    normalise_email,
    normalise_source,
    normalise_title,
    parse_budget,
    parse_date,
    parse_employees,
    read_leads_csv,
)

# --------------------------------------------------------------------------- #
# Dates - all six shapes found in `created`
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "raw,expected_iso,expected_status",
    [
        ("06/28/2024", "2024-06-28", "exact"),      # MM/DD/YYYY  (89 rows)
        ("2024-06-08", "2024-06-08", "exact"),      # YYYY-MM-DD  (82 rows)
        ("2024-6-7", "2024-06-07", "exact"),        # unpadded ISO (69 rows)
        ("Jun 7 2024", "2024-06-07", "exact"),      # Mon D YYYY  (89 rows)
        ("19-06-2024", "2024-06-19", "exact"),      # DD-MM-YYYY  (72 rows)
        ("6/1/24", "2024-06-01", "exact"),          # M/D/YY      (91 rows)
        ("", None, "missing"),
        ("lead_id", None, "unparseable"),
        ("not a date", None, "unparseable"),
        ("13/45/2024", None, "unparseable"),        # shape matches, date invalid
    ],
)
def test_parse_date_formats(raw, expected_iso, expected_status):
    result = parse_date(raw)
    assert result.iso == expected_iso
    assert result.status == expected_status


def test_two_digit_year_is_not_swallowed_as_a_full_year():
    """Regression guard for the reason dates are dispatched by regex.

    `datetime.strptime("6/1/24", "%m/%d/%Y")` succeeds and yields year 24. A
    naive try-each-format cascade would therefore never reach the `%y` branch
    and would silently mis-date all 91 rows of this shape.
    """
    assert parse_date("6/1/24").iso == "2024-06-01"


@pytest.mark.parametrize(
    "raw,ambiguous",
    [
        ("19-06-2024", False),   # first component > 12 -> unambiguously day-first
        ("28-06-2024", False),
        ("04-06-2024", True),    # both <= 12 -> flagged for audit
        ("06/28/2024", False),   # slash shape is not affected
    ],
)
def test_day_first_ambiguity_flag(raw, ambiguous):
    assert parse_date(raw).ambiguous is ambiguous


def test_day_first_reading_is_applied():
    # 19-06-2024 read day-first is 19 June; month-first would be invalid.
    assert parse_date("19-06-2024").iso == "2024-06-19"
    # 04-06-2024 is ambiguous but resolved day-first for consistency.
    assert parse_date("04-06-2024").iso == "2024-06-04"


# --------------------------------------------------------------------------- #
# Employees - all five shapes found in the column
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "raw,low,high,status",
    [
        ("20", 20.0, 20.0, "exact"),      # plain int   (231 rows)
        ("35-55", 35.0, 55.0, "range"),   # range       (47 rows)
        ("51+", 51.0, None, "floor"),     # plus        (74 rows)
        ("~61", 61.0, 61.0, "approx"),    # approx      (35 rows)
        ("", None, None, "missing"),      # blank       (131 rows)
        ("asdf", None, None, "unparseable"),
        ("employees", None, None, "unparseable"),
        ("55-35", 35.0, 55.0, "range"),   # reversed range is normalised
        ("1", 1.0, 1.0, "exact"),
    ],
)
def test_parse_employees(raw, low, high, status):
    result = parse_employees(raw)
    assert result.low == low
    assert result.high == high
    assert result.status == status


def test_floor_notation_invents_no_upper_bound():
    """`50+` asserts a lower bound and nothing else.

    Collapsing it to 50 would claim a precision the source never gave, and
    would land the lead inside a 10-70 band it may sit far above.
    """
    result = parse_employees("50+")
    assert result.low == 50.0
    assert result.high is None
    assert result.unbounded_above is True
    assert result.is_uncertain is True


def test_range_is_preserved_not_averaged():
    """`100-200` stays a band; 150 is a number nobody wrote."""
    result = parse_employees("100-200")
    assert (result.low, result.high) == (100.0, 200.0)
    assert result.is_uncertain is True


def test_exact_value_is_not_marked_uncertain():
    result = parse_employees("26")
    assert (result.low, result.high) == (26.0, 26.0)
    assert result.is_uncertain is False
    assert result.unbounded_above is False


def test_approximate_value_is_flagged_but_not_widened():
    """`~61` is soft, but inventing a band around it would also be invention."""
    result = parse_employees("~61")
    assert (result.low, result.high) == (61.0, 61.0)
    assert result.status == "approx"
    assert result.is_uncertain is True


def test_raw_employee_value_is_retained_for_audit():
    assert parse_employees("51+").raw == "51+"
    assert parse_employees(" 35-55 ").raw == "35-55"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("26", "26"),
        ("35-55", "35-55"),
        ("51+", "51+"),
        ("~61", "~61"),
        ("", "unknown"),
        ("asdf", "unknown"),
    ],
)
def test_employee_display_rendering(raw, expected):
    assert describe_measure(parse_employees(raw)) == expected


def test_unknown_headcount_is_never_assumed_small():
    assert parse_employees("").low is None
    assert parse_employees("asdf").low is None
    assert parse_employees("").is_known is False


# --------------------------------------------------------------------------- #
# Budget - every distinct value present in `monthly_budget`
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "raw,low,high,status",
    [
        ("$11k/mo", 11000.0, 11000.0, "exact"),
        ("$14k/mo", 14000.0, 14000.0, "exact"),
        ("$4k/mo", 4000.0, 4000.0, "exact"),
        ("$6k/mo", 6000.0, 6000.0, "exact"),
        ("12k/mo", 12000.0, 12000.0, "exact"),
        ("15k/mo", 15000.0, 15000.0, "exact"),
        ("16k/mo", 16000.0, 16000.0, "exact"),
        ("5,000/mo", 5000.0, 5000.0, "exact"),
        ("$8,000/mo", 8000.0, 8000.0, "exact"),
        ("$8,500", 8500.0, 8500.0, "exact"),
        ("$9,000", 9000.0, 9000.0, "exact"),
        ("$7k", 7000.0, 7000.0, "exact"),
        ("10,000", 10000.0, 10000.0, "exact"),
        ("18k", 18000.0, 18000.0, "exact"),
        ("2000", 2000.0, 2000.0, "exact"),
        ("4000", 4000.0, 4000.0, "exact"),
        ("500", 500.0, 500.0, "exact"),
        ("6k", 6000.0, 6000.0, "exact"),
        ("8k", 8000.0, 8000.0, "exact"),
        ("$6-8k", 6000.0, 8000.0, "range"),
        ("5k-7k", 5000.0, 7000.0, "range"),
        ("8k-12k", 8000.0, 12000.0, "range"),
        ("0", 0.0, 0.0, "zero"),
        ("", None, None, "missing"),
        ("TBD", None, None, "placeholder"),
        ("depends", None, None, "placeholder"),
        ("budget", None, None, "placeholder"),
        ("asdf", None, None, "placeholder"),
    ],
)
def test_parse_budget_every_observed_value(raw, low, high, status):
    result = parse_budget(raw)
    assert result.low == low
    assert result.high == high
    assert result.status == status


def test_bare_left_operand_inherits_k_from_the_right():
    """`$6-8k` means 6k-8k, not 6-8000 - and stays a band."""
    result = parse_budget("$6-8k")
    assert (result.low, result.high) == (6000.0, 8000.0)


def test_budget_range_is_preserved_not_averaged():
    result = parse_budget("8k-12k")
    assert (result.low, result.high) == (8000.0, 12000.0)
    assert result.is_uncertain is True


def test_confirmed_zero_is_distinct_from_missing():
    """A recorded 0 is evidence of no budget; a blank is absence of evidence."""
    assert parse_budget("0")[:3] == (0.0, 0.0, "zero")
    assert parse_budget("")[:3] == (None, None, "missing")
    assert parse_budget("TBD")[:3] == (None, None, "placeholder")


def test_raw_budget_value_is_retained_for_audit():
    assert parse_budget("$6-8k").raw == "$6-8k"


# --------------------------------------------------------------------------- #
# Email / source / title
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "raw,address,is_valid,repaired",
    [
        ("gbenga@luxauto.io", "gbenga@luxauto.io", True, False),
        ("GBENGA@LuxAuto.io", "gbenga@luxauto.io", True, False),
        ("  ola@pipegtm.co  ", "ola@pipegtm.co", True, False),
        ("kunle[at]meridianbound.com", "kunle@meridianbound.com", True, True),
        ("sophie[at]omniside.agency", "sophie@omniside.agency", True, True),
        ("weird-email-no-domain", "weird-email-no-domain", False, False),
        ("email", "email", False, False),
        ("", None, False, False),
    ],
)
def test_normalise_email(raw, address, is_valid, repaired):
    result = normalise_email(raw)
    assert result.address == address
    assert result.is_valid is is_valid
    assert result.was_repaired is repaired


def test_obfuscated_addresses_are_repaired_not_discarded():
    """11 real prospects use `name[at]domain.com`; they must survive cleaning."""
    result = normalise_email("kunle[at]meridianbound.com")
    assert result.address == "kunle@meridianbound.com"
    assert result.is_valid


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("webform", "webform"),
        ("referral", "referral"),
        ("linkedin", "linkedin"),
        ("event", "event"),
        ("cold reply", "cold_reply"),
        ("Cold Reply", "cold_reply"),
        ("  referral ", "referral"),
        ("test", "unknown"),
        ("source", "unknown"),
        ("", "unknown"),
    ],
)
def test_normalise_source(raw, expected):
    assert normalise_source(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("CEO", "CEO"),
        ("COO", "COO"),
        ("VP Growth", "VP Growth"),
        ("  Head of Ops  ", "Head of Ops"),
        ("asdf", None),
        ("test", None),
        ("title", None),
        ("", None),
    ],
)
def test_normalise_title(raw, expected):
    assert normalise_title(raw) == expected


def test_acronym_titles_are_not_mangled_by_title_casing():
    assert normalise_title("CEO") == "CEO"
    assert normalise_title("COO") == "COO"


# --------------------------------------------------------------------------- #
# Row-level exclusion
# --------------------------------------------------------------------------- #

def _row(**overrides) -> dict:
    base = {
        "lead_id": "L-1168", "created": "2024-06-08", "name": "Ola",
        "email": "ola@pipegtm.co", "company": "PipeGTM", "employees": "26",
        "website": "pipegtm.co", "title": "VP Growth", "source": "linkedin",
        "monthly_budget": "$6k/mo",
        "notes": "We're a influencer marketing agency, 26 people. Budget approved.",
    }
    base.update(overrides)
    return base


def test_genuine_lead_is_not_excluded():
    assert detect_exclusion(_row()) is None


def test_empty_row_excluded():
    assert detect_exclusion({col: "" for col in _row()}) == EXCL_EMPTY_ROW


def test_column_shifted_header_leak_is_caught():
    """The leaked header is shifted: lead_id="header", created="lead_id", ...

    Plan §2 specified `lead_id == "lead_id"`, which matches zero rows in this
    file. Detection is structural instead.
    """
    leaked = {
        "lead_id": "header", "created": "lead_id", "name": "name",
        "email": "email", "company": "company", "employees": "employees",
        "website": "website", "title": "title", "source": "source",
        "monthly_budget": "budget", "notes": "notes",
    }
    assert detect_exclusion(leaked) == EXCL_HEADER_LEAK


def test_unshifted_header_leak_would_also_be_caught():
    unshifted = {col: col for col in _row()}
    assert detect_exclusion(unshifted) == EXCL_HEADER_LEAK


@pytest.mark.parametrize(
    "overrides",
    [
        {"lead_id": "asdf", "title": "asdf", "source": "test", "notes": "test test ignore this"},
        {"lead_id": "TESTROW", "title": "test", "source": "test", "notes": "QA test entry, please ignore."},
        {"source": "test"},
        {"title": "asdf"},
    ],
)
def test_test_rows_excluded(overrides):
    assert detect_exclusion(_row(**overrides)) == EXCL_TEST_ROW


def test_newsletter_signup_excluded():
    row = _row(notes="This is a newsletter signup that ended up in the leads sheet by mistake.")
    assert detect_exclusion(row) == EXCL_NEWSLETTER


def test_broken_email_row_excluded_on_notes_evidence():
    row = _row(notes="broken email. one line: 'call me'.")
    assert detect_exclusion(row) == EXCL_NOT_A_LEAD


def test_blank_notes_excluded_since_the_llm_has_no_input():
    assert detect_exclusion(_row(notes="")) == EXCL_NO_NOTES


def test_malformed_address_alone_never_excludes_a_row():
    """B3: address shape is not evidence that something is not a lead."""
    assert detect_exclusion(_row(email="weird-email-no-domain")) is None
    assert detect_exclusion(_row(email="kunle[at]meridianbound.com")) is None
    assert detect_exclusion(_row(email="")) is None


def test_student_and_recruiter_titles_are_not_structurally_excluded():
    """Buyer qualification is the LLM's job at Stage 3/4, not the cleaner's."""
    assert detect_exclusion(_row(title="Student")) is None
    assert detect_exclusion(_row(title="Recruiter")) is None


# --------------------------------------------------------------------------- #
# Row cleaning
# --------------------------------------------------------------------------- #

def test_clean_lead_row_preserves_every_raw_value():
    record = clean_lead_row(_row(), 1)
    assert record["monthly_budget_raw"] == "$6k/mo"
    assert record["budget_low"] == 6000.0
    assert record["employees_raw"] == "26"
    assert record["employees_low"] == 26.0
    assert record["source_raw"] == "linkedin"
    assert record["source_clean"] == "linkedin"


def test_cleaned_row_carries_headcount_uncertainty_through_to_scoring():
    """Everything scoring needs to make its own call about `51+` is present."""
    record = clean_lead_row(_row(employees="51+"), 1)
    assert record["employees_raw"] == "51+"        # auditable original
    assert record["employees_low"] == 51.0         # what the source asserts
    assert record["employees_high"] is None        # no invented ceiling
    assert record["employees_status"] == "floor"
    assert record["employees_uncertain"] is True
    assert record["employees_display"] == "51+"


def test_cleaned_row_carries_headcount_range_through_to_scoring():
    record = clean_lead_row(_row(employees="100-200"), 1)
    assert record["employees_low"] == 100.0
    assert record["employees_high"] == 200.0
    assert record["employees_status"] == "range"
    assert record["employees_display"] == "100-200"


def test_notes_are_trimmed_only_and_never_altered():
    notes = "  We're a influencer marketing agency, 26 people.  Budget approved, ASAP.  "
    record = clean_lead_row(_row(notes=notes), 1)
    assert record["notes_clean"] == notes.strip()
    assert "influencer marketing agency, 26 people.  Budget approved" in record["notes_clean"]


def test_missing_lead_id_gets_synthetic_id_and_flag():
    record = clean_lead_row(_row(lead_id=""), 7)
    assert record["lead_id"] == "SYN-0007"
    assert record["id_missing"] is True


def test_present_lead_id_is_not_flagged():
    record = clean_lead_row(_row(), 7)
    assert record["lead_id"] == "L-1168"
    assert record["id_missing"] is False


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #

def test_duplicate_notes_marker_flagged():
    records = [clean_lead_row(_row(notes="(duplicate submission) We're a SEO agency."), 1)]
    flag_duplicates(records)
    assert records[0]["duplicate_flag"] is True
    assert "notes_marker" in records[0]["duplicate_reasons"]


def test_dup_id_suffix_flagged():
    records = [clean_lead_row(_row(lead_id="L-1205-dup"), 1)]
    flag_duplicates(records)
    assert "id_suffix" in records[0]["duplicate_reasons"]


def test_repeated_email_company_pair_flagged_on_both_rows():
    records = [
        clean_lead_row(_row(lead_id="L-1", email="bola@nimblepods.co", company="NimblePods"), 1),
        clean_lead_row(_row(lead_id="L-2", email="bola@nimblepods.co", company="NimblePods"), 2),
    ]
    flag_duplicates(records)
    assert all("email_company_match" in r["duplicate_reasons"] for r in records)


def test_placeholder_addresses_do_not_form_duplicate_clusters():
    """B4: `weird-email-no-domain` occurs 9 times across unrelated companies."""
    records = [
        clean_lead_row(_row(lead_id=f"L-{i}", email="weird-email-no-domain", company=f"Co{i}"), i)
        for i in range(1, 6)
    ]
    flag_duplicates(records)
    assert not any(r["duplicate_flag"] for r in records)


def test_blank_addresses_do_not_form_duplicate_clusters():
    records = [
        clean_lead_row(_row(lead_id=f"L-{i}", email="", company=f"Co{i}"), i)
        for i in range(1, 5)
    ]
    flag_duplicates(records)
    assert not any(r["duplicate_flag"] for r in records)


def test_duplicates_are_flagged_not_removed():
    df = pd.DataFrame([
        _row(lead_id="L-1", email="a@one.co", company="One",
             notes="(duplicate submission) We're a SEO agency, 21 people."),
        _row(lead_id="L-2", email="b@two.co", company="Two"),
    ])
    clean_df, _ = clean_dataframe(df)
    assert len(clean_df) == 2          # the duplicate survives cleaning
    assert clean_df["duplicate_flag"].sum() == 1


# --------------------------------------------------------------------------- #
# DataFrame boundary
# --------------------------------------------------------------------------- #

def test_clean_dataframe_splits_kept_from_excluded():
    df = pd.DataFrame([
        _row(lead_id="L-1"),
        _row(lead_id="TESTROW", source="test", notes="QA test entry, please ignore."),
        _row(lead_id="L-3", notes="This is a newsletter signup that ended up in the leads sheet by mistake."),
    ])
    clean_df, excluded_df = clean_dataframe(df)

    assert len(clean_df) == 1
    assert len(excluded_df) == 2
    assert set(excluded_df["exclusion_reason"]) == {EXCL_TEST_ROW, EXCL_NEWSLETTER}


def test_excluded_rows_retain_their_original_values():
    df = pd.DataFrame([_row(lead_id="TESTROW", source="test", notes="QA test entry, please ignore.")])
    _, excluded_df = clean_dataframe(df)
    assert excluded_df.iloc[0]["lead_id"] == "TESTROW"
    assert excluded_df.iloc[0]["notes"] == "QA test entry, please ignore."


def test_empty_frame_returns_empty_results_with_columns():
    clean_df, excluded_df = clean_dataframe(pd.DataFrame(columns=list(_row())))
    assert clean_df.empty and excluded_df.empty
    assert "budget_low" in clean_df.columns
    assert "employees_status" in clean_df.columns
    assert "exclusion_reason" in excluded_df.columns


def test_missing_required_column_is_rejected_loudly():
    import io
    with pytest.raises(ValueError, match="missing required column"):
        read_leads_csv(io.StringIO("lead_id,notes\nL-1,hello\n"))


# --------------------------------------------------------------------------- #
# Sample-fixture regression - real rows drawn from the assessment CSV
# --------------------------------------------------------------------------- #

SAMPLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sample_data", "leads_sample.csv",
)


@pytest.fixture(scope="module")
def sample_frames():
    if not os.path.exists(SAMPLE_PATH):
        pytest.skip("sample_data/leads_sample.csv not present")
    return clean_dataframe(read_leads_csv(SAMPLE_PATH))


def test_sample_exercises_every_exclusion_reason(sample_frames):
    _, excluded_df = sample_frames
    assert set(excluded_df["exclusion_reason"]) == {
        EXCL_EMPTY_ROW, EXCL_NO_NOTES, EXCL_HEADER_LEAK,
        EXCL_TEST_ROW, EXCL_NEWSLETTER, EXCL_NOT_A_LEAD,
    }


def test_sample_rows_are_fully_accounted_for(sample_frames):
    clean_df, excluded_df = sample_frames
    raw = read_leads_csv(SAMPLE_PATH)
    assert len(clean_df) + len(excluded_df) == len(raw)


def test_every_scoreable_sample_row_has_notes_for_the_llm(sample_frames):
    clean_df, _ = sample_frames
    assert (clean_df["notes_clean"].str.len() > 0).all()


def test_every_scoreable_sample_row_has_a_canonical_source(sample_frames):
    clean_df, _ = sample_frames
    allowed = {"event", "referral", "linkedin", "webform", "cold_reply", "unknown"}
    assert set(clean_df["source_clean"]) <= allowed
