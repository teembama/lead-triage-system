"""Stage 2 - non-destructive cleaning and normalisation of the raw leads CSV.

Design notes
------------
* Field-level parsers are pure stdlib functions over strings. They take a raw
  string and return a small NamedTuple; they never touch pandas. This keeps them
  trivially unit-testable and fast (no API calls, no DataFrame overhead).
* pandas appears only at the DataFrame boundary (`read_leads_csv`,
  `clean_dataframe`).
* Cleaning is non-destructive: every raw column survives as `<name>_raw`, junk
  rows are routed to a separate excluded frame rather than deleted, and
  duplicates are *flagged* for a human rather than auto-removed.
* Nothing here guesses at missing data. A value that cannot be parsed becomes
  `None` with an explanatory status string - never a fabricated number and never
  a silent zero.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Iterable, NamedTuple, Optional

import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

CANONICAL_COLUMNS = [
    "lead_id",
    "created",
    "name",
    "email",
    "company",
    "employees",
    "website",
    "title",
    "source",
    "monthly_budget",
    "notes",
]

# Canonical source keys consumed by the scorer (scoring.py, plan §12).
SOURCE_ALIASES = {
    "event": "event",
    "referral": "referral",
    "linkedin": "linkedin",
    "webform": "webform",
    "cold reply": "cold_reply",
    "cold_reply": "cold_reply",
    "coldreply": "cold_reply",
}

# Values that are placeholders rather than data.
JUNK_TITLES = {"asdf", "test", "title"}
JUNK_LEAD_IDS = {"asdf", "test", "testrow"}
BUDGET_PLACEHOLDERS = {"tbd", "depends", "budget", "asdf", "n/a", "na", "none", "unknown", "-"}
EMAIL_PLACEHOLDERS = {"weird-email-no-domain", "email", "none", "n/a", "na"}

# Exclusion reasons (plan §2 row-level junk filter, §8 structural data junk).
EXCL_EMPTY_ROW = "empty_row"
EXCL_HEADER_LEAK = "header_leak"
EXCL_TEST_ROW = "test_row"
EXCL_NEWSLETTER = "newsletter_signup"
EXCL_NOT_A_LEAD = "not_a_lead"
EXCL_NO_NOTES = "no_notes"

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


# --------------------------------------------------------------------------- #
# Return types
# --------------------------------------------------------------------------- #

class ParsedDate(NamedTuple):
    """`iso` is None when the value could not be parsed; `raw` is always kept."""
    iso: Optional[str]
    status: str          # exact | missing | unparseable
    ambiguous: bool      # True when DD-MM/MM-DD could not be told apart


class Measure(NamedTuple):
    """An interval, not a point. What the source actually asserts, nothing more.

    The cleaning layer's job is to represent what the data tells us; deciding
    how that maps to a scoring band is the scoring layer's job (plan §12), so
    nothing here collapses uncertainty into an invented single figure.

        "26"      -> low=26,  high=26,   status="exact"
        "35-55"   -> low=35,  high=55,   status="range"      (kept as a range)
        "51+"     -> low=51,  high=None, status="floor"      (no upper bound invented)
        "~61"     -> low=61,  high=61,   status="approx"     (stated, but soft)
        ""        -> low=None, high=None, status="missing"
        "asdf"    -> low=None, high=None, status="unparseable"

    `high is None` means *unbounded above* when `status == "floor"`, and
    *unknown* otherwise; `status` is always the disambiguator, so callers should
    branch on it rather than on `None` alone.

    Deviation from plan §16, which specified `int | "unknown"`: a
    number-or-magic-string union forces every caller to type-check before doing
    arithmetic, and cannot express "at least 51" at all.
    """
    low: Optional[float]
    high: Optional[float]
    status: str          # exact | range | floor | approx | zero
                         # | missing | placeholder | unparseable
    raw: str = ""

    @property
    def is_known(self) -> bool:
        return self.low is not None

    @property
    def is_uncertain(self) -> bool:
        """True when the source gives a band or a bound rather than a figure."""
        return self.status in {"range", "floor", "approx"}

    @property
    def unbounded_above(self) -> bool:
        return self.status == "floor"


def describe_measure(measure: Measure) -> str:
    """Human-readable rendering for the UI, CSV export and LLM context."""
    if not measure.is_known:
        return "unknown"
    low = _trim_number(measure.low)
    if measure.status == "floor":
        return f"{low}+"
    if measure.status == "range":
        return f"{low}-{_trim_number(measure.high)}"
    if measure.status == "approx":
        return f"~{low}"
    return low


def _trim_number(value: Optional[float]) -> str:
    if value is None:
        return ""
    return str(int(value)) if float(value).is_integer() else str(value)


class ParsedEmail(NamedTuple):
    address: Optional[str]   # lowercased, for matching
    is_valid: bool
    was_repaired: bool       # True when "[at]" / "(at)" was rewritten to "@"


# --------------------------------------------------------------------------- #
# Field parsers - pure functions, no pandas
# --------------------------------------------------------------------------- #

def clean_text(raw: object) -> str:
    """Trim whitespace and normalise nulls to an empty string."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    return str(raw).strip()


def parse_date(raw: object) -> ParsedDate:
    """Normalise the six `created` formats present in the dataset to ISO.

    Formats are dispatched by regex rather than by trying `strptime` formats in
    sequence. A blind cascade is unsafe here: `%Y` happily consumes a two-digit
    year, so `strptime("6/1/24", "%m/%d/%Y")` silently yields year 24 instead of
    failing over to the `%m/%d/%y` branch. Matching the shape first removes that
    class of bug entirely.

    Day-first is used for the `DD-MM-YYYY` shape: across the dataset 38 rows of
    that shape have a first component > 12 and none have a second component > 12,
    which settles the format even though 34 individual rows stay ambiguous in
    isolation. Those 34 are marked `ambiguous=True` for audit.
    """
    text = clean_text(raw)
    if not text:
        return ParsedDate(None, "missing", False)

    # ISO, zero-padded or not: 2024-06-08 / 2024-6-7
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if m:
        return _build_date(int(m[1]), int(m[2]), int(m[3]), ambiguous=False)

    # US slash with 4-digit year: 06/28/2024
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if m:
        return _build_date(int(m[3]), int(m[1]), int(m[2]), ambiguous=False)

    # US slash with 2-digit year: 6/1/24
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2})", text)
    if m:
        return _build_date(2000 + int(m[3]), int(m[1]), int(m[2]), ambiguous=False)

    # Month-name: Jun 7 2024 / June 7, 2024
    m = re.fullmatch(r"([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})", text)
    if m:
        month = _MONTH_ABBR.get(m[1][:3].lower())
        if month:
            return _build_date(int(m[3]), month, int(m[2]), ambiguous=False)
        return ParsedDate(None, "unparseable", False)

    # Dash with 4-digit year, day-first: 19-06-2024
    m = re.fullmatch(r"(\d{1,2})-(\d{1,2})-(\d{4})", text)
    if m:
        day, month = int(m[1]), int(m[2])
        return _build_date(int(m[3]), month, day, ambiguous=day <= 12 and month <= 12)

    return ParsedDate(None, "unparseable", False)


def _build_date(year: int, month: int, day: int, ambiguous: bool) -> ParsedDate:
    try:
        return ParsedDate(date(year, month, day).isoformat(), "exact", ambiguous)
    except ValueError:
        return ParsedDate(None, "unparseable", False)


def parse_employees(raw: object) -> Measure:
    """Parse headcount into an interval that preserves the source's uncertainty.

    `35-55` stays a range, `51+` stays a lower bound with no upper bound
    invented, `~61` is recorded as approximate. Nothing is collapsed to a single
    figure here: a midpoint would assert a precision the source never gave, and
    "at least 51" is genuinely different information from "51". Per plan §2, an
    unknown headcount is never assumed to be small.

    How these representations map onto the Company Size /10 rubric is decided in
    scoring.py, not here.
    """
    text = clean_text(raw)
    if not text:
        return Measure(None, None, "missing", text)

    if re.fullmatch(r"\d+", text):
        value = float(text)
        return Measure(value, value, "exact", text)

    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", text)
    if m:
        low, high = float(m[1]), float(m[2])
        if low > high:
            low, high = high, low
        return Measure(low, high, "range", text)

    m = re.fullmatch(r"(\d+)\s*\+", text)
    if m:
        # No upper bound is invented. "50+" asserts a floor and nothing else.
        return Measure(float(m[1]), None, "floor", text)

    m = re.fullmatch(r"~\s*(\d+)", text)
    if m:
        value = float(m[1])
        return Measure(value, value, "approx", text)

    return Measure(None, None, "unparseable", text)


def parse_budget(raw: object) -> Measure:
    """Normalise `monthly_budget` to a monthly USD interval.

    Handles `$6k/mo`, `5,000/mo`, `6k-8k`, `$8,500`, plain integers and `0`.
    Values carrying no `/mo` suffix are read as monthly, since that is what the
    column asserts. A literal `0` is recorded as a confirmed zero, kept distinct
    from a missing value - "no budget" and "not recorded" are different facts and
    only the first is evidence.

    Ranges are preserved rather than averaged, for the same reason headcount
    ranges are: `6k-8k` is a stated band, and 7000 is a number nobody wrote. The
    budget-capacity rubric in scoring.py decides how a band maps to a tier.
    """
    text = clean_text(raw)
    if not text:
        return Measure(None, None, "missing", text)
    if text.lower() in BUDGET_PLACEHOLDERS:
        return Measure(None, None, "placeholder", text)

    body = re.sub(r"\s*/\s*(mo|month|mth)\b\.?", "", text, flags=re.I)
    body = re.sub(r"\bper\s+month\b", "", body, flags=re.I).strip()

    m = re.fullmatch(r"\$?\s*([\d,.]+)\s*(k?)\s*-\s*\$?\s*([\d,.]+)\s*(k?)", body, flags=re.I)
    if m:
        left, left_k, right, right_k = m[1], m[2], m[3], m[4]
        low = _to_number(left)
        high = _to_number(right)
        if low is None or high is None:
            return Measure(None, None, "unparseable", text)
        # "$6-8k": the k on the right operand governs a bare left operand too.
        right_is_k = bool(right_k)
        low *= 1000 if (left_k or (right_is_k and low < 1000)) else 1
        high *= 1000 if right_is_k else 1
        if low > high:
            low, high = high, low
        return Measure(low, high, "range", text)

    m = re.fullmatch(r"\$?\s*([\d,.]+)\s*(k?)", body, flags=re.I)
    if m:
        value = _to_number(m[1])
        if value is None:
            return Measure(None, None, "unparseable", text)
        if m[2]:
            value *= 1000
        return Measure(value, value, "zero" if value == 0 else "exact", text)

    return Measure(None, None, "unparseable", text)


def _to_number(token: str) -> Optional[float]:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def normalise_email(raw: object) -> ParsedEmail:
    """Lowercase for matching and repair `[at]`-style obfuscation.

    A malformed address is flagged, never used as grounds for dropping the row.
    Eleven rows in the dataset use `name[at]domain.com` and are genuine
    prospects; excluding them on address shape alone would discard real leads
    (plan §2 lists "broken email" under junk, but the only true non-lead there is
    identified by its notes, not its address).
    """
    text = clean_text(raw)
    if not text:
        return ParsedEmail(None, False, False)

    repaired = re.sub(r"\[at\]|\(at\)", "@", text, flags=re.I)
    was_repaired = repaired != text
    lowered = repaired.lower()

    if lowered in EMAIL_PLACEHOLDERS:
        return ParsedEmail(lowered, False, was_repaired)
    return ParsedEmail(lowered, bool(_EMAIL_RE.fullmatch(lowered)), was_repaired)


def normalise_source(raw: object) -> str:
    """Map to a canonical scoring key; unrecognised/junk values become `unknown`."""
    text = clean_text(raw).lower()
    return SOURCE_ALIASES.get(re.sub(r"\s+", " ", text), "unknown")


def normalise_title(raw: object) -> Optional[str]:
    """Trim and strip placeholder titles; `None` means no usable title.

    Original casing is preserved rather than title-cased as plan §2 suggested,
    because title-casing mangles the acronyms that dominate this column
    (`CEO` -> `Ceo`, `COO` -> `Coo`). The column is already consistently cased.
    """
    text = clean_text(raw)
    if not text or text.lower() in JUNK_TITLES:
        return None
    return text


# --------------------------------------------------------------------------- #
# Row-level classification
# --------------------------------------------------------------------------- #

def detect_exclusion(row: dict) -> Optional[str]:
    """Return an exclusion reason for structural junk, else None.

    These are rows that are not leads at all, so they never reach the LLM or the
    buyer gate. The notes-based checks match documented literal phrases only -
    interpreting notes is the LLM's job, and a loose keyword rule here would
    quietly disqualify real prospects.
    """
    values = {col: clean_text(row.get(col)) for col in CANONICAL_COLUMNS}

    if not any(values.values()):
        return EXCL_EMPTY_ROW

    # Header row leaked into the data. It is column-shifted in this dataset
    # (lead_id="header", created="lead_id", ...), so keying on `lead_id ==
    # "lead_id"` as plan §2 specified matches nothing. Detect it structurally
    # instead: a real lead never has several fields equal to their own name.
    name_matches = sum(1 for col, value in values.items() if value.lower() == col)
    if name_matches >= 3 or values["lead_id"].lower() == "header":
        return EXCL_HEADER_LEAK

    if (
        values["lead_id"].lower() in JUNK_LEAD_IDS
        or values["source"].lower() == "test"
        or values["title"].lower() in {"asdf", "test"}
    ):
        return EXCL_TEST_ROW

    notes = values["notes"].lower()
    if "newsletter signup" in notes:
        return EXCL_NEWSLETTER
    if "broken email" in notes:
        return EXCL_NOT_A_LEAD
    if not notes:
        return EXCL_NO_NOTES

    return None


def clean_lead_row(row: dict, row_number: int) -> dict:
    """Normalise one raw row into the cleaned record consumed by later stages.

    Every raw value survives as `<column>_raw`; derived values sit alongside
    under explicit names, so nothing is overwritten and every number on screen
    can be traced back to what was in the file.
    """
    raw = {col: clean_text(row.get(col)) for col in CANONICAL_COLUMNS}

    lead_id = raw["lead_id"]
    id_missing = not lead_id
    if id_missing:
        lead_id = f"SYN-{row_number:04d}"

    created = parse_date(raw["created"])
    employees = parse_employees(raw["employees"])
    budget = parse_budget(raw["monthly_budget"])
    email = normalise_email(raw["email"])

    record = {f"{col}_raw": value for col, value in raw.items()}
    record.update(
        {
            "row_number": row_number,
            "lead_id": lead_id,
            "id_missing": id_missing,
            "created_clean": created.iso,
            "created_status": created.status,
            "created_ambiguous": created.ambiguous,
            "name_clean": raw["name"] or None,
            "email_clean": email.address,
            "email_valid": email.is_valid,
            "email_repaired": email.was_repaired,
            "company_clean": raw["company"] or None,
            # Headcount is carried as an interval plus a status. `employees_high`
            # is None for a "51+" floor because the source gives no upper bound;
            # `employees_display` is the audit-friendly rendering ("51+", "35-55").
            "employees_low": employees.low,
            "employees_high": employees.high,
            "employees_status": employees.status,
            "employees_display": describe_measure(employees),
            "employees_uncertain": employees.is_uncertain,
            "website_clean": raw["website"] or None,
            "title_clean": normalise_title(raw["title"]),
            "source_clean": normalise_source(raw["source"]),
            "budget_low": budget.low,
            "budget_high": budget.high,
            "budget_status": budget.status,
            "budget_display": describe_measure(budget),
            "budget_uncertain": budget.is_uncertain,
            # Notes are trimmed only. This is the LLM's input and its content is
            # never altered (plan §2).
            "notes_clean": raw["notes"],
        }
    )
    return record


# --------------------------------------------------------------------------- #
# Duplicate detection
# --------------------------------------------------------------------------- #

def flag_duplicates(records: list[dict]) -> None:
    """Annotate records in place with duplicate evidence. Never removes a row.

    Three independent signals, unioned:
      * notes containing "(duplicate submission)"
      * a `lead_id` carrying a `-dup` suffix
      * a repeated (email, company) pair

    The pair check ignores invalid/placeholder addresses: the literal string
    `weird-email-no-domain` occurs 9 times and blank addresses 8 times, so
    matching on those would manufacture large bogus duplicate clusters out of
    rows that have nothing to do with each other.
    """
    pair_counts: dict[tuple[str, str], int] = {}
    for record in records:
        key = _duplicate_key(record)
        if key:
            pair_counts[key] = pair_counts.get(key, 0) + 1

    for record in records:
        reasons = []
        if "(duplicate submission)" in record["notes_clean"].lower():
            reasons.append("notes_marker")
        if record["lead_id"].lower().endswith("-dup"):
            reasons.append("id_suffix")
        key = _duplicate_key(record)
        if key and pair_counts[key] > 1:
            reasons.append("email_company_match")

        record["duplicate_flag"] = bool(reasons)
        record["duplicate_reasons"] = ";".join(reasons)


def _duplicate_key(record: dict) -> Optional[tuple[str, str]]:
    if not record["email_valid"] or not record["email_clean"]:
        return None
    company = (record["company_clean"] or "").lower()
    return (record["email_clean"], company)


# --------------------------------------------------------------------------- #
# DataFrame boundary
# --------------------------------------------------------------------------- #

def read_leads_csv(source) -> pd.DataFrame:
    """Read the leads CSV as raw strings, preserving every cell verbatim.

    `dtype=str` and `keep_default_na=False` stop pandas from coercing IDs to
    floats or turning the literal budget text `NA` into a null before the
    parsers get a chance to classify it.
    """
    try:
        df = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except UnicodeDecodeError:
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="latin-1")

    missing = [col for col in CANONICAL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")
    return df


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean a raw leads frame into (scoreable rows, excluded junk rows).

    Excluded rows keep their original values plus an `exclusion_reason`, so the
    junk filter is auditable and reversible rather than a silent delete.
    """
    kept: list[dict] = []
    excluded: list[dict] = []

    for position, (_, row) in enumerate(df.iterrows(), start=1):
        row_dict = row.to_dict()
        reason = detect_exclusion(row_dict)
        if reason:
            entry = {col: clean_text(row_dict.get(col)) for col in CANONICAL_COLUMNS}
            entry["row_number"] = position
            entry["exclusion_reason"] = reason
            excluded.append(entry)
        else:
            kept.append(clean_lead_row(row_dict, position))

    flag_duplicates(kept)

    clean_df = pd.DataFrame(kept, columns=_clean_columns()) if kept else pd.DataFrame(columns=_clean_columns())
    excluded_df = (
        pd.DataFrame(excluded, columns=["row_number", "exclusion_reason", *CANONICAL_COLUMNS])
        if excluded
        else pd.DataFrame(columns=["row_number", "exclusion_reason", *CANONICAL_COLUMNS])
    )
    return clean_df, excluded_df


def _clean_columns() -> list[str]:
    derived = [
        "row_number", "lead_id", "id_missing",
        "created_clean", "created_status", "created_ambiguous",
        "name_clean", "email_clean", "email_valid", "email_repaired",
        "company_clean",
        "employees_low", "employees_high", "employees_status",
        "employees_display", "employees_uncertain",
        "website_clean", "title_clean", "source_clean",
        "budget_low", "budget_high", "budget_status",
        "budget_display", "budget_uncertain",
        "notes_clean", "duplicate_flag", "duplicate_reasons",
    ]
    return derived + [f"{col}_raw" for col in CANONICAL_COLUMNS]


def summarise_cleaning(clean_df: pd.DataFrame, excluded_df: pd.DataFrame) -> dict:
    """Small counts dict for the UI and for eyeballing a run's sanity."""
    reason_counts: dict[str, int] = {}
    if not excluded_df.empty:
        reason_counts = excluded_df["exclusion_reason"].value_counts().to_dict()
    return {
        "total_rows": len(clean_df) + len(excluded_df),
        "scoreable": len(clean_df),
        "excluded": len(excluded_df),
        "exclusion_reasons": reason_counts,
        "duplicates_flagged": int(clean_df["duplicate_flag"].sum()) if not clean_df.empty else 0,
        "budget_known": int(clean_df["budget_low"].notna().sum()) if not clean_df.empty else 0,
        "employees_known": int(clean_df["employees_low"].notna().sum()) if not clean_df.empty else 0,
        "employees_uncertain": int(clean_df["employees_uncertain"].sum()) if not clean_df.empty else 0,
        "dates_unparsed": int((clean_df["created_clean"].isna()).sum()) if not clean_df.empty else 0,
        "dates_ambiguous": int(clean_df["created_ambiguous"].sum()) if not clean_df.empty else 0,
    }
