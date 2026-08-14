"""Production-hardening tests: rate limiting, 429 handling, and the rule that a
provider failure is never presented as a genuine judgement about a lead.

These span extraction, pipeline and UI because that is the path the reported bug
travelled: a stringified Gemini 429 reached the user under "Why this needs a
person". No API key and no network - providers are fakes throughout.
"""

import json
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm_extraction as lx  # noqa: E402
import pipeline  # noqa: E402
import providers  # noqa: E402
import schema  # noqa: E402
import ui  # noqa: E402
from llm_extraction import ExtractionCache, extract_signals  # noqa: E402
from providers import ProviderError, ProviderResponse  # noqa: E402
from providers.gemini_provider import _classify  # noqa: E402

from test_llm_extraction import LEAD, FakeProvider, no_sleep, valid_payload  # noqa: E402

# The verbatim shape of the error that reached a user, kept as a fixture so the
# leak guards below test the real thing rather than a paraphrase.
REAL_429 = (
    "ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'message': 'You exceeded your current "
    "quota, please check your plan and billing details. Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, model: "
    "gemini-3.5-flash-lite, limit: 15. Please retry in 33.514422342s.', "
    "'status': 'RESOURCE_EXHAUSTED'}}"
)

LEAK_MARKERS = ("429", "RESOURCE_EXHAUSTED", "googleapis", "gemini-3.5", "quota",
                "retry in", "ClientError", "RateLimitError", "limit: 15")


def run(lead, provider, **kwargs):
    kwargs.setdefault("sleep", no_sleep)
    return extract_signals(lead, provider, **kwargs)


def failing(code="rate_limited", detail=REAL_429, times=6):
    return FakeProvider([ProviderError("request failed", code=code, detail=detail)] * times)


# --------------------------------------------------------------------------- #
# Rate limiter
# --------------------------------------------------------------------------- #

class _Stop(Exception):
    """Aborts acquire() at its first wait, so a test can read the requested delay
    without a no-op sleep spinning through it in real time."""


def _first_wait(limiter) -> float:
    waits = []

    def record(seconds):
        waits.append(seconds)
        raise _Stop

    try:
        limiter.acquire(record)
    except _Stop:
        pass
    return waits[0] if waits else 0.0


def test_limiter_spaces_requests_to_the_budget():
    limiter = lx.RateLimiter(rpm=60)               # one per second
    limiter.acquire(lambda _s: None)               # drain the single token
    assert _first_wait(limiter) == pytest.approx(1.0, abs=0.1)


def test_limiter_allows_a_burst_up_to_capacity():
    limiter = lx.RateLimiter(rpm=600)              # capacity 10
    slept = []
    for _ in range(10):
        limiter.acquire(slept.append)
    assert sum(slept) == 0, "a full bucket should not block"


def test_limiter_can_be_disabled():
    limiter = lx.RateLimiter(rpm=0)
    assert limiter.enabled is False
    slept = []
    for _ in range(50):
        limiter.acquire(slept.append)
    assert sum(slept) == 0


def test_pause_holds_every_worker():
    """A 429 is a run-wide fact, not a per-worker one."""
    limiter = lx.RateLimiter(rpm=6000)
    limiter.pause(5.0)
    assert _first_wait(limiter) == pytest.approx(5.0, abs=0.2)


def test_limiter_is_thread_safe():
    limiter = lx.RateLimiter(rpm=6000)
    taken, lock = [], threading.Lock()

    def worker():
        for _ in range(20):
            limiter.acquire(lambda _s: None)
            with lock:
                taken.append(1)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(taken) == 160           # no grant lost or double-counted


def test_rpm_is_configurable(monkeypatch):
    monkeypatch.setattr(providers, "_secrets_file", lambda: {})
    monkeypatch.delenv("LLM_RPM", raising=False)
    assert lx.rpm_setting() == lx.DEFAULT_RPM == 15.0

    monkeypatch.setenv("LLM_RPM", "300")
    assert lx.rpm_setting() == 300.0
    assert lx.rpm_setting(60) == 60.0

    monkeypatch.setenv("LLM_RPM", "nonsense")
    assert lx.rpm_setting() == lx.DEFAULT_RPM


def test_worker_count_is_sized_to_the_budget():
    """The pool is derived from the budget: enough threads to keep the quota
    spent, and never more than the cap."""
    assert lx.workers_for_rpm(15) == 2
    assert lx.workers_for_rpm(60) == 5
    assert lx.workers_for_rpm(10_000) == lx.DEFAULT_MAX_WORKERS      # capped
    assert lx.workers_for_rpm(0) == lx.DEFAULT_MAX_WORKERS           # unlimited


def test_the_worker_cap_is_twenty():
    assert lx.DEFAULT_MAX_WORKERS == 20


def test_the_assumed_latency_matches_what_was_measured():
    """Sizing is only correct if this tracks reality. Measured at 4.25s mean."""
    assert lx.ASSUMED_CALL_SECONDS == 4.25


def test_a_tier_one_budget_is_not_capped_at_the_old_eight():
    """The regression this change exists to prevent: the cap used to bind at 8,
    so every RPM above ~120 bought nothing."""
    workers = lx.workers_for_rpm(240)
    assert workers == 17
    assert workers > 8
    assert workers < lx.DEFAULT_MAX_WORKERS          # derived, not clamped


def test_the_pool_can_absorb_the_budget_it_is_given():
    """A pool of W workers at L seconds a call sustains W/L calls per second.
    If that is below the limiter's rate the quota goes unspent - which is the
    bug that made LLM_RPM inert above 120."""
    for rpm in (15, 60, 120, 240):
        workers = lx.workers_for_rpm(rpm)
        pool_rate = workers / lx.ASSUMED_CALL_SECONDS
        assert pool_rate >= rpm / 60.0 - 1e-9, (
            f"{rpm} rpm needs {rpm / 60.0:.2f} calls/s, pool sustains {pool_rate:.2f}"
        )


@pytest.mark.parametrize("low,high", [(15, 60), (60, 120), (120, 240)])
def test_raising_the_budget_actually_raises_concurrency(low, high):
    """Guards against LLM_RPM becoming a number that changes nothing."""
    assert lx.workers_for_rpm(high) > lx.workers_for_rpm(low)


def test_the_limiter_still_enforces_the_configured_rate():
    """240 rpm is four calls a second. Measured through `_first_wait`, which
    stops at the first sleep - a recording sleep that returns would spin, since
    it never advances the clock the limiter is reading."""
    limiter = lx.RateLimiter(rpm=240)
    assert limiter.rate == 4.0
    for _ in range(int(limiter.capacity)):       # drain the bucket
        limiter.acquire(lambda _s: None)
    assert _first_wait(limiter) == pytest.approx(0.25, abs=0.05)


def test_the_burst_is_bounded_by_the_budget():
    """Capacity is the burst ceiling: at 240 rpm four requests may leave
    together, not seventeen. A pool of 17 cannot stampede the provider."""
    limiter = lx.RateLimiter(rpm=240)
    assert limiter.capacity == 4.0
    assert limiter.capacity < lx.workers_for_rpm(240)


def test_max_workers_resolves_from_secrets_not_just_the_environment(monkeypatch):
    """The Secrets panel writes to st.secrets, never to the environment. This
    setting used to read os.environ alone, so on a deployment it was inert."""
    import providers

    monkeypatch.delenv("LLM_MAX_WORKERS", raising=False)
    providers._secrets_file.cache_clear()
    monkeypatch.setattr(providers, "_secrets_file", lambda: {"LLM_MAX_WORKERS": "6"})
    assert lx.max_workers_setting() == 6

    # The environment still wins over the file, as it does for every setting.
    monkeypatch.setenv("LLM_MAX_WORKERS", "9")
    assert lx.max_workers_setting() == 9


def test_a_secrets_worker_override_is_honoured_by_the_batch(monkeypatch):
    """The override branch in extract_batch read os.environ too, so a secrets
    value was ignored even where max_workers_setting would have honoured it."""
    import providers

    monkeypatch.delenv("LLM_MAX_WORKERS", raising=False)
    providers._secrets_file.cache_clear()
    monkeypatch.setattr(providers, "_secrets_file", lambda: {"LLM_MAX_WORKERS": "2"})

    seen = {}
    real_pool = lx.ThreadPoolExecutor

    def spy(max_workers, *a, **kw):
        seen["workers"] = max_workers
        return real_pool(max_workers=max_workers, *a, **kw)

    monkeypatch.setattr(lx, "ThreadPoolExecutor", spy)
    leads = [dict(LEAD, lead_id=f"L-{i}", notes_clean=f"note {i}") for i in range(6)]
    lx.extract_batch(leads, FakeProvider([valid_payload()] * 6), rpm=0,
                     cache=lx.ExtractionCache(path=None), sleep=no_sleep)
    assert seen["workers"] == 2


def test_concurrency_is_preserved():
    """The limiter must throttle, not serialize.

    A barrier of 4 can only be cleared if four workers are inside `extract` at
    the same moment; if the pipeline were serialized this times out instead.
    """
    barrier = threading.Barrier(4, timeout=5)
    cleared = []

    class Concurrent:
        name, model = "t", "t"

        def extract(self, *_a):
            barrier.wait()
            cleared.append(1)
            return ProviderResponse(text=json.dumps(valid_payload()))

    leads = [{**LEAD, "lead_id": f"L-{i}", "notes_clean": f"Enquiry {i}."} for i in range(4)]
    lx.extract_batch(leads, Concurrent(), cache=ExtractionCache(path=None),
                     rpm=0, max_workers=4, sleep=no_sleep)
    assert len(cleared) == 4, "workers did not run concurrently"


# --------------------------------------------------------------------------- #
# 429 classification and backoff
# --------------------------------------------------------------------------- #

def test_real_429_is_classified_and_its_retry_delay_read():
    code, retry_after = _classify(REAL_429)
    assert code == "rate_limited"
    assert retry_after == pytest.approx(33.514422342)


@pytest.mark.parametrize("text,expected", [
    ("ClientError: 429 quota exceeded", "rate_limited"),
    ("ServerError: RESOURCE_EXHAUSTED", "rate_limited"),
    ("DeadlineExceeded: request timed out", "timeout"),
    ("ValueError: something else entirely", "provider_error"),
])
def test_error_classification(text, expected):
    assert _classify(text)[0] == expected


def test_backoff_prefers_the_providers_own_delay():
    exc = ProviderError("rate limited", code="rate_limited", retry_after=30.0)
    for attempt in (1, 2, 3):
        assert 30.0 <= lx._backoff_delay(exc, attempt) <= 30.0 * (1 + lx.RETRY_JITTER)


def test_backoff_is_exponential_without_a_provider_delay():
    exc = ProviderError("boom", code="provider_error")
    assert lx._backoff_delay(exc, 1) >= lx.RETRY_BASE_DELAY
    assert lx._backoff_delay(exc, 3) >= lx.RETRY_BASE_DELAY * 4
    assert lx._backoff_delay(exc, 20) <= lx.RETRY_MAX_DELAY * (1 + lx.RETRY_JITTER)


def test_backoff_adds_jitter():
    """Without jitter, every worker that trips one 429 wakes together and re-trips it."""
    exc = ProviderError("rate limited", code="rate_limited", retry_after=10.0)
    assert len({lx._backoff_delay(exc, 1) for _ in range(20)}) > 1


class SpyLimiter(lx.RateLimiter):
    """Records pauses instead of enforcing them, so a no-op sleep in tests cannot
    spin away the very pause being measured."""

    def __init__(self, rpm=6000.0):
        super().__init__(rpm)
        self.pauses: list[float] = []

    def pause(self, seconds: float) -> None:
        self.pauses.append(seconds)


def test_a_429_pauses_the_shared_budget():
    limiter = SpyLimiter()
    provider = FakeProvider([
        ProviderError("rate limited", code="rate_limited", retry_after=5.0, detail=REAL_429),
        valid_payload(),
    ])
    result = extract_signals(LEAD, provider, cache=None, limiter=limiter, sleep=no_sleep)
    assert result.status == "recovered"
    assert limiter.pauses, "the 429 should have paused every worker, not just this one"
    assert limiter.pauses[0] >= 5.0, "the provider's own retry delay should be honoured"


def test_a_contract_failure_does_not_pause_the_budget():
    """Only rate limiting is a run-wide fact; a malformed reply is one lead's problem."""
    limiter = SpyLimiter()
    extract_signals(LEAD, FakeProvider(["{not json", valid_payload()]),
                    cache=None, limiter=limiter, sleep=no_sleep)
    assert limiter.pauses == []


def test_retries_are_bounded():
    provider = failing(times=20)
    assert run(LEAD, provider).status == "failed"
    assert len(provider.calls) == lx.MAX_TRANSPORT_ATTEMPTS


def test_contract_failures_keep_the_single_corrective_retry():
    """§16 Stage 4: a malformed response is retried once, then flagged."""
    provider = FakeProvider(["{not json"] * 5)
    result = run(LEAD, provider)
    assert result.status == "failed"
    assert result.failure_code == "invalid_response"
    assert len(provider.calls) == lx.MAX_CONTRACT_ATTEMPTS


# --------------------------------------------------------------------------- #
# A failure is not an ambiguity
# --------------------------------------------------------------------------- #

def test_failure_record_carries_no_vendor_text():
    result = run(LEAD, failing())
    blob = json.dumps(result.extraction)
    for marker in LEAK_MARKERS:
        assert marker not in blob, f"{marker!r} leaked into the extraction record"
    assert result.extraction["ambiguity_reason"] == schema.ASSESSMENT_FAILED_REASON
    assert "429" in result.error          # ...but retained internally for debugging


def test_failure_is_distinguishable_from_genuine_ambiguity():
    genuine = valid_payload(is_potential_buyer="ambiguous",
                            ambiguity_reason="fellow agency owner, intent unclear")
    ok = run(LEAD, FakeProvider([genuine]))
    bad = run(LEAD, failing())

    # Both are 'ambiguous', so routing is unchanged...
    assert ok.extraction["is_potential_buyer"] == "ambiguous"
    assert bad.extraction["is_potential_buyer"] == "ambiguous"
    # ...but only one is a failure, and only it carries a code.
    assert (ok.status, ok.failure_code) == ("ok", None)
    assert (bad.status, bad.failure_code) == ("failed", "rate_limited")


@pytest.mark.parametrize("code", ["rate_limited", "timeout", "provider_error"])
def test_failure_code_reflects_the_cause(code):
    assert run(LEAD, failing(code=code)).failure_code == code


def test_refusal_is_recorded_and_not_retried():
    provider = FakeProvider([ProviderResponse(text=None, refused=True,
                                              refusal_reason="blocked by policy")])
    result = run(LEAD, provider)
    assert result.status == "failed"
    assert result.failure_code == "refused"
    assert len(provider.calls) == 1


# --------------------------------------------------------------------------- #
# Cache key
# --------------------------------------------------------------------------- #

def test_company_is_no_longer_part_of_the_cache_key():
    a = {**LEAD, "company_clean": "PipeGTM"}
    b = {**LEAD, "company_clean": "SomethingElse"}
    assert ExtractionCache.key(a, "gemini", "m") == ExtractionCache.key(b, "gemini", "m")


@pytest.mark.parametrize("field,other", [
    ("notes_clean", "completely different notes"), ("title_clean", "CEO"),
    ("employees_display", "40"), ("budget_display", "9000"),
])
def test_other_context_fields_still_separate_the_key(field, other):
    assert ExtractionCache.key(LEAD, "g", "m") != ExtractionCache.key({**LEAD, field: other}, "g", "m")


def test_company_only_difference_reuses_one_call():
    cache = ExtractionCache(path=None)
    provider = FakeProvider([valid_payload()])
    run({**LEAD, "company_clean": "A"}, provider, cache=cache)
    second = run({**LEAD, "lead_id": "L-2", "company_clean": "B"}, provider, cache=cache)
    assert len(provider.calls) == 1
    assert second.from_cache is True


def test_cache_dedup_on_the_real_file():
    """Measured effect of the key fix on the assessment data."""
    from data_cleaning import clean_dataframe, read_leads_csv

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv = os.path.join(root, "Cohort 3 Assessment — Task 1 Leads (messy).csv")
    if not os.path.exists(csv):
        pytest.skip("assessment CSV not present")

    clean, _ = clean_dataframe(read_leads_csv(csv))
    leads = clean.to_dict("records")
    keys = {ExtractionCache.key(x, "gemini", "m") for x in leads}
    assert len(leads) == 509
    assert len(keys) == 446, f"expected 446 calls after the key fix, got {len(keys)}"


# --------------------------------------------------------------------------- #
# Nothing technical reaches the user
# --------------------------------------------------------------------------- #

def _routed(provider, lead_overrides=None):
    """Drive one lead through extraction -> scoring -> routing, as the app does."""
    import pandas as pd
    from data_cleaning import CANONICAL_COLUMNS

    row = {c: "" for c in CANONICAL_COLUMNS}
    row.update(lead_id="L-1261", created="2024-06-07", name="Sam",
               email="sam@cladwell.ng", company="CladwellWorks", title="Owner",
               source="referral",
               notes="Fellow agency owner here, mostly researching the market.")
    row.update(lead_overrides or {})
    out = pipeline.run_pipeline(pd.DataFrame([row]).astype(str), provider=provider,
                                cache=ExtractionCache(path=None), rpm=0,
                                sleep=no_sleep)
    return out["rows"][0]


def test_assessment_failure_reaches_the_ui_as_clean_copy():
    row = _routed(failing())
    assert row["recommendation"] == "REVIEW"
    assert row["assessment_failed"] is True
    assert row["failure_code"] == "rate_limited"
    assert row["review_reason"] == pipeline.ASSESSMENT_FAILED_REVIEW_REASON

    html = ui.lead_detail(row)
    assert "could not be assessed automatically" in html
    for marker in LEAK_MARKERS:
        assert marker not in html, f"{marker!r} rendered to the user"


def test_assessment_failure_shows_no_fabricated_signal_table():
    """Nothing was observed, so nothing may be presented as observed."""
    html = ui.lead_detail(_routed(failing()))
    assert "Signals observed" not in html


def test_genuine_ambiguity_still_renders_its_own_reason():
    genuine = valid_payload(is_potential_buyer="ambiguous",
                            ambiguity_reason="fellow agency owner researching the market")
    row = _routed(FakeProvider([genuine]))
    assert row["recommendation"] == "REVIEW"
    assert row["assessment_failed"] is False
    assert row["failure_code"] is None

    html = ui.lead_detail(row)
    assert "researching the market" in html
    assert "could not be assessed automatically" not in html
    assert "Signals observed" in html          # a real classification has signals


def test_failure_does_not_leak_into_the_export():
    csv = pipeline.export_csv([_routed(failing())]).decode("utf-8")
    for marker in LEAK_MARKERS:
        assert marker not in csv, f"{marker!r} reached the exported CSV"
    assert "assessment_failed" in csv.splitlines()[0]


def test_scored_lead_is_not_marked_as_failed():
    row = _routed(FakeProvider([valid_payload()]))
    assert row["assessment_failed"] is False
    assert row["recommendation"] in ("CONTACT_NOW", "NURTURE", "DISQUALIFY")


def test_one_failure_does_not_stop_the_rest_of_the_run():
    """Partial failure must leave the other leads intact."""
    import pandas as pd
    from data_cleaning import CANONICAL_COLUMNS

    def row(lead_id, notes):
        r = {c: "" for c in CANONICAL_COLUMNS}
        r.update(lead_id=lead_id, created="2024-06-07", name="X", email="x@y.co",
                 company="Co", title="Owner", source="event", notes=notes)
        return r

    class Flaky:
        name, model = "t", "t"
        seen = 0

        def extract(self, _s, user_message, _j):
            Flaky.seen += 1
            if "boom" in user_message:
                raise ProviderError("rate limited", code="rate_limited", detail=REAL_429)
            return ProviderResponse(text=json.dumps(valid_payload()))

    frame = pd.DataFrame([
        row("L-1", "We're a marketing agency, 26 people. Budget approved, ASAP."),
        row("L-2", "boom - this one fails"),
        row("L-3", "We're a growth agency, 30 people. Budget approved, ASAP."),
    ]).astype(str)

    out = pipeline.run_pipeline(frame, provider=Flaky(), cache=ExtractionCache(path=None),
                                rpm=0, sleep=no_sleep)
    assert out["summary"]["total_processed"] == 3
    assert out["failed"] == 1
    failed = [r for r in out["rows"] if r["assessment_failed"]]
    assert len(failed) == 1 and failed[0]["lead_id"] == "L-2"
    assert all(not r["assessment_failed"] for r in out["rows"] if r["lead_id"] != "L-2")


# --------------------------------------------------------------------------- #
# No Python internals on screen
# --------------------------------------------------------------------------- #

def test_the_generic_failure_handler_renders_no_python_internals():
    """A stale-module reload once put "run_pipeline() got an unexpected keyword
    argument 'profile'" in front of a user. Function names, signatures and
    exception classes are internals, exactly like a provider's 429 body."""
    import ast

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tree = ast.parse(open(os.path.join(root, "app.py"), encoding="utf-8").read())

    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        rendered = [
            n for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "markdown"
        ]
        for call in rendered:
            for arg in ast.walk(call):
                # An f-string interpolating the caught exception is the leak.
                if isinstance(arg, ast.FormattedValue):
                    names = {n.id for n in ast.walk(arg) if isinstance(n, ast.Name)}
                    assert not (names & {"exc", "e", "err", "detail"}), (
                        "exception text is interpolated into user-facing markup"
                    )
