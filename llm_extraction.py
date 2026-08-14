"""Stage 4 - the single LLM call per lead, and the batch machinery around it.

This module contains no vendor SDK code. It talks to `providers.LLMProvider`,
so the active backend is a configuration choice (`LLM_PROVIDER=gemini`).

What this layer owns, once, for every provider: prompt assembly, JSON parsing,
contract validation, the corrective retry, the all-unknown fallback, the
content cache, and concurrency. Keeping those here rather than in each provider
means a new backend cannot accidentally relax the rules.

What it deliberately does NOT do: score anything, bucket anything, resolve which
of two conflicting facts wins, or decide what should happen to a lead. It
extracts facts and signals. Every downstream decision belongs to Python modules
that do not exist yet.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import prompts
from providers import (
    CODE_INVALID_RESPONSE,
    CODE_PROVIDER_ERROR,
    CODE_RATE_LIMITED,
    CODE_REFUSED,
    LLMProvider,
    ProviderConfigError,
    ProviderError,
    _setting,
    get_provider,
    resolve_provider_name,
)
from schema import build_json_schema, unknown_extraction, validate_extraction

log = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = 20
DEFAULT_CACHE_PATH = Path(".cache") / "extractions.json"

# Requests per minute the provider will accept. 15 is the Gemini free-tier
# limit, so it is the safe default; a paid tier raises this through LLM_RPM
# with no code change.
DEFAULT_RPM = 15.0

# Observed mean call latency, used only to size the worker pool against the RPM
# budget. Workers beyond what the quota can absorb add no throughput - they just
# queue on the limiter.
#
# 3.05 is the median call latency measured across ~900 live calls in a
# concurrency sweep (1, 8, 12, 17 and 20 workers against 100 leads). Latency is
# flat with respect to concurrency - median 2.99-3.33s at every level - so one
# constant is honest here.
#
# It replaces 4.25, which came from five serial calls whose first carried client
# construction and skewed the mean. That over-provisioned the pool: at 240 rpm it
# derived 17 workers, which measured only ~7% more throughput than 12 while
# spending ~104s per run blocked on the limiter, against ~13s at 12.
#
# The median rather than the mean, because the mean is dragged by a thin tail
# (p95 ~5.2s) that describes stragglers rather than the rate the pool sustains.
ASSUMED_CALL_SECONDS = 3.05

# Transport failures (429, timeout, connection) get several attempts with
# exponential backoff. Contract failures (malformed JSON, schema violation) keep
# the single corrective retry the architecture specifies in §16 Stage 4.
MAX_TRANSPORT_ATTEMPTS = 4
MAX_CONTRACT_ATTEMPTS = 2
RETRY_BASE_DELAY = 2.0
RETRY_MAX_DELAY = 60.0
RETRY_JITTER = 0.25

_SCHEMA = build_json_schema()


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #

class RateLimiter:
    """Token bucket shared by every worker thread in a run.

    Concurrency is kept - workers still run in parallel - but they draw from one
    budget, so the pool can no longer demand 123 requests/minute against a
    15/minute quota and turn a slow run into a failing one.

    `pause()` exists because a 429 is a run-wide fact, not a per-worker one.
    Without it, one worker learns it must wait 33 seconds while the other seven
    immediately re-discover the same 429 and burn quota confirming it.
    """

    def __init__(self, rpm: float = DEFAULT_RPM):
        self.rpm = max(0.0, float(rpm))
        self.rate = self.rpm / 60.0                       # tokens per second
        self.capacity = max(1.0, self.rate)               # never below one call
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._paused_until = 0.0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.rate > 0

    def acquire(self, sleep: Callable[[float], None] = time.sleep) -> None:
        """Block until this thread may issue one request."""
        if not self.enabled:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                if now < self._paused_until:
                    wait = self._paused_until - now
                else:
                    elapsed = now - self._updated
                    self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                    self._updated = now
                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        return
                    wait = (1.0 - self._tokens) / self.rate
            sleep(max(wait, 0.0))

    def pause(self, seconds: float) -> None:
        """Hold every worker for `seconds` - used when the provider says to wait."""
        if seconds <= 0:
            return
        with self._lock:
            self._paused_until = max(self._paused_until, time.monotonic() + seconds)


def rpm_setting(explicit: Optional[float] = None) -> float:
    if explicit is not None:
        return max(0.0, float(explicit))
    raw = _setting("LLM_RPM")
    try:
        return max(0.0, float(raw)) if raw else DEFAULT_RPM
    except (TypeError, ValueError):
        return DEFAULT_RPM


def workers_for_rpm(rpm: float, cap: int = DEFAULT_MAX_WORKERS) -> int:
    """Workers needed to keep an RPM budget saturated, never more than `cap`.

    At 15 RPM one worker already saturates the quota, so the historical eight
    were seven threads' worth of 429s.
    """
    if rpm <= 0:
        return cap
    return max(1, min(cap, math.ceil(rpm * ASSUMED_CALL_SECONDS / 60.0)))


def _backoff_delay(exc: ProviderError, attempt: int) -> float:
    """Provider's own requested wait when offered, else exponential, plus jitter.

    Jitter matters with a shared pool: without it, every worker that trips the
    same 429 wakes at the same instant and trips it again together.
    """
    base = exc.retry_after if exc.retry_after else RETRY_BASE_DELAY * (2 ** (attempt - 1))
    base = min(float(base), RETRY_MAX_DELAY)
    return base + random.uniform(0.0, base * RETRY_JITTER)


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #

@dataclass
class ExtractionResult:
    lead_id: str
    extraction: dict[str, Any]
    status: str                      # ok | recovered | failed
    provider: str = ""
    model: str = ""
    attempts: int = 1
    from_cache: bool = False
    # `error` holds vendor detail for logs and debugging only. It must never be
    # rendered: the UI reads `failure_code`, which is a closed vocabulary.
    error: Optional[str] = None
    failure_code: Optional[str] = None
    problems: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    @property
    def needs_review(self) -> bool:
        """`failed` never means 'drop the lead' - it means a human should look."""
        return self.status == "failed"


def max_workers_setting(explicit: Optional[int] = None) -> int:
    """Resolved like every other setting: environment, then secrets.toml, then
    Streamlit's injected store.

    It previously read `os.environ` alone, which made it the one setting that
    could not be configured on a Streamlit deployment - the Secrets panel writes
    to `st.secrets`, not to the environment, so an operator setting it there got
    silence rather than an effect.
    """
    if explicit:
        return max(1, explicit)
    raw = _setting("LLM_MAX_WORKERS")
    try:
        return max(1, int(raw)) if raw else DEFAULT_MAX_WORKERS
    except (TypeError, ValueError):
        return DEFAULT_MAX_WORKERS


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #

class ExtractionCache:
    """Maps a lead's notes (plus the context that reaches the model) to a result.

    Keyed on content rather than lead_id, so repeated note templates in the
    dataset collapse to a single API call. Provider, model, prompt version and
    schema version are all part of the key: changing any of them must not serve
    a response produced under the old configuration.
    """

    def __init__(self, path: Optional[Path] = DEFAULT_CACHE_PATH, enabled: bool = True):
        self.path = Path(path) if path else None
        self.enabled = enabled
        self._lock = threading.Lock()
        self._store: dict[str, dict[str, Any]] = {}
        if self.enabled and self.path and self.path.exists():
            try:
                self._store = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._store = {}   # a corrupt cache is a cold cache, not a crash

    @staticmethod
    def key(lead: dict[str, Any], provider: str, model: str, profile: Any = None) -> str:
        # `company_clean` is deliberately excluded. It was making every key
        # unique - 509 keys for 509 leads, so the cache never hit once on the
        # assessment file - while contributing nothing to the classification:
        # no scored factor reads the company name, and the model is instructed
        # to classify from the notes. Dropping it collapses 509 calls to 446.
        payload = "|".join([
            prompts.cache_signature(profile),
            provider,
            model,
            (lead.get("notes_clean") or "").strip(),
            str(lead.get("title_clean") or ""),
            str(lead.get("employees_display") or ""),
            str(lead.get("budget_display") or ""),
        ])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Returns a deep copy: callers must never hold a reference into the store."""
        if not self.enabled:
            return None
        with self._lock:
            entry = self._store.get(key)
        return json.loads(json.dumps(entry)) if entry else None

    def put(self, key: str, extraction: dict[str, Any]) -> None:
        """Stores a deep copy, so a later mutation by the caller cannot reach in."""
        if not self.enabled:
            return
        snapshot = json.loads(json.dumps(extraction))
        with self._lock:
            self._store[key] = snapshot

    def save(self) -> None:
        if not (self.enabled and self.path):
            return
        with self._lock:
            snapshot = json.loads(json.dumps(self._store))
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(snapshot, indent=1), encoding="utf-8")
        except OSError:
            pass   # a cache we cannot persist is still a cache for this run

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# --------------------------------------------------------------------------- #
# Single extraction
# --------------------------------------------------------------------------- #

def extract_signals(
    lead: dict[str, Any],
    provider: LLMProvider,
    *,
    cache: Optional[ExtractionCache] = None,
    limiter: Optional["RateLimiter"] = None,
    profile: Any = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ExtractionResult:
    """Classify one lead. Never raises - failures come back as a flagged record."""
    lead_id = str(lead.get("lead_id") or "unknown")
    provider_name = getattr(provider, "name", "unknown")
    model = getattr(provider, "model", "unknown")

    # `is not None`, not truthiness: ExtractionCache defines __len__, so an empty
    # cache is falsy and a bare `if cache` would disable caching for the whole run.
    cache_key = (ExtractionCache.key(lead, provider_name, model, profile)
                 if cache is not None else None)
    if cache_key:
        hit = cache.get(cache_key)
        if hit is not None:
            return ExtractionResult(
                lead_id, hit, status="ok", provider=provider_name,
                model=model, from_cache=True,
            )

    base_message = prompts.build_user_message(lead, profile)
    problems: list[str] = []
    last_detail: Optional[str] = None
    failure_code: Optional[str] = CODE_PROVIDER_ERROR
    tokens = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}

    schema_for_run = build_json_schema(profile) if profile is not None else _SCHEMA
    transport_tries = 0     # 429 / timeout / connection - backed off and retried
    contract_tries = 0      # malformed or schema-violating response - corrected once
    calls = 0

    while True:
        message = base_message if not problems else base_message + prompts.build_retry_suffix(problems)

        if limiter is not None:
            limiter.acquire(sleep)
        calls += 1

        try:
            response = provider.extract(prompts.SYSTEM_PROMPT, message, schema_for_run)
        except ProviderConfigError:
            raise                                   # misconfiguration is not per-lead
        except ProviderError as exc:
            transport_tries += 1
            failure_code, last_detail = exc.code, exc.detail
            # Vendor detail goes to the log and nowhere else.
            log.warning(
                "extraction transport failure lead=%s code=%s attempt=%d/%d: %s",
                lead_id, exc.code, transport_tries, MAX_TRANSPORT_ATTEMPTS, exc.detail,
            )
            if transport_tries >= MAX_TRANSPORT_ATTEMPTS:
                break
            delay = _backoff_delay(exc, transport_tries)
            if exc.code == CODE_RATE_LIMITED and limiter is not None:
                # Run-wide fact: hold every worker, not just this one.
                limiter.pause(delay)
            sleep(delay)
            continue
        except Exception as exc:                    # noqa: BLE001 - batch must survive
            transport_tries += 1
            failure_code = CODE_PROVIDER_ERROR
            last_detail = f"{type(exc).__name__}: {exc}"
            log.warning("extraction error lead=%s attempt=%d: %s",
                        lead_id, transport_tries, last_detail)
            if transport_tries >= MAX_TRANSPORT_ATTEMPTS:
                break
            sleep(min(RETRY_BASE_DELAY * (2 ** (transport_tries - 1)), RETRY_MAX_DELAY))
            continue

        tokens = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cached_tokens": response.cached_tokens,
        }

        if response.refused:
            failure_code = CODE_REFUSED
            last_detail = response.refusal_reason or "provider declined this lead"
            log.warning("extraction refused lead=%s: %s", lead_id, last_detail)
            break                                   # a retry would refuse again

        parsed, parse_error = _parse_json(response.text)
        if parse_error:
            contract_tries += 1
            failure_code, last_detail = CODE_INVALID_RESPONSE, parse_error
            problems = [parse_error]
            if contract_tries >= MAX_CONTRACT_ATTEMPTS:
                break
            continue

        problems = validate_extraction(parsed)
        if not problems:
            if cache_key:
                cache.put(cache_key, parsed)
            return ExtractionResult(
                lead_id, parsed,
                status="ok" if calls == 1 else "recovered",
                provider=provider_name, model=model, attempts=calls, **tokens,
            )

        contract_tries += 1
        failure_code, last_detail = CODE_INVALID_RESPONSE, "; ".join(problems)
        if contract_tries >= MAX_CONTRACT_ATTEMPTS:
            break

    # Repeated failure degrades to a valid all-unknown record flagged for review.
    # Nothing is fabricated, the lead is not dropped, and this is never cached.
    # The record itself carries no vendor text - only `error` does, for logs.
    return ExtractionResult(
        lead_id, unknown_extraction(), status="failed",
        provider=provider_name, model=model, attempts=calls,
        error=last_detail or "extraction failed",
        failure_code=failure_code or CODE_PROVIDER_ERROR,
        problems=problems, **tokens,
    )


def _parse_json(text: Optional[str]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if not text or not text.strip():
        return None, "response contained no text"
    body = text.strip()
    # Some models wrap JSON in a markdown fence even when asked not to.
    if body.startswith("```"):
        body = body.split("\n", 1)[-1]
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3]
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        return None, f"response was not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "response JSON was not an object"
    return parsed, None


# --------------------------------------------------------------------------- #
# Batch
# --------------------------------------------------------------------------- #

def extract_batch(
    leads: list[dict[str, Any]],
    provider: Optional[LLMProvider] = None,
    *,
    max_workers: Optional[int] = None,
    cache: Optional[ExtractionCache] = None,
    rpm: Optional[float] = None,
    limiter: Optional[RateLimiter] = None,
    profile: Any = None,
    sleep: Callable[[float], None] = time.sleep,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[ExtractionResult]:
    """Classify many leads concurrently, returning results in input order.

    Concurrency is preserved; what changed is that every worker now draws from
    one shared RPM budget, so the pool cannot outrun the provider's quota.

    Demo mode and Full mode differ only in the length of `leads` - both call this
    same function, so what is tuned on a handful of rows is what runs on all of
    them.
    """
    if not leads:
        return []

    provider = provider or get_provider()
    cache = cache if cache is not None else ExtractionCache()

    effective_rpm = rpm_setting(rpm)
    if limiter is None:
        limiter = RateLimiter(effective_rpm)

    results: list[Optional[ExtractionResult]] = [None] * len(leads)
    completed = 0
    lock = threading.Lock()

    def run(index: int) -> None:
        nonlocal completed
        results[index] = extract_signals(leads[index], provider, cache=cache,
                                         limiter=limiter, profile=profile, sleep=sleep)
        # Advanced only once the lead has actually finished, never on submission.
        with lock:
            completed += 1
            if progress_callback:
                progress_callback(completed, len(leads))

    # Explicit LLM_MAX_WORKERS still wins; otherwise the pool is sized to what
    # the RPM budget can actually absorb rather than a fixed cap.
    if max_workers is not None or _setting("LLM_MAX_WORKERS"):
        workers = max_workers_setting(max_workers)
    else:
        workers = workers_for_rpm(effective_rpm)
    workers = max(1, min(workers, len(leads)))

    log.info("extracting %d leads: %d workers, %.0f rpm", len(leads), workers, effective_rpm)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, range(len(leads))))

    cache.save()

    # Index-aligned with `leads`, always, and never filtered. This is
    # load-bearing: callers pair the two lists positionally, so dropping an
    # entry would silently attach one lead's extraction to a different lead.
    # `extract_signals` never raises, so a gap here means a worker died - which
    # is worth failing loudly for rather than papering over by shifting rows up.
    missing = [index for index, result in enumerate(results) if result is None]
    if missing:
        raise RuntimeError(f"extraction produced no result for lead index {missing[0]}")
    return [result for result in results if result is not None]


def summarise_batch(results: list[ExtractionResult]) -> dict[str, Any]:
    """Run-level counts, for the UI and for sanity-checking a run.

    Counts extraction outcomes only. It reports how many leads the model judged
    to be potential buyers; it does not rank them, score them, or suggest what to
    do about any of them.
    """
    return {
        "total": len(results),
        "ok": sum(1 for r in results if r.status == "ok"),
        "recovered": sum(1 for r in results if r.status == "recovered"),
        "failed": sum(1 for r in results if r.status == "failed"),
        "from_cache": sum(1 for r in results if r.from_cache),
        "api_calls": sum(r.attempts for r in results if not r.from_cache),
        "input_tokens": sum(r.input_tokens for r in results),
        "output_tokens": sum(r.output_tokens for r in results),
        "cached_tokens": sum(r.cached_tokens for r in results),
        "buyer_yes": sum(1 for r in results if r.extraction.get("is_potential_buyer") == "yes"),
        "buyer_no": sum(1 for r in results if r.extraction.get("is_potential_buyer") == "no"),
        "buyer_ambiguous": sum(
            1 for r in results if r.extraction.get("is_potential_buyer") == "ambiguous"
        ),
    }


def active_provider_name() -> str:
    return resolve_provider_name()
