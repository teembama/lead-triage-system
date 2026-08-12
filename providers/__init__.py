"""Provider abstraction for the extraction call.

`llm_extraction.py` imports only from this package, never from a vendor SDK, so
swapping Gemini for Anthropic (or adding Groq) is a configuration change rather
than a code change.

## Where the boundary sits

A provider owns exactly one thing: turning (system prompt, user message, JSON
schema) into raw response text from some vendor's API. It does **not** parse,
validate, retry, cache or build fallbacks.

That split is deliberate. If each provider returned a finished, validated
extraction, then every provider would carry its own copy of the retry and
validation logic, and Gemini and Anthropic could silently drift apart in how
strictly they enforce the contract. Keeping parse/validate/retry in the pipeline
means there is exactly one implementation of the rules, and a new provider
cannot weaken them.

(This is a deliberate deviation from a `extract(notes, context) -> ExtractionResult`
provider interface, for the reason above. The vendor-facing surface is still a
single method with a single return type.)

## What a provider must guarantee

* Return `ProviderResponse.text` containing a JSON document, or raise.
* Raise `ProviderError` for anything retryable (timeouts, rate limits, 5xx).
* Set `refused=True` rather than raising when the vendor declines the content -
  a refusal is a decision, and retrying it just burns quota.

No provider decides anything about a lead. The model reads text and classifies
it; scoring, qualification and recommendation live in Python that does not exist
yet, and no provider may anticipate them.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

DEFAULT_PROVIDER = "gemini"


@dataclass
class ProviderResponse:
    """Raw result of one vendor call. Deliberately dumb - no interpretation."""

    text: Optional[str]
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    refused: bool = False
    refusal_reason: Optional[str] = None


# Machine-readable failure codes. These are the only failure vocabulary the rest
# of the system sees; vendor text never travels past this boundary.
CODE_RATE_LIMITED = "rate_limited"
CODE_TIMEOUT = "timeout"
CODE_INVALID_RESPONSE = "invalid_response"
CODE_REFUSED = "refused"
CODE_PROVIDER_ERROR = "provider_error"

FAILURE_CODES = (
    CODE_RATE_LIMITED, CODE_TIMEOUT, CODE_INVALID_RESPONSE,
    CODE_REFUSED, CODE_PROVIDER_ERROR,
)


class ProviderError(RuntimeError):
    """A retryable transport/vendor failure (timeout, rate limit, 5xx).

    Carries a machine-readable `code` so the retry logic can branch without
    string-matching, and keeps the vendor's own text in `detail` - which is for
    logs only. Nothing downstream reads `detail`, and nothing user-facing may
    ever render it: putting a stringified vendor exception into the extraction
    record is exactly the bug this type exists to prevent.

    `retry_after` is the provider's own requested wait in seconds, when it tells
    us. Honouring it beats guessing at a backoff curve.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = CODE_PROVIDER_ERROR,
        retry_after: Optional[float] = None,
        detail: Optional[str] = None,
    ):
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after
        self.detail = detail or message


class ProviderConfigError(RuntimeError):
    """Provider cannot be constructed: missing SDK, missing key, unknown name."""


@runtime_checkable
class LLMProvider(Protocol):
    """The whole vendor-facing surface of the system."""

    name: str
    model: str

    def extract(
        self,
        system_prompt: str,
        user_message: str,
        json_schema: dict[str, Any],
    ) -> ProviderResponse:
        """Send one classification request and return the raw response text."""
        ...


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Anchored to this file, not the working directory: a relative path resolves
# against wherever the process happens to have been started, so a run launched
# from another directory would silently find no credentials.
_SECRETS_PATH = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"


@lru_cache(maxsize=1)
def _secrets_file() -> dict[str, Any]:
    """Parse .streamlit/secrets.toml directly.

    Deliberately not routed through `st.secrets`: that requires Streamlit to be
    installed, and this package must resolve credentials from plain pytest, a
    CLI script or a cron job where it is not. secrets.toml is ordinary TOML and
    `tomllib` is stdlib on 3.11+, so reading it needs no third-party dependency.
    """
    try:
        with _SECRETS_PATH.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _setting(name: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve a setting: environment, then secrets.toml, then the default.

    Environment wins, so a shell export overrides a committed secrets file.
    """
    value = os.environ.get(name)
    if value:
        return value

    secret = _secrets_file().get(name)
    if secret:
        return str(secret)

    # Streamlit's own store, for deployments where secrets are injected by the
    # platform rather than written to disk (Community Cloud's Secrets panel).
    try:
        import streamlit as st

        injected = st.secrets.get(name)
        if injected:
            return str(injected)
    except Exception:
        pass

    return default


def resolve_provider_name(explicit: Optional[str] = None) -> str:
    return (explicit or _setting("LLM_PROVIDER", DEFAULT_PROVIDER) or DEFAULT_PROVIDER).strip().lower()


def get_provider(
    name: Optional[str] = None,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMProvider:
    """Construct the configured provider.

    Imports are deferred to the selected branch so that having only one vendor
    SDK installed is not an error - a Gemini-only environment must not need the
    Anthropic package present, and vice versa.
    """
    resolved = resolve_provider_name(name)

    if resolved == "gemini":
        from providers.gemini_provider import GeminiProvider

        return GeminiProvider(model=model, api_key=api_key)

    if resolved == "anthropic":
        from providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(model=model, api_key=api_key)

    raise ProviderConfigError(
        f"Unknown LLM_PROVIDER {resolved!r}. Supported: 'gemini', 'anthropic'."
    )


def available_providers() -> tuple[str, ...]:
    return ("gemini", "anthropic")


__all__ = [
    "LLMProvider",
    "ProviderResponse",
    "ProviderError",
    "ProviderConfigError",
    "get_provider",
    "resolve_provider_name",
    "available_providers",
    "DEFAULT_PROVIDER",
    "FAILURE_CODES",
    "CODE_RATE_LIMITED",
    "CODE_TIMEOUT",
    "CODE_INVALID_RESPONSE",
    "CODE_REFUSED",
    "CODE_PROVIDER_ERROR",
]
