"""Gemini provider - the active backend.

Uses the current Interactions API (`client.interactions.create`), not the older
`generate_content()` surface.

## Schema adaptation, not schema change

Gemini's structured-output schema is an OpenAPI 3.0 subset. It expresses
nullability as `nullable: true` on the field rather than as
`anyOf: [{type: X}, {type: "null"}]`, and it does not use `additionalProperties`.
So the canonical schema in schema.py is translated to that dialect on the way
out by `to_gemini_schema()`.

This is a wire-format translation, not a contract change - the field names, the
enum values and the nullability semantics are identical, and every response is
validated against the canonical contract on the way back in by
`schema.validate_extraction()`. Anthropic gets the canonical schema untouched.
If a future provider genuinely could not express part of the contract, that
would be a real incompatibility and belongs in a conversation, not in a silent
simplification here.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from providers import (
    CODE_PROVIDER_ERROR,
    CODE_RATE_LIMITED,
    CODE_TIMEOUT,
    ProviderConfigError,
    ProviderError,
    ProviderResponse,
    _setting,
)

DEFAULT_MODEL = "gemini-3.5-flash-lite"

# Interaction.status values that mean "no usable output, but trying again may work".
_RETRYABLE_STATUSES = {"failed", "incomplete", "cancelled", "budget_exceeded"}

# Gemini reports its own requested wait inside the 429 body, e.g.
# "Please retry in 33.514422342s." Honouring that beats guessing a backoff.
_RETRY_AFTER_RE = re.compile(r"retry(?:\s+in|[-_ ]after)[:\s]*([0-9]+(?:\.[0-9]+)?)\s*s", re.I)

_RATE_LIMIT_MARKERS = ("429", "resource_exhausted", "quota", "rate limit", "ratelimit")
_TIMEOUT_MARKERS = ("timeout", "timed out", "deadline")


def _classify(text: str) -> tuple[str, Optional[float]]:
    """Map a vendor error string onto a failure code plus any requested wait.

    Classification happens here, at the vendor boundary, so that exactly one
    place in the system has to understand Gemini's error vocabulary - and so
    the raw text can be dropped immediately afterwards.
    """
    lowered = text.lower()
    if any(marker in lowered for marker in _RATE_LIMIT_MARKERS):
        match = _RETRY_AFTER_RE.search(text)
        return CODE_RATE_LIMITED, float(match.group(1)) if match else None
    if any(marker in lowered for marker in _TIMEOUT_MARKERS):
        return CODE_TIMEOUT, None
    return CODE_PROVIDER_ERROR, None


def to_gemini_schema(node: Any) -> Any:
    """Translate canonical JSON Schema into Gemini's OpenAPI-subset dialect.

    * `anyOf: [{type: X}, {type: "null"}]`  ->  `{type: X, nullable: true}`
    * `additionalProperties` is dropped (not part of the dialect)
    * `propertyOrdering` is emitted from `required`, which materially improves
      field-ordering stability in Gemini's structured output
    """
    if isinstance(node, list):
        return [to_gemini_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    # Collapse a nullable union into the dialect's `nullable` flag.
    if "anyOf" in node:
        options = node["anyOf"]
        non_null = [o for o in options if o.get("type") != "null"]
        has_null = len(non_null) != len(options)
        if has_null and len(non_null) == 1:
            collapsed = {**to_gemini_schema(non_null[0]), "nullable": True}
            for key in ("description", "title"):
                if key in node:
                    collapsed[key] = node[key]
            return collapsed
        return {**{k: v for k, v in node.items() if k != "anyOf"},
                "anyOf": [to_gemini_schema(o) for o in non_null or options]}

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "additionalProperties":
            continue
        out[key] = to_gemini_schema(value)

    if out.get("type") == "object" and isinstance(out.get("properties"), dict):
        required = node.get("required") or list(out["properties"])
        out["propertyOrdering"] = [k for k in required if k in out["properties"]]

    return out


class GeminiProvider:
    """Active provider. One `interactions.create` call per lead."""

    name = "gemini"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or _setting("GEMINI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
        self._api_key = api_key or _setting("GEMINI_API_KEY") or _setting("GOOGLE_API_KEY")
        self._client = None

    # -- client ------------------------------------------------------------- #

    @property
    def client(self):
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self):
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - environment problem
            raise ProviderConfigError(
                "google-genai is not installed. Run: pip install google-genai"
            ) from exc

        if not self._api_key:
            raise ProviderConfigError(
                "GEMINI_API_KEY is not set. Export it, or add it to "
                ".streamlit/secrets.toml (which is gitignored)."
            )
        # Passed explicitly rather than relying on ambient auto-detection, so the
        # resolution order (argument > env > secrets) is the same everywhere.
        return genai.Client(api_key=self._api_key)

    def is_configured(self) -> bool:
        return bool(self._api_key)

    # -- the one vendor call ------------------------------------------------ #

    def extract(
        self,
        system_prompt: str,
        user_message: str,
        json_schema: dict[str, Any],
    ) -> ProviderResponse:
        try:
            interaction = self.client.interactions.create(
                model=self.model,
                input=user_message,
                system_instruction=system_prompt,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": to_gemini_schema(json_schema),
                },
            )
        except ProviderConfigError:
            raise
        except Exception as exc:  # noqa: BLE001 - vendor exceptions are not a stable taxonomy
            detail = f"{type(exc).__name__}: {exc}"
            if _looks_like_refusal(exc):
                # Refusal reasons are internal too - the UI renders its own copy.
                return ProviderResponse(text=None, refused=True, refusal_reason=detail[:300])
            code, retry_after = _classify(detail)
            # The message is generic on purpose; the vendor text lives in
            # `detail`, which only ever reaches the logs.
            raise ProviderError(
                f"gemini request failed ({code})",
                code=code, retry_after=retry_after, detail=detail,
            ) from exc

        return self._read(interaction)

    def _read(self, interaction) -> ProviderResponse:
        usage = getattr(interaction, "usage", None)
        tokens = {
            "input_tokens": _int(getattr(usage, "total_input_tokens", 0)),
            "output_tokens": _int(getattr(usage, "total_output_tokens", 0)),
            "cached_tokens": _int(getattr(usage, "total_cached_tokens", 0)),
        }

        status = str(getattr(interaction, "status", "") or "")
        text = getattr(interaction, "output_text", None)

        if status in _RETRYABLE_STATUSES and not text:
            errors = getattr(interaction, "errors", None) or []
            detail = "; ".join(str(getattr(e, "message", e)) for e in errors) or status
            # A content-policy block is a decision, not a transient fault, so it is
            # reported as a refusal rather than raised for retry.
            if _mentions_safety(detail):
                return ProviderResponse(
                    text=None, refused=True, refusal_reason=detail[:300], **tokens
                )
            code, retry_after = _classify(detail)
            raise ProviderError(
                f"gemini interaction {status} ({code})",
                code=code, retry_after=retry_after,
                detail=f"interaction {status}: {detail}",
            )

        if not text:
            raise ProviderError(
                "gemini returned an empty response",
                code=CODE_PROVIDER_ERROR,
                detail=f"interaction returned no output_text (status={status!r})",
            )

        return ProviderResponse(text=text, **tokens)


def _int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _mentions_safety(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in ("safety", "blocked", "prohibited", "policy"))


def _looks_like_refusal(exc: Exception) -> bool:
    return _mentions_safety(f"{type(exc).__name__}: {exc}")
