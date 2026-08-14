"""Anthropic provider - working alternate backend.

Not deleted or stubbed when Gemini became active: it is the same code that ran
before, moved behind the shared interface. Selecting it is a one-line change:

    LLM_PROVIDER=anthropic

It takes the canonical schema unmodified (Anthropic's structured outputs accept
`additionalProperties: false` and `anyOf` nullable unions directly), which is
why only the Gemini provider carries a schema adapter.
"""

from __future__ import annotations

import threading
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

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "medium"
DEFAULT_MAX_TOKENS = 8192


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        effort: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 5,
    ):
        self.model = model or _setting("ANTHROPIC_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
        self.effort = effort or _setting("ANTHROPIC_EFFORT", DEFAULT_EFFORT) or DEFAULT_EFFORT
        self.max_tokens = max_tokens
        self._max_retries = max_retries
        self._api_key = api_key or _setting("ANTHROPIC_API_KEY")
        self._client = None
        self._client_lock = threading.Lock()

    @property
    def client(self):
        """One client per provider, built once however many threads ask at once.

        Same double-checked construction as the Gemini provider, for the same
        reason: an unsynchronised lazy init let every worker build its own
        client, and the orphaned ones were closed by the garbage collector
        while other threads were still using them. This backend is not the
        active one, but it carried the identical defect and would have shown
        the identical symptom the moment it was selected.
        """
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = self._build_client()
        return self._client

    def _build_client(self):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ProviderConfigError(
                "anthropic is not installed. Run: pip install anthropic"
            ) from exc

        # A missing key is not fatal here: the SDK also resolves an
        # `ant auth login` profile on its own.
        if self._api_key:
            return anthropic.Anthropic(api_key=self._api_key, max_retries=self._max_retries)
        return anthropic.Anthropic(max_retries=self._max_retries)

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def extract(
        self,
        system_prompt: str,
        user_message: str,
        json_schema: dict[str, Any],
    ) -> ProviderResponse:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                # Static across the run, so the breakpoint bills it once rather
                # than once per lead.
                system=[{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                thinking={"type": "adaptive"},
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": json_schema},
                },
                messages=[{"role": "user", "content": user_message}],
            )
        except ProviderConfigError:
            raise
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {exc}"
            lowered = detail.lower()
            if "429" in detail or "rate" in lowered and "limit" in lowered:
                code = CODE_RATE_LIMITED
            elif "timeout" in lowered or "timed out" in lowered:
                code = CODE_TIMEOUT
            else:
                code = CODE_PROVIDER_ERROR
            retry_after = getattr(getattr(exc, "response", None), "headers", {})
            try:
                retry_after = float(retry_after.get("retry-after")) if retry_after else None
            except (TypeError, ValueError):
                retry_after = None
            raise ProviderError(
                f"anthropic request failed ({code})",
                code=code, retry_after=retry_after, detail=detail,
            ) from exc

        usage = getattr(response, "usage", None)
        tokens = {
            "input_tokens": _int(getattr(usage, "input_tokens", 0)),
            "output_tokens": _int(getattr(usage, "output_tokens", 0)),
            "cached_tokens": _int(getattr(usage, "cache_read_input_tokens", 0)),
        }

        if getattr(response, "stop_reason", None) == "refusal":
            return ProviderResponse(
                text=None, refused=True,
                refusal_reason="model declined to classify this lead", **tokens,
            )

        # Thinking blocks precede text, so select by type rather than position.
        text = next(
            (b.text for b in getattr(response, "content", [])
             if getattr(b, "type", None) == "text"),
            None,
        )
        if not text:
            raise ProviderError(
                "anthropic returned an empty response",
                code=CODE_PROVIDER_ERROR, detail="response contained no text block",
            )

        return ProviderResponse(text=text, **tokens)


def _int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0
