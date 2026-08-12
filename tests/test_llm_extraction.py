"""Stage 4 offline tests - contract, provider abstraction, cache, retries.

No API key and no network: providers are replaced with fakes, so these run free
and fast on every edit. Live model behaviour is covered separately in
test_extraction_live.py.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm_extraction  # noqa: E402
import prompts  # noqa: E402
import providers  # noqa: E402
import schema  # noqa: E402
from llm_extraction import ExtractionCache, extract_signals, summarise_batch  # noqa: E402
from providers import ProviderError, ProviderResponse  # noqa: E402
from providers.gemini_provider import GeminiProvider, to_gemini_schema  # noqa: E402


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

class FakeProvider:
    """Implements the LLMProvider protocol; returns queued items in order.

    A queued item may be a dict (serialised to JSON), a raw string, an Exception
    (raised), or a ProviderResponse (returned as-is).
    """

    name = "fake"
    model = "fake-model-1"

    def __init__(self, items):
        self._items = list(items)
        self.calls = []

    def extract(self, system_prompt, user_message, json_schema):
        self.calls.append({
            "system_prompt": system_prompt,
            "user_message": user_message,
            "json_schema": json_schema,
        })
        item = self._items.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, ProviderResponse):
            return item
        body = item if isinstance(item, str) else json.dumps(item)
        return ProviderResponse(
            text=body, input_tokens=100, output_tokens=50, cached_tokens=900,
        )


def no_sleep(_seconds):
    """Retry backoff is real in production and instant in tests."""


def valid_payload(**overrides):
    payload = {
        "is_potential_buyer": "yes",
        "buyer_type": "business_prospect",
        "non_buyer_signal": "none",
        "disqualification_evidence": None,
        "ambiguity_reason": None,
        "industry_fit": {"level": "high", "evidence": "influencer marketing agency"},
        "employee_count_in_notes": {
            "stated": True, "low": 26, "high": 26, "kind": "exact", "evidence": "26 people",
        },
        "budget_in_notes": {
            "stated": False, "low": None, "high": None, "kind": "missing",
            "evidence": "not specified",
        },
        "location_in_notes": {"stated": False, "value": None, "evidence": "not specified"},
        "seniority": {"level": "high_influence", "evidence": "VP Growth"},
        "urgency": {"level": "high", "evidence": "start ASAP"},
        "pain_severity": {"level": "high", "evidence": "eating our week"},
        "purchasing_readiness": {"level": "approved", "evidence": "Budget approved"},
        "timeline": {"level": "near_term", "evidence": "ASAP"},
        "buying_stage": {"level": "committed", "evidence": "automated end to end"},
        "price_sensitive_flag": False,
    }
    payload.update(overrides)
    return payload


LEAD = {
    "lead_id": "L-1168",
    "title_clean": "VP Growth",
    "company_clean": "PipeGTM",
    "employees_display": "unknown",
    "budget_display": "6000",
    "notes_clean": "We're a influencer marketing agency, 26 people. Budget approved, ASAP.",
}


def run(lead, provider, **kwargs):
    kwargs.setdefault("sleep", no_sleep)
    return extract_signals(lead, provider, **kwargs)


# --------------------------------------------------------------------------- #
# Schema shape
# --------------------------------------------------------------------------- #

def test_schema_requires_every_property_and_forbids_extras():
    s = schema.build_json_schema()
    assert set(s["required"]) == set(s["properties"])
    assert s["additionalProperties"] is False
    for name, prop in s["properties"].items():
        if prop.get("type") == "object":
            assert prop["additionalProperties"] is False, name
            assert set(prop["required"]) == set(prop["properties"]), name


def test_buyer_status_is_a_three_way_enum_not_a_boolean():
    assert schema.build_json_schema()["properties"]["is_potential_buyer"]["enum"] == [
        "yes", "no", "ambiguous",
    ]


def test_timeline_enum_is_a_single_shared_vocabulary():
    levels = schema.build_json_schema()["properties"]["timeline"]["properties"]["level"]["enum"]
    assert levels == ["near_term", "medium_term", "none"]


def test_employee_count_is_an_interval_fact_not_a_level():
    """Bucketing a headcount is a scoring decision, so the model never sees a band."""
    block = schema.build_json_schema()["properties"]["employee_count_in_notes"]
    assert set(block["properties"]) == {"stated", "low", "high", "kind", "evidence"}
    assert "level" not in block["properties"]
    assert block["properties"]["kind"]["enum"] == ["exact", "range", "floor", "approx", "missing"]


def test_budget_is_an_interval_fact_not_a_level():
    block = schema.build_json_schema()["properties"]["budget_in_notes"]
    assert set(block["properties"]) == {"stated", "low", "high", "kind", "evidence"}
    assert "level" not in block["properties"]


def test_no_scoring_vocabulary_leaks_into_the_contract():
    """Stage 4 must not be able to express a score, a band or a recommendation.

    Scoped to the contract surface - property names and enum values - rather
    than to descriptions, which legitimately quote lead phrases like "priority
    this quarter". `disqualification_evidence` is deliberately allowed: it is
    the evidence for a `no` on the buyer gate, which is a classification, not a
    verdict on what should happen to the lead.
    """
    surface: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"description", "title"}:
                    continue
                if key == "enum":
                    surface.extend(str(v).lower() for v in value)
                    continue
                if key in {"properties"} and isinstance(value, dict):
                    surface.extend(k.lower() for k in value)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema.build_json_schema())
    joined = " ".join(surface)
    for banned in ("score", "points", "contact_now", "nurture", "rank",
                   "priority", "recommendation", "tier", "grade"):
        assert banned not in joined, f"scoring vocabulary {banned!r} leaked into the contract"


def test_no_size_bucket_labels_leak_into_the_contract():
    """The eventual size bands (small / core fit / enterprise) are a later
    concern; Stage 4 emits raw bounds so that layer keeps its precision."""
    measure = schema.build_json_schema()["properties"]["employee_count_in_notes"]
    blob = json.dumps(measure).lower()
    for banned in ("small", "ideal", "core", "enterprise", "viable", "fit"):
        assert banned not in blob, f"bucket label {banned!r} leaked into the schema"


def test_measure_kinds_mirror_the_cleaning_layer_exactly():
    """A notes-stated `50+` and a CRM `50+` must arrive in the same shape."""
    from data_cleaning import parse_employees

    for raw, expected in [("26", "exact"), ("35-55", "range"),
                          ("51+", "floor"), ("~61", "approx"), ("", "missing")]:
        assert parse_employees(raw).status == expected
        assert expected in schema.MEASURE_KINDS


# --------------------------------------------------------------------------- #
# Validation - enums
# --------------------------------------------------------------------------- #

def test_valid_payload_passes():
    assert schema.validate_extraction(valid_payload()) == []


@pytest.mark.parametrize(
    "override,fragment",
    [
        ({"is_potential_buyer": "maybe"}, "is_potential_buyer"),
        ({"is_potential_buyer": True}, "is_potential_buyer"),
        ({"buyer_type": "wizard"}, "buyer_type"),
        ({"timeline": {"level": "within_1_month", "evidence": "2 weeks"}}, "timeline.level"),
        ({"urgency": {"level": "urgent", "evidence": "asap"}}, "urgency.level"),
        ({"seniority": {"level": "boss", "evidence": "owner"}}, "seniority.level"),
        ({"buying_stage": {"level": "browsing", "evidence": "looking"}}, "buying_stage.level"),
        ({"urgency": {"level": "high", "evidence": "  "}}, "urgency.evidence is empty"),
        ({"price_sensitive_flag": "no"}, "price_sensitive_flag"),
    ],
)
def test_enum_and_evidence_violations_are_reported(override, fragment):
    problems = schema.validate_extraction(valid_payload(**override))
    assert any(fragment in p for p in problems), problems


def test_missing_factor_is_reported():
    payload = valid_payload()
    del payload["urgency"]
    assert any("urgency" in p for p in schema.validate_extraction(payload))


# --------------------------------------------------------------------------- #
# Validation - the stated-purpose field
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("signal", schema.NON_BUYER_SIGNALS)
def test_every_allowed_non_buyer_signal_validates(signal):
    evidence = None if signal == "none" else "we're benchmarking our own pricing"
    payload = valid_payload(non_buyer_signal=signal, disqualification_evidence=evidence)
    assert schema.validate_extraction(payload) == []


@pytest.mark.parametrize("bad", ["benchmarking", "NONE", "", None, True, "competitor"])
def test_an_unrecognised_non_buyer_signal_is_rejected(bad):
    problems = schema.validate_extraction(valid_payload(non_buyer_signal=bad))
    assert any("non_buyer_signal" in p for p in problems), problems


def test_a_missing_non_buyer_signal_is_rejected():
    payload = valid_payload()
    del payload["non_buyer_signal"]
    assert any("non_buyer_signal" in p for p in schema.validate_extraction(payload))


@pytest.mark.parametrize("evidence", [None, "", "   "])
def test_a_stated_purpose_without_evidence_is_rejected(evidence):
    """An exclusion the notes cannot be quoted for is an inference, and an
    inference must not be able to disqualify a lead."""
    problems = schema.validate_extraction(valid_payload(
        non_buyer_signal="competitive_research", disqualification_evidence=evidence,
    ))
    assert any("disqualification_evidence" in p for p in problems), problems


def test_signal_none_does_not_require_evidence():
    assert schema.validate_extraction(valid_payload(
        non_buyer_signal="none", disqualification_evidence=None,
    )) == []


def test_the_signal_reaches_the_model_as_a_closed_enum():
    prop = schema.build_json_schema()["properties"]["non_buyer_signal"]
    assert prop["enum"] == list(schema.NON_BUYER_SIGNALS)
    assert "non_buyer_signal" in schema.build_json_schema()["required"]
    # The description must teach the distinction, not just list the values.
    assert "not about what" in prop["description"].lower() or (
        "purpose" in prop["description"].lower())


def test_ambiguous_verdict_must_say_what_is_unresolved():
    payload = valid_payload(is_potential_buyer="ambiguous", ambiguity_reason=None)
    assert any("ambiguity_reason" in p for p in schema.validate_extraction(payload))
    payload["ambiguity_reason"] = "researching the market; intent unclear"
    assert schema.validate_extraction(payload) == []


# --------------------------------------------------------------------------- #
# Validation - employee uncertainty (all five cases)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "block",
    [
        {"stated": True, "low": 26, "high": 26, "kind": "exact", "evidence": "26 people"},
        {"stated": True, "low": 35, "high": 55, "kind": "range", "evidence": "35-55 people"},
        {"stated": True, "low": 51, "high": None, "kind": "floor", "evidence": "51+ people"},
        {"stated": True, "low": 61, "high": 61, "kind": "approx", "evidence": "around 61"},
        {"stated": False, "low": None, "high": None, "kind": "missing", "evidence": "not specified"},
    ],
)
def test_all_five_employee_uncertainty_shapes_are_valid(block):
    assert schema.validate_extraction(valid_payload(employee_count_in_notes=block)) == []


def test_employee_floor_must_not_carry_an_upper_bound():
    """`51+` asserts a lower bound and nothing else. 51-100 is invented data."""
    bad = valid_payload(employee_count_in_notes={
        "stated": True, "low": 51, "high": 100, "kind": "floor", "evidence": "51+ people",
    })
    assert any("floor" in p for p in schema.validate_extraction(bad))


def test_employee_range_keeps_both_bounds_and_rejects_inversion():
    assert schema.validate_extraction(valid_payload(employee_count_in_notes={
        "stated": True, "low": 35, "high": 55, "kind": "range", "evidence": "35-55",
    })) == []
    inverted = valid_payload(employee_count_in_notes={
        "stated": True, "low": 55, "high": 35, "kind": "range", "evidence": "35-55",
    })
    assert any("inverted" in p for p in schema.validate_extraction(inverted))


def test_range_must_not_be_collapsed_to_a_midpoint():
    """A midpoint asserts a precision the source never gave."""
    midpoint = valid_payload(employee_count_in_notes={
        "stated": True, "low": 45, "high": 45, "kind": "range", "evidence": "35-55 people",
    })
    assert any("range" in p for p in schema.validate_extraction(midpoint))


def test_unstated_employee_count_must_not_borrow_the_crm_value():
    payload = valid_payload(employee_count_in_notes={
        "stated": False, "low": 9, "high": 9, "kind": "missing", "evidence": "not specified",
    })
    assert any("nothing stated" in p for p in schema.validate_extraction(payload))


# --------------------------------------------------------------------------- #
# Validation - budget uncertainty (all five cases)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "block",
    [
        {"stated": True, "low": 8500, "high": 8500, "kind": "exact", "evidence": "$8,500"},
        {"stated": True, "low": 6000, "high": 8000, "kind": "range", "evidence": "$6-8k"},
        {"stated": True, "low": 8000, "high": None, "kind": "floor", "evidence": "8k+"},
        {"stated": True, "low": 7000, "high": 7000, "kind": "approx", "evidence": "~7k"},
        {"stated": False, "low": None, "high": None, "kind": "missing", "evidence": "not specified"},
    ],
)
def test_all_five_budget_uncertainty_shapes_are_valid(block):
    assert schema.validate_extraction(valid_payload(budget_in_notes=block)) == []


def test_budget_floor_must_not_carry_an_upper_bound():
    bad = valid_payload(budget_in_notes={
        "stated": True, "low": 8000, "high": 20000, "kind": "floor", "evidence": "8k+",
    })
    assert any("floor" in p for p in schema.validate_extraction(bad))


def test_zero_budget_is_a_stated_fact_not_an_absence():
    """"tiny budget, one-man shop" is evidence; a silent field is not."""
    stated_zero = valid_payload(budget_in_notes={
        "stated": True, "low": 0, "high": 0, "kind": "exact", "evidence": "tiny budget",
    })
    assert schema.validate_extraction(stated_zero) == []


# --------------------------------------------------------------------------- #
# Fallback record
# --------------------------------------------------------------------------- #

def test_unknown_extraction_is_itself_contract_valid():
    """The failure fallback must survive the same contract, or a failed lead
    would crash the next stage instead of being flagged."""
    record = schema.unknown_extraction()
    assert schema.validate_extraction(record) == []
    assert record["is_potential_buyer"] == "ambiguous"
    # The reason is a fixed, user-safe sentence - never caller-supplied text.
    assert record["ambiguity_reason"] == schema.ASSESSMENT_FAILED_REASON


def test_unknown_extraction_states_no_non_buying_purpose():
    """A lead the system never managed to read must not be excluded on the
    strength of that failure. Silence is not evidence of anything."""
    record = schema.unknown_extraction()
    assert record["non_buyer_signal"] == "none"
    assert record["disqualification_evidence"] is None

    from recommendation import recommend

    assert recommend({"is_potential_buyer": record["is_potential_buyer"],
                      "non_buyer_signal": record["non_buyer_signal"],
                      "total_score": 0, "buying_intent_score": 0}) == "REVIEW"


def test_unknown_extraction_invents_no_figures():
    record = schema.unknown_extraction()
    for factor in schema.MEASURE_FACTORS:
        assert record[factor]["stated"] is False
        assert record[factor]["low"] is None and record[factor]["high"] is None


def test_evidence_fields_flattens_every_factor():
    flat = schema.evidence_fields(valid_payload())
    assert flat["urgency"] == "start ASAP"
    assert flat["employee_count_in_notes"] == "26 people"
    assert set(flat) == (set(schema.LEVEL_FACTORS) | set(schema.MEASURE_FACTORS)
                         | {schema.LOCATION_FACTOR})


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

def test_user_message_carries_notes_and_context():
    msg = prompts.build_user_message(LEAD)
    assert "influencer marketing agency" in msg
    assert "VP Growth" in msg
    assert "6000" in msg


def test_blank_context_reads_as_not_recorded_not_as_zero():
    msg = prompts.build_user_message({**LEAD, "employees_display": "unknown", "budget_display": ""})
    # "CRM" is internal vocabulary; the prompt says "on file" like the interface.
    assert "Employees on file (may be stale or blank): not recorded" in msg


def test_system_prompt_forbids_scoring_and_prose():
    text = prompts.SYSTEM_PROMPT.lower()
    assert "never return a score" in text
    assert "never write a justification" in text


def test_system_prompt_says_yes_is_not_a_recommendation():
    text = prompts.SYSTEM_PROMPT.lower()
    assert "not a recommendation to contact them" in text


def test_system_prompt_warns_against_keyword_matching():
    """The recruitment-marketing-agency trap is named explicitly."""
    text = prompts.SYSTEM_PROMPT.lower()
    assert "recruitment marketing agency" in text
    assert "it is not a recruiter" in text


def test_prompt_does_not_disqualify_a_peer_agency_for_being_a_peer():
    """Regression: the hard-disqualify list used to read 'a competing agency
    doing market research', which matches L-1261 ('Fellow agency owner here,
    mostly researching the market') almost verbatim - the very lead §8 names as
    THE ambiguous exemplar. The bullet now requires a stated competitive
    purpose, and the ambiguous rule carries §8's worked example."""
    text = prompts.SYSTEM_PROMPT.lower()
    assert "a competing agency doing market research" not in text
    assert "stated purpose is competitive research" in text
    assert "researching the market" in text
    assert "not by itself a reason to say `no`" in text


def test_prompt_breaks_the_disqualify_vs_ambiguous_tie_toward_ambiguous():
    """§8: ambiguous leads are 'not auto-resolved either direction'. Resolving a
    genuine toss-up to `no` is auto-resolving."""
    text = prompts.SYSTEM_PROMPT.lower()
    assert "nothing in the notes decides between them, answer `ambiguous`" in text


def test_prompt_still_disqualifies_explicit_competitive_recon():
    """The fix must not over-correct: a stated benchmarking purpose is still a
    competitor (guards L-1144, 'curious about your pricing for benchmarking')."""
    text = prompts.SYSTEM_PROMPT.lower()
    assert "benchmarking" in text
    assert "comparing pricing against their own" in text


def test_system_prompt_forbids_inventing_bounds_and_midpoints():
    text = prompts.SYSTEM_PROMPT.lower()
    assert "never invent an upper bound" in text
    assert "never collapse a range to its midpoint" in text


def test_system_prompt_forbids_resolving_the_crm_conflict():
    text = prompts.SYSTEM_PROMPT.lower()
    assert "do not prefer the crm" in text


def test_retry_suffix_names_the_violations():
    assert "urgency.evidence is empty" in prompts.build_retry_suffix(["urgency.evidence is empty"])


# --------------------------------------------------------------------------- #
# Provider abstraction
# --------------------------------------------------------------------------- #

def test_pipeline_imports_no_vendor_sdk():
    """The whole point of the abstraction: llm_extraction stays vendor-free."""
    source = open(llm_extraction.__file__, encoding="utf-8").read()
    for vendor in ("import anthropic", "from anthropic", "from google", "import google"):
        assert vendor not in source, f"{vendor!r} leaked into llm_extraction.py"


def test_fake_provider_satisfies_the_protocol():
    assert isinstance(FakeProvider([]), providers.LLMProvider)


def test_gemini_is_the_default_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert providers.resolve_provider_name() == "gemini"


def test_provider_name_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert providers.resolve_provider_name() == "anthropic"


def test_explicit_provider_name_wins_over_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert providers.resolve_provider_name("gemini") == "gemini"


def test_unknown_provider_raises_a_clear_error():
    with pytest.raises(providers.ProviderConfigError, match="Unknown LLM_PROVIDER"):
        providers.get_provider("groq")


def test_both_providers_can_be_constructed(monkeypatch):
    """Anthropic must remain a working alternate, not a deleted stub."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    gemini = providers.get_provider("gemini")
    anthropic_provider = providers.get_provider("anthropic")
    assert (gemini.name, anthropic_provider.name) == ("gemini", "anthropic")
    assert isinstance(gemini, providers.LLMProvider)
    assert isinstance(anthropic_provider, providers.LLMProvider)


def test_gemini_model_defaults_and_overrides(monkeypatch):
    """Asserts the override mechanism, not a specific model string - the model
    is configuration, and pinning it here would make this test go stale every
    time the configured default moves."""
    import providers.gemini_provider as gemini_module

    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert providers.get_provider("gemini").model == gemini_module.DEFAULT_MODEL

    monkeypatch.setenv("GEMINI_MODEL", "some-other-model")
    assert providers.get_provider("gemini").model == "some-other-model"


@pytest.fixture
def no_credentials(monkeypatch):
    """Isolate from every credential source, not just the environment.

    `_setting` reads secrets.toml directly (so the package works without
    Streamlit installed), which means clearing env vars alone no longer
    guarantees an unconfigured provider on a developer machine that has a real
    secrets.toml on disk.
    """
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(providers, "_secrets_file", lambda: {})
    # Streamlit's own store is the third source, and it reads the same file from
    # disk - so on a machine with a real secrets.toml it must be blanked too.
    monkeypatch.setattr("streamlit.secrets", {}, raising=False)


def test_gemini_without_a_key_fails_only_when_called(no_credentials):
    """Construction must not explode; offline tests build providers freely."""
    provider = GeminiProvider()
    assert provider.is_configured() is False
    with pytest.raises(providers.ProviderConfigError, match="GEMINI_API_KEY"):
        _ = provider.client


def test_settings_resolve_from_secrets_file_without_streamlit(monkeypatch):
    """Regression: secrets.toml was previously only readable via `st.secrets`,
    so with Streamlit absent a configured key was silently invisible."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(providers, "_secrets_file", lambda: {"GEMINI_API_KEY": "from-file"})
    assert providers._setting("GEMINI_API_KEY") == "from-file"


def test_environment_overrides_the_secrets_file(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    monkeypatch.setattr(providers, "_secrets_file", lambda: {"GEMINI_API_KEY": "from-file"})
    assert providers._setting("GEMINI_API_KEY") == "from-env"


def test_max_workers_setting(monkeypatch):
    monkeypatch.delenv("LLM_MAX_WORKERS", raising=False)
    assert llm_extraction.max_workers_setting() == 8
    monkeypatch.setenv("LLM_MAX_WORKERS", "3")
    assert llm_extraction.max_workers_setting() == 3
    assert llm_extraction.max_workers_setting(5) == 5
    monkeypatch.setenv("LLM_MAX_WORKERS", "nonsense")
    assert llm_extraction.max_workers_setting() == 8


# --------------------------------------------------------------------------- #
# Gemini schema adaptation
# --------------------------------------------------------------------------- #

def test_nullable_union_becomes_the_gemini_nullable_flag():
    adapted = to_gemini_schema({"anyOf": [{"type": "number"}, {"type": "null"}]})
    assert adapted == {"type": "number", "nullable": True}


def test_additional_properties_is_dropped_for_gemini():
    adapted = to_gemini_schema({
        "type": "object", "properties": {"a": {"type": "string"}},
        "required": ["a"], "additionalProperties": False,
    })
    assert "additionalProperties" not in adapted
    assert adapted["propertyOrdering"] == ["a"]


def test_adaptation_preserves_field_names_and_enums():
    """A wire-format translation, not a contract change."""
    canonical = schema.build_json_schema()
    adapted = to_gemini_schema(canonical)
    assert set(adapted["properties"]) == set(canonical["properties"])
    assert adapted["properties"]["is_potential_buyer"]["enum"] == ["yes", "no", "ambiguous"]
    measure = adapted["properties"]["employee_count_in_notes"]["properties"]
    assert measure["kind"]["enum"] == list(schema.MEASURE_KINDS)
    assert measure["low"] == {"type": "number", "nullable": True,
                              "description": measure["low"]["description"]}


def test_adaptation_does_not_mutate_the_canonical_schema():
    canonical = schema.build_json_schema()
    before = json.dumps(canonical, sort_keys=True)
    to_gemini_schema(canonical)
    assert json.dumps(canonical, sort_keys=True) == before


# --------------------------------------------------------------------------- #
# Extraction - success and malformed responses
# --------------------------------------------------------------------------- #

def test_successful_extraction_records_provider_and_model():
    provider = FakeProvider([valid_payload()])
    result = run(LEAD, provider)
    assert result.status == "ok"
    assert (result.provider, result.model) == ("fake", "fake-model-1")
    assert result.extraction["urgency"]["level"] == "high"
    assert result.cached_tokens == 900


def test_provider_receives_the_system_prompt_and_canonical_schema():
    provider = FakeProvider([valid_payload()])
    run(LEAD, provider)
    call = provider.calls[0]
    assert call["system_prompt"] == prompts.SYSTEM_PROMPT
    assert call["json_schema"] == schema.build_json_schema()
    assert "influencer marketing agency" in call["user_message"]


def test_markdown_fenced_json_is_recovered():
    """Some models fence JSON even when told not to; that is not a failure."""
    fenced = "```json\n" + json.dumps(valid_payload()) + "\n```"
    assert run(LEAD, FakeProvider([fenced])).status == "ok"


def test_malformed_json_triggers_one_corrective_retry():
    provider = FakeProvider(["{not json", valid_payload()])
    result = run(LEAD, provider)
    assert result.status == "recovered"
    assert result.attempts == 2
    assert "did not satisfy the contract" in provider.calls[1]["user_message"]


def test_invalid_enum_triggers_a_corrective_retry_naming_the_field():
    broken = valid_payload(timeline={"level": "within_1_month", "evidence": "2 weeks"})
    provider = FakeProvider([broken, valid_payload()])
    result = run(LEAD, provider)
    assert result.status == "recovered"
    assert "timeline.level" in provider.calls[1]["user_message"]


def test_non_object_json_is_rejected():
    provider = FakeProvider(["[1, 2, 3]", valid_payload()])
    assert run(LEAD, provider).status == "recovered"


def test_empty_response_triggers_a_retry():
    provider = FakeProvider([ProviderResponse(text=""), valid_payload()])
    assert run(LEAD, provider).status == "recovered"


# --------------------------------------------------------------------------- #
# Extraction - failure handling
# --------------------------------------------------------------------------- #

def test_two_failures_yield_a_flagged_unknown_record_not_an_exception():
    broken = valid_payload(buyer_type="wizard")
    result = run(LEAD, FakeProvider([broken, broken]))
    assert result.status == "failed"
    assert result.needs_review
    assert schema.validate_extraction(result.extraction) == []
    assert result.extraction["is_potential_buyer"] == "ambiguous"


def test_provider_error_does_not_propagate():
    provider = FakeProvider([ProviderError("rate limited", code="rate_limited",
                                          detail="429 quota exceeded")] * 4)
    result = run(LEAD, provider)
    assert result.status == "failed"
    assert result.failure_code == "rate_limited"
    # Vendor detail is retained for logs, but only on `error`.
    assert "429" in result.error


def test_transient_error_then_success_recovers():
    provider = FakeProvider([ProviderError("503 unavailable"), valid_payload()])
    assert run(LEAD, provider).status == "recovered"


def test_retry_backs_off_before_the_second_attempt():
    """Free-tier quotas are easy to trip; a retry waits rather than hammering."""
    delays = []
    provider = FakeProvider([ProviderError("429"), valid_payload()])
    extract_signals(LEAD, provider, cache=None, sleep=delays.append)
    assert delays and delays[0] > 0


def test_refusal_is_not_retried():
    """A refusal is a decision, not a transient fault."""
    provider = FakeProvider([ProviderResponse(text=None, refused=True, refusal_reason="blocked")])
    result = run(LEAD, provider)
    assert result.status == "failed"
    assert "blocked" in result.error
    assert len(provider.calls) == 1


def test_misconfiguration_propagates_rather_than_flagging_every_lead():
    class Broken:
        name, model = "broken", "x"

        def extract(self, *_args):
            raise providers.ProviderConfigError("GEMINI_API_KEY is not set")

    with pytest.raises(providers.ProviderConfigError):
        run(LEAD, Broken())


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #

def test_identical_notes_reuse_one_call():
    cache = ExtractionCache(path=None)
    provider = FakeProvider([valid_payload()])
    first = run(LEAD, provider, cache=cache)
    second = run({**LEAD, "lead_id": "L-OTHER"}, provider, cache=cache)
    assert len(provider.calls) == 1
    assert (first.from_cache, second.from_cache) == (False, True)
    assert second.extraction == first.extraction


def test_the_contract_version_is_the_one_that_carries_the_stated_purpose():
    assert schema.SCHEMA_VERSION == "1.3"


def test_an_extraction_cached_under_the_previous_contract_is_not_reused():
    """The old contract has no stated-purpose field, so replaying one of its
    answers would route a lead on an extraction that was never asked the
    question. The version is part of the key precisely to prevent that."""
    cache = ExtractionCache(path=None)
    key_now = ExtractionCache.key(LEAD, "gemini", "m")
    assert f"schema={schema.SCHEMA_VERSION}" in prompts.cache_signature()

    # `prompts` binds the version at import, so that is the name the signature
    # actually reads - patching schema's copy would leave the key unchanged and
    # the test would pass while proving nothing.
    previous = prompts.SCHEMA_VERSION
    try:
        prompts.SCHEMA_VERSION = "1.2"
        key_then = ExtractionCache.key(LEAD, "gemini", "m")
    finally:
        prompts.SCHEMA_VERSION = previous

    assert key_then != key_now, "a schema bump must change the cache key"

    # Concretely: an answer stored under 1.2 is a miss for a 1.3 request.
    cache.put(key_then, valid_payload())
    assert cache.get(key_then) is not None
    assert cache.get(key_now) is None


def test_a_prompt_change_also_invalidates_the_cache():
    """The stated-purpose rules live in the prompt, not only in the schema."""
    assert f"prompt={prompts.PROMPT_VERSION}" in prompts.cache_signature()
    assert prompts.PROMPT_VERSION == "1.6"


def test_different_notes_are_not_confused():
    cache = ExtractionCache(path=None)
    provider = FakeProvider([valid_payload(), valid_payload()])
    run(LEAD, provider, cache=cache)
    run({**LEAD, "notes_clean": "Completely different enquiry."}, provider, cache=cache)
    assert len(provider.calls) == 2


def test_cache_returns_copies_not_mutable_references():
    cache = ExtractionCache(path=None)
    provider = FakeProvider([valid_payload()])
    first = run(LEAD, provider, cache=cache)
    first.extraction["urgency"]["level"] = "low"
    first.extraction["employee_count_in_notes"]["low"] = 999

    second = run({**LEAD, "lead_id": "L-2"}, provider, cache=cache)
    assert second.extraction["urgency"]["level"] == "high"
    assert second.extraction["employee_count_in_notes"]["low"] == 26


def test_cache_stores_a_copy_so_later_mutation_cannot_reach_in():
    cache = ExtractionCache(path=None)
    payload = valid_payload()
    cache.put("k", payload)
    payload["urgency"]["level"] = "low"
    assert cache.get("k")["urgency"]["level"] == "high"


def test_cache_key_tracks_provider_and_model():
    """Switching provider must not serve a response produced by the other one."""
    a = ExtractionCache.key(LEAD, "gemini", "gemini-2.5-flash-lite")
    b = ExtractionCache.key(LEAD, "anthropic", "claude-opus-5")
    c = ExtractionCache.key(LEAD, "gemini", "gemini-2.5-pro")
    assert len({a, b, c}) == 3


def test_cache_key_tracks_the_prompt_and_schema_versions():
    key = ExtractionCache.key(LEAD, "gemini", "m")
    original = prompts.PROMPT_VERSION
    try:
        prompts.PROMPT_VERSION = "9.9"
        assert ExtractionCache.key(LEAD, "gemini", "m") != key
    finally:
        prompts.PROMPT_VERSION = original


def test_switching_provider_bypasses_the_cache():
    cache = ExtractionCache(path=None)
    gemini_like = FakeProvider([valid_payload()])
    run(LEAD, gemini_like, cache=cache)

    other = FakeProvider([valid_payload()])
    other.name, other.model = "anthropic", "claude-opus-5"
    assert run(LEAD, other, cache=cache).from_cache is False


def test_failed_extraction_is_not_cached():
    """One transient outage must not poison the lead for the whole run."""
    cache = ExtractionCache(path=None)
    provider = FakeProvider([ProviderError("boom")] * 4 + [valid_payload()])
    assert run(LEAD, provider, cache=cache).status == "failed"
    assert run(LEAD, provider, cache=cache).status == "ok"


def test_refused_extraction_is_not_cached():
    cache = ExtractionCache(path=None)
    provider = FakeProvider([
        ProviderResponse(text=None, refused=True, refusal_reason="blocked"),
        valid_payload(),
    ])
    assert run(LEAD, provider, cache=cache).status == "failed"
    assert run(LEAD, provider, cache=cache).status == "ok"


def test_corrupt_cache_file_is_treated_as_a_cold_cache(tmp_path):
    path = tmp_path / "extractions.json"
    path.write_text("{ not json", encoding="utf-8")
    assert len(ExtractionCache(path=path)) == 0


def test_cache_round_trips_through_disk(tmp_path):
    path = tmp_path / "nested" / "extractions.json"
    cache = ExtractionCache(path=path)
    cache.put("k", valid_payload())
    cache.save()
    assert ExtractionCache(path=path).get("k")["urgency"]["level"] == "high"


def test_disabled_cache_never_serves_a_hit():
    cache = ExtractionCache(path=None, enabled=False)
    provider = FakeProvider([valid_payload(), valid_payload()])
    run(LEAD, provider, cache=cache)
    run(LEAD, provider, cache=cache)
    assert len(provider.calls) == 2


# --------------------------------------------------------------------------- #
# Batch
# --------------------------------------------------------------------------- #

def test_batch_preserves_input_order_under_concurrency():
    leads = [{**LEAD, "lead_id": f"L-{i}", "notes_clean": f"Enquiry number {i}."} for i in range(12)]
    provider = FakeProvider([valid_payload() for _ in leads])
    results = llm_extraction.extract_batch(
        leads, provider, rpm=0, cache=ExtractionCache(path=None), max_workers=4,
    )
    assert [r.lead_id for r in results] == [lead["lead_id"] for lead in leads]


def test_batch_reports_progress():
    leads = [{**LEAD, "lead_id": f"L-{i}", "notes_clean": f"Enquiry {i}."} for i in range(5)]
    provider = FakeProvider([valid_payload() for _ in leads])
    seen = []
    llm_extraction.extract_batch(
        leads, provider, rpm=0, cache=ExtractionCache(path=None), max_workers=2,
        progress_callback=lambda done, total: seen.append((done, total)),
    )
    assert len(seen) == 5 and seen[-1] == (5, 5)


def test_batch_survives_one_bad_lead():
    leads = [
        {**LEAD, "lead_id": "L-good", "notes_clean": "Good enquiry."},
        {**LEAD, "lead_id": "L-bad", "notes_clean": "Bad enquiry."},
    ]
    provider = FakeProvider([valid_payload()] + [ProviderError("boom")] * 4)
    results = llm_extraction.extract_batch(
        leads, provider, rpm=0, cache=ExtractionCache(path=None), max_workers=1,
        sleep=no_sleep,
    )
    assert len(results) == 2
    assert {r.status for r in results} == {"ok", "failed"}


def test_empty_batch_makes_no_calls():
    provider = FakeProvider([])
    assert llm_extraction.extract_batch([], provider, rpm=0) == []


def test_summarise_batch_counts_gate_outcomes_only():
    results = [
        llm_extraction.ExtractionResult("a", valid_payload(), "ok"),
        llm_extraction.ExtractionResult("b", valid_payload(is_potential_buyer="no"), "ok"),
        llm_extraction.ExtractionResult(
            "c", valid_payload(is_potential_buyer="ambiguous", ambiguity_reason="unclear"), "ok",
        ),
        llm_extraction.ExtractionResult("d", schema.unknown_extraction(), "failed"),
    ]
    summary = summarise_batch(results)
    assert (summary["buyer_yes"], summary["buyer_no"], summary["buyer_ambiguous"]) == (1, 1, 2)
    assert summary["failed"] == 1
    # No scoring, ranking or recommendation vocabulary in the run summary.
    for banned in ("score", "rank", "contact_now", "nurture", "disqualify"):
        assert not any(banned in k for k in summary)
