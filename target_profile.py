"""The buyer profile a run is assessed against.

Before this existed the ideal customer was hardcoded: the extraction schema
described industry fit as "marketing / growth / ops / lead-gen agency (core
ICP)" and the budget bands were fixed dollar figures. That made the app a
demonstration of one agency's pipeline rather than a tool anyone could point at
their own leads. The profile replaces both with values the operator supplies.

It reaches assessment two ways, and both matter:

* The **industries** and **location** are given to the model, because judging
  whether a lead's business matches a target industry is interpretation.
* The **budget range** is applied in Python, because comparing an interval to a
  range is arithmetic - the same division of labour the rest of the system uses.

What it deliberately does not do is change routing. A profile moves a lead's
fit score; the thresholds, the intent floor and the four routes are untouched,
so a perfect profile match with no buying intent still cannot reach
CONTACT_NOW.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

# Bump when the profile's meaning changes; participates in the extraction cache
# key so a re-run under a different profile can never be served stale results.
PROFILE_VERSION = "1.1"

MAX_INDUSTRY_CHARS = 120
MAX_LOCATION_CHARS = 120

ANY_INDUSTRY = "Any"
"""The operator's way of saying industry is not one of their criteria.

Stored as an empty `industries` tuple rather than as a magic string in the list,
so no code path can accidentally treat "Any" as the name of an industry and ask
the model whether a lead is in it."""


class TargetProfileError(ValueError):
    """The operator's profile is unusable - message is safe to show them."""


@dataclass(frozen=True)
class TargetProfile:
    """Industries and budget are required; location is optional.

    `industries` is a tuple because the profile is frozen and is used as part of
    a cache key - a list would be neither hashable nor safe to share. An empty
    tuple is the `Any` case: the operator has said industry is not one of their
    criteria, and no lead may be scored up or down for its industry.

    Company size is deliberately absent. The system still extracts and displays
    headcount, but it is not something the operator declares a target for -
    size is a fact about a lead, not a filter the product asks them to set.
    """

    industries: tuple[str, ...]
    budget_min: float
    budget_max: float
    location: Optional[str] = None

    # -- construction ------------------------------------------------------- #

    @classmethod
    def create(
        cls,
        industries: Any,
        budget_min: Any,
        budget_max: Any,
        location: Optional[str] = None,
    ) -> "TargetProfile":
        """Validate operator input. Raises `TargetProfileError` with copy that is
        safe to render - no field names, no exception types.

        `industries` accepts a single string or any sequence of them, so a
        caller that targets one industry reads exactly as it did before this
        field could hold several.
        """
        selected = _clean_industries(industries)

        if any(name.casefold() == ANY_INDUSTRY.casefold() for name in selected):
            if len(selected) > 1:
                raise TargetProfileError(
                    "Choose either Any, or specific industries - not both."
                )
            selected = []
        elif not selected:
            raise TargetProfileError(
                "Choose the industries you sell to, or Any."
            )

        for name in selected:
            if len(name) > MAX_INDUSTRY_CHARS:
                raise TargetProfileError("That industry description is too long.")

        low = _to_amount(budget_min, "smallest")
        high = _to_amount(budget_max, "largest")
        if low > high:
            raise TargetProfileError(
                "The smallest budget must not be larger than the largest budget."
            )

        place = (location or "").strip() or None
        if place and len(place) > MAX_LOCATION_CHARS:
            raise TargetProfileError("That location is too long.")

        return cls(industries=tuple(selected), budget_min=low, budget_max=high,
                   location=place)

    # -- queries ------------------------------------------------------------ #

    @property
    def accepts_any_industry(self) -> bool:
        """True means industry is not a criterion: it may neither reward nor
        penalise a lead, and cannot be used to tell two leads apart."""
        return not self.industries

    @property
    def has_location(self) -> bool:
        """False means location must have no effect on any score at all."""
        return bool(self.location)

    def industry_label(self) -> str:
        """The selection as the operator reads it back."""
        return ", ".join(self.industries) if self.industries else ANY_INDUSTRY

    def signature(self) -> str:
        """Identity for cache keys. A different profile is a different question,
        so it must not reuse an answer given for the previous one.

        The industries are sorted: picking Technology then Healthcare asks the
        same question as picking Healthcare then Technology, and paying twice
        for one answer would be waste, not caution.
        """
        place = (self.location or "").strip().lower()
        picked = "|".join(sorted(name.strip().lower() for name in self.industries))
        return (
            f"profile={PROFILE_VERSION};"
            f"industries={picked or 'any'};"
            f"budget={self.budget_min:.0f}-{self.budget_max:.0f};"
            f"location={place}"
        )

    def budget_label(self) -> str:
        return f"{_money(self.budget_min)} - {_money(self.budget_max)}"

    def describe_for_model(self) -> str:
        """The profile as the model is told about it.

        The single-industry wording is byte-identical to what it has always
        been, so selecting one industry asks the model exactly the question it
        was asked before this field could hold several.

        Location is omitted entirely when unset rather than sent as "any", so
        the model is never nudged into inventing a location judgement.
        """
        if self.accepts_any_industry:
            industry_line = (
                "Target industry: any - the operator has no industry preference, "
                "so do not treat any industry as a better or worse match"
            )
        elif len(self.industries) == 1:
            industry_line = f"Target industry: {self.industries[0]}"
        else:
            industry_line = (
                "Target industries (a lead in ANY ONE of these is a match, and "
                "matching more than one is no better than matching one): "
                + ", ".join(self.industries)
            )
        lines = [
            industry_line,
            f"Target monthly budget: {self.budget_label()}",
        ]
        if self.has_location:
            lines.append(f"Target location: {self.location}")
        return "\n".join(lines)

    def matches_location(self, stated: Optional[str]) -> Optional[bool]:
        """True match, False mismatch, None when there is nothing to compare.

        None is the important case: a lead that simply never mentions where it
        is must not be treated as failing a location test it was never given.
        """
        if not self.has_location or not stated:
            return None
        return _place_matches(self.location or "", stated)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _clean_industries(value: Any) -> list[str]:
    """Normalise operator input to an ordered, de-duplicated list of names.

    A bare string is accepted as a selection of one, so existing single-industry
    callers keep working unchanged. Duplicates are dropped case-insensitively -
    picking the same industry twice is not two criteria.
    """
    if value is None:
        return []
    if isinstance(value, str):
        candidates: Iterable[Any] = [value]
    elif isinstance(value, Sequence):
        candidates = value
    else:
        raise TargetProfileError("Choose the industries you sell to, or Any.")

    out: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        name = str(item or "").strip()
        if not name or name.casefold() in seen:
            continue
        seen.add(name.casefold())
        out.append(name)
    return out


_AMOUNT_RE = re.compile(r"^\$?\s*([\d,]+(?:\.\d+)?)\s*([km])?$", re.I)


def _to_amount(value: Any, which: str) -> float:
    """Accept a number, or operator text like '$10,000' / '10k' / '1.5m'."""
    if isinstance(value, bool):
        raise TargetProfileError(f"Enter a {which} budget as a number.")
    if isinstance(value, (int, float)):
        amount = float(value)
    else:
        match = _AMOUNT_RE.match(str(value or "").strip())
        if not match:
            raise TargetProfileError(f"Enter a {which} budget as a number.")
        amount = float(match.group(1).replace(",", ""))
        suffix = (match.group(2) or "").lower()
        amount *= {"k": 1_000, "m": 1_000_000}.get(suffix, 1)
    if amount < 0 or amount != amount:                      # negative or NaN
        raise TargetProfileError(f"Enter a {which} budget of zero or more.")
    return amount


def _money(amount: float) -> str:
    return f"${amount:,.0f}"


def _normalise_place(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", text.lower()) if len(token) > 1}


# Common demonyms and short forms, so "Nigerian agency" matches a "Nigeria"
# target. Deliberately small: an unrecognised place falls back to token
# comparison rather than guessing, because a wrong match here silently moves a
# lead's score.
_PLACE_ALIASES = {
    "nigeria": {"nigerian", "ng", "lagos", "abuja"},
    "united kingdom": {"uk", "britain", "british", "england", "london", "gb"},
    "united states": {"us", "usa", "american", "america"},
    "germany": {"german", "berlin", "deutschland"},
    "france": {"french", "paris"},
    "canada": {"canadian", "toronto"},
    "australia": {"australian", "sydney", "melbourne"},
    "india": {"indian", "mumbai", "bangalore", "bengaluru", "delhi"},
    "kenya": {"kenyan", "nairobi"},
    "south africa": {"south african", "johannesburg", "cape town"},
}


def _place_matches(target: str, stated: str) -> bool:
    target_l, stated_l = target.strip().lower(), stated.strip().lower()
    if not target_l or not stated_l:
        return False
    if target_l in stated_l or stated_l in target_l:
        return True

    target_tokens, stated_tokens = _normalise_place(target_l), _normalise_place(stated_l)
    if target_tokens & stated_tokens:
        return True

    for canonical, aliases in _PLACE_ALIASES.items():
        family = {canonical} | aliases
        target_hit = target_l in family or bool(target_tokens & family)
        stated_hit = stated_l in family or bool(stated_tokens & family)
        if target_hit and stated_hit:
            return True
    return False
