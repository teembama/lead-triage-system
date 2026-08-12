"""The buyer profile a run is assessed against.

Before this existed the ideal customer was hardcoded: the extraction schema
described industry fit as "marketing / growth / ops / lead-gen agency (core
ICP)" and the budget bands were fixed dollar figures. That made the app a
demonstration of one agency's pipeline rather than a tool anyone could point at
their own leads. The profile replaces both with values the operator supplies.

It reaches assessment two ways, and both matter:

* The **industry** and **location** are given to the model, because judging
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
from typing import Any, Optional

# Bump when the profile's meaning changes; participates in the extraction cache
# key so a re-run under a different profile can never be served stale results.
PROFILE_VERSION = "1.0"

MAX_INDUSTRY_CHARS = 120
MAX_LOCATION_CHARS = 120


class TargetProfileError(ValueError):
    """The operator's profile is unusable - message is safe to show them."""


@dataclass(frozen=True)
class TargetProfile:
    """Industry and budget are required; location is optional.

    Company size is deliberately absent. The system still extracts and displays
    headcount, but it is not something the operator declares a target for -
    size is a fact about a lead, not a filter the product asks them to set.
    """

    industry: str
    budget_min: float
    budget_max: float
    location: Optional[str] = None

    # -- construction ------------------------------------------------------- #

    @classmethod
    def create(
        cls,
        industry: str,
        budget_min: Any,
        budget_max: Any,
        location: Optional[str] = None,
    ) -> "TargetProfile":
        """Validate operator input. Raises `TargetProfileError` with copy that is
        safe to render - no field names, no exception types."""
        industry = (industry or "").strip()
        if not industry:
            raise TargetProfileError("Choose the industry you sell to.")
        if len(industry) > MAX_INDUSTRY_CHARS:
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

        return cls(industry=industry, budget_min=low, budget_max=high, location=place)

    # -- queries ------------------------------------------------------------ #

    @property
    def has_location(self) -> bool:
        """False means location must have no effect on any score at all."""
        return bool(self.location)

    def signature(self) -> str:
        """Identity for cache keys. A different profile is a different question,
        so it must not reuse an answer given for the previous one."""
        place = (self.location or "").strip().lower()
        return (
            f"profile={PROFILE_VERSION};"
            f"industry={self.industry.strip().lower()};"
            f"budget={self.budget_min:.0f}-{self.budget_max:.0f};"
            f"location={place}"
        )

    def budget_label(self) -> str:
        return f"{_money(self.budget_min)} - {_money(self.budget_max)}"

    def describe_for_model(self) -> str:
        """The profile as the model is told about it.

        Location is omitted entirely when unset rather than sent as "any", so
        the model is never nudged into inventing a location judgement.
        """
        lines = [
            f"Target industry: {self.industry}",
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
