"""Shared label contract for longitudinal response evaluation.

Response labels are deliberately separate from treatment phases and operative
states.  The legacy ``actions`` field remains for checkpoint compatibility,
but new manifests and prospective code should use these helpers instead.
"""
from __future__ import annotations

RANO_NAMES = ("PD", "SD", "PR", "CR")
RANO_TO_FLAT = {name: i for i, name in enumerate(RANO_NAMES)}
OPERATIVE_RATINGS = frozenset({"Pre-Op", "Post-Op", "Post-Op/PD"})


def normalize_rating(value) -> str:
    return "" if value is None else str(value).strip()


def response_label(value) -> int:
    """Return flat {PD,SD,PR,CR} id, or -1 for invalid/operative labels."""
    return RANO_TO_FLAT.get(normalize_rating(value), -1)


def response_valid(value) -> bool:
    return response_label(value) >= 0


def operative_event(value) -> bool:
    return normalize_rating(value) in OPERATIVE_RATINGS
