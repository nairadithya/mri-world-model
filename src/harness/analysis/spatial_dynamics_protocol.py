"""R37 cutoff-only history construction from caller-verified full chronology.

Prepared R35 visit/pair metadata is not itself complete chronology: unusable
visits can be absent. Callers must supply every original visit in order, with
explicit usability and measured preceding-visit gap evidence. IDs are audit-only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class ChronologyVisit:
    visit_id: str
    patient_id: str
    usable: bool
    # Actual elapsed days from the immediately preceding original visit; None
    # only where no prior visit exists or the source lacks interval evidence.
    gap_from_previous_days: float | None
    availability: tuple[bool, bool, bool, bool]
    target: object | None = None  # opaque held-out payload, never copied to HistoryInput


@dataclass(frozen=True)
class FullChronology:
    visits: tuple[ChronologyVisit, ...]
    complete_evidence: bool


@dataclass(frozen=True)
class HistoryVisitInput:
    availability: tuple[bool, bool, bool, bool]
    gap_log1p_days: float


@dataclass(frozen=True)
class HistoryInput:
    visits: tuple[HistoryVisitInput, ...]
    history_mask: tuple[bool, ...]


@dataclass(frozen=True)
class ForecastRow:
    row_id: str
    patient_id: str  # audit/grouping metadata; never predictor input
    cutoff_visit_id: str
    target_visit_id: str  # audit metadata; never predictor input
    history_visit_ids: tuple[str, ...]  # ordered audit IDs, excluded from predictor tensors
    history: HistoryInput
    target: object  # held separately; scorer/trainer only
    primary_history: bool


def build_history_rows(chronology: FullChronology, *, max_visits: int = 4) -> tuple[ForecastRow, ...]:
    """Build one canonical adjacent-usable target list; never bridge omissions.

    Rows require a usable target immediately following the usable cutoff in the
    *full* original chronology and valid positive gaps for retained history.
    The first retained input gap is reset to zero. Padding is represented by
    a false mask to the left. Target payloads are opaque and never enter input.
    """
    if not chronology.complete_evidence:
        raise ValueError("full original visit chronology evidence is required")
    if isinstance(max_visits, bool) or not isinstance(max_visits, int) or not 1 <= max_visits <= 4:
        raise ValueError("max_visits must be an integer from 1 through 4")
    visits = chronology.visits
    if len({v.visit_id for v in visits}) != len(visits):
        raise ValueError("duplicate chronology visit id")
    if any(not v.visit_id or not v.patient_id or len(v.availability) != 4 for v in visits):
        raise ValueError("invalid chronology identity or availability")
    if any(type(v.usable) is not bool or any(type(x) is not bool for x in v.availability) for v in visits):
        raise ValueError("usability and availability must be booleans")
    seen_patients = set()
    closed_patients = set()
    previous_patient = None
    for v in visits:
        if v.patient_id != previous_patient:
            if v.patient_id in closed_patients:
                raise ValueError("patient chronology must form contiguous blocks")
            if previous_patient is not None:
                closed_patients.add(previous_patient)
            previous_patient = v.patient_id
        seen_patients.add(v.patient_id)
    rows: list[ForecastRow] = []
    for i in range(len(visits) - 1):
        target = visits[i + 1]
        cutoff = visits[i]
        if cutoff.patient_id != target.patient_id:
            continue
        if not cutoff.usable or not target.usable or target.target is None:
            continue
        # History may only end at cutoff; retain the contiguous usable suffix.
        start = i
        while start > 0 and visits[start - 1].patient_id == cutoff.patient_id and visits[start - 1].usable:
            start -= 1
        candidates = list(visits[max(start, i - max_visits + 1): i + 1])
        gaps: list[float] = []
        valid = True
        for entry in candidates[1:]:
            gap = entry.gap_from_previous_days
            if gap is None or not math.isfinite(gap) or gap <= 0:
                valid = False
                break
            gaps.append(math.log1p(gap))
        if not valid:
            continue
        steps = [HistoryVisitInput(v.availability, 0.0 if j == 0 else gaps[j - 1])
                 for j, v in enumerate(candidates)]
        padding = max_visits - len(steps)
        history = HistoryInput(tuple([HistoryVisitInput((False,) * 4, 0.0)] * padding + steps),
                               tuple([False] * padding + [True] * len(steps)))
        rows.append(ForecastRow(f"{cutoff.visit_id}->{target.visit_id}", cutoff.patient_id,
                                cutoff.visit_id, target.visit_id,
                                tuple(v.visit_id for v in candidates), history, target.target,
                                primary_history=len(steps) >= 2))
    return tuple(rows)
