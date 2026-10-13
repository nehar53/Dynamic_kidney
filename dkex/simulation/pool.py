"""Pool entities: donor-recipient pairs and altruistic donors (Section 3.1)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ABO_TYPES = ("O", "A", "B", "AB")


def abo_compatible(donor: str, patient: str) -> bool:
    if donor == "O":
        return True
    if donor == "A":
        return patient in ("A", "AB")
    if donor == "B":
        return patient in ("B", "AB")
    return patient == "AB"  # donor AB


@dataclass
class Pair:
    """A donor-recipient pair, or an altruistic donor when ``altruist`` is True."""

    pid: int
    donor_abo: str
    centre: int                       # index into the logistics network
    arrival_round: int
    altruist: bool = False
    patient_abo: str | None = None
    cpra: float = 0.0                 # calculated panel reactive antibody, 0..1
    wait_months: float = 0.0

    def is_high_cpra(self, threshold: float = 0.80) -> bool:
        return (not self.altruist) and self.cpra >= threshold


@dataclass
class PoolState:
    """The active set V_t at the start of a clearing round."""

    pairs: dict[int, Pair] = field(default_factory=dict)
    round: int = 0

    def __len__(self) -> int:
        return len(self.pairs)

    def add(self, pairs: list[Pair]) -> None:
        for p in pairs:
            self.pairs[p.pid] = p

    def remove(self, pids) -> list[Pair]:
        return [self.pairs.pop(pid) for pid in list(pids) if pid in self.pairs]

    @property
    def patients(self) -> list[Pair]:
        return [p for p in self.pairs.values() if not p.altruist]

    @property
    def altruists(self) -> list[Pair]:
        return [p for p in self.pairs.values() if p.altruist]

    def n_patients(self) -> int:
        return sum(1 for p in self.pairs.values() if not p.altruist)

    def high_cpra_share(self, threshold: float = 0.80) -> float:
        pts = self.patients
        return float(np.mean([p.cpra >= threshold for p in pts])) if pts else 0.0
