"""Module 3 — Logistics feasibility constraint (Sections 3.3 and 4.3).

An edge between donor centre i and recipient centre j is kept only if

    T_ij^transport + T^prep <= CIT_max                          (Eq. 4 / 14)
"""

from __future__ import annotations

import numpy as np

from dkex.config import CENTRES, DISTANT_CENTRES, LogisticsConfig


class LogisticsNetwork:
    def __init__(self, cfg: LogisticsConfig, centres: tuple[str, ...] = CENTRES):
        self.cfg = cfg
        self.centres = centres
        self.index = {c: k for k, c in enumerate(centres)}
        n = len(centres)
        T = np.full((n, n), np.nan)
        for a, row in cfg.transport_hours.items():
            for b, hours in row.items():
                i, j = self.index[a], self.index[b]
                T[i, j] = T[j, i] = hours
        if np.isnan(T).any():
            missing = [(centres[i], centres[j]) for i, j in zip(*np.where(np.isnan(T)))]
            raise ValueError(f"transport_hours missing entries: {missing[:5]}")
        off_diag = ~np.eye(n, dtype=bool)
        T[off_diag] += cfg.inter_centre_delay_hours
        self.transport = T
        self.feasible = (T + cfg.prep_hours) <= cfg.cit_max_hours
        self.distant = np.array([c in DISTANT_CENTRES for c in centres])

    def is_feasible(self, donor_centre: int, recipient_centre: int) -> bool:
        return bool(self.feasible[donor_centre, recipient_centre])

    def binding_fraction(self) -> float:
        """Share of centre pairs ruled out by CIT (0 => the constraint never binds)."""
        return float(1.0 - self.feasible.mean())

    def summary(self) -> str:
        lines = [f"CIT_max={self.cfg.cit_max_hours}h, T_prep={self.cfg.prep_hours}h"]
        for i, a in enumerate(self.centres):
            cells = []
            for j, b in enumerate(self.centres):
                mark = "" if self.feasible[i, j] else "x"
                cells.append(f"{self.transport[i, j]:5.1f}{mark:1s}")
            lines.append(f"{a:>10s} " + " ".join(cells))
        lines.append(f"infeasible centre pairs: {self.binding_fraction():.1%}")
        return "\n".join(lines)
