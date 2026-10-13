"""Module 10 — Price of Fairness trajectory (Section 3.6, Eq. 13).

    PoF_t = (F1^ILP(t) - F1^MOEA(t)) / F1^ILP(t) * 100 %

F1^ILP(t) and F1^MOEA(t) must be computed on the identical pool V_t, graph E_t
and structure set — the simulator guarantees this by solving the ILP on the
same StructureSet handed to the MOEA.
"""

from __future__ import annotations

import numpy as np


def price_of_fairness(f1_ilp: float, f1_fair: float) -> float:
    if f1_ilp <= 0:
        return 0.0
    return float((f1_ilp - f1_fair) / f1_ilp * 100.0)


def trajectory_summary(pof: list[float] | np.ndarray) -> dict[str, float]:
    """Mean PoF and PoF variance across rounds (Section 4.5.2)."""
    x = np.asarray([p for p in pof if p is not None and np.isfinite(p)], dtype=float)
    if x.size == 0:
        return {"pof_mean": float("nan"), "pof_var": float("nan"), "pof_rounds": 0}
    return {"pof_mean": float(x.mean()), "pof_var": float(x.var(ddof=0)), "pof_rounds": int(x.size)}
