"""Module 6 — Fairness objectives F1-F4 (Section 3.4, Eqs. 5-9).

    F1  transplant efficiency          number of transplants in the round
    F2  waiting-time equity            Gini of waiting times of patients left waiting
    F3  sensitised-patient access      | |M∩S|/|S| - |M|/|V_t| |   (high-cPRA gap)
    F4  node-level logistics disparity | |M∩R|/|R| - |M∩U|/|U| |  (proxy only)

Sign convention.  Eq. 5 writes ``min (-F1, F2, -F3, F4)``, i.e. F3 maximised,
but Eq. 8 defines F3 as an absolute *gap*, which should be minimised.  Both
readings are supported via ``f3_mode``:

* ``"min_gap"``    (default) minimise the gap of Eq. 8 — matches the
  "Group Fairness Gap GFG_t" metric of Section 4.5.1.
* ``"max_access"`` maximise the high-cPRA match rate |M∩S|/|S| — matches the
  sign in Eq. 5.

F2 interpretation: Eq. 7 is the Gini over "all waiting patient times"; we
take the patients who remain waiting after the round's matching, so matching
long-waiting patients lowers F2.
"""

from __future__ import annotations

import numpy as np

from dkex.config import DISTANT_CENTRES
from dkex.optimisation.structures import StructureSet
from dkex.simulation.compatibility import CompatibilityGraph
from dkex.simulation.pool import PoolState

OBJ_NAMES = ("F1_transplants", "F2_wait_gini", "F3_cpra_gap", "F4_logistics_gap")


def gini(x: np.ndarray) -> float:
    """Gini coefficient, Eq. 7: sum_i sum_j |x_i - x_j| / (2 n sum x)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2 or x.sum() <= 0:
        return 0.0
    xs = np.sort(x)
    k = np.arange(1, n + 1)
    return float(np.sum((2 * k - n - 1) * xs) / (n * xs.sum()))


class ObjectiveEvaluator:
    def __init__(
        self,
        pool: PoolState,
        graph: CompatibilityGraph,
        structures: StructureSet,
        centres: tuple[str, ...],
        high_cpra_threshold: float = 0.80,
        f3_mode: str = "min_gap",
    ):
        pairs = [pool.pairs[int(pid)] for pid in graph.ids]
        self.S = structures
        self.n = graph.n
        self.patient = ~graph.altruist
        self.high = np.array([p.is_high_cpra(high_cpra_threshold) for p in pairs], dtype=bool)
        distant_idx = {centres.index(c) for c in DISTANT_CENTRES if c in centres}
        dist = np.array([p.centre in distant_idx for p in pairs], dtype=bool)
        self.R = dist & self.patient
        self.U = ~dist & self.patient
        self.wait = np.array([p.wait_months for p in pairs])
        self.n_patients = int(self.patient.sum())
        self.f3_mode = f3_mode
        self._rec = [np.array(r, dtype=np.int64) for r in structures.recipients]

    def matched_mask(self, selected) -> np.ndarray:
        m = np.zeros(self.n, dtype=bool)
        for s in selected:
            m[self._rec[s]] = True
        return m

    def evaluate(self, selected) -> np.ndarray:
        """Raw objective values (F1, F2, F3, F4)."""
        m = self.matched_mask(selected)
        f1 = float(m.sum())
        waiting = self.patient & ~m
        f2 = gini(self.wait[waiting] + 1e-9) if waiting.any() else 0.0
        nS = self.high.sum()
        if self.f3_mode == "max_access":
            f3 = float((m & self.high).sum() / nS) if nS else 0.0
        else:
            f3 = abs((m & self.high).sum() / nS - f1 / self.n_patients) if nS and self.n_patients else 0.0
        nR, nU = self.R.sum(), self.U.sum()
        f4 = abs((m & self.R).sum() / nR - (m & self.U).sum() / nU) if nR and nU else 0.0
        return np.array([f1, f2, float(f3), float(f4)])

    def to_min(self, raw: np.ndarray) -> np.ndarray:
        """Map raw values to a normalised minimisation vector in [0, 1]^4."""
        raw = np.atleast_2d(raw)
        out = np.empty_like(raw)
        out[:, 0] = 1.0 - raw[:, 0] / max(1, self.n_patients)
        out[:, 1] = raw[:, 1]
        out[:, 2] = 1.0 - raw[:, 2] if self.f3_mode == "max_access" else raw[:, 2]
        out[:, 3] = raw[:, 3]
        return out
