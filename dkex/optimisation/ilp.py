"""Module 7 — Exact baselines on the cycle/chain formulation (Section 4.4).

* ``solve_max_volume``     Myopic ILP: maximise transplants (F1^ILP(t) ceiling).
* ``solve_lexicographic``  Dickerson et al. (2014) style: first maximise the
  number of high-cPRA recipients, then maximise total volume subject to that.

Both are set-packing ILPs over the same StructureSet the MOEA uses, solved
with SciPy's HiGHS MILP interface (no external solver required).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from dkex.optimisation.structures import StructureSet


def _packing_constraint(S: StructureSet) -> LinearConstraint:
    rows, cols = [], []
    for s, nodes in enumerate(S.nodes):
        rows.extend(nodes)
        cols.extend([s] * len(nodes))
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(S.n_nodes, len(S)))
    return LinearConstraint(A, -np.inf, 1.0)


def _solve(c: np.ndarray, constraints: list[LinearConstraint], time_limit: float) -> np.ndarray:
    n = c.size
    res = milp(
        c=-c,  # milp minimises
        constraints=constraints,
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
        options={"time_limit": time_limit, "disp": False},
    )
    if res.x is None:
        raise RuntimeError(f"MILP failed: {res.message}")
    return np.nonzero(res.x > 0.5)[0]


def solve_max_volume(S: StructureSet, hla_tiebreak: bool = True, time_limit: float = 120.0) -> np.ndarray:
    """Return indices of selected structures maximising the transplant count."""
    if len(S) == 0:
        return np.array([], dtype=int)
    c = S.sizes.astype(float)
    if hla_tiebreak:
        # Tie-break by HLA quality without ever trading away a transplant.
        c = c + 1e-3 * S.hla / max(1.0, S.hla.sum())
    return _solve(c, [_packing_constraint(S)], time_limit)


def solve_lexicographic(S: StructureSet, high_mask: np.ndarray, time_limit: float = 120.0) -> np.ndarray:
    if len(S) == 0:
        return np.array([], dtype=int)
    pack = _packing_constraint(S)
    high_count = np.array([high_mask[list(r)].sum() for r in S.recipients], dtype=float)
    first = _solve(high_count, [pack], time_limit)
    best_high = high_count[first].sum()
    keep_high = LinearConstraint(high_count[None, :], best_high - 1e-6, np.inf)
    return _solve(S.sizes.astype(float), [pack, keep_high], time_limit)
