import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from module1_generator import Patient
from module3_logistics import build_compatibility_graph
from module5_nsga3_aos import MAX_CYCLE_LEN

# ============================================================
# MODULE 6: ILP UPPER BOUND (CYCLE FORMULATION, CBC)
# F1_ILP(t) = maximum number of transplants in round t
# Used as the volume ceiling in the Price of Fairness:
#   PoF_t = (F1_ILP(t) - F1_MOEA(t)) / F1_ILP(t) x 100%
# Paper 2 — Dynamic KEP Implementation
# ============================================================
#
# Identical feasibility (concept note Section 3.6):
# The ILP uses exactly the same compatibility graph (module 3)
# and the same structures as module 5's repair_to_cycles:
# disjoint cycles of length 2..MAX_CYCLE_LEN. So the ILP optimum
# is a true ceiling for every MOEA solution in the same round.
#
# Solver: CBC through PuLP (bundled with pulp 2.x).
# PuLP >= 3 no longer ships CBC; if CBC is missing we fall back
# to SciPy's HiGHS MILP solver, which gives the same optimum.

ILP_TIME_LIMIT = 120     # seconds per solve
CPRA_THRESHOLD = 0.80


# ============================================================
# CYCLE ENUMERATION
# ============================================================

def enumerate_cycles(
    edges: Dict[Tuple[int, int], float],
    pair_ids: List[int],
    max_len: int = MAX_CYCLE_LEN
) -> List[Tuple[int, ...]]:
    """
    Enumerate every simple cycle of length 2..max_len.

    Each cycle is listed once, starting from its smallest
    pair ID. A cycle (a, b, c) means: donor a -> patient b,
    donor b -> patient c, donor c -> patient a.
    """
    adj: Dict[int, List[int]] = {i: [] for i in pair_ids}
    for (i, j) in edges.keys():
        if i in adj and j in adj:
            adj[i].append(j)

    cycles: List[Tuple[int, ...]] = []
    for start in sorted(pair_ids):
        stack = [(start, (start,))]
        while stack:
            node, path = stack.pop()
            for nxt in adj[node]:
                if nxt == start and len(path) >= 2:
                    cycles.append(path)
                elif (nxt > start and nxt not in path
                      and len(path) < max_len):
                    stack.append((nxt, path + (nxt,)))
    return cycles


def cycle_weight(cycle: Tuple[int, ...],
                 edges: Dict[Tuple[int, int], float]) -> float:
    """Sum of omega_HLA along the cycle (used as tie-break)."""
    return sum(
        edges[(cycle[k], cycle[(k + 1) % len(cycle)])]
        for k in range(len(cycle))
    )


# ============================================================
# GENERIC SET-PACKING SOLVER
# ============================================================

def _solve_packing(
    cycles: List[Tuple[int, ...]],
    values: np.ndarray,
    extra_ge: Optional[Tuple[np.ndarray, float]] = None,
    time_limit: int = ILP_TIME_LIMIT,
    msg: bool = False
) -> Tuple[List[int], str]:
    """
    max  sum_c values[c] * x_c
    s.t. each pair appears in at most one chosen cycle
         (optional) sum_c a[c] * x_c >= rhs
         x_c in {0, 1}

    Returns (indices of chosen cycles, solver name).
    """
    if len(cycles) == 0:
        return [], "none"

    try:
        import pulp
        solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
        if not solver.available():
            raise ImportError
    except (ImportError, AttributeError):
        return _solve_packing_highs(cycles, values, extra_ge,
                                    time_limit), "highs"

    prob = pulp.LpProblem("KEP_cycle_formulation", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{c}", cat="Binary")
         for c in range(len(cycles))]
    prob += pulp.lpSum(float(values[c]) * x[c]
                       for c in range(len(cycles)))

    # Each pair in at most one cycle
    by_pair: Dict[int, List[int]] = {}
    for c, cyc in enumerate(cycles):
        for p in cyc:
            by_pair.setdefault(p, []).append(c)
    for p, cs in by_pair.items():
        if len(cs) > 1:
            prob += pulp.lpSum(x[c] for c in cs) <= 1, f"pair_{p}"

    if extra_ge is not None:
        a, rhs = extra_ge
        prob += pulp.lpSum(float(a[c]) * x[c]
                           for c in range(len(cycles))) >= rhs

    prob.solve(solver)
    chosen = [c for c in range(len(cycles))
              if x[c].value() is not None and x[c].value() > 0.5]
    return chosen, "cbc"


def _solve_packing_highs(cycles, values, extra_ge, time_limit):
    """Same model solved with SciPy HiGHS (fallback)."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix

    ids = sorted({p for cyc in cycles for p in cyc})
    row = {p: k for k, p in enumerate(ids)}
    r, c = [], []
    for k, cyc in enumerate(cycles):
        for p in cyc:
            r.append(row[p])
            c.append(k)
    A = csr_matrix((np.ones(len(r)), (r, c)),
                   shape=(len(ids), len(cycles)))
    cons = [LinearConstraint(A, -np.inf, 1.0)]
    if extra_ge is not None:
        a, rhs = extra_ge
        cons.append(LinearConstraint(a[None, :], rhs, np.inf))
    res = milp(c=-np.asarray(values, float), constraints=cons,
               integrality=np.ones(len(cycles)), bounds=Bounds(0, 1),
               options={"time_limit": time_limit})
    if res.x is None:
        raise RuntimeError(f"HiGHS failed: {res.message}")
    return [k for k in range(len(cycles)) if res.x[k] > 0.5]


# ============================================================
# BASELINE 7: MYOPIC ILP (maximum transplants)
# ============================================================

def solve_ilp_max_transplants(
    pool_pairs: List[Patient],
    edges: Optional[Dict] = None,
    max_len: int = MAX_CYCLE_LEN,
    hla_tiebreak: bool = True,
    time_limit: int = ILP_TIME_LIMIT
) -> Tuple[Set[int], float, Dict]:
    """
    Solve the maximum-transplant ILP for the current pool.

    Returns:
        matched_ids: set of pair IDs in the optimal cycles
        f1_ilp:      number of transplants (= F1_ILP(t))
        info:        solver statistics
    """
    t0 = time.time()
    if edges is None:
        edges = build_compatibility_graph(pool_pairs)
    pair_ids = [p.pair_id for p in pool_pairs]
    cycles = enumerate_cycles(edges, pair_ids, max_len)

    sizes = np.array([len(c) for c in cycles], dtype=float)
    values = sizes.copy()
    if hla_tiebreak and len(cycles):
        # Tiny HLA bonus breaks ties between equally large
        # solutions without ever trading away a transplant.
        hla = np.array([cycle_weight(c, edges) for c in cycles])
        values = values + 1e-3 * hla / max(1.0, hla.sum())

    chosen, solver = _solve_packing(cycles, values,
                                    time_limit=time_limit)
    matched: Set[int] = set()
    for k in chosen:
        matched.update(cycles[k])

    info = {
        "n_cycles": len(cycles),
        "n_edges": len(edges),
        "solver": solver,
        "seconds": time.time() - t0,
        "cycles_chosen": [cycles[k] for k in chosen],
    }
    return matched, float(len(matched)), info


# ============================================================
# BASELINE 8: DICKERSON LEXICOGRAPHIC FAIRNESS
# ============================================================

def solve_ilp_lexicographic(
    pool_pairs: List[Patient],
    edges: Optional[Dict] = None,
    max_len: int = MAX_CYCLE_LEN,
    time_limit: int = ILP_TIME_LIMIT
) -> Tuple[Set[int], float, Dict]:
    """
    Lexicographic fairness baseline (Dickerson et al. 2014):
      Stage 1: maximise number of high-cPRA patients matched.
      Stage 2: maximise total transplants, keeping stage-1 value.
    """
    t0 = time.time()
    if edges is None:
        edges = build_compatibility_graph(pool_pairs)
    pair_ids = [p.pair_id for p in pool_pairs]
    high = {p.pair_id for p in pool_pairs
            if p.is_highly_sensitised and not p.is_altruistic}
    cycles = enumerate_cycles(edges, pair_ids, max_len)

    n_high = np.array([sum(1 for p in c if p in high)
                       for c in cycles], dtype=float)
    sizes = np.array([len(c) for c in cycles], dtype=float)

    stage1, solver = _solve_packing(cycles, n_high,
                                    time_limit=time_limit)
    best_high = float(n_high[stage1].sum()) if stage1 else 0.0
    stage2, _ = _solve_packing(cycles, sizes,
                               extra_ge=(n_high, best_high - 1e-6),
                               time_limit=time_limit)

    matched: Set[int] = set()
    for k in stage2:
        matched.update(cycles[k])
    info = {
        "n_cycles": len(cycles),
        "high_cpra_matched": int(best_high),
        "solver": solver,
        "seconds": time.time() - t0,
    }
    return matched, float(len(matched)), info


def verify_disjoint(cycles: List[Tuple[int, ...]]) -> bool:
    """Check no pair appears in two chosen cycles."""
    seen: Set[int] = set()
    for c in cycles:
        if seen & set(c):
            return False
        seen.update(c)
    return True


# ============================================================
# TEST MODULE 6
# ============================================================

if __name__ == "__main__":

    from module1_generator import generate_initial_pool
    from module4_objectives import compute_pof
    from module5_nsga3_aos import (
        UCB1AOS, extract_best_solution, run_nsga3_with_aos
    )

    print("=" * 55)
    print("MODULE 6: ILP UPPER BOUND (CBC)")
    print(f"Cycle length <= {MAX_CYCLE_LEN} (same as module 5)")
    print("=" * 55)

    pool, next_id = generate_initial_pool(n_pairs=75, seed=42)
    edges = build_compatibility_graph(pool.pairs)

    matched, f1_ilp, info = solve_ilp_max_transplants(
        pool.pairs, edges
    )
    print(f"\nPool size:        {pool.size()}")
    print(f"Feasible edges:   {info['n_edges']}")
    print(f"Cycles (2..{MAX_CYCLE_LEN}):    {info['n_cycles']}")
    print(f"Solver:           {info['solver']}")
    print(f"F1_ILP:           {f1_ilp:.0f} transplants")
    print(f"Solve time:       {info['seconds']:.2f} s")
    print(f"Cycles disjoint:  "
          f"{verify_disjoint(info['cycles_chosen'])}")

    lex_matched, f1_lex, lex_info = solve_ilp_lexicographic(
        pool.pairs, edges
    )
    print(f"\n--- Dickerson Lexicographic ---")
    print(f"High-cPRA matched: {lex_info['high_cpra_matched']}")
    print(f"F1_LEX:            {f1_lex:.0f} transplants")
    print(f"PoF (lexicographic vs ILP): "
          f"{compute_pof(f1_ilp, f1_lex):.2f}%")

    print(f"\n--- MOEA vs ILP on the same pool ---")
    result, hv, arm, rate = run_nsga3_with_aos(
        pool.pairs, 'ucb1', ucb1_aos=UCB1AOS(), n_gen=30, seed=1
    )
    moea = extract_best_solution(result, pool.pairs, edges)
    print(f"F1_MOEA (max-F1 point): {len(moea)}")
    print(f"PoF:                    "
          f"{compute_pof(f1_ilp, len(moea)):.2f}%")
    assert len(moea) <= f1_ilp, "MOEA exceeded ILP ceiling!"

    print("\n" + "=" * 55)
    print("MODULE 6 COMPLETE")
    print("Ready for Module 7: Clearing Strategies")
    print("=" * 55)
