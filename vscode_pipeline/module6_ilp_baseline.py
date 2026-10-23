import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from module1_generator import Patient
from module3_logistics import (
    MAX_CHAIN_LEN, MAX_CYCLE_LEN, ExchangeStructures,
    build_compatibility_graph, classify_node, build_exchange_structures, enumerate_cycles,
)

# ============================================================
# MODULE 6: ILP BASELINES (CBC)
#   Baseline 7: Myopic ILP  - maximum transplants, F1_ILP(t)
#   Baseline 8: Dickerson lexicographic fairness (Dickerson 2014)
# Concept note Sections 3.6 and 4.4
# Paper 2 — Dynamic KEP Implementation
# ============================================================
#
# Identical feasibility (Section 3.6): both ILPs optimise over
# EXACTLY the exchange structures of module 3 (2-/3-cycles and
# altruistic chains up to 4) that module 5's NSGA-III repairs to.
# So F1_ILP(t) is a true ceiling for every MOEA solution and
# PoF_t = (F1_ILP - F1_MOEA) / F1_ILP measures only the cost of
# fairness.
#
# Set-packing (cycle-and-chain) formulation:
#   max  sum_s |recipients(s)| x_s
#   s.t. sum_{s ∋ v} x_s <= 1      for every pair / altruist v
#        x_s in {0, 1}
#
# Exact chain check (optional, for the paper): module 3 caps the
# number of chains (MAX_CHAINS). solve_ilp_picef() solves the
# position-indexed chain formulation (PICEF, Dickerson et al. 2016)
# with ALL chains up to length L, to show how many transplants
# the cap costs (normally 0).
#
# Solver: CBC via PuLP 2.x (bundled). PuLP >= 3 no longer ships
# CBC; then SciPy's HiGHS is used, which gives the same optimum.

ILP_TIME_LIMIT = 120       # seconds per solve
TWO_STAGE_TIME_LIMIT = 60  # stage 2 of the two-stage ILP; if reached, the
                           # best matching found so far is used (reported)
TIE_EPS_FAIR = 1e-5        # two-stage tie-break: secondary fairness terms
TIE_EPS_HLA = 1e-7         # two-stage tie-break: HLA weight


# ============================================================
# GENERIC SET-PACKING SOLVER
# ============================================================

def _cbc_solver(time_limit: int, msg: bool = False):
    try:
        import pulp
        solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
        return solver if solver.available() else None
    except (ImportError, AttributeError):
        return None


def _cbc_solve(prob, time_limit: int) -> bool:
    """
    Solve a PuLP model with CBC. On Windows the CBC executable very
    occasionally exits without writing a solution (PulpSolverError);
    it is retried once. Returns False if CBC still fails, so the
    caller can fall back to HiGHS.
    """
    import pulp
    for _ in range(2):
        solver = _cbc_solver(time_limit) or pulp.PULP_CBC_CMD(msg=False)
        try:
            prob.solve(solver)
            return True
        except (pulp.PulpSolverError, MemoryError):
            continue
    return False


def _solve_packing(
    nodes: List[Tuple[int, ...]],
    values: np.ndarray,
    extra_ge: Optional[Tuple[np.ndarray, float]] = None,
    time_limit: int = ILP_TIME_LIMIT,
) -> Tuple[List[int], str]:
    """
    max  sum_k values[k] x_k
    s.t. every node in at most one chosen structure
         (optional) sum_k a[k] x_k >= rhs
    Returns (chosen structure indices, solver name).
    """
    if len(nodes) == 0:
        return [], "none"
    solver = _cbc_solver(time_limit)
    if solver is None:
        return _solve_packing_highs(nodes, values, extra_ge, time_limit), "highs"

    import pulp
    prob = pulp.LpProblem("KEP_set_packing", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{k}", cat="Binary") for k in range(len(nodes))]
    prob += pulp.lpSum(float(values[k]) * x[k] for k in range(len(nodes)))
    by_node: Dict[int, List[int]] = {}
    for k, s in enumerate(nodes):
        for v in s:
            by_node.setdefault(v, []).append(k)
    for v, ks in by_node.items():
        if len(ks) > 1:
            prob += pulp.lpSum(x[k] for k in ks) <= 1, f"node_{v}"
    if extra_ge is not None:
        a, rhs = extra_ge
        prob += pulp.lpSum(float(a[k]) * x[k] for k in range(len(nodes))) >= rhs
    if not _cbc_solve(prob, time_limit):
        return _solve_packing_highs(nodes, values, extra_ge, time_limit), "highs"
    return [k for k in range(len(nodes))
            if x[k].value() is not None and x[k].value() > 0.5], "cbc"


def _solve_packing_highs(nodes, values, extra_ge, time_limit):
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix

    ids = sorted({v for s in nodes for v in s})
    row = {v: r for r, v in enumerate(ids)}
    r, c = [], []
    for k, s in enumerate(nodes):
        for v in s:
            r.append(row[v])
            c.append(k)
    A = csr_matrix((np.ones(len(r)), (r, c)), shape=(len(ids), len(nodes)))
    cons = [LinearConstraint(A, -np.inf, 1.0)]
    if extra_ge is not None:
        a, rhs = extra_ge
        cons.append(LinearConstraint(np.asarray(a, float)[None, :], rhs, np.inf))
    res = milp(c=-np.asarray(values, float), constraints=cons,
               integrality=np.ones(len(nodes)), bounds=Bounds(0, 1),
               options={"time_limit": time_limit})
    if res.x is None:
        raise RuntimeError(f"HiGHS failed: {res.message}")
    return [k for k in range(len(nodes)) if res.x[k] > 0.5]


def _structures(pool_pairs, edges, structures):
    if edges is None:
        edges = build_compatibility_graph(pool_pairs)
    if structures is None:
        structures = build_exchange_structures(pool_pairs, edges)
    return edges, structures


def _result(structures: ExchangeStructures, chosen: List[int]):
    matched: Set[int] = set()
    for k in chosen:
        matched.update(structures.nodes[k])     # includes altruists used
    f1 = float(sum(structures.size(k) for k in chosen))
    return matched, f1


# ============================================================
# BASELINE 7: MYOPIC ILP (maximum transplants)
# ============================================================

def solve_ilp_max_transplants(
    pool_pairs: List[Patient],
    edges: Optional[Dict] = None,
    structures: Optional[ExchangeStructures] = None,
    hla_tiebreak: bool = True,
    time_limit: int = ILP_TIME_LIMIT,
    max_len: int = MAX_CYCLE_LEN,
) -> Tuple[Set[int], float, Dict]:
    """
    Maximum-transplant ILP for the current pool.

    Returns
      matched_ids  pair IDs in the chosen structures (incl. altruists)
      f1_ilp       transplants = F1_ILP(t)
      info         'structure_indices' (pass to module 5 as
                   guide_structures), counts, solver, time
    """
    t0 = time.time()
    edges, structures = _structures(pool_pairs, edges, structures)
    sizes = np.array([structures.size(k) for k in range(len(structures))], float)
    values = sizes.copy()
    if hla_tiebreak and len(structures):
        # tiny HLA bonus breaks ties without trading away a transplant
        w = np.asarray(structures.weight, float)
        values = values + 1e-3 * w / max(1.0, w.sum())
    chosen, solver = _solve_packing(structures.nodes, values, time_limit=time_limit)
    matched, f1 = _result(structures, chosen)
    info = {
        "structure_indices": chosen,
        "cycles_chosen": [structures.nodes[k] for k in chosen
                          if structures.kind[k] == "cycle"],
        "chains_chosen": [structures.nodes[k] for k in chosen
                          if structures.kind[k] == "chain"],
        "n_cycles": structures.n_cycles(),
        "n_chains": structures.n_chains(),
        "chains_truncated": structures.truncated,
        "n_edges": len(edges),
        "solver": solver,
        "seconds": time.time() - t0,
        "structures": structures,
        "edges": edges,
    }
    return matched, f1, info


# ============================================================
# BASELINE 8: DICKERSON LEXICOGRAPHIC FAIRNESS
# ============================================================

def solve_ilp_lexicographic(
    pool_pairs: List[Patient],
    edges: Optional[Dict] = None,
    structures: Optional[ExchangeStructures] = None,
    time_limit: int = ILP_TIME_LIMIT,
    max_len: int = MAX_CYCLE_LEN,
) -> Tuple[Set[int], float, Dict]:
    """
    Dickerson et al. (2014) lexicographic fairness:
      stage 1: maximise high-cPRA patients transplanted
      stage 2: maximise total transplants, keeping stage 1 optimal
    """
    t0 = time.time()
    edges, structures = _structures(pool_pairs, edges, structures)
    high = {p.pair_id for p in pool_pairs
            if p.is_highly_sensitised and not p.is_altruistic}
    n_high = np.array([sum(v in high for v in structures.recipients[k])
                       for k in range(len(structures))], float)
    sizes = np.array([structures.size(k) for k in range(len(structures))], float)

    stage1, solver = _solve_packing(structures.nodes, n_high, time_limit=time_limit)
    best_high = float(n_high[stage1].sum()) if stage1 else 0.0
    stage2, _ = _solve_packing(structures.nodes, sizes,
                               extra_ge=(n_high, best_high - 1e-6),
                               time_limit=time_limit)
    matched, f1 = _result(structures, stage2)
    return matched, f1, {
        "structure_indices": stage2,
        "high_cpra_matched": int(best_high),
        "solver": solver,
        "seconds": time.time() - t0,
    }


# ============================================================
# EXACT CHAIN CHECK: PICEF (all chains up to L, no cap)
# ============================================================

# ============================================================
# BASELINE 9: TWO-STAGE ILP (fairest maximum matching)
# ============================================================
#
# Stage 1 is the volume-maximising ILP (F1* transplants). Stage 2
# keeps exactly F1* transplants and minimises fairness instead:
#   "f3"   F3 alone (high-cPRA access gap)
#   "sum"  F2/F2_ref + F3/F3_ref + F4/F4_ref, with the values of the
#          stage-1 matching as references (floored at 0.01), i.e. the
#          total relative improvement over the volume-optimal matching.
# With F1 fixed all three objectives are linear or linearisable:
#   F3, F4  differences of match rates with constant denominators
#           (|.| via one auxiliary variable each);
#   F2      Gini of w'_i = w_i + 1 - m_i (integer waits). The sum of
#           w' is fixed, and |w'_i - w'_j| is linear in m when
#           w_i != w_j, so only equal-wait pairs need |m_i - m_j|.
# This is the exact answer to "pick the fairest maximum matching"
# for these objectives (Section 6, reviewer question "why an MOEA?").
# Ties (several equally fair maximum matchings) are broken
# deterministically by small secondary terms: for "f3" the logistics
# gap F4 (weight TIE_EPS_FAIR), and for both objectives the
# HLA weight (TIE_EPS_HLA), as in the volume ILP. Both weights are
# far below the smallest possible change of the primary objective,
# and CBC is deterministic, so the same pool always gives the same
# matching.

def solve_ilp_two_stage(
    pool_pairs: List[Patient],
    edges: Dict,
    structures: ExchangeStructures,
    f1_star: float,
    ref: Tuple[float, float, float],
    objective: str = "sum",
    time_limit: int = TWO_STAGE_TIME_LIMIT,
) -> Tuple[Set[int], Dict]:
    import pulp
    t0 = time.time()
    pts = [p for p in pool_pairs if not p.is_altruistic]
    if not pts or len(structures) == 0 or f1_star <= 0:
        return set(), {"seconds": time.time() - t0, "status": "trivial"}
    n = len(pts)
    pid = [p.pair_id for p in pts]
    pos = {q: i for i, q in enumerate(pid)}
    by_patient: Dict[int, List[int]] = {}
    for k in range(len(structures)):
        for v in structures.recipients[k]:
            if v in pos:
                by_patient.setdefault(pos[v], []).append(k)

    prob = pulp.LpProblem("KEP_two_stage", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{k}", cat="Binary") for k in range(len(structures))]
    by_node: Dict[int, List[int]] = {}
    for k, s in enumerate(structures.nodes):
        for v in s:
            by_node.setdefault(v, []).append(k)
    for v, ks in by_node.items():
        if len(ks) > 1:
            prob += pulp.lpSum(x[k] for k in ks) <= 1
    prob += pulp.lpSum(structures.size(k) * x[k]
                       for k in range(len(structures))) == round(f1_star)
    m = [pulp.lpSum(x[k] for k in by_patient.get(i, [])) for i in range(n)]

    # F3
    S = [i for i, p in enumerate(pts) if p.is_highly_sensitised]
    f3 = 0
    if S:
        t3 = pulp.LpVariable("t3", lowBound=0)
        rate = pulp.lpSum(m[i] for i in S) / len(S) - round(f1_star) / n
        prob += t3 >= rate
        prob += t3 >= -rate
        f3 = t3
    # F4
    R = [i for i, p in enumerate(pts) if classify_node(p.hospital_node) == 'R']
    U = [i for i, p in enumerate(pts) if classify_node(p.hospital_node) == 'U']
    f4 = 0
    if R and U:
        t4 = pulp.LpVariable("t4", lowBound=0)
        diff = (pulp.lpSum(m[i] for i in R) / len(R)
                - pulp.lpSum(m[i] for i in U) / len(U))
        prob += t4 >= diff
        prob += t4 >= -diff
        f4 = t4
    # F2: Gini(w'), w'_i = w_i + 1 - m_i, sum(w') fixed ("sum" only)
    w = np.array([p.waiting_time for p in pts], dtype=float)
    total = w.sum() + n - round(f1_star)
    f2 = 0
    if objective != "f3" and n >= 2 and total > 0:
        # sum_{i,j} |w'_i - w'_j| = 2 sum_{i<j} |.|
        const, coef = 0.0, np.zeros(n)
        eq_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                c = w[i] - w[j]
                if c > 0:            # w'_i - w'_j = c - m_i + m_j >= 0
                    const += c; coef[i] -= 1; coef[j] += 1
                elif c < 0:
                    const -= c; coef[i] += 1; coef[j] -= 1
                else:
                    eq_pairs.append((i, j))
        d = []
        for (i, j) in eq_pairs:
            v = pulp.LpVariable(f"d_{i}_{j}", lowBound=0)
            prob += v >= m[i] - m[j]
            prob += v >= m[j] - m[i]
            d.append(v)
        num = const + pulp.lpSum(coef[i] * m[i] for i in range(n)) + pulp.lpSum(d)
        f2 = (2.0 / (2.0 * n * total)) * num
    r2, r3, r4 = (max(r, 0.01) for r in ref)
    wmax = max(float(np.max(structures.weight)) if len(structures) else 1.0, 1e-9)
    hla = pulp.lpSum((structures.weight[k] / wmax) * x[k]
                     for k in range(len(structures)))
    if objective == "f3":
        prob += f3 + TIE_EPS_FAIR * f4 * (1.0 / r4) - TIE_EPS_HLA * hla
    else:
        prob += f2 * (1.0 / r2) + f3 * (1.0 / r3) + f4 * (1.0 / r4) - TIE_EPS_HLA * hla

    if not _cbc_solve(prob, time_limit):
        return set(), {"seconds": time.time() - t0, "status": "failed"}
    chosen = [k for k in range(len(structures)) if (x[k].value() or 0) > 0.5]
    matched: Set[int] = set()
    for k in chosen:
        matched.update(structures.nodes[k])
    secs = time.time() - t0
    return matched, {"seconds": secs, "structure_indices": chosen,
                     "status": "time_limit" if secs >= time_limit - 1 else "ok"}


def solve_ilp_picef(
    pool_pairs: List[Patient],
    edges: Optional[Dict] = None,
    max_cycle_len: int = MAX_CYCLE_LEN,
    max_chain_len: int = MAX_CHAIN_LEN,
    time_limit: int = ILP_TIME_LIMIT,
) -> Tuple[float, Dict]:
    """
    Position-Indexed Chain-Edge Formulation (Dickerson et al. 2016):
    cycles enumerated explicitly, chains modelled by arc-position
    variables y[i,j,k] (arc i->j used at position k of a chain).
    Gives the TRUE optimum with all chains up to max_chain_len.
    Used only to verify that module 3's chain cap loses nothing.
    """
    import pulp
    if edges is None:
        edges = build_compatibility_graph(pool_pairs)
    t0 = time.time()
    alts = {p.pair_id for p in pool_pairs if p.is_altruistic}
    pts = [p.pair_id for p in pool_pairs if not p.is_altruistic]
    cycles = enumerate_cycles(edges, [p.pair_id for p in pool_pairs], max_cycle_len)

    prob = pulp.LpProblem("PICEF", pulp.LpMaximize)
    x = {k: pulp.LpVariable(f"c_{k}", cat="Binary") for k in range(len(cycles))}
    y = {}
    for (i, j) in edges:
        positions = [1] if i in alts else range(2, max_chain_len + 1)
        for k in positions:
            y[(i, j, k)] = pulp.LpVariable(f"y_{i}_{j}_{k}", cat="Binary")
    prob += (pulp.lpSum(len(cycles[k]) * x[k] for k in x) + pulp.lpSum(y.values()))

    into: Dict[Tuple[int, int], list] = {}
    out_of: Dict[Tuple[int, int], list] = {}
    for (i, j, k), var in y.items():
        into.setdefault((j, k), []).append(var)
        out_of.setdefault((i, k), []).append(var)
    in_cycle: Dict[int, list] = {}
    for k, c in enumerate(cycles):
        for v in c:
            in_cycle.setdefault(v, []).append(x[k])

    for v in pts:       # each patient receives at most one kidney
        prob += (pulp.lpSum(in_cycle.get(v, [])) +
                 pulp.lpSum(var for k in range(1, max_chain_len + 1)
                            for var in into.get((v, k), []))) <= 1
    for a in alts:      # each altruist starts at most one chain
        prob += pulp.lpSum(out_of.get((a, 1), [])) <= 1
    for v in pts:       # chain continues only if v received at position k
        for k in range(1, max_chain_len):
            prob += (pulp.lpSum(out_of.get((v, k + 1), [])) <=
                     pulp.lpSum(into.get((v, k), [])))

    if not _cbc_solve(prob, time_limit):
        # PICEF only adds chains the capped enumeration missed; if CBC
        # fails twice the round keeps its enumerated chains.
        return 0, {"n_cycles": len(cycles), "n_arc_vars": len(y),
                   "chains": [], "failed": True,
                   "seconds": time.time() - t0}
    f1 = float(pulp.value(prob.objective) or 0.0)
    # chains of the optimal matching: follow the used arcs from each altruist
    used = {(i, k): j for (i, j, k), var in y.items()
            if (var.value() or 0) > 0.5}
    chains = []
    for a in sorted(alts):
        if (a, 1) in used:
            chain, node, k = [a], a, 1
            while (node, k) in used:
                node = used[(node, k)]
                chain.append(node)
                k += 1
            chains.append(tuple(chain))
    return round(f1), {"n_cycles": len(cycles), "n_arc_vars": len(y),
                       "chains": chains, "seconds": time.time() - t0}


def verify_disjoint(structs: List[Tuple[int, ...]]) -> bool:
    """No pair or altruist appears in two chosen structures."""
    seen: Set[int] = set()
    for s in structs:
        if seen & set(s):
            return False
        seen.update(s)
    return True


# ============================================================
# TEST MODULE 6
# ============================================================

if __name__ == "__main__":

    from module1_generator import generate_initial_pool
    from module4_objectives import compute_pof, round_metrics
    from module5_nsga3_aos import UCB1AOS, run_moea_round

    print("=" * 60)
    print("MODULE 6: ILP BASELINES (CBC)")
    print(f"Structures: cycles <= {MAX_CYCLE_LEN}, chains <= {MAX_CHAIN_LEN} "
          f"(same list as module 5)")
    print("=" * 60)

    pool, _ = generate_initial_pool(n_pairs=75, seed=42, initial_wait_max=12)
    pairs = pool.pairs
    edges = build_compatibility_graph(pairs)
    st = build_exchange_structures(pairs, edges)
    print(f"\nPool: {len(pool.patients())} pairs + {len(pool.altruists())} altruists, "
          f"{len(edges)} edges, {st.n_cycles()} cycles, {st.n_chains()} chains "
          f"(truncated={st.truncated})")

    # --- Baseline 7 ---
    m_ilp, f1_ilp, info = solve_ilp_max_transplants(pairs, edges, st)
    print("\n--- Baseline 7: Myopic ILP (max transplants) ---")
    print(f"  Solver:            {info['solver']}  ({info['seconds']:.2f} s)")
    print(f"  F1_ILP:            {f1_ilp:.0f} transplants")
    print(f"  Cycles chosen:     {len(info['cycles_chosen'])}  "
          f"Chains chosen: {len(info['chains_chosen'])}")
    print(f"  Disjoint:          "
          f"{verify_disjoint(info['cycles_chosen'] + info['chains_chosen'])}")
    rm = round_metrics(m_ilp, pairs)
    print(f"  High-cPRA match rate: {rm['match_rate_high_cpra'] * 100:.1f}%  "
          f"low-cPRA: {rm['match_rate_low_cpra'] * 100:.1f}%")

    # --- Cycles only (old module 6) for comparison ---
    cyc = [k for k in range(len(st)) if st.kind[k] == "cycle"]
    sizes = np.array([st.size(k) for k in cyc], float)
    ch, _ = _solve_packing([st.nodes[k] for k in cyc], sizes)
    print(f"  (cycles only, no chains: {sizes[ch].sum():.0f} transplants)")

    # --- Exact chain check ---
    f1_exact, pinfo = solve_ilp_picef(pairs, edges)
    print("\n--- Exact check: PICEF with ALL chains up to L ---")
    print(f"  F1 exact:          {f1_exact:.0f}  ({pinfo['seconds']:.2f} s)")
    print(f"  Chain cap costs:   {f1_exact - f1_ilp:.0f} transplant(s)")

    # --- Baseline 8 ---
    m_lex, f1_lex, linfo = solve_ilp_lexicographic(pairs, edges, st)
    rml = round_metrics(m_lex, pairs)
    print("\n--- Baseline 8: Dickerson lexicographic ---")
    print(f"  High-cPRA matched: {linfo['high_cpra_matched']}  "
          f"(ILP: {sum(1 for p in pool.get_highly_sensitised() if p.pair_id in m_ilp)})")
    print(f"  F1_LEX:            {f1_lex:.0f} transplants")
    print(f"  High-cPRA match rate: {rml['match_rate_high_cpra'] * 100:.1f}%")
    print(f"  PoF (lexicographic vs ILP): {compute_pof(f1_ilp, f1_lex):.2f}%")

    # --- MOEA on identical structures ---
    print("\n--- NSGA-III (ILP-guided) on the same structures ---")
    res = run_moea_round(pairs, UCB1AOS(seed=1), None, "nsga3", n_gen=40,
                         seed=1, edges=edges, structures=st,
                         guide_structures=info["structure_indices"])
    f1s = -res.F[:, 0]
    print(f"  Front: {len(f1s)} solutions, F1 {f1s.min():.0f}..{f1s.max():.0f}")
    print(f"  Max F1 on front == F1_ILP: {f1s.max() == f1_ilp}")
    print(f"  No MOEA solution above the ILP ceiling: {bool((f1s <= f1_ilp).all())}")

    print("\n" + "=" * 60)
    print("MODULE 6 COMPLETE")
    print("Ready for Module 7: Clearing Strategies")
    print("=" * 60)
