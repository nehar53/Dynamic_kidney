import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from module1_generator import Patient
from module3_logistics import (
    MAX_CHAIN_LEN, MAX_CYCLE_LEN, ExchangeStructures,
    build_compatibility_graph, build_exchange_structures, enumerate_cycles,
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
    prob.solve(solver)
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

    solver = _cbc_solver(time_limit) or pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    f1 = float(pulp.value(prob.objective) or 0.0)
    return round(f1), {"n_cycles": len(cycles), "n_arc_vars": len(y),
                       "seconds": time.time() - t0}


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
