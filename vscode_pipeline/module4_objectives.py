import numpy as np
from typing import List, Dict, Tuple, Set
from module1_generator import Patient, ANZKX_NODES
from module3_logistics import (
    DISTANT_NODES,
    CENTRAL_NODES,
    classify_node
)

# ============================================================
# MODULE 4: FOUR FAIRNESS OBJECTIVES
# F1: Transplant Efficiency
# F2: Waiting-Time Equity (Gini)
# F3: Sensitised Patient Access (GFG)
# F4: Node-Level Logistics Disparity
# Paper 2 — Dynamic KEP Implementation
# ============================================================

# High cPRA threshold for sensitised patients
CPRA_THRESHOLD = 0.80


def compute_f1_efficiency(
    matched_pair_ids: Set[int],
    pool_pairs: List[Patient]
) -> float:
    """
    F1: Transplant Efficiency
    Total matched pairs per round.
   
    F1(x) = sum of |c| for all cycles c in C(x)
   
    Higher is better.
    Minimisation: negate F1 in NSGA-III.
    """
    return float(len(matched_pair_ids))


def compute_f2_gini(
    pool_pairs: List[Patient]
) -> float:
    """
    F2: Waiting-Time Equity
    Gini coefficient across all waiting times.
   
    F2(x) = sum_i sum_j |w_i - w_j|
            / (2n * sum_i w_i)
   
    Range: 0 (perfect equality) to 1 (maximum inequality).
    Lower is better.
    Minimise F2 directly.
    """
    n = len(pool_pairs)
    if n == 0:
        return 0.0

    waiting_times = np.array([
        p.waiting_time for p in pool_pairs
    ], dtype=float)

    total_waiting = np.sum(waiting_times)
    if total_waiting == 0:
        return 0.0

    # Compute sum of absolute differences
    diff_sum = 0.0
    for i in range(n):
        for j in range(n):
            diff_sum += abs(
                waiting_times[i] - waiting_times[j]
            )

    gini = diff_sum / (2 * n * total_waiting)
    return float(np.clip(gini, 0.0, 1.0))


def compute_f2_gini_fast(
    pool_pairs: List[Patient]
) -> float:
    """
    F2: Fast vectorised Gini computation.
    Same result as compute_f2_gini but faster.
    Used in actual NSGA-III runs.
    """
    n = len(pool_pairs)
    if n == 0:
        return 0.0

    waiting_times = np.array([
        p.waiting_time for p in pool_pairs
    ], dtype=float)

    total_waiting = np.sum(waiting_times)
    if total_waiting == 0:
        return 0.0

    # Vectorised absolute difference
    w = waiting_times.reshape(-1, 1)
    diff_sum = np.sum(np.abs(w - w.T))

    gini = diff_sum / (2 * n * total_waiting)
    return float(np.clip(gini, 0.0, 1.0))


def compute_f3_group_fairness(
    matched_pair_ids: Set[int],
    pool_pairs: List[Patient]
) -> float:
    """
    F3: Sensitised Patient Access
    Group Fairness Gap for high-cPRA patients.
   
    F3(x) = |matched_rate_sensitised
              - matched_rate_overall|
   
    Where:
    matched_rate_sensitised = |M ∩ S| / |S|
    matched_rate_overall    = |M| / |V_t|
   
    S = high-cPRA patients (cPRA >= 80%)
    M = matched pairs
    V_t = all active pairs
   
    Range: 0 (perfect group fairness) to 1.
    Lower is better.
    Minimise F3 directly.
    """
    n_total = len(pool_pairs)
    if n_total == 0:
        return 0.0

    # Get sensitised patients
    sensitised = [
        p for p in pool_pairs
        if p.is_highly_sensitised
    ]
    n_sensitised = len(sensitised)

    if n_sensitised == 0:
        return 0.0

    # Count matched sensitised
    matched_sensitised = sum(
        1 for p in sensitised
        if p.pair_id in matched_pair_ids
    )

    # Matched rates
    rate_sensitised = matched_sensitised / n_sensitised
    rate_overall = len(matched_pair_ids) / n_total

    gfg = abs(rate_sensitised - rate_overall)
    return float(np.clip(gfg, 0.0, 1.0))


def compute_f4_logistics_disparity(
    matched_pair_ids: Set[int],
    pool_pairs: List[Patient]
) -> float:
    """
    F4: Node-Level Logistics Disparity
    Modelled disparity in match rates between
    geographically distant and central nodes.
   
    F4(x) = ||M ∩ R| / |R| - |M ∩ U| / |U||
   
    R = geographically distant nodes
        (Perth, Auckland)
    U = centrally located nodes
        (Adelaide, Melbourne, Sydney, Brisbane)
   
    Classification based on transport distances only.
    Does NOT imply rural vs metropolitan inequity.
    This is a modelled logistics proxy only.
   
    Range: 0 (no disparity) to 1.
    Lower is better.
    Minimise F4 directly.
    """
    # Get distant and central patients
    distant_patients = [
        p for p in pool_pairs
        if classify_node(p.hospital_node) == 'R'
    ]
    central_patients = [
        p for p in pool_pairs
        if classify_node(p.hospital_node) == 'U'
    ]

    n_distant = len(distant_patients)
    n_central = len(central_patients)

    if n_distant == 0 or n_central == 0:
        return 0.0

    # Count matched in each group
    matched_distant = sum(
        1 for p in distant_patients
        if p.pair_id in matched_pair_ids
    )
    matched_central = sum(
        1 for p in central_patients
        if p.pair_id in matched_pair_ids
    )

    rate_distant = matched_distant / n_distant
    rate_central = matched_central / n_central

    disparity = abs(rate_distant - rate_central)
    return float(np.clip(disparity, 0.0, 1.0))


def compute_all_objectives(
    matched_pair_ids: Set[int],
    pool_pairs: List[Patient]
) -> Dict[str, float]:
    """
    Compute all four objectives for a given solution.
   
    Returns dict with F1 F2 F3 F4 values.
   
    For NSGA-III minimisation:
    F1 is negated (maximise transplants).
    F2 minimised directly (lower Gini = more equal).
    F3 minimised directly (lower GFG = more fair).
    F4 minimised directly (lower disparity = more fair).
    """
    f1 = compute_f1_efficiency(
        matched_pair_ids, pool_pairs
    )
    f2 = compute_f2_gini_fast(pool_pairs)
    f3 = compute_f3_group_fairness(
        matched_pair_ids, pool_pairs
    )
    f4 = compute_f4_logistics_disparity(
        matched_pair_ids, pool_pairs
    )

    return {
        'F1': f1,           # Raw value (higher better)
        'F1_neg': -f1,      # Negated for minimisation
        'F2': f2,           # Gini (lower better)
        'F3': f3,           # GFG (lower better)
        'F4': f4,           # Disparity (lower better)
        'n_matched': len(matched_pair_ids),
        'n_pool': len(pool_pairs)
    }


def compute_pof(
    f1_ilp: float,
    f1_moea: float
) -> float:
    """
    Compute Price of Fairness at round t.
   
    PoF_t = (F1_ILP(t) - F1_MOEA(t))
            / F1_ILP(t) × 100%
   
    F1_ILP and F1_MOEA computed on identical
    pool state, compatibility graph, and
    operational feasibility constraints.
   
    Returns PoF as percentage.
    """
    if f1_ilp <= 0:
        return 0.0
    pof = (f1_ilp - f1_moea) / f1_ilp * 100.0
    return float(np.clip(pof, 0.0, 100.0))

def repair_to_feasible_matching(
    selected_pair_ids: Set[int],
    edges: Dict
) -> Set[int]:
    """
    Given a set of selected pair IDs from NSGA-III,
    find the largest feasible matching using valid
    2-cycles and 3-cycles only.
    
    This repair ensures MOEA solutions are feasible
    so F1_MOEA is comparable to F1_ILP.
    """
    selected = list(selected_pair_ids)
    edge_set = set(edges.keys())
    matched = set()

    # Find all valid 2-cycles among selected pairs
    two_cycles = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            a = selected[i]
            b = selected[j]
            if (a, b) in edge_set and (b, a) in edge_set:
                two_cycles.append((a, b))

    # Find all valid 3-cycles among selected pairs
    three_cycles = []
    for i in range(len(selected)):
        for j in range(len(selected)):
            if i == j:
                continue
            for k in range(len(selected)):
                if k == i or k == j:
                    continue
                a, b, c = selected[i], selected[j], selected[k]
                if ((a, b) in edge_set and
                    (b, c) in edge_set and
                    (c, a) in edge_set):
                    key = tuple(sorted([a, b, c]))
                    if key not in [
                        tuple(sorted(x)) for x in three_cycles
                    ]:
                        three_cycles.append((a, b, c))

    # Greedy: prefer 3-cycles first (more transplants)
    for cycle in three_cycles:
        if not any(p in matched for p in cycle):
            matched.update(cycle)

    # Then add 2-cycles
    for cycle in two_cycles:
        if not any(p in matched for p in cycle):
            matched.update(cycle)

    return matched

# ============================================================
# TEST MODULE 4
# ============================================================

if __name__ == "__main__":

    from module1_generator import generate_initial_pool
    from module3_logistics import build_compatibility_graph

    print("=" * 55)
    print("MODULE 4: FOUR OBJECTIVES TEST")
    print("=" * 55)

    # Generate pool
    pool, next_id = generate_initial_pool(
        n_pairs=75,
        seed=42
    )

    # Simulate some waiting times
    import random
    random.seed(42)
    for p in pool.pairs:
        p.waiting_time = random.randint(0, 10)

    print(f"\nPool size: {pool.size()}")

    # Build compatibility graph
    edges = build_compatibility_graph(pool.pairs)
    print(f"Feasible edges: {len(edges)}")

    # Simulate two scenarios:
    # Scenario 1: No matches (empty solution)
    # Scenario 2: Matched some pairs

    # Get first 10 feasible pair IDs as mock matches
    all_pair_ids = [p.pair_id for p in pool.pairs]
    mock_matched = set(all_pair_ids[:8])

    print("\n--- Scenario 1: No Matches ---")
    obj_none = compute_all_objectives(
        matched_pair_ids=set(),
        pool_pairs=pool.pairs
    )
    print(f"  F1 (efficiency):    {obj_none['F1']:.4f}")
    print(f"  F2 (Gini):          {obj_none['F2']:.4f}")
    print(f"  F3 (GFG):           {obj_none['F3']:.4f}")
    print(f"  F4 (disparity):     {obj_none['F4']:.4f}")

    print("\n--- Scenario 2: 8 Pairs Matched ---")
    obj_matched = compute_all_objectives(
        matched_pair_ids=mock_matched,
        pool_pairs=pool.pairs
    )
    print(f"  F1 (efficiency):    {obj_matched['F1']:.4f}")
    print(f"  F2 (Gini):          {obj_matched['F2']:.4f}")
    print(f"  F3 (GFG):           {obj_matched['F3']:.4f}")
    print(f"  F4 (disparity):     {obj_matched['F4']:.4f}")

    # Test PoF computation
    print("\n--- PoF Computation Test ---")
    f1_ilp_mock = 15.0   # ILP volume ceiling
    f1_moea_mock = 13.0  # MOEA fairness solution

    pof = compute_pof(f1_ilp_mock, f1_moea_mock)
    print(f"  F1 ILP ceiling:     {f1_ilp_mock}")
    print(f"  F1 MOEA solution:   {f1_moea_mock}")
    print(f"  PoF:                {pof:.2f}%")

    # Pool composition breakdown
    sensitised = [
        p for p in pool.pairs
        if p.is_highly_sensitised
    ]
    distant = [
        p for p in pool.pairs
        if classify_node(p.hospital_node) == 'R'
    ]
    central = [
        p for p in pool.pairs
        if classify_node(p.hospital_node) == 'U'
    ]

    print("\n--- Pool Composition ---")
    print(f"  Total pairs:        {pool.size()}")
    print(f"  Sensitised (S):     {len(sensitised)} "
          f"({len(sensitised)/pool.size()*100:.1f}%)")
    print(f"  Distant nodes (R):  {len(distant)} "
          f"({len(distant)/pool.size()*100:.1f}%)")
    print(f"  Central nodes (U):  {len(central)} "
          f"({len(central)/pool.size()*100:.1f}%)")

    print("\n" + "=" * 55)
    print("MODULE 4 COMPLETE")
    print("Ready for Module 5: NSGA-III with AOS")
    print("=" * 55)