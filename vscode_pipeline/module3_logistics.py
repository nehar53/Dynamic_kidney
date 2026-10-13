import numpy as np
from typing import Dict, Tuple, List
from module1_generator import Patient, ANZKX_NODES

# ============================================================
# MODULE 3: LOGISTICS FEASIBILITY NETWORK
# Six-node ANZKX inter-hospital CIT constraint
# T_transport + T_prep <= CIT_max
# Paper 2 — Dynamic KEP Implementation
# ============================================================

# --- Logistics Parameters ---

# Maximum cold ischaemia time for living donor kidneys
# Conservative modelling assumption
# Requires ANZKX-specific clinical data for validation
CIT_MAX_HOURS = 24.0

# Surgical preparation time at donor and recipient centres
# Modelling assumption
T_PREP_HOURS = 6.0

# Effective transport budget available
# CIT_max - T_prep = 18 hours for actual transport
TRANSPORT_BUDGET = CIT_MAX_HOURS - T_PREP_HOURS

# --- Inter-Hospital Air Transport Times (hours) ---
# Based on approximate published air transport times
# between ANZKX transplant centres.
# Source: Approximate flight durations
# Perth-Auckland via east coast approximately 8-9 hours
# These are approximate values for modelling purposes.
# ANZKX-specific transport protocols would refine these.

TRANSPORT_TIMES: Dict[Tuple[str, str], float] = {
    # Perth connections
    ('Perth', 'Adelaide'):   3.0,
    ('Perth', 'Melbourne'):  4.0,
    ('Perth', 'Sydney'):     5.0,
    ('Perth', 'Brisbane'):   5.5,
    ('Perth', 'Auckland'):   8.5,

    # Adelaide connections
    ('Adelaide', 'Perth'):   3.0,
    ('Adelaide', 'Melbourne'): 1.5,
    ('Adelaide', 'Sydney'):  2.0,
    ('Adelaide', 'Brisbane'): 2.5,
    ('Adelaide', 'Auckland'): 4.0,

    # Melbourne connections
    ('Melbourne', 'Perth'):  4.0,
    ('Melbourne', 'Adelaide'): 1.5,
    ('Melbourne', 'Sydney'):  1.5,
    ('Melbourne', 'Brisbane'): 2.5,
    ('Melbourne', 'Auckland'): 3.5,

    # Sydney connections
    ('Sydney', 'Perth'):     5.0,
    ('Sydney', 'Adelaide'):  2.0,
    ('Sydney', 'Melbourne'): 1.5,
    ('Sydney', 'Brisbane'):  1.5,
    ('Sydney', 'Auckland'):  3.0,

    # Brisbane connections
    ('Brisbane', 'Perth'):   5.5,
    ('Brisbane', 'Adelaide'): 2.5,
    ('Brisbane', 'Melbourne'): 2.5,
    ('Brisbane', 'Sydney'):  1.5,
    ('Brisbane', 'Auckland'): 3.0,

    # Auckland connections
    ('Auckland', 'Perth'):   8.5,
    ('Auckland', 'Adelaide'): 4.0,
    ('Auckland', 'Melbourne'): 3.5,
    ('Auckland', 'Sydney'):  3.0,
    ('Auckland', 'Brisbane'): 3.0,

    # Same node — no transport needed
    ('Perth', 'Perth'):       0.0,
    ('Adelaide', 'Adelaide'): 0.0,
    ('Melbourne', 'Melbourne'): 0.0,
    ('Sydney', 'Sydney'):     0.0,
    ('Brisbane', 'Brisbane'): 0.0,
    ('Auckland', 'Auckland'): 0.0,
}

# Node classification for F4 objective
# Based solely on inter-node transport distances
# Does NOT imply rural vs metropolitan distinction
# Perth and Auckland are geographically distant nodes
# Others are centrally located nodes

DISTANT_NODES = ['Perth', 'Auckland']      # R set
CENTRAL_NODES = [
    'Adelaide', 'Melbourne',
    'Sydney', 'Brisbane'
]                                           # U set


def get_transport_time(
    node_i: str,
    node_j: str
) -> float:
    """
    Get air transport time between two ANZKX nodes.
    Returns hours.
    """
    key = (node_i, node_j)
    if key in TRANSPORT_TIMES:
        return TRANSPORT_TIMES[key]
    # Try reverse
    reverse_key = (node_j, node_i)
    if reverse_key in TRANSPORT_TIMES:
        return TRANSPORT_TIMES[reverse_key]
    # Unknown route — assume infeasible
    return float('inf')


def is_cit_feasible(
    node_donor: str,
    node_recipient: str,
    t_prep: float = T_PREP_HOURS,
    cit_max: float = CIT_MAX_HOURS
) -> bool:
    """
    Check if a match between two nodes is
    logistically feasible under CIT constraint.
   
    Feasible if:
    T_transport(i,j) + T_prep <= CIT_max
   
    Modelling assumption:
    CIT_max = 24 hours (conservative estimate)
    T_prep = 6 hours (preparation at both centres)
    """
    t_transport = get_transport_time(
        node_donor,
        node_recipient
    )
    return (t_transport + t_prep) <= cit_max


def compute_edge_weight(
    patient_i: 'Patient',
    patient_j: 'Patient'
) -> float:
    """
    Compute edge weight w_ij for compatibility graph.
   
    w_ij = omega_HLA(i,j) if ABO compatible
                            AND CIT feasible
           0               otherwise
   
    This is the full feasibility weight combining:
    ABO compatibility + CIT logistics + HLA quality
    """
    from module1_generator import (
        is_abo_compatible,
        compute_omega_hla
    )

    # Check ABO compatibility
    # Donor i gives kidney to recipient j
    abo_ok = is_abo_compatible(
        patient_i.blood_type_donor,
        patient_j.blood_type_patient
    )
    if not abo_ok:
        return 0.0

    # Check CIT feasibility
    cit_ok = is_cit_feasible(
        patient_i.hospital_node,
        patient_j.hospital_node
    )
    if not cit_ok:
        return 0.0

    # Compute HLA mismatch quality score
    omega = compute_omega_hla(
        patient_i.hla_donor,
        patient_j.hla_patient
    )

    return omega


def build_compatibility_graph(
    pool_pairs: List['Patient']
) -> Dict[Tuple[int, int], float]:
    """
    Build compatibility graph E_t for current pool.
   
    Returns dict of (i,j) -> edge_weight
    Only includes edges with weight > 0.
    Edges represent potential donor-recipient matches.
    """
    edges = {}
    n = len(pool_pairs)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # Skip if same pair
            # (cannot match with yourself)
            if pool_pairs[i].pair_id == \
               pool_pairs[j].pair_id:
                continue

            weight = compute_edge_weight(
                pool_pairs[i],  # donor from pair i
                pool_pairs[j]   # recipient in pair j
            )

            if weight > 0:
                edges[(
                    pool_pairs[i].pair_id,
                    pool_pairs[j].pair_id
                )] = weight

    return edges


def get_logistics_summary(
    pool_pairs: List['Patient'],
    edges: Dict
) -> Dict:
    """
    Summarise logistics network statistics.
    """
    total_possible = len(pool_pairs) * (
        len(pool_pairs) - 1
    )
    feasible_edges = len(edges)

    # Count edges removed by CIT constraint
    cit_removed = 0
    abo_removed = 0

    from module1_generator import is_abo_compatible

    for i in range(len(pool_pairs)):
        for j in range(len(pool_pairs)):
            if i == j:
                continue
            abo_ok = is_abo_compatible(
                pool_pairs[i].blood_type_donor,
                pool_pairs[j].blood_type_patient
            )
            if abo_ok:
                cit_ok = is_cit_feasible(
                    pool_pairs[i].hospital_node,
                    pool_pairs[j].hospital_node
                )
                if not cit_ok:
                    cit_removed += 1
            else:
                abo_removed += 1

    return {
        'total_possible_edges': total_possible,
        'feasible_edges': feasible_edges,
        'removed_by_abo': abo_removed,
        'removed_by_cit': cit_removed,
        'feasibility_rate': (
            feasible_edges / max(1, total_possible) * 100
        )
    }


def classify_node(node: str) -> str:
    """
    Classify node as distant R or central U.
    Used for F4 objective calculation.
    Classification based solely on
    inter-node transport distances.
    Does not imply rural vs metropolitan.
    """
    if node in DISTANT_NODES:
        return 'R'  # Geographically distant
    return 'U'      # Centrally located


# ============================================================
# TEST MODULE 3
# ============================================================

if __name__ == "__main__":

    from module1_generator import generate_initial_pool

    print("=" * 55)
    print("MODULE 3: LOGISTICS FEASIBILITY NETWORK")
    print(f"CIT_max = {CIT_MAX_HOURS} hours")
    print(f"T_prep  = {T_PREP_HOURS} hours")
    print(f"Transport budget = {TRANSPORT_BUDGET} hours")
    print("=" * 55)

    # Print transport time matrix
    print("\n--- Transport Time Matrix (hours) ---")
    print(f"{'':12}", end="")
    for node in ANZKX_NODES:
        print(f"{node:>12}", end="")
    print()

    for node_i in ANZKX_NODES:
        print(f"{node_i:12}", end="")
        for node_j in ANZKX_NODES:
            t = get_transport_time(node_i, node_j)
            print(f"{t:>12.1f}", end="")
        print()

    # Print feasibility matrix
    print("\n--- CIT Feasibility Matrix ---")
    print(f"{'':12}", end="")
    for node in ANZKX_NODES:
        print(f"{node:>12}", end="")
    print()

    for node_i in ANZKX_NODES:
        print(f"{node_i:12}", end="")
        for node_j in ANZKX_NODES:
            feasible = is_cit_feasible(node_i, node_j)
            symbol = "YES" if feasible else "NO"
            print(f"{symbol:>12}", end="")
        print()

    # Show which routes are infeasible
    print("\n--- Infeasible Routes (exceed CIT) ---")
    infeasible_found = False
    for node_i in ANZKX_NODES:
        for node_j in ANZKX_NODES:
            if node_i == node_j:
                continue
            t = get_transport_time(node_i, node_j)
            total = t + T_PREP_HOURS
            if total > CIT_MAX_HOURS:
                print(
                    f"  {node_i} -> {node_j}: "
                    f"{t:.1f}h transport + "
                    f"{T_PREP_HOURS}h prep = "
                    f"{total:.1f}h > "
                    f"{CIT_MAX_HOURS}h CIT_max"
                )
                infeasible_found = True

    if not infeasible_found:
        print(
            "  All routes feasible within "
            f"{CIT_MAX_HOURS}h CIT constraint."
        )

    # Node classification for F4
    print("\n--- Node Classification for F4 ---")
    print("Geographically distant nodes (R set):")
    for n in DISTANT_NODES:
        print(f"  {n}")
    print("Centrally located nodes (U set):")
    for n in CENTRAL_NODES:
        print(f"  {n}")

    # Build compatibility graph on small pool
    print("\n--- Compatibility Graph Test ---")
    pool, next_id = generate_initial_pool(
        n_pairs=20,
        seed=42
    )
    print(f"Pool size: {pool.size()} pairs")

    edges = build_compatibility_graph(pool.pairs)
    summary = get_logistics_summary(pool.pairs, edges)

    print(f"Total possible edges: "
          f"{summary['total_possible_edges']}")
    print(f"Feasible edges: "
          f"{summary['feasible_edges']}")
    print(f"Removed by ABO: "
          f"{summary['removed_by_abo']}")
    print(f"Removed by CIT: "
          f"{summary['removed_by_cit']}")
    print(f"Feasibility rate: "
          f"{summary['feasibility_rate']:.1f}%")

    print("\n" + "=" * 55)
    print("MODULE 3 COMPLETE")
    print("Ready for Module 4: Four Objectives")
    print("=" * 55)