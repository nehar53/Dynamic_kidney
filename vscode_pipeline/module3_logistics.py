from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from module1_generator import (
    ANZKX_NODES, Patient, compute_omega_hla, is_abo_compatible,
)

# ============================================================
# MODULE 3: LOGISTICS FEASIBILITY + COMPATIBILITY GRAPH
# + EXCHANGE STRUCTURES (cycles and altruistic chains)
# Concept note Sections 3.2, 3.3, 4.3 and Table 2
# Paper 2 — Dynamic KEP Implementation
# ============================================================
#
# Edge i -> j  (donor of i gives a kidney to the patient of j):
#
#   w_ij = omega_HLA(i,j)  if ABO compatible
#                          AND virtual crossmatch negative
#                          AND CIT feasible          (Eq. 2)
#          no edge         otherwise
#
#   omega_HLA(i,j) = 1 - m_ij / 6                    (Eq. 3)
#
# NOTE: an edge with omega_HLA = 0 (all 6 loci mismatched) is
# still a FEASIBLE transplant, so it is kept with weight 0.0.
# Feasibility (is there an edge?) and quality (the weight) are
# separate. Dropping weight-0 edges would silently remove about
# a quarter of all feasible transplants.
#
# Altruistic donors only GIVE: they have out-edges but never
# in-edges, so they can start chains but never sit in a cycle.
#
# CIT feasibility (Eq. 4):  T_transport(i,j) + T_prep <= CIT_max
#   T_transport = door-to-door time = flight + ground transfers
#                 at both ends + wait for a scheduled flight.
#   All times are approximate modelling assumptions; replace with
#   ANZKX transport data when available.

# --- Logistics parameters (concept note 3.3 / 4.3) ---
CIT_MAX_HOURS = 24.0          # conservative living-donor CIT limit
T_PREP_HOURS = 6.0            # surgical preparation at both centres
GROUND_HOURS_EACH_END = 1.0   # hospital <-> airport, each end
SCHEDULING_WAIT_HOURS = {     # wait for the next suitable flight
    'domestic': 2.0,
    'trans_tasman': 4.0,      # international, fewer flights
}

# --- Exchange structure limits (Table 2) ---
MAX_CYCLE_LEN = 3             # K: pairs per cycle
MAX_CHAIN_LEN = 4             # L: transplants per altruistic chain
MAX_CHAINS = 5000             # cap on enumerated chains (see below)

# --- Approximate scheduled flight times (hours) ---
FLIGHT_HOURS: Dict[Tuple[str, str], float] = {
    ('Perth', 'Adelaide'):     3.0,
    ('Perth', 'Melbourne'):    4.0,
    ('Perth', 'Sydney'):       5.0,
    ('Perth', 'Brisbane'):     5.5,
    ('Perth', 'Auckland'):     8.5,
    ('Adelaide', 'Melbourne'): 1.5,
    ('Adelaide', 'Sydney'):    2.0,
    ('Adelaide', 'Brisbane'):  2.5,
    ('Adelaide', 'Auckland'):  4.0,
    ('Melbourne', 'Sydney'):   1.5,
    ('Melbourne', 'Brisbane'): 2.5,
    ('Melbourne', 'Auckland'): 3.5,
    ('Sydney', 'Brisbane'):    1.5,
    ('Sydney', 'Auckland'):    3.0,
    ('Brisbane', 'Auckland'):  3.0,
}

# Node classification for F4 (Eq. 9). Based only on transport
# distance; does NOT imply a rural vs metropolitan distinction.
DISTANT_NODES = ['Perth', 'Auckland']                          # R
CENTRAL_NODES = ['Adelaide', 'Melbourne', 'Sydney', 'Brisbane']  # U


# ============================================================
# LOGISTICS
# ============================================================

def get_flight_time(node_i: str, node_j: str) -> float:
    if node_i == node_j:
        return 0.0
    if (node_i, node_j) in FLIGHT_HOURS:
        return FLIGHT_HOURS[(node_i, node_j)]
    if (node_j, node_i) in FLIGHT_HOURS:
        return FLIGHT_HOURS[(node_j, node_i)]
    return float('inf')


def get_transport_time(node_i: str, node_j: str) -> float:
    """
    Door-to-door transport time (hours) for a kidney from centre
    i to centre j. Same centre = 0 (no transport needed).
    """
    if node_i == node_j:
        return 0.0
    flight = get_flight_time(node_i, node_j)
    tasman = 'Auckland' in (node_i, node_j)
    wait = SCHEDULING_WAIT_HOURS['trans_tasman' if tasman else 'domestic']
    return flight + 2 * GROUND_HOURS_EACH_END + wait


# Backwards-compatible name used by earlier code
TRANSPORT_TIMES: Dict[Tuple[str, str], float] = {
    (a, b): get_transport_time(a, b) for a in ANZKX_NODES for b in ANZKX_NODES
}


def is_cit_feasible(node_donor: str, node_recipient: str,
                    t_prep: float = T_PREP_HOURS,
                    cit_max: float = CIT_MAX_HOURS) -> bool:
    """T_transport(i,j) + T_prep <= CIT_max   (Eq. 4 / Eq. 14)"""
    return get_transport_time(node_donor, node_recipient) + t_prep <= cit_max


def cit_binding_fraction(cit_max: float = CIT_MAX_HOURS,
                         t_prep: float = T_PREP_HOURS) -> float:
    """Share of ordered centre pairs ruled out by CIT (0 = never binds)."""
    pairs = [(a, b) for a in ANZKX_NODES for b in ANZKX_NODES if a != b]
    bad = sum(not is_cit_feasible(a, b, t_prep, cit_max) for a, b in pairs)
    return bad / len(pairs)


def classify_node(node: str) -> str:
    """'R' = geographically distant node, 'U' = central node (for F4)."""
    return 'R' if node in DISTANT_NODES else 'U'


# ============================================================
# COMPATIBILITY GRAPH
# ============================================================

@lru_cache(maxsize=2_000_000)
def _crossmatch_negative(salt_donor: int, salt_patient: int,
                         cpra: float) -> bool:
    """Cached virtual crossmatch (same rule as module 1)."""
    u = np.random.default_rng([salt_donor, salt_patient, 7]).random()
    return u >= cpra


def edge_status(donor_pair: Patient, patient_pair: Patient,
                cit_max: float = CIT_MAX_HOURS) -> str:
    """
    Why an edge donor_pair -> patient_pair exists or not:
    'ok', 'self', 'altruist' (no patient), 'abo', 'crossmatch', 'cit'.
    """
    if donor_pair.pair_id == patient_pair.pair_id:
        return 'self'
    if patient_pair.is_altruistic:
        return 'altruist'
    if not is_abo_compatible(donor_pair.blood_type_donor,
                             patient_pair.blood_type_patient):
        return 'abo'
    if not _crossmatch_negative(donor_pair.xm_salt, patient_pair.xm_salt,
                                patient_pair.cpra):
        return 'crossmatch'
    if not is_cit_feasible(donor_pair.hospital_node,
                           patient_pair.hospital_node, cit_max=cit_max):
        return 'cit'
    return 'ok'


def compute_edge_weight(patient_i: Patient, patient_j: Patient,
                        cit_max: float = CIT_MAX_HOURS) -> Optional[float]:
    """
    w_ij for donor of i -> patient of j (Eq. 2).
    Returns omega_HLA in [0, 1] if the transplant is feasible,
    None if it is not (ABO, crossmatch, CIT, altruist, self).
    """
    if edge_status(patient_i, patient_j, cit_max) != 'ok':
        return None
    return compute_omega_hla(patient_i.hla_donor, patient_j.hla_patient)


def build_compatibility_graph(pool_pairs: List[Patient],
                              cit_max: float = CIT_MAX_HOURS
                              ) -> Dict[Tuple[int, int], float]:
    """
    Compatibility graph E_t of the current pool.
    Returns {(donor_pair_id, patient_pair_id): omega_HLA} for every
    FEASIBLE transplant (weight may be 0.0, see header).
    """
    edges = {}
    for donor in pool_pairs:
        for patient in pool_pairs:
            w = compute_edge_weight(donor, patient, cit_max)
            if w is not None:
                edges[(donor.pair_id, patient.pair_id)] = w
    return edges


def get_logistics_summary(pool_pairs: List[Patient], edges: Dict,
                          cit_max: float = CIT_MAX_HOURS) -> Dict:
    """Edge counts by reason of removal, plus degree statistics."""
    reasons = {'ok': 0, 'abo': 0, 'crossmatch': 0, 'cit': 0,
               'altruist': 0, 'self': 0}
    for d in pool_pairs:
        for p in pool_pairs:
            reasons[edge_status(d, p, cit_max)] += 1
    n = len(pool_pairs)
    possible = n * (n - 1)
    patients = [p for p in pool_pairs if not p.is_altruistic]
    indeg = {p.pair_id: 0 for p in pool_pairs}
    for (_, j) in edges:
        indeg[j] += 1
    hi = [indeg[p.pair_id] for p in patients if p.is_highly_sensitised]
    lo = [indeg[p.pair_id] for p in patients if not p.is_highly_sensitised]
    return {
        'total_possible_edges': possible,
        'feasible_edges': len(edges),
        'removed_by_abo': reasons['abo'],
        'removed_by_crossmatch': reasons['crossmatch'],
        'removed_by_cit': reasons['cit'],
        'into_altruists': reasons['altruist'],
        'feasibility_rate': len(edges) / max(1, possible) * 100,
        'mean_in_degree_high_cpra': float(np.mean(hi)) if hi else 0.0,
        'mean_in_degree_low_cpra': float(np.mean(lo)) if lo else 0.0,
        'patients_with_no_donor': sum(indeg[p.pair_id] == 0 for p in patients),
    }


# ============================================================
# EXCHANGE STRUCTURES: CYCLES AND ALTRUISTIC CHAINS
# ============================================================
#
# Shared by module 5 (MOEA repair) and module 6 (ILP), so both
# optimise over EXACTLY the same feasible set and PoF_t compares
# like with like (concept note Section 3.6).
#
#   cycle (a, b, c):   a->b, b->c, c->a   transplants = 3
#   chain (A, p1..pk): altruist A -> p1 -> ... -> pk, the last
#                      donor gives to the deceased-donor waitlist
#                      transplants = k (A is not a patient)
#
# Chains grow combinatorially (d^L). If there are more than
# MAX_CHAINS, all chains with 1-2 transplants are kept and longer
# ones are sampled with a seed derived from the pool, so the same
# pool always gives the same structures. `truncated` reports it.

@dataclass
class ExchangeStructures:
    nodes: List[Tuple[int, ...]] = field(default_factory=list)
    recipients: List[Tuple[int, ...]] = field(default_factory=list)
    kind: List[str] = field(default_factory=list)       # 'cycle' / 'chain'
    weight: List[float] = field(default_factory=list)   # sum of omega_HLA
    truncated: bool = False

    def __len__(self):
        return len(self.nodes)

    def size(self, k: int) -> int:
        """Transplants delivered by structure k."""
        return len(self.recipients[k])

    def n_cycles(self) -> int:
        return self.kind.count('cycle')

    def n_chains(self) -> int:
        return self.kind.count('chain')


def _adjacency(edges: Dict[Tuple[int, int], float]) -> Dict[int, List[int]]:
    adj: Dict[int, List[int]] = {}
    for (i, j) in edges:
        adj.setdefault(i, []).append(j)
    for i in adj:
        adj[i].sort()
    return adj


def enumerate_cycles(edges: Dict[Tuple[int, int], float],
                     pair_ids: List[int],
                     max_len: int = MAX_CYCLE_LEN) -> List[Tuple[int, ...]]:
    """All simple cycles of length 2..max_len, each listed once."""
    adj = _adjacency(edges)
    ids = set(pair_ids)
    cycles = []
    for start in sorted(ids):
        stack = [(start, (start,))]
        while stack:
            node, path = stack.pop()
            for nxt in adj.get(node, ()):
                if nxt == start and len(path) >= 2:
                    cycles.append(path)
                elif nxt > start and nxt in ids and nxt not in path \
                        and len(path) < max_len:
                    stack.append((nxt, path + (nxt,)))
    return cycles


def enumerate_chains(edges: Dict[Tuple[int, int], float],
                     altruist_ids: List[int],
                     max_len: int = MAX_CHAIN_LEN,
                     max_chains: int = MAX_CHAINS,
                     seed: int = 0) -> Tuple[List[Tuple[int, ...]], bool]:
    """
    Altruistic chains (A, p1, ..., pk), 1 <= k <= max_len.
    Returns (chains, truncated).
    """
    adj = _adjacency(edges)
    chains: List[Tuple[int, ...]] = []
    # exact enumeration with a budget
    for a in sorted(altruist_ids):
        stack = [(a, (a,))]
        while stack:
            node, path = stack.pop()
            if len(path) > 1:
                chains.append(path)
                if len(chains) > max_chains:
                    break
            if len(path) - 1 < max_len:
                for nxt in adj.get(node, ()):
                    if nxt not in path:
                        stack.append((nxt, path + (nxt,)))
        if len(chains) > max_chains:
            break
    if len(chains) <= max_chains:
        return chains, False

    # too many: keep all short chains, sample long ones
    rng = np.random.default_rng(seed)
    keep = set()
    for a in altruist_ids:
        for b in adj.get(a, ()):
            keep.add((a, b))
            for c in adj.get(b, ()):
                if c != a:
                    keep.add((a, b, c))
    alts = sorted(altruist_ids)
    tries = 0
    while len(keep) < max_chains and tries < 20 * max_chains and max_len >= 3:
        tries += 1
        path = [alts[rng.integers(len(alts))]]
        target = int(rng.integers(3, max_len + 1))
        while len(path) - 1 < target:
            nxt = [x for x in adj.get(path[-1], ()) if x not in path]
            if not nxt:
                break
            path.append(nxt[rng.integers(len(nxt))])
        if len(path) - 1 >= 3:
            keep.add(tuple(path))
    return sorted(keep), True


def build_exchange_structures(pool_pairs: List[Patient],
                              edges: Optional[Dict] = None,
                              max_cycle_len: int = MAX_CYCLE_LEN,
                              max_chain_len: int = MAX_CHAIN_LEN,
                              max_chains: int = MAX_CHAINS
                              ) -> ExchangeStructures:
    """All cycles and chains of the current pool (shared by modules 5/6)."""
    if edges is None:
        edges = build_compatibility_graph(pool_pairs)
    ids = [p.pair_id for p in pool_pairs]
    alts = [p.pair_id for p in pool_pairs if p.is_altruistic]
    seed = int(np.random.default_rng(sorted(ids) or [0]).integers(2 ** 31))

    cycles = enumerate_cycles(edges, ids, max_cycle_len)
    chains, truncated = (enumerate_chains(edges, alts, max_chain_len,
                                          max_chains, seed)
                         if max_chain_len > 0 else ([], False))
    s = ExchangeStructures(truncated=truncated)
    for c in cycles:
        s.nodes.append(c)
        s.recipients.append(c)
        s.kind.append('cycle')
        s.weight.append(sum(edges[(c[k], c[(k + 1) % len(c)])]
                            for k in range(len(c))))
    for ch in chains:
        s.nodes.append(ch)
        s.recipients.append(ch[1:])
        s.kind.append('chain')
        s.weight.append(sum(edges[(ch[k], ch[k + 1])]
                            for k in range(len(ch) - 1)))
    return s


# ============================================================
# TEST MODULE 3
# ============================================================

if __name__ == "__main__":

    import time
    from module1_generator import generate_initial_pool

    print("=" * 60)
    print("MODULE 3: LOGISTICS + COMPATIBILITY GRAPH + STRUCTURES")
    print(f"CIT_max = {CIT_MAX_HOURS} h, T_prep = {T_PREP_HOURS} h")
    print("T_transport = flight + 2 x ground + scheduling wait")
    print("=" * 60)

    print("\n--- Door-to-door transport time (hours) ---")
    print(f"{'':11}" + "".join(f"{n[:9]:>10}" for n in ANZKX_NODES))
    for a in ANZKX_NODES:
        print(f"{a:11}" + "".join(f"{get_transport_time(a, b):>10.1f}"
                                  for b in ANZKX_NODES))

    print("\n--- CIT feasibility (transport + prep <= CIT_max) ---")
    print(f"{'':11}" + "".join(f"{n[:9]:>10}" for n in ANZKX_NODES))
    for a in ANZKX_NODES:
        print(f"{a:11}" + "".join(
            f"{('YES' if is_cit_feasible(a, b) else 'NO'):>10}"
            for b in ANZKX_NODES))

    print("\n--- CIT sensitivity: share of centre pairs ruled out ---")
    for cit in (24, 20, 18, 16, 12):
        bad = [f"{a}-{b}" for a in ANZKX_NODES for b in ANZKX_NODES
               if a < b and not is_cit_feasible(a, b, cit_max=cit)]
        print(f"  CIT_max={cit:>2} h: {cit_binding_fraction(cit) * 100:5.1f}%  "
              f"{', '.join(bad) if bad else '(none)'}")

    print("\n--- Node classification for F4 ---")
    print(f"  Distant (R): {DISTANT_NODES}")
    print(f"  Central (U): {CENTRAL_NODES}")

    print("\n--- Compatibility graph (75-pair pool) ---")
    pool, _ = generate_initial_pool(n_pairs=75, seed=42)
    t0 = time.time()
    edges = build_compatibility_graph(pool.pairs)
    t_graph = time.time() - t0
    s = get_logistics_summary(pool.pairs, edges)
    print(f"  Pool: {len(pool.patients())} pairs + "
          f"{len(pool.altruists())} altruists")
    print(f"  Possible edges:          {s['total_possible_edges']}")
    print(f"  Removed by ABO:          {s['removed_by_abo']}")
    print(f"  Removed by crossmatch:   {s['removed_by_crossmatch']}")
    print(f"  Removed by CIT:          {s['removed_by_cit']}")
    print(f"  Into altruists (none):   {s['into_altruists']}")
    print(f"  Feasible edges:          {s['feasible_edges']} "
          f"({s['feasibility_rate']:.1f}%)")
    print(f"  Mean compatible donors, high-cPRA patient: "
          f"{s['mean_in_degree_high_cpra']:.1f}")
    print(f"  Mean compatible donors, low-cPRA patient:  "
          f"{s['mean_in_degree_low_cpra']:.1f}")
    print(f"  Patients with no compatible donor: {s['patients_with_no_donor']}")
    w = np.array(list(edges.values()))
    print(f"  omega_HLA: mean {w.mean():.3f}, "
          f"edges with omega=0 kept: {(w == 0).sum()}")
    alt_ids = {p.pair_id for p in pool.altruists()}
    print(f"  Edges INTO altruists: {sum(j in alt_ids for (_, j) in edges)}")
    print(f"  Build time: {t_graph:.2f} s")

    print("\n--- Exchange structures (K=3 cycles, L=4 chains) ---")
    t0 = time.time()
    st = build_exchange_structures(pool.pairs, edges)
    print(f"  2-cycles: {sum(1 for k in range(len(st)) if st.kind[k] == 'cycle' and len(st.nodes[k]) == 2)}")
    print(f"  3-cycles: {sum(1 for k in range(len(st)) if st.kind[k] == 'cycle' and len(st.nodes[k]) == 3)}")
    for L in range(1, MAX_CHAIN_LEN + 1):
        n_l = sum(1 for k in range(len(st))
                  if st.kind[k] == 'chain' and st.size(k) == L)
        print(f"  chains with {L} transplant(s): {n_l}")
    print(f"  Chains truncated (sampled): {st.truncated}")
    print(f"  Altruists inside cycles: "
          f"{any(n in alt_ids for k in range(len(st)) if st.kind[k] == 'cycle' for n in st.nodes[k])}")
    print(f"  Enumeration time: {time.time() - t0:.2f} s")

    print("\n" + "=" * 60)
    print("MODULE 3 COMPLETE")
    print("Ready for Module 4: Objectives")
    print("=" * 60)
