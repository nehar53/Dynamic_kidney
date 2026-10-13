import warnings
from typing import Dict, Iterable, List, Optional, Set

import numpy as np

from module1_generator import CPRA_THRESHOLD, Patient
from module3_logistics import CENTRAL_NODES, DISTANT_NODES, classify_node

# ============================================================
# MODULE 4: FOUR FAIRNESS OBJECTIVES + PRICE OF FAIRNESS
# Concept note Section 3.4 (Eqs. 5-9) and Section 3.6 (Eq. 13)
# Paper 2 — Dynamic KEP Implementation
# ============================================================
#
#   F1  Transplant efficiency      number of patients transplanted
#   F2  Waiting-time equity        Gini of waiting times of ALL
#                                  patients in V_t after the round:
#                                  matched keep w_i, others w_i+1 (Eq. 7)
#   F3  Sensitised patient access  | |M∩S|/|S| - |M|/|V_t| |   (Eq. 8)
#   F4  Logistics disparity        | |M∩R|/|R| - |M∩U|/|U| |   (Eq. 9)
#
#   NSGA-III minimises  (-F1, F2, F3, F4)                      (Eq. 5)
#
# Conventions (state these in the paper):
#   * All sets are PATIENTS. Altruistic donors have no patient, so
#     they never count in F1, |S|, |V_t|, R, U or the Gini. An
#     altruist used in a chain is "matched" as a donor only.
#   * F2 ("post_round", default). A Gini of the current waiting
#     times of the whole pool is the same for every solution and
#     cannot be optimised. A Gini over only the patients LEFT
#     waiting ("residual") has the wrong incentive: removing the
#     longest waiters leaves a relatively more unequal pool, so it
#     rewards matching SHORT waiters. "post_round" uses every
#     patient in V_t, with the waiting time each will have after
#     this round (matched: w_i, final; unmatched: w_i + 1). Serving
#     long waiters then lowers F2, as waiting-time equity requires.
#   * F3 sign. Eq. 5 writes -F3 (maximise) while Eq. 8 defines F3 as
#     an absolute gap (minimise). Default F3_MODE = "gap" minimises
#     Eq. 8. F3_MODE = "access" maximises |M∩S|/|S| instead.
#   * The empty matching gives F3 = F4 = 0 (no gap if nobody is
#     matched), so it is always Pareto-optimal. The solution
#     selection rule (module 7) must ignore it when a non-empty
#     solution exists.
#
# PoF_t = (F1_ILP(t) - F1_MOEA(t)) / F1_ILP(t) x 100%          (Eq. 13)
#   undefined (NaN) when F1_ILP(t) = 0 (no transplant possible).

F3_MODE = "gap"            # "gap" (Eq. 8) or "access" (Eq. 5 sign)
F2_MODE = "post_round"     # "post_round" (default) or "residual"
ROUND_INCREMENT = 1        # months added to unmatched patients
OBJECTIVE_NAMES = ("F1_transplants", "F2_wait_gini",
                   "F3_cpra_gap", "F4_logistics_gap")


# ============================================================
# BASIC BUILDING BLOCKS
# ============================================================

def gini(values: Iterable[float]) -> float:
    """
    Gini coefficient (Eq. 7):
        G = sum_i sum_j |w_i - w_j| / (2 n sum_i w_i)
    Computed in O(n log n) with the sorted-values identity.
    0 = perfect equality, -> 1 = maximal inequality.
    """
    w = np.sort(np.asarray(list(values), dtype=float))
    n = w.size
    if n < 2 or w.sum() <= 0:
        return 0.0
    k = np.arange(1, n + 1)
    return float(np.sum((2 * k - n - 1) * w) / (n * w.sum()))


def gini_pairwise(values: Iterable[float]) -> float:
    """Direct double-sum Gini (Eq. 7), used only to verify gini()."""
    w = np.asarray(list(values), dtype=float)
    n = w.size
    if n < 2 or w.sum() <= 0:
        return 0.0
    return float(np.abs(w[:, None] - w[None, :]).sum() / (2 * n * w.sum()))


def _patients(pool_pairs: List[Patient]) -> List[Patient]:
    return [p for p in pool_pairs if not p.is_altruistic]


# ============================================================
# THE FOUR OBJECTIVES (readable versions)
# ============================================================

def compute_f1_efficiency(matched_pair_ids: Set[int],
                          pool_pairs: List[Patient]) -> float:
    """F1 (Eq. 6): number of PATIENTS transplanted this round."""
    return float(sum(1 for p in _patients(pool_pairs)
                     if p.pair_id in matched_pair_ids))


def compute_f2_gini(pool_pairs: List[Patient],
                    matched_pair_ids: Optional[Set[int]] = None,
                    mode: str = F2_MODE) -> float:
    """
    F2 (Eq. 7), waiting-time equity of a solution.
      "post_round": Gini over all patients, matched w_i,
                    unmatched w_i + 1  (default, see header)
      "residual":   Gini over unmatched patients only
    With matched_pair_ids=None it is the Gini of the current
    waiting times of all patients (a pool descriptor, e.g. for the
    LinUCB context).
    """
    pts = _patients(pool_pairs)
    if matched_pair_ids is None:
        return gini([p.waiting_time for p in pts])
    if mode == "residual":
        return gini([p.waiting_time for p in pts
                     if p.pair_id not in matched_pair_ids])
    return gini([p.waiting_time if p.pair_id in matched_pair_ids
                 else p.waiting_time + ROUND_INCREMENT for p in pts])


# Kept for compatibility with earlier modules (same result)
compute_f2_gini_fast = compute_f2_gini


def compute_f3_group_fairness(matched_pair_ids: Set[int],
                              pool_pairs: List[Patient],
                              mode: str = F3_MODE) -> float:
    """
    F3 (Eq. 8) for high-cPRA patients S (cPRA >= 80%).
      mode "gap":    | |M∩S|/|S| - |M|/|V_t| |   (minimise)
      mode "access": |M∩S|/|S|                   (maximise)
    """
    pts = _patients(pool_pairs)
    S = [p for p in pts if p.is_highly_sensitised]
    if not pts or not S:
        return 0.0
    rate_s = sum(p.pair_id in matched_pair_ids for p in S) / len(S)
    if mode == "access":
        return float(rate_s)
    rate_all = sum(p.pair_id in matched_pair_ids for p in pts) / len(pts)
    return float(abs(rate_s - rate_all))


def compute_f4_logistics_disparity(matched_pair_ids: Set[int],
                                   pool_pairs: List[Patient]) -> float:
    """
    F4 (Eq. 9): | |M∩R|/|R| - |M∩U|/|U| |
    R = patients at distant nodes (Perth, Auckland),
    U = patients at central nodes. Modelled logistics proxy only;
    not a validated measure of geographic inequity.
    """
    pts = _patients(pool_pairs)
    R = [p for p in pts if classify_node(p.hospital_node) == 'R']
    U = [p for p in pts if classify_node(p.hospital_node) == 'U']
    if not R or not U:
        return 0.0
    rate_r = sum(p.pair_id in matched_pair_ids for p in R) / len(R)
    rate_u = sum(p.pair_id in matched_pair_ids for p in U) / len(U)
    return float(abs(rate_r - rate_u))


def compute_all_objectives(matched_pair_ids: Set[int],
                           pool_pairs: List[Patient],
                           mode: str = F3_MODE,
                           f2_mode: str = F2_MODE) -> Dict[str, float]:
    """
    All four objectives for one solution (same keys as before).
    For NSGA-III use ObjectiveEvaluator below (much faster).
    """
    matched = set(matched_pair_ids)
    f1 = compute_f1_efficiency(matched, pool_pairs)
    return {
        'F1': f1,
        'F1_neg': -f1,
        'F2': compute_f2_gini(pool_pairs, matched, f2_mode),
        'F3': compute_f3_group_fairness(matched, pool_pairs, mode),
        'F4': compute_f4_logistics_disparity(matched, pool_pairs),
        'n_matched': int(f1),
        'n_pool': len(_patients(pool_pairs)),
    }


# ============================================================
# FAST EVALUATOR (used inside NSGA-III)
# ============================================================

class ObjectiveEvaluator:
    """
    Precomputes pool attributes once per round so each solution is
    evaluated with a few NumPy operations.

        ev = ObjectiveEvaluator(pool_pairs)
        f  = ev.evaluate(matched_ids)        # raw (F1, F2, F3, F4)
        g  = ev.minimisation_vector(f)       # normalised, all minimised
    """

    def __init__(self, pool_pairs: List[Patient], mode: str = F3_MODE,
                 f2_mode: str = F2_MODE):
        self.mode = mode
        self.f2_mode = f2_mode
        self.ids = np.array([p.pair_id for p in pool_pairs], dtype=np.int64)
        self.index = {int(pid): k for k, pid in enumerate(self.ids)}
        self.patient = np.array([not p.is_altruistic for p in pool_pairs])
        self.high = np.array([p.is_highly_sensitised and not p.is_altruistic
                              for p in pool_pairs])
        distant = np.array([classify_node(p.hospital_node) == 'R'
                            for p in pool_pairs])
        self.R = distant & self.patient
        self.U = ~distant & self.patient
        self.wait = np.array([p.waiting_time for p in pool_pairs], dtype=float)
        self.n_patients = int(self.patient.sum())
        self.n_high = int(self.high.sum())
        self.n_R, self.n_U = int(self.R.sum()), int(self.U.sum())

    def mask(self, matched_pair_ids: Iterable[int]) -> np.ndarray:
        m = np.zeros(len(self.ids), dtype=bool)
        for pid in matched_pair_ids:
            k = self.index.get(int(pid))
            if k is not None:
                m[k] = True
        return m

    def evaluate(self, matched_pair_ids: Iterable[int]) -> np.ndarray:
        """Raw objectives (F1, F2, F3, F4) of one solution."""
        m = self.mask(matched_pair_ids) & self.patient
        f1 = float(m.sum())
        if self.f2_mode == "residual":
            f2 = gini(self.wait[self.patient & ~m])
        else:
            w = np.where(m, self.wait, self.wait + ROUND_INCREMENT)
            f2 = gini(w[self.patient])
        if self.n_high == 0 or self.n_patients == 0:
            f3 = 0.0
        else:
            rate_s = (m & self.high).sum() / self.n_high
            f3 = float(rate_s if self.mode == "access"
                       else abs(rate_s - f1 / self.n_patients))
        f4 = (abs((m & self.R).sum() / self.n_R - (m & self.U).sum() / self.n_U)
              if self.n_R and self.n_U else 0.0)
        return np.array([f1, f2, f3, float(f4)])

    def minimisation_vector(self, raw: np.ndarray) -> np.ndarray:
        """
        Normalised objectives in [0, 1], all to be MINIMISED:
          (1 - F1/n_patients, F2, F3 or 1-F3 for "access", F4)
        Use this for NSGA-III and hypervolume, so the four
        objectives are on the same scale.
        """
        raw = np.atleast_2d(raw).astype(float)
        out = raw.copy()
        out[:, 0] = 1.0 - raw[:, 0] / max(1, self.n_patients)
        if self.mode == "access":
            out[:, 2] = 1.0 - raw[:, 2]
        return out if out.shape[0] > 1 else out[0]


# ============================================================
# EXTRA PER-ROUND METRICS (for Section 4.5 reporting)
# ============================================================

def round_metrics(matched_pair_ids: Set[int],
                  pool_pairs: List[Patient]) -> Dict[str, float]:
    """
    Descriptive metrics that explain the objectives in the paper:
    match rates per group, waiting times, altruists used.
    """
    matched = set(matched_pair_ids)
    pts = _patients(pool_pairs)
    alts = [p for p in pool_pairs if p.is_altruistic]
    S = [p for p in pts if p.is_highly_sensitised]
    R = [p for p in pts if classify_node(p.hospital_node) == 'R']
    U = [p for p in pts if classify_node(p.hospital_node) == 'U']

    def rate(group):
        return (sum(p.pair_id in matched for p in group) / len(group)
                if group else float('nan'))

    m_pts = [p for p in pts if p.pair_id in matched]
    rest = [p for p in pts if p.pair_id not in matched]
    return {
        'n_patients': len(pts),
        'n_high_cpra': len(S),
        'n_altruists': len(alts),
        'altruists_used': sum(p.pair_id in matched for p in alts),
        'match_rate_all': rate(pts),
        'match_rate_high_cpra': rate(S),            # |M∩S|/|S|
        'match_rate_low_cpra': rate([p for p in pts if not p.is_highly_sensitised]),
        'match_rate_distant': rate(R),              # |M∩R|/|R|
        'match_rate_central': rate(U),              # |M∩U|/|U|
        'mean_wait_matched': (float(np.mean([p.waiting_time for p in m_pts]))
                              if m_pts else float('nan')),
        'mean_wait_remaining': (float(np.mean([p.waiting_time for p in rest]))
                                if rest else float('nan')),
        'pool_wait_gini': gini([p.waiting_time for p in pts]),
    }


# ============================================================
# PRICE OF FAIRNESS
# ============================================================

def compute_pof(f1_ilp: float, f1_moea: float) -> float:
    """
    PoF_t = (F1_ILP(t) - F1_MOEA(t)) / F1_ILP(t) x 100%   (Eq. 13)

    F1_ILP and F1_MOEA must come from the identical pool, graph and
    exchange structures. Returns NaN if F1_ILP = 0 (undefined).
    A negative value means the MOEA beat the 'ceiling', i.e. the
    two did not use the same feasible set; this is warned about,
    not hidden by clipping.
    """
    if f1_ilp <= 0:
        return float('nan')
    pof = (f1_ilp - f1_moea) / f1_ilp * 100.0
    if pof < -1e-9:
        warnings.warn(f"PoF < 0 (F1_MOEA={f1_moea} > F1_ILP={f1_ilp}): "
                      "ILP and MOEA are not using the same feasible set.")
    return float(pof)


# ============================================================
# TEST MODULE 4
# ============================================================

if __name__ == "__main__":

    import time
    from module1_generator import generate_initial_pool
    from module3_logistics import (build_compatibility_graph,
                                   build_exchange_structures)

    print("=" * 60)
    print("MODULE 4: FOUR OBJECTIVES + PRICE OF FAIRNESS")
    print(f"F2 mode: {F2_MODE}   F3 mode: {F3_MODE}")
    print("=" * 60)

    pool, _ = generate_initial_pool(n_pairs=75, seed=42, initial_wait_max=12)
    pairs = pool.pairs
    edges = build_compatibility_graph(pairs)
    st = build_exchange_structures(pairs, edges)

    def greedy(order):
        """Pack structures greedily in the given order (test helper)."""
        used, chosen = set(), set()
        for k in order:
            if not used & set(st.nodes[k]):
                used |= set(st.nodes[k])
                chosen |= set(st.nodes[k])
        return chosen

    by_id = {p.pair_id: p for p in pairs}
    n_high = lambda k: sum(by_id[i].is_highly_sensitised for i in st.recipients[k])
    wait_sum = lambda k: sum(by_id[i].waiting_time for i in st.recipients[k])
    distant = lambda k: sum(classify_node(by_id[i].hospital_node) == 'R'
                            for i in st.recipients[k])
    K = range(len(st))
    solutions = {
        "empty matching":        set(),
        "largest first":         greedy(sorted(K, key=lambda k: -st.size(k))),
        "high-cPRA first":       greedy(sorted(K, key=lambda k: (-n_high(k), -st.size(k)))),
        "longest-waiting first": greedy(sorted(K, key=lambda k: (-wait_sum(k), -st.size(k)))),
        "distant nodes first":   greedy(sorted(K, key=lambda k: (-distant(k), -st.size(k)))),
    }

    ev = ObjectiveEvaluator(pairs)
    print(f"\nPool: {ev.n_patients} patients, {ev.n_high} high-cPRA, "
          f"R={ev.n_R} distant, U={ev.n_U} central, "
          f"{len(pool.altruists())} altruists")
    print(f"\n{'Solution':<23}{'F1':>5}{'F2':>8}{'F3':>8}{'F4':>8}"
          f"{'hiCPRA%':>9}{'alt used':>9}")
    print("-" * 70)
    for name, sol in solutions.items():
        f = ev.evaluate(sol)
        rm = round_metrics(sol, pairs)
        print(f"{name:<23}{f[0]:>5.0f}{f[1]:>8.3f}{f[2]:>8.3f}{f[3]:>8.3f}"
              f"{rm['match_rate_high_cpra'] * 100:>8.1f}%{rm['altruists_used']:>9}")
    print("  -> F2 now differs between solutions (it was constant before).")
    print("  -> The empty matching has F3 = F4 = 0: always Pareto-optimal.")

    print("\n--- F2 definition check: who should F2 reward? ---")
    sw = lambda k: wait_sum(k) / max(1, st.size(k))
    short = greedy(sorted(K, key=lambda k: (sw(k), -st.size(k))))
    longw = solutions["longest-waiting first"]
    for mode in ("post_round", "residual"):
        a = compute_f2_gini(pairs, longw, mode)
        b = compute_f2_gini(pairs, short, mode)
        verdict = "correct" if a < b else "WRONG incentive"
        print(f"  {mode:<11} longest-waiting first F2={a:.4f}  "
              f"shortest-waiting first F2={b:.4f}  -> {verdict}")

    print("\n--- Consistency checks ---")
    sol = solutions["largest first"]
    slow = compute_all_objectives(sol, pairs)
    fast = ev.evaluate(sol)
    print(f"  Fast evaluator == readable functions: "
          f"{np.allclose(fast, [slow['F1'], slow['F2'], slow['F3'], slow['F4']])}")
    w = [p.waiting_time for p in pool.patients()]
    print(f"  O(n log n) Gini == double-sum Gini (Eq. 7): "
          f"{np.isclose(gini(w), gini_pairwise(w))}")
    alt_ids = {p.pair_id for p in pool.altruists()}
    print(f"  Altruists never counted in F1: "
          f"{compute_f1_efficiency(alt_ids, pairs) == 0}")
    print(f"  Minimisation vector (largest first): "
          f"{np.round(ev.minimisation_vector(fast), 3)}")

    t0 = time.time()
    for _ in range(10000):
        ev.evaluate(sol)
    print(f"  10,000 evaluations: {time.time() - t0:.2f} s")

    print("\n--- Per-round metrics (largest first) ---")
    for k, v in round_metrics(solutions["largest first"], pairs).items():
        print(f"  {k:<22} {v:.3f}" if isinstance(v, float) else f"  {k:<22} {v}")

    print("\n--- PoF (Eq. 13) ---")
    print(f"  F1_ILP=15, F1_MOEA=13 -> PoF = {compute_pof(15, 13):.2f}%")
    print(f"  F1_ILP=15, F1_MOEA=15 -> PoF = {compute_pof(15, 15):.2f}%")
    print(f"  F1_ILP=0,  F1_MOEA=0  -> PoF = {compute_pof(0, 0)}  (undefined)")

    print("\n" + "=" * 60)
    print("MODULE 4 COMPLETE")
    print("Ready for Module 5: NSGA-III with AOS")
    print("=" * 60)
