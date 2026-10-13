import contextlib
import io
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.ux import UX
from pymoo.util.ref_dirs import get_reference_directions

from module1_generator import Patient
from module3_logistics import (
    MAX_CHAIN_LEN, MAX_CYCLE_LEN, ExchangeStructures,
    build_compatibility_graph, build_exchange_structures,
)
from module4_objectives import ObjectiveEvaluator, compute_f2_gini

# ============================================================
# MODULE 5: NSGA-III WITH ADAPTIVE OPERATOR SELECTION
# UCB1 (stateless, Eq. 10) and LinUCB (contextual, Eqs. 11-12)
# Four objectives F1-F4 (module 4)
# Concept note Sections 3.5 and 4.2
# Paper 2 — Dynamic KEP Implementation
# ============================================================
#
# Encoding (as in Paper 1): x_i in {0,1} says pair i is ELIGIBLE.
# Repair: the eligible pairs are packed into disjoint exchange
# structures (2-/3-cycles and altruistic chains up to 4) taken
# from module 3's shared list, largest structures first. The
# repaired x (= the pairs actually matched) is written back
# (Lamarckian repair), so genotype = phenotype. Module 6's ILP
# uses the SAME structure list, so F1_MOEA <= F1_ILP always.
#
# ILP-guided hybrid (recommended, used by module 7):
#   Packing cycles and chains is NP-hard and a greedy repair alone
#   reaches only ~70% of the optimum, which would make PoF measure
#   search weakness instead of fairness. So the volume-optimal ILP
#   solution (module 6) is (a) injected into the initial population
#   and (b) its structures get top repair priority. The front then
#   always contains the F1-optimal point, and PoF_t measures only
#   the efficiency given up by choosing a fairer solution (Eq. 13).
#
# Adaptive operator selection, PER GENERATION (Section 3.5.1):
#   at every generation g the bandit picks a mutation rate
#   a in A = {0.01, 0.05, 0.10, 0.25}, NSGA-III runs generation g
#   with bit-flip rate a, and the bandit is rewarded with that
#   generation's hypervolume gain of the non-dominated set.
#   Reward = HV gain / largest gain seen so far in the round, so
#   rewards are in [0, 1] for both UCB1 and LinUCB.
#
# LinUCB context s_t (Eq. 11), each feature scaled to [0, 1]:
#   [ n_t / N_SCALE,  p_PRA,  x_t / n_t,  t / T,  Gini_t,  PoF_{t-1}/100 ]
#   The context is fixed within a round (it describes the pool),
#   so LinUCB adapts BETWEEN rounds; UCB1 only uses rewards.
#
# Reference directions: concept note sets N = 100 and p = 12,
# which gives C(15,3) = 455 Das-Dennis directions (> N). NSGA-III
# still works (niching uses the directions closest to the
# population), but pymoo prints a warning; it is silenced here.
# Report this, or set N_POPULATION = 455 if compute allows.

# --- NSGA-III parameters (Section 4.2.1) ---
N_POPULATION = 100
N_GENERATIONS = 150
N_REF_PARTITIONS = 12          # p for four objectives
CROSSOVER_PROB = 0.9
FIXED_MUTATION_RATE = 0.10     # baseline 2: no AOS

# --- AOS parameters (Sections 4.2.2 / 4.2.3) ---
MUTATION_RATES = [0.01, 0.05, 0.10, 0.25]
N_ARMS = len(MUTATION_RATES)
WINDOW_SIZE = 15               # UCB1 sliding window W
ALPHA_LINUCB = 1.0
CONTEXT_DIM = 6
N_SCALE = 150.0                # pool size scaling for the context

HV_REF_POINT = np.full(4, 1.1)  # normalised objectives are in [0, 1]

# Backwards-compatible names
N_REF_DIRS = N_REF_PARTITIONS
UCB1_C = 1.0


# ============================================================
# AOS 1: UCB1 (stateless, sliding window)            Eq. 10
# ============================================================

class UCB1AOS:
    """
    a* = argmax_a  Q(a) + sqrt(2 ln t / N(a))
    Q, N, t computed over the last W pulls (sliding window).
    """

    def __init__(self, n_arms: int = N_ARMS, window_size: int = WINDOW_SIZE,
                 c: float = UCB1_C, seed: int = 0):
        self.n_arms = n_arms
        self.window = deque(maxlen=window_size)        # (arm, reward)
        self.c = c
        self.total_pulls = np.zeros(n_arms, dtype=int)
        self.rng = np.random.default_rng(seed)

    def _stats(self):
        counts, sums = np.zeros(self.n_arms), np.zeros(self.n_arms)
        for arm, reward in self.window:
            counts[arm] += 1
            sums[arm] += reward
        return counts, sums

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        counts, sums = self._stats()
        unplayed = np.flatnonzero(counts == 0)
        if unplayed.size:                     # play every arm in the window once
            return int(self.rng.choice(unplayed))
        t = counts.sum()
        ucb = sums / counts + self.c * np.sqrt(2.0 * np.log(t) / counts)
        return int(np.argmax(ucb))

    def get_mutation_rate(self, context=None) -> Tuple[int, float]:
        arm = self.select_arm(context)
        self.total_pulls[arm] += 1
        return arm, MUTATION_RATES[arm]

    def update(self, arm: int, reward: float, context=None) -> None:
        self.window.append((arm, float(reward)))

    def get_arm_distribution(self) -> Dict[int, float]:
        total = self.total_pulls.sum()
        return {a: (self.total_pulls[a] / total if total else 0.0)
                for a in range(self.n_arms)}


# ============================================================
# AOS 2: LinUCB (contextual, disjoint model)      Eqs. 11-12
# ============================================================

def build_context_vector(pool_pairs: List[Patient], competing_exits: int,
                         clearing_round: int, total_rounds: int,
                         pof_prev: float) -> np.ndarray:
    """
    s_t = [n_t, p_PRA, x_t, t, Gini_t, PoF_{t-1}]  (Eq. 11), in [0, 1].
      n_t     patients in the pool / N_SCALE
      p_PRA   share of high-cPRA patients
      x_t     competing exits last round / n_t
      t       clearing round / T
      Gini_t  waiting-time Gini of the pool
      PoF     previous round's PoF in % / 100
    """
    pts = [p for p in pool_pairs if not p.is_altruistic]
    n = max(1, len(pts))
    s = np.array([
        len(pts) / N_SCALE,
        sum(p.is_highly_sensitised for p in pts) / n,
        competing_exits / n,
        clearing_round / max(1, total_rounds),
        compute_f2_gini(pool_pairs),
        (0.0 if pof_prev is None or np.isnan(pof_prev) else pof_prev) / 100.0,
    ])
    return np.clip(s, 0.0, 1.0)


class LinUCBAOS:
    """
    a* = argmax_a  theta_a^T s_t + alpha sqrt(s_t^T A_a^{-1} s_t)
    """

    def __init__(self, n_arms: int = N_ARMS, context_dim: int = CONTEXT_DIM,
                 alpha: float = ALPHA_LINUCB, seed: int = 0):
        self.n_arms, self.d, self.alpha = n_arms, context_dim, alpha
        self.A = [np.eye(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]
        self.total_pulls = np.zeros(n_arms, dtype=int)
        self.rng = np.random.default_rng(seed)

    def build_context_vector(self, pool_pairs, competing_exits,
                             clearing_round, gini_t=None, pof_prev=0.0,
                             total_rounds: int = 25) -> np.ndarray:
        """Kept for compatibility; gini_t is recomputed from the pool."""
        return build_context_vector(pool_pairs, competing_exits,
                                    clearing_round, total_rounds,
                                    100.0 * pof_prev if pof_prev <= 1 else pof_prev)

    def select_arm(self, context: np.ndarray) -> int:
        scores = np.empty(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            scores[a] = theta @ context + self.alpha * np.sqrt(context @ A_inv @ context)
        best = np.flatnonzero(np.isclose(scores, scores.max()))
        return int(self.rng.choice(best))

    def get_mutation_rate(self, context: np.ndarray) -> Tuple[int, float]:
        arm = self.select_arm(context)
        self.total_pulls[arm] += 1
        return arm, MUTATION_RATES[arm]

    def update(self, arm: int, reward: float, context: np.ndarray = None) -> None:
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context

    def get_arm_distribution(self) -> Dict[int, float]:
        total = self.total_pulls.sum()
        return {a: (self.total_pulls[a] / total if total else 0.0)
                for a in range(self.n_arms)}


class FixedAOS:
    """Baseline 2: always the same mutation rate."""

    def __init__(self, rate: float = FIXED_MUTATION_RATE):
        self.arm = int(np.argmin([abs(r - rate) for r in MUTATION_RATES]))
        self.total_pulls = np.zeros(N_ARMS, dtype=int)

    def get_mutation_rate(self, context=None):
        self.total_pulls[self.arm] += 1
        return self.arm, MUTATION_RATES[self.arm]

    def update(self, arm, reward, context=None):
        pass

    def get_arm_distribution(self):
        total = self.total_pulls.sum()
        return {a: (self.total_pulls[a] / total if total else 0.0)
                for a in range(N_ARMS)}


# ============================================================
# REPAIR: ELIGIBLE PAIRS -> DISJOINT CYCLES AND CHAINS
# ============================================================

class StructureRepair:
    """
    Packs eligible pairs into disjoint structures from module 3,
    in a fixed priority order: more transplants first, then higher
    HLA quality. Identical structures to the ILP (module 6).
    """

    def __init__(self, pool_pairs: List[Patient], structures: ExchangeStructures,
                 preferred: Optional[List[int]] = None):
        """preferred: structure indices repaired first (e.g. ILP optimum)."""
        self.ids = [p.pair_id for p in pool_pairs]
        pos = {pid: k for k, pid in enumerate(self.ids)}
        self.n = len(self.ids)
        self.structures = structures
        self.nodes = [tuple(pos[i] for i in s) for s in structures.nodes]
        self.lens = np.array([len(s) for s in self.nodes], dtype=int)
        rows, cols = [], []
        for k, s in enumerate(self.nodes):
            rows.extend([k] * len(s))
            cols.extend(s)
        from scipy.sparse import csr_matrix
        self.A = csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)),
                            shape=(len(self.nodes), self.n))
        pref = set(preferred or [])
        priority = sorted(range(len(self.nodes)),
                          key=lambda k: (k not in pref, -structures.size(k),
                                         -structures.weight[k], k))
        self.rank = np.empty(len(self.nodes), dtype=np.int64)
        self.rank[priority] = np.arange(len(self.nodes))

    def repair(self, x: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Returns (matched node mask, chosen structure indices)."""
        if len(self.nodes) == 0:
            return np.zeros(self.n, dtype=bool), []
        cnt = self.A @ x.astype(np.int32)
        feas = np.flatnonzero(cnt == self.lens)
        feas = feas[np.argsort(self.rank[feas])]
        used = [False] * self.n
        chosen = []
        for k in feas:
            nodes = self.nodes[k]
            if not any(used[v] for v in nodes):
                for v in nodes:
                    used[v] = True
                chosen.append(int(k))
        return np.array(used, dtype=bool), chosen

    def matched_ids(self, x: np.ndarray) -> Set[int]:
        mask, _ = self.repair(x)
        return {self.ids[k] for k in np.flatnonzero(mask)}


# ============================================================
# PYMOO PROBLEM AND OPERATORS
# ============================================================

class KEPProblem(Problem):
    """
    x_i in {0,1}: pair i eligible. Each x is repaired to disjoint
    structures; objectives are module 4's normalised minimisation
    vector (1 - F1/n, F2, F3, F4), all in [0, 1].
    """

    def __init__(self, pool_pairs: List[Patient], edges: Optional[Dict] = None,
                 mutation_rate: float = FIXED_MUTATION_RATE,
                 structures: Optional[ExchangeStructures] = None,
                 preferred: Optional[List[int]] = None):
        self.pool_pairs = pool_pairs
        self.edges = edges if edges is not None else build_compatibility_graph(pool_pairs)
        self.structures = (structures if structures is not None
                           else build_exchange_structures(pool_pairs, self.edges))
        self.repairer = StructureRepair(pool_pairs, self.structures, preferred)
        self.evaluator = ObjectiveEvaluator(pool_pairs)
        self.mutation_rate = mutation_rate
        super().__init__(n_var=len(pool_pairs), n_obj=4, n_ieq_constr=0,
                         xl=0, xu=1, vtype=bool)

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((X.shape[0], 4))
        for i in range(X.shape[0]):
            mask, _ = self.repairer.repair(np.asarray(X[i], dtype=bool))
            F[i] = self.evaluator.minimisation_vector(
                self.evaluator.evaluate(self.repairer.ids[k] for k in np.flatnonzero(mask)))
        out["F"] = F


class LamarckianRepair(Repair):
    """Write the repaired solution (pairs actually matched) back into x."""

    def _do(self, problem, X, **kwargs):
        X = np.asarray(X, dtype=bool).copy()
        for i in range(len(X)):
            X[i], _ = problem.repairer.repair(X[i])
        return X


class AdaptiveBitflip(Mutation):
    """Bit-flip mutation whose per-bit rate is set by the AOS each generation."""

    def __init__(self, rate: float = FIXED_MUTATION_RATE):
        super().__init__(prob=1.0)
        self.rate = rate

    def _do(self, problem, X, random_state=None, **kwargs):
        rng = random_state if random_state is not None else np.random.default_rng()
        X = np.asarray(X, dtype=bool).copy()
        flip = rng.random(X.shape) < self.rate
        X[flip] = ~X[flip]
        return X


class SeededSampling(Sampling):
    """
    Initial population: optional warm-start solutions (previous
    round's front, dynamic variants), the rest random with a
    per-individual eligibility rate in [0.2, 1.0] for diversity.
    """

    def __init__(self, seeds: Optional[np.ndarray] = None):
        super().__init__()
        self.seeds = seeds

    def _do(self, problem, n_samples, random_state=None, **kwargs):
        rng = random_state if random_state is not None else np.random.default_rng()
        rates = rng.uniform(0.2, 1.0, size=(n_samples, 1))
        X = rng.random((n_samples, problem.n_var)) < rates
        if self.seeds is not None and len(self.seeds):
            k = min(len(self.seeds), n_samples // 2)
            X[:k] = self.seeds[:k]
        return X


# ============================================================
# HYPERVOLUME
# ============================================================

def compute_hypervolume_norm(F_norm: np.ndarray) -> float:
    """HV of normalised minimisation vectors, reference point 1.1^4."""
    if F_norm is None or len(F_norm) == 0:
        return 0.0
    G = np.clip(np.atleast_2d(F_norm), 0.0, HV_REF_POINT - 1e-9)
    return float(HV(ref_point=HV_REF_POINT)(G))


def compute_hypervolume(F: np.ndarray, n_pairs: int) -> float:
    """Backwards-compatible: F in the old (-F1, F2, F3, F4) form."""
    if F is None or len(F) == 0:
        return 0.0
    G = np.atleast_2d(F).astype(float).copy()
    G[:, 0] = 1.0 + G[:, 0] / max(1, n_pairs)
    return compute_hypervolume_norm(G)


# ============================================================
# RESULT OF ONE CLEARING ROUND
# ============================================================

@dataclass
class MOEAResult:
    X: np.ndarray                 # final non-dominated solutions (bool, repaired)
    F: np.ndarray                 # same front as (-F1, F2, F3, F4) raw values
    F_norm: np.ndarray            # normalised minimisation vectors
    matched_sets: List[Set[int]]  # matched pair IDs per front solution
    hv_trace: List[float] = field(default_factory=list)
    arm_trace: List[int] = field(default_factory=list)
    reward_trace: List[float] = field(default_factory=list)
    structures: Optional[ExchangeStructures] = None
    edges: Optional[Dict] = None

    @property
    def final_hv(self) -> float:
        return self.hv_trace[-1] if self.hv_trace else 0.0

    def arm_counts(self) -> np.ndarray:
        return np.bincount(self.arm_trace, minlength=N_ARMS)


# ============================================================
# ONE CLEARING ROUND: NSGA-III (or NSGA-II) + PER-GENERATION AOS
# ============================================================

def _make_algorithm(name: str, mutation: Mutation, sampling: Sampling,
                    pop_size: int):
    common = dict(sampling=sampling, crossover=UX(prob=CROSSOVER_PROB),
                  mutation=mutation, repair=LamarckianRepair(),
                  eliminate_duplicates=True)
    if name == "nsga2":
        return NSGA2(pop_size=pop_size, **common)
    ref_dirs = get_reference_directions("das-dennis", 4,
                                        n_partitions=N_REF_PARTITIONS)
    with contextlib.redirect_stdout(io.StringIO()):   # N < |ref_dirs| warning
        return NSGA3(ref_dirs=ref_dirs, pop_size=pop_size, **common)


def run_moea_round(
    pool_pairs: List[Patient],
    aos=None,
    context: Optional[np.ndarray] = None,
    algorithm: str = "nsga3",
    n_gen: int = N_GENERATIONS,
    pop_size: int = N_POPULATION,
    seed: Optional[int] = None,
    edges: Optional[Dict] = None,
    structures: Optional[ExchangeStructures] = None,
    warm_start: Optional[List[Set[int]]] = None,
    guide_structures: Optional[List[int]] = None,
) -> MOEAResult:
    """
    Run NSGA-III (or NSGA-II) for one clearing round, with the
    mutation rate chosen by `aos` at EVERY generation.

    aos:        UCB1AOS, LinUCBAOS, FixedAOS or None (= fixed 0.10)
    context:    LinUCB context s_t (build_context_vector)
    edges / structures: pass the ones used by the ILP (module 6)
                so both optimise over the identical feasible set
    warm_start: matched sets from the previous round (dynamic runs)
    guide_structures: indices (into `structures`) of the ILP optimum;
                injected as a seed and repaired first (see header)
    """
    if edges is None:
        edges = build_compatibility_graph(pool_pairs)
    if structures is None:
        structures = build_exchange_structures(pool_pairs, edges)
    aos = aos if aos is not None else FixedAOS()
    problem = KEPProblem(pool_pairs, edges, structures=structures,
                         preferred=guide_structures)
    ids = problem.repairer.ids

    warm_start = list(warm_start or [])
    if guide_structures:
        guide = set()
        for k in guide_structures:
            guide.update(structures.nodes[k])
        warm_start = [guide] + warm_start

    seeds = None
    if warm_start:
        pos = {pid: k for k, pid in enumerate(ids)}
        seeds = np.zeros((len(warm_start), len(ids)), dtype=bool)
        for r, sol in enumerate(warm_start):
            for pid in sol:
                if pid in pos:
                    seeds[r, pos[pid]] = True

    mutation = AdaptiveBitflip()
    algo = _make_algorithm(algorithm, mutation, SeededSampling(seeds), pop_size)
    algo.setup(problem, termination=("n_gen", n_gen + 1), seed=seed, verbose=False)

    algo.next()                                    # generation 0: initial population
    hv_prev = compute_hypervolume_norm(algo.opt.get("F"))
    hv_trace, arm_trace, reward_trace = [hv_prev], [], []
    best_gain = 1e-12

    for _ in range(n_gen):
        arm, rate = aos.get_mutation_rate(context)
        mutation.rate = rate
        algo.next()
        hv_now = compute_hypervolume_norm(algo.opt.get("F"))
        gain = max(0.0, hv_now - hv_prev)
        best_gain = max(best_gain, gain)
        reward = gain / best_gain
        aos.update(arm, reward, context)
        hv_prev = hv_now
        hv_trace.append(hv_now)
        arm_trace.append(arm)
        reward_trace.append(reward)

    X = np.atleast_2d(algo.opt.get("X")).astype(bool)
    F_norm = np.atleast_2d(algo.opt.get("F"))
    matched_sets = [problem.repairer.matched_ids(x) for x in X]
    raw = np.array([problem.evaluator.evaluate(m) for m in matched_sets])
    F_old = raw.copy()
    F_old[:, 0] = -raw[:, 0]
    return MOEAResult(X, F_old, F_norm, matched_sets, hv_trace, arm_trace,
                      reward_trace, structures, edges)


def run_nsga3_with_aos(
    pool_pairs: List[Patient],
    aos_mechanism: str = 'ucb1',
    ucb1_aos: Optional[UCB1AOS] = None,
    linucb_aos: Optional[LinUCBAOS] = None,
    context: Optional[np.ndarray] = None,
    prev_hv: float = 0.0,
    n_gen: int = N_GENERATIONS,
    seed: Optional[int] = None,
    **kwargs,
):
    """
    Backwards-compatible wrapper used by module 7.
    Returns (result, final_hv, most_used_arm, its_mutation_rate).
    """
    if aos_mechanism == 'ucb1' and ucb1_aos is not None:
        aos = ucb1_aos
    elif aos_mechanism == 'linucb' and linucb_aos is not None and context is not None:
        aos = linucb_aos
    else:
        aos = FixedAOS()
    res = run_moea_round(pool_pairs, aos, context, "nsga3", n_gen,
                         seed=seed, **kwargs)
    arm = int(np.argmax(res.arm_counts())) if res.arm_trace else 0
    return res, res.final_hv, arm, MUTATION_RATES[arm]


def extract_best_solution(result: Optional[MOEAResult], pool_pairs=None,
                          edges=None) -> Set[int]:
    """Highest-F1 solution of the front (ties: smaller F3)."""
    if result is None or len(result.matched_sets) == 0:
        return set()
    idx = int(np.lexsort((result.F[:, 2], result.F[:, 0]))[0])
    return set(result.matched_sets[idx])


def repair_to_cycles(selected: Set[int], edges: Dict,
                     max_len: int = MAX_CYCLE_LEN) -> Set[int]:
    """
    Kept for compatibility with older code: packs `selected` pairs
    into disjoint cycles only (no chains). New code should use
    MOEAResult.matched_sets, which already contain the repaired
    cycles AND chains.
    """
    adj: Dict[int, Set[int]] = {}
    for (i, j) in edges:
        adj.setdefault(i, set()).add(j)
    avail, matched = set(selected), set()
    for a in sorted(selected):
        if a not in avail:
            continue
        found = None
        for b in sorted(adj.get(a, ())):
            if b == a or b not in avail:
                continue
            if a in adj.get(b, ()):
                found = (a, b)
                break
            if max_len >= 3:
                for c in sorted(adj.get(b, ())):
                    if c not in (a, b) and c in avail and a in adj.get(c, ()):
                        found = (a, b, c)
                        break
                if found:
                    break
        if found:
            matched.update(found)
            avail.difference_update(found)
    return matched


# ============================================================
# TEST MODULE 5
# ============================================================

if __name__ == "__main__":

    import time
    from module1_generator import generate_initial_pool

    TEST_GEN = 40
    print("=" * 62)
    print("MODULE 5: NSGA-III WITH PER-GENERATION AOS")
    print(f"N={N_POPULATION}, p={N_REF_PARTITIONS}, pc={CROSSOVER_PROB}, "
          f"rates={MUTATION_RATES}, W={WINDOW_SIZE}")
    print(f"(test uses {TEST_GEN} generations; full runs use {N_GENERATIONS})")
    print("=" * 62)

    pool, _ = generate_initial_pool(n_pairs=75, seed=42, initial_wait_max=12)
    pairs = pool.pairs
    edges = build_compatibility_graph(pairs)
    st = build_exchange_structures(pairs, edges)
    print(f"\nPool: {len(pool.patients())} pairs + {len(pool.altruists())} altruists, "
          f"{len(edges)} edges, {st.n_cycles()} cycles, {st.n_chains()} chains")

    ctx = build_context_vector(pairs, competing_exits=2, clearing_round=1,
                               total_rounds=25, pof_prev=0.0)
    print(f"LinUCB context s_t = {np.round(ctx, 3)}")

    for name, aos in [("UCB1", UCB1AOS(seed=1)),
                      ("LinUCB", LinUCBAOS(seed=1)),
                      ("Fixed 0.10", FixedAOS())]:
        t0 = time.time()
        res = run_moea_round(pairs, aos, ctx, "nsga3", n_gen=TEST_GEN, seed=1,
                             edges=edges, structures=st)
        f1 = -res.F[:, 0]
        alt_used = max(sum(1 for p in pool.altruists() if p.pair_id in m)
                       for m in res.matched_sets)
        print(f"\n--- {name} ---")
        print(f"  Front size:          {len(res.matched_sets)}")
        print(f"  F1 range on front:   {f1.min():.0f} .. {f1.max():.0f}")
        print(f"  F2 range on front:   {res.F[:, 1].min():.3f} .. {res.F[:, 1].max():.3f}")
        print(f"  F3 range on front:   {res.F[:, 2].min():.3f} .. {res.F[:, 2].max():.3f}")
        print(f"  F4 range on front:   {res.F[:, 3].min():.3f} .. {res.F[:, 3].max():.3f}")
        print(f"  Altruists used (max): {alt_used}")
        print(f"  HV: {res.hv_trace[0]:.4f} -> {res.final_hv:.4f}")
        counts = res.arm_counts()
        print("  Arm use per generation: " +
              ", ".join(f"{r}:{c}" for r, c in zip(MUTATION_RATES, counts)))
        print(f"  Time: {time.time() - t0:.1f} s")
        if name == "UCB1":
            res_ucb = res

    print("\n--- ILP-guided hybrid vs unguided (same structures) ---")
    from module6_ilp_baseline import _solve_packing
    sizes = np.array([st.size(k) for k in range(len(st))], dtype=float)
    ilp_k, _ = _solve_packing(st.nodes, sizes)
    print(f"  ILP optimum F1:                {sizes[ilp_k].sum():.0f}")
    print(f"  Unguided UCB1 front, max F1:   {-res_ucb.F[:, 0].min():.0f}")
    res = run_moea_round(pairs, UCB1AOS(seed=1), ctx, "nsga3", n_gen=TEST_GEN,
                         seed=1, edges=edges, structures=st,
                         guide_structures=list(ilp_k))
    f1 = -res.F[:, 0]
    print(f"  ILP-guided UCB1 front, max F1: {f1.max():.0f}  "
          f"(front size {len(f1)}, F1 range {f1.min():.0f}..{f1.max():.0f})")
    print(f"  Guided front F3 range: {res.F[:, 2].min():.3f} .. {res.F[:, 2].max():.3f}")

    print("\n--- Feasibility check ---")
    # re-repair with the same rule the guided run used: a valid
    # packing is a fixed point of the repair
    rep = StructureRepair(pairs, st, preferred=list(ilp_k))
    fixed_point = all(
        rep.matched_ids(np.array([p.pair_id in m for p in pairs])) == m
        for m in res.matched_sets)
    disjoint = True
    for m in res.matched_sets:
        mask, chosen = rep.repair(np.array([p.pair_id in m for p in pairs]))
        nodes = [v for k in chosen for v in st.nodes[k]]
        disjoint &= len(nodes) == len(set(nodes)) == len(m)
    print(f"  Every front solution is a disjoint cycle/chain packing: "
          f"{fixed_point and disjoint}")
    alt_ids = {p.pair_id for p in pool.altruists()}
    print(f"  Altruists only appear in chains: "
          f"{all(not (set(st.nodes[k]) & alt_ids) for k in range(len(st)) if st.kind[k] == 'cycle')}")

    print("\n--- UCB1 learns the best arm (toy rewards) ---")
    u = UCB1AOS(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(300):
        a, _r = u.get_mutation_rate()
        u.update(a, float(rng.random() < (0.8 if a == 2 else 0.2)))
    print(f"  Most pulled arm: rate {MUTATION_RATES[int(np.argmax(u.total_pulls))]} "
          f"(best = 0.10)")

    print("\n--- LinUCB learns a context-dependent arm (toy rewards) ---")
    lin = LinUCBAOS(alpha=0.3, seed=0)
    ctxs = [np.array([1, 0, 0, 0, 0, 0.]), np.array([0, 1, 0, 0, 0, 0.])]
    best_arm = [0, 3]
    for _ in range(400):
        k = rng.integers(2)
        a, _r = lin.get_mutation_rate(ctxs[k])
        lin.update(a, 1.0 if a == best_arm[k] else 0.0, ctxs[k])
    picks = [lin.select_arm(c) for c in ctxs]
    print(f"  Context A -> rate {MUTATION_RATES[picks[0]]} (best 0.01), "
          f"context B -> rate {MUTATION_RATES[picks[1]]} (best 0.25)")

    print("\n" + "=" * 62)
    print("MODULE 5 COMPLETE")
    print("Ready for Module 6: ILP Baseline")
    print("=" * 62)
