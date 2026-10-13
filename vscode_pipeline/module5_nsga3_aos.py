import random
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.core.result import Result
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from module1_generator import Patient, Pool
from module3_logistics import (
    build_compatibility_graph,
    DISTANT_NODES, CENTRAL_NODES
)
from module4_objectives import (
    compute_f1_efficiency,
    compute_f2_gini_fast,
    compute_f3_group_fairness,
    compute_f4_logistics_disparity
)

# ============================================================
# MODULE 5: NSGA-III WITH ADAPTIVE OPERATOR SELECTION
# UCB1 Stateless Bandit + LinUCB Contextual Bandit
# Four Objectives: F1 F2 F3 F4
# Paper 2 - Dynamic KEP Implementation
# ============================================================

# --- NSGA-III Parameters ---
N_POPULATION = 100
N_GENERATIONS = 150
N_REF_DIRS = 5            # Das-Dennis partitions (4 obj -> 56 directions)
MAX_CYCLE_LEN = 3         # MUST match the ILP in module 6

# --- AOS Parameters ---
MUTATION_RATES = [0.01, 0.05, 0.10, 0.25]
N_ARMS = len(MUTATION_RATES)
WINDOW_SIZE = 10          # sliding window for UCB1
UCB1_C = 1.0              # exploration constant
ALPHA_LINUCB = 1.0
CONTEXT_DIM = 6


# ============================================================
# CYCLE-BASED REPAIR (same notion of "matched" as the ILP)
# ============================================================

def repair_to_cycles(
    selected: Set[int],
    edges: Dict,
    max_len: int = MAX_CYCLE_LEN
) -> Set[int]:
    """
    Keep only pairs that can be placed in disjoint exchange
    cycles (length 2 or 3) built from the selected pairs.
    Each pair appears in at most one cycle, as in the ILP.
    """
    adj: Dict[int, Set[int]] = {}
    for (i, j) in edges.keys():
        adj.setdefault(i, set()).add(j)

    avail = set(selected)
    matched: Set[int] = set()

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
                    if c in (a, b) or c not in avail:
                        continue
                    if a in adj.get(c, ()):
                        found = (a, b, c)
                        break
                if found:
                    break
        if found:
            matched.update(found)
            avail.difference_update(found)

    return matched


def compute_all_objectives(
    matched_ids: Set[int],
    pool_pairs: List[Patient]
) -> Tuple[float, float, float, float]:
    """
    Returns (F1, F2, F3, F4) for a matched set.
    F1 is a count to MAXIMISE, F2-F4 are to MINIMISE.
    """
    f1 = compute_f1_efficiency(matched_ids, pool_pairs)
    f2 = compute_f2_gini_fast(pool_pairs)
    f3 = compute_f3_group_fairness(matched_ids, pool_pairs)
    f4 = compute_f4_logistics_disparity(matched_ids, pool_pairs)
    return f1, f2, f3, f4


# ============================================================
# AOS 1: UCB1 (stateless, sliding window)
# ============================================================

class UCB1AOS:
    def __init__(self, n_arms: int = N_ARMS,
                 window_size: int = WINDOW_SIZE,
                 c: float = UCB1_C):
        self.n_arms = n_arms
        self.window = deque(maxlen=window_size)  # (arm, reward)
        self.c = c
        self.total_pulls = np.zeros(n_arms, dtype=int)

    def _stats(self):
        counts = np.zeros(self.n_arms)
        sums = np.zeros(self.n_arms)
        for arm, reward in self.window:
            counts[arm] += 1
            sums[arm] += reward
        return counts, sums

    def select_arm(self) -> int:
        counts, sums = self._stats()
        for arm in range(self.n_arms):       # try every arm once
            if counts[arm] == 0:
                return arm
        n = counts.sum()
        means = sums / counts
        bonus = self.c * np.sqrt(2.0 * np.log(n) / counts)
        return int(np.argmax(means + bonus))

    def get_mutation_rate(self) -> Tuple[int, float]:
        arm = self.select_arm()
        self.total_pulls[arm] += 1
        return arm, MUTATION_RATES[arm]

    def update(self, arm: int, reward: float) -> None:
        self.window.append((arm, reward))

    def get_arm_distribution(self) -> Dict[int, float]:
        total = self.total_pulls.sum()
        if total == 0:
            return {a: 0.0 for a in range(self.n_arms)}
        return {a: self.total_pulls[a] / total
                for a in range(self.n_arms)}


# ============================================================
# AOS 2: LinUCB (contextual)
# ============================================================

class LinUCBAOS:
    def __init__(self, n_arms: int = N_ARMS,
                 context_dim: int = CONTEXT_DIM,
                 alpha: float = ALPHA_LINUCB):
        self.n_arms = n_arms
        self.d = context_dim
        self.alpha = alpha
        self.A = [np.eye(self.d) for _ in range(n_arms)]
        self.b = [np.zeros(self.d) for _ in range(n_arms)]
        self.total_pulls = np.zeros(n_arms, dtype=int)

    def build_context_vector(
        self,
        pool_pairs: List[Patient],
        competing_exits: int,
        clearing_round: int,
        gini_t: float,
        pof_prev: float
    ) -> np.ndarray:
        n = max(len(pool_pairs), 1)
        waits = [getattr(p, "waiting_time", 0) for p in pool_pairs]
        mean_wait = float(np.mean(waits)) if waits else 0.0
        x = np.array([
            min(n / 100.0, 1.0),
            min(mean_wait / 10.0, 1.0),
            min(competing_exits / 10.0, 1.0),
            min(clearing_round / 10.0, 1.0),
            float(np.clip(gini_t, 0.0, 1.0)),
            float(np.clip(pof_prev, 0.0, 1.0)),
        ])
        return x

    def select_arm(self, context: np.ndarray) -> int:
        scores = []
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p = theta @ context + self.alpha * np.sqrt(
                context @ A_inv @ context
            )
            scores.append(p)
        return int(np.argmax(scores))

    def get_mutation_rate(self, context: np.ndarray) -> Tuple[int, float]:
        arm = self.select_arm(context)
        self.total_pulls[arm] += 1
        return arm, MUTATION_RATES[arm]

    def update(self, arm: int, context: np.ndarray,
               reward: float) -> None:
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context

    def get_arm_distribution(self) -> Dict[int, float]:
        total = self.total_pulls.sum()
        if total == 0:
            return {a: 0.0 for a in range(self.n_arms)}
        return {a: self.total_pulls[a] / total
                for a in range(self.n_arms)}


# ============================================================
# PROBLEM DEFINITION
# ============================================================

class KEPProblem(Problem):
    """
    Four-objective kidney exchange problem.
    Variable x_i in {0,1}: pair i is selected.
    Selected pairs are repaired to disjoint cycles, so the
    reported F1 is a count of pairs in real exchanges.
    Objectives: [-F1, F2, F3, F4] (all minimised).
    """

    def __init__(
        self,
        pool_pairs: List[Patient],
        edges: Dict,
        mutation_rate: float = 0.10
    ):
        self.pool_pairs = pool_pairs
        self.edges = edges
        self.mutation_rate = mutation_rate
        self.n_pairs = len(pool_pairs)

        super().__init__(
            n_var=self.n_pairs,
            n_obj=4,
            n_ieq_constr=0,
            xl=0,
            xu=1,
            vtype=bool
        )

    def _evaluate(self, X, out, *args, **kwargs):
        n_pop = X.shape[0]
        F = np.zeros((n_pop, 4))

        for i in range(n_pop):
            solution = np.asarray(X[i]).astype(int)

            selected = {
                self.pool_pairs[j].pair_id
                for j in range(self.n_pairs)
                if solution[j] == 1
            }
            matched_ids = repair_to_cycles(selected, self.edges)

            f1, f2, f3, f4 = compute_all_objectives(
                matched_ids, self.pool_pairs
            )
            F[i] = [-f1, f2, f3, f4]

        out["F"] = F


# ============================================================
# HYPERVOLUME
# ============================================================

def compute_hypervolume(F: np.ndarray, n_pairs: int) -> float:
    """Hypervolume of a front, with F1 scaled to [0,1]."""
    if F is None or len(F) == 0:
        return 0.0
    G = F.copy()
    G[:, 0] = G[:, 0] / max(n_pairs, 1)      # -F1 in [-1, 0]
    ref_point = np.array([1.0, 1.1, 1.1, 1.1])
    G = np.clip(G, None, ref_point - 1e-9)
    return float(HV(ref_point=ref_point)(G))  # type: ignore[arg-type]


# ============================================================
# ONE CLEARING ROUND
# ============================================================

def run_nsga3_with_aos(
    pool_pairs: List[Patient],
    aos_mechanism: str = 'ucb1',
    ucb1_aos: Optional[UCB1AOS] = None,
    linucb_aos: Optional[LinUCBAOS] = None,
    context: Optional[np.ndarray] = None,
    prev_hv: float = 0.0,
    n_gen: int = N_GENERATIONS,
    seed: Optional[int] = None
) -> Tuple[Optional[Result], float, int, float]:
    """
    Run NSGA-III with AOS for one clearing round.

    Returns:
        result: NSGA-III result object
        new_hv: hypervolume of result
        selected_arm: which mutation rate was selected
        mutation_rate: actual mutation rate used
    """
    if len(pool_pairs) == 0:
        return None, 0.0, 0, 0.10

    edges = build_compatibility_graph(pool_pairs)

    # Select mutation rate via AOS
    if aos_mechanism == 'ucb1' and ucb1_aos:
        arm, mutation_rate = ucb1_aos.get_mutation_rate()
    elif aos_mechanism == 'linucb' and linucb_aos \
            and context is not None:
        arm, mutation_rate = linucb_aos.get_mutation_rate(context)
    else:
        arm, mutation_rate = 0, 0.10

    problem = KEPProblem(
        pool_pairs=pool_pairs,
        edges=edges,
        mutation_rate=mutation_rate
    )

    ref_dirs = get_reference_directions(
        "das-dennis", 4, n_partitions=N_REF_DIRS
    )
    pop_size = max(N_POPULATION, len(ref_dirs))

    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        sampling=BinaryRandomSampling(),  # type: ignore[arg-type]
        crossover=TwoPointCrossover(),  # type: ignore[arg-type]
        # the mutation rate chosen by the bandit is applied here
        mutation=BitflipMutation(prob=1.0, prob_var=mutation_rate),  # type: ignore[arg-type]
        eliminate_duplicates=True
    )

    result = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        seed=seed,
        verbose=False
    )

    new_hv = 0.0
    if result is not None and result.F is not None:
        new_hv = compute_hypervolume(result.F, len(pool_pairs))

    # Reward = hypervolume improvement over previous round
    hv_improvement = new_hv - prev_hv

    if aos_mechanism == 'ucb1' and ucb1_aos:
        ucb1_aos.update(arm, hv_improvement)
    elif aos_mechanism == 'linucb' and linucb_aos \
            and context is not None:
        linucb_aos.update(arm, context, hv_improvement)

    return result, new_hv, arm, mutation_rate


def extract_best_solution(
    result: Optional[Result],
    pool_pairs: List[Patient],
    edges: Dict
) -> Set[int]:
    """
    Extract the solution with the highest F1 from the Pareto
    front, repaired to disjoint cycles. In the full
    implementation this will use a selection policy based on
    the clearing strategy.
    """
    if result is None or result.F is None:
        return set()

    F = np.atleast_2d(result.F)
    X = np.atleast_2d(result.X)  # type: ignore[arg-type]

    best_idx = int(np.argmin(F[:, 0]))
    best_x = np.asarray(X[best_idx]).astype(int)

    selected = {
        pool_pairs[j].pair_id
        for j in range(len(pool_pairs))
        if best_x[j] == 1
    }
    return repair_to_cycles(selected, edges)


# ============================================================
# TEST MODULE 5
# ============================================================

if __name__ == "__main__":

    from module1_generator import generate_initial_pool

    print("=" * 55)
    print("MODULE 5: NSGA-III WITH AOS TEST")
    print("Running 3 rounds with UCB1 and LinUCB")
    print("Note: Using reduced generations for speed")
    print("=" * 55)

    pool, next_id = generate_initial_pool(n_pairs=30, seed=42)

    random.seed(42)
    for p in pool.pairs:
        p.waiting_time = random.randint(0, 5)

    ucb1 = UCB1AOS(n_arms=N_ARMS, window_size=WINDOW_SIZE)
    linucb = LinUCBAOS(
        n_arms=N_ARMS,
        context_dim=CONTEXT_DIM,
        alpha=ALPHA_LINUCB
    )

    prev_hv = 0.0
    prev_pof = 0.0

    print(f"\nPool size: {pool.size()}")
    print("\nRunning 3 rounds (reduced n_gen=10 for test)...")

    print(f"\n{'Round':<8} {'Arm':>5} {'MutRate':>9} "
          f"{'F1':>6} {'F2':>8} {'F3':>8} {'F4':>8}")
    print("-" * 55)

    for r in range(1, 4):

        edges = build_compatibility_graph(pool.pairs)
        gini_t = compute_f2_gini_fast(pool.pairs)

        context = linucb.build_context_vector(
            pool_pairs=pool.pairs,
            competing_exits=1,
            clearing_round=r,
            gini_t=gini_t,
            pof_prev=prev_pof
        )

        result, new_hv, arm, mut_rate = run_nsga3_with_aos(
            pool_pairs=pool.pairs,
            aos_mechanism='ucb1',
            ucb1_aos=ucb1,
            prev_hv=prev_hv,
            n_gen=10,
            seed=r          # different seed per round
        )

        prev_hv = new_hv

        if result is not None and result.F is not None:
            F = np.atleast_2d(result.F)
            best_idx = int(np.argmin(F[:, 0]))
            f1 = -F[best_idx, 0]
            f2 = F[best_idx, 1]
            f3 = F[best_idx, 2]
            f4 = F[best_idx, 3]
            prev_pof = 0.0
        else:
            f1 = f2 = f3 = f4 = 0.0

        print(f"  {r:<6} {arm:>5} {mut_rate:>9.2f} "
              f"{f1:>6.1f} {f2:>8.4f} "
              f"{f3:>8.4f} {f4:>8.4f}")

    print("\n--- UCB1 Arm Selection Distribution ---")
    dist = ucb1.get_arm_distribution()
    for arm_idx, pct in dist.items():
        print(f"  Rate {MUTATION_RATES[arm_idx]}: {pct*100:.1f}%")

    print("\n" + "=" * 55)
    print("MODULE 5 COMPLETE")
    print("Ready for Module 6: ILP Baseline")
    print("=" * 55)