import time
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from module1_generator import Patient, Pool, generate_arrivals, generate_initial_pool
from module3_logistics import build_compatibility_graph

try:                                     # file name used in VS Code project
    from module2_pool_evolution import LAMBDA_ARRIVALS, MU_D, PoolEvolution
except ImportError:                      # alternative name in the plan
    from module2_evolution import LAMBDA_ARRIVALS, MU_D, PoolEvolution

from module4_objectives import compute_all_objectives, compute_f2_gini_fast, compute_pof
from module5_nsga3_aos import (
    MUTATION_RATES, N_GENERATIONS, N_POPULATION,
    KEPProblem, LinUCBAOS, UCB1AOS,
    compute_hypervolume, repair_to_cycles, run_nsga3_with_aos,
)
from module6_ilp_baseline import solve_ilp_lexicographic, solve_ilp_max_transplants

# ============================================================
# MODULE 7: CLEARING STRATEGIES + ONE DYNAMIC TRAJECTORY
# pi_M: Myopic  - clear every round
# pi_B: Batched - clear only when pool reaches n* = 90 pairs
# Paper 2 — Dynamic KEP Implementation
# ============================================================
#
# The clearing policy decides WHETHER round t is cleared.
# The AOS (UCB1 / LinUCB) works INSIDE a cleared round.
# These are separate decisions (concept note Section 3.7).

BATCH_THRESHOLD = 90       # n* (upper ANZKX pool size)
N_ROUNDS = 25              # T in 20..30
INITIAL_POOL_SIZE = 75


# ============================================================
# CLEARING POLICIES
# ============================================================

def count_patients(pool_pairs: List[Patient]) -> int:
    """Patients waiting (altruistic donors excluded)."""
    return sum(1 for p in pool_pairs if not p.is_altruistic)


class MyopicClearing:
    """pi_M: clear the pool at every round."""
    name = "myopic"

    def should_clear(self, pool: Pool, round_number: int) -> bool:
        return count_patients(pool.pairs) > 0


class BatchedClearing:
    """
    pi_B: clear only when the pool holds n* patients.

    max_interval (optional, default None = pure rule from the
    concept note): also clear if this many rounds passed since
    the last clearing. Without it pi_B never clears when the
    steady-state pool (about lambda x tau) stays below n*.
    """
    name = "batched"

    def __init__(self, threshold: int = BATCH_THRESHOLD,
                 max_interval: Optional[int] = None):
        self.threshold = threshold
        self.max_interval = max_interval
        self.last_clear = 0

    def should_clear(self, pool: Pool, round_number: int) -> bool:
        due = (self.max_interval is not None and
               round_number - self.last_clear >= self.max_interval)
        clear = count_patients(pool.pairs) >= self.threshold or due
        if clear:
            self.last_clear = round_number
        return clear


def make_policy(name: str, threshold: int = BATCH_THRESHOLD,
                max_interval: Optional[int] = None):
    if name == "myopic":
        return MyopicClearing()
    if name == "batched":
        return BatchedClearing(threshold, max_interval)
    raise ValueError(f"Unknown clearing policy: {name}")


# ============================================================
# PAIRED POOL EVOLUTION (common random numbers)
# ============================================================

class PairedPoolEvolution(PoolEvolution):
    """
    Same pool equation as module 2, but every random event is
    tied to (run seed, round) or (run seed, pair, round).

    Why: all 8 baselines in run r must see the SAME arrivals
    and the SAME competing exits, so they differ only through
    their matching decisions. This is what makes the paired
    Wilcoxon signed-rank test in module 8 valid.
    (Module 2 draws exits and arrivals from one shared RNG, so
    different matchings would shift every later random draw.)
    """

    def __init__(self, *args, base_seed: int = 42, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_seed = int(base_seed)

    def compute_competing_exits(self) -> List[Patient]:
        exits, remaining = [], []
        for pair in self.pool.pairs:
            if pair.is_altruistic:
                remaining.append(pair)
                continue
            u = np.random.default_rng(
                [self.base_seed, pair.pair_id, self.current_round, 3]
            ).random()
            (exits if u < self.mu_d else remaining).append(pair)
        self.pool.pairs = remaining
        return exits

    def compute_arrivals(self) -> List[Patient]:
        rng = np.random.RandomState(
            (self.base_seed * 7919 + self.current_round) % (2 ** 32))
        n_arrivals = max(1, rng.poisson(self.lambda_arrivals))
        # module 1 uses the global numpy RNG; seed it per round
        np.random.seed(
            (self.base_seed * 104729 + self.current_round) % (2 ** 32))
        new_pairs, self.next_pair_id = generate_arrivals(
            n_arrivals=n_arrivals,
            next_pair_id=self.next_pair_id,
            arrival_round=self.current_round,
        )
        self.pool.pairs.extend(new_pairs)
        return new_pairs


# ============================================================
# NSGA-II (baseline 1: Paper 1 framework, static)
# ============================================================

def run_nsga2_with_aos(
    pool_pairs: List[Patient],
    ucb1_aos: UCB1AOS,
    prev_hv: float = 0.0,
    n_gen: int = N_GENERATIONS,
    seed: Optional[int] = None,
    graph_builder: Callable = build_compatibility_graph,
):
    """Same interface as module 5's run_nsga3_with_aos, NSGA-II."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.pntx import TwoPointCrossover
    from pymoo.operators.mutation.bitflip import BitflipMutation
    from pymoo.operators.sampling.rnd import BinaryRandomSampling
    from pymoo.optimize import minimize

    edges = graph_builder(pool_pairs)
    arm, mutation_rate = ucb1_aos.get_mutation_rate()
    problem = KEPProblem(pool_pairs, edges, mutation_rate)
    algorithm = NSGA2(
        pop_size=N_POPULATION,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(prob=1.0, prob_var=mutation_rate),
        eliminate_duplicates=True,
    )
    result = minimize(problem, algorithm, ('n_gen', n_gen),
                      seed=seed, verbose=False)
    new_hv = 0.0
    if result is not None and result.F is not None:
        new_hv = compute_hypervolume(result.F, len(pool_pairs))
    ucb1_aos.update(arm, new_hv - prev_hv)
    return result, new_hv, arm, mutation_rate


# ============================================================
# WHICH PARETO SOLUTION IS IMPLEMENTED
# ============================================================

def select_solution(result, pool_pairs: List[Patient], edges: Dict,
                    rule: str = "floor",
                    min_f1_ratio: float = 0.9) -> Set[int]:
    """
    Pick one solution from the final front and repair it.

    IMPORTANT: F3 and F4 are absolute gaps, so the EMPTY matching
    (nobody transplanted) gives F3 = F4 = 0 and is always on the
    Pareto front. Every rule below therefore ignores solutions
    with F1 = 0 whenever a non-empty solution exists.

    rule:
      "max_f1"     highest F1 (module 5's extract_best_solution).
                   PoF then only measures the MOEA's distance from
                   the ILP optimum, not a fairness trade-off.
      "floor"      (default) among solutions with
                   F1 >= min_f1_ratio x best F1 on the front, take
                   the fairest one (compromise on F2-F4).
      "compromise" closest to the ideal point after scaling all
                   four objectives to [0, 1] over the front.
      "min_f3"     smallest high-cPRA gap, ties -> higher F1.

    The rule (and min_f1_ratio) directly shapes PoF_t, so report
    it in the paper.
    """
    if result is None or result.F is None:
        return set()
    F = np.atleast_2d(result.F)
    X = np.atleast_2d(result.X)

    cand = np.arange(len(F))
    nonempty = cand[F[:, 0] < 0]                 # F[:,0] = -F1
    if len(nonempty):
        cand = nonempty

    def compromise(idx, cols):
        G = F[np.ix_(idx, cols)]
        lo, hi = G.min(axis=0), G.max(axis=0)
        span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        return int(idx[np.argmin(np.linalg.norm((G - lo) / span,
                                                axis=1))])

    if rule == "max_f1":
        best = int(cand[np.lexsort((F[cand, 2], F[cand, 0]))[0]])
    elif rule == "min_f3":
        best = int(cand[np.lexsort((F[cand, 0], F[cand, 2]))[0]])
    elif rule == "compromise":
        best = compromise(cand, [0, 1, 2, 3])
    elif rule == "floor":
        best_f1 = -F[cand, 0].min()
        ok = cand[-F[cand, 0] >= min_f1_ratio * best_f1 - 1e-9]
        best = compromise(ok, [1, 2, 3])
    else:
        raise ValueError(f"Unknown selection rule: {rule}")

    x = np.asarray(X[best]).astype(int)
    selected = {pool_pairs[j].pair_id
                for j in range(len(pool_pairs)) if x[j] == 1}
    return repair_to_cycles(selected, edges)


# ============================================================
# ONE TRAJECTORY: T ROUNDS FOR ONE CONFIGURATION
# ============================================================

def simulate_trajectory(
    config: Dict,
    tau: int,
    run_seed: int,
    n_rounds: int = N_ROUNDS,
    n_gen: int = N_GENERATIONS,
    n_pairs: int = INITIAL_POOL_SIZE,
    selection_rule: str = "floor",
    min_f1_ratio: float = 0.9,
    batch_threshold: int = BATCH_THRESHOLD,
    batch_max_interval: Optional[int] = None,
    mu_d: float = MU_D,
    lambda_arrivals: float = LAMBDA_ARRIVALS,
    graph_builder: Callable = build_compatibility_graph,
    initial_pool: Optional[Tuple[Pool, int]] = None,
    evolution_factory: Optional[Callable] = None,
    initial_wait_max: int = 6,
    verbose: bool = False,
) -> List[Dict]:
    """
    Simulate one configuration for n_rounds clearing rounds.

    config keys:
      name     label, e.g. "dyn_nsga3_ucb1_myopic"
      method   "nsga3" | "nsga2" | "ilp" | "lex"
      aos      "ucb1" | "linucb" | "none"
      clearing "myopic" | "batched"
      dynamic  True  -> AOS state + previous HV kept across rounds
               False -> fresh AOS every round (static baselines)

    initial_wait_max: module 1 starts every initial pair at
      waiting_time 0, so the whole initial pool withdraws in the
      same round (round tau). To avoid that cliff, initial pairs
      get a seeded waiting time uniform in 0..initial_wait_max-1.
      Set 0 to keep module 1's behaviour.

    Returns one dict per round.
    """
    method = config["method"]
    aos_name = config.get("aos", "none")
    dynamic = config.get("dynamic", True)

    if initial_pool is None:
        pool, next_id = generate_initial_pool(n_pairs=n_pairs,
                                              seed=run_seed)
    else:
        pool, next_id = initial_pool
    if initial_wait_max > 0:
        wrng = np.random.default_rng([run_seed, 11])
        for p in pool.pairs:
            p.waiting_time = int(wrng.integers(0, initial_wait_max))
    if evolution_factory is None:
        evo = PairedPoolEvolution(
            initial_pool=pool, next_pair_id=next_id, tau_months=tau,
            mu_d=mu_d, lambda_arrivals=lambda_arrivals,
            seed=run_seed, base_seed=run_seed)
    else:
        evo = evolution_factory(pool, next_id, tau)
    policy = make_policy(config["clearing"], batch_threshold,
                         batch_max_interval)

    ucb1 = UCB1AOS()
    linucb = LinUCBAOS()
    prev_hv, prev_pof, prev_exits = 0.0, 0.0, 0
    rows: List[Dict] = []

    for r in range(1, n_rounds + 1):
        pairs = evo.pool.pairs
        high = [p for p in pairs
                if p.is_highly_sensitised and not p.is_altruistic]
        row = {
            "config": config["name"], "tau": tau, "run_seed": run_seed,
            "round": r, "pool_size": len(pairs),
            "n_patients": count_patients(pairs),
            "n_high_cpra": len(high), "cleared": False,
        }
        matched: Set[int] = set()

        if policy.should_clear(evo.pool, r):
            t0 = time.time()
            edges = graph_builder(pairs)
            ilp_ids, f1_ilp, ilp_info = solve_ilp_max_transplants(
                pairs, edges)
            hv, arm, rate = np.nan, -1, np.nan
            round_seed = (run_seed * 1000 + r) % (2 ** 31)

            if method == "ilp":
                matched = ilp_ids
            elif method == "lex":
                matched, _, _ = solve_ilp_lexicographic(pairs, edges)
            else:
                if not dynamic:              # static baseline
                    ucb1, linucb, prev_hv = UCB1AOS(), LinUCBAOS(), 0.0
                if method == "nsga2":
                    result, hv, arm, rate = run_nsga2_with_aos(
                        pairs, ucb1, prev_hv, n_gen, round_seed,
                        graph_builder)
                else:
                    context = linucb.build_context_vector(
                        pool_pairs=pairs,
                        competing_exits=prev_exits,
                        clearing_round=r,
                        gini_t=compute_f2_gini_fast(pairs),
                        pof_prev=prev_pof / 100.0,  # PoF as fraction
                    )
                    result, hv, arm, rate = run_nsga3_with_aos(
                        pool_pairs=pairs, aos_mechanism=aos_name,
                        ucb1_aos=ucb1, linucb_aos=linucb,
                        context=context, prev_hv=prev_hv,
                        n_gen=n_gen, seed=round_seed)
                prev_hv = hv
                matched = select_solution(result, pairs, edges,
                                          selection_rule, min_f1_ratio)

            obj = compute_all_objectives(matched, pairs)
            # PoF is undefined when no transplant was possible
            # (F1_ILP = 0); store NaN so it is excluded from means.
            pof = compute_pof(f1_ilp, obj["F1"]) if f1_ilp > 0 \
                else float("nan")
            if method in ("nsga3", "nsga2") and f1_ilp > 0:
                prev_pof = pof
            row.update({
                "cleared": True,
                "n_edges": len(edges),
                "n_cycles": ilp_info["n_cycles"],
                "F1_ILP": f1_ilp,
                "F1": obj["F1"], "F2": obj["F2"],
                "F3": obj["F3"], "F4": obj["F4"],
                "PoF": pof, "HV": hv, "arm": arm,
                "mutation_rate": rate,
                "high_cpra_matched": sum(1 for p in high
                                         if p.pair_id in matched),
                "seconds": time.time() - t0,
            })

        stats = evo.evolve_round(matched_pair_ids=list(matched))
        prev_exits = stats["competing_exits"]
        row.update({
            "matched": stats["matched"],
            "withdrawn": stats["withdrawn"],
            "competing_exits": stats["competing_exits"],
            "arrivals": stats["arrivals"],
        })
        rows.append(row)

        if verbose:
            if row["cleared"]:
                pof_txt = ("   n/a" if np.isnan(row["PoF"])
                           else f"{row['PoF']:6.2f}")
                print(f"  r={r:<3} pool={row['pool_size']:>3} "
                      f"F1={row['F1']:>4.0f}/{row['F1_ILP']:<4.0f} "
                      f"PoF={pof_txt}%  F3={row['F3']:.3f} "
                      f"arm={row['arm']}")
            else:
                print(f"  r={r:<3} pool={row['pool_size']:>3} "
                      f"(not cleared)")
    return rows


# ============================================================
# TEST MODULE 7
# ============================================================

if __name__ == "__main__":

    print("=" * 55)
    print("MODULE 7: MYOPIC vs BATCHED CLEARING")
    print("=" * 55)

    TEST_ROUNDS, TEST_GEN, TAU = 12, 20, 12

    for clearing in ("myopic", "batched"):
        cfg = {"name": f"dyn_nsga3_ucb1_{clearing}", "method": "nsga3",
               "aos": "ucb1", "clearing": clearing, "dynamic": True}
        print(f"\n--- {cfg['name']} (tau={TAU}, "
              f"{TEST_ROUNDS} rounds, n_gen={TEST_GEN}) ---")
        rows = simulate_trajectory(cfg, tau=TAU, run_seed=1,
                                   n_rounds=TEST_ROUNDS,
                                   n_gen=TEST_GEN, verbose=True)
        cleared = [r for r in rows if r["cleared"]]
        if cleared:
            print(f"  Rounds cleared:    {len(cleared)}")
            print(f"  Total transplants: "
                  f"{sum(r['F1'] for r in cleared):.0f}")
            print(f"  Mean PoF:          "
                  f"{np.nanmean([r['PoF'] for r in cleared]):.2f}%")
        else:
            print("  Pool never reached n* - batched never cleared.")
            print("  Use batch_max_interval or a higher lambda.")

    # Paired design check: same arrivals for different configs
    a = simulate_trajectory({"name": "ilp", "method": "ilp",
                             "clearing": "myopic"}, 12, 7, n_rounds=5)
    b = simulate_trajectory({"name": "lex", "method": "lex",
                             "clearing": "myopic"}, 12, 7, n_rounds=5)
    same = [x["arrivals"] for x in a] == [y["arrivals"] for y in b]
    print(f"\nPaired design (identical arrivals across configs): "
          f"{same}")

    print("\n" + "=" * 55)
    print("MODULE 7 COMPLETE")
    print("Ready for Module 8: Statistics")
    print("=" * 55)
