import time
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from module1_generator import Patient, Pool, generate_initial_pool
from module2_pool_evolution import LAMBDA_ARRIVALS, MU_D, PoolEvolution
from module3_logistics import add_chains, build_compatibility_graph, build_exchange_structures
from module4_objectives import compute_all_objectives, compute_pof, round_metrics
from module5_nsga3_aos import (
    MUTATION_RATES, N_GENERATIONS, N_POPULATION,
    FixedAOS, LinUCBAOS, MOEAResult, UCB1AOS,
    build_context_vector, run_moea_round,
)
from module6_ilp_baseline import (solve_ilp_lexicographic, solve_ilp_max_transplants,
                                  solve_ilp_picef, solve_ilp_two_stage)

# ============================================================
# MODULE 7: CLEARING STRATEGIES + ONE DYNAMIC TRAJECTORY
#   pi_M  Myopic:  clear the pool at every round
#   pi_B  Batched: clear only when the pool holds n* = 90 patients
# Concept note Sections 3.6, 3.7 and 4.4
# Paper 2 — Dynamic KEP Implementation
# ============================================================
#
# One clearing round t (when the policy says "clear"):
#   1. G_t   = compatibility graph of V_t             (module 3)
#   2. S_t   = cycles + chains of G_t, built ONCE     (module 3)
#   3. ILP on S_t -> F1_ILP(t) and its structures     (module 6)
#   4. the configuration's matching on S_t:
#        ILP / Dickerson lexicographic                (module 6)
#        NSGA-III / NSGA-II + AOS, ILP-guided         (module 5)
#   5. one solution of the Pareto front is IMPLEMENTED
#      (selection rule, see select_solution)
#   6. PoF_t = (F1_ILP - F1_implemented) / F1_ILP      (Eq. 13)
# Then the pool evolves: V_{t+1} = V_t \ M_t \ X_t \ W_t U A_t
# (module 2). If the policy says "wait", only step 6's pool
# evolution happens (no matching).
#
# Clearing policy (WHETHER to clear) and AOS (HOW to search
# inside a round) are separate decisions (Section 3.7).
#
# Dynamic vs static configurations (Section 4.4):
#   dynamic: AOS state persists across rounds and the previous
#            round's front warm-starts the next round
#   static:  fresh AOS and no warm start in every round (Paper 1)

BATCH_THRESHOLD = 90       # n* (upper ANZKX pool size)
N_ROUNDS = 25              # T in 20..30
INITIAL_POOL_SIZE = 75
INITIAL_WAIT_MAX = 6       # spread initial waiting times over 0..5
SELECTION_RULE = "floor"
MIN_F1_RATIO = 0.9         # efficiency floor for the "floor" rule


# ============================================================
# CLEARING POLICIES
# ============================================================

def count_patients(pool_pairs: List[Patient]) -> int:
    """Patients waiting (altruistic donors excluded)."""
    return sum(1 for p in pool_pairs if not p.is_altruistic)


class MyopicClearing:
    """pi_M: clear the pool at every round (primary baseline policy)."""
    name = "myopic"

    def should_clear(self, pool: Pool, round_number: int) -> bool:
        return count_patients(pool.pairs) > 0


class BatchedClearing:
    """
    pi_B: clear only when the pool holds n* patients.

    max_interval (optional, default None = the rule as written):
    also clear if this many rounds passed since the last clearing.
    Without it pi_B never clears when the steady-state pool stays
    below n* (e.g. small lambda with tau = 6).
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


# Module 2 now has paired random numbers built in; this alias keeps
# module 9 (which subclasses it) working.
PairedPoolEvolution = PoolEvolution


# ============================================================
# WHICH PARETO SOLUTION IS IMPLEMENTED
# ============================================================

def select_solution(result: MOEAResult, rule: str = SELECTION_RULE,
                    min_f1_ratio: float = MIN_F1_RATIO) -> Tuple[Set[int], int]:
    """
    Choose the solution of the front that is actually implemented.
    Returns (matched pair IDs, index in the front).

    The EMPTY matching has F3 = F4 = 0 and is always Pareto-optimal,
    so every rule ignores F1 = 0 when a non-empty solution exists.

    rule
      "floor"      (default) among solutions with
                   F1 >= min_f1_ratio x max F1 on the front, the one
                   closest to the ideal point on F2, F3, F4
                   (each scaled to [0, 1] over that subset)
      "compromise" closest to the ideal point on all four objectives
      "max_f1"     most transplants (ties -> smaller F3); PoF = 0
      "min_f3"     smallest high-cPRA gap (ties -> more transplants)

    The rule and min_f1_ratio shape PoF_t: report them in the paper.
    """
    if result is None or len(result.matched_sets) == 0:
        return set(), -1
    F = np.atleast_2d(result.F)           # (-F1, F2, F3, F4)
    cand = np.arange(len(F))
    nonempty = cand[F[:, 0] < 0]
    if len(nonempty):
        cand = nonempty

    def compromise(idx, cols):
        G = F[np.ix_(idx, cols)]
        lo, hi = G.min(axis=0), G.max(axis=0)
        span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        return int(idx[np.argmin(np.linalg.norm((G - lo) / span, axis=1))])

    if rule == "max_f1":
        best = int(cand[np.lexsort((F[cand, 2], F[cand, 0]))[0]])
    elif rule == "min_f3":
        best = int(cand[np.lexsort((F[cand, 0], F[cand, 2]))[0]])
    elif rule == "compromise":
        best = compromise(cand, [0, 1, 2, 3])
    elif rule == "floor":
        f1_max = -F[cand, 0].min()
        ok = cand[-F[cand, 0] >= min_f1_ratio * f1_max - 1e-9]
        best = compromise(ok, [1, 2, 3])
    else:
        raise ValueError(f"Unknown selection rule: {rule}")
    return set(result.matched_sets[best]), best


# ============================================================
# ONE TRAJECTORY: T ROUNDS FOR ONE CONFIGURATION
# ============================================================

def make_aos(name: str, seed: int):
    if name == "ucb1":
        return UCB1AOS(seed=seed)
    if name == "linucb":
        return LinUCBAOS(seed=seed)
    return FixedAOS()


def _shadow_two_stage(res, pairs, edges, structures, ilp_ids, f1_ilp) -> Dict:
    """
    Same-pool comparison (module 12): on the pool of this round, solve
    the two-stage ILP (both objectives) and report, next to it, the
    MOEA front solutions with the maximum volume F1 = F1_ILP. Does not
    change the trajectory.
    """
    out: Dict = {}
    o = compute_all_objectives(ilp_ids, pairs)
    ref = (o["F2"], o["F3"], o["F4"])
    out.update({"ilp_F2": o["F2"], "ilp_F3": o["F3"], "ilp_F4": o["F4"]})
    for obj in ("sum", "f3"):
        M, info = solve_ilp_two_stage(pairs, edges, structures, f1_ilp, ref,
                                      objective=obj)
        q = compute_all_objectives(M if (M or f1_ilp == 0) else ilp_ids, pairs)
        out.update({f"ts_{obj}_F1": q["F1"], f"ts_{obj}_F2": q["F2"],
                    f"ts_{obj}_F3": q["F3"], f"ts_{obj}_F4": q["F4"],
                    f"ts_{obj}_sec": info.get("seconds", np.nan),
                    f"ts_{obj}_status": info.get("status", "")})
    F = np.atleast_2d(res.F) if res is not None and len(res.matched_sets) else None
    if F is None:
        out["moea_maxvol_n"] = 0
        return out
    mask = -F[:, 0] >= f1_ilp - 1e-9
    out["moea_maxvol_n"] = int(mask.sum())
    if mask.any():
        out.update({"moea_maxvol_minF2": float(F[mask, 1].min()),
                    "moea_maxvol_minF3": float(F[mask, 2].min()),
                    "moea_maxvol_minF4": float(F[mask, 3].min())})
        pick, _ = select_solution(res, "floor", 1.0)      # 100% rule
        q = compute_all_objectives(pick, pairs)
        out.update({"moea_pick100_F1": q["F1"], "moea_pick100_F2": q["F2"],
                    "moea_pick100_F3": q["F3"], "moea_pick100_F4": q["F4"]})
    return out


def simulate_trajectory(
    config: Dict,
    tau: int,
    run_seed: int,
    n_rounds: int = N_ROUNDS,
    n_gen: int = N_GENERATIONS,
    pop_size: int = N_POPULATION,
    n_pairs: int = INITIAL_POOL_SIZE,
    selection_rule: str = SELECTION_RULE,
    min_f1_ratio: float = MIN_F1_RATIO,
    batch_threshold: int = BATCH_THRESHOLD,
    batch_max_interval: Optional[int] = None,
    mu_d: float = MU_D,
    lambda_arrivals: float = LAMBDA_ARRIVALS,
    initial_wait_max: int = INITIAL_WAIT_MAX,
    ilp_guided: bool = True,
    graph_builder: Callable = build_compatibility_graph,
    initial_pool: Optional[Tuple[Pool, int]] = None,
    evolution_factory: Optional[Callable] = None,
    verbose: bool = False,
) -> List[Dict]:
    """
    Simulate one configuration for n_rounds clearing rounds.

    config keys
      name      label, e.g. "dyn_nsga3_ucb1_myopic"
      method    "nsga3" | "nsga2" | "ilp" | "lex" | "ilp2"
      objective (ilp2 only) "sum" | "f3", see module 6
      shadow_two_stage (MOEA only) also solve the two-stage ILP on the
                same pool each round and record both (module 12)
      aos       "ucb1" | "linucb" | "none"
      clearing  "myopic" | "batched"
      dynamic   True: AOS persists + warm start; False: static

    All configurations with the same run_seed see the same initial
    pool, arrivals and competing exits (paired design).
    Returns one dict per round.
    """
    method = config["method"]
    aos_name = config.get("aos", "none")
    dynamic = config.get("dynamic", True)

    if initial_pool is None:
        pool, next_id = generate_initial_pool(n_pairs=n_pairs, seed=run_seed,
                                              initial_wait_max=initial_wait_max)
    else:
        pool, next_id = initial_pool
    if evolution_factory is None:
        evo = PoolEvolution(pool, next_id, tau_months=tau, mu_d=mu_d,
                            lambda_arrivals=lambda_arrivals, seed=run_seed)
    else:
        evo = evolution_factory(pool, next_id, tau)
    policy = make_policy(config["clearing"], batch_threshold, batch_max_interval)

    aos = make_aos(aos_name, run_seed)
    prev_pof, prev_exits = 0.0, 0
    warm: List[Set[int]] = []
    rows: List[Dict] = []

    for r in range(1, n_rounds + 1):
        pairs = evo.pool.pairs
        pts = [p for p in pairs if not p.is_altruistic]
        high = [p for p in pts if p.is_highly_sensitised]
        row = {
            "config": config["name"], "tau": tau, "run_seed": run_seed,
            "round": r, "pool_size": len(pairs), "n_patients": len(pts),
            "n_altruists": len(pairs) - len(pts), "n_high_cpra": len(high),
            "high_cpra_share": len(high) / max(1, len(pts)),
            "mean_wait": float(np.mean([p.waiting_time for p in pts])) if pts else 0.0,
            "cleared": False,
        }
        matched: Set[int] = set()

        if policy.should_clear(evo.pool, r):
            t0 = time.time()
            edges = graph_builder(pairs)
            structures = build_exchange_structures(pairs, edges)
            picef_added, picef_failed = 0, False
            two_stage_status = ""
            shadow: Dict = {}
            if structures.truncated:
                # chain list was capped: add the chains of the exact
                # (PICEF) optimum, so that F1_ILP is the true optimum
                _, pinfo = solve_ilp_picef(pairs, edges)
                picef_added = add_chains(structures, pinfo["chains"], edges)
                picef_failed = bool(pinfo.get("failed", False))
            ilp_ids, f1_ilp, ilp_info = solve_ilp_max_transplants(
                pairs, edges, structures)
            hv, front_size, exact = np.nan, 0, False
            arm_counts = np.zeros(len(MUTATION_RATES), dtype=int)

            if method == "ilp":
                matched = ilp_ids
            elif method == "lex":
                matched, _, _ = solve_ilp_lexicographic(pairs, edges, structures)
            elif method == "ilp2":
                # two-stage ILP: fairest matching among the maximum ones
                o = compute_all_objectives(ilp_ids, pairs)
                matched, ts_info = solve_ilp_two_stage(
                    pairs, edges, structures, f1_ilp,
                    (o["F2"], o["F3"], o["F4"]),
                    objective=config.get("objective", "sum"))
                if f1_ilp > 0 and not matched:      # solver failed
                    matched = ilp_ids
                two_stage_status = ts_info.get("status", "")
            else:
                if not dynamic:                    # static: Paper 1 style
                    aos = make_aos(aos_name, run_seed * 1000 + r)
                context = build_context_vector(pairs, prev_exits, r,
                                               n_rounds, prev_pof)
                res = run_moea_round(
                    pairs, aos, context,
                    algorithm=method, n_gen=n_gen, pop_size=pop_size,
                    seed=(run_seed * 1000 + r) % (2 ** 31),
                    edges=edges, structures=structures,
                    warm_start=warm if dynamic else None,
                    guide_structures=(ilp_info["structure_indices"]
                                      if ilp_guided else None),
                )
                matched, _ = select_solution(res, selection_rule, min_f1_ratio)
                hv, front_size = res.final_hv, len(res.matched_sets)
                exact = bool(getattr(res, "exact", False))
                if config.get("shadow_two_stage"):
                    shadow = _shadow_two_stage(res, pairs, edges, structures,
                                               ilp_ids, f1_ilp)
                arm_counts = res.arm_counts()
                if dynamic:
                    warm = res.matched_sets

            obj = compute_all_objectives(matched, pairs)
            pof = compute_pof(f1_ilp, obj["F1"])    # NaN if F1_ILP = 0
            if method in ("nsga3", "nsga2") and not np.isnan(pof):
                prev_pof = pof
            rm = round_metrics(matched, pairs)
            row.update({
                "cleared": True,
                "n_edges": len(edges),
                "n_cycles": structures.n_cycles(),
                "n_chains": structures.n_chains(),
                "chains_truncated": structures.truncated,
                "picef_chains_added": picef_added,
                "picef_failed": picef_failed,
                "ilp_solver": ilp_info.get("solver", ""),
                "two_stage_status": two_stage_status,
                **shadow,
                "F1_ILP": f1_ilp,
                "F1": obj["F1"], "F2": obj["F2"], "F3": obj["F3"], "F4": obj["F4"],
                "PoF": pof, "HV": hv, "front_size": front_size,
                "exact_front": exact,
                "arm": int(np.argmax(arm_counts)) if arm_counts.sum() else -1,
                **{f"arm_{rate}": int(c) for rate, c in zip(MUTATION_RATES, arm_counts)},
                "high_cpra_matched": sum(1 for p in high if p.pair_id in matched),
                "match_rate_high_cpra": rm["match_rate_high_cpra"],
                "match_rate_low_cpra": rm["match_rate_low_cpra"],
                "match_rate_distant": rm["match_rate_distant"],
                "match_rate_central": rm["match_rate_central"],
                "altruists_used": rm["altruists_used"],
                "seconds": time.time() - t0,
            })

        stats = evo.evolve_round(matched_pair_ids=list(matched))
        prev_exits = stats["competing_exits"]
        row.update({
            "matched": stats["matched"],
            "withdrawn": stats["withdrawn"],
            "withdrawn_high_cpra": stats.get("withdrawn_high_cpra", 0),
            "competing_exits": stats["competing_exits"],
            "exits_high_cpra": stats.get("exits_high_cpra", 0),
            "arrivals": stats["arrivals"],
        })
        rows.append(row)

        if verbose:
            if row["cleared"]:
                pof_txt = "  n/a" if np.isnan(row["PoF"]) else f"{row['PoF']:5.1f}"
                print(f"  r={r:<3} patients={row['n_patients']:>3} "
                      f"F1={row['F1']:>3.0f}/{row['F1_ILP']:<3.0f} "
                      f"PoF={pof_txt}%  F3={row['F3']:.3f}  "
                      f"hiCPRA={row['high_cpra_matched']}/{row['n_high_cpra']}")
            else:
                print(f"  r={r:<3} patients={row['n_patients']:>3}  (not cleared)")
    return rows


# ============================================================
# LAMBDA CALIBRATION (pool 60-90, ~50-66 transplants / year)
# ============================================================

def calibrate_lambda(lambdas=(6, 9, 12, 15), taus=(6, 12, 18),
                     seeds=(1, 2, 3), n_rounds: int = 24,
                     burn_in: int = 6) -> List[Dict]:
    """
    Runs the myopic ILP (fast) for each lambda and tau and reports
    the steady-state number of waiting patients and transplants per
    year. ANZKX targets: 60-90 registered pairs, ~50-66 transplants
    per year (concept note, refs [5][6]).
    """
    cfg = {"name": "calib", "method": "ilp", "clearing": "myopic"}
    out = []
    for lam in lambdas:
        for tau in taus:
            pool_sizes, tx = [], []
            for s in seeds:
                rows = simulate_trajectory(cfg, tau, s, n_rounds=n_rounds,
                                           lambda_arrivals=lam)
                after = [x for x in rows if x["round"] > burn_in]
                pool_sizes += [x["n_patients"] for x in after]
                tx.append(sum(x.get("F1", 0) for x in after) / len(after) * 12)
            out.append({"lambda": lam, "tau": tau,
                        "patients_mean": float(np.mean(pool_sizes)),
                        "patients_min": int(np.min(pool_sizes)),
                        "patients_max": int(np.max(pool_sizes)),
                        "transplants_per_year": float(np.mean(tx))})
    return out


# ============================================================
# TEST MODULE 7
# ============================================================

if __name__ == "__main__":

    print("=" * 66)
    print("MODULE 7: MYOPIC vs BATCHED CLEARING (ILP-guided NSGA-III)")
    print("=" * 66)

    TEST_ROUNDS, TEST_GEN, TAU = 12, 20, 12
    tests = [("myopic", None), ("batched", None), ("batched", 3)]
    for clearing, interval in tests:
        cfg = {"name": f"dyn_nsga3_ucb1_{clearing}", "method": "nsga3",
               "aos": "ucb1", "clearing": clearing, "dynamic": True}
        label = cfg["name"] + (f" (max interval {interval})" if interval else "")
        print(f"\n--- {label}: tau={TAU}, {TEST_ROUNDS} rounds, n_gen={TEST_GEN} ---")
        rows = simulate_trajectory(cfg, tau=TAU, run_seed=1, n_rounds=TEST_ROUNDS,
                                   n_gen=TEST_GEN, batch_max_interval=interval,
                                   verbose=True)
        c = [x for x in rows if x["cleared"]]
        if c:
            print(f"  Rounds cleared: {len(c)}   transplants: {sum(x['F1'] for x in c):.0f}"
                  f"   high-cPRA transplanted: {sum(x['high_cpra_matched'] for x in c)}"
                  f"   mean PoF: {np.nanmean([x['PoF'] for x in c]):.2f}%")
        else:
            print("  Never cleared: the pool never reached n* patients.")

    print("\n--- Paired design: identical arrivals across configurations ---")
    a = simulate_trajectory({"name": "ilp", "method": "ilp", "clearing": "myopic"},
                            12, 7, n_rounds=6)
    b = simulate_trajectory({"name": "lex", "method": "lex", "clearing": "myopic"},
                            12, 7, n_rounds=6)
    print(f"  Same arrivals: {[x['arrivals'] for x in a] == [y['arrivals'] for y in b]}")
    print(f"  Same F1_ILP in round 1: {a[0]['F1_ILP'] == b[0]['F1_ILP']}")

    print("\n--- Lambda calibration (myopic ILP, tau=12, 2 seeds, 18 rounds) ---")
    print("  ANZKX targets: 60-90 patients waiting, ~50-66 transplants/year")
    print(f"  {'lambda':>6} {'patients mean':>14} {'min-max':>9} {'transplants/yr':>15}")
    for c in calibrate_lambda(lambdas=(6, 10, 14), taus=(12,), seeds=(1, 2),
                              n_rounds=18):
        print(f"  {c['lambda']:>6} {c['patients_mean']:>14.1f} "
              f"{c['patients_min']:>4}-{c['patients_max']:<4} "
              f"{c['transplants_per_year']:>15.1f}")

    print("\n" + "=" * 66)
    print("MODULE 7 COMPLETE")
    print("Ready for Module 8: Statistics")
    print("=" * 66)
