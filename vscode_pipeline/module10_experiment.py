import os

# One BLAS thread per process: the experiment is parallel over
# trajectories (--jobs), not inside numpy. Must be set before numpy.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import warnings
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from module1_generator import generate_initial_pool
from module2_pool_evolution import LAMBDA_ARRIVALS, MU_D, PoolEvolution
from module3_logistics import CIT_MAX_HOURS, build_compatibility_graph
from module5_nsga3_aos import MUTATION_RATES, N_GENERATIONS, N_POPULATION
from module7_clearing import (BATCH_THRESHOLD, INITIAL_POOL_SIZE,
                              INITIAL_WAIT_MAX, MIN_F1_RATIO, N_ROUNDS,
                              SELECTION_RULE, simulate_trajectory)
from module8_stats import (compare_many_metrics, format_mean_sd,
                           friedman_test, pof_trend, summary_table,
                           to_latex_table)

# ============================================================
# MODULE 10: EXPERIMENT RUNNER AND ANALYSIS
# Concept note Section 4 (Experimental Setup)
# Paper 2 — Dynamic KEP Implementation
# ============================================================
#
# Experiments (--experiment):
#   main         8 baselines x tau {6, 12, 18} x 100 runs x 25 rounds
#                N = 100, G = 150                    (Sections 4.1-4.5)
#   sens_select  implemented solution: 90% floor vs 80% floor vs
#                compromise                          (Section 3.6.1)
#   sens_cit     CIT limit 24 h vs 16 h vs 12 h       (Table 1)
#   sens_batch   batched clearing with a 3-round maximum interval
#                (n* = 90 is never reached at tau = 6) (Section 3.7)
#   sens_mud     competing exit rate mu_d 0.01 / 0.02 / 0.04, and
#                high-cPRA patients leaving at 0.3 mu_d (Table 1)
#   scalability  initial pools of 200, 500, 1000 vertices (Table 1)
#   all          main + every sensitivity experiment
#
# A "variant" is one setting of the simulation (base = Table 1).
# Every result file is   raw/{variant}__{config}__tau{tau}__run{r}.csv
# so the base runs of `main` are reused by the sensitivity
# experiments (same output folder, same budget).
#
# Pairing: run r uses seed BASE_SEED + r in EVERY configuration and
# variant -> same initial pool, arrivals and competing-exit random
# numbers (module 2) -> paired Wilcoxon tests (module 8).
#
# Usage (folder that contains module1..module10):
#   python module10_experiment.py --quick               # ~5 min test
#   python module10_experiment.py --pilot --jobs 4      # 5 runs, timing
#   python module10_experiment.py --jobs 4              # main design
#   python module10_experiment.py --experiment sens_cit --jobs 4
#   python module10_experiment.py --analyse-only        # tables+figures
#   python module10_experiment.py --shard 1/3 --jobs 8  # split over PCs
#
# Resumable: every trajectory is written as soon as it finishes.
# Stop with Ctrl+C at any time and re-run the same command.

BASE_SEED = 42

# ============================================================
# THE EIGHT BASELINES (Section 4.4)
# ============================================================

BASELINES: Dict[str, Dict] = {
    # 1. Paper 1 framework applied to each round independently
    "static_nsga2_ucb1": {"method": "nsga2", "aos": "ucb1",
                          "clearing": "myopic", "dynamic": False},
    # 2. NSGA-III with fixed p_m = 0.10, no AOS
    "static_nsga3_fixed": {"method": "nsga3", "aos": "none",
                           "clearing": "myopic", "dynamic": False},
    # 3-6. Proposed dynamic framework
    "dyn_nsga3_ucb1_myopic": {"method": "nsga3", "aos": "ucb1",
                              "clearing": "myopic", "dynamic": True},
    "dyn_nsga3_linucb_myopic": {"method": "nsga3", "aos": "linucb",
                                "clearing": "myopic", "dynamic": True},
    "dyn_nsga3_ucb1_batched": {"method": "nsga3", "aos": "ucb1",
                               "clearing": "batched", "dynamic": True},
    "dyn_nsga3_linucb_batched": {"method": "nsga3", "aos": "linucb",
                                 "clearing": "batched", "dynamic": True},
    # 7. Volume-maximising ILP (Constantino et al. 2013)
    "myopic_ilp": {"method": "ilp", "clearing": "myopic"},
    # 8. Dickerson et al. (2014): high-cPRA first, then volume
    "dickerson_lex": {"method": "lex", "clearing": "myopic"},
}
for _name, _cfg in BASELINES.items():
    _cfg["name"] = _name

MOEA_CONFIGS = [c for c, v in BASELINES.items() if v["method"] in ("nsga2", "nsga3")]
SHORT = {
    "static_nsga2_ucb1": "S-NSGA-II+UCB1",
    "static_nsga3_fixed": "S-NSGA-III fixed",
    "dyn_nsga3_ucb1_myopic": "D-UCB1 myopic",
    "dyn_nsga3_linucb_myopic": "D-LinUCB myopic",
    "dyn_nsga3_ucb1_batched": "D-UCB1 batched",
    "dyn_nsga3_linucb_batched": "D-LinUCB batched",
    "myopic_ilp": "Myopic ILP",
    "dickerson_lex": "Dickerson lex",
}

# Pre-registered pairwise comparisons. Bonferroni is applied over
# this list, separately for every metric and tau.
COMPARISONS: List[Tuple[str, str]] = [
    # Central question: myopic vs batched clearing (same AOS)
    ("dyn_nsga3_ucb1_myopic", "dyn_nsga3_ucb1_batched"),
    ("dyn_nsga3_linucb_myopic", "dyn_nsga3_linucb_batched"),
    # Supporting question: stateless vs contextual AOS
    ("dyn_nsga3_ucb1_myopic", "dyn_nsga3_linucb_myopic"),
    ("dyn_nsga3_ucb1_batched", "dyn_nsga3_linucb_batched"),
    # Dynamic framework vs static baselines
    ("dyn_nsga3_ucb1_myopic", "static_nsga2_ucb1"),
    ("dyn_nsga3_ucb1_myopic", "static_nsga3_fixed"),
    ("dyn_nsga3_linucb_myopic", "static_nsga3_fixed"),
    # Multi-objective framework vs the two ILP baselines
    ("dyn_nsga3_ucb1_myopic", "myopic_ilp"),
    ("dyn_nsga3_ucb1_myopic", "dickerson_lex"),
]

# ============================================================
# VARIANTS (one setting of the simulator) AND EXPERIMENTS
# ============================================================
# Keys a variant may override (defaults = Table 1 / module 7):
#   selection_rule, min_f1_ratio, cit_max, batch_max_interval,
#   mu_d, mu_d_high_mult, n_pairs, lambda_arrivals, batch_threshold

VARIANTS: Dict[str, Dict] = {
    "base":       {},
    # implemented solution (Section 3.6.1)
    "floor80":    {"min_f1_ratio": 0.8},
    "compromise": {"selection_rule": "compromise"},
    # logistics (Table 1: 24 h, sensitivity 16 h and 12 h)
    "cit16":      {"cit_max": 16},
    "cit12":      {"cit_max": 12},
    # batched clearing with a maximum waiting interval
    "maxint3":    {"batch_max_interval": 3},
    # competing exits (Table 1)
    "mud001":     {"mu_d": 0.01},
    "mud004":     {"mu_d": 0.04},
    "mudhi03":    {"mu_d_high_mult": 0.3},
    # scalability: arrivals and n* scaled with the initial pool
    "n200":       {"n_pairs": 200},
    "n500":       {"n_pairs": 500},
    "n1000":      {"n_pairs": 1000},
}

# Exhaustive cycle enumeration (K = 3) and the joint cycle/chain ILP
# grow quickly with the pool; above this size a variant needs
# --allow-large (see the scalability note in the README).
MAX_SAFE_POOL = 300

DYN_CORE = ["dyn_nsga3_ucb1_myopic", "dyn_nsga3_ucb1_batched",
            "dyn_nsga3_linucb_myopic", "dyn_nsga3_linucb_batched"]

EXPERIMENTS: Dict[str, Dict] = {
    "main": {
        "variants": ["base"], "configs": list(BASELINES),
        "taus": [6, 12, 18], "runs": 100,
        "about": "8 baselines x 3 tau (Sections 4.4-4.5)"},
    "sens_select": {
        "variants": ["base", "floor80", "compromise"],
        "configs": ["dyn_nsga3_ucb1_myopic", "dyn_nsga3_ucb1_batched"],
        "taus": [12], "runs": 30,
        "about": "which Pareto solution is implemented"},
    "sens_cit": {
        "variants": ["base", "cit16", "cit12"],
        "configs": ["dyn_nsga3_ucb1_myopic", "dyn_nsga3_ucb1_batched",
                    "myopic_ilp", "dickerson_lex"],
        "taus": [12], "runs": 30,
        "about": "CIT limit 24 / 16 / 12 hours"},
    "sens_batch": {
        "variants": ["base", "maxint3"],
        "configs": ["dyn_nsga3_ucb1_batched", "dyn_nsga3_linucb_batched"],
        "taus": [6, 12, 18], "runs": 30,
        "about": "batched clearing with a 3-round maximum interval"},
    "sens_mud": {
        "variants": ["base", "mud001", "mud004", "mudhi03"],
        "configs": ["dyn_nsga3_ucb1_myopic", "dyn_nsga3_ucb1_batched",
                    "myopic_ilp"],
        "taus": [12], "runs": 30,
        "about": "competing exit rate mu_d"},
    "scalability": {
        "variants": ["n200", "n500", "n1000"],
        "configs": ["dyn_nsga3_ucb1_myopic", "myopic_ilp", "dickerson_lex"],
        "taus": [12], "runs": 10, "rounds": 10,
        "about": "initial pools of 200 / 500 / 1000 vertices"},
}
SENSITIVITY = ["sens_select", "sens_cit", "sens_batch", "sens_mud"]

# Rough seconds per NSGA-III clearing round at N = 100, G = 150 for a
# 50-70 patient pool on one core (measured ~6-15 s after the module 5
# speed-up; small rounds are solved exactly in < 1 s). Planning only.
SEC_PER_MOEA_ROUND = 12.0

# Parallel processes when --jobs is not given: all CPU cores but one
DEFAULT_JOBS = max(1, (os.cpu_count() or 2) - 1)
SEC_PER_ILP_ROUND = 0.1


def variant_settings(variant: str) -> Dict:
    """Full simulation settings of a variant (Table 1 defaults)."""
    s = {"selection_rule": SELECTION_RULE, "min_f1_ratio": MIN_F1_RATIO,
         "cit_max": CIT_MAX_HOURS, "batch_max_interval": None,
         "mu_d": MU_D, "mu_d_high_mult": 1.0,
         "n_pairs": INITIAL_POOL_SIZE, "lambda_arrivals": LAMBDA_ARRIVALS,
         "batch_threshold": BATCH_THRESHOLD}
    s.update(VARIANTS[variant])
    scale = s["n_pairs"] / INITIAL_POOL_SIZE
    if scale != 1.0:                       # scalability: keep the ratios
        s["lambda_arrivals"] = LAMBDA_ARRIVALS * scale
        s["batch_threshold"] = int(round(BATCH_THRESHOLD * scale))
    return s


def label(config: str, variant: str) -> str:
    """Configuration label used in the tables: config or config@variant."""
    return config if variant == "base" else f"{config}@{variant}"


# ============================================================
# ONE TASK = ONE TRAJECTORY
# ============================================================

def raw_path(out: Path, variant: str, config: str, tau: int, run: int) -> Path:
    return out / "raw" / f"{variant}__{config}__tau{tau}__run{run:03d}.csv"


def run_task(task: Dict) -> Tuple[str, float]:
    """Worker: simulate one trajectory and save it as CSV."""
    t0 = time.time()
    s = variant_settings(task["variant"])
    cfg = BASELINES[task["config"]]
    run_seed = BASE_SEED + task["run"]

    builder = (build_compatibility_graph if s["cit_max"] == CIT_MAX_HOURS
               else partial(build_compatibility_graph, cit_max=s["cit_max"]))

    def evolution(pool, next_id, tau):
        return PoolEvolution(pool, next_id, tau_months=tau, mu_d=s["mu_d"],
                             lambda_arrivals=s["lambda_arrivals"],
                             seed=run_seed,
                             mu_d_high_cpra_multiplier=s["mu_d_high_mult"])

    initial = generate_initial_pool(n_pairs=s["n_pairs"], seed=run_seed,
                                    initial_wait_max=task["initial_wait_max"])
    rows = simulate_trajectory(
        cfg, tau=task["tau"], run_seed=run_seed,
        n_rounds=task["rounds"], n_gen=task["gen"], pop_size=task["pop"],
        n_pairs=s["n_pairs"],
        selection_rule=s["selection_rule"], min_f1_ratio=s["min_f1_ratio"],
        batch_threshold=s["batch_threshold"],
        batch_max_interval=s["batch_max_interval"],
        mu_d=s["mu_d"], lambda_arrivals=s["lambda_arrivals"],
        initial_wait_max=task["initial_wait_max"],
        graph_builder=builder, initial_pool=initial,
        evolution_factory=evolution,
    )
    df = pd.DataFrame(rows)
    df.insert(0, "variant", task["variant"])
    df.insert(4, "run", task["run"])
    path = Path(task["path"])
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)                      # atomic: never a half file
    return str(path), time.time() - t0


def plan(args) -> List[Dict]:
    """All (variant, config, tau, run) of the chosen experiments."""
    out = Path(args.out)
    seen, tasks = set(), []
    for exp in args.experiments:
        e = EXPERIMENTS[exp]
        configs = [c for c in e["configs"] if c in args.configs]
        taus = args.taus or e["taus"]
        runs = args.runs if args.runs is not None else e["runs"]
        rounds = args.rounds or e.get("rounds", N_ROUNDS)
        for run in range(runs):                 # run-major: any prefix
            for variant in e["variants"]:       # of runs is complete and
                for tau in taus:                # paired across configs
                    for config in configs:
                        key = (variant, config, tau, run)
                        if key in seen:
                            continue
                        seen.add(key)
                        tasks.append({
                            "experiment": exp, "variant": variant,
                            "config": config, "tau": tau, "run": run,
                            "rounds": rounds, "gen": args.gen,
                            "pop": args.pop,
                            "initial_wait_max": args.initial_wait_max,
                            "path": str(raw_path(out, variant, config,
                                                 tau, run))})
    large = sorted({t["variant"] for t in tasks
                    if variant_settings(t["variant"])["n_pairs"] > MAX_SAFE_POOL})
    if large and not args.allow_large:
        print(f"NOTE: skipping {', '.join(large)}: above {MAX_SAFE_POOL} "
              f"vertices exhaustive 3-cycle enumeration and the joint ILP "
              f"take minutes per round (500 vertices: ~180k cycles, "
              f"~3 min ILP). Add --allow-large to include them.")
        tasks = [t for t in tasks if t["variant"] not in large]
    if args.shard:
        i, k = args.shard
        tasks = [t for n, t in enumerate(tasks) if n % k == i - 1]
    return tasks


def is_moea(config: str) -> bool:
    return BASELINES[config]["method"] in ("nsga2", "nsga3")


def estimate_seconds(task: Dict) -> float:
    """Rough CPU time of one trajectory (for planning only)."""
    s = variant_settings(task["variant"])
    if not is_moea(task["config"]):
        return task["rounds"] * SEC_PER_ILP_ROUND * (s["n_pairs"] / 75) ** 2
    frac = 1.0
    if BASELINES[task["config"]]["clearing"] == "batched":
        frac = 0.4 if s["batch_max_interval"] is None else 0.45
    per = SEC_PER_MOEA_ROUND * (task["gen"] / 150) * (task["pop"] / 100)
    return task["rounds"] * frac * per * (s["n_pairs"] / 75) ** 1.5


def run_all(args, tasks: List[Dict]) -> None:
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks if not Path(t["path"]).exists()]
    # cheap trajectories first, then the MOEA ones (still run-major)
    todo.sort(key=lambda t: (is_moea(t["config"]), t["run"]))
    est = sum(estimate_seconds(t) for t in todo) / 3600
    print(f"Trajectories: {len(tasks) - len(todo)}/{len(tasks)} done, "
          f"{len(todo)} to run on {args.jobs} process(es)")
    if todo:
        print(f"Rough estimate: {est:.1f} CPU-hours "
              f"(~{est / max(1, args.jobs):.1f} h wall-clock with "
              f"--jobs {args.jobs}); measured rates follow below.\n")
    if not todo:
        return

    t0, done_moea, sec_moea = time.time(), 0, 0.0

    def report(k, t, secs):
        nonlocal done_moea, sec_moea
        if is_moea(t["config"]):
            done_moea += 1
            sec_moea += secs
        left = [x for x in todo[k:] if is_moea(x["config"])]
        eta = ""
        if done_moea:
            eta_h = len(left) * (sec_moea / done_moea) / args.jobs / 3600
            eta = f"  ETA ~{eta_h:.1f} h"
        print(f"[{k}/{len(todo)}] {t['variant']:<10} {t['config']:<25} "
              f"tau={t['tau']:<2} run={t['run']:<3} {secs:6.0f}s  "
              f"elapsed {(time.time() - t0) / 60:.1f} min{eta}", flush=True)

    if args.jobs > 1:
        # "spawn" = the only start method on Windows; also safe on Linux
        with ProcessPoolExecutor(max_workers=args.jobs,
                                 mp_context=get_context("spawn")) as ex:
            futures = {ex.submit(run_task, t): t for t in todo}
            for k, f in enumerate(as_completed(futures), 1):
                _, secs = f.result()
                report(k, futures[f], secs)
    else:
        for k, t in enumerate(todo, 1):
            _, secs = run_task(t)
            report(k, t, secs)


# ============================================================
# COLLECT AND PER-RUN (TRAJECTORY) METRICS — Section 4.5.2
# ============================================================

def collect(tasks: List[Dict]) -> pd.DataFrame:
    frames = [pd.read_csv(t["path"]) for t in tasks if Path(t["path"]).exists()]
    if not frames:
        raise SystemExit("No results found - run the experiment first.")
    rounds = pd.concat(frames, ignore_index=True)
    rounds["cleared"] = rounds["cleared"].astype(bool)
    rounds["label"] = [label(c, v) for c, v in
                       zip(rounds["config"], rounds["variant"])]
    return rounds


def _slope(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~np.isnan(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0:
        return np.nan
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def per_run_metrics(rounds: pd.DataFrame) -> pd.DataFrame:
    """One row per (variant, config, tau, run)."""
    rows = []
    keys = ["variant", "config", "tau", "run"]
    for (variant, cfg, tau, run), g in rounds.groupby(keys):
        g = g.sort_values("round")
        c = g[g["cleared"]]
        pof = c["PoF"].dropna()
        f1, hi = c["F1"].sum(), c["high_cpra_matched"].sum()
        arm_cols = [f"arm_{r}" for r in MUTATION_RATES if f"arm_{r}" in c]
        arm_total = c[arm_cols].to_numpy().sum() if arm_cols else 0
        row = {
            "variant": variant, "config": cfg, "label": label(cfg, variant),
            "tau": tau, "run": run, "run_seed": g["run_seed"].iloc[0],
            "rounds": len(g), "rounds_cleared": len(c),
            # Price of Fairness trajectory
            "pof_mean": pof.mean() if len(pof) else np.nan,
            "pof_var": pof.var(ddof=0) if len(pof) else np.nan,
            "pof_max": pof.max() if len(pof) else np.nan,
            "pof_slope": _slope(c["round"], c["PoF"]),
            # efficiency
            "total_transplants": f1,
            "total_F1_ILP": c["F1_ILP"].sum(),
            "transplants_per_round": f1 / len(g),
            # equity
            "total_high_cpra_transplants": hi,
            "high_cpra_share_transplants": hi / f1 if f1 else np.nan,
            "mean_match_rate_high_cpra": c["match_rate_high_cpra"].mean(),
            "mean_match_rate_low_cpra": c["match_rate_low_cpra"].mean(),
            "mean_F2": c["F2"].mean(), "mean_F3": c["F3"].mean(),
            "mean_F4": c["F4"].mean(), "mean_HV": c["HV"].mean(),
            "high_cpra_accumulation": _slope(g["round"], g["n_high_cpra"]),
            "final_high_cpra_in_pool": g["n_high_cpra"].iloc[-1],
            # patients lost from the programme
            "total_withdrawn": g["withdrawn"].sum(),
            "total_withdrawn_high_cpra": g["withdrawn_high_cpra"].sum(),
            "total_competing_exits": g["competing_exits"].sum(),
            "total_exits_high_cpra": g["exits_high_cpra"].sum(),
            "mean_pool_size": g["pool_size"].mean(),
            "mean_wait": g["mean_wait"].mean(),
            "mean_seconds_per_clearing": c["seconds"].mean(),
            # rounds small enough to be solved by exact enumeration
            "share_exact_rounds": (c["exact_front"].astype(bool).mean()
                                   if "exact_front" in c and len(c) else np.nan),
        }
        for col in arm_cols:
            row[f"share_{col}"] = (c[col].sum() / arm_total
                                   if arm_total else np.nan)
        rows.append(row)
    runs = pd.DataFrame(rows)

    # Trajectory-level efficiency loss against the myopic ILP on the
    # SAME pool sequence (seed, tau, variant). Unlike PoF_t this is
    # comparable between myopic and batched clearing.
    ilp = runs[runs["config"] == "myopic_ilp"].set_index(
        ["variant", "tau", "run"])["total_transplants"]
    key = pd.MultiIndex.from_frame(runs[["variant", "tau", "run"]])
    ref = ilp.reindex(key).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        runs["pof_vs_myopic_ilp"] = np.where(
            ref > 0, 100.0 * (ref - runs["total_transplants"]) / ref, np.nan)
    return runs


def complete_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Keep only runs finished for every label of their (tau) group,
    so that means and tests use the same paired runs."""
    keep = []
    for tau, g in runs.groupby("tau"):
        n_labels = g["label"].nunique()
        ok = g.groupby("run")["label"].nunique()
        ok = ok[ok == n_labels].index
        keep.append(g[g["run"].isin(ok)])
    return pd.concat(keep, ignore_index=True) if keep else runs


RUN_METRICS = [
    "pof_mean", "pof_var", "pof_slope", "pof_vs_myopic_ilp",
    "total_transplants", "total_high_cpra_transplants",
    "high_cpra_share_transplants", "mean_match_rate_high_cpra",
    "mean_F2", "mean_F3", "mean_F4", "mean_HV",
    "high_cpra_accumulation", "total_withdrawn",
    "total_withdrawn_high_cpra", "total_competing_exits",
    "rounds_cleared", "mean_pool_size",
]
# Metrics tested with Wilcoxon (the rest are descriptive)
TEST_METRICS = [
    "pof_mean", "pof_var", "pof_vs_myopic_ilp", "total_transplants",
    "total_high_cpra_transplants", "mean_match_rate_high_cpra",
    "mean_F2", "mean_F3", "mean_F4", "mean_HV",
    "high_cpra_accumulation", "total_withdrawn_high_cpra",
]
# Columns of the main paper table
PAPER_COLUMNS = [
    ("pof_mean", "PoF (%)", 2),
    ("pof_var", "Var PoF", 2),
    ("total_transplants", "Transplants", 1),
    ("total_high_cpra_transplants", "High-cPRA tx", 1),
    ("mean_F2", "F2 Gini", 3),
    ("mean_F3", "F3 gap", 3),
    ("mean_F4", "F4", 3),
    ("mean_HV", "HV", 3),
    ("total_withdrawn_high_cpra", "High-cPRA lost", 1),
]


# ============================================================
# FIGURES
# ============================================================

COLORS = {
    "dyn_nsga3_ucb1_myopic": "#2a6fdb",
    "dyn_nsga3_linucb_myopic": "#e0702b",
    "dyn_nsga3_ucb1_batched": "#2a9d8f",
    "dyn_nsga3_linucb_batched": "#c0392b",
    "static_nsga2_ucb1": "#7f8c8d",
    "static_nsga3_fixed": "#8e6cc2",
    "dickerson_lex": "#b8860b",
    "myopic_ilp": "#333333",
}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    return plt


def _band(ax, sub, metric, color, name, cumulative=False):
    """Mean over runs with a 95% CI band, per round."""
    if cumulative:
        sub = sub.sort_values("round").copy()
        sub[metric] = sub.groupby("run")[metric].cumsum()
    g = sub.groupby("round")[metric]
    if g.ngroups == 0 or g.mean().isna().all():
        return
    m, se = g.mean(), g.sem().fillna(0)
    lo = m - 1.96 * se
    if (sub[metric].dropna() >= 0).all():   # counts, PoF, F2-F4, HV
        lo = lo.clip(lower=0)
    ax.plot(m.index, m.values, color=color, lw=1.5, label=name,
            marker="o", ms=2)
    ax.fill_between(m.index, lo, m + 1.96 * se,
                    color=color, alpha=0.15, lw=0)


def _legend(fig, axes, order):
    handles = {}
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    names = [n for n in order if n in handles]
    fig.legend([handles[n] for n in names], names, loc="lower center",
               ncol=4, fontsize=8, frameon=False)


def figures_main(rounds: pd.DataFrame, runs: pd.DataFrame,
                 fig_dir: Path) -> List[Path]:
    plt = _plt()
    paths = []
    taus = sorted(rounds["tau"].unique())
    order = [SHORT[c] for c in COLORS]

    # 1. Per-round trajectories (Section 4.5.1)
    panels = [("PoF", r"PoF$_t$ (%)", True, False),
              ("F2", r"Gini $F_2$", True, False),
              ("F3", r"High-cPRA gap $F_3$", True, False),
              ("F4", r"Logistics disparity $F_4$", True, False),
              ("HV", r"Hypervolume $HV_t$", True, False),
              ("n_high_cpra", "High-cPRA patients in pool", False, False),
              ("pool_size", "Pool size", False, False),
              ("F1", "Cumulative transplants", False, True),
              ("high_cpra_matched", "Cumulative high-cPRA transplants",
               False, True)]
    for metric, ylab, cleared_only, cum in panels:
        if metric not in rounds:
            continue
        fig, axes = plt.subplots(1, len(taus), sharey=True,
                                 figsize=(4.2 * len(taus), 3.3),
                                 squeeze=False)
        for ax, tau in zip(axes[0], taus):
            sub = rounds[rounds["tau"] == tau]
            if cum:
                sub = sub.assign(**{metric: sub[metric].fillna(0)})
            elif cleared_only:
                sub = sub[sub["cleared"]]
            for cfg, color in COLORS.items():
                _band(ax, sub[sub["config"] == cfg], metric, color,
                      SHORT[cfg], cumulative=cum)
            ax.set_title(rf"$\tau$ = {tau} months")
            ax.set_xlabel("Clearing round $t$")
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel(ylab)
        _legend(fig, axes, order)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        p = fig_dir / f"trajectory_{metric}{'_cum' if cum else ''}.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        paths.append(p)

    # 2. Clearing strategy comparison: PoF_t under pi_M vs pi_B
    pairs = [("dyn_nsga3_ucb1_myopic", "dyn_nsga3_ucb1_batched", "UCB1"),
             ("dyn_nsga3_linucb_myopic", "dyn_nsga3_linucb_batched", "LinUCB")]
    if rounds["config"].isin([p[1] for p in pairs]).any():
        fig, axes = plt.subplots(2, len(taus), figsize=(4.2 * len(taus), 5.6),
                                 squeeze=False, sharex=True)
        for j, tau in enumerate(taus):
            sub = rounds[(rounds["tau"] == tau)]
            for myo, bat, aos in pairs:
                for cfg, ls in ((myo, "-"), (bat, "--")):
                    s = sub[sub["config"] == cfg]
                    if s.empty:
                        continue
                    g = s[s["cleared"]].groupby("round")["PoF"].mean()
                    axes[0][j].plot(g.index, g.values, ls, marker="o", ms=2,
                                    color=COLORS[cfg], label=SHORT[cfg])
                    _band(axes[1][j], s.assign(F1=s["F1"].fillna(0)),
                          "F1", COLORS[cfg], SHORT[cfg], cumulative=True)
            axes[0][j].set_title(rf"$\tau$ = {tau} months")
            axes[1][j].set_xlabel("Clearing round $t$")
            for ax in axes[:, j]:
                ax.grid(alpha=0.3)
        axes[0][0].set_ylabel(r"PoF$_t$ (%)")
        axes[1][0].set_ylabel("Cumulative transplants")
        _legend(fig, axes, [SHORT[c] for p in pairs for c in p[:2]])
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        p = fig_dir / "clearing_myopic_vs_batched.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        paths.append(p)

    # 3. Mean PoF per configuration and tau (box plots over runs)
    moea = [c for c in COLORS if c in set(runs["config"]) and is_moea(c)]
    if moea:
        fig, axes = plt.subplots(1, len(taus), sharey=True,
                                 figsize=(4.2 * len(taus), 3.4),
                                 squeeze=False)
        for ax, tau in zip(axes[0], taus):
            sub = runs[(runs["tau"] == tau) & (runs["variant"] == "base")]
            data = [sub.loc[sub["config"] == c, "pof_mean"].dropna()
                    for c in moea]
            bp = ax.boxplot(data, patch_artist=True, widths=0.6)
            for patch, c in zip(bp["boxes"], moea):
                patch.set_facecolor(COLORS[c])
                patch.set_alpha(0.6)
            ax.set_xticks(range(1, len(moea) + 1))
            ax.set_xticklabels([SHORT[c] for c in moea], rotation=35,
                               ha="right", fontsize=7)
            ax.set_title(rf"$\tau$ = {tau} months")
            ax.grid(alpha=0.3, axis="y")
        axes[0][0].set_ylabel(r"$\overline{\mathrm{PoF}}$ per run (%)")
        fig.tight_layout()
        p = fig_dir / "pof_mean_boxplot.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        paths.append(p)

    # 4. AOS arm-selection distribution across rounds
    arm_cols = [f"arm_{r}" for r in MUTATION_RATES]
    aos_cfgs = [c for c in ("dyn_nsga3_ucb1_myopic", "dyn_nsga3_linucb_myopic",
                            "dyn_nsga3_ucb1_batched", "dyn_nsga3_linucb_batched",
                            "static_nsga2_ucb1")
                if c in set(rounds["config"])]
    if aos_cfgs and all(c in rounds for c in arm_cols):
        tau = 12 if 12 in taus else taus[0]
        fig, axes = plt.subplots(1, len(aos_cfgs), sharey=True,
                                 figsize=(3.2 * len(aos_cfgs), 3.0),
                                 squeeze=False)
        for ax, cfg in zip(axes[0], aos_cfgs):
            sub = rounds[(rounds["config"] == cfg) & (rounds["tau"] == tau)
                         & rounds["cleared"]]
            share = sub.groupby("round")[arm_cols].sum()
            share = share.div(share.sum(axis=1).replace(0, np.nan), axis=0)
            ax.stackplot(share.index, share.T.fillna(0).values,
                         labels=[f"$p_m$={r}" for r in MUTATION_RATES])
            ax.set_title(SHORT[cfg], fontsize=9)
            ax.set_xlabel("Clearing round $t$")
            ax.set_ylim(0, 1)
        axes[0][0].set_ylabel(rf"Share of generations ($\tau$={tau})")
        axes[0][-1].legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        p = fig_dir / "aos_arm_selection.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        paths.append(p)
    return paths


def figure_sensitivity(runs: pd.DataFrame, exp: str, fig_dir: Path) -> Path:
    """Mean ± 95% CI of the key metrics per variant and configuration."""
    plt = _plt()
    metrics = [("pof_mean", "PoF (%)"), ("total_transplants", "Transplants"),
               ("total_high_cpra_transplants", "High-cPRA transplants"),
               ("mean_F3", "F3 gap")]
    variants = EXPERIMENTS[exp]["variants"]
    configs = [c for c in EXPERIMENTS[exp]["configs"] if c in set(runs["config"])]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3.2))
    x = np.arange(len(variants))
    w = 0.8 / max(1, len(configs))
    for ax, (m, lab) in zip(axes, metrics):
        for k, cfg in enumerate(configs):
            mean, ci = [], []
            for v in variants:
                s = runs[(runs["config"] == cfg) & (runs["variant"] == v)][m].dropna()
                mean.append(s.mean() if len(s) else np.nan)
                ci.append(1.96 * s.sem() if len(s) > 1 else 0)
            ax.bar(x + k * w, mean, w, yerr=ci, color=COLORS[cfg],
                   alpha=0.8, label=SHORT[cfg], capsize=2)
        ax.set_xticks(x + w * (len(configs) - 1) / 2)
        ax.set_xticklabels(variants, rotation=20, fontsize=8)
        ax.set_title(lab)
        ax.grid(alpha=0.3, axis="y")
    _legend(fig, axes, [SHORT[c] for c in configs])
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    p = fig_dir / f"{exp}.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


# ============================================================
# ANALYSIS
# ============================================================

def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(
            f"{v:.4g}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def paper_table(runs: pd.DataFrame, by=("config", "tau")) -> pd.DataFrame:
    tab = summary_table(runs, RUN_METRICS, by=by)
    for m, name, dec in PAPER_COLUMNS:
        if f"{m}_mean" in tab:
            tab[name] = format_mean_sd(tab, m, dec)
    return tab


def analyse_main(rounds: pd.DataFrame, runs: pd.DataFrame, out: Path,
                 md: List[str]) -> None:
    base_r = rounds[rounds["variant"] == "base"]
    base = complete_runs(runs[runs["variant"] == "base"])
    if base.empty:
        return
    configs = [c for c in BASELINES if c in set(base["config"])]
    n_runs = base.groupby("tau")["run"].nunique().to_dict()

    # Table: mean ± sd over runs
    tab = paper_table(base)
    tab["order"] = tab["config"].map({c: i for i, c in enumerate(BASELINES)})
    tab = tab.sort_values(["tau", "order"]).drop(columns="order")
    tab.to_csv(out / "table_main_results.csv", index=False)
    show = ["config", "tau"] + [n for _, n, _ in PAPER_COLUMNS if n in tab]
    pretty = tab[show].assign(config=tab["config"].map(SHORT))
    (out / "table_main_results.tex").write_text(to_latex_table(
        pretty, show,
        caption=("Trajectory results over 25 clearing rounds "
                 "(mean $\\pm$ sd over paired runs)"),
        label="tab:main_results"), encoding="utf-8")

    # Friedman omnibus per metric and tau, then pairwise Wilcoxon
    fr = []
    for tau, g in base.groupby("tau"):
        for m in TEST_METRICS:
            cols = configs if m != "mean_HV" else [c for c in configs if is_moea(c)]
            r = friedman_test(g, m, cols, pair_col="run")
            fr.append({"tau": tau, "metric": m, "chi2": r["chi2"],
                       "p": r["p"], "n": r["n"],
                       "mean_ranks": json.dumps(r["mean_ranks"])})
    fr = pd.DataFrame(fr)
    fr.to_csv(out / "table_friedman.csv", index=False)

    comps = [c for c in COMPARISONS if c[0] in configs and c[1] in configs]
    stats = compare_many_metrics(base, TEST_METRICS, comps,
                                 group_col="tau", pair_col="run")
    stats.to_csv(out / "table_wilcoxon.csv", index=False)

    # PoF trend per configuration (does PoF_t drift?)
    trend = pof_trend(base_r[base_r["config"].isin(MOEA_CONFIGS)
                             & base_r["run"].isin(base["run"].unique())],
                      pair_col="run")
    trend.to_csv(out / "table_pof_trend.csv", index=False)

    # PoF_t trajectory and clearing comparison per round
    traj = (base_r[base_r["cleared"]]
            .groupby(["tau", "config", "round"])
            .agg(PoF_mean=("PoF", "mean"), PoF_sd=("PoF", "std"),
                 F1_mean=("F1", "mean"), F1_ILP_mean=("F1_ILP", "mean"),
                 F3_mean=("F3", "mean"), HV_mean=("HV", "mean"),
                 n_cleared=("PoF", "size")).reset_index())
    traj.to_csv(out / "table_pof_trajectory.csv", index=False)

    # AOS arm distribution
    arm_cols = [c for c in base if c.startswith("share_arm_")]
    if arm_cols:
        arms = (base[base["config"].isin(MOEA_CONFIGS)]
                .groupby(["config", "tau"])[arm_cols].mean().reset_index())
        arms.columns = [c.replace("share_arm_", "p_m=") for c in arms.columns]
        arms.to_csv(out / "table_arm_distribution.csv", index=False)
    else:
        arms = pd.DataFrame()

    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    figs = figures_main(base_r[base_r["run"].isin(base["run"].unique())],
                        base, fig_dir)

    # ---------- console + summary.md ----------
    print("\n=== MAIN RESULTS (mean ± sd over paired runs) ===")
    print(f"runs per tau: {n_runs}")
    print(pretty.to_string(index=False))
    md += ["## Main experiment (8 baselines)", "",
           f"Paired runs per tau: {n_runs}", "", _md_table(pretty), ""]

    if len(stats):
        key = stats[stats["metric"].isin(
            ["pof_mean", "total_transplants",
             "total_high_cpra_transplants", "mean_F3"])].copy()
        key["config_a"] = key["config_a"].map(SHORT)
        key["config_b"] = key["config_b"].map(SHORT)
        cols = ["tau", "metric", "config_a", "config_b", "mean_a", "mean_b",
                "p_bonferroni", "cohens_d", "effect", "direction"]
        print("\n=== PAIRED WILCOXON (Bonferroni per metric and tau) ===")
        print(key[cols].round(4).to_string(index=False))
        md += ["### Paired Wilcoxon signed-rank (Bonferroni)", "",
               _md_table(key[cols].round(4)), ""]
    if len(fr):
        k = fr[fr["metric"].isin(["pof_mean", "total_transplants", "mean_F3"])]
        print("\n=== FRIEDMAN OMNIBUS ===")
        print(k[["tau", "metric", "chi2", "p", "n"]].round(4).to_string(index=False))
    if len(trend):
        t = trend.assign(config=trend["config"].map(SHORT))
        print("\n=== PoF TREND (slope of PoF_t over rounds, pp/round) ===")
        print(t.round(4).to_string(index=False))
        md += ["### PoF trend over rounds", "", _md_table(t.round(4)), ""]
    if len(arms):
        print("\n=== AOS ARM DISTRIBUTION (share of generations) ===")
        print(arms.round(3).to_string(index=False))
        md += ["### AOS arm distribution", "", _md_table(arms.round(3)), ""]
    md += ["Figures: " + ", ".join(p.name for p in figs), ""]


def analyse_sensitivity(runs: pd.DataFrame, exp: str, out: Path,
                        md: List[str]) -> None:
    e = EXPERIMENTS[exp]
    sub = runs[runs["variant"].isin(e["variants"])
               & runs["config"].isin(e["configs"])
               & runs["tau"].isin(e["taus"])]
    sub = complete_runs(sub)
    if sub.empty or sub["variant"].nunique() < 2:
        return
    tab = paper_table(sub, by=("variant", "config", "tau"))
    tab.to_csv(out / f"table_{exp}.csv", index=False)

    # every non-base variant vs base for the same configuration,
    # plus myopic vs batched inside every variant
    comps = []
    ref = e["variants"][0]
    for v in e["variants"][1:]:
        for c in e["configs"]:
            comps.append((label(c, v), label(c, ref)))
    for v in e["variants"]:
        for myo, bat in (("dyn_nsga3_ucb1_myopic", "dyn_nsga3_ucb1_batched"),):
            if myo in e["configs"] and bat in e["configs"]:
                comps.append((label(myo, v), label(bat, v)))
    stats = compare_many_metrics(sub, TEST_METRICS, comps, group_col="tau",
                                 config_col="label", pair_col="run")
    stats.to_csv(out / f"table_{exp}_wilcoxon.csv", index=False)

    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig = figure_sensitivity(sub, exp, fig_dir)

    show = ["variant", "config", "tau"] + [n for _, n, _ in PAPER_COLUMNS[:4]]
    pretty = tab[show].assign(config=tab["config"].map(SHORT))
    print(f"\n=== SENSITIVITY: {exp} ({e['about']}) ===")
    print(pretty.to_string(index=False))
    md += [f"## Sensitivity: {exp} — {e['about']}", "", _md_table(pretty), ""]
    if len(stats):
        k = stats[stats["metric"].isin(["pof_mean", "total_transplants",
                                        "total_high_cpra_transplants"])]
        cols = ["tau", "metric", "config_a", "config_b", "mean_a", "mean_b",
                "p_bonferroni", "cohens_d", "direction"]
        print(k[cols].round(4).to_string(index=False))
        md += [_md_table(k[cols].round(4)), ""]
    md += [f"Figure: {fig.name}", ""]


def analyse_scalability(rounds: pd.DataFrame, out: Path, md: List[str]) -> None:
    v = EXPERIMENTS["scalability"]["variants"]
    sub = rounds[rounds["variant"].isin(v) & rounds["cleared"]]
    if sub.empty:
        return
    tab = (sub.groupby(["variant", "config"])
           .agg(pool_size=("pool_size", "mean"), edges=("n_edges", "mean"),
                cycles=("n_cycles", "mean"), chains=("n_chains", "mean"),
                truncated=("chains_truncated", "mean"),
                seconds_per_round=("seconds", "mean"),
                F1=("F1", "mean"), F1_ILP=("F1_ILP", "mean"),
                PoF=("PoF", "mean"), F3=("F3", "mean"),
                runs=("run", "nunique")).reset_index())
    tab["order"] = tab["variant"].map({x: i for i, x in enumerate(v)})
    tab = tab.sort_values(["order", "config"]).drop(columns="order")
    tab.to_csv(out / "table_scalability.csv", index=False)
    print("\n=== SCALABILITY (means per clearing round) ===")
    print(tab.round(3).to_string(index=False))
    md += ["## Scalability", "", _md_table(tab.round(3)), ""]


def analyse(args, tasks: List[Dict]) -> None:
    out = Path(args.out)
    rounds = collect(tasks)
    rounds.to_csv(out / "rounds.csv", index=False)
    runs = per_run_metrics(rounds)
    runs.to_csv(out / "runs.csv", index=False)

    md = ["# Paper 2 — experiment summary", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} from "
          f"{len(rounds)} rounds / {len(runs)} trajectories.", "",
          f"Budget: N = {args.pop}, G = {args.gen}; selection rule "
          f"{SELECTION_RULE} (F1 >= {MIN_F1_RATIO} x max).", ""]
    if "main" in args.experiments:
        analyse_main(rounds, runs, out, md)
    for exp in SENSITIVITY:
        if exp in args.experiments:
            analyse_sensitivity(runs, exp, out, md)
    if "scalability" in args.experiments:
        analyse_scalability(rounds, out, md)
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nAll tables, figures/ and summary.md are in {out}/")


# ============================================================
# PREFLIGHT, MANIFEST, ARGUMENTS
# ============================================================

def preflight(args, tasks: List[Dict]) -> None:
    """Warn about settings that make a configuration uninformative."""
    batched = [t for t in tasks if "batched" in t["config"]]
    for tau in sorted({t["tau"] for t in batched}):
        for v in sorted({t["variant"] for t in batched if t["tau"] == tau}):
            s = variant_settings(v)
            # ~ steady-state patients under myopic clearing (Section 4.1)
            steady = s["lambda_arrivals"] * min(tau, 1 / max(s["mu_d"], 1e-9)) * 0.9
            if s["batch_max_interval"] is None and steady < s["batch_threshold"]:
                print(f"NOTE: tau={tau}, variant={v}: the pool rarely reaches "
                      f"n*={s['batch_threshold']} patients; batched clearing "
                      f"may clear only a few times (reported as a finding; "
                      f"see experiment sens_batch).")


def check_manifest(args) -> None:
    """One output folder = one budget, so results stay comparable."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    budget = {"gen": args.gen, "pop": args.pop,
              "initial_wait_max": args.initial_wait_max,
              "base_seed": BASE_SEED}
    p = out / "manifest.json"
    if p.exists():
        old = json.loads(p.read_text())
        diff = {k: (old["budget"].get(k), v) for k, v in budget.items()
                if old["budget"].get(k) != v}
        if diff and not args.analyse_only:
            raise SystemExit(
                f"{out}/ holds results with a different budget {diff}. "
                f"Use another --out folder.")
        return
    import pymoo
    import scipy
    p.write_text(json.dumps({
        "budget": budget,
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "pymoo": pymoo.__version__,
        "table1": {"lambda": LAMBDA_ARRIVALS, "mu_d": MU_D,
                   "cit_max": CIT_MAX_HOURS, "n_star": BATCH_THRESHOLD,
                   "initial_pool": INITIAL_POOL_SIZE,
                   "selection": SELECTION_RULE,
                   "min_f1_ratio": MIN_F1_RATIO},
    }, indent=2))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Paper 2 experiments (module 10)")
    ap.add_argument("--experiment", nargs="+", default=["main"],
                    choices=list(EXPERIMENTS) + ["all", "sensitivity"])
    ap.add_argument("--runs", type=int, default=None,
                    help="runs per configuration (default: per experiment)")
    ap.add_argument("--rounds", type=int, default=None,
                    help=f"clearing rounds T (default {N_ROUNDS})")
    ap.add_argument("--gen", type=int, default=N_GENERATIONS)
    ap.add_argument("--pop", type=int, default=N_POPULATION)
    ap.add_argument("--taus", type=int, nargs="+", default=None)
    ap.add_argument("--configs", nargs="+", default=list(BASELINES),
                    choices=list(BASELINES))
    ap.add_argument("--initial-wait-max", type=int, default=INITIAL_WAIT_MAX)
    ap.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
                    help="parallel processes (default: all CPU cores but one)")
    ap.add_argument("--shard", default=None,
                    help="i/k: run only every k-th trajectory starting at i "
                         "(split one experiment over k computers)")
    ap.add_argument("--allow-large", action="store_true",
                    help=f"allow pools above {MAX_SAFE_POOL} vertices")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true",
                    help="tiny budget: checks the whole pipeline in minutes")
    ap.add_argument("--pilot", action="store_true",
                    help="5 runs at full budget: real timing + first look")
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args(argv)

    exps = []
    for e in args.experiment:
        exps += (list(EXPERIMENTS) if e == "all" else
                 SENSITIVITY if e == "sensitivity" else [e])
    args.experiments = list(dict.fromkeys(exps))
    if args.shard:
        i, k = (int(x) for x in args.shard.split("/"))
        if not 1 <= i <= k:
            ap.error("--shard must be i/k with 1 <= i <= k")
        args.shard = (i, k)
    if args.quick:
        args.runs = args.runs or 3
        args.rounds = args.rounds or 6
        args.gen, args.pop = 10, 40
        args.taus = args.taus or [6, 12]
        if args.out == "results":
            args.out = "results_quick"
    elif args.pilot:
        args.runs = args.runs or 5
        if args.out == "results":
            args.out = "results_pilot"
    return args


def main(argv=None) -> None:
    # identical columns (e.g. both batched configs never clearing) make
    # scipy divide by zero inside Friedman/Wilcoxon: result is NaN anyway
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args(argv)
    tasks = plan(args)
    print("=" * 66)
    print("MODULE 10: EXPERIMENTS")
    for exp in args.experiments:
        e = EXPERIMENTS[exp]
        print(f"  {exp:<12} {e['about']}  (variants: {', '.join(e['variants'])})")
    print(f"  N={args.pop}  G={args.gen}  trajectories={len(tasks)}  "
          f"output={args.out}/")
    print("=" * 66)
    check_manifest(args)
    preflight(args, tasks)
    if not args.analyse_only:
        try:
            run_all(args, tasks)
        except KeyboardInterrupt:
            print("\nStopped. Finished trajectories are saved; re-run the "
                  "same command to continue.")
            return
    analyse(args, tasks)
    print("\n" + "=" * 66)
    print("MODULE 10 COMPLETE")
    print("=" * 66)


# ============================================================
# RUN SETTINGS (used when you press Run / run with no arguments)
# ============================================================
# Which experiments to run, in order. Each is resumable: stop at any
# time (Ctrl+C or close the terminal), press Run again and it
# continues. Results so far can be analysed at any time by adding
# "--analyse-only" to RUN_ARGS.
#   ["--experiment", "main"]          8 baselines x tau 6/12/18 x 100 runs
#   ["--experiment", "sensitivity"]   selection rule, CIT, batching, mu_d
#   ["--experiment", "scalability"]   pools of 200 vertices
#   ["--experiment", "all"]           everything above
#   ["--quick"]                       1-minute test of the whole pipeline
RUN_ARGS = ["--experiment", "main"]


if __name__ == "__main__":
    main(None if len(sys.argv) > 1 else RUN_ARGS)
