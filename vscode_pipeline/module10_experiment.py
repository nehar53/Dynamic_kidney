import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from module7_clearing import BATCH_THRESHOLD, LAMBDA_ARRIVALS, simulate_trajectory
from module8_stats import compare_many_metrics, format_mean_sd, summary_table

# ============================================================
# MODULE 10: FULL EXPERIMENT RUNNER
# 8 baselines x 3 tau x 100 independent runs x T rounds
# Paper 2 — Dynamic KEP Implementation (Section 4.4 - 4.5)
# ============================================================
#
# Usage (from the folder that contains module1..module10):
#   python module10_experiment.py --quick              # 2-3 min test
#   python module10_experiment.py --runs 10 --jobs 8   # pilot
#   python module10_experiment.py --jobs 8             # full design
#   python module10_experiment.py --analyse-only       # stats+plots
#
# Resumable: each trajectory is saved to results/raw/ as soon as
# it finishes. Re-running the same command skips finished ones.

BASE_SEED = 42

BASELINES: Dict[str, Dict] = {
    # 1. Paper 1 framework, applied to each round independently
    "static_nsga2_ucb1": {"method": "nsga2", "aos": "ucb1",
                          "clearing": "myopic", "dynamic": False},
    # 2. NSGA-III, fixed mutation rate 0.10, no AOS
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
    # 7. Volume-maximising ILP (no fairness)
    "myopic_ilp": {"method": "ilp", "clearing": "myopic"},
    # 8. Dickerson et al. (2014) lexicographic fairness
    "dickerson_lex": {"method": "lex", "clearing": "myopic"},
}
for _name, _cfg in BASELINES.items():
    _cfg["name"] = _name

# Pre-defined comparisons (Bonferroni over this list, per metric/tau)
COMPARISONS = [
    # Central question: myopic vs batched (same AOS)
    ("dyn_nsga3_ucb1_myopic", "dyn_nsga3_ucb1_batched"),
    ("dyn_nsga3_linucb_myopic", "dyn_nsga3_linucb_batched"),
    # Supporting question: UCB1 vs LinUCB (same clearing)
    ("dyn_nsga3_ucb1_myopic", "dyn_nsga3_linucb_myopic"),
    ("dyn_nsga3_ucb1_batched", "dyn_nsga3_linucb_batched"),
    # Dynamic vs static framework
    ("static_nsga2_ucb1", "dyn_nsga3_ucb1_myopic"),
    ("static_nsga3_fixed", "dyn_nsga3_ucb1_myopic"),
    # Fairness-aware MOEA vs fairness baseline
    ("dyn_nsga3_ucb1_myopic", "dickerson_lex"),
]

RUN_METRICS = [
    "pof_mean", "pof_var", "total_transplants",
    "total_high_cpra_transplants", "mean_F2", "mean_F3", "mean_F4",
    "mean_HV", "high_cpra_accumulation", "total_withdrawn",
    "total_competing_exits", "rounds_cleared",
]


# ============================================================
# ONE TASK = ONE TRAJECTORY
# ============================================================

def raw_path(out_dir: Path, name: str, tau: int, run: int) -> Path:
    return out_dir / "raw" / f"{name}__tau{tau}__run{run:03d}.csv"


def run_task(task: Dict) -> str:
    """Worker: simulate one trajectory and save it to CSV."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cfg = BASELINES[task["config"]]
    rows = simulate_trajectory(
        cfg, tau=task["tau"], run_seed=BASE_SEED + task["run"],
        n_rounds=task["rounds"], n_gen=task["gen"],
        selection_rule=task["selection"],
        min_f1_ratio=task["min_f1_ratio"],
        batch_max_interval=task["batch_max_interval"],
        initial_wait_max=task["initial_wait_max"],
    )
    df = pd.DataFrame(rows)
    df.insert(2, "run", task["run"])
    tmp = Path(task["path"]).with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(task["path"])                # atomic: no half files
    return task["path"]


def run_all(args) -> None:
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    tasks = []
    for tau in args.taus:
        for name in args.configs:
            for run in range(args.runs):
                p = raw_path(out, name, tau, run)
                if not p.exists():
                    tasks.append({
                        "config": name, "tau": tau, "run": run,
                        "rounds": args.rounds, "gen": args.gen,
                        "selection": args.selection,
                        "min_f1_ratio": args.min_f1_ratio,
                        "batch_max_interval": args.batch_max_interval,
                        "initial_wait_max": args.initial_wait_max,
                        "path": str(p)})
    total = len(args.taus) * len(args.configs) * args.runs
    print(f"Trajectories: {total - len(tasks)}/{total} already done, "
          f"{len(tasks)} to run on {args.jobs} process(es)")

    t0 = time.time()
    if args.jobs > 1 and tasks:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = {ex.submit(run_task, t): t for t in tasks}
            for k, f in enumerate(as_completed(futures), 1):
                f.result()
                t = futures[f]
                print(f"[{k}/{len(tasks)}] {t['config']} tau={t['tau']} "
                      f"run={t['run']}  ({(time.time() - t0) / 60:.1f} min)",
                      flush=True)
    else:
        for k, t in enumerate(tasks, 1):
            run_task(t)
            print(f"[{k}/{len(tasks)}] {t['config']} tau={t['tau']} "
                  f"run={t['run']}  ({(time.time() - t0) / 60:.1f} min)",
                  flush=True)


# ============================================================
# COLLECT + PER-RUN METRICS
# ============================================================

def collect(args) -> pd.DataFrame:
    out = Path(args.out)
    frames = []
    for tau in args.taus:
        for name in args.configs:
            for run in range(args.runs):
                p = raw_path(out, name, tau, run)
                if p.exists():
                    frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit("No results found - run the experiment first.")
    rounds = pd.concat(frames, ignore_index=True)
    rounds.to_csv(out / "rounds.csv", index=False)
    return rounds


def per_run_metrics(rounds: pd.DataFrame) -> pd.DataFrame:
    """Trajectory metrics of Section 4.5.2, one row per run."""
    rows = []
    for (cfg, tau, run), g in rounds.groupby(["config", "tau", "run"]):
        g = g.sort_values("round")
        c = g[g["cleared"]]
        slope = (np.polyfit(g["round"], g["n_high_cpra"], 1)[0]
                 if len(g) > 1 else 0.0)
        rows.append({
            "config": cfg, "tau": tau, "run": run,
            "run_seed": g["run_seed"].iloc[0],
            "rounds_cleared": len(c),
            "pof_mean": c["PoF"].mean() if len(c) else np.nan,
            "pof_var": c["PoF"].var(ddof=0) if len(c) else np.nan,
            "total_transplants": c["F1"].sum() if len(c) else 0,
            "total_high_cpra_transplants":
                c["high_cpra_matched"].sum() if len(c) else 0,
            "mean_F2": c["F2"].mean() if len(c) else np.nan,
            "mean_F3": c["F3"].mean() if len(c) else np.nan,
            "mean_F4": c["F4"].mean() if len(c) else np.nan,
            "mean_HV": c["HV"].mean() if len(c) else np.nan,
            "high_cpra_accumulation": slope,
            "total_withdrawn": g["withdrawn"].sum(),
            "total_competing_exits": g["competing_exits"].sum(),
            "mean_pool_size": g["pool_size"].mean(),
        })
    return pd.DataFrame(rows)


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


def make_figures(rounds: pd.DataFrame, out: Path) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    paths = []
    taus = sorted(rounds["tau"].unique())

    for metric, label in [("PoF", "Price of Fairness PoF$_t$ (%)"),
                          ("F3", "High-cPRA gap F3"),
                          ("F4", "Logistics disparity F4 (proxy)"),
                          ("HV", "Hypervolume HV$_t$"),
                          ("n_high_cpra", "High-cPRA patients in pool"),
                          ("pool_size", "Pool size")]:
        fig, axes = plt.subplots(1, len(taus), sharey=True,
                                 figsize=(5 * len(taus), 3.8),
                                 squeeze=False)
        for ax, tau in zip(axes[0], taus):
            sub = rounds[rounds["tau"] == tau]
            if metric not in ("n_high_cpra", "pool_size"):
                sub = sub[sub["cleared"]]
            for cfg, color in COLORS.items():
                g = sub[sub["config"] == cfg].groupby("round")[metric]
                if g.ngroups == 0 or g.mean().isna().all():
                    continue
                m, se = g.mean(), g.sem().fillna(0)
                ax.plot(m.index, m.values, color=color, lw=1.6,
                        label=cfg, marker="o", ms=2.5)
                ax.fill_between(m.index, m - 1.96 * se, m + 1.96 * se,
                                color=color, alpha=0.15, lw=0)
            ax.set_title(f"$\\tau$ = {tau} months")
            ax.set_xlabel("Clearing round t")
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel(label)
        handles, labels = {}, []
        for ax in axes[0]:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(l, h)
        labels = [c for c in COLORS if c in handles]
        fig.legend([handles[l] for l in labels], labels,
                   loc="lower center", ncol=4, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        p = fig_dir / f"trajectory_{metric}.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        paths.append(p)

    # AOS arm selection per round (UCB1 vs LinUCB)
    arm_rows = rounds[rounds["cleared"] & (rounds["arm"] >= 0)
                      & rounds["config"].str.startswith("dyn_")]
    if len(arm_rows):
        from module5_nsga3_aos import MUTATION_RATES
        cfgs = sorted(arm_rows["config"].unique())
        fig, axes = plt.subplots(1, len(cfgs), sharey=True,
                                 figsize=(4.2 * len(cfgs), 3.4),
                                 squeeze=False)
        for ax, cfg in zip(axes[0], cfgs):
            sub = arm_rows[arm_rows["config"] == cfg]
            share = (pd.crosstab(sub["round"], sub["arm"],
                                 normalize="index")
                     .reindex(columns=range(len(MUTATION_RATES)),
                              fill_value=0))
            ax.stackplot(share.index, share.T.values,
                         labels=[f"p_m={r}" for r in MUTATION_RATES])
            ax.set_title(cfg, fontsize=9)
            ax.set_xlabel("Clearing round t")
            ax.set_ylim(0, 1)
        axes[0][0].set_ylabel("Share of runs choosing arm")
        axes[0][-1].legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        p = fig_dir / "aos_arm_selection.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        paths.append(p)
    return paths


# ============================================================
# ANALYSIS
# ============================================================

def analyse(args) -> None:
    out = Path(args.out)
    rounds = collect(args)
    runs = per_run_metrics(rounds)
    runs.to_csv(out / "runs.csv", index=False)

    table = summary_table(runs, RUN_METRICS)
    for m in ("pof_mean", "total_transplants",
              "total_high_cpra_transplants", "mean_F3"):
        table[f"{m} (mean ± sd)"] = format_mean_sd(table, m)
    table.to_csv(out / "table_main_results.csv", index=False)

    comps = [c for c in COMPARISONS
             if c[0] in args.configs and c[1] in args.configs]
    stats = compare_many_metrics(runs, RUN_METRICS, comps,
                                 group_col="tau")
    stats.to_csv(out / "table_wilcoxon.csv", index=False)

    # Central question: per-round PoF, myopic vs batched
    dyn = rounds[rounds["cleared"]
                 & rounds["config"].str.startswith("dyn_nsga3_")].copy()
    if len(dyn):
        dyn["aos"] = dyn["config"].str.split("_").str[2]
        dyn["clearing"] = dyn["config"].str.split("_").str[3]
        (dyn.groupby(["tau", "aos", "clearing", "round"])["PoF"]
            .agg(["mean", "std", "count"]).reset_index()
            .to_csv(out / "table_pof_trajectory.csv", index=False))

    figs = make_figures(rounds, out)

    show = ["config", "tau", "pof_mean (mean ± sd)",
            "total_transplants (mean ± sd)",
            "total_high_cpra_transplants (mean ± sd)",
            "mean_F3 (mean ± sd)"]
    print("\n=== MAIN RESULTS (mean ± sd over runs) ===")
    print(table[show].to_string(index=False))
    if len(stats):
        key = stats[stats["metric"].isin(["pof_mean",
                                          "total_transplants"])]
        print("\n=== PAIRED WILCOXON (Bonferroni) ===")
        print(key[["tau", "metric", "config_a", "config_b", "mean_a",
                   "mean_b", "p_bonferroni", "significant",
                   "cohens_d", "effect"]].round(4).to_string(index=False))
    print(f"\nTables:  {out}/table_main_results.csv, "
          f"table_wilcoxon.csv, table_pof_trajectory.csv")
    print(f"Raw:     {out}/rounds.csv, runs.csv")
    print(f"Figures: {', '.join(p.name for p in figs)}")


def preflight(args) -> None:
    """Warn about settings that make a configuration meaningless."""
    steady = LAMBDA_ARRIVALS * min(args.taus)
    batched = any("batched" in c for c in args.configs)
    if batched and args.batch_max_interval is None and \
            steady < BATCH_THRESHOLD:
        print(f"WARNING: lambda={LAMBDA_ARRIVALS}/round x tau="
              f"{min(args.taus)} gives at most ~{steady:.0f} pairs, "
              f"below n*={BATCH_THRESHOLD}. Batched clearing may never "
              f"fire at this tau. Consider --batch-max-interval 3.\n")


# ============================================================
# MAIN
# ============================================================

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Paper 2 full experiment")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--gen", type=int, default=150)
    ap.add_argument("--taus", type=int, nargs="+", default=[6, 12, 18])
    ap.add_argument("--configs", nargs="+", default=list(BASELINES),
                    choices=list(BASELINES))
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel processes (number of CPU cores)")
    ap.add_argument("--selection", default="floor",
                    choices=["floor", "compromise", "max_f1", "min_f3"])
    ap.add_argument("--min-f1-ratio", type=float, default=0.9,
                    help="efficiency floor for --selection floor")
    ap.add_argument("--batch-max-interval", type=int, default=None)
    ap.add_argument("--initial-wait-max", type=int, default=6,
                    help="spread initial waiting times over 0..k-1 "
                         "(0 = module 1 behaviour, all start at 0)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true",
                    help="tiny budget to test the whole pipeline")
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args(argv)
    if args.quick:
        args.runs, args.rounds, args.gen = 3, 6, 10
        args.taus = [6, 12]
        args.out = args.out if args.out != "results" else "results_quick"
    return args


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("MODULE 10: FULL EXPERIMENT")
    print(f"configs={len(args.configs)}  taus={args.taus}  "
          f"runs={args.runs}  rounds={args.rounds}  n_gen={args.gen}")
    print(f"selection={args.selection} (min_f1_ratio="
          f"{args.min_f1_ratio})  output={args.out}/")
    print("=" * 60)
    preflight(args)
    if not args.analyse_only:
        run_all(args)
    analyse(args)
    print("\n" + "=" * 60)
    print("MODULE 10 COMPLETE")
    print("=" * 60)
