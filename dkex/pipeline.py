"""Stage-wise pipeline: calibrate -> simulate -> analyse -> plot (-> ood).

Each stage reads/writes files under ``<output_dir>/<experiment name>/`` so
stages can be re-run independently:

    calibration.json          stage 1  pool/graph calibration check
    raw/*.json                stage 2  one file per (baseline, tau, run) — resumable
    rounds.csv, runs.csv      stage 2  combined per-round / per-run tables
    trajectories.csv, ...     stage 3  metrics, Wilcoxon/Bonferroni tables, summary.md
    figures/*.png             stage 4  PoF_t and other trajectories, AOS arm usage
    ood/                      stage 5  PrefLib 00036 out-of-distribution run
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from dkex.config import CENTRES, Config


def out_dir(cfg: Config) -> Path:
    p = Path(cfg.experiment.output_dir) / cfg.experiment.name
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------- 1. calibrate

def calibrate(cfg: Config, n_samples: int = 5) -> dict:
    """Check the synthetic pool against Table 2 before running experiments."""
    from dkex.experiments import run_seed
    from dkex.optimisation.ilp import solve_max_volume
    from dkex.optimisation.structures import enumerate_structures
    from dkex.simulation.compatibility import build_graph
    from dkex.simulation.dynamics import DynamicPool
    from dkex.simulation.logistics import LogisticsNetwork

    net = LogisticsNetwork(cfg.logistics)
    stats = []
    pat_abo, don_abo, centres = Counter(), Counter(), Counter()
    for k in range(n_samples):
        seed = run_seed(cfg.experiment.seed, k)
        pool = DynamicPool(cfg, seed).state
        g = build_graph(pool, net, seed, cfg.pool.hla_mismatch_p)
        S = enumerate_structures(g, cfg.structures, np.random.default_rng(seed))
        ilp = solve_max_volume(S)
        f1 = int(sum(S.sizes[ilp]))
        for p in pool.patients:
            pat_abo[p.patient_abo] += 1
            don_abo[p.donor_abo] += 1
        for p in pool.pairs.values():
            centres[CENTRES[p.centre]] += 1
        stats.append({
            "pool": len(pool), "patients": pool.n_patients(), "altruists": len(pool.altruists),
            "high_cpra_share": pool.high_cpra_share(cfg.pool.high_cpra_threshold),
            "edges": g.n_edges, "edges_pre_cit": g.n_edges_abo, "density": g.density(),
            "cycles": int((~S.is_chain).sum()), "chains": int(S.is_chain.sum()),
            "chains_truncated": bool(S.truncated), "ilp_transplants": f1,
        })
    df = pd.DataFrame(stats)
    tot = lambda c: {k: v / sum(c.values()) for k, v in sorted(c.items())}  # noqa: E731
    report = {
        "samples": stats,
        "mean": df.drop(columns=["chains_truncated"]).mean().to_dict(),
        "patient_abo": tot(pat_abo), "donor_abo": tot(don_abo), "centre_share": tot(centres),
        "cit_infeasible_centre_pairs": net.binding_fraction(),
        "logistics": net.summary(),
        "warnings": [],
    }
    if net.binding_fraction() == 0:
        report["warnings"].append(
            "CIT constraint never binds: with CIT_max=%.0fh and T_prep=%.0fh every centre pair is "
            "feasible, so F4 differences cannot arise from logistics. Supply published transport "
            "times or raise inter_centre_delay_hours / lower cit_max_hours."
            % (cfg.logistics.cit_max_hours, cfg.logistics.prep_hours))
    steady = cfg.pool.arrival_rate * (1 - cfg.pool.altruist_rate) * min(cfg.experiment.taus) \
        / cfg.dynamics.round_length_months
    if steady < cfg.clearing.batch_threshold and cfg.clearing.batch_max_interval is None:
        report["warnings"].append(
            "Batched policy may never clear: with lambda=%.1f/round and tau=%g months the pool "
            "holds at most ~%.0f patients before withdrawals, below n*=%d. Raise arrival_rate, "
            "lower batch_threshold, or set clearing.batch_max_interval."
            % (cfg.pool.arrival_rate, min(cfg.experiment.taus), steady, cfg.clearing.batch_threshold))
    if df["chains_truncated"].any():
        report["warnings"].append(
            "Chain enumeration hit max_chains; long chains were sampled. ILP and MOEA share the "
            "same candidate set, so PoF_t is unaffected, but F1^ILP is a ceiling over that set.")
    (out_dir(cfg) / "calibration.json").write_text(json.dumps(report, indent=2, default=float))
    return report


# ------------------------------------------------------------ 2. simulate

def _init_worker():
    # One BLAS thread per process; avoids oversubscription with n_jobs workers.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def _task(args):
    cfg_dict, baseline, tau, run, raw_path = args
    from dkex.experiments import simulate

    cfg = Config.from_dict(cfg_dict)
    rows, summary = simulate(cfg, baseline, tau, run)
    Path(raw_path).write_text(json.dumps({"rows": rows, "summary": summary}, default=_json_default))
    return raw_path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def simulate_all(cfg: Config, progress: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    e = cfg.experiment
    raw = out_dir(cfg) / "raw"
    raw.mkdir(exist_ok=True)
    tasks = []
    for tau in e.taus:
        for b in e.baselines:
            for r in range(e.runs):
                p = raw / f"{b}__tau{tau:g}__run{r:03d}.json"
                if not p.exists():                       # resumable
                    tasks.append((cfg.to_dict(), b, float(tau), r, str(p)))
    total = len(e.taus) * len(e.baselines) * e.runs
    if progress:
        print(f"[simulate] {total - len(tasks)}/{total} trajectories cached, {len(tasks)} to run "
              f"on {e.n_jobs} worker(s)")
    if e.n_jobs > 1 and tasks:
        # "spawn" (not fork): forking after HiGHS/BLAS initialise threads can deadlock.
        with ProcessPoolExecutor(e.n_jobs, mp_context=mp.get_context("spawn"),
                                 initializer=_init_worker) as ex:
            futs = [ex.submit(_task, t) for t in tasks]
            for k, f in enumerate(as_completed(futs), 1):
                f.result()
                if progress:
                    print(f"[simulate] {k}/{len(tasks)} done", flush=True)
    else:
        for k, t in enumerate(tasks, 1):
            _task(t)
            if progress:
                print(f"[simulate] {k}/{len(tasks)} {t[1]} tau={t[2]:g} run={t[3]}", flush=True)
    return collect(cfg)


def collect(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    e = cfg.experiment
    rows, runs = [], []
    wanted = {(b, float(t)) for b in e.baselines for t in e.taus}
    for p in sorted((out_dir(cfg) / "raw").glob("*.json")):
        d = json.loads(p.read_text())
        s = d["summary"]
        if (s["config"], float(s["tau"])) in wanted and s["run"] < e.runs:
            rows.extend(d["rows"])
            runs.append(s)
    rounds_df, runs_df = pd.DataFrame(rows), pd.DataFrame(runs)
    rounds_df.to_csv(out_dir(cfg) / "rounds.csv", index=False)
    runs_df.to_csv(out_dir(cfg) / "runs.csv", index=False)
    return rounds_df, runs_df


# ------------------------------------------------------------- 3. analyse

FOCUSED = [
    # Central question: clearing strategy (same AOS).
    ("dyn_nsga3_ucb1_myopic", "dyn_nsga3_ucb1_batched"),
    ("dyn_nsga3_linucb_myopic", "dyn_nsga3_linucb_batched"),
    # Supporting question: contextual vs stateless AOS (same clearing).
    ("dyn_nsga3_ucb1_myopic", "dyn_nsga3_linucb_myopic"),
    ("dyn_nsga3_ucb1_batched", "dyn_nsga3_linucb_batched"),
    # Dynamic vs static framework.
    ("static_nsga2_ucb1", "dyn_nsga3_ucb1_myopic"),
    ("static_nsga3_fixed", "dyn_nsga3_ucb1_myopic"),
]
TEST_METRICS = ["pof_mean", "pof_var", "total_transplants", "total_high_cpra_transplants",
                "mean_F3", "mean_HV", "high_cpra_accumulation"]


def analyse(cfg: Config) -> dict[str, pd.DataFrame]:
    from dkex.analysis.metrics import (arm_distribution, clearing_comparison,
                                       round_trajectories, trajectory_table)
    from dkex.analysis.stats import pairwise_wilcoxon

    d = out_dir(cfg)
    rounds = pd.read_csv(d / "rounds.csv")
    runs = pd.read_csv(d / "runs.csv")
    out = {
        "trajectories": round_trajectories(rounds),
        "trajectory_metrics": trajectory_table(runs),
        "arm_distribution": arm_distribution(rounds),
        "clearing_comparison": clearing_comparison(rounds),
    }
    tests = []
    present = set(runs["config"])
    comps = [c for c in FOCUSED if set(c) <= present]
    for tau, sub in runs.groupby("tau"):
        for m in TEST_METRICS:
            if m in sub and sub[m].notna().any() and comps:
                t = pairwise_wilcoxon(sub, m, comparisons=comps)
                t.insert(0, "tau", tau)
                tests.append(t)
    out["wilcoxon"] = pd.concat(tests, ignore_index=True) if tests else pd.DataFrame()
    for name, df in out.items():
        df.to_csv(d / f"{name}.csv", index=False)
    (d / "summary.md").write_text(_summary_md(cfg, out))
    return out


def _summary_md(cfg: Config, out: dict[str, pd.DataFrame]) -> str:
    tm = out["trajectory_metrics"]
    cols = ["config", "tau", "pof_mean_mean", "pof_var_mean", "total_transplants_mean",
            "total_high_cpra_transplants_mean", "mean_F3_mean", "high_cpra_accumulation_mean"]
    cols = [c for c in cols if c in tm]
    lines = [f"# Results — {cfg.experiment.name}", "",
             f"Runs per configuration: {cfg.experiment.runs}; rounds T = {cfg.dynamics.rounds}; "
             f"MOEA N = {cfg.moea.pop_size}, G = {cfg.moea.generations}.", "",
             "## Trajectory metrics (mean over runs)", "", _md_table(tm[cols].round(3)), ""]
    w = out["wilcoxon"]
    if not w.empty:
        lines += ["## Focused paired comparisons (Wilcoxon, Bonferroni-corrected)", "",
                  _md_table(w[["tau", "metric", "a", "b", "mean_a", "mean_b", "p_bonferroni",
                               "significant", "cohens_d"]].round(4)), ""]
    return "\n".join(lines)


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(none)_"
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


# ---------------------------------------------------------------- 4. plot

def plot(cfg: Config) -> list[Path]:
    from dkex.analysis.plots import plot_all

    d = out_dir(cfg)
    traj = pd.read_csv(d / "trajectories.csv")
    arms_p = d / "arm_distribution.csv"
    arms = pd.read_csv(arms_p) if arms_p.exists() and arms_p.stat().st_size > 1 else pd.DataFrame()
    return plot_all(traj, arms, d / "figures")


# ----------------------------------------------------------------- 5. ood

def ood(cfg: Config, wmd_path: str | Path) -> pd.DataFrame:
    """Run the configured baselines on a PrefLib 00036 instance (Section 4.6)."""
    from dkex.data.preflib import load_wmd
    from dkex.experiments import simulate

    graph = load_wmd(wmd_path)
    d = out_dir(cfg) / "ood"
    d.mkdir(exist_ok=True)
    rows, runs = [], []
    for tau in cfg.experiment.taus:
        for b in cfg.experiment.baselines:
            for r in range(cfg.experiment.runs):
                rr, s = simulate(cfg, b, float(tau), r, preflib_graph=graph)
                rows.extend(rr)
                runs.append(s)
    pd.DataFrame(rows).to_csv(d / "rounds.csv", index=False)
    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(d / "runs.csv", index=False)
    return runs_df
