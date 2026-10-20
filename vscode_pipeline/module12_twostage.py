import os

# one BLAS thread per worker (same as modules 9-11)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ============================================================
# MODULE 12: TWO-STAGE ILP BASELINE ("why an MOEA?")
# ============================================================
#
# Question a reviewer will ask: a two-stage ILP could first maximise
# the number of transplants and then pick the fairest maximum
# matching. Does the MOEA add anything?
#
# Configurations (myopic clearing, same seeds as the main experiment,
# so the framework and myopic-ILP runs are reused, not re-run):
#   dyn_nsga3_ucb1_myopic  proposed framework (90% floor)
#   myopic_ilp             volume-maximising ILP
#   twostage_ilp_sum       max F1, then min F2/F2_ref + F3/F3_ref + F4/F4_ref
#   twostage_ilp_f3        max F1, then min F3 only
# plus, if available, the framework with a 100% floor (module 10
# experiment sens_select, tau = 12): the MOEA restricted to maximum
# matchings, i.e. the same feasible set as the two-stage ILP.
#
# Stage 2 is solved exactly by CBC (all objectives are linear once F1
# is fixed, see module 6), with a 60 s limit per round; rounds that
# reach the limit use the best matching found and are counted.
#
# Press Run (no arguments), or:
#   python module12_twostage.py --taus 12 --runs 10 --jobs 6
#   python module12_twostage.py --analyse-only
# Resumable: finished trajectories are kept in results/raw/.

TAUS = [6, 12, 18]
RUNS = 30
RESULTS = "results"
JOBS = max(1, (os.cpu_count() or 2) - 1)

FRAME, ILP = "dyn_nsga3_ucb1_myopic", "myopic_ilp"
TS_SUM, TS_F3 = "twostage_ilp_sum", "twostage_ilp_f3"
CONFIGS = [FRAME, ILP, TS_SUM, TS_F3]
NAMES = {FRAME: "Framework (90% floor)", ILP: "Myopic ILP",
         TS_SUM: "Two-stage ILP (sum)", TS_F3: "Two-stage ILP (F3)",
         "frame100": "Framework (100% floor)"}
METRICS = [("total_transplants", "Transplants"),
           ("total_high_cpra_transplants", "High-cPRA tx"),
           ("mean_F2", "F2"), ("mean_F3", "F3"), ("mean_F4", "F4"),
           ("pof_mean", "PoF (%)")]
PAIRS = [(FRAME, TS_SUM), (FRAME, TS_F3), (TS_SUM, ILP), (TS_F3, ILP)]


def _p(p):
    return "n/a" if p is None or np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(
            f"{v:.4g}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def paired(a: pd.Series, b: pd.Series) -> Dict:
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(j) < 2:
        return {"n": len(j), "mean_a": np.nan, "mean_b": np.nan, "p": np.nan, "d_z": np.nan}
    d = j.iloc[:, 0] - j.iloc[:, 1]
    p = np.nan if np.allclose(d, 0) else float(wilcoxon(j.iloc[:, 0], j.iloc[:, 1]).pvalue)
    sd = d.std(ddof=1)
    return {"n": len(j), "mean_a": float(j.iloc[:, 0].mean()),
            "mean_b": float(j.iloc[:, 1].mean()), "p": p,
            "d_z": float(d.mean() / sd) if sd > 0 else 0.0}


def _module10():
    """Module 10 under either file name (module10_experiment(s).py)."""
    try:
        import module10_experiment as m10
    except ImportError:
        import module10_experiments as m10
    return m10


def build_args(a):
    m10 = _module10()
    m10.EXPERIMENTS["twostage"] = {
        "variants": ["base"], "configs": CONFIGS, "taus": a.taus,
        "runs": a.runs, "about": "two-stage ILP baseline (module 12)"}
    args = m10.parse_args(["--out", a.results, "--jobs", str(a.jobs)])
    args.experiments = ["twostage"]
    args.configs = CONFIGS
    return m10, args


def load_floor100(m10, out: Path, runs: int) -> pd.DataFrame:
    tasks = [{"variant": "floor100", "path": str(m10.raw_path(out, "floor100", FRAME, 12, r))}
             for r in range(runs)]
    tasks = [t for t in tasks if Path(t["path"]).exists()]
    if not tasks:
        return pd.DataFrame()
    r = m10.collect(tasks)
    pr = m10.per_run_metrics(r)
    pr["config"] = "frame100"
    return pr, r


def report(m10, args, tasks, a) -> None:
    out = Path(a.results)
    rounds = m10.collect(tasks)
    runs = m10.per_run_metrics(rounds)
    secs = (rounds[rounds["cleared"]].groupby(["config", "tau"])["seconds"]
            .mean().rename("seconds_per_round"))
    f100 = load_floor100(m10, out, a.runs)
    if len(f100):
        f100_runs, f100_rounds = f100
        runs = pd.concat([runs, f100_runs], ignore_index=True)
        s100 = (f100_rounds[f100_rounds["cleared"]].groupby("tau")["seconds"].mean())
        for tau, v in s100.items():
            secs.loc[("frame100", tau)] = v

    md = ["# Module 12 — two-stage ILP baseline", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} from {out}. "
          f"Myopic clearing, paired runs (same seeds as the main experiment).", ""]
    if "two_stage_status" in rounds:
        ts = rounds[rounds["config"].isin([TS_SUM, TS_F3]) & rounds["cleared"]]
        share = (ts["two_stage_status"] == "time_limit").groupby(ts["config"]).mean()
        md += ["Share of rounds where stage 2 hit the 60 s limit (best matching found used): "
               + ", ".join(f"{NAMES[c]} {v:.3f}" for c, v in share.items()), ""]
    J = {"table": [], "comparisons": []}
    for tau in sorted(runs["tau"].unique()):
        sub = runs[runs["tau"] == tau]
        order = [c for c in [FRAME, "frame100", TS_SUM, TS_F3, ILP] if c in set(sub["config"])]
        rows = []
        for c in order:
            g = sub[sub["config"] == c]
            row = {"configuration": NAMES[c], "runs": int(g["run"].nunique())}
            for col, lab in METRICS:
                if col in g:
                    row[lab] = f"{g[col].mean():.4g} ± {g[col].std():.3g}"
            row["s/round"] = (f"{secs.loc[(c, tau)]:.2f}" if (c, tau) in secs.index else "n/a")
            rows.append(row)
            J["table"].append({"tau": int(tau), "config": c,
                               **{lab: float(g[col].mean()) for col, lab in METRICS if col in g},
                               "seconds_per_round": float(secs.loc[(c, tau)])
                               if (c, tau) in secs.index else None})
        comps = []
        pairs = list(PAIRS) + ([("frame100", TS_SUM), ("frame100", TS_F3)]
                               if "frame100" in order else [])
        for x, y in pairs:
            if x not in order or y not in order:
                continue
            for col, lab in METRICS:
                q = paired(sub[sub["config"] == x].set_index("run")[col],
                           sub[sub["config"] == y].set_index("run")[col])
                comps.append({"a": NAMES[x], "b": NAMES[y], "metric": lab, **q})
        cmp_ = pd.DataFrame(comps)
        if len(cmp_):
            k = cmp_["p"].notna().sum()
            cmp_["p_bonf"] = np.minimum(1.0, cmp_["p"] * max(k, 1))
            J["comparisons"] += [{"tau": int(tau), **r} for r in cmp_.round(6).to_dict("records")]
        md += [f"## tau = {tau}", "", _md(pd.DataFrame(rows)), "",
               "Paired Wilcoxon, Bonferroni over all rows of this table:", "",
               _md(cmp_.assign(p=cmp_["p"].map(_p), p_bonf=cmp_["p_bonf"].map(_p)))
               if len(cmp_) else "n/a", ""]
    pdir = out / "paper"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "module12_twostage.md").write_text("\n".join(md), encoding="utf-8")
    (pdir / "module12_twostage.json").write_text(json.dumps(J, indent=2, default=float),
                                                 encoding="utf-8")
    print("\n".join(md))
    print(f"\nWritten: {pdir / 'module12_twostage.md'} (send this file)")


def main(argv=None) -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="Module 12: two-stage ILP baseline")
    ap.add_argument("--taus", type=int, nargs="+", default=TAUS)
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--jobs", type=int, default=JOBS)
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args(argv)
    print("=" * 66 + "\nMODULE 12: two-stage ILP baseline\n" + "=" * 66)
    m10, args = build_args(a)
    tasks = m10.plan(args)
    if not a.analyse_only:
        m10.run_all(args, tasks)
    report(m10, args, tasks, a)
    print("\n" + "=" * 66 + "\nMODULE 12 COMPLETE\n" + "=" * 66)


if __name__ == "__main__":
    main()
