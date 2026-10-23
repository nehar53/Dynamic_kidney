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
# A two-stage ILP first maximises the number of transplants and then,
# with that volume fixed, picks the fairest maximum matching. Does the
# MOEA add anything? Two stage-2 objectives (module 6):
#   twostage_ilp_sum  min F2/F2_ref + F3/F3_ref + F4/F4_ref
#   twostage_ilp_f3   min F3 only
# Same pool, compatibility graph, structure list, feasibility rules
# and objective code as the framework. Stage 2 is exact (all
# objectives are linear once F1 is fixed) with a 60 s limit per round
# (rounds that reach it are counted). Ties between equally fair
# maximum matchings are broken deterministically (module 6).
#
# Three comparisons, all written to results/paper/module12_twostage.md
#   A. Same pool, one round: the 30 pools of the module 11 ablations.
#      The MOEA front solutions with the maximum volume (the one the
#      100% rule picks; the best F3 and best F4 among them) against
#      the two-stage ILP on the identical pool. Cost per method.
#   B. Same pool, every round of a trajectory: the proposed framework
#      (tau = 12, 30 runs; unchanged, so its pools are those of the
#      main experiment) also solves the two-stage ILP on each round's
#      pool and records both. Exact vs searched rounds reported apart.
#   C. Over trajectories: two-stage ILP clearing every round vs the
#      framework (90% and, if present, 100% floor) and the myopic ILP
#      (framework / ILP runs are reused from the main experiment).
#
# Press Run (no arguments), or:
#   python module12_twostage.py --steps pools shadow    (A and B only)
#   python module12_twostage.py --analyse-only
# Resumable: finished pools and trajectories are kept on disk.

STEPS = ["pools", "shadow", "trajectory"]
POOLS = 30                          # A: seeds 42.. (as in module 11)
POOL_TIME_LIMIT = 300               # A: stage-2 limit per pool (75-vertex
                                    # pools are the largest; trajectories: 60 s)
SHADOW_TAUS = [12]                  # B
TAUS = [6, 12, 18]
RUNS = 30
RESULTS = "results"
JOBS = max(1, (os.cpu_count() or 2) - 1)

FRAME, ILP = "dyn_nsga3_ucb1_myopic", "myopic_ilp"
SHADOW = "dyn_nsga3_ucb1_myopic_shadow"
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
    m10.EXPERIMENTS["twostage_shadow"] = {
        "variants": ["base"], "configs": [SHADOW], "taus": a.shadow_taus,
        "runs": a.runs, "about": "same-pool two-stage ILP (module 12)"}
    args = m10.parse_args(["--out", a.results, "--jobs", str(a.jobs)])
    args.experiments = ["twostage"]
    args.configs = CONFIGS
    sargs = m10.parse_args(["--out", a.results, "--jobs", str(a.jobs)])
    sargs.experiments = ["twostage_shadow"]
    sargs.configs = [SHADOW]
    return m10, args, sargs


# ============================================================
# A. SAME POOL, ONE ROUND
# ============================================================

def _pool_task(seed: int) -> Dict:
    from module1_generator import generate_initial_pool
    from module3_logistics import (add_chains, build_compatibility_graph,
                                   build_exchange_structures)
    from module5_nsga3_aos import UCB1AOS, build_context_vector, run_moea_round
    from module6_ilp_baseline import solve_ilp_max_transplants, solve_ilp_picef
    from module7_clearing import _shadow_two_stage, select_solution
    from module4_objectives import compute_all_objectives
    pool, _ = generate_initial_pool(n_pairs=75, seed=seed, initial_wait_max=6)
    P = pool.pairs
    e = build_compatibility_graph(P)
    s = build_exchange_structures(P, e)
    t0 = time.time()
    if s.truncated:
        add_chains(s, solve_ilp_picef(P, e)[1]["chains"], e)
    ids, f1, info = solve_ilp_max_transplants(P, e, s)
    sec_ilp = time.time() - t0
    t0 = time.time()
    res = run_moea_round(P, UCB1AOS(seed=seed), build_context_vector(P, 0, 1, 25, 0.0),
                         n_gen=150, pop_size=100, seed=seed, edges=e, structures=s,
                         guide_structures=info["structure_indices"])
    sec_moea = time.time() - t0
    row = {"seed": seed, "F1_ILP": f1, "sec_ilp": sec_ilp, "sec_moea": sec_moea,
           "exact_front": bool(getattr(res, "exact", False)),
           "front_size": len(res.matched_sets)}
    row.update(_shadow_two_stage(res, P, e, s, ids, f1, time_limit=POOL_TIME_LIMIT))
    pick90, _ = select_solution(res, "floor", 0.9)          # main rule
    q = compute_all_objectives(pick90, P)
    row.update({"moea_pick90_F1": q["F1"], "moea_pick90_F2": q["F2"],
                "moea_pick90_F3": q["F3"], "moea_pick90_F4": q["F4"]})
    return row


def run_pools(a) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    out = Path(a.results) / "ablation_twostage_pools.csv"
    done = set(pd.read_csv(out)["seed"].astype(int)) if out.exists() else set()
    todo = [sd for sd in range(42, 42 + a.pools) if sd not in done]
    print(f"\nA. Same-pool comparison: {len(done)} pools done, {len(todo)} to run")

    out.parent.mkdir(parents=True, exist_ok=True)

    def save(row):
        # rewrite the whole file: rows can have different columns
        old = pd.read_csv(out) if out.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(out, index=False)

    if a.jobs > 1 and todo:
        with ProcessPoolExecutor(max_workers=a.jobs, mp_context=get_context("spawn")) as ex:
            futs = {ex.submit(_pool_task, sd): sd for sd in todo}
            for k, f in enumerate(as_completed(futs), 1):
                save(f.result())
                print(f"  [{k}/{len(todo)}] pool {futs[f]} done", flush=True)
    else:
        for k, sd in enumerate(todo, 1):
            save(_pool_task(sd))
            print(f"  [{k}/{len(todo)}] pool {sd} done", flush=True)


# rows of the same-pool tables: (label, column prefix)
SAME_POOL_ROWS = [("Volume ILP (stage 1 only)", "ilp_"),
                  ("Two-stage ILP (sum)", "ts_sum_"),
                  ("Two-stage ILP (F3)", "ts_f3_"),
                  ("MOEA, 100% rule pick", "moea_pick100_"),
                  ("MOEA, best F3 at max volume", None),
                  ("MOEA, best F4 at max volume", None)]


def _score(df: pd.DataFrame, pre: str) -> pd.Series:
    """Stage-2 'sum' objective of a solution: F_k / max(F_k(ILP), 0.01)."""
    r = {k: np.maximum(df[f"ilp_{k}"], 0.01) for k in ("F2", "F3", "F4")}
    return sum(df[f"{pre}{k}"] / r[k] for k in ("F2", "F3", "F4"))


def same_pool_table(df: pd.DataFrame, unit: str) -> (pd.DataFrame, pd.DataFrame, Dict):
    """df: one row per unit (pool, or run with per-run means)."""
    rows = []
    for lab, pre in SAME_POOL_ROWS:
        if pre is None:
            k = "F3" if "F3" in lab else "F4"
            col = f"moea_maxvol_min{k}"
            if col in df:
                rows.append({"method": lab, k: f"{df[col].mean():.4f}"})
            continue
        if f"{pre}F3" not in df:
            continue
        rows.append({"method": lab,
                     **{k: f"{df[f'{pre}{k}'].mean():.4f}" for k in ("F2", "F3", "F4")},
                     "score (sum objective)": f"{_score(df, pre).mean():.3f}"})
    comps = []
    tests = [("MOEA, 100% rule pick", "moea_pick100_", "Two-stage ILP (sum)", "ts_sum_"),
             ("MOEA, 100% rule pick", "moea_pick100_", "Two-stage ILP (F3)", "ts_f3_")]
    for la, pa, lb, pb in tests:
        if f"{pa}F3" not in df or f"{pb}F3" not in df:
            continue
        for k in ("F2", "F3", "F4"):
            comps.append({"a": la, "b": lb, "metric": k,
                          **paired(df[f"{pa}{k}"], df[f"{pb}{k}"])})
        comps.append({"a": la, "b": lb, "metric": "score",
                      **paired(_score(df, pa), _score(df, pb))})
    if "moea_maxvol_minF3" in df and "ts_f3_F3" in df:
        comps.append({"a": "MOEA, best F3 at max volume", "b": "Two-stage ILP (F3)",
                      "metric": "F3", **paired(df["moea_maxvol_minF3"], df["ts_f3_F3"])})
    cmp_ = pd.DataFrame(comps)
    if len(cmp_):
        k = cmp_["p"].notna().sum()
        cmp_["p_bonf"] = np.minimum(1.0, cmp_["p"] * max(k, 1))
    return pd.DataFrame(rows), cmp_, {}


def _optimality_shares(df: pd.DataFrame) -> Dict:
    """How often the MOEA's max-volume front contained the ILP optimum."""
    out = {}
    if "moea_maxvol_minF3" in df and "ts_f3_F3" in df:
        out["share MOEA best F3 = two-stage F3 optimum"] = float(
            (df["moea_maxvol_minF3"] <= df["ts_f3_F3"] + 1e-9).mean())
    if "moea_pick100_F3" in df and "ts_sum_F3" in df:
        out["share MOEA 100% pick score <= two-stage (sum) score"] = float(
            (_score(df, "moea_pick100_") <= _score(df, "ts_sum_") + 1e-9).mean())
    return out


def report_pools(a, md: List[str], J: Dict) -> None:
    p = Path(a.results) / "ablation_twostage_pools.csv"
    md += ["## A. Same pool, one round (30 pools of 75 vertices)", ""]
    if not p.exists():
        md += ["not available (step pools)", ""]
        return
    df = pd.read_csv(p).drop_duplicates("seed")
    df = df[df["F1_ILP"] > 0]
    tab, cmp_, _ = same_pool_table(df, "pool")
    sh = _optimality_shares(df)
    cost = {"ILP (stage 1)": df["sec_ilp"].mean(),
            "Two-stage stage 2 (sum)": df["ts_sum_sec"].mean(),
            "Two-stage stage 2 (F3)": df["ts_f3_sec"].mean(),
            "MOEA (N=100, G=150)": df["sec_moea"].mean()}
    md += [f"Pools: {len(df)}; exact MOEA fronts: {int(df['exact_front'].sum())}; "
           f"MOEA front solutions with maximum volume per pool: "
           f"{df['moea_maxvol_n'].mean():.1f} on average.", "",
           _md(tab), "",
           "Paired Wilcoxon over pools, Bonferroni over the rows below:", "",
           _md(cmp_.assign(p=cmp_["p"].map(_p), p_bonf=cmp_["p_bonf"].map(_p)))
           if len(cmp_) else "n/a", "",
           "Optimality: " + "; ".join(f"{k}: {v:.3f}" for k, v in sh.items()), "",
           "Mean seconds per pool: " + "; ".join(f"{k} {v:.2f}" for k, v in cost.items()), "",
           f"Stage-2 time-limit hits ({POOL_TIME_LIMIT} s): "
           f"sum {(df['ts_sum_status'] == 'time_limit').mean():.3f}, "
           f"F3 {(df['ts_f3_status'] == 'time_limit').mean():.3f}", ""]
    J["same_pool_one_round"] = {"table": tab.to_dict("records"),
                                "comparisons": cmp_.round(6).to_dict("records"),
                                "optimality": sh, "seconds": cost}


# ============================================================
# B. SAME POOL, EVERY ROUND OF THE FRAMEWORK'S TRAJECTORY
# ============================================================

def report_shadow(m10, sargs, stasks, a, md: List[str], J: Dict) -> None:
    md += ["## B. Same pool, every round (framework trajectories, "
           f"tau = {', '.join(map(str, a.shadow_taus))})", ""]
    if not any(Path(t["path"]).exists() for t in stasks):
        md += ["not available (step shadow)", ""]
        return
    r = m10.collect(stasks)
    c = r[r["cleared"] & (r["F1_ILP"] > 0) & r["ts_sum_F3"].notna()].copy()
    # consistency with the main experiment (same seeds, same code)
    same = []
    for t in stasks:
        main = Path(m10.raw_path(Path(a.results), "base", FRAME, t["tau"], t["run"]))
        if main.exists() and Path(t["path"]).exists():
            x = pd.read_csv(main)["F1"].fillna(0).sum()
            y = pd.read_csv(t["path"])["F1"].fillna(0).sum()
            same.append(abs(x - y) < 1e-9)
    if same:
        md += [f"Trajectory identical to the main-experiment framework run "
               f"(same seed): {sum(same)}/{len(same)} runs.", ""]
    for name, part in [("all rounds", c),
                       ("searched rounds only (MOEA front not exact)", c[~c["exact_front"].astype(bool)]),
                       ("exactly enumerated rounds", c[c["exact_front"].astype(bool)])]:
        if part.empty:
            continue
        cols = [x for x in part.columns if x.startswith(("ilp_F", "ts_", "moea_"))
                and pd.api.types.is_numeric_dtype(part[x])]
        per_run = part.groupby(["tau", "run"])[cols].mean().reset_index()
        tab, cmp_, _ = same_pool_table(per_run, "run")
        sh = _optimality_shares(part)
        md += [f"### {name}: {len(part)} rounds in {per_run['run'].nunique()} runs", "",
               _md(tab), "",
               "Paired Wilcoxon over runs (per-run means), Bonferroni over the rows below:", "",
               _md(cmp_.assign(p=cmp_["p"].map(_p), p_bonf=cmp_["p_bonf"].map(_p)))
               if len(cmp_) else "n/a", "",
               "Optimality (share of rounds): "
               + "; ".join(f"{k}: {v:.3f}" for k, v in sh.items()), ""]
        J.setdefault("same_pool_every_round", {})[name] = {
            "table": tab.to_dict("records"),
            "comparisons": cmp_.round(6).to_dict("records"), "optimality": sh}
    cost = {"ILP stage 1 + structures (framework round)": np.nan,
            "Two-stage stage 2 (sum)": c["ts_sum_sec"].mean(),
            "Two-stage stage 2 (F3)": c["ts_f3_sec"].mean()}
    md += ["Mean seconds per round: two-stage stage 2 (sum) "
           f"{cost['Two-stage stage 2 (sum)']:.2f}, (F3) {cost['Two-stage stage 2 (F3)']:.2f}; "
           "time-limit hits: "
           f"sum {(c['ts_sum_status'] == 'time_limit').mean():.3f}, "
           f"F3 {(c['ts_f3_status'] == 'time_limit').mean():.3f}", ""]


# ============================================================
# C. OVER TRAJECTORIES
# ============================================================

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


def report_trajectory(m10, args, tasks, a, md: List[str], J: Dict) -> None:
    out = Path(a.results)
    md += ["## C. Over trajectories (each method clears its own pool)", ""]
    if not any(Path(t["path"]).exists() for t in tasks):
        md += ["not available (step trajectory)", ""]
        return
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

    if "two_stage_status" in rounds:
        ts = rounds[rounds["config"].isin([TS_SUM, TS_F3]) & rounds["cleared"]]
        share = (ts["two_stage_status"] == "time_limit").groupby(ts["config"]).mean()
        md += ["Share of rounds where stage 2 hit the 60 s limit (best matching found used): "
               + ", ".join(f"{NAMES[c]} {v:.3f}" for c, v in share.items()), ""]
    J.setdefault("trajectory", {"table": [], "comparisons": []})
    JT = J["trajectory"]
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
            JT["table"].append({"tau": int(tau), "config": c,
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
            JT["comparisons"] += [{"tau": int(tau), **r} for r in cmp_.round(6).to_dict("records")]
        md += [f"### tau = {tau}", "", _md(pd.DataFrame(rows)), "",
               "Paired Wilcoxon, Bonferroni over all rows of this table:", "",
               _md(cmp_.assign(p=cmp_["p"].map(_p), p_bonf=cmp_["p_bonf"].map(_p)))
               if len(cmp_) else "n/a", ""]


def write_report(m10, args, tasks, sargs, stasks, a) -> None:
    out = Path(a.results)
    md = ["# Module 12 — two-stage ILP baseline", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} from {out}.", "",
          "Two-stage ILP: stage 1 maximises transplants; stage 2 fixes that volume "
          "and minimises F2/F2_ref + F3/F3_ref + F4/F4_ref ('sum') or F3 alone, "
          "with references from the stage-1 matching (floored at 0.01). Ties are "
          "broken deterministically (module 6). Same structure list, feasibility "
          "rules and objective code as the framework.", ""]
    J: Dict = {}
    report_pools(a, md, J)
    report_shadow(m10, sargs, stasks, a, md, J)
    report_trajectory(m10, args, tasks, a, md, J)
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
    ap.add_argument("--steps", nargs="+", default=STEPS, choices=STEPS)
    ap.add_argument("--taus", type=int, nargs="+", default=TAUS)
    ap.add_argument("--shadow-taus", type=int, nargs="+", default=SHADOW_TAUS)
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--pools", type=int, default=POOLS)
    ap.add_argument("--jobs", type=int, default=JOBS)
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args(argv)
    print("=" * 66 + "\nMODULE 12: two-stage ILP baseline\n" + "=" * 66)
    m10, args, sargs = build_args(a)
    tasks, stasks = m10.plan(args), m10.plan(sargs)
    if not a.analyse_only:
        if "pools" in a.steps:
            run_pools(a)
        if "shadow" in a.steps:
            print("\nB. Framework trajectories with a same-pool two-stage ILP")
            m10.run_all(sargs, stasks)
        if "trajectory" in a.steps:
            print("\nC. Two-stage ILP trajectories")
            m10.run_all(args, tasks)
    write_report(m10, args, tasks, sargs, stasks, a)
    print("\n" + "=" * 66 + "\nMODULE 12 COMPLETE\n" + "=" * 66)


if __name__ == "__main__":
    main()
