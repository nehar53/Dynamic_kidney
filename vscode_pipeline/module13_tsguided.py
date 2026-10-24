import os

# one BLAS thread per worker (same as modules 9-12)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ============================================================
# MODULE 13: TWO-STAGE-ILP-GUIDED NSGA-III
# ============================================================
#
# Module 12 showed that, at the maximum volume, the two-stage ILP
# finds fairer matchings than the NSGA-III front. This module tests
# the framework with the two-stage ILP as its guide instead of the
# volume ILP alone. Modules 1-12 are NOT changed: the new behaviour
# is added at run time, inside every worker, by wrapping
# module7.run_moea_round and module7.select_solution.
#
# In every clearing round of a "tsg_" configuration
#   1. small rounds (<= 2N matchings) are solved exactly, as before
#      (the exact front already contains the two-stage optima);
#   2. otherwise the two-stage ILP is solved on the round's pool with
#      both stage-2 objectives (sum and F3, module 6);
#   3. NSGA-III + AOS runs as before, but its seed and repair priority
#      come from the two-stage 'sum' matching, and the volume-ILP and
#      two-stage 'F3' matchings are added to the initial population;
#   4. the three exact matchings are inserted into the final front
#      (non-dominated filter, HV recomputed), so the front always
#      contains the fairest maximum matching.
#
# Implemented solution, rule "score" (selection variants below):
#   among front solutions with F1 >= rho * max F1, minimise
#   F2/r2 + F3/r3 + F4/r4, where r_k are the objectives of the
#   volume-ILP matching floored at 0.01: the stage-2 objective of the
#   two-stage ILP. With rho = 1 the pick equals the two-stage ILP
#   (sanity check); with rho = 0.9 the front is used to trade up to
#   10% of the round's transplants for fairness.
#
# Experiments (all written to results/paper/module13_tsguided.md)
#   tsg_main    4 tsg configurations (UCB1 / LinUCB x myopic / batched),
#               score rule, rho = 0.9, tau = 6/12/18
#   tsg_ref     reference runs reused from modules 10 and 12: the
#               original framework (4 configurations), myopic ILP,
#               Dickerson lex, two-stage ILP (sum, F3)
#   tsg_select  tau = 12: tsg UCB1 myopic with rho = 1.0 and 0.8, and
#               with the original selection rule (floor + compromise)
#   tsg_ablate  tau = 12: the ORIGINAL framework with the score rule
#               (separates the effect of the guide from the rule)
#
# Press Run (no arguments), or:
#   python module13_tsguided.py --steps main        (tsg_main only)
#   python module13_tsguided.py --quick             (pipeline check)
#   python module13_tsguided.py --analyse-only
# Resumable: finished trajectories are kept in results/raw.

STEPS = ["main", "select"]
TAUS = [6, 12, 18]
SELECT_TAUS = [12]
RUNS = 30                           # same runs as module 12 (seeds 42..71)
RESULTS = "results"
JOBS = max(1, (os.cpu_count() or 2) - 1)

TSG = ["tsg_nsga3_ucb1_myopic", "tsg_nsga3_linucb_myopic",
       "tsg_nsga3_ucb1_batched", "tsg_nsga3_linucb_batched"]
OLD = ["dyn_nsga3_ucb1_myopic", "dyn_nsga3_linucb_myopic",
       "dyn_nsga3_ucb1_batched", "dyn_nsga3_linucb_batched"]
ILP, LEX = "myopic_ilp", "dickerson_lex"
TS_SUM, TS_F3 = "twostage_ilp_sum", "twostage_ilp_f3"

NEW_CONFIGS: Dict[str, Dict] = {
    c: {**{"method": "nsga3", "aos": a, "clearing": cl, "dynamic": True},
        "ts_guided": True}
    for c, a, cl in zip(TSG, ["ucb1", "linucb", "ucb1", "linucb"],
                        ["myopic", "myopic", "batched", "batched"])}
NEW_VARIANTS: Dict[str, Dict] = {
    "score90":  {"selection_rule": "score"},
    "score100": {"selection_rule": "score", "min_f1_ratio": 1.0},
    "score80":  {"selection_rule": "score", "min_f1_ratio": 0.8},
}
SCORE_FLOOR = 0.01                  # same floor as the two-stage ILP (module 6)

NAMES = {
    ("tsg_nsga3_ucb1_myopic", "score90"): "TSG UCB1 myopic",
    ("tsg_nsga3_linucb_myopic", "score90"): "TSG LinUCB myopic",
    ("tsg_nsga3_ucb1_batched", "score90"): "TSG UCB1 batched",
    ("tsg_nsga3_linucb_batched", "score90"): "TSG LinUCB batched",
    ("dyn_nsga3_ucb1_myopic", "base"): "Old UCB1 myopic",
    ("dyn_nsga3_linucb_myopic", "base"): "Old LinUCB myopic",
    ("dyn_nsga3_ucb1_batched", "base"): "Old UCB1 batched",
    ("dyn_nsga3_linucb_batched", "base"): "Old LinUCB batched",
    (ILP, "base"): "Myopic ILP", (LEX, "base"): "Dickerson lex",
    (TS_SUM, "base"): "Two-stage ILP (sum)", (TS_F3, "base"): "Two-stage ILP (F3)",
    ("tsg_nsga3_ucb1_myopic", "score100"): "TSG UCB1 myopic, rho=1.0",
    ("tsg_nsga3_ucb1_myopic", "score80"): "TSG UCB1 myopic, rho=0.8",
    ("tsg_nsga3_ucb1_myopic", "base"): "TSG UCB1 myopic, original rule",
    ("dyn_nsga3_ucb1_myopic", "score90"): "Old UCB1 myopic, score rule",
}
METRICS = [("total_transplants", "Transplants"),
           ("total_high_cpra_transplants", "High-cPRA tx"),
           ("mean_F2", "F2"), ("mean_F3", "F3"), ("mean_F4", "F4"),
           ("pof_mean", "PoF (%)"), ("pof_vs_myopic_ilp", "Delta (%)"),
           ("total_withdrawn_high_cpra", "High-cPRA withdrawn")]
MAIN_PAIRS = [
    (("tsg_nsga3_ucb1_myopic", "score90"), ("dyn_nsga3_ucb1_myopic", "base")),
    (("tsg_nsga3_linucb_myopic", "score90"), ("dyn_nsga3_linucb_myopic", "base")),
    (("tsg_nsga3_ucb1_myopic", "score90"), (TS_SUM, "base")),
    (("tsg_nsga3_ucb1_myopic", "score90"), (TS_F3, "base")),
    (("tsg_nsga3_ucb1_myopic", "score90"), (ILP, "base")),
    (("tsg_nsga3_ucb1_myopic", "score90"), (LEX, "base")),
    (("tsg_nsga3_ucb1_myopic", "score90"), ("tsg_nsga3_ucb1_batched", "score90")),
    (("tsg_nsga3_linucb_myopic", "score90"), ("tsg_nsga3_linucb_batched", "score90")),
]
SELECT_ROWS = [("tsg_nsga3_ucb1_myopic", "score90"), ("tsg_nsga3_ucb1_myopic", "score100"),
               ("tsg_nsga3_ucb1_myopic", "score80"), ("tsg_nsga3_ucb1_myopic", "base"),
               ("dyn_nsga3_ucb1_myopic", "score90"), ("dyn_nsga3_ucb1_myopic", "base"),
               (TS_SUM, "base"), (ILP, "base")]
SELECT_PAIRS = [(("tsg_nsga3_ucb1_myopic", "score100"), (TS_SUM, "base")),
                (("tsg_nsga3_ucb1_myopic", "score90"), ("tsg_nsga3_ucb1_myopic", "base")),
                (("tsg_nsga3_ucb1_myopic", "score90"), ("dyn_nsga3_ucb1_myopic", "score90")),
                (("dyn_nsga3_ucb1_myopic", "score90"), ("dyn_nsga3_ucb1_myopic", "base"))]


# ============================================================
# RUN-TIME EXTENSION OF MODULES 7 AND 10 (no file is changed)
# ============================================================

_STATE: Dict = {"ready": False, "guided": False, "diag": []}


def _module10():
    """Module 10 under either file name (module10_experiment(s).py)."""
    try:
        import module10_experiment as m10
    except ImportError:
        import module10_experiments as m10
    return m10


def _setup():
    """Add the configurations/variants and wrap module 7 (once per process)."""
    m10 = _module10()
    if _STATE["ready"]:
        return m10
    import module7_clearing as m7
    for name, cfg in NEW_CONFIGS.items():
        m10.CONFIGS[name] = {**cfg, "name": name}
    m10.VARIANTS.update(NEW_VARIANTS)
    m10.SHORT.update({c: NAMES[(c, "score90")] for c in TSG})
    _STATE.update(ready=True, run_task=m10.run_task,
                  run_round=m7.run_moea_round, select=m7.select_solution)
    m7.run_moea_round = _guided_round
    m7.select_solution = _select
    return m10


def _union(structures, indices) -> Set[int]:
    out: Set[int] = set()
    for k in indices:
        out.update(structures.nodes[k])
    return out


def _guided_round(pool_pairs, aos=None, context=None, **kw):
    """
    Replaces module7.run_moea_round. Original configurations: the
    original round, unchanged (only the score references are attached).
    tsg configurations: the two-stage-ILP-guided round (header).
    """
    from module4_objectives import ObjectiveEvaluator, compute_all_objectives
    from module5_nsga3_aos import (EXACT_MAX_PACKINGS, ParetoArchive,
                                   compute_hypervolume_norm, enumerate_packings)
    from module6_ilp_baseline import solve_ilp_two_stage
    run_round = _STATE["run_round"]
    edges, structures = kw.get("edges"), kw.get("structures")
    guide = kw.get("guide_structures")
    ref = None
    if guide is not None and structures is not None:
        ilp_set = _union(structures, guide)
        o = compute_all_objectives(ilp_set, pool_pairs)
        ref = (o["F2"], o["F3"], o["F4"])
    diag = {"tsg_searched": False}

    small = (structures is not None and kw.get("allow_exact", True)
             and EXACT_MAX_PACKINGS > 0 and enumerate_packings(
                 structures, [p.pair_id for p in pool_pairs],
                 EXACT_MAX_PACKINGS) is not None)
    if not _STATE["guided"] or ref is None or small:
        res = run_round(pool_pairs, aos, context, **kw)
        res.score_ref, res.ts_pref = ref, None
        _STATE["diag"].append(diag)
        return res

    # two-stage ILP on this round's pool (same structures as the ILP)
    f1 = o["F1"]
    t0 = time.time()
    ts_sum, i_sum = solve_ilp_two_stage(pool_pairs, edges, structures, f1, ref,
                                        objective="sum")
    ts_f3, i_f3 = solve_ilp_two_stage(pool_pairs, edges, structures, f1, ref,
                                      objective="f3")
    ts_sec = time.time() - t0
    ts_sum = ts_sum or ilp_set                  # solver failure -> volume ILP
    ts_f3 = ts_f3 or ilp_set
    kw = dict(kw)
    kw["guide_structures"] = (i_sum.get("structure_indices") or list(guide))
    kw["warm_start"] = [ilp_set, ts_f3] + list(kw.get("warm_start") or [])
    res = run_round(pool_pairs, aos, context, **kw)

    # insert the exact matchings into the final front
    ev = ObjectiveEvaluator(pool_pairs)
    ids = [p.pair_id for p in pool_pairs]
    pos = {pid: k for k, pid in enumerate(ids)}
    inj = [ts_sum, ts_f3, ilp_set]
    Xi = np.zeros((len(inj), len(ids)), dtype=bool)
    for r, m in enumerate(inj):
        Xi[r, [pos[i] for i in m if i in pos]] = True
    Fi = np.atleast_2d(ev.minimisation_vector(np.array([ev.evaluate(m) for m in inj])))
    Fm = np.atleast_2d(res.F_norm)
    found = bool(((Fm <= Fi[0] + 1e-12).all(axis=1)).any())   # MOEA had it (or better)
    sets = {x.tobytes(): set(m) for x, m in zip(np.atleast_2d(res.X), res.matched_sets)}
    sets.update({x.tobytes(): set(m) for x, m in zip(Xi, inj)})
    arch = ParetoArchive()
    arch.update(np.atleast_2d(res.X), Fm)
    n_before = len(arch.X)
    keys_before = set(arch._keys)
    arch.update(Xi, Fi)
    res.X, res.F_norm = arch.X, arch.F
    res.matched_sets = [sets[x.tobytes()] for x in arch.X]
    raw = np.array([ev.evaluate(m) for m in res.matched_sets])
    res.F = raw.copy()
    res.F[:, 0] = -raw[:, 0]
    hv = compute_hypervolume_norm(res.F_norm)
    if res.hv_trace:
        res.hv_trace[-1] = hv
    else:
        res.hv_trace = [hv]
    res.score_ref, res.ts_pref = ref, set(ts_sum)
    diag.update({
        "tsg_searched": True, "tsg_ts_sec": ts_sec,
        "tsg_status_sum": i_sum.get("status", ""), "tsg_status_f3": i_f3.get("status", ""),
        "tsg_moea_found_ts_sum": found,
        "tsg_new_on_front": len(set(arch._keys) - keys_before),
        "tsg_front_before": n_before})
    _STATE["diag"].append(diag)
    return res


def _select(result, rule="floor", min_f1_ratio=0.9):
    """
    Replaces module7.select_solution: adds rule "score" (header);
    every other rule is the original function.
    """
    ref = getattr(result, "score_ref", None) if result is not None else None
    if rule != "score":
        return _STATE["select"](result, rule, min_f1_ratio)
    if ref is None:                             # no ILP reference: original rule
        return _STATE["select"](result, "floor", min_f1_ratio)
    if result is None or len(result.matched_sets) == 0:
        return set(), -1
    F = np.atleast_2d(result.F)                 # (-F1, F2, F3, F4)
    cand = np.arange(len(F))
    nonempty = cand[F[:, 0] < 0]
    if len(nonempty):
        cand = nonempty
    f1_max = -F[cand, 0].min()
    ok = cand[-F[cand, 0] >= min_f1_ratio * f1_max - 1e-9]
    r = np.maximum(np.asarray(ref, dtype=float), SCORE_FLOOR)
    score = np.round((F[ok, 1:4] / r).sum(axis=1), 9)
    pref = getattr(result, "ts_pref", None)
    is_ts = np.array([pref is not None and set(result.matched_sets[k]) == pref for k in ok])
    # ties: more transplants, then the two-stage ILP matching, then front order
    best = int(ok[np.lexsort((~is_ts, F[ok, 0], score))[0]])
    return set(result.matched_sets[best]), best


def _task13(task: Dict):
    """Worker: one trajectory with the run-time extension active."""
    m10 = _setup()
    _STATE["guided"] = bool(m10.CONFIGS[task["config"]].get("ts_guided", False))
    _STATE["diag"] = []
    path, secs = _STATE["run_task"](task)
    if _STATE["guided"]:                        # per-round diagnostics
        df = pd.read_csv(path)
        idx = df.index[df["cleared"].astype(bool)]
        d = pd.DataFrame(_STATE["diag"])
        if len(d) == len(idx):
            for col in d.columns:
                df.loc[idx, col] = d[col].to_numpy()
            tmp = Path(path).with_suffix(".tmp")
            df.to_csv(tmp, index=False)
            # OneDrive / antivirus can lock the CSV just written by
            # module 10 for a few seconds: retry, and if it stays locked
            # keep the trajectory without the diagnostic columns
            for wait in [0.5, 1, 2, 4, 8, 15, 30]:
                try:
                    tmp.replace(path)
                    break
                except PermissionError:
                    time.sleep(wait)
            else:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                print(f"NOTE: {path} locked; saved without diagnostics", flush=True)
    return path, secs


# ============================================================
# PLAN
# ============================================================

def build_args(a):
    m10 = _setup()
    E = m10.EXPERIMENTS
    E["tsg_main"] = {"variants": ["score90"], "configs": TSG, "taus": a.taus,
                     "runs": a.runs, "about": "two-stage-ILP-guided NSGA-III (module 13)"}
    E["tsg_ref"] = {"variants": ["base"], "configs": OLD + [ILP, LEX, TS_SUM, TS_F3],
                    "taus": a.taus, "runs": a.runs,
                    "about": "reference runs (modules 10 and 12)"}
    E["tsg_select"] = {"variants": ["score100", "score80", "base"],
                       "configs": ["tsg_nsga3_ucb1_myopic"], "taus": a.select_taus,
                       "runs": a.runs, "about": "selection rule (module 13)"}
    E["tsg_ablate"] = {"variants": ["score90"], "configs": ["dyn_nsga3_ucb1_myopic"],
                       "taus": a.select_taus, "runs": a.runs,
                       "about": "original framework with the score rule (module 13)"}
    base = ["--out", a.results, "--jobs", str(a.jobs)] + (["--quick"] if a.quick else [])
    if a.quick:
        base += ["--runs", str(a.runs if a.runs != RUNS else 2), "--rounds", "4"]
    groups = {}
    for key, exps in [("main", ["tsg_main", "tsg_ref"]),
                      ("select", ["tsg_select", "tsg_ablate", "tsg_ref"])]:
        args = m10.parse_args(base)
        args.experiments = exps
        args.configs = list(m10.CONFIGS)
        if key == "select":
            args.taus = a.select_taus if not a.quick else [12]
        elif a.quick:
            args.taus = [12]
        groups[key] = (args, m10.plan(args))
    return m10, groups


def run(m10, args, tasks):
    m10.run_task = _task13                       # run_all submits run_task
    try:
        m10.run_all(args, tasks)
    finally:
        m10.run_task = _STATE["run_task"]


# ============================================================
# REPORT
# ============================================================

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


def _load(m10, tasks):
    tasks = [t for t in tasks if Path(t["path"]).exists()]
    if not tasks:
        return pd.DataFrame(), pd.DataFrame()
    rounds = m10.collect(tasks)
    runs = m10.per_run_metrics(rounds)
    # Delta vs the myopic ILP: per_run_metrics pairs on (variant, tau, run);
    # ILP runs exist under 'base' only, so pair every variant with them
    ilp = runs[runs["config"] == ILP].set_index(["tau", "run"])["total_transplants"]
    ref = ilp.reindex(pd.MultiIndex.from_frame(runs[["tau", "run"]])).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        runs["pof_vs_myopic_ilp"] = np.where(
            ref > 0, 100.0 * (ref - runs["total_transplants"]) / ref, np.nan)
    secs = (rounds[rounds["cleared"].astype(bool)]
            .groupby(["config", "variant", "tau"])["seconds"].mean())
    runs["sec_round"] = [secs.get((c, v, t), np.nan) for c, v, t
                         in zip(runs["config"], runs["variant"], runs["tau"])]
    return rounds, runs


def _table(runs, order, tau):
    rows = []
    sub = runs[runs["tau"] == tau]
    for key in order:
        g = sub[(sub["config"] == key[0]) & (sub["variant"] == key[1])]
        if not len(g):
            continue
        row = {"configuration": NAMES.get(key, f"{key[0]}@{key[1]}"),
               "runs": int(g["run"].nunique())}
        for col, lab in METRICS:
            if col in g:
                row[lab] = f"{g[col].mean():.4g} ± {g[col].std():.3g}"
        row["s/round"] = f"{g['sec_round'].mean():.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def _tests(runs, pairs, tau):
    sub = runs[runs["tau"] == tau]
    comps = []
    for x, y in pairs:
        gx = sub[(sub["config"] == x[0]) & (sub["variant"] == x[1])].set_index("run")
        gy = sub[(sub["config"] == y[0]) & (sub["variant"] == y[1])].set_index("run")
        if not len(gx) or not len(gy):
            continue
        for col, lab in METRICS:
            if col in gx and col in gy:
                comps.append({"a": NAMES.get(x, str(x)), "b": NAMES.get(y, str(y)),
                              "metric": lab, **paired(gx[col], gy[col])})
    c = pd.DataFrame(comps)
    if len(c):
        k = c["p"].notna().sum()
        c["p_bonf"] = np.minimum(1.0, c["p"] * max(k, 1))
    return c


def _diag(rounds) -> Optional[pd.DataFrame]:
    if "tsg_searched" not in rounds:
        return None
    r = rounds[rounds["config"].isin(TSG) & rounds["cleared"].astype(bool)]
    if not len(r):
        return None
    rows = []
    for (c, v, tau), g in r.groupby(["config", "variant", "tau"]):
        s = g[g["tsg_searched"].astype(str).str.lower() == "true"]
        row = {"configuration": NAMES.get((c, v), f"{c}@{v}"), "tau": int(tau),
               "cleared rounds": len(g), "searched rounds": len(s)}
        if len(s):
            row.update({
                "mean front size": float(s["front_size"].mean()),
                "two-stage s/round": float(s["tsg_ts_sec"].mean()),
                "stage-2 time limit (sum)": float((s["tsg_status_sum"] == "time_limit").mean()),
                "stage-2 time limit (F3)": float((s["tsg_status_f3"] == "time_limit").mean())})
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(m10, groups, a) -> None:
    out = Path(a.results)
    md = ["# Module 13 — two-stage-ILP-guided NSGA-III", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} from {out}.", "",
          "TSG = NSGA-III + AOS seeded and repaired from the two-stage ILP ('sum') "
          "matching; volume-ILP and two-stage 'F3' matchings added to the initial "
          "population and, with the 'sum' matching, inserted into the final front. "
          "Implemented solution ('score' rule): among F1 >= rho x max F1, minimise "
          "F2/r2 + F3/r3 + F4/r4 with r_k from the volume-ILP matching (floor 0.01); "
          "rho = 0.9 unless stated. 'Old' = the framework of modules 10/11 "
          "(volume-ILP guide, floor + compromise rule). Delta = loss of transplants "
          "over the trajectory vs the myopic ILP on the same pool sequence.", ""]
    J: Dict = {"main": [], "main_tests": [], "select": [], "select_tests": [], "diag": []}

    md += ["## A. Main comparison", ""]
    rounds, runs = _load(m10, groups["main"][1])
    if not len(runs):
        md += ["not available (step main)", ""]
    else:
        order = [(c, "score90") for c in TSG] + [(c, "base") for c in OLD] + \
                [(TS_SUM, "base"), (TS_F3, "base"), (ILP, "base"), (LEX, "base")]
        for tau in sorted(runs["tau"].unique()):
            t, c = _table(runs, order, tau), _tests(runs, MAIN_PAIRS, tau)
            J["main"] += [{"tau": int(tau), **r} for r in t.to_dict("records")]
            J["main_tests"] += [{"tau": int(tau), **r} for r in c.round(6).to_dict("records")]
            md += [f"### tau = {tau}", "", _md(t), "",
                   "Paired Wilcoxon, Bonferroni over all rows of this table:", "",
                   _md(c.assign(p=c["p"].map(_p), p_bonf=c["p_bonf"].map(_p)))
                   if len(c) else "n/a", ""]
        d = _diag(rounds)
        if d is not None:
            J["diag"] = d.to_dict("records")
            md += ["### Per-round diagnostics of the guided search", "",
                   "Searched rounds = rounds with more than 2N matchings (smaller rounds "
                   "are solved exactly); the two-stage ILP is solved only in these. "
                   "Stage-2 limit: 60 s per objective (best matching found is used).",
                   "", _md(d), ""]

    md += ["## B. Selection rule and guide (tau = 12)", ""]
    _, sruns = _load(m10, groups["select"][1])
    if not len(sruns):
        md += ["not available (step select)", ""]
    else:
        for tau in sorted(set(sruns["tau"]) & set(a.select_taus if not a.quick else [12])):
            t, c = _table(sruns, SELECT_ROWS, tau), _tests(sruns, SELECT_PAIRS, tau)
            J["select"] += [{"tau": int(tau), **r} for r in t.to_dict("records")]
            J["select_tests"] += [{"tau": int(tau), **r} for r in c.round(6).to_dict("records")]
            md += [f"### tau = {tau}", "", _md(t), "",
                   "rho = 1.0 must reproduce the two-stage ILP (sanity check). Paired "
                   "Wilcoxon, Bonferroni over all rows:", "",
                   _md(c.assign(p=c["p"].map(_p), p_bonf=c["p_bonf"].map(_p)))
                   if len(c) else "n/a", ""]

    pdir = out / "paper"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "module13_tsguided.md").write_text("\n".join(md), encoding="utf-8")
    (pdir / "module13_tsguided.json").write_text(json.dumps(J, indent=2, default=str),
                                                 encoding="utf-8")
    print("\n".join(md))
    print(f"\nWritten: {pdir / 'module13_tsguided.md'} (send this file)")


def main(argv=None) -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="Module 13: two-stage-ILP-guided NSGA-III")
    ap.add_argument("--steps", nargs="+", default=STEPS, choices=STEPS)
    ap.add_argument("--taus", type=int, nargs="+", default=TAUS)
    ap.add_argument("--select-taus", type=int, nargs="+", default=SELECT_TAUS)
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--jobs", type=int, default=JOBS)
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--quick", action="store_true",
                    help="tiny budget in results_quick: checks the pipeline")
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args(argv)
    if a.quick and a.results == RESULTS:
        a.results = "results_quick"
    print("=" * 66 + "\nMODULE 13: two-stage-ILP-guided NSGA-III\n" + "=" * 66)
    m10, groups = build_args(a)
    if not a.analyse_only:
        for step in a.steps:
            args, tasks = groups[step]
            print(f"\nStep {step}: " + ", ".join(args.experiments))
            run(m10, args, tasks)
    write_report(m10, groups, a)
    print("\n" + "=" * 66 + "\nMODULE 13 COMPLETE\n" + "=" * 66)


if __name__ == "__main__":
    main()
