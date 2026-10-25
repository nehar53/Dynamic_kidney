import os

# one BLAS thread per worker (same as modules 9-15)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ============================================================
# MODULE 16: TRADE-OFF FRONT vs TWO-STAGE ILP (supervisor's points 2-3)
# ============================================================
#
# The proposed framework is UNCHANGED (volume-ILP-guided NSGA-III,
# modules 5-7). The two-stage ILP (module 6) is the benchmark at
# maximum volume. This module asks what the MOEA front offers BELOW
# maximum volume that one two-stage ILP solution cannot, and how close
# it is to the exact answer. Modules 1-15 are not changed.
#
# Volume levels rho in {1.0, 0.95, 0.90, 0.85, 0.80}; in a round with
# volume optimum F1*, level rho allows V_rho = ceil(rho * F1*)
# transplants or more. Fairness of a solution is reported as F2, F3,
# F4 and as the two-stage 'sum' score F2/r2 + F3/r3 + F4/r4 (r_k = the
# volume-ILP matching's values floored at 0.01; the volume-ILP matching
# itself scores 3 when no r_k is floored).
#
#   A  identical pools: the 30 initial pools of 75 vertices (seeds 42..,
#      as modules 11 and 12), one round
#   B  fixed trajectories: every round of the framework's own
#      trajectories (UCB1 myopic, 30 runs, tau = 6/12/18; identical to
#      the main experiment, checked), observed without changing them
#   C  selection rules over full trajectories: 90% floor (base), 100%,
#      80% and compromise at tau = 6/12/18 (module 10 'sens_select',
#      extended to all tau), with the two-stage ILP and the myopic ILP
#
# Per round / pool and level rho:
#   options    distinct front solutions with V_rho <= F1 <= F1*
#   best       lowest score, F2, F3, F4 among them
#   pick       the solution the floor rule would implement at rho
#   exact      two-stage ILP at exactly V_rho transplants (stage 2,
#              'sum', same refs; time limit, status recorded)
#   gap        MOEA best score among solutions with F1 = V_rho vs exact
#   beyond     front holds a solution with a lower score than the
#              two-stage ILP at maximum volume
# Ties: in exactly enumerated rounds the whole front is exact, so the
# number of maximum-volume matchings with the optimal score is exact:
# how often the deterministic tie-break actually chooses.
#
# Press Run (no arguments), or:
#   python module16_tradeoff.py --steps pools rounds
#   python module16_tradeoff.py --analyse-only
#   python module16_tradeoff.py --quick                (pipeline check)
# Resumable: finished pools / trajectories are kept on disk.

STEPS = ["pools", "rounds", "select"]
LEVELS = [1.0, 0.95, 0.90, 0.85, 0.80]
POOLS = 30
TAUS = [6, 12, 18]
RUNS = 30
POOL_TIME_LIMIT = 120               # A: per exact solve (75-vertex pools)
ROUND_TIME_LIMIT = 60               # B: as in module 12 trajectories
RESULTS = "results"
JOBS = max(1, (os.cpu_count() or 2) - 1)
FRAME = "dyn_nsga3_ucb1_myopic"
SCORE_FLOOR = 0.01

_STATE: Dict = {"ready": False, "rows": [], "time_limit": ROUND_TIME_LIMIT}


def _module10():
    try:
        import module10_experiment as m10
    except ImportError:
        import module10_experiments as m10
    return m10


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    """Atomic write with retries (OneDrive / antivirus locks)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    for wait in [0.5, 1, 2, 4, 8, 15, 30, 60]:
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(wait)
    raise PermissionError(f"{path} stays locked (pause OneDrive and re-run)")


# ============================================================
# ANALYSIS OF ONE FRONT (one round or one pool)
# ============================================================

def analyse_front(res, pairs, edges, structures, guide, time_limit: int) -> Dict:
    from module4_objectives import compute_all_objectives
    from module6_ilp_baseline import solve_ilp_two_stage
    from module7_clearing import select_solution
    ilp_set = set()
    for k in guide:
        ilp_set.update(structures.nodes[k])
    o = compute_all_objectives(ilp_set, pairs)
    f1s = int(round(o["F1"]))
    ref = (o["F2"], o["F3"], o["F4"])
    r = np.maximum(np.asarray(ref, float), SCORE_FLOOR)
    out = {"F1_star": f1s, "ilp_F2": o["F2"], "ilp_F3": o["F3"], "ilp_F4": o["F4"],
           "exact_front": bool(getattr(res, "exact", False)),
           "front_size": len(res.matched_sets)}
    F = np.atleast_2d(res.F) if len(res.matched_sets) else np.zeros((0, 4))
    f1 = -F[:, 0] if len(F) else np.zeros(0)
    score = (F[:, 1:4] / r).sum(axis=1) if len(F) else np.zeros(0)

    # exact two-stage ILP at each distinct volume level
    exact: Dict[int, Dict] = {}
    for v in sorted({max(1, math.ceil(rho * f1s - 1e-9)) for rho in LEVELS}, reverse=True):
        if f1s == 0:
            break
        t0 = time.time()
        M, info = solve_ilp_two_stage(pairs, edges, structures, v, ref, objective="sum",
                                      time_limit=time_limit)
        q = compute_all_objectives(M, pairs) if M else None
        ok = q is not None and int(round(q["F1"])) == v
        exact[v] = {"sec": time.time() - t0,
                    "status": info.get("status", "") if ok else "infeasible_or_failed",
                    "F2": q["F2"] if ok else np.nan, "F3": q["F3"] if ok else np.nan,
                    "F4": q["F4"] if ok else np.nan,
                    "score": float(sum(q[k] / rr for k, rr in zip(("F2", "F3", "F4"), r)))
                    if ok else np.nan}
    ts = exact.get(f1s, {})
    out.update({"ts_score": ts.get("score", np.nan), "ts_F2": ts.get("F2", np.nan),
                "ts_F3": ts.get("F3", np.nan), "ts_F4": ts.get("F4", np.nan),
                "ts_status": ts.get("status", ""), "ts_sec": ts.get("sec", np.nan),
                "exact_sec_total": float(sum(e["sec"] for e in exact.values()))})

    for rho in LEVELS:
        tag = f"{int(round(rho * 100))}"
        v = max(1, math.ceil(rho * f1s - 1e-9)) if f1s else 0
        inr = (f1 >= v - 1e-9) & (f1 > 0) if len(F) else np.zeros(0, bool)
        at = (np.abs(f1 - v) < 1e-9) & (f1 > 0) if len(F) else np.zeros(0, bool)
        row = {f"V_{tag}": v, f"opt_{tag}": int(inr.sum())}
        if inr.any():
            row.update({f"best_score_{tag}": float(score[inr].min()),
                        f"best_F2_{tag}": float(F[inr, 1].min()),
                        f"best_F3_{tag}": float(F[inr, 2].min()),
                        f"best_F4_{tag}": float(F[inr, 3].min()),
                        f"beyond_ts_{tag}": bool(score[inr].min() < out["ts_score"] - 1e-9)
                        if not np.isnan(out["ts_score"]) else np.nan})
            # options that beat the two-stage ILP on F2, F3 and F4 at once
            if not np.isnan(out["ts_score"]):
                t = np.array([out["ts_F2"], out["ts_F3"], out["ts_F4"]])
                G = F[inr, 1:4]
                dom = (G <= t + 1e-12).all(axis=1) & (G < t - 1e-12).any(axis=1)
                row[f"dominate_ts_{tag}"] = int(dom.sum())
            pick, _ = select_solution(res, "floor", rho)
            q = compute_all_objectives(pick, pairs)
            row.update({f"pick_F1_{tag}": q["F1"], f"pick_F2_{tag}": q["F2"],
                        f"pick_F3_{tag}": q["F3"], f"pick_F4_{tag}": q["F4"],
                        f"pick_score_{tag}": float(sum(q[k] / rr for k, rr in
                                                       zip(("F2", "F3", "F4"), r)))})
        if at.any():
            row[f"moea_at_score_{tag}"] = float(score[at].min())
        e = exact.get(v)
        if e:
            row.update({f"exact_score_{tag}": e["score"], f"exact_F2_{tag}": e["F2"],
                        f"exact_F3_{tag}": e["F3"], f"exact_F4_{tag}": e["F4"],
                        f"exact_status_{tag}": e["status"], f"exact_sec_{tag}": e["sec"]})
        out.update(row)

    # ties at maximum volume (exact only when the front is exact)
    mx = (np.abs(f1 - f1s) < 1e-9) & (f1 > 0) if len(F) else np.zeros(0, bool)
    if mx.any():
        best = score[mx].min()
        tied = mx & (np.abs(score - best) < 1e-9)
        Ft = F[tied, 1:4]
        out.update({"ties_n": int(tied.sum()),
                    "ties_differ": bool(len(Ft) > 1 and (np.ptp(Ft, axis=0) > 1e-12).any()),
                    "ties_range_F3": float(np.ptp(Ft[:, 1])) if len(Ft) > 1 else 0.0})
    return out


# ============================================================
# A. IDENTICAL POOLS
# ============================================================

def _pool_task(seed: int) -> Dict:
    from module1_generator import generate_initial_pool
    from module3_logistics import add_chains, build_compatibility_graph, build_exchange_structures
    from module5_nsga3_aos import UCB1AOS, build_context_vector, run_moea_round
    from module6_ilp_baseline import solve_ilp_max_transplants, solve_ilp_picef
    pool, _ = generate_initial_pool(n_pairs=75, seed=seed, initial_wait_max=6)
    P = pool.pairs
    e = build_compatibility_graph(P)
    s = build_exchange_structures(P, e)
    if s.truncated:
        add_chains(s, solve_ilp_picef(P, e)[1]["chains"], e)
    _, _, info = solve_ilp_max_transplants(P, e, s)
    t0 = time.time()
    res = run_moea_round(P, UCB1AOS(seed=seed), build_context_vector(P, 0, 1, 25, 0.0),
                         n_gen=_STATE.get("gen", 150), pop_size=_STATE.get("pop", 100),
                         seed=seed, edges=e, structures=s,
                         guide_structures=info["structure_indices"])
    row = {"seed": seed, "moea_sec": time.time() - t0}
    row.update(analyse_front(res, P, e, s, info["structure_indices"],
                             _STATE.get("pool_time_limit", POOL_TIME_LIMIT)))
    return row


def _pool_worker(args):
    seed, gen, pop, tl = args
    _STATE.update(gen=gen, pop=pop, pool_time_limit=tl)
    return _pool_task(seed)


def run_pools(a) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    out = Path(a.results) / "tradeoff16" / "pools.csv"
    done = set(pd.read_csv(out)["seed"].astype(int)) if out.exists() else set()
    todo = [sd for sd in range(42, 42 + a.pools) if sd not in done]
    print(f"\nA. Identical pools: {len(done)} done, {len(todo)} to run")
    rows = [pd.read_csv(out)] if out.exists() else []

    def save(row):
        rows.append(pd.DataFrame([row]))
        _save_csv(pd.concat(rows, ignore_index=True), out)

    jobs = [(sd, a.gen, a.pop, a.pool_time_limit) for sd in todo]
    if a.jobs > 1 and todo:
        with ProcessPoolExecutor(max_workers=a.jobs, mp_context=get_context("spawn")) as ex:
            futs = {ex.submit(_pool_worker, j): j[0] for j in jobs}
            for k, f in enumerate(as_completed(futs), 1):
                save(f.result())
                print(f"  [{k}/{len(todo)}] pool {futs[f]} done", flush=True)
    else:
        for k, j in enumerate(jobs, 1):
            save(_pool_worker(j))
            print(f"  [{k}/{len(todo)}] pool {j[0]} done", flush=True)


# ============================================================
# B. FIXED TRAJECTORIES (observe every round, change nothing)
# ============================================================

def _observed_round(pool_pairs, aos=None, context=None, **kw):
    """Wraps module7.run_moea_round: same round, then analyse its front."""
    res = _STATE["run_round"](pool_pairs, aos, context, **kw)
    guide = kw.get("guide_structures")
    if guide is not None:
        row = analyse_front(res, pool_pairs, kw["edges"], kw["structures"], guide,
                            _STATE["time_limit"])
        row["call"] = len(_STATE["rows"]) + 1
        _STATE["rows"].append(row)
    return res


def _round_task(task: Dict):
    m10 = _module10()
    if not _STATE["ready"]:
        import module7_clearing as m7
        _STATE.update(ready=True, run_round=m7.run_moea_round)
        m7.run_moea_round = _observed_round
    _STATE["rows"] = []
    _STATE["time_limit"] = task["time_limit"]
    path, secs = m10.run_task(task)            # trajectory file (same as main run)
    traj = pd.read_csv(path)
    cl = traj[traj["cleared"].astype(bool)].reset_index(drop=True)
    d = pd.DataFrame(_STATE["rows"])
    if len(d) == len(cl):
        d.insert(0, "round", cl["round"].to_numpy())
        d.insert(1, "F1_impl", cl["F1"].to_numpy())
        d.insert(2, "sec_round", cl["seconds"].to_numpy())
    d.insert(0, "tau", task["tau"])
    d.insert(1, "run", task["run"])
    _save_csv(d, Path(task["rounds_path"]))
    return path, secs


def round_tasks(a, m10) -> List[Dict]:
    out = Path(a.results) / "tradeoff16"
    tasks = []
    for run in range(a.runs):
        for tau in a.taus:
            tasks.append({
                "experiment": "tradeoff16", "variant": "base", "config": FRAME,
                "tau": tau, "run": run, "rounds": a.rounds, "gen": a.gen, "pop": a.pop,
                "initial_wait_max": m10.INITIAL_WAIT_MAX, "time_limit": a.round_time_limit,
                "path": str(out / "traj" / f"tau{tau}__run{run:03d}.csv"),
                "main_path": str(m10.raw_path(Path(a.results), "base", FRAME, tau, run)),
                "rounds_path": str(out / "rounds" / f"tau{tau}__run{run:03d}.csv")})
    return tasks


def run_rounds(a, m10, tasks) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    for t in tasks:
        Path(t["path"]).parent.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks if not Path(t["rounds_path"]).exists()]
    print(f"\nB. Fixed trajectories: {len(tasks) - len(todo)}/{len(tasks)} done, "
          f"{len(todo)} to run on {a.jobs} process(es)")
    t0 = time.time()
    if a.jobs > 1 and todo:
        with ProcessPoolExecutor(max_workers=a.jobs, mp_context=get_context("spawn")) as ex:
            futs = {ex.submit(_round_task, t): t for t in todo}
            for k, f in enumerate(as_completed(futs), 1):
                _, secs = f.result()
                t = futs[f]
                print(f"  [{k}/{len(todo)}] tau={t['tau']} run={t['run']} {secs:.0f}s  "
                      f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    else:
        for k, t in enumerate(todo, 1):
            _, secs = _round_task(t)
            print(f"  [{k}/{len(todo)}] tau={t['tau']} run={t['run']} {secs:.0f}s", flush=True)


# ============================================================
# C. SELECTION RULES OVER FULL TRAJECTORIES (module 10 runs)
# ============================================================

def select_args(a, m10):
    m10.EXPERIMENTS["tradeoff16_select"] = {
        "variants": ["base", "floor100", "floor80", "compromise"],
        "configs": [FRAME, "myopic_ilp", "twostage_ilp_sum"], "taus": a.taus,
        "runs": a.runs, "about": "selection rules vs two-stage ILP (module 16)"}
    base = ["--out", a.results, "--jobs", str(a.jobs)]
    if a.quick:
        base += ["--rounds", str(a.rounds), "--gen", str(a.gen), "--pop", str(a.pop)]
    args = m10.parse_args(base)
    args.experiments = ["tradeoff16_select"]
    args.configs = [FRAME, "myopic_ilp", "twostage_ilp_sum"]
    args.runs = a.runs
    return args, m10.plan(args)


# ============================================================
# REPORT
# ============================================================

def _p(p):
    return "n/a" if p is None or (isinstance(p, float) and np.isnan(p)) else \
        ("<0.001" if p < 0.001 else f"{p:.3f}")


def _md(df: pd.DataFrame) -> str:
    if df is None or not len(df):
        return "n/a"
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(
            f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
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


def _bonf(df: pd.DataFrame) -> pd.DataFrame:
    if len(df):
        k = df["p"].notna().sum()
        df["p_bonf"] = np.minimum(1.0, df["p"] * max(k, 1))
        df = df.assign(p=df["p"].map(_p), p_bonf=df["p_bonf"].map(_p))
    return df


def _mean(df, col):
    return float(df[col].mean()) if col in df and df[col].notna().any() else np.nan


def _share(df, col):
    if col not in df or not df[col].notna().any():
        return np.nan
    return float(df[col].dropna().astype(str).str.lower().isin(["true", "1", "1.0"]).mean())


def level_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per volume level over the given units (pools or rounds)."""
    rows = []
    for rho in LEVELS:
        tag = f"{int(round(rho * 100))}"
        sub = df[df["F1_star"] > 0]
        below = sub[sub[f"V_{tag}"] < sub["F1_star"]] if f"V_{tag}" in sub else sub.iloc[0:0]
        u = sub if rho == 1.0 else below
        ex = u[f"exact_status_{tag}"] if f"exact_status_{tag}" in u else pd.Series(dtype=str)
        cmp_ = u.dropna(subset=[c for c in (f"moea_at_score_{tag}", f"exact_score_{tag}")
                                if c in u]) if f"moea_at_score_{tag}" in u else u.iloc[0:0]
        rows.append({
            "level": f"{int(rho * 100)}%",
            "units": len(u),
            "share with V<F1*": (len(below) / len(sub)) if rho < 1 and len(sub) else np.nan,
            "options (mean)": _mean(u, f"opt_{tag}"),
            "MOEA best score": _mean(u, f"best_score_{tag}"),
            "MOEA best F2": _mean(u, f"best_F2_{tag}"),
            "MOEA best F3": _mean(u, f"best_F3_{tag}"),
            "MOEA best F4": _mean(u, f"best_F4_{tag}"),
            "rule pick score": _mean(u, f"pick_score_{tag}"),
            "two-stage @max score": _mean(u, "ts_score"),
            "share front beyond two-stage": _share(u, f"beyond_ts_{tag}"),
            "share front dominates two-stage": float((u[f"dominate_ts_{tag}"] > 0).mean())
            if f"dominate_ts_{tag}" in u and len(u) else np.nan,
            "exact @V score": _mean(u, f"exact_score_{tag}"),
            "MOEA @V coverage": float(u[f"moea_at_score_{tag}"].notna().mean())
            if f"moea_at_score_{tag}" in u and len(u) else np.nan,
            "MOEA @V score (covered)": _mean(cmp_, f"moea_at_score_{tag}"),
            "exact @V score (covered)": _mean(cmp_, f"exact_score_{tag}"),
            "exact time-limit share": float((ex == "time_limit").mean()) if len(ex) else np.nan,
            "exact s/solve": _mean(u, f"exact_sec_{tag}"),
        })
    return pd.DataFrame(rows)


def report_pools(a, md, J) -> None:
    md += ["## A. Identical pools (one round, 75-vertex pools)", ""]
    p = Path(a.results) / "tradeoff16" / "pools.csv"
    if not p.exists():
        md += ["not available (step pools)", ""]
        return
    df = pd.read_csv(p)
    t = level_table(df)
    J["pools"] = t.to_dict("records")
    md += [f"{len(df)} pools; MOEA {df['moea_sec'].mean():.1f} s per round; exact two-stage "
           f"at maximum volume {df['ts_sec'].mean():.1f} s (time-limit share "
           f"{(df['ts_status'] == 'time_limit').mean():.2f}, limit {a.pool_time_limit} s).", "",
           "Levels below 100% use only pools where the level allows fewer transplants than "
           "F1* (V < F1*). 'beyond two-stage': the front holds a solution with a lower score "
           "than the two-stage ILP at maximum volume; 'dominates': lower or equal on F2, F3 "
           "and F4 at once. '@V': at exactly V transplants (MOEA best on the front vs exact "
           "two-stage ILP at that volume); 'covered' = units where the front has a solution "
           "at exactly V.", "", _md(t), ""]


def report_rounds(a, m10, tasks, md, J) -> None:
    md += ["## B. Fixed trajectories (every round of the framework's own trajectories)", ""]
    files = [t for t in tasks if Path(t["rounds_path"]).exists()]
    if not files:
        md += ["not available (step rounds)", ""]
        return
    same = []
    for t in files:
        mp, tp = Path(t["main_path"]), Path(t["path"])
        if mp.exists() and tp.exists():
            x, y = pd.read_csv(mp), pd.read_csv(tp)
            same.append(len(x) == len(y) and np.allclose(x["F1"].fillna(-1), y["F1"].fillna(-1)))
    df = pd.concat([pd.read_csv(t["rounds_path"]) for t in files], ignore_index=True)
    md += [f"Trajectories identical to the main experiment (same seed): {sum(same)}/{len(same)}.",
           ""]
    J["rounds"] = {}
    for tau in sorted(df["tau"].unique()):
        sub = df[df["tau"] == tau]
        t = level_table(sub)
        J["rounds"][int(tau)] = t.to_dict("records")
        md += [f"### tau = {tau}: {len(sub)} rounds in {sub['run'].nunique()} runs "
               f"({sub['exact_front'].astype(bool).mean():.2f} exactly enumerated)", "",
               _md(t), ""]
        # per run: best fairness the front offers at 90 / 80% vs the two-stage ILP
        comps = []
        per = sub.groupby("run")
        for tag in ("90", "80"):
            if f"best_score_{tag}" not in sub:
                continue
            m = per[[f"best_score_{tag}", "ts_score", f"best_F3_{tag}", "ts_F3",
                     f"best_F2_{tag}", "ts_F2", f"best_F4_{tag}", "ts_F4"]].mean()
            for met, ts in [("score", "ts_score"), ("F2", "ts_F2"), ("F3", "ts_F3"), ("F4", "ts_F4")]:
                comps.append({"a": f"front best at {tag}%", "b": "two-stage ILP at max volume",
                              "metric": met, **paired(m[f"best_{met}_{tag}"], m[ts])})
        c = _bonf(pd.DataFrame(comps))
        J["rounds"][f"{int(tau)}_tests"] = c.to_dict("records")
        md += ["Per-run means over all rounds (rounds where the level allows no fewer "
               "transplants count with their maximum-volume value). Paired Wilcoxon, "
               "Bonferroni over the rows:", "", _md(c), ""]
    # ties
    ex = df[df["exact_front"].astype(bool) & df["ties_n"].notna()]
    if len(ex):
        tie = {"exact rounds": len(ex),
               "share with >1 optimal maximum matching": float((ex["ties_n"] > 1).mean()),
               "share where tied optima differ in F2/F3/F4": float(
                   ex["ties_differ"].astype(str).str.lower().eq("true").mean()),
               "mean F3 range among tied optima": float(ex["ties_range_F3"].mean())}
        J["ties"] = tie
        md += ["### Ties at maximum volume (exactly enumerated rounds: the front is exact)", "",
               "How often several maximum-volume matchings share the optimal two-stage score, "
               "so the deterministic tie-break (highest HLA weight, module 6) decides which is "
               "implemented, and whether the tied matchings differ in the reported F2/F3/F4.",
               "", _md(pd.DataFrame([tie])), ""]
    cost = pd.DataFrame([{
        "MOEA s/round (framework, incl. ILP)": _mean(df, "sec_round"),
        "two-stage at max volume s/round": _mean(df, "ts_sec"),
        "two-stage time-limit share": float((df["ts_status"] == "time_limit").mean()),
        "all exact level solves s/round": _mean(df, "exact_sec_total")}])
    J["rounds_cost"] = cost.to_dict("records")
    md += ["### Cost per round", "", _md(cost), ""]


def report_select(a, m10, stasks, md, J) -> None:
    md += ["## C. Selection rules over full trajectories", ""]
    have = [t for t in stasks if Path(t["path"]).exists()]
    if not have:
        md += ["not available (step select)", ""]
        return
    rounds = m10.collect(have)
    runs = m10.per_run_metrics(rounds)
    # ILP runs are reused for every variant (same file): keep the base copy
    ilp = runs[(runs["config"] == "myopic_ilp") & (runs["variant"] == "base")] \
        .set_index(["tau", "run"])["total_transplants"]
    ref = ilp.reindex(pd.MultiIndex.from_frame(runs[["tau", "run"]])).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        runs["delta"] = np.where(ref > 0, 100 * (ref - runs["total_transplants"]) / ref, np.nan)
    names = {(FRAME, "floor100"): "Framework, 100% floor", (FRAME, "base"): "Framework, 90% floor",
             (FRAME, "floor80"): "Framework, 80% floor", (FRAME, "compromise"): "Framework, compromise",
             ("twostage_ilp_sum", "base"): "Two-stage ILP (sum)", ("myopic_ilp", "base"): "Myopic ILP"}
    mets = [("total_transplants", "Transplants"), ("total_high_cpra_transplants", "High-cPRA tx"),
            ("mean_F2", "F2"), ("mean_F3", "F3"), ("mean_F4", "F4"), ("pof_mean", "PoF (%)"),
            ("delta", "Delta (%)"), ("total_withdrawn_high_cpra", "High-cPRA withdrawn")]
    J["select"] = {}
    for tau in sorted(runs["tau"].unique()):
        sub = runs[runs["tau"] == tau]
        rows, comps = [], []
        for key, nm in names.items():
            g = sub[(sub["config"] == key[0]) & (sub["variant"] == key[1])]
            if not len(g):
                continue
            row = {"configuration": nm, "runs": len(g)}
            for col, lab in mets:
                row[lab] = f"{g[col].mean():.4g} ± {g[col].std():.3g}"
            rows.append(row)
        ts = sub[(sub["config"] == "twostage_ilp_sum") & (sub["variant"] == "base")].set_index("run")
        for var in ("floor100", "base", "floor80", "compromise"):
            g = sub[(sub["config"] == FRAME) & (sub["variant"] == var)].set_index("run")
            if not len(g) or not len(ts):
                continue
            for col, lab in mets:
                comps.append({"a": names[(FRAME, var)], "b": "Two-stage ILP (sum)", "metric": lab,
                              **paired(g[col], ts[col])})
        c = _bonf(pd.DataFrame(comps))
        J["select"][int(tau)] = {"table": rows, "tests": c.to_dict("records")}
        md += [f"### tau = {tau}", "", _md(pd.DataFrame(rows)), "",
               "Paired Wilcoxon vs the two-stage ILP, Bonferroni over all rows:", "", _md(c), ""]


def write_report(a, m10, rtasks, stasks) -> None:
    md = ["# Module 16 — trade-off front vs two-stage ILP", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} from {a.results}.", "",
          "Framework unchanged (volume-ILP-guided NSGA-III, UCB1, N = 100, G = 150). "
          "Two-stage ILP (module 6): stage 1 maximises transplants F1*; stage 2 fixes the "
          "volume and minimises the score F2/r2 + F3/r3 + F4/r4, r_k = values of the "
          "volume-ILP matching floored at 0.01; ties broken by a 1e-7 weight on HLA quality. "
          "Here stage 2 is also solved at fixed lower volumes V = ceil(rho F1*) to give the "
          "exact reference at each level. All results are synthetic simulation informed by "
          "public ANZKX information.", ""]
    J: Dict = {}
    report_pools(a, md, J)
    report_rounds(a, m10, rtasks, md, J)
    report_select(a, m10, stasks, md, J)
    pdir = Path(a.results) / "paper"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "module16_tradeoff.md").write_text("\n".join(md), encoding="utf-8")
    (pdir / "module16_tradeoff.json").write_text(json.dumps(J, indent=2, default=str),
                                                 encoding="utf-8")
    print("\n".join(md))
    print(f"\nWritten: {pdir / 'module16_tradeoff.md'} (send this file)")


def main(argv=None) -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="Module 16: trade-off front vs two-stage ILP")
    ap.add_argument("--steps", nargs="+", default=STEPS, choices=STEPS)
    ap.add_argument("--taus", type=int, nargs="+", default=TAUS)
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--pools", type=int, default=POOLS)
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--gen", type=int, default=150)
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--pool-time-limit", type=int, default=POOL_TIME_LIMIT)
    ap.add_argument("--round-time-limit", type=int, default=ROUND_TIME_LIMIT)
    ap.add_argument("--jobs", type=int, default=JOBS)
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--quick", action="store_true", help="tiny budget in results_quick")
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args(argv)
    if a.quick:
        a.results, a.runs, a.pools, a.rounds, a.gen, a.pop = "results_quick", 2, 2, 4, 10, 40
        a.taus, a.pool_time_limit, a.round_time_limit = [12], 20, 20
    print("=" * 66 + "\nMODULE 16: trade-off front vs two-stage ILP\n" + "=" * 66)
    m10 = _module10()
    rtasks = round_tasks(a, m10)
    sargs, stasks = select_args(a, m10)
    if not a.analyse_only:
        if "pools" in a.steps:
            run_pools(a)
        if "rounds" in a.steps:
            run_rounds(a, m10, rtasks)
        if "select" in a.steps:
            print("\nC. Selection rules over full trajectories (module 10 runs; "
                  "finished ones are reused)")
            m10.run_all(sargs, stasks)
    write_report(a, m10, rtasks, stasks)
    print("\n" + "=" * 66 + "\nMODULE 16 COMPLETE\n" + "=" * 66)


if __name__ == "__main__":
    main()
