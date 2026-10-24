import os

# one BLAS thread per worker (same as modules 9-13)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ============================================================
# MODULE 14: TWO-STAGE-ILP-GUIDED NSGA-III ON PREFLIB 00036
# ============================================================
#
# The out-of-distribution check of module 9, repeated for the
# two-stage-ILP-guided framework (TSG) of module 13. Same instances,
# instance scaling, arrival order, seeds and settings as module 9's
# preset "main" (12 instances of 128-147 vertices, 10 runs, tau = 12,
# 25 rounds, G = 150), so every run is paired with the saved
# module 9 runs of the old framework, the myopic ILP and Dickerson lex.
# Modules 1-13 are NOT changed: TSG is switched on inside each worker
# exactly as in module 13.
#
# New trajectories (written to results/ood_tsg/raw):
#   tsg_nsga3_ucb1_myopic         TSG, score rule, rho = 0.9
#   tsg_nsga3_linucb_myopic       TSG, score rule, rho = 0.9
#   tsg_nsga3_ucb1_batched_mi3    TSG, batched with a 3-round maximum
#                                 interval (as module 9's batched run)
#   tsg_nsga3_ucb1_myopic_rho80   TSG, score rule, rho = 0.8
#   twostage_ilp_sum / _f3        two-stage ILP baselines (module 12)
# Reused from module 9 (results/ood_main/raw): dyn_nsga3_* (old
# framework), myopic_ilp, dickerson_lex. Missing ones are run here.
#
# F4 is not reported: on PrefLib the hospitals are random (module 9).
#
# Press Run (no arguments): the PrefLib folder is found automatically
# (the folder with .wmd files used by module 9), or:
#   python module14_tsg_preflib.py --wmd <PrefLib folder>
#   python module14_tsg_preflib.py --analyse-only
#   python module14_tsg_preflib.py --quick      (pipeline check)
# Resumable: finished trajectories are kept on disk.

OOD_MAIN = "results/ood_main"        # module 9 preset "main" / "batched"
OUT = "results/ood_tsg"
MIN_SIZE, MAX_SIZE, PER_SIZE = 128, 147, 3   # = module 9 preset "main"
RUNS, ROUNDS, GEN, TAU = 10, 25, 150, 12
JOBS = max(1, (os.cpu_count() or 2) - 1)

TSG_CFGS: Dict[str, Dict] = {
    "tsg_nsga3_ucb1_myopic": {"method": "nsga3", "aos": "ucb1", "clearing": "myopic",
                              "dynamic": True, "ts_guided": True, "rho": 0.9},
    "tsg_nsga3_linucb_myopic": {"method": "nsga3", "aos": "linucb", "clearing": "myopic",
                                "dynamic": True, "ts_guided": True, "rho": 0.9},
    "tsg_nsga3_ucb1_batched_mi3": {"method": "nsga3", "aos": "ucb1", "clearing": "batched",
                                   "dynamic": True, "batch_max_interval": 3,
                                   "ts_guided": True, "rho": 0.9},
    "tsg_nsga3_ucb1_myopic_rho80": {"method": "nsga3", "aos": "ucb1", "clearing": "myopic",
                                    "dynamic": True, "ts_guided": True, "rho": 0.8},
    "twostage_ilp_sum": {"method": "ilp2", "objective": "sum", "clearing": "myopic"},
    "twostage_ilp_f3": {"method": "ilp2", "objective": "f3", "clearing": "myopic"},
}
REF_CFGS = ["dyn_nsga3_ucb1_myopic", "dyn_nsga3_linucb_myopic",
            "dyn_nsga3_ucb1_batched_mi3", "myopic_ilp", "dickerson_lex"]
for _n, _c in TSG_CFGS.items():
    _c["name"] = _n

NAMES = {
    "tsg_nsga3_ucb1_myopic": "TSG UCB1 myopic",
    "tsg_nsga3_linucb_myopic": "TSG LinUCB myopic",
    "tsg_nsga3_ucb1_batched_mi3": "TSG UCB1 batched (max int. 3)",
    "tsg_nsga3_ucb1_myopic_rho80": "TSG UCB1 myopic, rho=0.8",
    "dyn_nsga3_ucb1_myopic": "Old UCB1 myopic",
    "dyn_nsga3_linucb_myopic": "Old LinUCB myopic",
    "dyn_nsga3_ucb1_batched_mi3": "Old UCB1 batched (max int. 3)",
    "twostage_ilp_sum": "Two-stage ILP (sum)",
    "twostage_ilp_f3": "Two-stage ILP (F3)",
    "myopic_ilp": "Myopic ILP",
    "dickerson_lex": "Dickerson lex",
}
ORDER = list(NAMES)
METRICS = [("total_transplants", "Transplants"),
           ("total_high_cpra_transplants", "High-cPRA tx"),
           ("mean_F2", "F2"), ("mean_F3", "F3"),
           ("pof_mean", "PoF (%)"), ("delta", "Delta (%)"),
           ("total_withdrawn_high_cpra", "High-cPRA withdrawn")]
PAIRS = [("tsg_nsga3_ucb1_myopic", "dyn_nsga3_ucb1_myopic"),
         ("tsg_nsga3_linucb_myopic", "dyn_nsga3_linucb_myopic"),
         ("tsg_nsga3_ucb1_myopic", "twostage_ilp_sum"),
         ("tsg_nsga3_ucb1_myopic", "twostage_ilp_f3"),
         ("tsg_nsga3_ucb1_myopic", "myopic_ilp"),
         ("tsg_nsga3_ucb1_myopic", "dickerson_lex"),
         ("tsg_nsga3_ucb1_myopic", "tsg_nsga3_ucb1_batched_mi3"),
         ("tsg_nsga3_ucb1_batched_mi3", "dyn_nsga3_ucb1_batched_mi3"),
         ("tsg_nsga3_ucb1_myopic_rho80", "twostage_ilp_sum")]


# ============================================================
# TASKS (same construction as module 9's run_preflib_ood)
# ============================================================

def find_wmd(start: Path) -> Optional[Path]:
    """Folder with the most .wmd files below the current folder."""
    best, n_best = None, 0
    for d in {p.parent for p in start.rglob("*.wmd")}:
        n = len(list(d.glob("*.wmd")))
        if n > n_best:
            best, n_best = d, n
    return best


def make_tasks(a) -> List[Dict]:
    import module9_preflib as m9
    from module7_clearing import BATCH_THRESHOLD
    files = m9.wmd_files(a.wmd, a.min_size, a.max_size, None, per_size=a.per_size)
    print(f"{len(files)} PrefLib instance(s) from {a.wmd}")
    configs = [TSG_CFGS[c] for c in TSG_CFGS] + [m9.ALL_CONFIGS[c] for c in REF_CFGS]
    tasks = []
    for f in files:
        n = m9.instance_size(f)
        # module 9, scale_all=True: half the vertices start in the pool
        init = max(2, n // 2)
        lam = max(0.5, (n - init) / a.rounds)
        n_star = max(3, int(round(BATCH_THRESHOLD / m9.OOD_INITIAL_POOL * init)))
        for run in range(a.runs):
            for cfg in configs:
                ref = cfg["name"] in REF_CFGS
                root = Path(a.ood_main if ref else a.out) / "raw"
                tasks.append({
                    "file": str(f), "cfg": cfg, "tau": TAU, "run": run,
                    "init": init, "lam": lam, "n_star": n_star,
                    "n_rounds": a.rounds, "n_gen": a.gen,
                    "selection_rule": "score" if cfg.get("ts_guided") else "floor",
                    "min_f1_ratio": cfg.get("rho", 0.9), "verbose": False,
                    "path": str(root / f"{f.stem}__{cfg['name']}__tau{TAU}__run{run:03d}.csv")})
    return tasks


# ============================================================
# WORKER: module 9 trajectory with TSG switched on (module 13)
# ============================================================

def _task14(task: Dict):
    import module13_tsguided as m13
    import module9_preflib as m9
    m13._setup()
    m13._STATE["guided"] = bool(task["cfg"].get("ts_guided", False))
    m13._STATE["diag"] = []
    path, secs = m9._ood_task(task)
    if m13._STATE["guided"]:                    # per-round diagnostics
        df = pd.read_csv(path)
        idx = df.index[df["cleared"].astype(bool)]
        d = pd.DataFrame(m13._STATE["diag"])
        if len(d) == len(idx):
            for col in d.columns:
                df.loc[idx, col] = d[col].to_numpy()
            tmp = Path(path).with_suffix(".tmp")
            df.to_csv(tmp, index=False)
            # OneDrive / antivirus can lock a file just written: retry
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


def run_tasks(tasks: List[Dict], jobs: int) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    for t in tasks:
        Path(t["path"]).parent.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks if not Path(t["path"]).exists()]
    todo.sort(key=lambda t: t["cfg"]["method"] in ("nsga3", "nsga2"))   # ILPs first
    n_ref = sum(t["cfg"]["name"] in REF_CFGS for t in todo)
    print(f"Trajectories: {len(tasks) - len(todo)}/{len(tasks)} saved, {len(todo)} to run "
          f"on {jobs} process(es) ({n_ref} module 9 reference runs missing)")
    if n_ref:
        print("  NOTE: module 9 reference runs not found in --ood-main; they are run "
              "now (check the folder if you expected them to exist)")
    t0 = time.time()

    def report(k, t, secs):
        print(f"[{k}/{len(todo)}] {Path(t['file']).stem} {t['cfg']['name']:<28} "
              f"run={t['run']:<2} {secs:6.0f}s  elapsed {(time.time() - t0) / 60:.1f} min",
              flush=True)

    if jobs > 1 and todo:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("spawn")) as ex:
            futs = {ex.submit(_task14, t): t for t in todo}
            for k, f in enumerate(as_completed(futs), 1):
                _, secs = f.result()
                report(k, futs[f], secs)
    else:
        for k, t in enumerate(todo, 1):
            _, secs = _task14(t)
            report(k, t, secs)


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


def per_run(tasks: List[Dict]) -> (pd.DataFrame, pd.DataFrame):
    frames = []
    for t in tasks:
        if Path(t["path"]).exists():
            df = pd.read_csv(t["path"])
            df["config"] = t["cfg"]["name"]       # rho80 runs carry their own name
            frames.append(df)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    rounds = pd.concat(frames, ignore_index=True)
    rounds["cleared"] = rounds["cleared"].astype(bool)
    rows = []
    for (ds, cfg, run), g in rounds.groupby(["dataset", "config", "run"]):
        c = g[g["cleared"]]
        rows.append({
            "dataset": ds, "config": cfg, "run": run, "pair_id": f"{ds}|{run}",
            "total_transplants": c["F1"].sum() if len(c) else 0,
            "total_F1_ILP": c["F1_ILP"].sum() if len(c) else 0,
            "total_high_cpra_transplants": c["high_cpra_matched"].sum() if len(c) else 0,
            "mean_F2": c["F2"].mean() if len(c) else np.nan,
            "mean_F3": c["F3"].mean() if len(c) else np.nan,
            "pof_mean": c["PoF"].mean() if len(c) else np.nan,
            "total_withdrawn_high_cpra": g["withdrawn_high_cpra"].sum()
            if "withdrawn_high_cpra" in g else np.nan,
            "sec_round": c["seconds"].mean() if len(c) else np.nan,
            "share_exact": c["exact_front"].astype(bool).mean()
            if "exact_front" in c and len(c) else np.nan})
    runs = pd.DataFrame(rows)
    ilp = runs[runs["config"] == "myopic_ilp"].set_index("pair_id")["total_transplants"]
    ref = runs["pair_id"].map(ilp)
    runs["delta"] = np.where(ref > 0, 100.0 * (ref - runs["total_transplants"]) / ref, np.nan)
    return rounds, runs


def write_report(tasks: List[Dict], a) -> None:
    md = ["# Module 14 — two-stage-ILP-guided NSGA-III on PrefLib 00036", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Instances and settings of module 9 "
          f"(preset main): sizes {a.min_size}-{a.max_size}, {a.per_size} per size, "
          f"{a.runs} runs, tau = {TAU}, {a.rounds} rounds, G = {a.gen}. Paired unit = "
          "(instance, run). TSG as in module 13 (score rule, rho = 0.9 unless stated); "
          "'Old' = module 9 runs of the original framework. Delta = transplants lost over "
          "the trajectory vs the myopic ILP. F4 is not reported (random hospitals).", ""]
    J: Dict = {"table": [], "tests": [], "diag": []}
    rounds, runs = per_run(tasks)
    if not len(runs):
        md += ["no trajectories found", ""]
    else:
        rows = []
        for c in ORDER:
            g = runs[runs["config"] == c]
            if not len(g):
                continue
            row = {"configuration": NAMES[c], "runs": len(g)}
            for col, lab in METRICS:
                row[lab] = f"{g[col].mean():.4g} ± {g[col].std():.3g}"
            row["exact rounds"] = f"{g['share_exact'].mean():.2f}" \
                if g["share_exact"].notna().any() else "n/a"
            row["s/round"] = f"{g['sec_round'].mean():.2f}"
            rows.append(row)
            J["table"].append({"config": c, **{lab: float(g[col].mean()) for col, lab in METRICS},
                               "seconds_per_round": float(g["sec_round"].mean())})
        comps = []
        for x, y in PAIRS:
            gx = runs[runs["config"] == x].set_index("pair_id")
            gy = runs[runs["config"] == y].set_index("pair_id")
            if not len(gx) or not len(gy):
                continue
            for col, lab in METRICS:
                comps.append({"a": NAMES[x], "b": NAMES[y], "metric": lab,
                              **paired(gx[col], gy[col])})
        cmp_ = pd.DataFrame(comps)
        if len(cmp_):
            k = cmp_["p"].notna().sum()
            cmp_["p_bonf"] = np.minimum(1.0, cmp_["p"] * max(k, 1))
            J["tests"] = cmp_.round(6).to_dict("records")
        md += ["## Results", "", _md(pd.DataFrame(rows)), "",
               "Paired Wilcoxon over (instance, run), Bonferroni over all rows:", "",
               _md(cmp_.assign(p=cmp_["p"].map(_p), p_bonf=cmp_["p_bonf"].map(_p)))
               if len(cmp_) else "n/a", ""]
        if "tsg_searched" in rounds:
            r = rounds[rounds["config"].str.startswith("tsg") & rounds["cleared"]]
            drows = []
            for c, g in r.groupby("config"):
                s = g[g["tsg_searched"].astype(str).str.lower() == "true"]
                row = {"configuration": NAMES[c], "cleared rounds": len(g),
                       "searched rounds": len(s)}
                if len(s):
                    row.update({"two-stage s/round": float(s["tsg_ts_sec"].mean()),
                                "stage-2 time limit (sum)":
                                    float((s["tsg_status_sum"] == "time_limit").mean())})
                drows.append(row)
            J["diag"] = drows
            md += ["### Guided search", "",
                   "Searched rounds = more than 2N matchings; smaller rounds are solved "
                   "exactly (the two-stage ILP is solved only in searched rounds).", "",
                   _md(pd.DataFrame(drows)), ""]
        ok = bool((rounds.loc[rounds["cleared"], "F1"]
                   <= rounds.loc[rounds["cleared"], "F1_ILP"] + 1e-9).all())
        md += [f"No configuration exceeds the ILP ceiling: {ok}", ""]
    pdir = Path(a.results) / "paper"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "module14_tsg_preflib.md").write_text("\n".join(md), encoding="utf-8")
    (pdir / "module14_tsg_preflib.json").write_text(json.dumps(J, indent=2, default=str),
                                                    encoding="utf-8")
    print("\n".join(md))
    print(f"\nWritten: {pdir / 'module14_tsg_preflib.md'} (send this file)")


def main(argv=None) -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="Module 14: TSG on PrefLib 00036")
    ap.add_argument("--wmd", default=None, help="PrefLib 00036 folder (default: found)")
    ap.add_argument("--ood-main", default=OOD_MAIN, help="module 9 output folder")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--gen", type=int, default=GEN)
    ap.add_argument("--min-size", type=int, default=MIN_SIZE)
    ap.add_argument("--max-size", type=int, default=MAX_SIZE)
    ap.add_argument("--per-size", type=int, default=PER_SIZE)
    ap.add_argument("--jobs", type=int, default=JOBS)
    ap.add_argument("--quick", action="store_true", help="tiny budget, separate folders")
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args(argv)
    if a.quick:
        a.runs, a.rounds, a.gen, a.per_size = 2, 4, 10, 1
        a.out, a.ood_main, a.results = "results_quick/ood_tsg", "results_quick/ood_main", \
            "results_quick"
    print("=" * 66 + "\nMODULE 14: two-stage-ILP-guided NSGA-III on PrefLib 00036\n" + "=" * 66)
    if a.wmd is None:
        d = find_wmd(Path("."))
        if d is None:
            raise SystemExit("No .wmd files found below this folder: give --wmd <PrefLib folder>")
        a.wmd = str(d)
    tasks = make_tasks(a)
    if not tasks:
        raise SystemExit(f"No instances of {a.min_size}-{a.max_size} vertices in {a.wmd}")
    if not a.analyse_only:
        run_tasks(tasks, a.jobs)
    write_report(tasks, a)
    print("\n" + "=" * 66 + "\nMODULE 14 COMPLETE\n" + "=" * 66)


if __name__ == "__main__":
    main()
