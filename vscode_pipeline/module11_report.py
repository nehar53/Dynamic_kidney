import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ============================================================
# MODULE 11: REMAINING EXPERIMENTS + PAPER NUMBERS
# Paper 2 — Dynamic KEP Implementation
# ============================================================
#
# Step "sensitivity"  module 10 sensitivity experiments (30 runs):
#                     selection rule, CIT 16/12 h, mu_d, batching interval
# Step "scalability"  module 10 scalability (200 vertices: all methods,
#                     500 vertices: ILP baselines only)
# Step "report"       re-creates every module 10 table and figure from
#                     all finished runs, then collects the numbers used
#                     in the paper (main experiment, sensitivity,
#                     scalability, PrefLib) into
#                         results/paper/paper_numbers.md   <- send this
#                         results/paper/paper_numbers.json
#                         results/paper/tab_*.tex           (LaTeX tables)
#
# Press Run (no arguments) to run RUN_STEPS below, or:
#   python module11_report.py --steps report
#   python module11_report.py --steps sensitivity --jobs 6
#
# Every step is resumable (finished trajectories are kept). The report
# only reads results; missing parts are listed as "not available".

RUN_STEPS = ["sensitivity", "scalability", "report"]
RESULTS = "results"                 # module 10 output folder
OOD_MAIN = "results/ood_main"       # module 9 preset "main" / "batched"
OOD_ILP = "results/ood_ilp_sizes"   # module 9 preset "ilp"

DYN_U, DYN_L = "dyn_nsga3_ucb1_myopic", "dyn_nsga3_linucb_myopic"
BAT_U, BAT_L = "dyn_nsga3_ucb1_batched", "dyn_nsga3_linucb_batched"
ILP, LEX = "myopic_ilp", "dickerson_lex"
SHORT = {
    "static_nsga2_ucb1": "S-NSGA-II+UCB1", "static_nsga3_fixed": "S-NSGA-III fixed",
    DYN_U: "D-UCB1 myopic", DYN_L: "D-LinUCB myopic",
    BAT_U: "D-UCB1 batched", BAT_L: "D-LinUCB batched",
    ILP: "Myopic ILP", LEX: "Dickerson lex",
    "dyn_nsga3_ucb1_batched_mi3": "D-UCB1 batched (max 3)",
}
ORDER = ["static_nsga2_ucb1", "static_nsga3_fixed", DYN_U, DYN_L, BAT_U,
         BAT_L, ILP, LEX]


# ============================================================
# SMALL HELPERS
# ============================================================

def _p(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def _paired(a: pd.Series, b: pd.Series) -> Dict:
    """Paired Wilcoxon on aligned series (index = run)."""
    df = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(df) < 2 or np.allclose(df["a"], df["b"]):
        return {"n": len(df), "mean_a": df["a"].mean(), "mean_b": df["b"].mean(),
                "p": np.nan, "d": 0.0}
    d = df["a"] - df["b"]
    sd = d.std(ddof=1)
    return {"n": len(df), "mean_a": df["a"].mean(), "mean_b": df["b"].mean(),
            "p": float(wilcoxon(df["a"], df["b"], zero_method="zsplit").pvalue),
            "d": float(d.mean() / sd) if sd > 0 else 0.0}


def _slope_test(g: pd.DataFrame) -> Dict:
    sl = [np.polyfit(x["round"], x["PoF"], 1)[0]
          for _, x in g.groupby("run") if x["round"].nunique() >= 3]
    if len(sl) < 2:
        return {"n": len(sl), "slope": np.nan, "p": np.nan}
    sl = np.asarray(sl)
    p = 1.0 if np.allclose(sl, 0) else float(wilcoxon(sl, zero_method="zsplit").pvalue)
    return {"n": len(sl), "slope": float(sl.mean()), "p": p}


def _md(df: pd.DataFrame, floatfmt: str = "{:.3g}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join(
            floatfmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v)
            else ("n/a" if isinstance(v, float) else str(v)) for v in row) + " |")
    return "\n".join(out)


def _tex(df: pd.DataFrame, caption: str, label: str, wide: bool = False) -> str:
    env = "table*" if wide else "table"
    cols = list(df.columns)
    esc = lambda v: str(v).replace("%", r"\%").replace("_", r"\_").replace("±", r"$\pm$")
    body = "\n".join(" & ".join(esc(v) for v in r) + r" \\"
                     for r in df.itertuples(index=False))
    return "\n".join([
        rf"\begin{{{env}}}[!t]", r"\centering", rf"\caption{{{caption}}}",
        rf"\label{{{label}}}", r"\footnotesize",
        r"\begin{tabular}{" + "l" * len(cols) + "}", r"\toprule",
        " & ".join(esc(c) for c in cols) + r" \\", r"\midrule", body,
        r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}", ""])


def _ms(s: pd.Series, dec: int) -> str:
    s = s.dropna()
    if not len(s):
        return "n/a"
    return f"{s.mean():.{dec}f} ± {s.std(ddof=1):.{dec}f}" if len(s) > 1 \
        else f"{s.mean():.{dec}f}"


# ============================================================
# STEPS THAT RUN EXPERIMENTS (via module 10)
# ============================================================

def run_module10(argv: List[str]) -> None:
    import module10_experiment as m10
    print("\n" + "#" * 66 + f"\n# module 10 {' '.join(argv)}\n" + "#" * 66)
    m10.main(argv)


# ============================================================
# REPORT: MAIN EXPERIMENT
# ============================================================

def report_main(res: Path, md: List[str], js: Dict, texdir: Path) -> None:
    runs_p, rounds_p = res / "runs.csv", res / "rounds.csv"
    if not runs_p.exists() or not rounds_p.exists():
        md += ["## Main experiment", "", "not available (run module 10 main)", ""]
        return
    runs = pd.read_csv(runs_p)
    runs = runs[runs["variant"] == "base"]
    rounds = pd.read_csv(rounds_p)
    rounds = rounds[rounds["variant"] == "base"]
    rounds["cleared"] = rounds["cleared"].astype(bool)
    taus = sorted(runs["tau"].unique())
    J = js.setdefault("main", {})
    n_runs = runs.groupby("tau")["run"].nunique().to_dict()
    md += ["## 1. Main experiment (base settings)", "",
           f"Paired runs per tau: {n_runs}", ""]

    def per(cfg, tau, col):
        s = runs[(runs["config"] == cfg) & (runs["tau"] == tau)]
        return s.set_index("run")[col]

    # ---- calibration (myopic ILP) ----
    ilp_r = rounds[rounds["config"] == ILP]
    cal = []
    for tau in taus:
        g = ilp_r[ilp_r["tau"] == tau]
        c = g[g["cleared"]]
        cal.append({"tau": tau,
                    "pool patients (rounds 13-25)": g[g["round"] > 12]["n_patients"].mean(),
                    "ILP transplants / year (rounds 2-25)": c[c["round"] > 1]["F1"].mean() * 12,
                    "round-1 transplants": c[c["round"] == 1]["F1"].mean(),
                    "withdrawn per run": g.groupby("run")["withdrawn"].sum().mean()})
    cal = pd.DataFrame(cal)
    md += ["### 1a. Calibration (myopic ILP)", "", _md(cal), ""]
    J["calibration"] = cal.round(3).to_dict("records")

    # ---- compact main table ----
    rows = []
    for tau in taus:
        for cfg in ORDER:
            s = runs[(runs["config"] == cfg) & (runs["tau"] == tau)]
            if s.empty:
                continue
            rows.append({"tau": tau, "Configuration": SHORT[cfg],
                         "PoF (%)": _ms(s["pof_mean"], 2),
                         "Loss vs ILP (%)": _ms(s["pof_vs_myopic_ilp"], 2),
                         "Transplants": _ms(s["total_transplants"], 1),
                         "High-cPRA tx": _ms(s["total_high_cpra_transplants"], 1),
                         "F2": _ms(s["mean_F2"], 3), "F3": _ms(s["mean_F3"], 3),
                         "F4": _ms(s["mean_F4"], 3),
                         "High-cPRA lost": _ms(s["total_withdrawn_high_cpra"], 1)})
    tab = pd.DataFrame(rows)
    md += ["### 1b. Trajectory results (mean ± sd over runs)", "", _md(tab), ""]
    (texdir / "tab_main_compact.tex").write_text(_tex(
        tab, "Trajectory results over 25 clearing rounds (mean $\\pm$ sd over "
        "100 paired runs). Loss vs ILP: trajectory-level loss $\\Delta$ against "
        "the myopic ILP.", "tab:main", wide=True), encoding="utf-8")

    # ---- framework vs myopic ILP / Dickerson / batched ----
    comp = []
    fam = []
    for tau in taus:
        for a, b, metrics in [
            (DYN_U, ILP, ["total_transplants", "mean_F2", "mean_F3", "mean_F4",
                          "total_high_cpra_transplants", "total_withdrawn_high_cpra"]),
            (DYN_U, LEX, ["total_transplants", "total_high_cpra_transplants",
                          "mean_F3", "pof_mean"]),
            (DYN_U, BAT_U, ["total_transplants", "total_high_cpra_transplants",
                            "total_withdrawn_high_cpra", "pof_mean"]),
            (DYN_L, BAT_L, ["total_transplants", "pof_mean"]),
            (DYN_U, DYN_L, ["pof_mean", "total_transplants", "mean_F3", "mean_HV"]),
            (DYN_U, "static_nsga3_fixed", ["pof_mean", "total_transplants", "mean_HV"]),
        ]:
            for m in metrics:
                r = _paired(per(a, tau, m), per(b, tau, m))
                if r["n"] == 0:
                    continue
                ch = (100 * (r["mean_a"] / r["mean_b"] - 1)
                      if r["mean_b"] not in (0, np.nan) and r["mean_b"] else np.nan)
                comp.append({"tau": tau, "a": SHORT[a], "b": SHORT[b], "metric": m,
                             "mean_a": r["mean_a"], "mean_b": r["mean_b"],
                             "change_%": ch, "p": r["p"], "d_z": r["d"], "n": r["n"]})
                fam.append(len(comp) - 1)
    comp = pd.DataFrame(comp)
    if len(comp):
        m = comp["p"].notna()
        comp["p_bonf"] = np.nan
        comp.loc[m, "p_bonf"] = np.minimum(1.0, comp.loc[m, "p"] * m.sum())
        show = comp.assign(p=comp["p"].map(_p), p_bonf=comp["p_bonf"].map(_p))
        md += ["### 1c. Key paired comparisons (Wilcoxon signed-rank; "
               "Bonferroni over all rows of this table)", "",
               _md(show[["tau", "a", "b", "metric", "mean_a", "mean_b", "change_%",
                         "p", "p_bonf", "d_z", "n"]]), ""]
        J["comparisons"] = comp.round(5).to_dict("records")

    # ---- PoF structure: round 1 vs later, floor rule ----
    pof = []
    for cfg in [DYN_U, DYN_L, BAT_U, BAT_L]:
        for tau in taus:
            g = rounds[(rounds["config"] == cfg) & (rounds["tau"] == tau)
                       & rounds["cleared"] & rounds["PoF"].notna()]
            if g.empty:
                continue
            later = g[g["round"] > 1]
            big = g[g["F1_ILP"] >= 10]
            t_all, t_late = _slope_test(g), _slope_test(later)
            pof.append({"config": SHORT[cfg], "tau": tau,
                        "PoF round 1": g[g["round"] == 1]["PoF"].mean(),
                        "PoF rounds 2-25": later["PoF"].mean(),
                        "share rounds F1_ILP>=10": (g["F1_ILP"] >= 10).mean(),
                        "PoF if F1_ILP>=10": big["PoF"].mean(),
                        "slope all (pp/round)": t_all["slope"], "p all": _p(t_all["p"]),
                        "slope rounds 2-25": t_late["slope"], "p 2-25": _p(t_late["p"])})
    pof = pd.DataFrame(pof)
    md += ["### 1d. PoF structure and trend", "",
           "With the 90% floor, PoF_t > 0 is possible only when F1_ILP >= 10; "
           "round 1 clears the initial backlog of 75 vertices.", "", _md(pof), ""]
    J["pof_structure"] = pof.round(5).to_dict("records")

    # ---- bandits ----
    arm_cols = [c for c in runs.columns if c.startswith("share_arm_")]
    if arm_cols:
        arms = runs[runs["config"].isin(["static_nsga2_ucb1", DYN_U, DYN_L, BAT_U, BAT_L])]
        a = arms.groupby(["config", "tau"])[arm_cols].mean()
        P = a.to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -(np.where(P > 0, P * np.log(P), 0)).sum(1) / np.log(len(arm_cols))
        a["normalised entropy (1 = uniform)"] = ent
        a = a.reset_index()
        a["config"] = a["config"].map(SHORT)
        md += ["### 1e. Selected mutation rates (share of generations)", "", _md(a), ""]
        J["arms"] = a.round(4).to_dict("records")

    # ---- implementation statistics ----
    c = rounds[rounds["cleared"]]
    st = {"share exact rounds (MOEA)":
          c[c["config"].str.contains("nsga")]["exact_front"].astype(bool).mean()
          if "exact_front" in c else np.nan,
          "share rounds chain list truncated": c["chains_truncated"].astype(bool).mean(),
          "share rounds PICEF chains added":
          (c["picef_chains_added"] > 0).mean() if "picef_chains_added" in c else np.nan,
          "MOEA rounds with F1 < 0.9 F1_ILP (should be 0)":
          int((c[c["config"].str.contains("nsga")]["F1"]
               < 0.9 * c[c["config"].str.contains("nsga")]["F1_ILP"] - 1e-9).sum()),
          "mean seconds per MOEA clearing":
          c[c["config"].str.contains("nsga")]["seconds"].mean()}
    md += ["### 1f. Implementation checks", ""] + \
          [f"- {k}: {v:.4g}" if isinstance(v, float) else f"- {k}: {v}"
           for k, v in st.items()] + [""]
    J["implementation"] = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                           for k, v in st.items()}


# ============================================================
# REPORT: SENSITIVITY AND SCALABILITY
# ============================================================

def report_tables(res: Path, md: List[str], js: Dict, texdir: Path) -> None:
    import module10_experiment as m10
    for exp in m10.SENSITIVITY:
        p = res / f"table_{exp}.csv"
        if not p.exists():
            md += [f"## Sensitivity: {exp}", "", "not available", ""]
            continue
        t = pd.read_csv(p)
        cols = ["variant", "config", "tau"] + [n for _, n, _ in m10.PAPER_COLUMNS[:4]
                                              if n in t]
        if "pof_vs_myopic_ilp_mean" in t:
            t["Loss vs ILP (%)"] = m10.format_mean_sd(t, "pof_vs_myopic_ilp", 2)
            cols.append("Loss vs ILP (%)")
        t = t[cols].assign(config=t["config"].map(SHORT))
        md += [f"## 2. Sensitivity: {exp} — {m10.EXPERIMENTS[exp]['about']}", "",
               _md(t), ""]
        w = res / f"table_{exp}_wilcoxon.csv"
        if w.exists():
            wt = pd.read_csv(w)
            wt = wt[wt["metric"].isin(["pof_mean", "total_transplants",
                                       "total_high_cpra_transplants", "mean_F3"])]
            if len(wt):
                wt = wt.assign(p_bonferroni=wt["p_bonferroni"].map(_p))
                md += [_md(wt[["tau", "metric", "config_a", "config_b", "mean_a",
                               "mean_b", "p_bonferroni", "cohens_d", "direction"]]), ""]
        js.setdefault("sensitivity", {})[exp] = t.to_dict("records")
        (texdir / f"tab_{exp}.tex").write_text(
            _tex(t, f"Sensitivity analysis: {m10.EXPERIMENTS[exp]['about']} "
                 "(mean $\\pm$ sd, 30 paired runs).", f"tab:{exp}"), encoding="utf-8")
    p = res / "table_scalability.csv"
    if p.exists():
        t = pd.read_csv(p)
        md += ["## 3. Scalability (means per clearing round)", "", _md(t), ""]
        js["scalability"] = t.round(4).to_dict("records")
        keep = [c for c in ["variant", "config", "pool_size", "cycles", "chains",
                            "seconds_per_round", "F1", "F1_ILP", "PoF"] if c in t]
        (texdir / "tab_scalability.tex").write_text(
            _tex(t[keep].round(2).assign(config=t["config"].map(SHORT)),
                 "Scalability: means per clearing round.", "tab:scalability"),
            encoding="utf-8")
    else:
        md += ["## 3. Scalability", "", "not available", ""]


# ============================================================
# REPORT: PREFLIB (MODULE 9)
# ============================================================

def report_ood(ood_main: Path, ood_ilp: Path, md: List[str], js: Dict,
               texdir: Path) -> None:
    J = js.setdefault("preflib", {})
    p = ood_main / "ood_runs.csv"
    if p.exists():
        r = pd.read_csv(p)
        # report the corrected batched form only
        r = r[r["config"] != BAT_U]
        cfgs = [c for c in [ILP, LEX, DYN_U, DYN_L, "dyn_nsga3_ucb1_batched_mi3"]
                if c in set(r["config"])]
        ilp = r[r["config"] == ILP].set_index("pair_id")["total_transplants"]
        rows = []
        for c in cfgs:
            s = r[r["config"] == c]
            loss = 100 * (1 - s.set_index("pair_id")["total_transplants"] / ilp)
            rows.append({"Configuration": SHORT.get(c, c),
                         "PoF (%)": _ms(s["pof_mean"], 2),
                         "Loss vs ILP (%)": _ms(loss, 2),
                         "Transplants": _ms(s["total_transplants"], 2),
                         "High-cPRA tx": _ms(s["total_high_cpra_transplants"], 2),
                         "F3": _ms(s["mean_F3"], 3)})
        t = pd.DataFrame(rows)
        n_units = r["pair_id"].nunique()
        md += ["## 4. PrefLib 00036 (module 9, preset main + batched)", "",
               f"Paired units (instance x run): {n_units}", "", _md(t), ""]
        (texdir / "tab_ood.tex").write_text(_tex(
            t, "PrefLib 00036, 12 instances of 128--147 vertices, 10 paired runs "
            "each, $\\tau = 12$, 25 rounds (mean $\\pm$ sd).", "tab:ood"),
            encoding="utf-8")
        comp = []
        for a, b in [(DYN_U, ILP), (DYN_U, LEX), (DYN_U, DYN_L),
                     (DYN_U, "dyn_nsga3_ucb1_batched_mi3")]:
            if a not in cfgs or b not in cfgs:
                continue
            for m in ["pof_mean", "total_transplants", "total_high_cpra_transplants",
                      "mean_F3"]:
                x = r[r["config"] == a].set_index("pair_id")[m]
                y = r[r["config"] == b].set_index("pair_id")[m]
                q = _paired(x, y)
                comp.append({"a": SHORT[a], "b": SHORT.get(b, b), "metric": m,
                             "mean_a": q["mean_a"], "mean_b": q["mean_b"],
                             "p": q["p"], "d_z": q["d"], "n": q["n"]})
        comp = pd.DataFrame(comp)
        if len(comp):
            mk = comp["p"].notna()
            comp["p_bonf"] = np.nan
            comp.loc[mk, "p_bonf"] = np.minimum(1.0, comp.loc[mk, "p"] * mk.sum())
            md += [_md(comp.assign(p=comp["p"].map(_p),
                                   p_bonf=comp["p_bonf"].map(_p))), ""]
        J["main"] = t.to_dict("records")
        J["comparisons"] = comp.round(5).to_dict("records")
    else:
        md += ["## 4. PrefLib main", "", "not available (run module 9 main/batched)", ""]
    p = ood_ilp / "ood_runs.csv"
    if p.exists():
        r = pd.read_csv(p)
        by = (r.groupby(["n_vertices", "config"])["pof_mean"].mean()
              .unstack().round(2).reset_index())
        allm = r.groupby("config")[["pof_mean", "total_transplants",
                                    "total_high_cpra_transplants", "mean_F3"]].mean()
        md += ["## 5. PrefLib ILP baselines on all instances up to 294 vertices", "",
               f"Instances: {r['dataset'].nunique()}", "",
               _md(allm.reset_index()), "", "Mean PoF by instance size:", "",
               _md(by), ""]
        J["ilp_sizes"] = allm.round(4).reset_index().to_dict("records")
    else:
        md += ["## 5. PrefLib ILP baselines", "", "not available", ""]


def report(args) -> None:
    res = Path(args.results)
    if (res / "manifest.json").exists():
        # refresh all module 10 tables/figures from every finished run
        try:
            run_module10(["--experiment", "all", "--analyse-only", "--allow-large",
                          "--out", str(res)])
        except SystemExit as e:
            print(f"(module 10 analysis skipped: {e})")
    out = res / "paper"
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Paper 2 — numbers for the manuscript", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} by module 11 from "
          f"`{res}`, `{args.ood_main}`, `{args.ood_ilp}`.", ""]
    js: Dict = {}
    report_main(res, md, js, out)
    report_tables(res, md, js, out)
    report_ood(Path(args.ood_main), Path(args.ood_ilp), md, js, out)
    if (res / "manifest.json").exists():
        js["manifest"] = json.loads((res / "manifest.json").read_text())
    (out / "paper_numbers.md").write_text("\n".join(md), encoding="utf-8")
    (out / "paper_numbers.json").write_text(
        json.dumps(js, indent=1, default=lambda o: None if o != o else str(o)),
        encoding="utf-8")
    print("\n" + "\n".join(md[:3]))
    print(f"\nWritten: {out}/paper_numbers.md (send this), paper_numbers.json, "
          f"tab_*.tex")


# ============================================================
# MAIN
# ============================================================

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Module 11: remaining experiments + paper numbers")
    ap.add_argument("--steps", nargs="+", default=RUN_STEPS,
                    choices=["sensitivity", "scalability", "report"])
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--ood-main", default=OOD_MAIN)
    ap.add_argument("--ood-ilp", default=OOD_ILP)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args(argv)
    jobs = ["--jobs", str(args.jobs)] if args.jobs else []
    print("=" * 66 + "\nMODULE 11: " + ", ".join(args.steps) + "\n" + "=" * 66)
    if "sensitivity" in args.steps:
        run_module10(["--experiment", "sensitivity", "--out", args.results] + jobs)
    if "scalability" in args.steps:
        run_module10(["--experiment", "scalability", "--allow-large",
                      "--out", args.results] + jobs)
    if "report" in args.steps:
        report(args)
    print("\n" + "=" * 66 + "\nMODULE 11 COMPLETE\n" + "=" * 66)


if __name__ == "__main__":
    main()
