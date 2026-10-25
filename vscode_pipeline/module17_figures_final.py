import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import t as student_t, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ============================================================
# MODULE 17: FINAL FIGURES AND TABLES (original framework +
#            two-stage ILP baseline, supervisor's decision)
# ============================================================
#
# The proposed framework is the ORIGINAL volume-ILP-guided NSGA-III
# (modules 5-7). The two-stage ILP (module 6) is reported as a separate,
# stronger baseline at maximum volume. Reads saved results only:
#   module 10  main experiment (100 runs), sens_select
#   module 12  identical pools (A), fixed trajectories (B)
#   module 13  two-stage ILP trajectories, 100 runs (reference runs)
#   module 16  trade-off front vs exact ILP per volume level, selection
#              rules at tau = 6/12/18
#   modules 9 / 14  PrefLib 00036 (framework, ILPs, two-stage ILP)
# No simulation is run. TSG results (modules 13-15) are NOT used.
#
# Output (existing figures / figures_tsg folders are not touched):
#   results/paper/figures_final/  (.png 300 dpi + .pdf)
#     fig1_pof_per_round        PoF_t per round (mean, 95% CI), tau 6/12/18
#     fig2_myopic_vs_batched    PoF per cleared round + cumulative transplants
#     fig3_three_levels         F2/F3/F4: identical pools, fixed trajectories,
#                               full trajectories (framework vs ILPs)
#     fig4_tradeoff_levels      fairness vs volume level: front best, rule
#                               pick, exact ILP at that volume, two-stage max
#     fig5_selection_rules      Delta vs F3 / F4 for the selection rules, tau 6/12/18
#     fig6_forest_tau{6,12,18}  paired differences framework - baselines
#     fig7_forest_preflib       same on PrefLib 00036
#   results/paper/tables_final/
#     tab_main_final.tex, tab_three_levels.tex, tab_tradeoff.tex,
#     tab_select.tex, tab_cost.tex, tab_ood_final.tex
#
# Press Run after modules 12 (--runs 100) and 16, or:
#   python module17_figures_final.py --results results

FIG_DIR, TAB_DIR = "figures_final", "tables_final"
TAUS = [6, 12, 18]
RUNS = 100
LEVELS = [100, 95, 90, 85, 80]
FRAME = "dyn_nsga3_ucb1_myopic"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
GREY = "#5f5e5a"

# config -> label, colour, marker, line style (variant 'base' unless keyed)
SPEC: Dict[Tuple[str, str], Tuple[str, str, str, str]] = {
    (FRAME, "base"): ("Framework UCB1 (90% floor)", BLUE, "o", "-"),
    ("dyn_nsga3_linucb_myopic", "base"): ("Framework LinUCB", VIOLET, "s", "-"),
    ("dyn_nsga3_ucb1_batched", "base"): ("Framework UCB1 batched", BLUE, "o", "--"),
    ("dyn_nsga3_linucb_batched", "base"): ("Framework LinUCB batched", VIOLET, "s", "--"),
    (FRAME, "floor100"): ("Framework, 100% floor", BLUE, "^", "-"),
    (FRAME, "floor80"): ("Framework, 80% floor", BLUE, "v", "-"),
    (FRAME, "compromise"): ("Framework, compromise", BLUE, "D", "-"),
    ("static_nsga2_ucb1", "base"): ("Static NSGA-II+UCB1", GREEN, "P", "-"),
    ("static_nsga3_fixed", "base"): ("Static NSGA-III fixed", GREEN, "X", "--"),
    ("twostage_ilp_sum", "base"): ("Two-stage ILP (sum)", AQUA, "D", "-"),
    ("twostage_ilp_f3", "base"): ("Two-stage ILP (F3)", AQUA, "d", "--"),
    ("myopic_ilp", "base"): ("Myopic ILP", GREY, "*", "-"),
    ("dickerson_lex", "base"): ("Dickerson lex", YELLOW, "h", "-"),
}
MAIN_KEYS = [(FRAME, "base"), ("dyn_nsga3_linucb_myopic", "base"),
             ("dyn_nsga3_ucb1_batched", "base"), ("dyn_nsga3_linucb_batched", "base"),
             ("static_nsga2_ucb1", "base"), ("static_nsga3_fixed", "base"),
             ("twostage_ilp_sum", "base"), ("twostage_ilp_f3", "base"),
             ("myopic_ilp", "base"), ("dickerson_lex", "base")]
FOREST_BASES = [("twostage_ilp_sum", "base"), ("twostage_ilp_f3", "base"),
                ("myopic_ilp", "base"), ("dickerson_lex", "base"),
                ("static_nsga2_ucb1", "base"), ("static_nsga3_fixed", "base")]
ML = {"delta": "$\\Delta$ (%)", "mean_F2": "$F_2$", "mean_F3": "$F_3$", "mean_F4": "$F_4$",
      "F2": "$F_2$", "F3": "$F_3$", "F4": "$F_4$", "score": "Score"}


def _style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
        "axes.linewidth": 0.6, "xtick.color": "#5f5e5a", "ytick.color": "#5f5e5a",
        "axes.grid": True, "grid.color": "#e4e3de", "grid.linewidth": 0.5,
        "lines.linewidth": 1.4, "lines.markersize": 4, "savefig.bbox": "tight",
        "savefig.dpi": 300})


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {out / (name + '.png')}")


def _ci(x) -> float:
    x = pd.Series(x).dropna()
    if len(x) < 2:
        return 0.0
    return float(student_t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def _module10():
    try:
        import module10_experiment as m10
    except ImportError:
        import module10_experiments as m10
    return m10


def _legend_below(fig, ax_or_handles, ncol=4, dy=-0.06, rect_bottom=0.06):
    if isinstance(ax_or_handles, tuple):
        h, l = ax_or_handles
    else:
        h, l = ax_or_handles.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, dy))
    fig.tight_layout(rect=(0, rect_bottom, 1, 1))


# ============================================================
# DATA
# ============================================================

def load_main(a, m10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Main experiment + two-stage ILP + sens_select variants (module 10 raw files)."""
    out = Path(a.results)
    tasks = []
    for tau in a.taus:
        for run in range(a.runs):
            for c, v in MAIN_KEYS:
                tasks.append({"variant": v, "config": c,
                              "path": str(m10.raw_path(out, v, c, tau, run))})
            for v in ("floor100", "floor80", "compromise"):
                tasks.append({"variant": v, "config": FRAME,
                              "path": str(m10.raw_path(out, v, FRAME, tau, run))})
    tasks = [t for t in tasks if Path(t["path"]).exists()]
    if not tasks:
        return pd.DataFrame(), pd.DataFrame()
    rounds = m10.collect(tasks)
    rounds["cleared"] = rounds["cleared"].astype(bool)
    runs = m10.per_run_metrics(rounds)
    ilp = runs[(runs["config"] == "myopic_ilp") & (runs["variant"] == "base")] \
        .set_index(["tau", "run"])["total_transplants"]
    ref = ilp.reindex(pd.MultiIndex.from_frame(runs[["tau", "run"]])).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        runs["delta"] = np.where(ref > 0, 100 * (ref - runs["total_transplants"]) / ref, np.nan)
    secs = rounds[rounds["cleared"]].groupby(["config", "variant", "tau"])["seconds"].mean()
    runs["sec_round"] = [secs.get((c, v, t), np.nan)
                         for c, v, t in zip(runs["config"], runs["variant"], runs["tau"])]
    return rounds, runs


def _sel(runs, key, tau=None):
    g = runs[(runs["config"] == key[0]) & (runs["variant"] == key[1])]
    return g if tau is None else g[g["tau"] == tau]


def _score(df, pre, ref="ilp_"):
    r = [np.maximum(df[f"{ref}F{k}"], 0.01) for k in (2, 3, 4)]
    return sum(df[f"{pre}F{k}"] / rr for k, rr in zip((2, 3, 4), r))


def load_levels(a, m10, runs) -> pd.DataFrame:
    """Rows: level (A/B/C), method, unit, F2, F3, F4 (score where defined)."""
    out, rows = Path(a.results), []
    p = out / "ablation_twostage_pools.csv"                 # module 12 A
    if p.exists():
        d = pd.read_csv(p)
        for nm, pre in [("Volume ILP", "ilp_"), ("Two-stage ILP (sum)", "ts_sum_"),
                        ("Two-stage ILP (F3)", "ts_f3_"), ("Framework, 100% pick", "moea_pick100_"),
                        ("Framework, 90% pick", "moea_pick90_")]:
            if f"{pre}F2" in d:
                sc = _score(d, pre)
                for i, r in d.iterrows():
                    rows.append({"level": "A. identical pools", "method": nm, "unit": r["seed"],
                                 "F2": r[f"{pre}F2"], "F3": r[f"{pre}F3"], "F4": r[f"{pre}F4"],
                                 "score": sc.iloc[i]})
    frames = []                                              # module 12 B
    for run in range(a.runs):
        f = m10.raw_path(out, "base", "dyn_nsga3_ucb1_myopic_shadow", 12, run)
        if Path(f).exists():
            frames.append(pd.read_csv(f).assign(run=run))
    if frames:
        b = pd.concat(frames, ignore_index=True)
        b = b[b["cleared"].astype(bool) & b["ilp_F2"].notna()]
        b = b.assign(impl_F2=b["F2"], impl_F3=b["F3"], impl_F4=b["F4"])
        for nm, pre in [("Volume ILP", "ilp_"), ("Two-stage ILP (sum)", "ts_sum_"),
                        ("Two-stage ILP (F3)", "ts_f3_"), ("Framework, 100% pick", "moea_pick100_"),
                        ("Framework, 90% pick", "impl_")]:
            if f"{pre}F2" not in b:
                continue
            bb = b.assign(score=_score(b, pre))
            m = bb.groupby("run")[[f"{pre}F2", f"{pre}F3", f"{pre}F4", "score"]].mean()
            for run, r in m.iterrows():
                rows.append({"level": "B. fixed trajectories", "method": nm, "unit": run,
                             "F2": r[f"{pre}F2"], "F3": r[f"{pre}F3"], "F4": r[f"{pre}F4"],
                             "score": r["score"]})
    if len(runs):                                            # full trajectories, tau 12
        for nm, key in [("Myopic ILP", ("myopic_ilp", "base")),
                        ("Two-stage ILP (sum)", ("twostage_ilp_sum", "base")),
                        ("Two-stage ILP (F3)", ("twostage_ilp_f3", "base")),
                        ("Framework, 100% floor", (FRAME, "floor100")),
                        ("Framework, 90% floor", (FRAME, "base"))]:
            g = _sel(runs, key, 12)
            for _, r in g.iterrows():
                rows.append({"level": "C. full trajectories (tau 12)", "method": nm,
                             "unit": r["run"], "F2": r["mean_F2"], "F3": r["mean_F3"],
                             "F4": r["mean_F4"], "score": np.nan,
                             "transplants": r["total_transplants"]})
    return pd.DataFrame(rows)


def load_tradeoff(a) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(a.results) / "tradeoff16"
    pools = pd.read_csv(base / "pools.csv") if (base / "pools.csv").exists() else pd.DataFrame()
    files = sorted((base / "rounds").glob("*.csv")) if (base / "rounds").exists() else []
    rounds = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    return pools, rounds


def load_preflib(a) -> pd.DataFrame:
    import module14_tsg_preflib as m14
    ts_dir, ref_dir = Path(a.results) / "ood_tsg" / "raw", Path(a.results) / "ood_main" / "raw"
    want_ts = {"twostage_ilp_sum", "twostage_ilp_f3"}
    want_ref = {"dyn_nsga3_ucb1_myopic", "dyn_nsga3_linucb_myopic",
                "dyn_nsga3_ucb1_batched_mi3", "myopic_ilp", "dickerson_lex"}
    ts_files = [f for f in sorted(ts_dir.glob("*.csv")) if f.stem.split("__")[1] in want_ts] \
        if ts_dir.exists() else []
    datasets = {f.name.split("__")[0] for f in ts_files}
    tasks = [{"path": str(f), "cfg": {"name": f.stem.split("__")[1]}} for f in ts_files]
    for f in (sorted(ref_dir.glob("*.csv")) if ref_dir.exists() else []):
        parts = f.stem.split("__")
        if len(parts) == 4 and parts[1] in want_ref and (not datasets or parts[0] in datasets) \
                and parts[2] == "tau12":
            tasks.append({"path": str(f), "cfg": {"name": parts[1]}})
    if not tasks:
        return pd.DataFrame()
    _, runs = m14.per_run(tasks)
    return runs


# ============================================================
# FIGURES
# ============================================================

def fig_pof(rounds, taus, out):
    keys = [(FRAME, "base"), ("dyn_nsga3_linucb_myopic", "base"),
            ("static_nsga2_ucb1", "base"), ("dickerson_lex", "base")]
    c = rounds[rounds["cleared"]]
    fig, axes = plt.subplots(1, len(taus), figsize=(7.16, 2.3), sharey=True, squeeze=False)
    for ax, tau in zip(axes[0], taus):
        for k in keys:
            g = c[(c["config"] == k[0]) & (c["variant"] == k[1]) & (c["tau"] == tau)]
            if not len(g):
                continue
            s = g.groupby("round")["PoF"]
            m, ci = s.mean(), s.apply(_ci)
            lab, col, mk, ls = SPEC[k]
            ax.plot(m.index, m.values, color=col, ls=ls, marker=mk, markevery=4, label=lab)
            ax.fill_between(m.index, m - ci, m + ci, color=col, alpha=0.12, lw=0)
        ax.set_title(f"$\\tau$ = {tau} months")
        ax.set_xlabel("Clearing round $t$")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0][0].set_ylabel("PoF$_t$ (%)")
    _legend_below(fig, axes[0][0], ncol=4, dy=-0.08)
    _save(fig, out, "fig1_pof_per_round")


def fig_batched(rounds, taus, out):
    keys = [(FRAME, "base"), ("dyn_nsga3_ucb1_batched", "base"), ("dyn_nsga3_linucb_batched", "base")]
    fig, axes = plt.subplots(2, len(taus), figsize=(7.16, 3.8), sharex=True, squeeze=False)
    for j, tau in enumerate(taus):
        for k in keys:
            g = rounds[(rounds["config"] == k[0]) & (rounds["variant"] == k[1]) & (rounds["tau"] == tau)]
            if not len(g):
                continue
            lab, col, mk, ls = SPEC[k]
            cl = g[g["cleared"]]
            if len(cl):
                s = cl.groupby("round")["PoF"].mean()
                axes[0][j].plot(s.index, s.values, color=col, ls=ls, marker=mk,
                                markevery=1 if "batched" in k[0] else 4, label=lab)
            elif k == ("dyn_nsga3_ucb1_batched", "base"):
                axes[0][j].text(0.5, 0.5, "batched: no round cleared",
                                transform=axes[0][j].transAxes, ha="center",
                                color="#5f5e5a", fontsize=7)
            f1 = g.assign(F1=g["F1"].where(g["cleared"], 0).fillna(0)).sort_values("round")
            cum = f1.groupby("run")["F1"].cumsum()
            m = cum.groupby(f1["round"]).mean()
            axes[1][j].plot(m.index, m.values, color=col, ls=ls, label=lab)
        axes[0][j].set_title(f"$\\tau$ = {tau} months")
        axes[1][j].set_xlabel("Round $t$")
        axes[1][j].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0][0].set_ylabel("PoF per cleared round (%)")
    axes[1][0].set_ylabel("Cumulative transplants")
    _legend_below(fig, axes[1][0], ncol=3, dy=-0.04, rect_bottom=0.04)
    _save(fig, out, "fig2_myopic_vs_batched")


LEVEL_COLORS = {"Volume ILP": GREY, "Myopic ILP": GREY, "Two-stage ILP (sum)": AQUA,
                "Two-stage ILP (F3)": GREEN, "Framework, 100% pick": VIOLET,
                "Framework, 100% floor": VIOLET, "Framework, 90% pick": BLUE,
                "Framework, 90% floor": BLUE}


def fig_three_levels(lv, out):
    if not len(lv):
        return
    levels = [l for l in ["A. identical pools", "B. fixed trajectories",
                          "C. full trajectories (tau 12)"] if l in set(lv["level"])]
    mets = ["F2", "F3", "F4"]
    fig, axes = plt.subplots(len(levels), 3, figsize=(7.16, 1.35 * len(levels) + 0.6),
                             squeeze=False)
    for i, lev in enumerate(levels):
        sub = lv[lv["level"] == lev]
        methods = list(dict.fromkeys(sub["method"]))
        for j, met in enumerate(mets):
            ax = axes[i][j]
            for y, nm in enumerate(methods):
                g = sub[sub["method"] == nm][met]
                ax.errorbar(g.mean(), y, xerr=_ci(g), fmt="o", color=LEVEL_COLORS.get(nm, GREY),
                            ms=4.5, elinewidth=0.9, capsize=2)
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels(methods if j == 0 else [""] * len(methods))
            ax.invert_yaxis()
            ax.grid(axis="y", visible=False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            if i == len(levels) - 1:
                ax.set_xlabel(ML[met] + " (lower = fairer)")
        axes[i][0].set_title(lev, loc="left", fontsize=7.5, fontweight="bold")
    fig.text(0.5, -0.01, "Mean and 95% CI. A: 30 pools, one round; B: every round of the "
             "framework's trajectories (per-run means, same pools for all methods); C: each "
             "method clears its own pool sequence (per-run means). Maximum-volume methods: "
             "volume ILP, two-stage ILP, 100% pick/floor.", ha="center", fontsize=6.3,
             color="#5f5e5a", wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, out, "fig3_three_levels")


def exact_upto(d: pd.DataFrame, m: str, L: int) -> pd.Series:
    """Best exact two-stage value over the volumes solved at levels >= L
    (each level's V, up to F1*): the like-for-like counterpart of the
    front's best solution with F1 >= V."""
    cols = [f"exact_{m}_{k}" for k in LEVELS if k >= L and f"exact_{m}_{k}" in d]
    return d[cols].min(axis=1, skipna=True) if cols else pd.Series(np.nan, index=d.index)


def fig_tradeoff(pools, rounds, out, tau=12):
    panels = []
    if len(pools):
        panels.append(("A. identical pools (30 pools)", pools))
    if len(rounds) and (rounds["tau"] == tau).any():
        panels.append((f"B. fixed trajectories, $\\tau$ = {tau}", rounds[rounds["tau"] == tau]))
    if not panels:
        return
    mets = [("score", "Two-stage score"), ("F3", "$F_3$"), ("F4", "$F_4$")]
    fig, axes = plt.subplots(len(panels), 3, figsize=(7.16, 2.2 * len(panels) + 0.4),
                             squeeze=False, sharex=True)
    for i, (title, d) in enumerate(panels):
        d = d[d["F1_star"] > 0]
        for j, (m, lab) in enumerate(mets):
            ax = axes[i][j]
            for kind, col, mk, ls, name in [("best", BLUE, "o", "-", "Front best ($F_1 \\geq V$)"),
                                            ("pick", VIOLET, "s", "--", "Floor-rule pick at $\\rho$"),
                                            ("exact", AQUA, "D", "-",
                                             "Exact two-stage ILP, best of solved volumes $\\geq V$")]:
                ys, es = [], []
                for L in LEVELS:
                    col_name = f"{kind}_{m}_{L}"
                    if kind == "exact":
                        v = exact_upto(d, m, L)
                    else:
                        v = d[col_name] if col_name in d else pd.Series(dtype=float)
                    ys.append(v.mean())
                    es.append(_ci(v))
                ax.errorbar(LEVELS, ys, yerr=es, color=col, marker=mk, ls=ls, ms=4,
                            elinewidth=0.7, capsize=1.5, label=name)
            ts = d[f"ts_{m}"] if f"ts_{m}" in d else pd.Series(dtype=float)
            ax.axhline(ts.mean(), color=GREY, ls=":", lw=1.1,
                       label="Two-stage ILP at max volume")
            ax.invert_xaxis()
            ax.set_ylabel(lab)
            if i == len(panels) - 1:
                ax.set_xlabel("Volume level $\\rho$ (% of $F_1^*$)")
            if j == 0:
                tl = 0.0
                cols = [f"exact_status_{L}" for L in LEVELS if L < 100 and f"exact_status_{L}" in d]
                if cols:
                    tl = float((d[cols] == "time_limit").to_numpy().mean())
                ax.set_title(f"{title}; exact solves time-limited: {tl:.0%}", loc="left",
                             fontsize=7.5)
    _legend_below(fig, axes[0][0], ncol=4, dy=-0.05, rect_bottom=0.05)
    _save(fig, out, "fig4_tradeoff_levels")


def fig_selection(runs, taus, out):
    pts = [(FRAME, "floor100"), (FRAME, "base"), (FRAME, "floor80"), (FRAME, "compromise")]
    refs = [("twostage_ilp_sum", "base"), ("twostage_ilp_f3", "base"),
            ("myopic_ilp", "base"), ("dickerson_lex", "base")]
    taus = [t for t in taus if len(_sel(runs, (FRAME, "floor80"), t))]
    if not taus:
        return
    fig, axes = plt.subplots(2, len(taus), figsize=(7.16, 4.0), squeeze=False)
    for j, tau in enumerate(taus):
        for i, met in enumerate(["mean_F3", "mean_F4"]):
            ax = axes[i][j]
            xs, ys = [], []
            for k in pts:
                g = _sel(runs, k, tau)
                if not len(g):
                    continue
                lab, col, mk, _ = SPEC[k]
                x, y = g["delta"].mean(), g[met].mean()
                xs.append(x)
                ys.append(y)
                ax.errorbar(x, y, xerr=_ci(g["delta"]), yerr=_ci(g[met]), fmt=mk, color=col,
                            ms=5, elinewidth=0.7, capsize=1.5, mec="white", mew=0.5, label=lab)
            ax.plot(xs, ys, color=BLUE, lw=0.9, zorder=1)
            for k in refs:
                g = _sel(runs, k, tau)
                if not len(g):
                    continue
                lab, col, mk, _ = SPEC[k]
                ax.errorbar(g["delta"].mean(), g[met].mean(), xerr=_ci(g["delta"]),
                            yerr=_ci(g[met]), fmt=mk, color=col, ms=5, elinewidth=0.7,
                            capsize=1.5, mec="white", mew=0.5, label=lab)
            ax.set_ylabel(ML[met])
            if i == 0:
                ax.set_title(f"$\\tau$ = {tau} months")
            else:
                ax.set_xlabel("$\\Delta$ vs myopic ILP (%)")
    _legend_below(fig, axes[0][0], ncol=4, dy=-0.07, rect_bottom=0.08)
    _save(fig, out, "fig5_selection_rules")


def _forest(runs, pair_col, a_key, bases, metrics, title, out, name, select):
    t = select(runs, a_key).set_index(pair_col)
    rows = []
    for k in bases:
        b = select(runs, k).set_index(pair_col)
        if not len(b) or not len(t):
            continue
        for met in metrics:
            j = pd.concat([t[met], b[met]], axis=1, join="inner").dropna()
            if len(j) < 2:
                continue
            d = j.iloc[:, 0] - j.iloc[:, 1]
            p = np.nan if np.allclose(d, 0) else float(wilcoxon(j.iloc[:, 0], j.iloc[:, 1]).pvalue)
            rows.append({"base": k, "metric": met, "mean": d.mean(), "ci": _ci(d), "p": p})
    if not rows:
        return
    R = pd.DataFrame(rows)
    R["p_bonf"] = np.minimum(1.0, R["p"] * R["p"].notna().sum())
    bases = [k for k in bases if k in set(R["base"])]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.16, 0.35 * len(bases) + 1.0),
                             sharey=True, squeeze=False)
    for ax, met in zip(axes[0], metrics):
        for i, k in enumerate(bases):
            r = R[(R["base"] == k) & (R["metric"] == met)]
            if not len(r):
                continue
            r = r.iloc[0]
            lab, col, mk, _ = SPEC[k]
            sig = bool(r["p_bonf"] < 0.05)
            ax.errorbar(r["mean"], i, xerr=r["ci"], fmt=mk, color=col, ms=5,
                        mfc=col if sig else "white", mec=col, elinewidth=0.9, capsize=2)
        ax.axvline(0, color="#8a8984", lw=0.8, zorder=0)
        ax.set_title(ML[met])
        ax.set_xlabel("Framework $-$ baseline")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(range(len(bases)))
    axes[0][0].set_yticklabels([SPEC[k][0] for k in bases])
    axes[0][0].invert_yaxis()
    fig.suptitle(title, fontsize=8, y=1.0)
    fig.text(0.5, -0.02, "Negative = framework lower (better) on every metric shown. Filled: "
             "paired Wilcoxon significant after Bonferroni (p < 0.05); bars: 95% CI.",
             ha="center", fontsize=6.5, color="#5f5e5a")
    fig.tight_layout()
    _save(fig, out, name)


# ============================================================
# LATEX TABLES
# ============================================================

def _fmt(g, d):
    g = pd.Series(g).dropna()
    return "--" if not len(g) else f"{g.mean():.{d}f} $\\pm$ {g.std():.{d}f}"


def _tex(header: List[str], body: List[List[str]], caption, label, wide=True, groups=None):
    env = "table*" if wide else "table"
    out = [f"\\begin{{{env}}}[!t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
           "\\begin{tabular}{@{}l" + "c" * (len(header) - 1) + "@{}}", "\\toprule",
           " & ".join(header) + " \\\\", "\\midrule"]
    for row in body:
        if isinstance(row, str):                      # group heading
            out.append(f"\\multicolumn{{{len(header)}}}{{@{{}}l}}{{\\textit{{{row}}}}} \\\\")
        else:
            out.append(" & ".join(row) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}"]
    return "\n".join(out)


def tables(runs, lv, pools, trounds, ood, taus, out, n_runs, results):
    out.mkdir(parents=True, exist_ok=True)
    cols = [("total_transplants", "Tx", 1), ("total_high_cpra_transplants", "High-cPRA tx", 1),
            ("mean_F2", "$F_2$", 3), ("mean_F3", "$F_3$", 3), ("mean_F4", "$F_4$", 3),
            ("pof_mean", "$\\pof$ (\\%)", 2), ("delta", "$\\Delta$ (\\%)", 2)]
    head = ["Configuration"] + [c[1] for c in cols]
    written = []
    if len(runs):
        body = []
        for tau in taus:
            body.append(f"$\\tau = {tau}$ months")
            for k in MAIN_KEYS:
                g = _sel(runs, k, tau)
                if len(g):
                    body.append([SPEC[k][0]] + [_fmt(g[c], d) for c, _, d in cols])
        (out / "tab_main_final.tex").write_text(_tex(
            head, body, f"Trajectory results over 25 rounds (mean $\\pm$ sd over {n_runs} "
            "paired runs). Framework: volume-ILP-guided NSGA-III, 90\\% floor. Two-stage ILP: "
            "maximum transplants, then the fairest maximum matching. $\\Delta$: transplants "
            "lost over the trajectory relative to the myopic ILP.", "tab:main_final"),
            encoding="utf-8")
        written.append("tab_main_final.tex")
        body = []
        for tau in taus:
            if not len(_sel(runs, (FRAME, "floor80"), tau)):
                continue
            body.append(f"$\\tau = {tau}$ months")
            for k in [(FRAME, "floor100"), (FRAME, "base"), (FRAME, "floor80"),
                      (FRAME, "compromise"), ("twostage_ilp_sum", "base"), ("myopic_ilp", "base")]:
                g = _sel(runs, k, tau)
                if len(g):
                    body.append([SPEC[k][0]] + [_fmt(g[c], d) for c, _, d in cols])
        if body:
            (out / "tab_select.tex").write_text(_tex(
                head, body, "Selection rules over full trajectories (mean $\\pm$ sd, 30 paired "
                "runs) compared with the two-stage ILP and the myopic ILP.", "tab:select"),
                encoding="utf-8")
            written.append("tab_select.tex")
    if len(lv):
        body = []
        for lev in dict.fromkeys(lv["level"]):
            body.append(lev)
            sub = lv[lv["level"] == lev]
            for nm in dict.fromkeys(sub["method"]):
                g = sub[sub["method"] == nm]
                body.append([nm] + [_fmt(g[m], 3) for m in ("F2", "F3", "F4")]
                            + [_fmt(g["score"], 3) if g["score"].notna().any() else "--"])
        (out / "tab_three_levels.tex").write_text(_tex(
            ["Method", "$F_2$", "$F_3$", "$F_4$", "Score"], body,
            "Framework and ILP baselines at three levels of comparison (mean $\\pm$ sd). "
            "A: same pool, one round (30 pools). B: same pool in every round of the "
            "framework's trajectories ($\\tau = 12$, per-run means). C: each method clears its "
            "own pools ($\\tau = 12$, per-run means). Volume ILP, two-stage ILP and the "
            "100\\% pick/floor are constrained to the maximum number of transplants; the 90\\% "
            "pick/floor may give up to 10\\%. Score: $F_2/r_2 + F_3/r_3 + F_4/r_4$ with $r_k$ "
            "from the volume-ILP matching (floored at 0.01).", "tab:three_levels", wide=False),
            encoding="utf-8")
        written.append("tab_three_levels.tex")
    if len(pools) or len(trounds):
        body = []
        for title, d in [("Identical pools", pools),
                         ("Fixed trajectories, $\\tau = 12$",
                          trounds[trounds["tau"] == 12] if len(trounds) else trounds)]:
            if not len(d):
                continue
            d = d[d["F1_star"] > 0]
            body.append(title)
            for L in LEVELS:
                def mm(c):
                    return f"{d[c].mean():.3f}" if c in d and d[c].notna().any() else "--"

                def em(m, L):
                    v = exact_upto(d, m, L)
                    return f"{v.mean():.3f}" if v.notna().any() else "--"
                ex = d.get(f"exact_status_{L}", pd.Series(dtype=str))
                body.append([f"{L}\\%", f"{d[f'opt_{L}'].mean():.1f}" if f"opt_{L}" in d else "--",
                             mm(f"best_score_{L}"), mm(f"pick_score_{L}"), em("score", L),
                             mm(f"best_F3_{L}"), em("F3", L),
                             f"{(ex == 'time_limit').mean():.2f}" if len(ex) else "--"])
        (out / "tab_tradeoff.tex").write_text(_tex(
            ["Level $\\rho$", "Options", "Front best", "Rule pick", "Exact ILP, volumes $\\ge V$",
             "Front best $F_3$", "Exact $F_3$", "ILP time limit"], body,
            "Fairness available below maximum volume. Level $\\rho$ allows $V = \\lceil \\rho "
            "F_1^* \\rceil$ or more transplants. Options: distinct front solutions with "
            "$F_1 \\ge V$. Scores as in Table~\\ref{tab:three_levels} (lower = fairer); "
            "exact ILP: best two-stage stage-2 solution among the volumes solved from $V$ up to $F_1^*$ (one solve per level, time-limited; share of solves at the limit in the last column).",
            "tab:tradeoff", wide=True), encoding="utf-8")
        written.append("tab_tradeoff.tex")
    # computational cost
    crow = []
    p12 = Path(results) / "ablation_twostage_pools.csv"
    pool = pd.read_csv(p12) if p12.exists() else pd.DataFrame()

    def ps(col):
        return f"{pool[col].mean():.1f}" if col in pool else "--"

    def tl(col):
        return f"{(pool[col] == 'time_limit').mean():.2f}" if col in pool else "--"

    def rs(key, tau=12):
        g = _sel(runs, key, tau) if len(runs) else pd.DataFrame()
        return f"{g['sec_round'].mean():.2f}" if len(g) else "--"

    def os_(c):
        g = ood[ood["config"] == c] if len(ood) else pd.DataFrame()
        return f"{g['sec_round'].mean():.2f}" if len(g) else "--"

    crow.append(["Volume (myopic) ILP", ps("sec_ilp"), "--", rs(("myopic_ilp", "base")), os_("myopic_ilp")])
    crow.append(["Two-stage ILP (sum)", ps("ts_sum_sec"), tl("ts_sum_status"),
                 rs(("twostage_ilp_sum", "base")), os_("twostage_ilp_sum")])
    crow.append(["Two-stage ILP (F3)", ps("ts_f3_sec"), tl("ts_f3_status"),
                 rs(("twostage_ilp_f3", "base")), os_("twostage_ilp_f3")])
    crow.append(["Framework (NSGA-III + AOS, incl. ILP)", ps("sec_moea"), "--",
                 rs((FRAME, "base")), os_("dyn_nsga3_ucb1_myopic")])
    crow.append(["Dickerson lex", "--", "--", rs(("dickerson_lex", "base")), os_("dickerson_lex")])
    if len(trounds):
        crow.append(["Exact ILP at all levels (Section trade-off)", "--",
                     f"{(trounds['ts_status'] == 'time_limit').mean():.2f}",
                     f"{trounds['exact_sec_total'].mean():.2f}", "--"])
    (out / "tab_cost.tex").write_text(_tex(
        ["Method", "Identical pool (s)", "Time-limit share", "Trajectory s/round ($\\tau=12$)",
         "PrefLib s/round"], crow,
        "Computational cost. Identical pool: one round on a 75-vertex pool (stage-2 limit "
        "300 s). Trajectory: mean seconds per cleared round (stage-2 limit 60 s). All on one "
        "core; times include building the compatibility graph only where stated in the text.",
        "tab:cost", wide=True), encoding="utf-8")
    written.append("tab_cost.tex")
    if len(ood):
        import module14_tsg_preflib as m14
        ocols = [("total_transplants", "Tx", 2), ("total_high_cpra_transplants", "High-cPRA tx", 2),
                 ("mean_F2", "$F_2$", 3), ("mean_F3", "$F_3$", 3),
                 ("pof_mean", "$\\pof$ (\\%)", 2), ("delta", "$\\Delta$ (\\%)", 2)]
        order = ["dyn_nsga3_ucb1_myopic", "dyn_nsga3_linucb_myopic", "dyn_nsga3_ucb1_batched_mi3",
                 "twostage_ilp_sum", "twostage_ilp_f3", "myopic_ilp", "dickerson_lex"]
        names = {"dyn_nsga3_ucb1_myopic": "Framework UCB1", "dyn_nsga3_linucb_myopic": "Framework LinUCB",
                 "dyn_nsga3_ucb1_batched_mi3": "Framework UCB1 batched (max int. 3)"}
        body = [[names.get(c, m14.NAMES.get(c, c))] + [_fmt(ood[ood["config"] == c][col], d)
                                                        for col, _, d in ocols]
                for c in order if (ood["config"] == c).any()]
        (out / "tab_ood_final.tex").write_text(_tex(
            ["Configuration"] + [c[1] for c in ocols], body,
            "PrefLib 00036, 12 instances of 128--147 vertices, 10 paired runs each, "
            "$\\tau = 12$, 25 rounds (mean $\\pm$ sd). $F_4$ is not reported (hospitals are "
            "random on PrefLib).", "tab:ood_final", wide=False), encoding="utf-8")
        written.append("tab_ood_final.tex")
    for w in written:
        print(f"  {out / w}")


# ============================================================
# MAIN
# ============================================================

def main(argv=None) -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="Module 17: final figures and tables")
    ap.add_argument("--results", default="results")
    ap.add_argument("--taus", type=int, nargs="+", default=TAUS)
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--quick", action="store_true", help="read the --quick outputs")
    a = ap.parse_args(argv)
    if a.quick:
        a.results, a.taus = "results_quick", [12]
    _style()
    fig_dir = Path(a.results) / "paper" / FIG_DIR
    tab_dir = Path(a.results) / "paper" / TAB_DIR
    print("=" * 66 + "\nMODULE 17: final figures and tables (original framework)\n" + "=" * 66)
    m10 = _module10()
    rounds, runs = load_main(a, m10)
    taus = [t for t in a.taus if len(runs) and (runs["tau"] == t).any()]
    print(f"\nMain experiment: {len(runs)} runs, tau {taus}")
    if len(runs):
        fig_pof(rounds, taus, fig_dir)
        fig_batched(rounds, taus, fig_dir)
    lv = load_levels(a, m10, runs)
    print(f"Three-level comparison: {len(lv)} rows")
    fig_three_levels(lv, fig_dir)
    pools, trounds = load_tradeoff(a)
    print(f"Trade-off (module 16): {len(pools)} pools, {len(trounds)} rounds")
    fig_tradeoff(pools, trounds, fig_dir)
    if len(runs):
        fig_selection(runs, taus, fig_dir)
        for tau in taus:
            _forest(runs, "run", (FRAME, "base"), FOREST_BASES,
                    ["delta", "mean_F2", "mean_F3", "mean_F4"],
                    f"Framework (UCB1, 90% floor) vs baselines, synthetic, $\\tau$ = {tau}",
                    fig_dir, f"fig6_forest_tau{tau}",
                    lambda df, k, tau=tau: _sel(df, k, tau))
    ood = load_preflib(a)
    print(f"PrefLib: {len(ood)} runs")
    if len(ood):
        o = ood.assign(variant="base")
        _forest(o, "pair_id", (FRAME, "base"),
                [("twostage_ilp_sum", "base"), ("twostage_ilp_f3", "base"),
                 ("myopic_ilp", "base"), ("dickerson_lex", "base")],
                ["delta", "mean_F2", "mean_F3"],
                "Framework (UCB1, 90% floor) vs baselines, PrefLib 00036 ($\\tau$ = 12)",
                fig_dir, "fig7_forest_preflib", lambda df, k: _sel(df, k))
    print("\nTables:")
    tables(runs, lv, pools, trounds, ood, taus, tab_dir, a.runs, a.results)
    print("\n" + "=" * 66 + "\nMODULE 17 COMPLETE\n" + "=" * 66)


if __name__ == "__main__":
    main()
