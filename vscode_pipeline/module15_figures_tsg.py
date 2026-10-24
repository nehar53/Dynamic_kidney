import argparse
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import t as student_t, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# ============================================================
# MODULE 15: FIGURES AND TABLES FOR THE TSG FRAMEWORK (option A)
# ============================================================
#
# Reads the saved results of modules 10, 12, 13 (synthetic) and
# 9, 14 (PrefLib) and writes the paper figures and LaTeX tables for
# the two-stage-ILP-guided NSGA-III (TSG). No simulation is run.
#
# Output (the existing results/paper/figures folder is NOT touched):
#   results/paper/figures_tsg/   fig1_pof_per_round       PoF_t per round, tau 6/12/18
#                                fig2_myopic_vs_batched   PoF per cleared round +
#                                                         cumulative transplants
#                                fig3_tradeoff_tau12      Delta vs F2/F3/F4, all methods,
#                                                         TSG rho = 1.0 / 0.9 / 0.8
#                                fig4_forest_tau{6,12,18} paired differences TSG - baseline
#                                fig5_forest_preflib      same on PrefLib 00036
#                                (each as .png 300 dpi and .pdf)
#   results/paper/tables_tsg/    tab_main_tsg.tex, tab_rho_tsg.tex, tab_ood_tsg.tex
#
# Press Run after modules 13 and 14 (about 2 minutes), or:
#   python module15_figures_tsg.py --results results

FIG_DIR, TAB_DIR = "figures_tsg", "tables_tsg"
TAUS = [6, 12, 18]
RUNS = 30

# One fixed colour per method family (validated categorical palette,
# colour-blind safe); marker and line style are the second encoding.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
GREY = "#5f5e5a"

# (config, variant) -> label, colour, marker, line style
SPEC: Dict[Tuple[str, str], Tuple[str, str, str, str]] = {
    ("tsg_nsga3_ucb1_myopic", "score90"): ("TSG UCB1", BLUE, "o", "-"),
    ("tsg_nsga3_linucb_myopic", "score90"): ("TSG LinUCB", VIOLET, "s", "-"),
    ("tsg_nsga3_ucb1_batched", "score90"): ("TSG UCB1 batched", BLUE, "o", "--"),
    ("tsg_nsga3_linucb_batched", "score90"): ("TSG LinUCB batched", VIOLET, "s", "--"),
    ("tsg_nsga3_ucb1_myopic", "score100"): ("TSG UCB1, $\\rho$=1.0", BLUE, "o", "-"),
    ("tsg_nsga3_ucb1_myopic", "score80"): ("TSG UCB1, $\\rho$=0.8", BLUE, "o", "-"),
    ("tsg_nsga3_ucb1_myopic", "base"): ("TSG UCB1, original rule", BLUE, "D", ":"),
    ("dyn_nsga3_ucb1_myopic", "base"): ("Old UCB1", ORANGE, "^", "-"),
    ("dyn_nsga3_linucb_myopic", "base"): ("Old LinUCB", ORANGE, "v", "--"),
    ("dyn_nsga3_ucb1_batched", "base"): ("Old UCB1 batched", ORANGE, "^", "--"),
    ("dyn_nsga3_linucb_batched", "base"): ("Old LinUCB batched", ORANGE, "v", ":"),
    ("dyn_nsga3_ucb1_myopic", "score90"): ("Old UCB1, score rule", ORANGE, "<", ":"),
    ("static_nsga2_ucb1", "base"): ("Static NSGA-II+UCB1", GREEN, "P", "-"),
    ("static_nsga3_fixed", "base"): ("Static NSGA-III fixed", GREEN, "X", "--"),
    ("twostage_ilp_sum", "base"): ("Two-stage ILP (sum)", AQUA, "D", "-"),
    ("twostage_ilp_f3", "base"): ("Two-stage ILP (F3)", AQUA, "d", "--"),
    ("myopic_ilp", "base"): ("Myopic ILP", GREY, "*", "-"),
    ("dickerson_lex", "base"): ("Dickerson lex", YELLOW, "h", "-"),
}
TSG = ("tsg_nsga3_ucb1_myopic", "score90")
BASELINES = [("dyn_nsga3_ucb1_myopic", "base"), ("twostage_ilp_sum", "base"),
             ("twostage_ilp_f3", "base"), ("myopic_ilp", "base"), ("dickerson_lex", "base"),
             ("static_nsga2_ucb1", "base"), ("static_nsga3_fixed", "base")]
METRIC_LABEL = {"pof_vs_myopic_ilp": "$\\Delta$ (%)", "delta": "$\\Delta$ (%)",
                "mean_F2": "$F_2$", "mean_F3": "$F_3$", "mean_F4": "$F_4$"}


def _style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
        "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6,
        "xtick.color": "#5f5e5a", "ytick.color": "#5f5e5a",
        "axes.grid": True, "grid.color": "#e4e3de", "grid.linewidth": 0.5,
        "lines.linewidth": 1.4, "lines.markersize": 4, "savefig.bbox": "tight",
        "savefig.dpi": 300})


def _save(fig, out: Path, name: str) -> List[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.png", out / f"{name}.pdf"]
    for p in paths:
        fig.savefig(p)
    plt.close(fig)
    print(f"  {paths[0]}")
    return paths


def _ci(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2:
        return 0.0
    return float(student_t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


# ============================================================
# DATA
# ============================================================

def load_synthetic(a) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import module13_tsguided as m13
    ns = SimpleNamespace(taus=a.taus, select_taus=[12], runs=a.runs, jobs=1,
                         results=a.results, quick=a.quick)
    m10, groups = m13.build_args(ns)
    tasks = groups["main"][1] + groups["select"][1]
    rounds, runs = m13._load(m10, tasks)
    if len(rounds):
        rounds["cleared"] = rounds["cleared"].astype(bool)
    return rounds, runs


def load_preflib(a) -> pd.DataFrame:
    import module14_tsg_preflib as m14
    tsg_dir, ref_dir = Path(a.ood_tsg) / "raw", Path(a.ood_main) / "raw"
    tsg_files = sorted(tsg_dir.glob("*.csv"))
    if not tsg_files:
        return pd.DataFrame()
    datasets = {f.name.split("__")[0] for f in tsg_files}
    keep = set(m14.TSG_CFGS) | set(m14.REF_CFGS)
    tasks = []
    for f in tsg_files + sorted(ref_dir.glob("*.csv")):
        parts = f.stem.split("__")
        if len(parts) == 4 and parts[0] in datasets and parts[1] in keep \
                and parts[2] == f"tau{m14.TAU}":
            tasks.append({"path": str(f), "cfg": {"name": parts[1]}})
    _, runs = m14.per_run(tasks)
    return runs


def _sel(runs: pd.DataFrame, key, tau=None) -> pd.DataFrame:
    g = runs[(runs["config"] == key[0]) & (runs["variant"] == key[1])]
    return g if tau is None else g[g["tau"] == tau]


# ============================================================
# FIGURES
# ============================================================

def fig1_pof(rounds: pd.DataFrame, taus, out: Path):
    keys = [TSG, ("tsg_nsga3_linucb_myopic", "score90"), ("dyn_nsga3_ucb1_myopic", "base"),
            ("dyn_nsga3_linucb_myopic", "base"), ("dickerson_lex", "base")]
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
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, out, "fig1_pof_per_round")


def fig2_batched(rounds: pd.DataFrame, taus, out: Path):
    # LinUCB behaves like UCB1 (tables); old batched shown as reference
    keys = [TSG, ("tsg_nsga3_ucb1_batched", "score90"), ("dyn_nsga3_ucb1_batched", "base")]
    fig, axes = plt.subplots(2, len(taus), figsize=(7.16, 3.8), sharex=True, squeeze=False)
    for j, tau in enumerate(taus):
        for k in keys:
            g = rounds[(rounds["config"] == k[0]) & (rounds["variant"] == k[1])
                       & (rounds["tau"] == tau)]
            if not len(g):
                continue
            lab, col, mk, ls = SPEC[k]
            cl = g[g["cleared"]]
            if len(cl):
                s = cl.groupby("round")["PoF"].mean()
                axes[0][j].plot(s.index, s.values, color=col, ls=ls, marker=mk,
                                markevery=1 if "batched" in k[0] else 4, label=lab)
            f1 = g.assign(F1=g["F1"].where(g["cleared"], 0).fillna(0)).sort_values("round")
            cum = f1.groupby("run")["F1"].cumsum()
            m = cum.groupby(f1["round"]).mean()
            axes[1][j].plot(m.index, m.values, color=col, ls=ls, label=lab)
            if k == ("tsg_nsga3_ucb1_batched", "score90") and not len(cl):
                axes[0][j].text(0.5, 0.5, "batched: no round cleared",
                                transform=axes[0][j].transAxes, ha="center",
                                color="#5f5e5a", fontsize=7)
        axes[0][j].set_title(f"$\\tau$ = {tau} months")
        axes[1][j].set_xlabel("Round $t$")
        axes[1][j].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0][0].set_ylabel("PoF per cleared round (%)")
    axes[1][0].set_ylabel("Cumulative transplants")
    h, l = axes[1][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out, "fig2_myopic_vs_batched")


def fig3_tradeoff(runs: pd.DataFrame, out: Path, tau: int = 12):
    keys = [("myopic_ilp", "base"), ("dickerson_lex", "base"), ("static_nsga2_ucb1", "base"),
            ("static_nsga3_fixed", "base"), ("dyn_nsga3_ucb1_myopic", "base"),
            ("twostage_ilp_f3", "base"), ("twostage_ilp_sum", "base")]
    rho = [("tsg_nsga3_ucb1_myopic", v) for v in ("score100", "score90", "score80")]
    metrics = ["mean_F2", "mean_F3", "mean_F4"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), sharex=True)
    for ax, met in zip(axes, metrics):
        for k in keys:
            g = _sel(runs, k, tau)
            if not len(g):
                continue
            lab, col, mk, _ = SPEC[k]
            ax.errorbar(g["pof_vs_myopic_ilp"].mean(), g[met].mean(),
                        xerr=_ci(g["pof_vs_myopic_ilp"]), yerr=_ci(g[met]), fmt=mk,
                        color=col, ms=5, elinewidth=0.7, capsize=1.5, label=lab,
                        mec="white", mew=0.5)
        pts = [(_sel(runs, k, tau)) for k in rho]
        pts = [(g["pof_vs_myopic_ilp"].mean(), g[met].mean(), g) for g in pts if len(g)]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=BLUE, lw=1.0, zorder=1)
            for (x, y, g), txt in zip(pts, ["1.0", "0.9", "0.8"][:len(pts)]):
                ax.errorbar(x, y, xerr=_ci(g["pof_vs_myopic_ilp"]), yerr=_ci(g[met]),
                            fmt="o", color=BLUE, ms=5, elinewidth=0.7, capsize=1.5,
                            mec="white", mew=0.5)
                ax.annotate(f"$\\rho$={txt}", (x, y), xytext=(4, 3),
                            textcoords="offset points", fontsize=6.5, color="#2c2c2a")
        ax.set_ylabel(METRIC_LABEL[met] + " (lower = fairer)")
        ax.set_xlabel("$\\Delta$ vs myopic ILP (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=BLUE, marker="o", lw=1.0, mec="white"))
    labels.append("TSG UCB1 ($\\rho$ = 1.0, 0.9, 0.8)")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return _save(fig, out, f"fig3_tradeoff_tau{tau}")


def _forest(runs: pd.DataFrame, pair_col: str, tsg_key, base_keys, metrics, title,
            out: Path, name: str, select=None):
    """Mean paired difference TSG - baseline with 95% CI; filled = Bonferroni p < 0.05."""
    select = select or (lambda df, k: _sel(df, k))
    t = select(runs, tsg_key).set_index(pair_col)
    rows = []
    for k in base_keys:
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
        return []
    R = pd.DataFrame(rows)
    R["p_bonf"] = np.minimum(1.0, R["p"] * R["p"].notna().sum())
    bases = [k for k in base_keys if k in set(R["base"])]
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
        ax.set_title(METRIC_LABEL[met])
        ax.set_xlabel("TSG $-$ baseline")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(range(len(bases)))
    axes[0][0].set_yticklabels([SPEC[k][0] for k in bases])
    axes[0][0].invert_yaxis()
    fig.suptitle(title, fontsize=8, y=1.0)
    fig.text(0.5, -0.02, "Negative = TSG lower (better) on every metric shown. Filled marker: "
             "paired Wilcoxon significant after Bonferroni (p < 0.05); bars: 95% CI.",
             ha="center", fontsize=6.5, color="#5f5e5a")
    fig.tight_layout()
    return _save(fig, out, name)


# ============================================================
# LATEX TABLES
# ============================================================

def _fmt(g: pd.Series, d: int) -> str:
    return f"{g.mean():.{d}f} $\\pm$ {g.std():.{d}f}"


TABLE_COLS = [("total_transplants", "Tx", 1), ("total_high_cpra_transplants", "High-cPRA tx", 1),
              ("mean_F2", "$F_2$", 3), ("mean_F3", "$F_3$", 3), ("mean_F4", "$F_4$", 3),
              ("pof_mean", "$\\pof$ (\\%)", 2), ("pof_vs_myopic_ilp", "$\\Delta$ (\\%)", 2)]


def _tex_table(blocks: List[Tuple[Optional[str], pd.DataFrame, List]], cols, caption, label,
               wide=True) -> str:
    env = "table*" if wide else "table"
    out = [f"\\begin{{{env}}}[!t]", "\\centering", f"\\caption{{{caption}}}",
           f"\\label{{{label}}}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
           "\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", "\\toprule",
           "Configuration & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule"]
    for head, runs, keys in blocks:
        if head:
            out.append(f"\\multicolumn{{{len(cols) + 1}}}{{@{{}}l}}{{\\textit{{{head}}}}} \\\\")
        for k in keys:
            g = _sel(runs, k) if "variant" in runs else runs[runs["config"] == k]
            if not len(g):
                continue
            name = SPEC[k][0] if isinstance(k, tuple) else k
            cells = [(_fmt(g[c], d) if c in g and g[c].notna().any() else "--")
                     for c, _, d in cols]
            out.append(f"{name} & " + " & ".join(cells) + " \\\\")
        out.append("\\midrule")
    out[-1] = "\\bottomrule"
    out += ["\\end{tabular}", f"\\end{{{env}}}"]
    return "\n".join(out)


def tables(runs: pd.DataFrame, ood: pd.DataFrame, taus, out: Path, n_runs: int):
    out.mkdir(parents=True, exist_ok=True)
    main_keys = [TSG, ("tsg_nsga3_linucb_myopic", "score90"),
                 ("tsg_nsga3_ucb1_batched", "score90"), ("tsg_nsga3_linucb_batched", "score90"),
                 ("dyn_nsga3_ucb1_myopic", "base"), ("dyn_nsga3_linucb_myopic", "base"),
                 ("dyn_nsga3_ucb1_batched", "base"), ("dyn_nsga3_linucb_batched", "base"),
                 ("static_nsga2_ucb1", "base"), ("static_nsga3_fixed", "base"),
                 ("twostage_ilp_sum", "base"), ("twostage_ilp_f3", "base"),
                 ("myopic_ilp", "base"), ("dickerson_lex", "base")]
    written = []
    if len(runs):
        blocks = [(f"$\\tau = {tau}$ months", runs[runs["tau"] == tau], main_keys)
                  for tau in taus if (runs["tau"] == tau).any()]
        (out / "tab_main_tsg.tex").write_text(_tex_table(
            blocks, TABLE_COLS,
            f"Trajectory results over 25 rounds (mean $\\pm$ sd over {n_runs} paired runs). "
            "TSG: two-stage-ILP-guided NSGA-III, score rule, $\\rho = 0.9$; Old: "
            "volume-ILP-guided NSGA-III. $\\Delta$: transplants lost over the trajectory "
            "relative to the myopic ILP.", "tab:main_tsg"), encoding="utf-8")
        written.append(out / "tab_main_tsg.tex")
        rho_keys = [("tsg_nsga3_ucb1_myopic", "score100"), TSG,
                    ("tsg_nsga3_ucb1_myopic", "score80"), ("tsg_nsga3_ucb1_myopic", "base"),
                    ("dyn_nsga3_ucb1_myopic", "score90"), ("dyn_nsga3_ucb1_myopic", "base"),
                    ("twostage_ilp_sum", "base"), ("myopic_ilp", "base")]
        sub = runs[runs["tau"] == 12]
        if len(sub):
            (out / "tab_rho_tsg.tex").write_text(_tex_table(
                [(None, sub, rho_keys)], TABLE_COLS,
                f"Floor $\\rho$, selection rule and guide ($\\tau = 12$, {n_runs} paired "
                "runs; mean $\\pm$ sd).", "tab:rho_tsg"), encoding="utf-8")
            written.append(out / "tab_rho_tsg.tex")
    if len(ood):
        import module14_tsg_preflib as m14
        cols = [("total_transplants", "Tx", 2), ("total_high_cpra_transplants", "High-cPRA tx", 2),
                ("mean_F2", "$F_2$", 3), ("mean_F3", "$F_3$", 3),
                ("pof_mean", "$\\pof$ (\\%)", 2), ("delta", "$\\Delta$ (\\%)", 2)]
        order = [c for c in m14.ORDER if c in set(ood["config"])]
        txt = _tex_table([(None, ood, order)], cols,
                         "PrefLib 00036, 12 instances of 128--147 vertices, 10 paired runs "
                         "each, $\\tau = 12$, 25 rounds (mean $\\pm$ sd). $F_4$ is not "
                         "reported (hospitals are random on PrefLib).", "tab:ood_tsg", wide=False)
        for c in order:                             # PrefLib config names -> labels
            txt = txt.replace(f"\n{c} &", f"\n{m14.NAMES[c]} &")
        (out / "tab_ood_tsg.tex").write_text(txt, encoding="utf-8")
        written.append(out / "tab_ood_tsg.tex")
    for p in written:
        print(f"  {p}")


# ============================================================
# MAIN
# ============================================================

def main(argv=None) -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="Module 15: TSG figures and tables")
    ap.add_argument("--results", default="results")
    ap.add_argument("--ood-tsg", default=None, help="module 14 folder (default results/ood_tsg)")
    ap.add_argument("--ood-main", default=None, help="module 9 folder (default results/ood_main)")
    ap.add_argument("--taus", type=int, nargs="+", default=TAUS)
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--quick", action="store_true", help="read the --quick outputs")
    a = ap.parse_args(argv)
    if a.quick:
        a.results, a.taus = "results_quick", [12]
    a.ood_tsg = a.ood_tsg or str(Path(a.results) / "ood_tsg")
    a.ood_main = a.ood_main or str(Path(a.results) / "ood_main")
    _style()
    fig_dir = Path(a.results) / "paper" / FIG_DIR
    tab_dir = Path(a.results) / "paper" / TAB_DIR
    print("=" * 66 + "\nMODULE 15: figures and tables for the TSG framework\n" + "=" * 66)

    rounds, runs = load_synthetic(a)
    taus = [t for t in a.taus if len(runs) and (runs["tau"] == t).any()]
    print(f"\nSynthetic: {len(runs)} runs, tau {taus}")
    if len(runs):
        fig1_pof(rounds, taus, fig_dir)
        fig2_batched(rounds, taus, fig_dir)
        if (runs["tau"] == 12).any():
            fig3_tradeoff(runs, fig_dir, 12)
        metrics = ["pof_vs_myopic_ilp", "mean_F2", "mean_F3", "mean_F4"]
        for tau in taus:
            _forest(runs, "run", TSG, BASELINES, metrics,
                    f"TSG UCB1 myopic vs baselines, synthetic, $\\tau$ = {tau}",
                    fig_dir, f"fig4_forest_tau{tau}",
                    select=lambda df, k, tau=tau: _sel(df, k, tau))

    ood = load_preflib(a)
    print(f"\nPrefLib: {len(ood)} runs")
    if len(ood):
        import module14_tsg_preflib as m14
        for k in m14.NAMES:                        # PrefLib names -> same styles
            if (k, "base") not in SPEC and k.startswith("tsg"):
                base = SPEC.get((k.replace("_mi3", "").replace("_rho80", ""), "score90"))
                if base:
                    SPEC[(k, "base")] = (m14.NAMES[k],) + base[1:]
            elif (k, "base") not in SPEC:
                SPEC[(k, "base")] = (m14.NAMES[k], ORANGE, "^", "--")
        ood = ood.assign(variant="base")
        _forest(ood, "pair_id", ("tsg_nsga3_ucb1_myopic", "base"),
                [("dyn_nsga3_ucb1_myopic", "base"), ("twostage_ilp_sum", "base"),
                 ("twostage_ilp_f3", "base"), ("myopic_ilp", "base"),
                 ("dickerson_lex", "base")],
                ["delta", "mean_F2", "mean_F3"],
                "TSG UCB1 myopic vs baselines, PrefLib 00036 ($\\tau$ = 12)",
                fig_dir, "fig5_forest_preflib")

    print("\nTables:")
    tables(runs, ood.drop(columns="variant", errors="ignore") if len(ood) else ood,
           taus, tab_dir, a.runs)
    print("\n" + "=" * 66 + "\nMODULE 15 COMPLETE\n" + "=" * 66)


if __name__ == "__main__":
    main()
