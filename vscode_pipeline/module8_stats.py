from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

# ============================================================
# MODULE 8: STATISTICAL VALIDATION
# Concept note Section 4.5.3:
#   Wilcoxon signed-rank test (paired runs)
#   Bonferroni correction for multiple comparisons
#   Cohen's d effect size (consistent with Paper 1)
# Paper 2 — Dynamic KEP Implementation
# ============================================================
#
# Pairing: run r of every configuration uses the same seed, the
# same initial pool, arrivals and competing exits (modules 2/7),
# so per-run metrics of two configurations are PAIRED samples ->
# Wilcoxon SIGNED-RANK test (not the rank-sum test).
#
# Reported for every comparison (all standard in IEEE TEVC):
#   W, p, Bonferroni-adjusted p (required), Holm-adjusted p
#   Cohen's d_z (paired) + magnitude label        (required)
#   rank-biserial correlation r_rb (non-parametric effect size
#     that matches the Wilcoxon test)
#   Hodges-Lehmann median difference + 95% bootstrap CI
# Plus:
#   Friedman omnibus test over >= 3 configurations (+ mean ranks)
#   PoF-trajectory trend test: per-run slope of PoF_t over rounds,
#     tested against 0 (does the price of fairness drift?)
#   LaTeX table export for the paper

ALPHA = 0.05
N_BOOTSTRAP = 5000


# ============================================================
# EFFECT SIZES
# ============================================================

def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d for paired samples (d_z): mean(a-b) / sd(a-b)."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    if diff.size < 2:
        return float("nan")
    sd = diff.std(ddof=1)
    return 0.0 if sd == 0 else float(diff.mean() / sd)


def cohens_d_pooled(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d with pooled SD (independent-samples version)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                 / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def effect_label(d: float) -> str:
    """Cohen (1988): |d| < 0.2 negligible, < 0.5 small, < 0.8 medium."""
    if d is None or np.isnan(d):
        return "n/a"
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def rank_biserial(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Matched-pairs rank-biserial correlation (Kerby 2014):
    r = (sum of positive ranks - sum of negative ranks) / total,
    in [-1, 1]. Positive = a tends to be larger than b.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def hodges_lehmann(a: Sequence[float], b: Sequence[float]) -> float:
    """Hodges-Lehmann estimate: median of Walsh averages of a-b."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if d.size == 0:
        return float("nan")
    i, j = np.triu_indices(d.size)
    return float(np.median((d[i] + d[j]) / 2.0))


def bootstrap_ci(a: Sequence[float], b: Sequence[float],
                 stat=np.mean, n_boot: int = N_BOOTSTRAP,
                 level: float = 0.95, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap CI of stat(a - b) over paired runs."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if d.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = stat(d[idx], axis=1)
    lo, hi = np.percentile(boots, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


# ============================================================
# TESTS AND CORRECTIONS
# ============================================================

def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """
    Two-sided Wilcoxon signed-rank test on paired samples.
    Pairs with a NaN are dropped. Identical samples -> p = 1.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    if len(a) < 2 or np.allclose(a, b):
        return float("nan"), 1.0
    stat, p = wilcoxon(a, b, zero_method="zsplit", alternative="two-sided")
    return float(stat), float(p)


def bonferroni(p_values: Sequence[float]) -> List[float]:
    """Bonferroni: min(1, p x m)  (required by the concept note)."""
    m = len(p_values)
    return [min(1.0, p * m) for p in p_values]


def holm(p_values: Sequence[float]) -> List[float]:
    """Holm step-down correction (same error control, more power)."""
    p = np.asarray(p_values, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, (m - rank) * p[k])
        adj[k] = min(1.0, running)
    return adj.tolist()


def friedman_test(runs: pd.DataFrame, metric: str, configs: List[str],
                  config_col: str = "config",
                  pair_col: str = "run_seed") -> Dict:
    """
    Friedman omnibus test: do >= 3 configurations differ at all?
    Also returns mean ranks (rank 1 = smallest value).
    """
    wide = runs.pivot_table(index=pair_col, columns=config_col,
                            values=metric)
    cols = [c for c in configs if c in wide]
    wide = wide[cols].dropna()
    if len(cols) < 3 or len(wide) < 2:
        return {"metric": metric, "chi2": np.nan, "p": np.nan,
                "n": len(wide), "mean_ranks": {}}
    chi2, p = friedmanchisquare(*[wide[c].to_numpy() for c in cols])
    ranks = wide.rank(axis=1).mean()
    return {"metric": metric, "chi2": float(chi2), "p": float(p),
            "n": len(wide), "mean_ranks": ranks.round(3).to_dict()}


# ============================================================
# PAIRWISE COMPARISON TABLES
# ============================================================

def compare_configs(
    runs: pd.DataFrame,
    metric: str,
    comparisons: Optional[List[Tuple[str, str]]] = None,
    config_col: str = "config",
    pair_col: str = "run_seed",
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """
    Paired Wilcoxon tests of `metric` between configurations.

    runs:        one row per (config, run) with a metric column
    comparisons: (config_a, config_b) pairs; default = all pairs.
                 Bonferroni and Holm corrections are applied over
                 this list (one family per metric, per tau).
    """
    wide = runs.pivot_table(index=pair_col, columns=config_col,
                            values=metric)
    if comparisons is None:
        comparisons = list(combinations(sorted(wide.columns), 2))

    rows = []
    for a, b in comparisons:
        if a not in wide or b not in wide:
            rows.append({"metric": metric, "config_a": a, "config_b": b,
                         "n": 0, "W": np.nan, "p": np.nan})
            continue
        sub = wide[[a, b]].dropna()
        x, y = sub[a].to_numpy(), sub[b].to_numpy()
        W, p = wilcoxon_paired(x, y)
        d = cohens_d_paired(x, y)
        lo, hi = bootstrap_ci(x, y)
        rows.append({
            "metric": metric, "config_a": a, "config_b": b, "n": len(sub),
            "mean_a": float(np.mean(x)) if len(x) else np.nan,
            "sd_a": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
            "mean_b": float(np.mean(y)) if len(y) else np.nan,
            "sd_b": float(np.std(y, ddof=1)) if len(y) > 1 else np.nan,
            "mean_diff": float(np.mean(x - y)) if len(x) else np.nan,
            "ci95_low": lo, "ci95_high": hi,
            "hl_median_diff": hodges_lehmann(x, y),
            "W": W, "p": p,
            "cohens_d": d, "effect": effect_label(d),
            "rank_biserial": rank_biserial(x, y) if len(x) else np.nan,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    valid = out["p"].notna()
    out["p_bonferroni"] = np.nan
    out["p_holm"] = np.nan
    if valid.any():
        out.loc[valid, "p_bonferroni"] = bonferroni(out.loc[valid, "p"].tolist())
        out.loc[valid, "p_holm"] = holm(out.loc[valid, "p"].tolist())
    out["significant"] = out["p_bonferroni"] < alpha
    out["direction"] = np.where(
        ~out["significant"], "n.s.",
        np.where(out["mean_diff"] > 0, "a > b", "a < b"))
    return out


def compare_many_metrics(
    runs: pd.DataFrame,
    metrics: List[str],
    comparisons: Optional[List[Tuple[str, str]]] = None,
    group_col: Optional[str] = "tau",
    **kwargs,
) -> pd.DataFrame:
    """compare_configs for several metrics, separately per tau."""
    tables = []
    groups = runs.groupby(group_col) if group_col else [(None, runs)]
    for g, sub in groups:
        for m in metrics:
            if m not in sub or sub[m].notna().sum() == 0:
                continue
            t = compare_configs(sub, m, comparisons, **kwargs)
            if t.empty:
                continue
            if group_col:
                t.insert(0, group_col, g)
            tables.append(t)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


# ============================================================
# PoF TRAJECTORY TREND
# ============================================================

def pof_trend(rounds: pd.DataFrame, config_col: str = "config",
              pair_col: str = "run_seed") -> pd.DataFrame:
    """
    Does PoF_t drift over the clearing rounds?
    For each run: OLS slope of PoF_t against the round index
    (cleared rounds with defined PoF only). Then per configuration
    and tau: mean slope (percentage points per round) and a
    one-sample Wilcoxon test of slope = 0.
    """
    sub = rounds[rounds["cleared"] & rounds["PoF"].notna()]
    slopes = []
    for (cfg, tau, run), g in sub.groupby([config_col, "tau", pair_col]):
        if g["round"].nunique() >= 3:
            slopes.append({"config": cfg, "tau": tau, "run": run,
                           "slope": np.polyfit(g["round"], g["PoF"], 1)[0]})
    s = pd.DataFrame(slopes)
    if s.empty:
        return s
    rows = []
    for (cfg, tau), g in s.groupby(["config", "tau"]):
        x = g["slope"].to_numpy()
        p = (wilcoxon(x, zero_method="zsplit").pvalue
             if len(x) >= 2 and not np.allclose(x, 0) else 1.0)
        rows.append({"config": cfg, "tau": tau, "n": len(x),
                     "mean_slope_pp_per_round": float(x.mean()),
                     "p_slope_zero": float(p),
                     "trend": ("rising" if p < ALPHA and x.mean() > 0 else
                               "falling" if p < ALPHA and x.mean() < 0 else
                               "stable")})
    return pd.DataFrame(rows)


# ============================================================
# DESCRIPTIVE TABLES (mean ± sd) AND LATEX EXPORT
# ============================================================

def summary_table(runs: pd.DataFrame, metrics: List[str],
                  by: Sequence[str] = ("config", "tau")) -> pd.DataFrame:
    """Mean and SD of each metric per configuration and tau."""
    cols = [m for m in metrics if m in runs]
    agg = runs.groupby(list(by))[cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def format_mean_sd(table: pd.DataFrame, metric: str,
                   decimals: int = 2) -> pd.Series:
    """'12.34 ± 1.56' strings; no data -> 'n/a'; single value -> '12.34 ± –'."""
    def fmt(m, s):
        if pd.isna(m):
            return "n/a"
        sd = "–" if pd.isna(s) else f"{s:.{decimals}f}"
        return f"{m:.{decimals}f} ± {sd}"
    return pd.Series([fmt(m, s) for m, s in
                      zip(table[f"{metric}_mean"], table[f"{metric}_std"])],
                     index=table.index)


def significance_marker(p_adj: float) -> str:
    """Stars for tables: * p<0.05, ** p<0.01, *** p<0.001 (adjusted)."""
    if p_adj is None or np.isnan(p_adj):
        return ""
    return "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else \
        "*" if p_adj < 0.05 else ""


def to_latex_table(table: pd.DataFrame, columns: List[str],
                   caption: str = "", label: str = "") -> str:
    """Simple booktabs LaTeX table from a DataFrame."""
    head = " & ".join(c.replace("_", r"\_").replace("±", r"$\pm$")
                      for c in columns) + r" \\"
    body = "\n".join(
        " & ".join(str(v).replace("_", r"\_").replace("±", r"$\pm$")
                   for v in row) + r" \\"
        for row in table[columns].itertuples(index=False))
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{caption}}}" if caption else "",
        rf"\label{{{label}}}" if label else "",
        r"\begin{tabular}{" + "l" * len(columns) + "}", r"\toprule",
        head, r"\midrule", body, r"\bottomrule", r"\end{tabular}",
        r"\end{table}"])


# ============================================================
# TEST MODULE 8
# ============================================================

if __name__ == "__main__":

    print("=" * 62)
    print("MODULE 8: STATISTICAL VALIDATION TEST")
    print("=" * 62)

    rng = np.random.default_rng(0)
    n = 100
    base = rng.normal(10, 2, n)               # shared run effect (pairing)
    runs = pd.DataFrame({
        "run_seed": np.tile(np.arange(n), 3),
        "tau": 12,
        "config": np.repeat(["A_myopic", "B_batched", "C_same"], n),
        "pof_mean": np.concatenate([
            base,
            base + 1.0 + rng.normal(0, 1, n),     # truly higher PoF
            base + rng.normal(0, 1, n),           # no real difference
        ]),
    })

    print("\n--- Friedman omnibus test (3 configurations) ---")
    fr = friedman_test(runs, "pof_mean", ["A_myopic", "B_batched", "C_same"])
    print(f"  chi2 = {fr['chi2']:.2f}, p = {fr['p']:.2e}, n = {fr['n']}")
    print(f"  mean ranks (1 = lowest PoF): {fr['mean_ranks']}")

    print("\n--- Paired Wilcoxon + Bonferroni/Holm (3 comparisons) ---")
    res = compare_configs(runs, "pof_mean")
    print(res[["config_a", "config_b", "mean_a", "mean_b", "mean_diff",
               "ci95_low", "ci95_high", "p_bonferroni", "p_holm",
               "significant", "cohens_d", "effect", "rank_biserial",
               "direction"]].round(4).to_string(index=False))
    print("\n  Expected: A vs B and B vs C significant, A vs C not.")

    print("\n--- Correction check (p = 0.01, 0.02, 0.04) ---")
    print(f"  Bonferroni: {bonferroni([0.01, 0.02, 0.04])}")
    print(f"  Holm:       {[round(x, 3) for x in holm([0.01, 0.02, 0.04])]}")

    print("\n--- PoF trajectory trend (synthetic rounds) ---")
    rows = []
    for cfg, slope in [("rising_cfg", 0.3), ("flat_cfg", 0.0)]:
        for run in range(30):
            for t in range(1, 21):
                rows.append({"config": cfg, "tau": 12, "run_seed": run,
                             "round": t, "cleared": True,
                             "PoF": 5 + slope * t + rng.normal(0, 1)})
    print(pof_trend(pd.DataFrame(rows)).round(4).to_string(index=False))

    print("\n--- Table for the paper ---")
    tab = summary_table(runs, ["pof_mean"])
    tab["PoF (mean ± sd)"] = format_mean_sd(tab, "pof_mean")
    print(tab[["config", "tau", "PoF (mean ± sd)"]].to_string(index=False))
    print("\nLaTeX:")
    print(to_latex_table(tab, ["config", "tau", "PoF (mean ± sd)"],
                         caption="Mean PoF over 100 runs.", label="tab:pof"))

    print("\n" + "=" * 62)
    print("MODULE 8 COMPLETE")
    print("Ready for Module 9: PrefLib OOD Validation")
    print("=" * 62)
