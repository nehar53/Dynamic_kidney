from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ============================================================
# MODULE 8: STATISTICAL VALIDATION
# Wilcoxon signed-rank test (paired runs)
# Bonferroni correction for multiple comparisons
# Cohen's d effect size
# Paper 2 — Dynamic KEP Implementation (Section 4.5.3)
# ============================================================
#
# Pairing: run r of every configuration uses the same seed,
# initial pool, arrivals and competing exits (module 7's
# PairedPoolEvolution). So per-run metrics of two configurations
# form PAIRED samples -> Wilcoxon signed-rank (not rank-sum).

ALPHA = 0.05


# ============================================================
# EFFECT SIZE
# ============================================================

def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Cohen's d for paired samples (d_z):
        d = mean(a - b) / sd(a - b)
    """
    diff = np.asarray(a, float) - np.asarray(b, float)
    if diff.size < 2:
        return float("nan")
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(diff.mean() / sd)


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
    """Cohen (1988) thresholds."""
    if np.isnan(d):
        return "n/a"
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


# ============================================================
# WILCOXON + BONFERRONI
# ============================================================

def wilcoxon_paired(a: Sequence[float],
                    b: Sequence[float]) -> Tuple[float, float]:
    """
    Two-sided Wilcoxon signed-rank test.
    Returns (W statistic, p-value). Identical samples -> p = 1.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2 or np.allclose(a, b):
        return float("nan"), 1.0
    stat, p = wilcoxon(a, b, zero_method="zsplit",
                       alternative="two-sided")
    return float(stat), float(p)


def bonferroni(p_values: Sequence[float]) -> List[float]:
    """Bonferroni-adjusted p-values: min(1, p x m)."""
    m = len(p_values)
    return [min(1.0, p * m) for p in p_values]


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
    comparisons: list of (config_a, config_b); default = all pairs
    Bonferroni correction is applied over this list.
    """
    wide = runs.pivot_table(index=pair_col, columns=config_col,
                            values=metric)
    if comparisons is None:
        comparisons = list(combinations(sorted(wide.columns), 2))

    rows = []
    for a, b in comparisons:
        if a not in wide or b not in wide:
            rows.append({"metric": metric, "config_a": a,
                         "config_b": b, "n": 0, "W": np.nan,
                         "p": np.nan})
            continue
        sub = wide[[a, b]].dropna()
        W, p = wilcoxon_paired(sub[a], sub[b])
        d = cohens_d_paired(sub[a], sub[b])
        rows.append({
            "metric": metric, "config_a": a, "config_b": b,
            "n": len(sub),
            "mean_a": sub[a].mean(), "sd_a": sub[a].std(ddof=1),
            "mean_b": sub[b].mean(), "sd_b": sub[b].std(ddof=1),
            "median_diff": float(np.median(sub[a] - sub[b]))
            if len(sub) else np.nan,
            "W": W, "p": p, "cohens_d": d,
            "effect": effect_label(d),
        })

    out = pd.DataFrame(rows)
    if out.empty:                        # no comparisons requested
        return out
    valid = out["p"].notna()
    out["p_bonferroni"] = np.nan
    out.loc[valid, "p_bonferroni"] = bonferroni(out.loc[valid, "p"])
    out["significant"] = out["p_bonferroni"] < alpha
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
    return pd.concat(tables, ignore_index=True) if tables \
        else pd.DataFrame()


# ============================================================
# DESCRIPTIVE TABLE (mean ± sd) FOR THE PAPER
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
    """'12.34 ± 1.56' strings for a LaTeX/Word table.
    No data -> 'n/a'; a single value (sd undefined) -> '12.34 ± –'."""
    def fmt(m, s):
        if pd.isna(m):
            return "n/a"
        sd = "–" if pd.isna(s) else f"{s:.{decimals}f}"
        return f"{m:.{decimals}f} ± {sd}"
    return pd.Series([fmt(m, s) for m, s in
                      zip(table[f"{metric}_mean"], table[f"{metric}_std"])],
                     index=table.index)


# ============================================================
# TEST MODULE 8
# ============================================================

if __name__ == "__main__":

    print("=" * 55)
    print("MODULE 8: STATISTICAL VALIDATION TEST")
    print("=" * 55)

    rng = np.random.default_rng(0)
    n = 100
    base = rng.normal(10, 2, n)              # shared run effect
    runs = pd.DataFrame({
        "run_seed": np.tile(np.arange(n), 3),
        "tau": 12,
        "config": np.repeat(["A_myopic", "B_batched", "C_same"], n),
        "pof_mean": np.concatenate([
            base,
            base - 1.0 + rng.normal(0, 1, n),    # truly lower
            base + rng.normal(0, 1, n),          # no real difference
        ]),
    })

    res = compare_configs(runs, "pof_mean")
    print("\nPaired Wilcoxon + Bonferroni (3 comparisons):")
    print(res[["config_a", "config_b", "mean_a", "mean_b", "p",
               "p_bonferroni", "significant", "cohens_d",
               "effect"]].round(4).to_string(index=False))

    print("\nExpected: A vs B significant (large effect),")
    print("          A vs C not significant.")

    tab = summary_table(runs, ["pof_mean"])
    tab["pof_mean (mean ± sd)"] = format_mean_sd(tab, "pof_mean")
    print("\n" + tab[["config", "tau",
                      "pof_mean (mean ± sd)"]].to_string(index=False))

    print("\n" + "=" * 55)
    print("MODULE 8 COMPLETE")
    print("Ready for Module 9: PrefLib OOD Validation")
    print("=" * 55)
