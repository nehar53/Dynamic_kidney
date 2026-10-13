"""Module 12 — Statistical validation (Section 4.5.3).

Wilcoxon signed-rank tests on paired runs (common random numbers make runs
paired across configurations), Bonferroni correction, Cohen's d.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1) if d.size > 1 else 0.0
    return float(d.mean() / sd) if sd > 0 else 0.0


def pairwise_wilcoxon(
    df: pd.DataFrame, metric: str, group: str = "config", pair_on: str = "run",
    alpha: float = 0.05, comparisons: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank tests between every pair of groups."""
    wide = df.pivot_table(index=pair_on, columns=group, values=metric)
    groups = list(wide.columns)
    comparisons = comparisons or list(combinations(groups, 2))
    m = len(comparisons)
    rows = []
    for a, b in comparisons:
        if a not in wide or b not in wide:     # e.g. a policy that never cleared
            rows.append({"metric": metric, "a": a, "b": b, "n": 0, "p": np.nan,
                         "p_bonferroni": np.nan, "significant": False})
            continue
        sub = wide[[a, b]].dropna()
        x, y = sub[a].to_numpy(), sub[b].to_numpy()
        if len(sub) < 2 or np.allclose(x, y):
            p = 1.0
            stat = float("nan")
        else:
            stat, p = wilcoxon(x, y, zero_method="zsplit")
        p_adj = min(1.0, p * m)
        rows.append({
            "metric": metric, "a": a, "b": b, "n": len(sub),
            "mean_a": float(np.mean(x)) if len(x) else np.nan,
            "mean_b": float(np.mean(y)) if len(y) else np.nan,
            "W": stat, "p": p, "p_bonferroni": p_adj,
            "significant": p_adj < alpha, "cohens_d": cohens_d_paired(x, y) if len(x) > 1 else np.nan,
        })
    return pd.DataFrame(rows)
