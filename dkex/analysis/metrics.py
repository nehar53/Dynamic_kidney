"""Module 12 — Per-round and trajectory metrics (Section 4.5.1-4.5.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

PER_ROUND = ["HV", "PoF", "F2", "F3", "F4", "F1", "F1_ilp", "pool_size", "n_high_cpra"]
TRAJECTORY = [
    "pof_mean", "pof_var", "total_transplants", "total_high_cpra_transplants",
    "high_cpra_accumulation", "total_withdrawn", "total_withdrawn_high_cpra",
    "total_deceased_exits", "mean_F2", "mean_F3", "mean_F4", "mean_HV",
]


def round_trajectories(rounds: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95 % normal CI of each per-round metric by (config, tau, round)."""
    cols = [c for c in PER_ROUND if c in rounds]
    g = rounds.groupby(["config", "tau", "round"])[cols]
    mean, sem, n = g.mean(), g.sem(), g.count()
    out = mean.add_suffix("_mean").join((1.96 * sem).add_suffix("_ci95"))
    out["n_runs"] = n[cols[0]]
    return out.reset_index()


def trajectory_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Trajectory metrics (mean ± sd over runs) by (config, tau)."""
    cols = [c for c in TRAJECTORY if c in runs]
    agg = runs.groupby(["config", "tau"])[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def arm_distribution(rounds: pd.DataFrame) -> pd.DataFrame:
    """Share of generations spent on each mutation-rate arm, per (config, tau, round)."""
    arms = [c for c in rounds.columns if c.startswith("arm_")]
    if not arms:
        return pd.DataFrame()
    sub = rounds.dropna(subset=arms, how="all")
    g = sub.groupby(["config", "tau", "round"])[arms].sum()
    share = g.div(g.sum(axis=1).replace(0, np.nan), axis=0)
    return share.reset_index()


def clearing_comparison(rounds: pd.DataFrame) -> pd.DataFrame:
    """PoF_t under myopic vs batched clearing, paired by AOS method (Section 4.5.2)."""
    sub = rounds[rounds["config"].str.startswith("dyn_nsga3_") & rounds["cleared"]].copy()
    if sub.empty:
        return sub
    sub["aos"] = sub["config"].str.split("_").str[2]
    sub["clearing"] = sub["config"].str.split("_").str[3]
    return (sub.groupby(["aos", "clearing", "tau", "round"])["PoF"].mean()
               .unstack("clearing").reset_index())
