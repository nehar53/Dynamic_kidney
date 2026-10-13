"""Figures for the PoF trajectory study (one PNG per figure)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed categorical order/colours so every figure reads the same way.
_ORDER = [
    "dyn_nsga3_ucb1_myopic", "dyn_nsga3_linucb_myopic",
    "dyn_nsga3_ucb1_batched", "dyn_nsga3_linucb_batched",
    "static_nsga2_ucb1", "static_nsga3_fixed", "dickerson_lex", "myopic_ilp",
]
_COLORS = dict(zip(_ORDER, [
    "#2a6fdb", "#e0702b", "#2a9d8f", "#c0392b", "#7f8c8d", "#8e6cc2", "#b8860b", "#333333",
]))


def _line(ax, df, metric, configs):
    for cfg in configs:
        d = df[df["config"] == cfg].sort_values("round")
        if d.empty or d[f"{metric}_mean"].isna().all():
            continue
        c = _COLORS.get(cfg, None)
        ax.plot(d["round"], d[f"{metric}_mean"], label=cfg, color=c, lw=1.8)
        ax.fill_between(d["round"], d[f"{metric}_mean"] - d[f"{metric}_ci95"],
                        d[f"{metric}_mean"] + d[f"{metric}_ci95"], color=c, alpha=0.15, lw=0)


def plot_all(traj: pd.DataFrame, arms: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    configs = [c for c in _ORDER if c in set(traj["config"])]
    for metric, ylabel in [("PoF", "Price of Fairness PoF_t (%)"), ("HV", "Hypervolume HV_t"),
                           ("F3", "High-cPRA gap F3"), ("F2", "Waiting-time Gini F2"),
                           ("F4", "Logistics disparity F4 (proxy)"),
                           ("n_high_cpra", "High-cPRA patients in pool")]:
        if f"{metric}_mean" not in traj:
            continue
        taus = sorted(traj["tau"].unique())
        fig, axes = plt.subplots(1, len(taus), figsize=(5 * len(taus), 3.8), sharey=True, squeeze=False)
        for ax, tau in zip(axes[0], taus):
            _line(ax, traj[traj["tau"] == tau], metric, configs)
            ax.set_title(f"tau = {tau:g} months")
            ax.set_xlabel("Clearing round t")
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel(ylabel)
        legend = {}
        for ax in axes[0]:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                legend.setdefault(lab, h)
        labels = [c for c in configs if c in legend]
        handles = [legend[c] for c in labels]
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        p = out / f"trajectory_{metric}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    if not arms.empty:
        arm_cols = [c for c in arms.columns if c.startswith("arm_")]
        for cfg in sorted(arms["config"].unique()):
            d = arms[arms["config"] == cfg].groupby("round")[arm_cols].mean()
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.stackplot(d.index, d.T.values, labels=[c.replace("arm_", "p_m=") for c in arm_cols])
            ax.set_ylim(0, 1)
            ax.set_xlabel("Clearing round t")
            ax.set_ylabel("Share of generations")
            ax.set_title(f"AOS arm selection — {cfg}")
            ax.legend(fontsize=7, loc="upper right")
            fig.tight_layout()
            p = out / f"arms_{cfg}.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(p)
    return paths
