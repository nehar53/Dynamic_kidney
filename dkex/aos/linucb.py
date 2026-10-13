"""Contextual LinUCB (Section 3.5.2, Eqs. 11-12; Li et al. 2010, disjoint model).

Context s_t = [n_t, p_PRA, x_t, t, Gini_t, PoF_{t-1}], each scaled to [0, 1].

    a* = argmax_a  theta_a^T s_t + alpha * sqrt(s_t^T A_a^{-1} s_t)
"""

from __future__ import annotations

import numpy as np

from dkex.aos.base import OperatorSelector


def build_context(
    pool_size: int,
    high_cpra_share: float,
    competing_exits: int,
    round_idx: int,
    total_rounds: int,
    gini_wait: float,
    prev_pof: float,
    pool_scale: float = 180.0,
) -> np.ndarray:
    """Context vector of Eq. 11 normalised to [0, 1]."""
    s = np.array(
        [
            pool_size / pool_scale,
            high_cpra_share,
            competing_exits / max(1.0, pool_size),
            round_idx / max(1, total_rounds),
            gini_wait,
            prev_pof / 100.0,
        ],
        dtype=float,
    )
    return np.clip(s, 0.0, 1.0)


class LinUCB(OperatorSelector):
    name = "linucb"

    def __init__(self, n_arms: int, dim: int = 6, alpha: float = 1.0, seed: int = 0):
        super().__init__(n_arms)
        self.dim, self.alpha = dim, alpha
        self.rng = np.random.default_rng(seed)
        self.reset()
        self.context = np.zeros(dim)

    def reset(self) -> None:
        super().reset()
        self.A = np.stack([np.eye(self.dim) for _ in range(self.n_arms)])
        self.b = np.zeros((self.n_arms, self.dim))

    def set_context(self, context: np.ndarray) -> None:
        context = np.asarray(context, dtype=float)
        if context.shape != (self.dim,):
            raise ValueError(f"context must have shape ({self.dim},)")
        self.context = context

    def select(self) -> int:
        s = self.context
        scores = np.empty(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            scores[a] = theta @ s + self.alpha * np.sqrt(s @ A_inv @ s)
        best = np.flatnonzero(np.isclose(scores, scores.max()))
        return int(self.rng.choice(best))

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        s = self.context
        self.A[arm] += np.outer(s, s)
        self.b[arm] += reward * s
