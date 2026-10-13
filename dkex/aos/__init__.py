"""Module 8 — Adaptive operator selection (Section 3.5).

Arms are candidate mutation rates A = {0.01, 0.05, 0.10, 0.25}; the reward is
the hypervolume improvement produced by a generation.
"""

from dkex.aos.base import FixedOperator, OperatorSelector
from dkex.aos.linucb import LinUCB, build_context
from dkex.aos.ucb1 import SlidingWindowUCB1


def make_selector(method: str, n_arms: int, *, window: int = 15, alpha: float = 1.0,
                  fixed_arm: int = 0, context_dim: int = 6, seed: int = 0) -> OperatorSelector:
    if method == "ucb1":
        return SlidingWindowUCB1(n_arms, window=window, seed=seed)
    if method == "linucb":
        return LinUCB(n_arms, dim=context_dim, alpha=alpha, seed=seed)
    if method == "fixed":
        return FixedOperator(n_arms, fixed_arm)
    raise ValueError(f"unknown AOS method {method!r}")


__all__ = ["OperatorSelector", "FixedOperator", "SlidingWindowUCB1", "LinUCB",
           "build_context", "make_selector"]
