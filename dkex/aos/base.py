from __future__ import annotations

import numpy as np


class OperatorSelector:
    """Bandit interface used by the MOEA once per generation."""

    name = "base"

    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.pulls = np.zeros(n_arms, dtype=int)  # lifetime counts (for reporting)

    def set_context(self, context: np.ndarray) -> None:  # contextual bandits only
        pass

    def select(self) -> int:
        raise NotImplementedError

    def update(self, arm: int, reward: float) -> None:
        self.pulls[arm] += 1

    def reset(self) -> None:
        self.pulls[:] = 0


class FixedOperator(OperatorSelector):
    """No AOS: always the same mutation rate (baseline 2)."""

    name = "fixed"

    def __init__(self, n_arms: int, arm: int):
        super().__init__(n_arms)
        self.arm = arm

    def select(self) -> int:
        return self.arm
