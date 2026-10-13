"""Stateless sliding-window UCB1 (Section 3.5.1, Eq. 10).

    a* = argmax_a  Q(a) + sqrt(2 ln t / N(a))

Q, N and t are computed over the last W pulls (W = 15).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from dkex.aos.base import OperatorSelector


class SlidingWindowUCB1(OperatorSelector):
    name = "ucb1"

    def __init__(self, n_arms: int, window: int = 15, seed: int = 0):
        super().__init__(n_arms)
        self.window = window
        self.history: deque[tuple[int, float]] = deque(maxlen=window)
        self.rng = np.random.default_rng(seed)

    def _stats(self):
        N = np.zeros(self.n_arms)
        R = np.zeros(self.n_arms)
        for a, r in self.history:
            N[a] += 1
            R[a] += r
        return N, R

    def select(self) -> int:
        N, R = self._stats()
        unplayed = np.nonzero(N == 0)[0]
        if unplayed.size:
            return int(self.rng.choice(unplayed))
        t = N.sum()
        ucb = R / N + np.sqrt(2.0 * np.log(t) / N)
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        self.history.append((arm, float(reward)))

    def reset(self) -> None:
        super().reset()
        self.history.clear()
