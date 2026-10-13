"""Module 9 — Clearing strategies (Section 3.7).

* Myopic  (pi_M): clear at every round.
* Batched (pi_B): clear only once the pool holds n* (= 90) patient pairs.

The clearing policy decides *whether* to match in round t; the AOS mechanism
operates *inside* a clearing round and is independent of this decision.
"""

from __future__ import annotations

from dkex.config import ClearingConfig
from dkex.simulation.pool import PoolState


class ClearingPolicy:
    name = "base"

    def should_clear(self, pool: PoolState, round_idx: int, last_round: bool) -> bool:
        raise NotImplementedError


class MyopicPolicy(ClearingPolicy):
    name = "myopic"

    def should_clear(self, pool, round_idx, last_round):
        return True


class BatchedPolicy(ClearingPolicy):
    name = "batched"

    def __init__(self, threshold: int = 90, max_interval: int | None = None):
        self.threshold = threshold
        self.max_interval = max_interval
        self._last = -1

    def should_clear(self, pool, round_idx, last_round):
        due = self.max_interval is not None and round_idx - self._last >= self.max_interval
        clear = pool.n_patients() >= self.threshold or due
        if clear:
            self._last = round_idx
        return clear


def make_policy(cfg: ClearingConfig) -> ClearingPolicy:
    if cfg.policy == "myopic":
        return MyopicPolicy()
    if cfg.policy == "batched":
        return BatchedPolicy(cfg.batch_threshold, cfg.batch_max_interval)
    raise ValueError(f"unknown clearing policy {cfg.policy!r}")
