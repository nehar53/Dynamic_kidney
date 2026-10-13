"""Module 1 — ANZKX-calibrated synthetic pool generator (Section 4.1).

Saidman et al. (2006) probabilistic model, calibrated with Table 2:
patient/donor ABO from the national distribution, a 30 % high-cPRA share,
and only *incompatible* pairs enter the exchange pool (ABO-incompatible or a
positive crossmatch with their own donor).
"""

from __future__ import annotations

import itertools

import numpy as np

from dkex.config import CENTRES, PoolConfig
from dkex.simulation.pool import ABO_TYPES, Pair, abo_compatible


class PoolGenerator:
    def __init__(self, cfg: PoolConfig, rng: np.random.Generator, centres=CENTRES):
        self.cfg = cfg
        self.rng = rng
        self.abo_p = np.array([cfg.abo_dist[t] for t in ABO_TYPES], dtype=float)
        self.abo_p /= self.abo_p.sum()
        w = np.array([cfg.centre_weights[c] for c in centres], dtype=float)
        self.centre_p = w / w.sum()
        self._ids = itertools.count()

    # ---------------------------------------------------------------- helpers
    def _abo(self) -> str:
        return ABO_TYPES[self.rng.choice(4, p=self.abo_p)]

    def _centre(self) -> int:
        return int(self.rng.choice(len(self.centre_p), p=self.centre_p))

    def _cpra(self, high: bool) -> float:
        thr = self.cfg.high_cpra_threshold
        if high:
            return float(self.rng.uniform(thr, 0.99))
        # Low-cPRA mixture: half unsensitised (~0), half moderately sensitised.
        if self.rng.random() < 0.5:
            return float(self.rng.uniform(0.0, 0.05))
        return float(self.rng.uniform(0.05, thr - 1e-6))

    # ------------------------------------------------------------------ API
    def make_pair(self, round_idx: int) -> Pair:
        high = self.rng.random() < self.cfg.high_cpra_rate
        cpra = self._cpra(high)
        while True:  # rejection-sample until the pair is incompatible
            p_abo, d_abo = self._abo(), self._abo()
            incompatible = (not abo_compatible(d_abo, p_abo)) or (self.rng.random() < cpra)
            if incompatible:
                break
        return Pair(
            pid=next(self._ids), donor_abo=d_abo, patient_abo=p_abo, cpra=cpra,
            centre=self._centre(), arrival_round=round_idx,
        )

    def make_altruist(self, round_idx: int) -> Pair:
        return Pair(
            pid=next(self._ids), donor_abo=self._abo(), centre=self._centre(),
            arrival_round=round_idx, altruist=True,
        )

    def make_arrival(self, round_idx: int) -> Pair:
        if self.rng.random() < self.cfg.altruist_rate:
            return self.make_altruist(round_idx)
        return self.make_pair(round_idx)

    def initial_pool(self, n: int | None = None) -> list[Pair]:
        n = self.cfg.initial_pool_size if n is None else n
        pairs = [self.make_arrival(0) for _ in range(n)]
        # Stagger initial waiting times so round 0 is not artificially uniform.
        for p in pairs:
            p.wait_months = float(self.rng.uniform(0, 6))
        return pairs

    def arrivals(self, round_idx: int) -> list[Pair]:
        k = self.rng.poisson(self.cfg.arrival_rate)
        return [self.make_arrival(round_idx) for _ in range(k)]
