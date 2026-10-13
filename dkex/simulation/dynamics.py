"""Module 4 — Dynamic pool model (Section 3.1, Eq. 1).

    V_{t+1} = V_t \\ M_t \\ W_t \\ X_t  U  A_t

* M_t — pairs matched in round t (supplied by the clearing policy)
* X_t — competing exits to the deceased-donor pathway, Bernoulli(mu_d) per pair
* W_t — withdrawals once waiting time exceeds tau months
* A_t — Poisson(lambda) arrivals

All stochastic events are pre-generated (arrivals) or hashed on
(seed, pair id, round) (competing exits), so every configuration compared in
the same run sees *identical* exogenous randomness and differs only through
its matching decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dkex.config import Config
from dkex.simulation.compatibility import hash_uniform
from dkex.simulation.generator import PoolGenerator
from dkex.simulation.pool import Pair, PoolState


@dataclass
class RoundEvents:
    round: int
    matched: list[int] = field(default_factory=list)
    deceased_exits: list[int] = field(default_factory=list)
    withdrawn: list[int] = field(default_factory=list)
    arrivals: list[int] = field(default_factory=list)
    # Per-exit detail so equity metrics can track who leaves unmatched.
    withdrawn_high_cpra: int = 0
    deceased_high_cpra: int = 0


class DynamicPool:
    def __init__(self, cfg: Config, seed: int, source=None):
        """``source`` must offer ``initial_pool()`` and ``arrivals(t)``; the
        default is the ANZKX-calibrated Saidman generator."""
        self.cfg = cfg
        self.seed = int(seed)
        gen = source or PoolGenerator(cfg.pool, np.random.default_rng(self.seed))
        self.state = PoolState(round=0)
        self.state.add(gen.initial_pool())
        # Pre-generate the arrival stream A_1..A_T.
        self._arrivals = [gen.arrivals(t) for t in range(1, cfg.dynamics.rounds + 2)]
        self.history: list[RoundEvents] = []

    @property
    def round(self) -> int:
        return self.state.round

    def step(self, matched_pids: list[int]) -> RoundEvents:
        d = self.cfg.dynamics
        thr = self.cfg.pool.high_cpra_threshold
        t = self.state.round
        ev = RoundEvents(round=t, matched=list(matched_pids))
        self.state.remove(matched_pids)

        # X_t: competing exit to the deceased-donor waitlist.
        pts = self.state.patients
        if pts:
            pids = np.array([p.pid for p in pts])
            u = hash_uniform(self.seed, pids, np.full(len(pids), t), salt=3)
            rate = np.array(
                [d.deceased_exit_rate * (d.deceased_exit_high_cpra_multiplier if p.cpra >= thr else 1.0) for p in pts]
            )
            exits = pids[u < rate].tolist()
            ev.deceased_high_cpra = sum(self.state.pairs[i].cpra >= thr for i in exits)
            ev.deceased_exits = exits
            self.state.remove(exits)

        # Waiting time advances; W_t: withdrawal after tau.
        for p in self.state.pairs.values():
            p.wait_months += d.round_length_months
        tau = d.withdrawal_threshold_months
        withdrawn = [
            p.pid for p in self.state.pairs.values()
            if p.wait_months > tau and (d.altruists_withdraw or not p.altruist)
        ]
        ev.withdrawn_high_cpra = sum(self.state.pairs[i].is_high_cpra(thr) for i in withdrawn)
        ev.withdrawn = withdrawn
        self.state.remove(withdrawn)

        # A_t: arrivals (deep-copied so runs never share mutable Pair objects).
        new = [Pair(**vars(p)) for p in self._arrivals[t]] if t < len(self._arrivals) else []
        ev.arrivals = [p.pid for p in new]
        self.state.add(new)
        self.state.round = t + 1
        self.history.append(ev)
        return ev
