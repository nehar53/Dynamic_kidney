from typing import Dict, List, Optional

import numpy as np

from module1_generator import ANZKX_NODES, Patient, Pool, generate_arrivals

# ============================================================
# MODULE 2: DYNAMIC POOL EVOLUTION
# Implements the pool equation (concept note Eq. 1):
#   V_{t+1} = V_t \ M_t \ W_t \ X_t  U  A_t
# Paper 2 — Dynamic KEP Implementation
# ============================================================
#
#   M_t  pairs matched and transplanted in round t
#   X_t  competing exits: patient receives a deceased-donor kidney
#        (ANZKX is a living-donor programme; deceased donors are
#        NOT pool nodes, they are a parallel exit pathway)
#   W_t  withdrawals after waiting tau months (sensitivity
#        parameter tau in {6, 12, 18}; Ferrari et al. 2012 report
#        mean KPD participation of 153 +/- 92 days)
#   A_t  new arrivals, |A_t| ~ Poisson(lambda)
#
# Order inside one round t (1 round = 1 month):
#   1. remove M_t           (decided on V_t by the optimiser)
#   2. remove X_t           (competing exit, prob mu_d per patient)
#   3. waiting time + 1     (for everyone still in the pool)
#   4. remove W_t           (waiting_time >= tau)
#   5. add A_t              (arrive with waiting_time 0)
#
# Paired random numbers (common random numbers):
#   Every random event is tied to (seed, round) or
#   (seed, pair_id, round) instead of one shared RNG stream.
#   So two configurations with the same seed see IDENTICAL
#   arrivals and IDENTICAL exit draws, and differ only through
#   their matching decisions. This is what makes the paired
#   Wilcoxon signed-rank test (module 8) valid.

# --- Simulation Parameters ---

# Withdrawal threshold tau (sensitivity parameter, months)
TAU_VALUES = {
    'tau_6':  6,
    'tau_12': 12,
    'tau_18': 18
}

# Competing exit rate mu_d per patient per round (month).
# Modelling assumption estimated from DonateLife 2024 aggregate
# statistics; not directly observed from ANZKX.
MU_D = 0.02

# Highly sensitised patients receive far fewer deceased-donor
# offers. 1.0 = same rate for everyone (concept note as written).
# Set e.g. 0.3 as a sensitivity analysis.
MU_D_HIGH_CPRA_MULTIPLIER = 1.0

# Poisson arrival rate lambda (pairs + altruists per round).
# Calibrated so the pool stays in the ANZKX range of 60-90.
LAMBDA_ARRIVALS = 6.0

# Altruistic donors who are not used within tau months donate
# to the deceased-donor waitlist instead (leave the pool).
ALTRUISTS_WITHDRAW = True


class PoolEvolution:
    """Dynamic pool V_t evolving over clearing rounds."""

    def __init__(
        self,
        initial_pool: Pool,
        next_pair_id: int,
        tau_months: int = 12,
        mu_d: float = MU_D,
        lambda_arrivals: float = LAMBDA_ARRIVALS,
        seed: int = 42,
        mu_d_high_cpra_multiplier: float = MU_D_HIGH_CPRA_MULTIPLIER,
        altruists_withdraw: bool = ALTRUISTS_WITHDRAW,
        base_seed: Optional[int] = None,
    ):
        self.pool = initial_pool
        self.next_pair_id = next_pair_id
        self.tau = tau_months
        self.mu_d = mu_d
        self.mu_d_high = mu_d * mu_d_high_cpra_multiplier
        self.lambda_arrivals = lambda_arrivals
        self.altruists_withdraw = altruists_withdraw
        self.base_seed = int(seed if base_seed is None else base_seed)
        self.current_round = 0

        # Tracking across rounds
        self.history = {
            'pool_sizes': [],
            'patients': [],
            'arrivals': [],
            'withdrawals': [],
            'competing_exits': [],
            'matched': [],
            'high_cpra_counts': [],
            'withdrawn_high_cpra': [],
            'exits_high_cpra': [],
            'round_numbers': []
        }

    # --------------------------------------------------------
    # Paired random draws
    # --------------------------------------------------------

    def _uniform(self, *keys: int) -> float:
        """U(0,1) that depends only on (seed, keys)."""
        return float(np.random.default_rng(
            [self.base_seed, *[int(k) for k in keys]]).random())

    def _round_rng(self, stream: int) -> np.random.RandomState:
        """RandomState that depends only on (seed, round, stream)."""
        s = np.random.default_rng(
            [self.base_seed, self.current_round, stream]
        ).integers(0, 2 ** 31 - 1)
        return np.random.RandomState(int(s))

    # --------------------------------------------------------
    # Pool equation components
    # --------------------------------------------------------

    def apply_matches(self, matched_pair_ids: List[int]) -> List[Patient]:
        """M_t: remove matched pairs (and altruists used in the matching)."""
        matched_set = set(matched_pair_ids)
        matched = [p for p in self.pool.pairs if p.pair_id in matched_set]
        self.pool.pairs = [p for p in self.pool.pairs
                           if p.pair_id not in matched_set]
        return matched

    def compute_competing_exits(self) -> List[Patient]:
        """
        X_t: patients leaving for a deceased-donor kidney.
        Each patient exits with probability mu_d this round
        (mu_d x multiplier if highly sensitised).
        Altruistic donors have no patient and cannot exit this way.
        """
        exits, remaining = [], []
        for pair in self.pool.pairs:
            if pair.is_altruistic:
                remaining.append(pair)
                continue
            rate = self.mu_d_high if pair.is_highly_sensitised else self.mu_d
            if self._uniform(pair.pair_id, self.current_round, 3) < rate:
                exits.append(pair)
            else:
                remaining.append(pair)
        self.pool.pairs = remaining
        return exits

    def increment_waiting_times(self):
        """Everyone still in the pool has waited one more round."""
        for pair in self.pool.pairs:
            pair.waiting_time += 1

    def compute_withdrawals(self) -> List[Patient]:
        """
        W_t: pairs withdrawn after waiting tau months.
        tau is a sensitivity parameter, not an ANZKX policy.
        """
        withdrawn, remaining = [], []
        for pair in self.pool.pairs:
            if pair.waiting_time >= self.tau and \
                    (self.altruists_withdraw or not pair.is_altruistic):
                withdrawn.append(pair)
            else:
                remaining.append(pair)
        self.pool.pairs = remaining
        return withdrawn

    def compute_arrivals(self) -> List[Patient]:
        """
        A_t: |A_t| ~ Poisson(lambda) new arrivals (pairs and
        altruistic donors, generated by module 1).
        Both the count and the arrivals themselves depend only on
        (seed, round), so every configuration gets the same A_t.
        """
        n_arrivals = int(self._round_rng(1).poisson(self.lambda_arrivals))
        new_pairs, self.next_pair_id = generate_arrivals(
            n_arrivals=n_arrivals,
            next_pair_id=self.next_pair_id,
            arrival_round=self.current_round,
            rng=self._round_rng(2),
        )
        self.pool.pairs.extend(new_pairs)
        return new_pairs

    # --------------------------------------------------------
    # One round
    # --------------------------------------------------------

    def evolve_round(self, matched_pair_ids: Optional[List[int]] = None) -> Dict:
        """
        Execute one round of the pool equation and return
        statistics for the round.
        """
        self.current_round += 1
        self.pool.round_number = self.current_round

        matched = self.apply_matches(matched_pair_ids or [])     # M_t
        exits = self.compute_competing_exits()                    # X_t
        self.increment_waiting_times()
        withdrawn = self.compute_withdrawals()                    # W_t
        arrivals = self.compute_arrivals()                        # A_t

        patients = self.pool.patients()
        high_cpra = sum(p.is_highly_sensitised for p in patients)
        withdrawn_high = sum(p.is_highly_sensitised and not p.is_altruistic
                             for p in withdrawn)
        exits_high = sum(p.is_highly_sensitised for p in exits)

        round_stats = {
            'round': self.current_round,
            'pool_size': self.pool.size(),
            'patients': len(patients),
            'altruists': self.pool.size() - len(patients),
            'matched': len(matched),
            'withdrawn': len(withdrawn),
            'withdrawn_high_cpra': withdrawn_high,
            'competing_exits': len(exits),
            'exits_high_cpra': exits_high,
            'arrivals': len(arrivals),
            'high_cpra_in_pool': high_cpra,
        }

        h = self.history
        h['pool_sizes'].append(self.pool.size())
        h['patients'].append(len(patients))
        h['arrivals'].append(len(arrivals))
        h['withdrawals'].append(len(withdrawn))
        h['competing_exits'].append(len(exits))
        h['matched'].append(len(matched))
        h['high_cpra_counts'].append(high_cpra)
        h['withdrawn_high_cpra'].append(withdrawn_high)
        h['exits_high_cpra'].append(exits_high)
        h['round_numbers'].append(self.current_round)
        return round_stats

    def get_pool_summary(self) -> Dict:
        """Current pool statistics (high-cPRA % among patients)."""
        patients = self.pool.patients()
        high_cpra = sum(p.is_highly_sensitised for p in patients)
        node_dist = {n: 0 for n in ANZKX_NODES}
        for p in self.pool.pairs:
            node_dist[p.hospital_node] += 1
        return {
            'pool_size': self.pool.size(),
            'patients': len(patients),
            'altruists': self.pool.size() - len(patients),
            'high_cpra_count': high_cpra,
            'high_cpra_pct': high_cpra / max(1, len(patients)) * 100,
            'mean_waiting_time': (float(np.mean([p.waiting_time for p in patients]))
                                  if patients else 0.0),
            'node_distribution': node_dist,
            'current_round': self.current_round
        }


def expected_steady_state(lambda_arrivals: float = LAMBDA_ARRIVALS,
                          tau: int = 12, mu_d: float = MU_D,
                          match_rate: float = 0.0) -> float:
    """
    Rough steady-state pool size without batching effects:
    each arrival stays until matched, exited or tau is reached,
    so E[pool] = lambda * sum_{k=0}^{tau-1} (1 - mu_d - m)^k,
    with m = per-round probability of being matched.
    Used to check that lambda keeps the pool in 60-90.
    """
    s = max(0.0, 1.0 - mu_d - match_rate)
    return float(lambda_arrivals * sum(s ** k for k in range(tau)))


# ============================================================
# TEST MODULE 2
# ============================================================

if __name__ == "__main__":

    from module1_generator import generate_initial_pool

    print("=" * 60)
    print("MODULE 2: DYNAMIC POOL EVOLUTION TEST")
    print("Pool Equation: V_{t+1} = V_t - M_t - W_t - X_t + A_t")
    print(f"lambda={LAMBDA_ARRIVALS}  mu_d={MU_D}  "
          f"(high-cPRA multiplier {MU_D_HIGH_CPRA_MULTIPLIER})")
    print("=" * 60)

    for tau_name, tau_val in TAU_VALUES.items():
        print(f"\n--- {tau_name} = {tau_val} months "
              f"(20 rounds, no matching) ---")
        pool, next_id = generate_initial_pool(n_pairs=75, seed=42,
                                              initial_wait_max=6)
        evo = PoolEvolution(pool, next_id, tau_months=tau_val, seed=42)
        print(f"{'Round':>5} {'Pool':>5} {'Pat':>5} {'Alt':>4} "
              f"{'W_t':>4} {'X_t':>4} {'A_t':>4} {'HighCPRA':>9}")
        for r in range(20):
            s = evo.evolve_round([])
            if s['round'] in (1, 2, 3, 5, 8, 10, 12, 15, 20):
                print(f"{s['round']:>5} {s['pool_size']:>5} "
                      f"{s['patients']:>5} {s['altruists']:>4} "
                      f"{s['withdrawn']:>4} {s['competing_exits']:>4} "
                      f"{s['arrivals']:>4} {s['high_cpra_in_pool']:>9}")
        summ = evo.get_pool_summary()
        print(f"  Final: {summ['patients']} patients, "
              f"{summ['high_cpra_pct']:.1f}% high-cPRA, "
              f"mean wait {summ['mean_waiting_time']:.1f} months")
        print(f"  Steady state without matching (formula): "
              f"{expected_steady_state(tau=tau_val):.0f}")

    print("\n--- Pool equation check ---")
    pool, next_id = generate_initial_pool(75, seed=7)
    evo = PoolEvolution(pool, next_id, tau_months=12, seed=7)
    before = {p.pair_id for p in evo.pool.pairs}
    matched = sorted(before)[:6]
    s = evo.evolve_round(matched)
    after = {p.pair_id for p in evo.pool.pairs}
    arrived = set(range(next_id, evo.next_pair_id))
    n_out = len(before) - len(after - arrived)
    print(f"  |V_t|={len(before)}  M={s['matched']} X={s['competing_exits']} "
          f"W={s['withdrawn']} A={s['arrivals']}  |V_t+1|={len(after)}")
    print(f"  |V_t| - M - X - W + A = |V_t+1| : "
          f"{len(before) - s['matched'] - s['competing_exits'] - s['withdrawn'] + s['arrivals'] == len(after)}")
    print(f"  Removed pairs = M + X + W : "
          f"{n_out == s['matched'] + s['competing_exits'] + s['withdrawn']}")

    print("\n--- Paired design: different matchings, same randomness ---")
    runs = []
    for matching in ("none", "first10"):
        pool, next_id = generate_initial_pool(75, seed=5)
        evo = PoolEvolution(pool, next_id, tau_months=12, seed=5)
        arrivals = []
        for r in range(10):
            ids = ([] if matching == "none"
                   else [p.pair_id for p in evo.pool.pairs[:10]])
            arrivals.append(evo.evolve_round(ids)['arrivals'])
        runs.append((arrivals, evo.next_pair_id))
    print(f"  Identical arrival counts:  {runs[0][0] == runs[1][0]}")
    print(f"  Identical arrival IDs:     {runs[0][1] == runs[1][1]}")

    print("\n--- Altruistic arrivals (expected ~8%) ---")
    pool, next_id = generate_initial_pool(75, seed=9)
    evo = PoolEvolution(pool, next_id, tau_months=12, seed=9)
    n_alt = n_arr = 0
    for r in range(200):
        start = evo.next_pair_id
        evo.evolve_round([])
        new = [p for p in evo.pool.pairs if p.pair_id >= start]
        n_alt += sum(p.is_altruistic for p in new)
        n_arr += evo.history['arrivals'][-1]
    print(f"  {n_alt}/{n_arr} arrivals altruistic "
          f"({n_alt / max(1, n_arr) * 100:.1f}%); "
          f"mean arrivals/round {n_arr / 200:.2f} (lambda={LAMBDA_ARRIVALS})")

    print("\n" + "=" * 60)
    print("MODULE 2 COMPLETE")
    print("Ready for Module 3: Logistics Network")
    print("=" * 60)
