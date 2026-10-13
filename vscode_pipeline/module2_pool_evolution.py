import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from module1_generator import (
    Patient, Pool,
    generate_arrivals,
    ANZKX_NODES
)

# ============================================================
# MODULE 2: DYNAMIC POOL EVOLUTION
# Implements pool equation:
# V_{t+1} = V_t \ M_t \ W_t \ X_t ∪ A_t
# Paper 2 — Dynamic KEP Implementation
# ============================================================

# --- Simulation Parameters ---

# Withdrawal threshold tau (sensitivity parameter)
# Tested across 6, 12, 18 months
# Modelling assumption — Ferrari 2012
# 1 round = approximately 1 month
TAU_VALUES = {
    'tau_6':  6,   # 6 months
    'tau_12': 12,  # 12 months
    'tau_18': 18   # 18 months
}

# Competing exit rate mu_d
# Estimated from DonateLife 2024 aggregate statistics
# Modelling assumption — not directly observed from ANZKX
# Probability a patient exits to deceased donor per round
MU_D = 0.02  # approximately 2% per round

# Poisson arrival rate lambda
# Calibrated from ANZKX programme statistics
# Pool maintains 60 to 90 pairs
# Approximate 5 to 8 new pairs per round
LAMBDA_ARRIVALS = 6.0


class PoolEvolution:
    

    def __init__(
        self,
        initial_pool: Pool,
        next_pair_id: int,
        tau_months: int = 12,
        mu_d: float = MU_D,
        lambda_arrivals: float = LAMBDA_ARRIVALS,
        seed: int = 42
    ):
        self.pool = initial_pool
        self.next_pair_id = next_pair_id
        self.tau = tau_months
        self.mu_d = mu_d
        self.lambda_arrivals = lambda_arrivals
        self.current_round = 0
        self.rng = np.random.RandomState(seed)

        # Tracking across rounds
        self.history = {
            'pool_sizes': [],
            'arrivals': [],
            'withdrawals': [],
            'competing_exits': [],
            'matched': [],
            'high_cpra_counts': [],
            'round_numbers': []
        }

    def increment_waiting_times(self):
        """
        Increment waiting time for all
        active pairs by 1 round.
        Called at start of each round.
        """
        for pair in self.pool.pairs:
            pair.waiting_time += 1

    def compute_withdrawals(self) -> List[Patient]:
        """
        Compute W_t: pairs withdrawn after
        exceeding sensitivity threshold tau.
       
        Pairs who have waited longer than tau months
        are removed from pool.
        Tau is a sensitivity parameter not
        an established ANZKX policy.
        Ferrari et al. 2012: mean 153 days.
        """
        withdrawn = []
        remaining = []

        for pair in self.pool.pairs:
            if pair.waiting_time >= self.tau:
                withdrawn.append(pair)
            else:
                remaining.append(pair)

        self.pool.pairs = remaining
        return withdrawn

    def compute_competing_exits(self) -> List[Patient]:
        """
        Compute X_t: pairs exiting to
        deceased donor pathway.
       
        Each pair has probability mu_d per round
        of receiving a deceased donor kidney.
        ANZKX is living donor programme only.
        Deceased donors are competing exit pathway.
        mu_d estimated from DonateLife 2024
        aggregate national statistics.
        Modelling assumption only.
        """
        exits = []
        remaining = []

        for pair in self.pool.pairs:
            # Altruistic donors cannot exit
            # to deceased donor pathway
            if pair.is_altruistic:
                remaining.append(pair)
                continue

            if self.rng.random() < self.mu_d:
                exits.append(pair)
            else:
                remaining.append(pair)

        self.pool.pairs = remaining
        return exits

    def compute_arrivals(self) -> List[Patient]:
        """
        Compute A_t: new incompatible pairs
        arriving this round.
       
        Arrival rate lambda follows Poisson process.
        Calibrated from ANZKX programme statistics.
        """
        n_arrivals = self.rng.poisson(self.lambda_arrivals)
        n_arrivals = max(1, n_arrivals)  # at least 1 arrival

        new_pairs, self.next_pair_id = generate_arrivals(
            n_arrivals=n_arrivals,
            next_pair_id=self.next_pair_id,
            arrival_round=self.current_round
        )

        self.pool.pairs.extend(new_pairs)
        return new_pairs

    def apply_matches(
        self,
        matched_pair_ids: List[int]
    ) -> List[Patient]:
        """
        Compute M_t: remove matched pairs from pool.
        Called after NSGA-III finds solution.
        matched_pair_ids: list of pair IDs to remove.
        """
        matched = []
        remaining = []

        matched_set = set(matched_pair_ids)

        for pair in self.pool.pairs:
            if pair.pair_id in matched_set:
                matched.append(pair)
            else:
                remaining.append(pair)

        self.pool.pairs = remaining
        return matched

    def evolve_round(
        self,
        matched_pair_ids: Optional[List[int]] = None
    ) -> Dict:
        """
        Execute one full round of pool evolution.
       
        Steps:
        1. Increment waiting times.
        2. Remove matched pairs M_t.
        3. Remove withdrawals W_t.
        4. Remove competing exits X_t.
        5. Add new arrivals A_t.
        6. Record statistics.
       
        Returns dict of round statistics.
        """
        self.current_round += 1
        self.pool.round_number = self.current_round

        # Step 1: Increment waiting times
        self.increment_waiting_times()

        # Step 2: Remove matched pairs M_t
        if matched_pair_ids is None:
            matched_pair_ids = []
        matched = self.apply_matches(matched_pair_ids)

        # Step 3: Remove withdrawals W_t
        withdrawn = self.compute_withdrawals()

        # Step 4: Remove competing exits X_t
        exits = self.compute_competing_exits()

        # Step 5: Add new arrivals A_t
        arrivals = self.compute_arrivals()

        # Step 6: Record statistics
        high_cpra = len([
            p for p in self.pool.pairs
            if p.is_highly_sensitised
        ])

        round_stats = {
            'round': self.current_round,
            'pool_size': self.pool.size(),
            'matched': len(matched),
            'withdrawn': len(withdrawn),
            'competing_exits': len(exits),
            'arrivals': len(arrivals),
            'high_cpra_in_pool': high_cpra
        }

        # Update history
        self.history['pool_sizes'].append(
            self.pool.size()
        )
        self.history['arrivals'].append(len(arrivals))
        self.history['withdrawals'].append(len(withdrawn))
        self.history['competing_exits'].append(len(exits))
        self.history['matched'].append(len(matched))
        self.history['high_cpra_counts'].append(high_cpra)
        self.history['round_numbers'].append(
            self.current_round
        )

        return round_stats

    def get_pool_summary(self) -> Dict:
        """Return current pool statistics."""
        high_cpra = len([
            p for p in self.pool.pairs
            if p.is_highly_sensitised
        ])
        node_dist = {n: 0 for n in ANZKX_NODES}
        for p in self.pool.pairs:
            node_dist[p.hospital_node] += 1

        return {
            'pool_size': self.pool.size(),
            'high_cpra_count': high_cpra,
            'high_cpra_pct': (
                high_cpra / max(1, self.pool.size()) * 100
            ),
            'node_distribution': node_dist,
            'current_round': self.current_round
        }


# ============================================================
# TEST MODULE 2
# ============================================================

if __name__ == "__main__":

    from module1_generator import generate_initial_pool

    print("=" * 55)
    print("MODULE 2: DYNAMIC POOL EVOLUTION TEST")
    print("Pool Equation: V_{t+1} = Vt - Mt - Wt - Xt + At")
    print("=" * 55)

    # Generate initial pool
    pool, next_id = generate_initial_pool(
        n_pairs=75,
        seed=42
    )
    print(f"\nInitial pool size: {pool.size()}")

    # Test all three tau values
    for tau_name, tau_val in TAU_VALUES.items():

        print(f"\n{'='*55}")
        print(f"Sensitivity Parameter: {tau_name} "
              f"= {tau_val} months")
        print(f"{'='*55}")

        # Re-generate fresh pool for each tau
        pool, next_id = generate_initial_pool(
            n_pairs=75,
            seed=42
        )

        # Create pool evolution manager
        evolution = PoolEvolution(
            initial_pool=pool,
            next_pair_id=next_id,
            tau_months=tau_val,
            mu_d=MU_D,
            lambda_arrivals=LAMBDA_ARRIVALS,
            seed=42
        )

        print(f"\n{'Round':<8} {'Pool':>6} "
              f"{'Matched':>8} {'W_t':>8} "
              f"{'X_t':>8} {'A_t':>8} "
              f"{'HighCPRA':>10}")
        print("-" * 55)

        # Run 10 rounds without matching
        # (matching will come in Module 5)
        for r in range(10):
            stats = evolution.evolve_round(
                matched_pair_ids=[]
            )
            print(
                f"  {stats['round']:<6} "
                f"{stats['pool_size']:>6} "
                f"{stats['matched']:>8} "
                f"{stats['withdrawn']:>8} "
                f"{stats['competing_exits']:>8} "
                f"{stats['arrivals']:>8} "
                f"{stats['high_cpra_in_pool']:>10}"
            )

        # Final summary
        summary = evolution.get_pool_summary()
        print(f"\nFinal pool size: {summary['pool_size']}")
        print(f"High cPRA patients: "
              f"{summary['high_cpra_count']} "
              f"({summary['high_cpra_pct']:.1f}%)")
        print(f"\nNode distribution:")
        for node, count in \
                summary['node_distribution'].items():
            print(f"  {node}: {count}")

    print("\n" + "=" * 55)
    print("MODULE 2 COMPLETE")
    print("Ready for Module 3: Logistics Network")
    print("=" * 55)