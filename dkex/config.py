"""Central configuration (concept note, Table 2 and Section 4.2).

Every number that the concept note labels a *modelling assumption* lives here
so that it can be changed from a YAML/JSON file without touching code.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

# The six ANZKX transplant centres (Section 3.3 / 4.3).
CENTRES = ("Perth", "Adelaide", "Melbourne", "Sydney", "Brisbane", "Auckland")
# F4 grouping (Eq. 9): distant nodes R vs. central nodes U.
DISTANT_CENTRES = ("Perth", "Auckland")


@dataclass
class PoolConfig:
    """Synthetic ANZKX-calibrated pool (Table 2)."""

    # Blood type distribution O, A, B, AB  [ANZDATA, ref 23]
    abo_dist: dict[str, float] = field(
        default_factory=lambda: {"O": 0.49, "A": 0.38, "B": 0.10, "AB": 0.03}
    )
    high_cpra_rate: float = 0.30          # share of patients with cPRA >= 80 %
    high_cpra_threshold: float = 0.80
    altruist_rate: float = 0.08           # share of arrivals that are altruistic donors
    initial_pool_size: int = 75           # active pool 60-90 pairs [ref 5]
    # Pair arrivals per clearing round, A_t ~ Poisson(lambda).  With one round
    # per month, 8/month (~96/yr) keeps the pool in the 60-90 band given
    # ~50-66 transplants/yr plus withdrawals/competing exits.  Calibration
    # assumption — tune against ANZKX program statistics.
    arrival_rate: float = 8.0
    # HLA mismatch m_ij ~ Binomial(6, p) — population-frequency proxy (Eq. 3).
    hla_mismatch_p: float = 0.55
    # Relative pair volume at each centre (used to assign patients to nodes).
    centre_weights: dict[str, float] = field(
        default_factory=lambda: {
            "Perth": 0.12, "Adelaide": 0.10, "Melbourne": 0.25,
            "Sydney": 0.28, "Brisbane": 0.15, "Auckland": 0.10,
        }
    )


@dataclass
class DynamicsConfig:
    """Pool evolution V_{t+1} = V_t \\ M_t \\ W_t \\ X_t U A_t (Eq. 1)."""

    rounds: int = 25                      # T in 20-30
    round_length_months: float = 1.0
    withdrawal_threshold_months: float = 12.0   # tau in {6, 12, 18} (sensitivity)
    # Competing exit to the deceased-donor pathway, per pair per round.
    # Modelling assumption from aggregate DonateLife 2024 statistics [ref 6].
    deceased_exit_rate: float = 0.03
    # Highly sensitised patients receive deceased-donor offers far less often.
    deceased_exit_high_cpra_multiplier: float = 0.3
    # Altruistic donors are also withdrawn after tau (they donate to the waitlist).
    altruists_withdraw: bool = True


@dataclass
class LogisticsConfig:
    """Six-node logistics network and CIT feasibility (Eq. 4 / 14)."""

    cit_max_hours: float = 24.0
    prep_hours: float = 6.0
    # Door-to-door transport time (hours) between centres.  Approximate
    # scheduled-flight time + ground transfer + handover.  Placeholder values —
    # replace with published inter-hospital air transport times (Section 4.3).
    transport_hours: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "Perth":     {"Perth": 1.0, "Adelaide": 5.5, "Melbourne": 6.5, "Sydney": 7.0, "Brisbane": 7.5, "Auckland": 13.0},
            "Adelaide":  {"Adelaide": 1.0, "Melbourne": 3.5, "Sydney": 4.0, "Brisbane": 5.0, "Auckland": 9.5},
            "Melbourne": {"Melbourne": 1.0, "Sydney": 3.5, "Brisbane": 4.0, "Auckland": 8.0},
            "Sydney":    {"Sydney": 1.0, "Brisbane": 3.5, "Auckland": 7.5},
            "Brisbane":  {"Brisbane": 1.0, "Auckland": 7.5},
            "Auckland":  {"Auckland": 1.0},
        }
    )
    # Extra stochastic-free delay added to every inter-centre leg (e.g. waiting
    # for the next scheduled flight).  Raise this to make CIT binding.
    inter_centre_delay_hours: float = 0.0


@dataclass
class StructureConfig:
    max_cycle_length: int = 3             # K
    max_chain_length: int = 4             # L (recipients per chain)
    # Chain enumeration explodes combinatorially; when the number of chains
    # exceeds this cap a uniform random subset is kept.  The SAME candidate set
    # is given to the ILP and the MOEA so PoF_t compares like with like (3.6).
    max_chains: int = 20000
    max_cycles: int = 200000


@dataclass
class MOEAConfig:
    """NSGA-III setup (Section 4.2.1)."""

    algorithm: str = "nsga3"              # "nsga3" | "nsga2"
    pop_size: int = 100
    generations: int = 150
    ref_divisions: int = 12               # p for four objectives
    crossover_prob: float = 0.9
    mutation_rates: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25)
    fixed_mutation_rate: float = 0.10     # used when AOS is disabled
    # Carry still-feasible non-dominated solutions into the next round's
    # initial population (dynamic variants only).
    warm_start: bool = True
    # Rule for choosing the implemented solution from the final Pareto front:
    # "compromise" (closest to the front's normalised ideal point),
    # "max_f1", or "min_f3".
    selection_rule: str = "compromise"
    # F3 sign: "min_gap" minimises Eq. 8's absolute gap; "max_access" follows
    # the -F3 sign of Eq. 5 and maximises the high-cPRA match rate.
    f3_mode: str = "min_gap"
    hv_samples: int = 4000                # Monte-Carlo hypervolume samples
    fill_attempts: int = 30               # repair: candidates tried per free vertex


@dataclass
class AOSConfig:
    """Adaptive operator selection (Section 3.5 / 4.2.2-4.2.3)."""

    method: str = "ucb1"                  # "ucb1" | "linucb" | "fixed"
    ucb_window: int = 15                  # sliding window W
    linucb_alpha: float = 1.0
    persist_across_rounds: bool = True    # False == "static" (reset each round)


@dataclass
class ClearingConfig:
    policy: str = "myopic"                # "myopic" | "batched"
    batch_threshold: int = 90             # n*
    # Optional safety valve for pi_B: also clear if this many rounds passed
    # since the last clearing.  None == pure threshold rule (concept note).
    # Without it pi_B can stall forever when the steady-state pool
    # (~ lambda * tau) never reaches n*, e.g. tau = 6 months.
    batch_max_interval: int | None = None


@dataclass
class ExperimentConfig:
    name: str = "default"
    seed: int = 12345
    runs: int = 100
    taus: tuple[float, ...] = (6.0, 12.0, 18.0)
    baselines: tuple[str, ...] = (
        "static_nsga2_ucb1",
        "static_nsga3_fixed",
        "dyn_nsga3_ucb1_myopic",
        "dyn_nsga3_linucb_myopic",
        "dyn_nsga3_ucb1_batched",
        "dyn_nsga3_linucb_batched",
        "myopic_ilp",
        "dickerson_lex",
    )
    n_jobs: int = 1
    output_dir: str = "results"


@dataclass
class Config:
    pool: PoolConfig = field(default_factory=PoolConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    logistics: LogisticsConfig = field(default_factory=LogisticsConfig)
    structures: StructureConfig = field(default_factory=StructureConfig)
    moea: MOEAConfig = field(default_factory=MOEAConfig)
    aos: AOSConfig = field(default_factory=AOSConfig)
    clearing: ClearingConfig = field(default_factory=ClearingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def copy(self) -> "Config":
        return copy.deepcopy(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        cfg = cls()
        _merge(cfg, data or {})
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            import yaml  # optional dependency

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)


def _merge(obj: Any, data: dict[str, Any]) -> None:
    """Recursively overwrite dataclass fields of *obj* with values in *data*."""
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise KeyError(f"Unknown config key '{key}' in {type(obj).__name__}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, value)
