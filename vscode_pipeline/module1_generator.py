import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Tuple

# ============================================================
# MODULE 1: SAIDMAN PATIENT GENERATOR
# ANZKX-Calibrated Synthetic Pool Generator
# Paper 2 — Dynamic KEP Implementation
# ============================================================

# --- Blood Type Frequencies (ANZDATA 2026) ---
BLOOD_TYPE_PROBS = {
    'O': 0.49,
    'A': 0.38,
    'B': 0.10,
    'AB': 0.03
}

# ABO Compatibility Matrix
# Key: donor blood type
# Value: list of compatible recipient blood types
ABO_COMPATIBILITY = {
    'O':  ['O', 'A', 'B', 'AB'],  # O is universal donor
    'A':  ['A', 'AB'],
    'B':  ['B', 'AB'],
    'AB': ['AB']                   # AB only donates to AB
}

# HLA Loci considered (HLA-A, HLA-B, HLA-DR = 6 alleles)
HLA_LOCI = 6

@dataclass
class Patient:
    """Represents a single donor-recipient incompatible pair."""
    pair_id: int
    blood_type_patient: str      # Recipient blood type
    blood_type_donor: str        # Donor blood type
    cpra: float                  # Panel Reactive Antibody (0 to 1)
    is_highly_sensitised: bool   # cPRA >= 0.80
    hla_patient: np.ndarray      # HLA profile recipient (6 loci)
    hla_donor: np.ndarray        # HLA profile donor (6 loci)
    is_altruistic: bool          # Altruistic donor flag
    hospital_node: str           # ANZKX centre
    waiting_time: int            # Rounds waiting (starts at 0)
    arrival_round: int           # Which round they arrived

@dataclass
class Pool:
    """Represents the active KEP pool at a given round."""
    pairs: List[Patient] = field(default_factory=list)
    round_number: int = 0

    def size(self):
        return len(self.pairs)

    def get_highly_sensitised(self):
        return [p for p in self.pairs if p.is_highly_sensitised]


# ANZKX Hospital Nodes
ANZKX_NODES = [
    'Perth',
    'Adelaide',
    'Melbourne',
    'Sydney',
    'Brisbane',
    'Auckland'
]

# Node contribution weights (approximated from DonateLife 2024)
# State level distribution
NODE_WEIGHTS = {
    'Perth':     0.12,
    'Adelaide':  0.10,
    'Melbourne': 0.28,
    'Sydney':    0.28,
    'Brisbane':  0.14,
    'Auckland':  0.08
}


def generate_blood_type() -> str:
    """Generate blood type using ANZDATA 2026 frequencies."""
    types = list(BLOOD_TYPE_PROBS.keys())
    probs = list(BLOOD_TYPE_PROBS.values())
    return np.random.choice(types, p=probs)


def generate_cpra() -> float:
    """
    Generate cPRA value.
    30% of patients have cPRA >= 0.80 (highly sensitised).
    Remaining 70% have cPRA < 0.80.
    Source: ANZDATA 2026.
    """
    if np.random.random() < 0.30:
        # Highly sensitised: cPRA between 0.80 and 1.0
        return np.random.uniform(0.80, 1.0)
    else:
        # Not highly sensitised: cPRA between 0.0 and 0.80
        return np.random.uniform(0.0, 0.80)


def generate_hla_profile() -> np.ndarray:
    """
    Generate simplified HLA profile across 6 loci.
    Each locus has allele value 0 or 1 (simplified binary proxy).
    This is a population-level proxy only.
    Not representative of actual HLA diversity.
    Source: Population frequency approximation.
    """
    return np.random.randint(0, 2, size=HLA_LOCI)


def compute_hla_mismatch(
    hla_donor: np.ndarray,
    hla_patient: np.ndarray
) -> int:
    """
    Compute HLA mismatch count between donor and patient.
    m_ij = number of mismatched loci out of 6.
    """
    return int(np.sum(hla_donor != hla_patient))


def compute_omega_hla(
    hla_donor: np.ndarray,
    hla_patient: np.ndarray
) -> float:
    """
    Compute HLA compatibility weight omega_HLA.
    omega_HLA(i,j) = 1 - m_ij / 6
    Range: 0 (worst) to 1 (perfect match).
    Proxy measure only — not actual clinical crossmatch.
    """
    m_ij = compute_hla_mismatch(hla_donor, hla_patient)
    return 1.0 - (m_ij / HLA_LOCI)


def is_abo_compatible(
    donor_blood: str,
    recipient_blood: str
) -> bool:
    """
    Check ABO blood type compatibility.
    Returns True if donor can donate to recipient.
    """
    return recipient_blood in ABO_COMPATIBILITY[donor_blood]


def generate_hospital_node() -> str:
    """
    Assign patient pair to an ANZKX hospital node.
    Weighted by approximate state contribution.
    """
    nodes = list(NODE_WEIGHTS.keys())
    weights = list(NODE_WEIGHTS.values())
    return np.random.choice(nodes, p=weights)


def generate_patient_pair(
    pair_id: int,
    arrival_round: int,
    is_altruistic: bool = False
) -> Patient:
    """
    Generate a single incompatible donor-recipient pair.
    Patient and donor are ABO incompatible by definition
    (they would not need KEP otherwise).
    """
    # Generate blood types ensuring incompatibility
    # Keep generating until we get an incompatible pair
    max_attempts = 100
    for _ in range(max_attempts):
        blood_donor = generate_blood_type()
        blood_patient = generate_blood_type()
        # Pair joins KEP because they are incompatible
        if not is_abo_compatible(blood_donor, blood_patient):
            break
   
    cpra = generate_cpra()
    hla_patient = generate_hla_profile()
    hla_donor = generate_hla_profile()
    node = generate_hospital_node()

    return Patient(
        pair_id=pair_id,
        blood_type_patient=blood_patient,
        blood_type_donor=blood_donor,
        cpra=cpra,
        is_highly_sensitised=(cpra >= 0.80),
        hla_patient=hla_patient,
        hla_donor=hla_donor,
        is_altruistic=is_altruistic,
        hospital_node=node,
        waiting_time=0,
        arrival_round=arrival_round
    )


def generate_initial_pool(
    n_pairs: int = 75,
    seed: int = 42
) -> Tuple[Pool, int]:
    """
    Generate initial ANZKX-calibrated synthetic pool.
    Default pool size 75 (midpoint of 60 to 90 range).
    8% altruistic donors included.
    Source: ANZKX Programme Page.
    """
    np.random.seed(seed)
    random.seed(seed)

    pool = Pool(round_number=0)
    n_altruistic = max(1, int(n_pairs * 0.08))
    n_incompatible = n_pairs - n_altruistic

    pair_id = 0

    # Generate incompatible pairs
    for _ in range(n_incompatible):
        pair = generate_patient_pair(
            pair_id=pair_id,
            arrival_round=0,
            is_altruistic=False
        )
        pool.pairs.append(pair)
        pair_id += 1

    # Generate altruistic donors
    for _ in range(n_altruistic):
        pair = generate_patient_pair(
            pair_id=pair_id,
            arrival_round=0,
            is_altruistic=True
        )
        pool.pairs.append(pair)
        pair_id += 1

    return pool, pair_id  # Return pool and next available ID


def generate_arrivals(
    n_arrivals: int,
    next_pair_id: int,
    arrival_round: int
) -> Tuple[List[Patient], int]:
    """
    Generate new arriving pairs for a given round.
    Used in pool evolution equation A_t.
    """
    new_pairs = []
    n_altruistic = max(0, int(n_arrivals * 0.08))
    n_incompatible = n_arrivals - n_altruistic

    for _ in range(n_incompatible):
        pair = generate_patient_pair(
            pair_id=next_pair_id,
            arrival_round=arrival_round,
            is_altruistic=False
        )
        new_pairs.append(pair)
        next_pair_id += 1

    for _ in range(n_altruistic):
        pair = generate_patient_pair(
            pair_id=next_pair_id,
            arrival_round=arrival_round,
            is_altruistic=True
        )
        new_pairs.append(pair)
        next_pair_id += 1

    return new_pairs, next_pair_id


# ============================================================
# TEST MODULE 1
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("MODULE 1: SAIDMAN GENERATOR TEST")
    print("ANZKX-Calibrated Synthetic Pool")
    print("=" * 50)

    # Generate initial pool
    pool, next_id = generate_initial_pool(n_pairs=75, seed=42)

    print(f"\nInitial Pool Size: {pool.size()}")

    # Blood type distribution check
    blood_counts = {'O': 0, 'A': 0, 'B': 0, 'AB': 0}
    for p in pool.pairs:
        blood_counts[p.blood_type_patient] += 1

    print("\n--- Patient Blood Type Distribution ---")
    for bt, count in blood_counts.items():
        pct = count / pool.size() * 100
        print(f"  {bt}: {count} ({pct:.1f}%)")
    print("  Expected: O=49% A=38% B=10% AB=3%")

    # cPRA distribution check
    highly_sensitised = pool.get_highly_sensitised()
    pra_pct = len(highly_sensitised) / pool.size() * 100
    print(f"\n--- cPRA Distribution ---")
    print(f"  Highly sensitised (cPRA>=80%): "
          f"{len(highly_sensitised)} ({pra_pct:.1f}%)")
    print(f"  Expected: ~30%")

    # Altruistic donors
    altruistic = [p for p in pool.pairs if p.is_altruistic]
    alt_pct = len(altruistic) / pool.size() * 100
    print(f"\n--- Altruistic Donors ---")
    print(f"  Count: {len(altruistic)} ({alt_pct:.1f}%)")
    print(f"  Expected: ~8%")

    # Hospital node distribution
    print(f"\n--- Hospital Node Distribution ---")
    node_counts = {n: 0 for n in ANZKX_NODES}
    for p in pool.pairs:
        node_counts[p.hospital_node] += 1
    for node, count in node_counts.items():
        print(f"  {node}: {count}")

    # HLA mismatch example
    p1 = pool.pairs[0]
    p2 = pool.pairs[1]
    omega = compute_omega_hla(p1.hla_donor, p2.hla_patient)
    mismatch = compute_hla_mismatch(p1.hla_donor, p2.hla_patient)
    print(f"\n--- HLA Mismatch Example ---")
    print(f"  Pair 0 donor vs Pair 1 patient:")
    print(f"  Mismatch count: {mismatch}/6")
    print(f"  omega_HLA: {omega:.3f}")

    # ABO compatibility example
    print(f"\n--- ABO Compatibility Example ---")
    print(f"  Pair 0: donor={p1.blood_type_donor} "
          f"patient={p1.blood_type_patient}")
    print(f"  Pair 1: donor={p2.blood_type_donor} "
          f"patient={p2.blood_type_patient}")
    compat = is_abo_compatible(
        p1.blood_type_donor,
        p2.blood_type_patient
    )
    print(f"  Pair 0 donor -> Pair 1 patient compatible: {compat}")

    # Test arrivals
    print(f"\n--- Testing Arrivals ---")
    new_arrivals, next_id = generate_arrivals(
        n_arrivals=5,
        next_pair_id=next_id,
        arrival_round=1
    )
    print(f"  Generated {len(new_arrivals)} new arrivals")
    print(f"  Next pair ID: {next_id}")

    print("\n" + "=" * 50)
    print("MODULE 1 COMPLETE")
    print("Ready for Module 2: Pool Evolution")
    print("=" * 50)
