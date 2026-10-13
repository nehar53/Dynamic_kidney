import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ============================================================
# MODULE 1: SAIDMAN PATIENT GENERATOR
# ANZKX-Calibrated Synthetic Pool Generator
# Paper 2 — Dynamic KEP Implementation
# Concept note Section 4.1, Table 2
# ============================================================
#
# Saidman et al. (2006) probabilistic model, calibrated to ANZKX:
#   - patient and donor blood types from ANZDATA frequencies
#   - 30% of patients highly sensitised (cPRA >= 80%)
#   - a pair joins the KEP only if it is INCOMPATIBLE:
#       ABO-incompatible  OR  positive crossmatch with own donor
#       (positive crossmatch happens with probability = cPRA)
#   - 8% of arrivals are altruistic (non-directed) donors.
#     They have a donor but NO patient, can only give a kidney,
#     and never count as patients in F2-F4.
#
# Virtual crossmatch (used by module 3 for every edge i -> j):
#   positive with probability cPRA_j, drawn ONCE per
#   (donor, patient) couple, so the same edge gives the same
#   answer in every round and every configuration.
#
# HLA limitation (concept note Section 3.2): HLA alleles are a
# population-frequency proxy, NOT clinical HLA typing or a real
# virtual crossmatch.
#
# Randomness: every function takes an optional `rng`
# (np.random.Generator or RandomState). If omitted, the global
# numpy RNG is used, so module 7's per-round seeding still works.

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

# --- Sensitisation (ANZDATA 2026) ---
HIGH_CPRA_RATE = 0.30          # share of patients with cPRA >= 0.80
CPRA_THRESHOLD = 0.80
CPRA_HIGH_MAX = 0.99           # cPRA = 1.00 could never be matched

# --- Altruistic donors (ANZKX programme) ---
ALTRUISTIC_RATE = 0.08

# --- HLA proxy: HLA-A, HLA-B, HLA-DR, two alleles each ---
HLA_LOCI = 6
N_ALLELES_PER_LOCUS = 8
# Skewed (Zipf-like) allele frequencies: a few common alleles,
# many rare ones. Expected mismatches between unrelated people
# is then about 4-5 of 6, close to unrelated-donor reality.
_ALLELE_P = 1.0 / np.arange(1, N_ALLELES_PER_LOCUS + 1)
ALLELE_FREQS = _ALLELE_P / _ALLELE_P.sum()

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
NODE_WEIGHTS = {
    'Perth':     0.12,
    'Adelaide':  0.10,
    'Melbourne': 0.28,
    'Sydney':    0.28,
    'Brisbane':  0.14,
    'Auckland':  0.08
}


@dataclass
class Patient:
    """
    A donor-recipient incompatible pair, or an altruistic donor.

    For an altruistic donor (is_altruistic=True):
      blood_type_patient = None, hla_patient = None, cpra = 0.0,
      is_highly_sensitised = False  (there is no patient).
    """
    pair_id: int
    blood_type_patient: Optional[str]   # Recipient blood type
    blood_type_donor: str               # Donor blood type
    cpra: float                         # Panel Reactive Antibody (0 to 1)
    is_highly_sensitised: bool          # cPRA >= 0.80
    hla_patient: Optional[np.ndarray]   # HLA profile recipient (6 loci)
    hla_donor: np.ndarray               # HLA profile donor (6 loci)
    is_altruistic: bool                 # Altruistic donor flag
    hospital_node: str                  # ANZKX centre
    waiting_time: int                   # Rounds waiting (starts at 0)
    arrival_round: int                  # Which round they arrived
    xm_salt: int = 0                    # random key for virtual crossmatch
    incompatibility: str = ""           # 'ABO', 'XM' (crossmatch) or ''


@dataclass
class Pool:
    """Represents the active KEP pool at a given round."""
    pairs: List[Patient] = field(default_factory=list)
    round_number: int = 0

    def size(self):
        return len(self.pairs)

    def patients(self):
        """Pairs with a patient (altruistic donors excluded)."""
        return [p for p in self.pairs if not p.is_altruistic]

    def altruists(self):
        return [p for p in self.pairs if p.is_altruistic]

    def get_highly_sensitised(self):
        return [p for p in self.pairs
                if p.is_highly_sensitised and not p.is_altruistic]


# ============================================================
# RANDOM HELPERS
# ============================================================

def _rng(rng=None):
    """Use the given RNG, or the global numpy RNG."""
    return np.random if rng is None else rng


def _uniform(rng, low=0.0, high=1.0) -> float:
    return float(_rng(rng).uniform(low, high))


def _choice(options, probs, rng=None):
    idx = _rng(rng).choice(len(options), p=probs)
    return options[int(idx)]


# ============================================================
# ATTRIBUTE GENERATORS
# ============================================================

def generate_blood_type(rng=None) -> str:
    """Generate blood type using ANZDATA 2026 frequencies."""
    return _choice(list(BLOOD_TYPE_PROBS.keys()),
                   list(BLOOD_TYPE_PROBS.values()), rng)


def generate_cpra(high: Optional[bool] = None, rng=None) -> float:
    """
    Generate cPRA value (0 to 1).

    high=True  -> highly sensitised, cPRA in [0.80, 0.99]
    high=False -> cPRA < 0.80: half unsensitised (0-0.05),
                  half moderately sensitised (0.05-0.80)
    high=None  -> high with probability 30% (ANZDATA 2026)
    """
    if high is None:
        high = _uniform(rng) < HIGH_CPRA_RATE
    if high:
        return _uniform(rng, CPRA_THRESHOLD, CPRA_HIGH_MAX)
    if _uniform(rng) < 0.5:
        return _uniform(rng, 0.0, 0.05)
    return _uniform(rng, 0.05, CPRA_THRESHOLD - 1e-6)


def generate_hla_profile(rng=None) -> np.ndarray:
    """
    HLA profile across 6 loci (HLA-A, B, DR x 2 alleles).
    Each locus holds an allele index drawn from skewed
    population frequencies. Population-level proxy only.
    """
    r = _rng(rng)
    return np.array([r.choice(N_ALLELES_PER_LOCUS, p=ALLELE_FREQS)
                     for _ in range(HLA_LOCI)])


def generate_hospital_node(rng=None) -> str:
    """Assign a pair to an ANZKX centre, weighted by state share."""
    return _choice(list(NODE_WEIGHTS.keys()),
                   list(NODE_WEIGHTS.values()), rng)


# ============================================================
# COMPATIBILITY FUNCTIONS (used by module 3)
# ============================================================

def is_abo_compatible(donor_blood: str,
                      recipient_blood: Optional[str]) -> bool:
    """True if a donor of this blood type can give to the recipient."""
    if recipient_blood is None:          # altruistic donor: no patient
        return False
    return recipient_blood in ABO_COMPATIBILITY[donor_blood]


def compute_hla_mismatch(hla_donor: np.ndarray,
                         hla_patient: np.ndarray) -> int:
    """m_ij = number of mismatched loci out of 6."""
    return int(np.sum(hla_donor != hla_patient))


def compute_omega_hla(hla_donor: np.ndarray,
                      hla_patient: np.ndarray) -> float:
    """
    omega_HLA(i,j) = 1 - m_ij / 6   (concept note Eq. 3)
    Range 0 (worst) to 1 (perfect). Proxy measure only.
    """
    return 1.0 - compute_hla_mismatch(hla_donor, hla_patient) / HLA_LOCI


def is_crossmatch_negative(donor_pair: Patient,
                           patient_pair: Patient) -> bool:
    """
    Virtual crossmatch for donor of `donor_pair` -> patient of
    `patient_pair`. Positive (incompatible) with probability
    cPRA of the patient. The draw is fixed for each
    (donor, patient) couple, so it never changes between rounds.
    """
    if patient_pair.is_altruistic:
        return False
    u = np.random.default_rng(
        [donor_pair.xm_salt, patient_pair.xm_salt, 7]).random()
    return u >= patient_pair.cpra


def is_compatible(donor_pair: Patient, patient_pair: Patient) -> bool:
    """Medical compatibility: ABO AND negative virtual crossmatch."""
    return (is_abo_compatible(donor_pair.blood_type_donor,
                              patient_pair.blood_type_patient)
            and is_crossmatch_negative(donor_pair, patient_pair))


# ============================================================
# PAIR / ALTRUIST GENERATION
# ============================================================

def generate_patient_pair(pair_id: int, arrival_round: int,
                          is_altruistic: bool = False,
                          rng=None) -> Patient:
    """
    Generate one incompatible donor-recipient pair (Saidman).

    The sensitisation class is drawn first (30% high), then blood
    types and the own-donor crossmatch are redrawn until the pair
    is incompatible. This keeps the 30% high-cPRA share in the
    POOL, as in Table 2.
    If is_altruistic=True an altruistic donor is returned instead.
    """
    if is_altruistic:
        return generate_altruistic_donor(pair_id, arrival_round, rng)

    r = _rng(rng)
    cpra = generate_cpra(rng=rng)
    while True:
        blood_donor = generate_blood_type(rng)
        blood_patient = generate_blood_type(rng)
        if not is_abo_compatible(blood_donor, blood_patient):
            reason = "ABO"
            break
        if _uniform(rng) < cpra:          # positive own-donor crossmatch
            reason = "XM"
            break
        # compatible pair: transplanted directly, never joins KEP

    return Patient(
        pair_id=pair_id,
        blood_type_patient=blood_patient,
        blood_type_donor=blood_donor,
        cpra=cpra,
        is_highly_sensitised=(cpra >= CPRA_THRESHOLD),
        hla_patient=generate_hla_profile(rng),
        hla_donor=generate_hla_profile(rng),
        is_altruistic=False,
        hospital_node=generate_hospital_node(rng),
        waiting_time=0,
        arrival_round=arrival_round,
        xm_salt=int(r.randint(0, 2 ** 31 - 1)),
        incompatibility=reason,
    )


def generate_altruistic_donor(pair_id: int, arrival_round: int,
                              rng=None) -> Patient:
    """An altruistic (non-directed) donor: donor only, no patient."""
    r = _rng(rng)
    return Patient(
        pair_id=pair_id,
        blood_type_patient=None,
        blood_type_donor=generate_blood_type(rng),
        cpra=0.0,
        is_highly_sensitised=False,
        hla_patient=None,
        hla_donor=generate_hla_profile(rng),
        is_altruistic=True,
        hospital_node=generate_hospital_node(rng),
        waiting_time=0,
        arrival_round=arrival_round,
        xm_salt=int(r.randint(0, 2 ** 31 - 1)),
        incompatibility="",
    )


def generate_arrival(pair_id: int, arrival_round: int,
                     rng=None) -> Patient:
    """One arrival: altruistic donor with probability 8%, else a pair."""
    if _uniform(rng) < ALTRUISTIC_RATE:
        return generate_altruistic_donor(pair_id, arrival_round, rng)
    return generate_patient_pair(pair_id, arrival_round, rng=rng)


def generate_initial_pool(n_pairs: int = 75, seed: int = 42,
                          initial_wait_max: int = 0,
                          rng=None) -> Tuple[Pool, int]:
    """
    Generate the initial ANZKX-calibrated pool.

    n_pairs          pool size (75 = midpoint of 60-90)
    seed             seeds the global numpy/random RNGs (as before)
    initial_wait_max if > 0, waiting times are drawn uniformly from
                     0..initial_wait_max-1 so the initial pool does
                     not all reach the withdrawal threshold tau in
                     the same round. 0 = everyone starts at 0.

    Each member is altruistic with probability 8%.
    Returns (pool, next available pair ID).
    """
    np.random.seed(seed)
    random.seed(seed)
    pool = Pool(round_number=0)
    for pair_id in range(n_pairs):
        p = generate_arrival(pair_id, arrival_round=0, rng=rng)
        if initial_wait_max > 0:
            p.waiting_time = int(_rng(rng).randint(0, initial_wait_max))
        pool.pairs.append(p)
    return pool, n_pairs


def generate_arrivals(n_arrivals: int, next_pair_id: int,
                      arrival_round: int,
                      rng=None) -> Tuple[List[Patient], int]:
    """
    New arrivals A_t for one round (count comes from module 2's
    Poisson draw). Each arrival is altruistic with probability 8%,
    so altruists keep arriving even when n_arrivals is small.
    """
    new_pairs = []
    for _ in range(n_arrivals):
        new_pairs.append(generate_arrival(next_pair_id, arrival_round, rng))
        next_pair_id += 1
    return new_pairs, next_pair_id


# ============================================================
# TEST MODULE 1
# ============================================================

if __name__ == "__main__":

    print("=" * 55)
    print("MODULE 1: SAIDMAN GENERATOR TEST")
    print("ANZKX-Calibrated Synthetic Pool (concept note 4.1)")
    print("=" * 55)

    pool, next_id = generate_initial_pool(n_pairs=75, seed=42)
    pts = pool.patients()
    print(f"\nInitial pool: {pool.size()} "
          f"({len(pts)} pairs + {len(pool.altruists())} altruists)")

    # Large sample for calibration checks
    big, _ = generate_initial_pool(n_pairs=5000, seed=1)
    bp = big.patients()

    print("\n--- Altruistic Donors (expected ~8%) ---")
    print(f"  {len(big.altruists()) / big.size() * 100:.1f}% "
          f"of 5000 generated")

    print("\n--- cPRA (expected 30% highly sensitised) ---")
    print(f"  Highly sensitised: "
          f"{len(big.get_highly_sensitised()) / len(bp) * 100:.1f}%")

    print("\n--- Why pairs are in the KEP ---")
    abo = sum(p.incompatibility == "ABO" for p in bp)
    print(f"  ABO-incompatible:      {abo / len(bp) * 100:.1f}%")
    print(f"  Positive crossmatch:   {(len(bp) - abo) / len(bp) * 100:.1f}%")

    print("\n--- Blood Types in KEP pool (patients / donors) ---")
    print("  Population: O=49% A=38% B=10% AB=3%")
    print("  (KEP pools are skewed: O patients are hard to match)")
    for bt in BLOOD_TYPE_PROBS:
        pat = sum(p.blood_type_patient == bt for p in bp) / len(bp) * 100
        don = sum(p.blood_type_donor == bt for p in bp) / len(bp) * 100
        print(f"  {bt:>2}: patients {pat:5.1f}%   donors {don:5.1f}%")

    print("\n--- Hospital Nodes (initial pool) ---")
    for node in ANZKX_NODES:
        n = sum(p.hospital_node == node for p in pool.pairs)
        print(f"  {node:<10} {n}")

    print("\n--- HLA Mismatch (expected mean ~4-5 of 6) ---")
    m = [compute_hla_mismatch(bp[i].hla_donor, bp[i + 1].hla_patient)
         for i in range(0, 2000, 2)]
    print(f"  Mean mismatches: {np.mean(m):.2f} / 6   "
          f"mean omega_HLA: {1 - np.mean(m) / 6:.3f}")

    print("\n--- Compatibility between pairs (ABO + crossmatch) ---")
    sample = bp[:300]
    abo_ok = sum(is_abo_compatible(a.blood_type_donor, b.blood_type_patient)
                 for a in sample for b in sample if a is not b)
    full_ok = sum(is_compatible(a, b)
                  for a in sample for b in sample if a is not b)
    tot = len(sample) * (len(sample) - 1)
    print(f"  ABO compatible:            {abo_ok / tot * 100:.1f}%")
    print(f"  ABO + negative crossmatch: {full_ok / tot * 100:.1f}%")
    hi = [p for p in sample if p.is_highly_sensitised]
    lo = [p for p in sample if not p.is_highly_sensitised]
    hi_in = np.mean([sum(is_compatible(d, p) for d in sample if d is not p)
                     for p in hi])
    lo_in = np.mean([sum(is_compatible(d, p) for d in sample if d is not p)
                     for p in lo])
    print(f"  Compatible donors per high-cPRA patient: {hi_in:.1f}")
    print(f"  Compatible donors per low-cPRA patient:  {lo_in:.1f}")

    print("\n--- Crossmatch is fixed per (donor, patient) couple ---")
    a, b = bp[0], bp[1]
    print(f"  Same answer on repeat: "
          f"{all(is_crossmatch_negative(a, b) == is_crossmatch_negative(a, b) for _ in range(5))}")

    print("\n--- Altruists never receive a kidney ---")
    alt = big.altruists()[0]
    print(f"  Any donor -> altruist compatible: "
          f"{any(is_compatible(d, alt) for d in bp[:200])}")

    print("\n--- Arrivals (8% altruistic even for small batches) ---")
    np.random.seed(3)
    alts = 0
    nid = next_id
    for r in range(1, 201):
        new, nid = generate_arrivals(6, nid, r)
        alts += sum(p.is_altruistic for p in new)
    print(f"  200 rounds x 6 arrivals: {alts} altruists "
          f"({alts / 1200 * 100:.1f}%)")

    print("\n--- Reproducibility ---")
    p1, _ = generate_initial_pool(75, seed=42)
    p2, _ = generate_initial_pool(75, seed=42)
    print(f"  Same seed -> same pool: "
          f"{[x.cpra for x in p1.pairs] == [y.cpra for y in p2.pairs]}")

    print("\n" + "=" * 55)
    print("MODULE 1 COMPLETE")
    print("Ready for Module 2: Pool Evolution")
    print("=" * 55)
