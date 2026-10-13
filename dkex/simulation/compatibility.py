"""Module 2 — Compatibility graph G_t = (V_t, E_t) (Section 3.2).

    w_ij = omega_HLA(i, j)   if ABO compatible AND CIT feasible   (Eq. 2)
           0                 otherwise
    omega_HLA(i, j) = 1 - m_ij / 6                                 (Eq. 3)

Edge i -> j means the donor of node i can give to the patient of node j.
Virtual-crossmatch outcomes and HLA mismatch counts are *properties of the
(donor, patient) couple*, so they are drawn with a deterministic hash of
(seed, i, j): the same edge looks identical in every round and in every
competing configuration (common random numbers).

HLA limitation: m_ij is a population-frequency proxy, not a clinical virtual
crossmatch — see concept note Section 3.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dkex.simulation.logistics import LogisticsNetwork
from dkex.simulation.pool import ABO_TYPES, PoolState

_ABO_IDX = {t: k for k, t in enumerate(ABO_TYPES)}
# _ABO_OK[donor, patient]
_ABO_OK = np.array(
    [
        [1, 1, 1, 1],  # O
        [0, 1, 0, 1],  # A
        [0, 0, 1, 1],  # B
        [0, 0, 0, 1],  # AB
    ],
    dtype=bool,
)

_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = (x + np.uint64(0x9E3779B97F4A7C15)) & _M64
        x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _M64
        x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _M64
        return x ^ (x >> np.uint64(31))


def hash_uniform(seed: int, a: np.ndarray, b: np.ndarray, salt: int) -> np.ndarray:
    """Deterministic U(0,1) for each (a, b) couple."""
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = _splitmix64(np.uint64(seed) ^ (np.uint64(salt) * np.uint64(0x100000001B3)))
        x = _splitmix64(x ^ a)
        x = _splitmix64(x ^ (b * np.uint64(0x9E3779B1)))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


@dataclass
class CompatibilityGraph:
    ids: np.ndarray            # node index -> pair id
    altruist: np.ndarray       # bool per node
    succ: list[np.ndarray]     # out-neighbours (node indices)
    weight: dict[tuple[int, int], float]  # omega_HLA on feasible edges
    n_edges_abo: int           # edges before the CIT filter (diagnostics)

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def n_edges(self) -> int:
        return len(self.weight)

    def density(self) -> float:
        n = self.n
        return self.n_edges / max(1, n * (n - 1))


def build_graph(
    pool: PoolState,
    logistics: LogisticsNetwork,
    seed: int,
    hla_mismatch_p: float = 0.55,
    external_edges: set[tuple[int, int]] | None = None,
) -> CompatibilityGraph:
    """Build G_t.  ``external_edges`` (pair-id couples) replaces the ABO and
    crossmatch tests, e.g. for PrefLib graphs; CIT feasibility still applies."""
    pairs = list(pool.pairs.values())
    n = len(pairs)
    ids = np.array([p.pid for p in pairs], dtype=np.int64)
    altruist = np.array([p.altruist for p in pairs], dtype=bool)
    donor_abo = np.array([_ABO_IDX[p.donor_abo] for p in pairs], dtype=int)
    patient_abo = np.array([_ABO_IDX[p.patient_abo] if p.patient_abo else 0 for p in pairs], dtype=int)
    cpra = np.array([p.cpra for p in pairs])
    centre = np.array([p.centre for p in pairs], dtype=int)

    if n == 0:
        return CompatibilityGraph(ids, altruist, [], {}, 0)

    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    if external_edges is None:
        abo_ok = _ABO_OK[donor_abo[I], patient_abo[J]]
        u_xm = hash_uniform(seed, ids[I], ids[J], salt=1)
        compatible = abo_ok & (u_xm >= cpra[J])
    else:
        compatible = np.zeros((n, n), dtype=bool)
        pos = {int(pid): k for k, pid in enumerate(ids)}
        for a, b in external_edges:
            if a in pos and b in pos:
                compatible[pos[a], pos[b]] = True
    valid = compatible & ~altruist[J] & (I != J)
    n_edges_abo = int(valid.sum())
    cit_ok = logistics.feasible[centre[I], centre[J]]
    valid &= cit_ok

    # HLA mismatch m_ij ~ Binomial(6, p) via inverse CDF on a hashed uniform.
    u_hla = hash_uniform(seed, ids[I], ids[J], salt=2)
    from scipy.stats import binom

    cdf = binom.cdf(np.arange(7), 6, hla_mismatch_p)
    mismatches = np.searchsorted(cdf, u_hla)
    omega = 1.0 - np.minimum(mismatches, 6) / 6.0

    succ = [np.nonzero(valid[i])[0] for i in range(n)]
    src, dst = np.nonzero(valid)
    weight = {(int(i), int(j)): float(omega[i, j]) for i, j in zip(src, dst)}
    return CompatibilityGraph(ids, altruist, succ, weight, n_edges_abo)
