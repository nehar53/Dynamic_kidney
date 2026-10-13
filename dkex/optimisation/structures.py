"""Module 5 — Exchange structures: cycles (<= K) and altruistic chains (<= L).

The decision space of every optimiser (ILP, lexicographic, MOEA) is a set
packing over this candidate list, so PoF_t is computed on identical
feasibility (Section 3.6).  Chain enumeration grows like d^L; when the exact
count would exceed ``max_chains`` the long chains (length >= 3) are replaced
by a uniform-ish random-walk sample and ``truncated`` is set so it can be
reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dkex.config import StructureConfig
from dkex.simulation.compatibility import CompatibilityGraph


@dataclass
class StructureSet:
    nodes: list[tuple[int, ...]]          # node indices covered (donor + recipients)
    recipients: list[tuple[int, ...]]     # nodes whose patient receives a kidney
    is_chain: np.ndarray
    hla: np.ndarray                       # summed omega_HLA along the structure
    n_nodes: int
    vertex_structs: list[np.ndarray] = field(default_factory=list)
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def sizes(self) -> np.ndarray:
        return np.array([len(r) for r in self.recipients], dtype=int)

    def finalise(self) -> "StructureSet":
        buckets: list[list[int]] = [[] for _ in range(self.n_nodes)]
        for s, nodes in enumerate(self.nodes):
            for v in nodes:
                buckets[v].append(s)
        self.vertex_structs = [np.array(b, dtype=np.int64) for b in buckets]
        return self


def _cycles(g: CompatibilityGraph, K: int, cap: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for s in range(g.n):
        if g.altruist[s]:
            continue
        stack = [(s, (s,))]
        while stack:
            v, path = stack.pop()
            for w in g.succ[v]:
                w = int(w)
                if w == s and len(path) >= 2:
                    out.append(path)
                elif w > s and w not in path and len(path) < K and not g.altruist[w]:
                    stack.append((w, path + (w,)))
            if len(out) > cap:
                raise RuntimeError(
                    f"cycle count exceeds max_cycles={cap}; lower K or raise the cap"
                )
    return out


def _chains_exact(g, L, budget):
    out = []
    for a in np.nonzero(g.altruist)[0]:
        stack = [(int(a), (int(a),))]
        while stack:
            v, path = stack.pop()
            if len(path) > 1:
                out.append(path)
                if len(out) > budget:
                    return None
            if len(path) - 1 < L:
                for w in g.succ[v]:
                    w = int(w)
                    if w not in path:
                        stack.append((w, path + (w,)))
    return out


def _chains_sampled(g, L, cap, rng):
    import random

    py = random.Random(int(rng.integers(2**63)))
    succ = [s.tolist() for s in g.succ]
    seen: set[tuple[int, ...]] = set()
    altruists = np.nonzero(g.altruist)[0].tolist()
    # Short chains (1-2 recipients) are cheap: keep them all.
    for a in altruists:
        for w in succ[a]:
            seen.add((a, w))
            for x in succ[w]:
                if x != a:
                    seen.add((a, w, x))
    attempts, max_attempts = 0, 20 * cap
    while len(seen) < cap and attempts < max_attempts and altruists and L >= 3:
        attempts += 1
        target = py.randint(3, L)
        path = [py.choice(altruists)]
        while len(path) - 1 < target:
            nxt = [w for w in succ[path[-1]] if w not in path]
            if not nxt:
                break
            path.append(py.choice(nxt))
        if len(path) - 1 >= 3:
            seen.add(tuple(path))
    return sorted(seen)


def enumerate_structures(
    g: CompatibilityGraph, cfg: StructureConfig, rng: np.random.Generator | None = None
) -> StructureSet:
    rng = rng or np.random.default_rng(0)
    cycles = _cycles(g, cfg.max_cycle_length, cfg.max_cycles)
    truncated = False
    chains = _chains_exact(g, cfg.max_chain_length, cfg.max_chains) if cfg.max_chain_length > 0 else []
    if chains is None:
        chains = _chains_sampled(g, cfg.max_chain_length, cfg.max_chains, rng)
        truncated = True

    nodes = cycles + chains
    recipients = [c for c in cycles] + [ch[1:] for ch in chains]
    is_chain = np.array([False] * len(cycles) + [True] * len(chains), dtype=bool)
    hla = np.empty(len(nodes))
    for k, path in enumerate(nodes):
        if is_chain[k]:
            edges = zip(path[:-1], path[1:])
        else:
            edges = zip(path, path[1:] + path[:1])
        hla[k] = sum(g.weight[(int(i), int(j))] for i, j in edges)
    return StructureSet(nodes, recipients, is_chain, hla, g.n, truncated=truncated).finalise()
