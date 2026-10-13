"""Module 13 — Out-of-distribution validation on PrefLib 00036 (Section 4.6).

PrefLib kidney instances (https://preflib.org/dataset/00036) are compatibility
graphs in the weighted-matching-data (``.wmd``) format.  Two layouts exist:

* modern:  ``# NUMBER ALTERNATIVES: n`` / ``# ALTERNATIVE NAME i: ...`` headers
           followed by ``u,v,w`` edge lines;
* legacy:  ``n`` then ``i,name`` lines, then ``n,m`` then ``u,v,w`` lines.

Both are parsed.  Vertex names containing ``altruist``/``ndd`` mark
non-directed donors.  The files carry no ABO / cPRA / centre data, so:

* the *edges* are taken from the file (ABO + crossmatch replaced);
* high-cPRA status is proxied by low in-degree (bottom ``high_cpra_rate``
  share of patients receive cPRA = 0.9, others 0.3);
* centres are assigned with the configured centre weights.

Dynamic extension: vertices enter in random order — an initial pool, then
Poisson(lambda) arrivals per round — and only edges between present vertices
are active.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dkex.config import CENTRES, PoolConfig
from dkex.simulation.pool import Pair


@dataclass
class PrefLibGraph:
    n: int
    names: dict[int, str]
    edges: set[tuple[int, int]]
    altruists: set[int]


_EDGE = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*([-+0-9.eE]+)\s*$")


def load_wmd(path: str | Path) -> PrefLibGraph:
    lines = Path(path).read_text().splitlines()
    names: dict[int, str] = {}
    edges: set[tuple[int, int]] = set()
    n = 0
    header_done = False
    for k, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = re.match(r"#\s*NUMBER ALTERNATIVES:\s*(\d+)", line, re.I)
            if m:
                n = int(m.group(1))
            m = re.match(r"#\s*ALTERNATIVE NAME (\d+):\s*(.*)", line, re.I)
            if m:
                names[int(m.group(1))] = m.group(2)
            continue
        m = _EDGE.match(line)
        if m and (header_done or n):
            u, v = int(m.group(1)), int(m.group(2))
            edges.add((u, v))
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 1 and parts[0].isdigit() and not n:
            n = int(parts[0])                                  # legacy vertex count
        elif len(parts) == 2 and parts[0].isdigit() and not parts[1].replace(".", "").isdigit():
            names[int(parts[0])] = parts[1]                    # legacy "i,name"
        elif len(parts) == 2 and all(p.isdigit() for p in parts):
            header_done = True                                 # legacy "n,m"
    n = n or (max(max(e) for e in edges) if edges else 0)
    altruists = {i for i, nm in names.items() if re.search(r"altruist|ndd|non.?directed", nm, re.I)}
    return PrefLibGraph(n, names, edges, altruists)


class PrefLibSource:
    """Arrival source for ``DynamicPool`` built from a PrefLib graph."""

    def __init__(self, graph: PrefLibGraph, cfg: PoolConfig, rng: np.random.Generator,
                 centres=CENTRES):
        self.graph, self.cfg, self.rng = graph, cfg, rng
        vids = sorted(set(graph.names) | {v for e in graph.edges for v in e})
        indeg = {v: 0 for v in vids}
        for _, b in graph.edges:
            indeg[b] = indeg.get(b, 0) + 1
        patients = [v for v in vids if v not in graph.altruists]
        order = sorted(patients, key=lambda v: indeg[v])
        n_high = int(round(cfg.high_cpra_rate * len(patients)))
        self.high = set(order[:n_high])
        w = np.array([cfg.centre_weights[c] for c in centres], float)
        self.centre_p = w / w.sum()
        self.queue = list(rng.permutation(vids))

    def _make(self, vid: int, t: int) -> Pair:
        centre = int(self.rng.choice(len(self.centre_p), p=self.centre_p))
        if vid in self.graph.altruists:
            return Pair(pid=int(vid), donor_abo="O", centre=centre, arrival_round=t, altruist=True)
        cpra = 0.9 if vid in self.high else 0.3
        # ABO is irrelevant here: edges come from the file.
        return Pair(pid=int(vid), donor_abo="O", patient_abo="AB", cpra=cpra,
                    centre=centre, arrival_round=t)

    def _pop(self, k: int, t: int) -> list[Pair]:
        out = [self._make(int(v), t) for v in self.queue[:k]]
        self.queue = self.queue[k:]
        return out

    def initial_pool(self) -> list[Pair]:
        pairs = self._pop(self.cfg.initial_pool_size, 0)
        for p in pairs:
            p.wait_months = float(self.rng.uniform(0, 6))
        return pairs

    def arrivals(self, t: int) -> list[Pair]:
        return self._pop(int(self.rng.poisson(self.cfg.arrival_rate)), t)
