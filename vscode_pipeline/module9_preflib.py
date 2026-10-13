import argparse
import contextlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

import module5_nsga3_aos
from module1_generator import HLA_LOCI, NODE_WEIGHTS, Patient, Pool
from module7_clearing import PairedPoolEvolution, simulate_trajectory

# ============================================================
# MODULE 9: OUT-OF-DISTRIBUTION VALIDATION ON PREFLIB 00036
# Concept note Section 4.6
# Paper 2 — Dynamic KEP Implementation
# ============================================================
#
# PrefLib kidney instances (https://preflib.org/dataset/00036)
# are compatibility graphs in the .wmd format. Two layouts exist:
#
#   modern:  "# NUMBER ALTERNATIVES: n"
#            "# ALTERNATIVE NAME i: Pair i"   (or "Altruist ...")
#            then edge lines  "u,v,w"
#   legacy:  "n" / "i,name" lines / "n,m" / edge lines "u,v,w"
#
# What is taken from the file: vertices, altruists, EDGES.
# What is NOT in the file and is filled in (state in the paper):
#   - cPRA: proxy from in-degree. The 30% of patients with the
#     fewest incoming edges get cPRA 0.90 (highly sensitised),
#     the rest 0.30.
#   - hospital node: random, with module 1's NODE_WEIGHTS.
#   - blood types / HLA: placeholders (edges come from the file).
#
# Dynamic extension: vertices enter in a seeded random order:
# an initial pool, then Poisson(lambda) arrivals per round.
# Only edges between vertices currently in the pool are active.

HIGH_CPRA_SHARE = 0.30
DEFAULT_CONFIGS = [
    {"name": "dyn_nsga3_ucb1_myopic", "method": "nsga3",
     "aos": "ucb1", "clearing": "myopic", "dynamic": True},
    {"name": "dyn_nsga3_linucb_myopic", "method": "nsga3",
     "aos": "linucb", "clearing": "myopic", "dynamic": True},
    {"name": "myopic_ilp", "method": "ilp", "clearing": "myopic"},
    {"name": "dickerson_lex", "method": "lex", "clearing": "myopic"},
]


# ============================================================
# LOADER
# ============================================================

@dataclass
class PrefLibGraph:
    n: int
    names: Dict[int, str]
    edges: Dict[Tuple[int, int], float]
    altruists: Set[int]

    @property
    def vertices(self) -> List[int]:
        v = set(self.names) | {i for e in self.edges for i in e}
        return sorted(v)


_EDGE = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*([-+0-9.eE]+)\s*$")


def load_wmd(path: str) -> PrefLibGraph:
    """Parse a PrefLib .wmd file (modern or legacy layout)."""
    names: Dict[int, str] = {}
    edges: Dict[Tuple[int, int], float] = {}
    n = 0
    in_edges = False
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = re.match(r"#\s*NUMBER ALTERNATIVES:\s*(\d+)", line, re.I)
            if m:
                n = int(m.group(1))
            m = re.match(r"#\s*ALTERNATIVE NAME (\d+):\s*(.*)", line,
                         re.I)
            if m:
                names[int(m.group(1))] = m.group(2).strip()
            continue
        m = _EDGE.match(line)
        if m and (in_edges or n):
            edges[(int(m.group(1)), int(m.group(2)))] = float(m.group(3))
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 1 and parts[0].isdigit() and not n:
            n = int(parts[0])                       # legacy vertex count
        elif (len(parts) == 2 and parts[0].isdigit()
              and not parts[1].replace(".", "").isdigit()):
            names[int(parts[0])] = parts[1]          # legacy "i,name"
        elif len(parts) == 2 and all(p.isdigit() for p in parts):
            in_edges = True                          # legacy "n,m"
    if not n:
        n = max((max(e) for e in edges), default=0)
    altruists = {i for i, nm in names.items()
                 if re.search(r"altruist|ndd|non.?directed", nm, re.I)}
    return PrefLibGraph(n, names, edges, altruists)


# ============================================================
# PATIENTS + GRAPH BUILDER FROM A PREFLIB GRAPH
# ============================================================

def make_patients(graph: PrefLibGraph, seed: int) -> Dict[int, Patient]:
    """One Patient object per PrefLib vertex (see header notes)."""
    rng = np.random.default_rng(seed)
    indeg = {v: 0 for v in graph.vertices}
    for (_, j) in graph.edges:
        indeg[j] = indeg.get(j, 0) + 1
    patients = [v for v in graph.vertices if v not in graph.altruists]
    order = sorted(patients, key=lambda v: (indeg[v], v))
    high = set(order[:int(round(HIGH_CPRA_SHARE * len(patients)))])

    nodes = list(NODE_WEIGHTS.keys())
    w = np.array(list(NODE_WEIGHTS.values()), float)
    w = w / w.sum()
    out = {}
    for v in graph.vertices:
        cpra = 0.90 if v in high else 0.30
        out[v] = Patient(
            pair_id=int(v),
            blood_type_patient="AB", blood_type_donor="O",  # placeholders
            cpra=cpra, is_highly_sensitised=cpra >= 0.80,
            hla_patient=rng.integers(0, 2, HLA_LOCI),
            hla_donor=rng.integers(0, 2, HLA_LOCI),
            is_altruistic=v in graph.altruists,
            hospital_node=str(rng.choice(nodes, p=w)),
            waiting_time=0, arrival_round=0,
        )
    return out


def make_graph_builder(graph: PrefLibGraph) -> Callable:
    """Replacement for module 3's build_compatibility_graph."""
    def builder(pool_pairs: List[Patient]) -> Dict[Tuple[int, int], float]:
        present = {p.pair_id for p in pool_pairs}
        out = {}
        for (i, j), w in graph.edges.items():
            if i in present and j in present and i != j \
                    and j not in graph.altruists:
                out[(i, j)] = w if 0 < w <= 1 else 1.0
        return out
    return builder


@contextlib.contextmanager
def preflib_edges(builder: Callable):
    """
    Module 5 calls build_compatibility_graph() internally, so for
    the OOD run we temporarily point that name at the PrefLib
    builder and restore it afterwards.
    """
    original = module5_nsga3_aos.build_compatibility_graph
    module5_nsga3_aos.build_compatibility_graph = builder
    try:
        yield
    finally:
        module5_nsga3_aos.build_compatibility_graph = original


class PrefLibPoolEvolution(PairedPoolEvolution):
    """Arrivals are the next vertices of a seeded random queue."""

    def __init__(self, *args, queue: List[Patient], **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = queue

    def compute_arrivals(self) -> List[Patient]:
        rng = np.random.RandomState(
            (self.base_seed * 7919 + self.current_round) % (2 ** 32))
        k = int(rng.poisson(self.lambda_arrivals))
        new = self.queue[:k]
        self.queue = self.queue[k:]
        for p in new:
            p.arrival_round = self.current_round
        self.pool.pairs.extend(new)
        return new


# ============================================================
# OOD EXPERIMENT
# ============================================================

def run_preflib_ood(
    wmd_path: str,
    configs: Optional[List[Dict]] = None,
    taus: Tuple[int, ...] = (12,),
    n_runs: int = 10,
    n_rounds: int = 25,
    n_gen: int = 150,
    initial_size: int = 75,
    lambda_arrivals: float = 6.0,
    selection_rule: str = "floor",
    verbose: bool = False,
) -> pd.DataFrame:
    """Run configurations on a PrefLib graph; one row per round."""
    import copy

    configs = configs or DEFAULT_CONFIGS
    graph = load_wmd(wmd_path)
    builder = make_graph_builder(graph)
    rows: List[Dict] = []

    with preflib_edges(builder):
        for tau in taus:
            for run in range(n_runs):
                seed = 1000 + run
                base = make_patients(graph, seed)
                order = list(np.random.default_rng(seed)
                             .permutation(graph.vertices))
                for cfg in configs:
                    # fresh copies: matching mutates waiting times
                    pts = [copy.deepcopy(base[int(v)]) for v in order]
                    pool = Pool(pairs=pts[:initial_size], round_number=0)
                    queue = pts[initial_size:]

                    def factory(pool, next_id, tau, queue=queue,
                                seed=seed):
                        return PrefLibPoolEvolution(
                            initial_pool=pool, next_pair_id=next_id,
                            tau_months=tau,
                            lambda_arrivals=lambda_arrivals,
                            seed=seed, base_seed=seed, queue=queue)

                    traj = simulate_trajectory(
                        cfg, tau, seed, n_rounds=n_rounds, n_gen=n_gen,
                        selection_rule=selection_rule,
                        graph_builder=builder,
                        initial_pool=(pool, max(graph.vertices) + 1),
                        evolution_factory=factory, verbose=verbose)
                    for r in traj:
                        r["dataset"] = Path(wmd_path).name
                    rows.extend(traj)
                    print(f"[OOD] {cfg['name']} tau={tau} run={run} done")
    return pd.DataFrame(rows)


def write_synthetic_wmd(path: str, n: int = 120, p_edge: float = 0.08,
                        n_altruists: int = 8, seed: int = 0) -> str:
    """Small random .wmd file in PrefLib's modern layout (testing)."""
    rng = np.random.default_rng(seed)
    lines = [f"# FILE NAME: {Path(path).name}",
             f"# NUMBER ALTERNATIVES: {n}"]
    for i in range(1, n + 1):
        kind = "Altruist" if i > n - n_altruists else "Pair"
        lines.append(f"# ALTERNATIVE NAME {i}: {kind} {i}")
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j and j <= n - n_altruists and rng.random() < p_edge:
                lines.append(f"{i},{j},1.0")
    Path(path).write_text("\n".join(lines))
    return path


# ============================================================
# TEST / COMMAND LINE
# ============================================================

if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="PrefLib OOD validation")
    ap.add_argument("--wmd", default=None,
                    help="PrefLib 00036 .wmd file (omit for a self-test)")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--gen", type=int, default=20)
    ap.add_argument("--taus", type=int, nargs="+", default=[12])
    ap.add_argument("--out", default="results/ood_preflib.csv")
    args = ap.parse_args()

    print("=" * 55)
    print("MODULE 9: PREFLIB OUT-OF-DISTRIBUTION VALIDATION")
    print("=" * 55)

    wmd = args.wmd
    if wmd is None:
        Path("results").mkdir(exist_ok=True)
        wmd = write_synthetic_wmd("results/selftest.wmd")
        print("No --wmd given: self-test on a synthetic .wmd file.")

    g = load_wmd(wmd)
    print(f"\nFile:       {wmd}")
    print(f"Vertices:   {len(g.vertices)}")
    print(f"Altruists:  {len(g.altruists)}")
    print(f"Edges:      {len(g.edges)}")

    df = run_preflib_ood(wmd, taus=tuple(args.taus), n_runs=args.runs,
                         n_rounds=args.rounds, n_gen=args.gen)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    cleared = df[df["cleared"]]
    print("\n--- Mean per configuration ---")
    print(cleared.groupby("config")[["F1", "F1_ILP", "PoF", "F3"]]
          .mean().round(3).to_string())
    print(f"\nSaved: {args.out}")

    print("\n" + "=" * 55)
    print("MODULE 9 COMPLETE")
    print("Ready for Module 10: Full Experiment")
    print("=" * 55)
