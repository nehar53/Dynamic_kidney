import argparse
import copy
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from module1_generator import CPRA_THRESHOLD, HLA_LOCI, NODE_WEIGHTS, Patient, Pool
from module2_pool_evolution import LAMBDA_ARRIVALS, PoolEvolution
from module3_logistics import is_cit_feasible
from module7_clearing import BATCH_THRESHOLD, simulate_trajectory
from module8_stats import compare_many_metrics, format_mean_sd, summary_table

# ============================================================
# MODULE 9: OUT-OF-DISTRIBUTION VALIDATION ON PREFLIB 00036
# Concept note Section 4.6
# Paper 2 — Dynamic KEP Implementation
# ============================================================
#
# Purpose: show that the conclusions do not depend on the
# ANZKX-calibrated generator, by running the SAME framework
# (modules 2-8) on independent kidney-exchange graphs from
# PrefLib dataset 00036 (https://preflib.org/dataset/00036).
#
# Input: PrefLib .wmd files (weighted matching data). Layouts:
#   modern:  "# NUMBER ALTERNATIVES: n"
#            "# ALTERNATIVE NAME i: Pair i" / "Altruist i"
#            then edge lines "u,v,w"
#   legacy:  "n" / "i,name" lines / "n,m" / edge lines "u,v,w"
# Companion file with the same stem and extension .dat (PrefLib
# 00036 ships one per instance), header:
#   Pair,Patient,Donor,Wife-P?,%Pra,Out-Deg,Altruist
# Real blood types, cPRA (%Pra as a fraction) and the altruist
# flag are read from it. Only if it is missing are the values
# below filled in.
#
# Taken from the file:  vertices, altruists, COMPATIBILITY EDGES
#   (they replace module 3's ABO + crossmatch test)
# Filled in when missing (state these in the paper):
#   cPRA        proxy from in-degree: the HIGH_CPRA_SHARE of
#               patients with the fewest incoming edges get
#               cPRA 0.90 (highly sensitised), the rest 0.30
#   hospital    random, with module 1's NODE_WEIGHTS
#   HLA         random 6-locus profile (only for tie-breaks)
# CIT feasibility (module 3) is still applied on top of the file
# edges, so operational constraints are identical.
#
# Dynamic extension: vertices enter in a seeded random order:
# an initial pool, then Poisson(lambda) arrivals per round
# (module 2's paired random numbers). Only edges between
# vertices currently in the pool are active.

HIGH_CPRA_SHARE = 0.30
OOD_INITIAL_POOL = 75
ALL_CONFIGS = {
    "static_nsga2_ucb1": {"method": "nsga2", "aos": "ucb1", "clearing": "myopic", "dynamic": False},
    "static_nsga3_fixed": {"method": "nsga3", "aos": "none", "clearing": "myopic", "dynamic": False},
    "dyn_nsga3_ucb1_myopic": {"method": "nsga3", "aos": "ucb1", "clearing": "myopic", "dynamic": True},
    "dyn_nsga3_linucb_myopic": {"method": "nsga3", "aos": "linucb", "clearing": "myopic", "dynamic": True},
    "dyn_nsga3_ucb1_batched": {"method": "nsga3", "aos": "ucb1", "clearing": "batched", "dynamic": True},
    "dyn_nsga3_linucb_batched": {"method": "nsga3", "aos": "linucb", "clearing": "batched", "dynamic": True},
    "myopic_ilp": {"method": "ilp", "clearing": "myopic"},
    "dickerson_lex": {"method": "lex", "clearing": "myopic"},
}
for _n, _c in ALL_CONFIGS.items():
    _c["name"] = _n

# Default OOD comparison: the central question (myopic vs batched),
# the AOS comparison, and the two ILP reference points.
DEFAULT_CONFIGS = [
    {"name": "dyn_nsga3_ucb1_myopic", "method": "nsga3", "aos": "ucb1",
     "clearing": "myopic", "dynamic": True},
    {"name": "dyn_nsga3_linucb_myopic", "method": "nsga3", "aos": "linucb",
     "clearing": "myopic", "dynamic": True},
    {"name": "dyn_nsga3_ucb1_batched", "method": "nsga3", "aos": "ucb1",
     "clearing": "batched", "dynamic": True},
    {"name": "myopic_ilp", "method": "ilp", "clearing": "myopic"},
    {"name": "dickerson_lex", "method": "lex", "clearing": "myopic"},
]


# ============================================================
# LOADER
# ============================================================

@dataclass
class PrefLibGraph:
    name: str
    n: int
    names: Dict[int, str]
    edges: Dict[Tuple[int, int], float]
    altruists: Set[int]
    attributes: Dict[int, Dict] = field(default_factory=dict)

    @property
    def vertices(self) -> List[int]:
        return sorted(set(self.names) | {i for e in self.edges for i in e})


_EDGE = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*([-+0-9.eE]+)\s*$")


def load_wmd(path: str) -> PrefLibGraph:
    """Parse a PrefLib .wmd file (modern or legacy layout)."""
    names: Dict[int, str] = {}
    edges: Dict[Tuple[int, int], float] = {}
    n, in_edges = 0, False
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = re.match(r"#\s*NUMBER ALTERNATIVES:\s*(\d+)", line, re.I)
            if m:
                n = int(m.group(1))
            m = re.match(r"#\s*ALTERNATIVE NAME (\d+):\s*(.*)", line, re.I)
            if m:
                names[int(m.group(1))] = m.group(2).strip()
            continue
        m = _EDGE.match(line)
        if m and (in_edges or n):
            u, v = int(m.group(1)), int(m.group(2))
            if u != v:
                edges[(u, v)] = float(m.group(3))
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
    attributes = _load_companion(Path(path))
    altruists |= {v for v, a in attributes.items() if a.get("altruist")}
    for v in attributes:
        names.setdefault(v, f"Pair {v}")
    g = PrefLibGraph(Path(path).name, max(n, len(names)), names, edges, altruists)
    g.attributes = attributes
    return g


def _load_companion(path: Path) -> Dict[int, Dict]:
    """Optional per-vertex data (.dat / .csv next to the .wmd)."""
    for ext in (".dat", ".csv"):
        f = path.with_suffix(ext)
        if not f.exists():
            continue
        rows = list(csv.reader(f.read_text().splitlines()))
        if len(rows) < 2:
            return {}
        head = [h.strip().lower() for h in rows[0]]

        def col(*keys):
            for k in keys:
                for i, h in enumerate(head):
                    if k in h:
                        return i
            return None
        c_id = col("pair", "id", "vertex")
        c_pra = col("cpra", "pra")
        c_pb = col("patient")
        c_db = col("donor")
        c_alt = col("altruist", "ndd")
        out = {}
        for r in rows[1:]:
            try:
                vid = int(r[c_id]) if c_id is not None else len(out) + 1
            except (ValueError, IndexError):
                continue
            a = {}
            if c_pra is not None:
                try:
                    pra = float(r[c_pra])
                    a["cpra"] = pra / 100.0 if pra > 1 else pra
                except (ValueError, IndexError):
                    pass
            for key, c in (("patient_blood", c_pb), ("donor_blood", c_db)):
                if c is not None and c < len(r) and r[c].strip().upper() in ("O", "A", "B", "AB"):
                    a[key] = r[c].strip().upper()
            if c_alt is not None and c_alt < len(r):
                a["altruist"] = r[c_alt].strip() in ("1", "true", "True", "yes")
            out[vid] = a
        return out
    return {}


# ============================================================
# PATIENTS + GRAPH BUILDER FROM A PREFLIB GRAPH
# ============================================================

def make_patients(graph: PrefLibGraph, seed: int) -> Dict[int, Patient]:
    """One Patient per PrefLib vertex (see header for filled-in values)."""
    rng = np.random.default_rng(seed)
    indeg = {v: 0 for v in graph.vertices}
    for (_, j) in graph.edges:
        indeg[j] = indeg.get(j, 0) + 1
    patients = [v for v in graph.vertices if v not in graph.altruists]
    order = sorted(patients, key=lambda v: (indeg[v], v))
    proxy_high = set(order[:int(round(HIGH_CPRA_SHARE * len(patients)))])

    nodes = list(NODE_WEIGHTS.keys())
    w = np.array(list(NODE_WEIGHTS.values()), float)
    w /= w.sum()
    out = {}
    for v in graph.vertices:
        a = graph.attributes.get(v, {})
        alt = v in graph.altruists
        cpra = 0.0 if alt else a.get("cpra", 0.90 if v in proxy_high else 0.30)
        out[v] = Patient(
            pair_id=int(v),
            blood_type_patient=None if alt else a.get("patient_blood", "AB"),
            blood_type_donor=a.get("donor_blood", "O"),
            cpra=float(cpra),
            is_highly_sensitised=(not alt) and cpra >= CPRA_THRESHOLD,
            hla_patient=None if alt else rng.integers(0, 8, HLA_LOCI),
            hla_donor=rng.integers(0, 8, HLA_LOCI),
            is_altruistic=alt,
            hospital_node=str(rng.choice(nodes, p=w)),
            waiting_time=0, arrival_round=0,
            xm_salt=int(rng.integers(0, 2 ** 31 - 1)),
        )
    return out


def make_graph_builder(graph: PrefLibGraph) -> Callable:
    """
    Replacement for module 3's build_compatibility_graph:
    file edges among vertices currently in the pool, never into an
    altruist, filtered by the CIT constraint. Weights are the file
    weights scaled to [0, 1] (used only as ILP tie-breaks).
    """
    wmax = max((w for w in graph.edges.values() if w > 0), default=1.0)

    def builder(pool_pairs: List[Patient]) -> Dict[Tuple[int, int], float]:
        by_id = {p.pair_id: p for p in pool_pairs}
        out = {}
        for (i, j), w in graph.edges.items():
            if i in by_id and j in by_id and not by_id[j].is_altruistic \
                    and is_cit_feasible(by_id[i].hospital_node,
                                        by_id[j].hospital_node):
                out[(i, j)] = max(0.0, w) / wmax
        return out
    return builder


class PrefLibPoolEvolution(PoolEvolution):
    """
    Module 2's pool equation, but arrivals are the next vertices of
    a seeded random queue (count ~ Poisson(lambda), paired draws).
    """

    def __init__(self, *args, queue: List[Patient], **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = queue

    def compute_arrivals(self) -> List[Patient]:
        k = int(self._round_rng(1).poisson(self.lambda_arrivals))
        new, self.queue = self.queue[:k], self.queue[k:]
        for p in new:
            p.arrival_round = self.current_round
            p.waiting_time = 0
        self.pool.pairs.extend(new)
        return new


# ============================================================
# OOD EXPERIMENT
# ============================================================

def instance_size(f: Path) -> int:
    """Number of vertices (pairs + altruists) of a PrefLib instance."""
    dat = f.with_suffix(".dat")
    if dat.exists():
        return max(0, sum(1 for line in dat.read_text().splitlines() if line.strip()) - 1)
    return len(load_wmd(str(f)).vertices)


def wmd_files(path: str, min_size: int = 0, max_size: int = 10 ** 9,
              sample: Optional[int] = None, seed: int = 0,
              per_size: Optional[int] = None) -> List[Path]:
    """
    A single .wmd file, or the .wmd files of a folder, optionally
    filtered by instance size and reduced to a reproducible subset
    (PrefLib 00036 has ~310 instances):
      per_size  seeded random choice of this many instances for every
                distinct instance size (size-stratified, recommended)
      sample    seeded random choice of this many instances overall
    """
    p = Path(path)
    if not p.is_dir():
        return [p]
    sized = [(f, instance_size(f)) for f in sorted(p.glob("*.wmd"))]
    sized = [(f, n) for f, n in sized if min_size <= n <= max_size]
    rng = np.random.default_rng(seed)
    if per_size is not None:
        chosen = []
        for n in sorted({n for _, n in sized}):
            group = [f for f, m in sized if m == n]
            k = min(per_size, len(group))
            idx = sorted(rng.choice(len(group), k, replace=False))
            chosen += [group[i] for i in idx]
        files = chosen
    else:
        files = [f for f, _ in sized]
    if sample is not None and sample < len(files):
        idx = rng.choice(len(files), sample, replace=False)
        files = [files[i] for i in sorted(idx)]
    return files


def list_instances(path: str) -> pd.DataFrame:
    """Size table of all instances in a folder (to choose a subset)."""
    rows = []
    for f in sorted(Path(path).glob("*.wmd")):
        g = load_wmd(str(f))
        rows.append({"file": f.name, "vertices": len(g.vertices),
                     "altruists": len(g.altruists), "edges": len(g.edges),
                     "high_cpra": sum(1 for a in g.attributes.values()
                                      if a.get("cpra", 0) >= CPRA_THRESHOLD)})
    return pd.DataFrame(rows)


def run_preflib_ood(
    wmd_path: str,
    configs: Optional[List[Dict]] = None,
    taus: Tuple[int, ...] = (12,),
    n_runs: int = 10,
    n_rounds: int = 25,
    n_gen: int = 150,
    initial_size: int = OOD_INITIAL_POOL,
    lambda_arrivals: float = LAMBDA_ARRIVALS,
    selection_rule: str = "floor",
    min_f1_ratio: float = 0.9,
    verbose: bool = False,
    min_size: int = 0,
    max_size: int = 10 ** 9,
    sample: Optional[int] = None,
    per_size: Optional[int] = None,
    scale_all: bool = True,
) -> pd.DataFrame:
    """
    Run the configurations on one PrefLib file or a folder of
    files. Returns one row per (dataset, config, tau, run, round).
    All configurations of the same run share the arrival order and
    the paired random numbers.

    Instance scaling (scale_all=True, default; same rule for every
    instance so all are treated alike): half of the instance's
    vertices form the initial pool, the rest arrive spread over the
    n_rounds rounds (lambda = remaining / n_rounds), and the batching
    threshold scales with the pool (n* = 90/75 x initial pool).
    scale_all=False keeps the ANZKX defaults (pool 75, lambda 10,
    n* 90) for instances large enough and scales only small ones.
    """
    configs = configs or DEFAULT_CONFIGS
    rows: List[Dict] = []
    files = wmd_files(wmd_path, min_size, max_size, sample, per_size=per_size)
    print(f"[OOD] {len(files)} instance(s) selected")
    for f in files:
        graph = load_wmd(str(f))
        builder = make_graph_builder(graph)
        verts = graph.vertices
        init, lam, n_star = initial_size, lambda_arrivals, BATCH_THRESHOLD
        if scale_all or len(verts) < initial_size + n_rounds * lambda_arrivals:
            # half the vertices start in the pool, the rest arrive
            # spread over the rounds (same dynamics, scaled)
            init = max(2, len(verts) // 2)
            lam = max(0.5, (len(verts) - init) / n_rounds)
            # batched threshold scaled like the pool: n* = 90/75 x initial pool
            n_star = max(3, int(round(BATCH_THRESHOLD / initial_size * init)))
            print(f"[OOD] {f.name}: {len(verts)} vertices -> initial pool {init}, "
                  f"lambda {lam:.2f}/round, n* {n_star} (scaled to the instance)")
        for tau in taus:
            for run in range(n_runs):
                seed = 1000 + run
                base = make_patients(graph, seed)
                order = [int(v) for v in np.random.default_rng(seed).permutation(verts)]
                for cfg in configs:
                    pts = [copy.deepcopy(base[v]) for v in order]
                    w_rng = np.random.default_rng([seed, 11])
                    for p in pts[:init]:                      # spread initial waits
                        p.waiting_time = int(w_rng.integers(0, 6))
                    pool = Pool(pairs=pts[:init], round_number=0)
                    queue = pts[init:]

                    def factory(pool, next_id, tau, queue=queue, seed=seed):
                        return PrefLibPoolEvolution(
                            pool, next_id, tau_months=tau,
                            lambda_arrivals=lam, seed=seed,
                            queue=queue)

                    traj = simulate_trajectory(
                        cfg, tau, seed, n_rounds=n_rounds, n_gen=n_gen,
                        selection_rule=selection_rule,
                        min_f1_ratio=min_f1_ratio,
                        batch_threshold=n_star,
                        graph_builder=builder,
                        initial_pool=(pool, max(verts) + 1),
                        evolution_factory=factory, verbose=verbose)
                    for r in traj:
                        r["dataset"] = f.name
                        r["n_vertices"] = len(verts)
                        r["run"] = run
                    rows.extend(traj)
                    print(f"[OOD] {f.name} {cfg['name']} tau={tau} run={run} done",
                          flush=True)
    return pd.DataFrame(rows)


def summarise_ood(rounds: pd.DataFrame) -> pd.DataFrame:
    """Per-run trajectory metrics (same definitions as module 10)."""
    out = []
    for (ds, cfg, tau, run), g in rounds.groupby(["dataset", "config", "tau", "run"]):
        c = g[g["cleared"]]
        out.append({
            "dataset": ds, "config": cfg, "tau": tau, "run": run,
            "run_seed": g["run_seed"].iloc[0],
            # paired unit = (instance, run): all configurations of the
            # same instance and run share arrivals and exits
            "pair_id": f"{ds}|{run}",
            "n_vertices": int(g["n_vertices"].iloc[0]) if "n_vertices" in g else np.nan,
            "pof_mean": c["PoF"].mean() if len(c) else np.nan,
            "total_transplants": c["F1"].sum() if len(c) else 0,
            "total_high_cpra_transplants": c["high_cpra_matched"].sum() if len(c) else 0,
            "mean_F3": c["F3"].mean() if len(c) else np.nan,
            "mean_HV": c["HV"].mean() if len(c) else np.nan,
        })
    return pd.DataFrame(out)


def write_synthetic_wmd(path: str, n: int = 160, p_edge: float = 0.06,
                        n_altruists: int = 12, seed: int = 0,
                        legacy: bool = False) -> str:
    """Random .wmd file in PrefLib layout (for the self-test only)."""
    rng = np.random.default_rng(seed)
    alt = set(range(n - n_altruists + 1, n + 1))
    edges = [(i, j) for i in range(1, n + 1) for j in range(1, n + 1)
             if i != j and j not in alt and rng.random() < p_edge]
    if legacy:
        lines = [str(n)] + [f"{i},{'Altruist' if i in alt else 'Pair'} {i}"
                            for i in range(1, n + 1)]
        lines += [f"{n},{len(edges)}"] + [f"{i},{j},1" for i, j in edges]
    else:
        lines = [f"# FILE NAME: {Path(path).name}", f"# NUMBER ALTERNATIVES: {n}",
                 f"# NUMBER EDGES: {len(edges)}"]
        lines += [f"# ALTERNATIVE NAME {i}: {'Altruist' if i in alt else 'Pair'} {i}"
                  for i in range(1, n + 1)]
        lines += [f"{i},{j},1.0" for i, j in edges]
    Path(path).write_text("\n".join(lines))
    return path


# ============================================================
# TEST / COMMAND LINE
# ============================================================

if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="PrefLib OOD validation (Section 4.6)")
    ap.add_argument("--wmd", default=None,
                    help="PrefLib 00036 .wmd file or folder (omit = self-test)")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--gen", type=int, default=20)
    ap.add_argument("--taus", type=int, nargs="+", default=[12])
    ap.add_argument("--out", default="results/ood")
    ap.add_argument("--min-size", type=int, default=0,
                    help="only instances with at least this many vertices")
    ap.add_argument("--max-size", type=int, default=10 ** 9,
                    help="only instances with at most this many vertices")
    ap.add_argument("--per-size", type=int, default=None,
                    help="size-stratified: this many instances per instance size")
    ap.add_argument("--sample", type=int, default=None,
                    help="seeded random sample of this many instances")
    ap.add_argument("--configs", nargs="+", default=[c["name"] for c in DEFAULT_CONFIGS],
                    choices=list(ALL_CONFIGS),
                    help="configurations to run (e.g. only myopic_ilp dickerson_lex "
                         "to cover all instances cheaply)")
    ap.add_argument("--list", action="store_true",
                    help="only print the size table of the instances")
    args = ap.parse_args()

    if args.list and args.wmd:
        t = list_instances(args.wmd)
        print(t.to_string(index=False))
        print(f"\n{len(t)} instances. Instances per size (vertices):")
        print(t.groupby("vertices").agg(instances=("file", "count"),
                                        mean_altruists=("altruists", "mean"),
                                        mean_edges=("edges", "mean"),
                                        mean_high_cpra=("high_cpra", "mean"))
                .round(1).to_string())
        raise SystemExit

    print("=" * 62)
    print("MODULE 9: PREFLIB OUT-OF-DISTRIBUTION VALIDATION")
    print("=" * 62)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wmd = args.wmd
    if wmd is None:
        wmd = write_synthetic_wmd(str(out / "selftest.wmd"))
        legacy = write_synthetic_wmd(str(out / "selftest_legacy.txt"), legacy=True)
        g1, g2 = load_wmd(wmd), load_wmd(legacy)
        print("\nNo --wmd given: self-test on synthetic PrefLib-format files.")
        print(f"  Modern and legacy layouts parse identically: "
              f"{set(g1.edges) == set(g2.edges) and g1.altruists == g2.altruists}")

    files = wmd_files(wmd, args.min_size, args.max_size, args.sample,
                      per_size=args.per_size)
    sizes = pd.Series([instance_size(f) for f in files])
    print(f"\nSelected {len(files)} instance(s); sizes: "
          f"{sizes.value_counts().sort_index().to_dict()}")
    if Path(wmd).is_dir() and len(files) > 40 and \
            any(c["method"] in ("nsga3", "nsga2") for c in DEFAULT_CONFIGS
                if c["name"] in args.configs):
        print("  NOTE: many instances with NSGA-III configurations is expensive; "
              "consider --per-size 3 (or run only myopic_ilp dickerson_lex on all).")

    for f in files[:3]:
        g = load_wmd(str(f))
        b = make_graph_builder(g)
        pts = list(make_patients(g, 1).values())
        e = b(pts)
        alt_ids = {p.pair_id for p in pts if p.is_altruistic}
        print(f"\nFile: {f.name}")
        print(f"  Vertices: {len(g.vertices)}   altruists: {len(g.altruists)}   "
              f"edges: {len(g.edges)}   companion .dat: {bool(g.attributes)}   "
              f"high-cPRA: {sum(p.is_highly_sensitised for p in pts)}")
        print(f"  Edges into altruists after building: "
              f"{sum(j in alt_ids for (_, j) in e)}")

    configs = [ALL_CONFIGS[n] for n in args.configs]
    df = run_preflib_ood(wmd, configs=configs, taus=tuple(args.taus),
                         n_runs=args.runs, n_rounds=args.rounds, n_gen=args.gen,
                         min_size=args.min_size, max_size=args.max_size,
                         sample=args.sample, per_size=args.per_size)
    df.to_csv(out / "ood_rounds.csv", index=False)
    runs = summarise_ood(df)
    runs.to_csv(out / "ood_runs.csv", index=False)

    metrics = ["pof_mean", "total_transplants", "total_high_cpra_transplants", "mean_F3"]
    tab = summary_table(runs, metrics)
    for m in metrics:
        tab[m] = format_mean_sd(tab, m, 3 if m == "mean_F3" else 2)
    print("\n--- OOD results over all selected instances (mean ± sd) ---")
    print(tab[["config", "tau"] + metrics].to_string(index=False))
    tab.to_csv(out / "ood_table_overall.csv", index=False)

    if runs["n_vertices"].nunique() > 1:
        by_size = summary_table(runs, ["pof_mean", "total_transplants"],
                                by=("n_vertices", "config"))
        by_size.to_csv(out / "ood_table_by_size.csv", index=False)
        print("\n--- By instance size (mean PoF %) ---")
        print(by_size.pivot(index="n_vertices", columns="config",
                            values="pof_mean_mean").round(2).to_string())

    ilp_ok = all((df.loc[df["cleared"], "F1"] <= df.loc[df["cleared"], "F1_ILP"] + 1e-9))
    print(f"\n  No configuration exceeds the ILP ceiling: {ilp_ok}")

    comps = [c for c in [("dyn_nsga3_ucb1_myopic", "dyn_nsga3_ucb1_batched"),
                         ("dyn_nsga3_ucb1_myopic", "dyn_nsga3_linucb_myopic"),
                         ("dyn_nsga3_ucb1_myopic", "dickerson_lex")]
             if set(c) <= set(args.configs)]
    if comps and runs["pair_id"].nunique() >= 2:
        st = compare_many_metrics(runs, ["pof_mean", "total_transplants",
                                         "total_high_cpra_transplants"],
                                  comps, pair_col="pair_id")
        st.to_csv(out / "ood_wilcoxon.csv", index=False)
        print("\n--- Paired Wilcoxon (unit = instance x run, Bonferroni) ---")
        print(st[["tau", "metric", "config_a", "config_b", "n", "mean_a", "mean_b",
                  "p_bonferroni", "significant", "cohens_d", "effect"]]
              .round(4).to_string(index=False))
        if runs["pair_id"].nunique() < 6:
            print("  (fewer than 6 paired units: significance is not reachable yet)")
    print(f"\nSaved in {out}/: ood_rounds.csv, ood_runs.csv, ood_table_overall.csv"
          f"{', ood_table_by_size.csv' if runs['n_vertices'].nunique() > 1 else ''}"
          f"{', ood_wilcoxon.csv' if comps else ''}")

    print("\n" + "=" * 62)
    print("MODULE 9 COMPLETE")
    print("Ready for Module 10: Full Experiment")
    print("=" * 62)
