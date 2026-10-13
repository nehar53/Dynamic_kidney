"""Module 8 — Many-objective evolutionary optimiser (Section 4.2).

NSGA-III (Deb & Jain 2014) with reference-point niching, plus NSGA-II for the
Paper-1 static baseline.  The mutation rate of every generation is chosen by
an ``OperatorSelector`` (UCB1 / LinUCB / fixed) and rewarded with the
generation's hypervolume improvement.

Representation: a solution is a conflict-free set of structure indices (set
packing).  Variation operators always return feasible solutions:

* crossover  keep structures common to both parents, inherit the others with
             probability 1/2 when they do not conflict, then *fill*;
* mutation   drop each structure with probability p_m, force-insert
             Binomial(|x|, p_m) random structures (evicting conflicts), then
             *fill*;
* fill       visit free vertices in random order and add the first fully-free
             candidate structure found among ``fill_attempts`` random draws.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import numpy as np

from dkex.aos.base import OperatorSelector
from dkex.config import MOEAConfig
from dkex.optimisation.objectives import ObjectiveEvaluator
from dkex.optimisation.structures import StructureSet

# ----------------------------------------------------------------- utilities


def das_dennis(n_obj: int, p: int) -> np.ndarray:
    """Structured reference points on the unit simplex (C(p+M-1, M-1) points)."""
    def rec(left, depth):
        if depth == n_obj - 1:
            yield (left,)
            return
        for k in range(left + 1):
            for rest in rec(left - k, depth + 1):
                yield (k,) + rest

    pts = np.array(list(rec(p, 0)), dtype=float) / p
    assert len(pts) == comb(p + n_obj - 1, n_obj - 1)
    return pts


def non_dominated_sort(F: np.ndarray) -> list[np.ndarray]:
    n = len(F)
    if n == 0:
        return []
    le = (F[:, None, :] <= F[None, :, :]).all(-1)
    lt = (F[:, None, :] < F[None, :, :]).any(-1)
    dom = le & lt                       # dom[i, j]: i dominates j
    n_dom = dom.sum(0)                  # how many dominate j
    fronts, current = [], np.nonzero(n_dom == 0)[0]
    while current.size:
        fronts.append(current)
        n_dom = n_dom - dom[current].sum(0)
        n_dom[current] = -1
        current = np.nonzero(n_dom == 0)[0]
    return fronts


def crowding_distance(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        order = np.argsort(F[:, j])
        span = F[order[-1], j] - F[order[0], j]
        d[order[0]] = d[order[-1]] = np.inf
        if span > 0:
            d[order[1:-1]] += (F[order[2:], j] - F[order[:-2], j]) / span
    return d


class MonteCarloHV:
    """Hypervolume of a minimisation set in [0, ref]^M by Monte-Carlo integration.

    A fixed sample set per round (common random numbers) makes successive
    generations directly comparable, which is what the AOS reward needs.
    """

    def __init__(self, n_obj: int, n_samples: int, rng: np.random.Generator, ref: float = 1.1):
        self.ref = ref
        self.samples = rng.uniform(0.0, ref, size=(n_samples, n_obj))
        self.volume = ref ** n_obj

    def __call__(self, F: np.ndarray) -> float:
        if len(F) == 0:
            return 0.0
        dominated = np.zeros(len(self.samples), dtype=bool)
        for f in F:
            dominated |= (self.samples >= f).all(1)
        return float(dominated.mean() * self.volume)


# ----------------------------------------------------------------- survival


def _nsga3_survival(F: np.ndarray, n_select: int, refs: np.ndarray, rng) -> np.ndarray:
    fronts = non_dominated_sort(F)
    chosen: list[int] = []
    k = 0
    while k < len(fronts) and len(chosen) + len(fronts[k]) <= n_select:
        chosen.extend(fronts[k].tolist())
        k += 1
    if len(chosen) == n_select or k == len(fronts):
        return np.array(chosen)
    last = fronts[k]
    St = np.array(chosen + last.tolist())
    Fs = F[St]

    # Normalisation (ideal point, extreme points, hyperplane intercepts).
    ideal = Fs.min(0)
    Ft = Fs - ideal
    m = F.shape[1]
    W = np.eye(m) + 1e-6
    asf = (Ft[:, None, :] / W[None, :, :]).max(-1)          # (|St|, m)
    extremes = Ft[asf.argmin(0)]
    try:
        b = np.linalg.solve(extremes, np.ones(m))
        with np.errstate(divide="ignore", invalid="ignore"):
            intercepts = 1.0 / b
        if np.any(intercepts <= 1e-9) or not np.all(np.isfinite(intercepts)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        intercepts = Ft.max(0)
    intercepts = np.where(intercepts <= 1e-9, 1.0, intercepts)
    Fn = Ft / intercepts

    # Association to reference lines (perpendicular distance).
    rn = refs / np.linalg.norm(refs, axis=1, keepdims=True)
    proj = Fn @ rn.T                                           # (|St|, R)
    dist = np.sqrt(np.maximum((Fn ** 2).sum(1)[:, None] - proj ** 2, 0.0))
    niche = dist.argmin(1)
    ndist = dist[np.arange(len(St)), niche]

    n_chosen = len(chosen)
    counts = np.bincount(niche[:n_chosen], minlength=len(refs))
    pool = list(range(n_chosen, len(St)))                     # positions of last-front members
    selected = []
    excluded_refs = np.zeros(len(refs), dtype=bool)
    while len(selected) < n_select - n_chosen:
        cand_refs = np.nonzero(~excluded_refs)[0]
        cmin = counts[cand_refs].min()
        j = rng.choice(cand_refs[counts[cand_refs] == cmin])
        members = [i for i in pool if niche[i] == j]
        if not members:
            excluded_refs[j] = True
            continue
        if counts[j] == 0:
            pick = members[int(np.argmin(ndist[members]))]
        else:
            pick = members[int(rng.integers(len(members)))]
        selected.append(pick)
        pool.remove(pick)
        counts[j] += 1
    return np.concatenate([np.array(chosen, dtype=int), St[selected]])


def _nsga2_survival(F: np.ndarray, n_select: int, rng=None) -> np.ndarray:
    chosen: list[int] = []
    for front in non_dominated_sort(F):
        if len(chosen) + len(front) <= n_select:
            chosen.extend(front.tolist())
        else:
            cd = crowding_distance(F[front])
            order = np.argsort(-cd)
            chosen.extend(front[order[: n_select - len(chosen)]].tolist())
            break
    return np.array(chosen)


# ---------------------------------------------------------------- operators


class PackingOperators:
    """Feasibility-preserving variation on set packings.

    Uses plain Python lists/tuples internally: solutions contain a few dozen
    small structures, where per-call NumPy overhead would dominate.
    """

    def __init__(self, S: StructureSet, rng: np.random.Generator, fill_attempts: int = 30):
        import random

        self.S = S
        self.rng = rng
        self.py = random.Random(int(rng.integers(2**63)))
        self.fill_attempts = fill_attempts
        self.nodes = [tuple(int(v) for v in n) for n in S.nodes]
        self.vstructs = [v.tolist() for v in S.vertex_structs]
        self.coverable = [v for v in range(S.n_nodes) if self.vstructs[v]]

    def _used(self, sol) -> list[bool]:
        used = [False] * self.S.n_nodes
        for s in sol:
            for v in self.nodes[s]:
                used[v] = True
        return used

    def _free(self, s: int, used: list[bool]) -> bool:
        for v in self.nodes[s]:
            if used[v]:
                return False
        return True

    def _take(self, s: int, used: list[bool]) -> None:
        for v in self.nodes[s]:
            used[v] = True

    def fill(self, sol: list[int], used: list[bool] | None = None) -> list[int]:
        if len(self.S) == 0:
            return sol
        used = self._used(sol) if used is None else used
        free = [v for v in self.coverable if not used[v]]
        self.py.shuffle(free)
        k = self.fill_attempts
        for v in free:
            if used[v]:
                continue
            cands = self.vstructs[v]
            m = len(cands)
            rnd = self.py.random
            # k draws with replacement (cheaper than random.sample, same intent).
            for _ in range(k if m > k else m):
                s = cands[int(rnd() * m)]
                if self._free(s, used):
                    sol.append(s)
                    self._take(s, used)
                    break
        return sol

    def random_solution(self) -> list[int]:
        return self.fill([])

    def crossover(self, a: list[int], b: list[int]) -> list[int]:
        sa, sb = set(a), set(b)
        child = list(sa & sb)
        used = self._used(child)
        rest = list(sa ^ sb)
        self.py.shuffle(rest)
        for s in rest:
            if self.py.random() < 0.5 and self._free(s, used):
                child.append(s)
                self._take(s, used)
        return self.fill(child, used)

    def mutate(self, sol: list[int], pm: float) -> list[int]:
        keep = [s for s in sol if self.py.random() >= pm]
        n_insert = int(self.rng.binomial(max(1, len(sol)), pm))
        for _ in range(n_insert if len(self.S) else 0):
            s = self.py.randrange(len(self.S))
            ns = set(self.nodes[s])
            keep = [t for t in keep if ns.isdisjoint(self.nodes[t])]
            keep.append(s)
        return self.fill(keep)


# ------------------------------------------------------------------- driver


@dataclass
class MOEAResult:
    solutions: list[list[int]]            # final population
    raw: np.ndarray                       # raw objectives (F1..F4) of population
    fmin: np.ndarray                      # normalised minimisation vectors
    front: np.ndarray                     # indices of the first non-dominated front
    chosen: int                           # index of the implemented solution
    hv_trace: list[float] = field(default_factory=list)
    arm_trace: list[int] = field(default_factory=list)

    @property
    def chosen_solution(self) -> list[int]:
        return self.solutions[self.chosen]

    @property
    def final_hv(self) -> float:
        return self.hv_trace[-1] if self.hv_trace else 0.0


def select_from_front(fmin: np.ndarray, raw: np.ndarray, front: np.ndarray, rule: str) -> int:
    Ff = fmin[front]
    if rule == "max_f1":
        return int(front[np.lexsort((Ff[:, 2], -raw[front, 0]))[0]])
    if rule == "min_f3":
        return int(front[np.lexsort((-raw[front, 0], Ff[:, 2]))[0]])
    if rule == "compromise":
        lo, hi = Ff.min(0), Ff.max(0)
        span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        d = np.linalg.norm((Ff - lo) / span, axis=1)
        return int(front[np.argmin(d)])
    raise ValueError(f"unknown selection rule {rule!r}")


class MOEA:
    def __init__(self, cfg: MOEAConfig, selector: OperatorSelector, rng: np.random.Generator):
        self.cfg = cfg
        self.selector = selector
        self.rng = rng
        self.refs = das_dennis(4, cfg.ref_divisions)

    def run(self, evaluator: ObjectiveEvaluator, S: StructureSet,
            seeds: list[list[int]] | None = None) -> MOEAResult:
        cfg, rng = self.cfg, self.rng
        ops = PackingOperators(S, rng, cfg.fill_attempts)
        hv = MonteCarloHV(4, cfg.hv_samples, rng)
        N = cfg.pop_size

        pop = [ops.fill(list(s)) for s in (seeds or [])][: N // 2]
        while len(pop) < N:
            pop.append(ops.random_solution())
        raw = np.array([evaluator.evaluate(s) for s in pop])
        fmin = evaluator.to_min(raw)
        hv_prev = hv(fmin[non_dominated_sort(fmin)[0]])
        hv_trace, arm_trace = [hv_prev], []
        best_gain = 1e-12

        for _ in range(cfg.generations):
            arm = self.selector.select()
            pm = cfg.mutation_rates[arm]
            children = []
            for _ in range(N):
                if cfg.algorithm == "nsga2":
                    pa, pb = self._tournament(fmin), self._tournament(fmin)
                else:
                    pa, pb = rng.integers(N, size=2)
                child = ops.crossover(pop[pa], pop[pb]) if rng.random() < cfg.crossover_prob else list(pop[pa])
                children.append(ops.mutate(child, pm))
            c_raw = np.array([evaluator.evaluate(s) for s in children])
            all_pop = pop + children
            all_raw = np.vstack([raw, c_raw])
            all_min = evaluator.to_min(all_raw)
            keep = self._survive(all_pop, all_min, N)
            pop = [all_pop[i] for i in keep]
            raw, fmin = all_raw[keep], all_min[keep]

            hv_now = hv(fmin[non_dominated_sort(fmin)[0]])
            gain = max(0.0, hv_now - hv_prev)
            best_gain = max(best_gain, gain)
            self.selector.update(arm, gain / best_gain)      # reward normalised to [0, 1]
            hv_prev = hv_now
            hv_trace.append(hv_now)
            arm_trace.append(arm)

        front = non_dominated_sort(fmin)[0]
        chosen = select_from_front(fmin, raw, front, cfg.selection_rule)
        return MOEAResult(pop, raw, fmin, front, chosen, hv_trace, arm_trace)

    # ------------------------------------------------------------ internals
    def _tournament(self, fmin: np.ndarray) -> int:
        i, j = self.rng.integers(len(fmin), size=2)
        if (fmin[i] <= fmin[j]).all() and (fmin[i] < fmin[j]).any():
            return int(i)
        if (fmin[j] <= fmin[i]).all() and (fmin[j] < fmin[i]).any():
            return int(j)
        return int(i if self.rng.random() < 0.5 else j)

    def _survive(self, pop: list[list[int]], F: np.ndarray, N: int) -> np.ndarray:
        # De-duplicate identical packings before environmental selection.
        seen, uniq, dup = set(), [], []
        for i, s in enumerate(pop):
            key = tuple(sorted(s))
            (dup if key in seen else uniq).append(i)
            seen.add(key)
        uniq = np.array(uniq)
        if len(uniq) <= N:
            return np.concatenate([uniq, np.array(dup[: N - len(uniq)], dtype=int)]).astype(int)
        if self.cfg.algorithm == "nsga2":
            sel = _nsga2_survival(F[uniq], N)
        else:
            sel = _nsga3_survival(F[uniq], N, self.refs, self.rng)
        return uniq[sel]
