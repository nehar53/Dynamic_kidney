"""Module 11 — Experiment runner and the eight baselines (Section 4.4).

One *trajectory* = one (baseline, tau, run) triple simulated for T rounds:

    for t in 0..T-1:
        if clearing policy says clear:
            G_t   <- build_graph(V_t)                       (3.2, 3.3)
            S_t   <- enumerate cycles/chains                 (K, L)
            ILP   <- max-volume ceiling on S_t               (F1^ILP(t))
            M_t   <- baseline's matching on S_t              (MOEA / ILP / lex)
            PoF_t <- Eq. 13
        V_{t+1} <- V_t \\ M_t \\ W_t \\ X_t U A_t            (3.1)

All baselines in the same run share the same seed, hence the same initial
pool, arrival stream, crossmatch/HLA draws and competing-exit draws, so the
comparison is paired (needed for Wilcoxon signed-rank tests).
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from dkex.aos import build_context, make_selector
from dkex.analysis.pof import price_of_fairness, trajectory_summary
from dkex.config import CENTRES, Config
from dkex.optimisation.ilp import solve_lexicographic, solve_max_volume
from dkex.optimisation.moea import MOEA
from dkex.optimisation.objectives import ObjectiveEvaluator, gini
from dkex.optimisation.structures import enumerate_structures
from dkex.simulation.clearing import make_policy
from dkex.simulation.compatibility import build_graph
from dkex.simulation.dynamics import DynamicPool
from dkex.simulation.logistics import LogisticsNetwork


@dataclass(frozen=True)
class Baseline:
    name: str
    solver: str                 # "moea" | "ilp" | "lex"
    algorithm: str = "nsga3"
    aos: str = "ucb1"
    clearing: str = "myopic"
    dynamic: bool = True        # persist AOS state + warm start across rounds


BASELINES: dict[str, Baseline] = {
    b.name: b
    for b in [
        Baseline("static_nsga2_ucb1", "moea", "nsga2", "ucb1", "myopic", dynamic=False),
        Baseline("static_nsga3_fixed", "moea", "nsga3", "fixed", "myopic", dynamic=False),
        Baseline("dyn_nsga3_ucb1_myopic", "moea", "nsga3", "ucb1", "myopic"),
        Baseline("dyn_nsga3_linucb_myopic", "moea", "nsga3", "linucb", "myopic"),
        Baseline("dyn_nsga3_ucb1_batched", "moea", "nsga3", "ucb1", "batched"),
        Baseline("dyn_nsga3_linucb_batched", "moea", "nsga3", "linucb", "batched"),
        Baseline("myopic_ilp", "ilp", clearing="myopic"),
        Baseline("dickerson_lex", "lex", clearing="myopic"),
    ]
}


def run_seed(base_seed: int, run: int) -> int:
    return int(base_seed) * 1_000_003 + int(run)


def _stable_hash(text: str) -> int:
    return zlib.crc32(text.encode())


def simulate(cfg: Config, baseline: Baseline | str, tau: float, run: int,
             verbose: bool = False, preflib_graph=None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Simulate one trajectory; return (per-round rows, per-run summary).

    ``preflib_graph`` (a ``dkex.data.preflib.PrefLibGraph``) switches to the
    out-of-distribution setting of Section 4.6.
    """
    b = BASELINES[baseline] if isinstance(baseline, str) else baseline
    cfg = cfg.copy()
    cfg.dynamics.withdrawal_threshold_months = float(tau)
    cfg.clearing.policy = b.clearing
    cfg.moea.algorithm = b.algorithm

    seed = run_seed(cfg.experiment.seed, run)
    logistics = LogisticsNetwork(cfg.logistics)
    source, external_edges = None, None
    if preflib_graph is not None:
        from dkex.data.preflib import PrefLibSource

        source = PrefLibSource(preflib_graph, cfg.pool, np.random.default_rng(seed))
        external_edges = preflib_graph.edges
    pool = DynamicPool(cfg, seed, source)
    policy = make_policy(cfg.clearing)
    T = cfg.dynamics.rounds
    thr = cfg.pool.high_cpra_threshold
    rates = cfg.moea.mutation_rates

    algo_rng = np.random.default_rng([seed, _stable_hash(b.name)])
    fixed_arm = int(np.argmin([abs(r - cfg.moea.fixed_mutation_rate) for r in rates]))
    selector = make_selector(b.aos, len(rates), window=cfg.aos.ucb_window,
                             alpha=cfg.aos.linucb_alpha, fixed_arm=fixed_arm,
                             seed=int(algo_rng.integers(2**31)))
    moea = MOEA(cfg.moea, selector, algo_rng)

    rows: list[dict[str, Any]] = []
    prev_pof, prev_exits = 0.0, 0
    warm: list[tuple[tuple[int, ...], ...]] = []   # previous population, as pair-id tuples
    high_in_pool: list[int] = []

    for t in range(T):
        state = pool.state
        pts = state.patients
        n_high = sum(p.cpra >= thr for p in pts)
        high_in_pool.append(n_high)
        row: dict[str, Any] = {
            "config": b.name, "tau": tau, "run": run, "round": t,
            "pool_size": len(state), "n_patients": len(pts), "n_altruists": len(state.altruists),
            "n_high_cpra": n_high, "high_cpra_share": n_high / max(1, len(pts)),
            "mean_wait": float(np.mean([p.wait_months for p in pts])) if pts else 0.0,
            "pool_wait_gini": gini(np.array([p.wait_months for p in pts]) + 1e-9) if pts else 0.0,
            "cleared": False,
        }
        matched_pids: list[int] = []

        if policy.should_clear(state, t, last_round=(t == T - 1)) and pts:
            tic = time.perf_counter()
            g = build_graph(state, logistics, seed, cfg.pool.hla_mismatch_p, external_edges)
            S = enumerate_structures(g, cfg.structures, np.random.default_rng([seed, t]))
            ev = ObjectiveEvaluator(state, g, S, CENTRES, thr, cfg.moea.f3_mode)
            ilp_sol = solve_max_volume(S)
            f1_ilp = ev.evaluate(ilp_sol)[0]
            hv, arm_counts = np.nan, None

            if b.solver == "ilp":
                chosen = list(ilp_sol)
            elif b.solver == "lex":
                chosen = list(solve_lexicographic(S, ev.high))
            else:
                if not b.dynamic:
                    selector.reset()
                selector.set_context(build_context(
                    len(state), state.high_cpra_share(thr), prev_exits, t, T,
                    row["pool_wait_gini"], prev_pof,
                    pool_scale=2.0 * cfg.clearing.batch_threshold,
                ))
                seeds = _map_warm_start(warm, g.ids, S) if (b.dynamic and cfg.moea.warm_start) else None
                res = moea.run(ev, S, seeds)
                chosen = res.chosen_solution
                hv = res.final_hv
                arm_counts = np.bincount(res.arm_trace, minlength=len(rates))
                warm = [tuple(tuple(int(g.ids[v]) for v in S.nodes[s]) for s in sol)
                        for sol in (res.solutions[i] for i in res.front)]

            raw = ev.evaluate(chosen)
            pof = price_of_fairness(f1_ilp, raw[0])
            matched_pids = sorted({int(g.ids[v]) for s in chosen for v in S.nodes[s]})
            row.update({
                "cleared": True, "n_edges": g.n_edges, "n_edges_pre_cit": g.n_edges_abo,
                "density": g.density(), "n_structures": len(S), "chains_truncated": S.truncated,
                "F1_ilp": f1_ilp, "F1": raw[0], "F2": raw[1], "F3": raw[2], "F4": raw[3],
                "PoF": pof, "HV": hv,
                "high_cpra_matched": int((ev.matched_mask(chosen) & ev.high).sum()),
                "solve_seconds": time.perf_counter() - tic,
            })
            if arm_counts is not None:
                for r, c in zip(rates, arm_counts):
                    row[f"arm_{r}"] = int(c)
            if b.solver == "moea":
                prev_pof = pof

        events = pool.step(matched_pids)
        prev_exits = len(events.deceased_exits)
        row.update({
            "matched": len(matched_pids), "deceased_exits": len(events.deceased_exits),
            "withdrawn": len(events.withdrawn), "withdrawn_high_cpra": events.withdrawn_high_cpra,
            "deceased_high_cpra": events.deceased_high_cpra, "arrivals": len(events.arrivals),
        })
        rows.append(row)
        if verbose:
            print(f"[{b.name} tau={tau} run={run}] t={t:2d} n={row['pool_size']:3d} "
                  f"cleared={row['cleared']!s:5s} F1={row.get('F1', 0):.0f}/{row.get('F1_ilp', 0):.0f} "
                  f"PoF={row.get('PoF', float('nan')):.1f}")

    summary = summarise_run(rows, high_in_pool)
    summary.update({"config": b.name, "tau": tau, "run": run})
    return rows, summary


def _map_warm_start(warm, ids: np.ndarray, S) -> list[list[int]]:
    """Translate last round's front (pair-id tuples) into this round's structure indices."""
    if not warm:
        return []
    pos = {int(pid): k for k, pid in enumerate(ids)}
    index = {tuple(n): k for k, n in enumerate(S.nodes)}
    seeds = []
    for sol in warm:
        mapped = []
        for struct in sol:
            if all(pid in pos for pid in struct):
                k = index.get(tuple(pos[pid] for pid in struct))
                if k is not None:
                    mapped.append(k)
        seeds.append(mapped)
    return seeds


def summarise_run(rows: list[dict[str, Any]], high_in_pool: list[int]) -> dict[str, Any]:
    cleared = [r for r in rows if r["cleared"]]
    pof = [r["PoF"] for r in cleared]
    out = {
        "rounds_cleared": len(cleared),
        "total_transplants": int(sum(r["F1"] for r in cleared)),
        "total_high_cpra_transplants": int(sum(r.get("high_cpra_matched", 0) for r in cleared)),
        "total_withdrawn": int(sum(r["withdrawn"] for r in rows)),
        "total_withdrawn_high_cpra": int(sum(r["withdrawn_high_cpra"] for r in rows)),
        "total_deceased_exits": int(sum(r["deceased_exits"] for r in rows)),
        "mean_F2": float(np.mean([r["F2"] for r in cleared])) if cleared else np.nan,
        "mean_F3": float(np.mean([r["F3"] for r in cleared])) if cleared else np.nan,
        "mean_F4": float(np.mean([r["F4"] for r in cleared])) if cleared else np.nan,
        "mean_HV": float(np.nanmean([r["HV"] for r in cleared])) if cleared and any(np.isfinite(r["HV"]) for r in cleared) else np.nan,
        # High-cPRA accumulation rate: OLS slope of #high-cPRA patients in pool per round.
        "high_cpra_accumulation": float(np.polyfit(np.arange(len(high_in_pool)), high_in_pool, 1)[0])
        if len(high_in_pool) > 1 else 0.0,
    }
    out.update(trajectory_summary(pof))
    return out
