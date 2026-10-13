import numpy as np
import pytest

from dkex.aos import LinUCB, SlidingWindowUCB1, build_context
from dkex.analysis.pof import price_of_fairness
from dkex.config import CENTRES, Config
from dkex.optimisation.ilp import solve_lexicographic, solve_max_volume
from dkex.optimisation.moea import (MOEA, MonteCarloHV, PackingOperators, das_dennis,
                                    non_dominated_sort)
from dkex.optimisation.objectives import ObjectiveEvaluator, gini
from dkex.optimisation.structures import enumerate_structures
from dkex.aos.base import FixedOperator
from dkex.simulation.compatibility import build_graph
from dkex.simulation.dynamics import DynamicPool
from dkex.simulation.logistics import LogisticsNetwork


@pytest.fixture
def instance(small_cfg):
    pool = DynamicPool(small_cfg, 21).state
    g = build_graph(pool, LogisticsNetwork(small_cfg.logistics), 21)
    S = enumerate_structures(g, small_cfg.structures, np.random.default_rng(0))
    ev = ObjectiveEvaluator(pool, g, S, CENTRES)
    return small_cfg, pool, g, S, ev


def _is_packing(S, sol):
    seen = set()
    for s in sol:
        if seen & set(S.nodes[s]):
            return False
        seen |= set(S.nodes[s])
    return True


def test_structures_respect_limits(instance):
    cfg, _, g, S, _ = instance
    for nodes, rec, chain in zip(S.nodes, S.recipients, S.is_chain):
        if chain:
            assert g.altruist[nodes[0]] and 1 <= len(rec) <= cfg.structures.max_chain_length
            assert all((a, b) in g.weight for a, b in zip(nodes, nodes[1:]))
        else:
            assert 2 <= len(nodes) <= cfg.structures.max_cycle_length
            assert all((a, b) in g.weight for a, b in zip(nodes, nodes[1:] + nodes[:1]))


def test_ilp_is_packing_and_dominates_lexicographic(instance):
    _, _, _, S, ev = instance
    best = solve_max_volume(S)
    lex = solve_lexicographic(S, ev.high)
    assert _is_packing(S, best) and _is_packing(S, lex)
    assert ev.evaluate(best)[0] >= ev.evaluate(lex)[0]
    # Lexicographic serves at least as many high-cPRA patients.
    assert (ev.matched_mask(lex) & ev.high).sum() >= (ev.matched_mask(best) & ev.high).sum()


def test_operators_keep_feasibility(instance):
    _, _, _, S, _ = instance
    ops = PackingOperators(S, np.random.default_rng(3))
    a, b = ops.random_solution(), ops.random_solution()
    for _ in range(50):
        c = ops.mutate(ops.crossover(a, b), 0.25)
        assert _is_packing(S, c)
        a, b = b, c


def test_moea_bounded_by_ilp(instance):
    cfg, _, _, S, ev = instance
    res = MOEA(cfg.moea, FixedOperator(4, 2), np.random.default_rng(0)).run(ev, S)
    f1_ilp = ev.evaluate(solve_max_volume(S))[0]
    assert all(_is_packing(S, s) for s in res.solutions)
    assert res.raw[:, 0].max() <= f1_ilp + 1e-9
    assert 0 <= price_of_fairness(f1_ilp, res.raw[res.chosen, 0]) <= 100
    assert len(res.arm_trace) == cfg.moea.generations


def test_gini_matches_definition():
    x = np.array([1.0, 2.0, 3.0, 10.0])
    brute = np.abs(x[:, None] - x[None, :]).sum() / (2 * len(x) * x.sum())
    assert gini(x) == pytest.approx(brute)
    assert gini(np.ones(5)) == pytest.approx(0.0)


def test_pof():
    assert price_of_fairness(40, 36) == pytest.approx(10.0)
    assert price_of_fairness(0, 0) == 0.0


def test_non_dominated_sort_and_refs():
    F = np.array([[0, 1], [1, 0], [1, 1], [2, 2]], float)
    fronts = non_dominated_sort(F)
    assert set(fronts[0]) == {0, 1} and set(fronts[1]) == {2} and set(fronts[2]) == {3}
    refs = das_dennis(4, 12)
    assert refs.shape == (455, 4) and np.allclose(refs.sum(1), 1)


def test_hypervolume_monotone():
    hv = MonteCarloHV(2, 20000, np.random.default_rng(0), ref=1.0)
    assert hv(np.array([[0.5, 0.5]])) == pytest.approx(0.25, abs=0.02)
    assert hv(np.array([[0.5, 0.5], [0.2, 0.8]])) > hv(np.array([[0.5, 0.5]]))


def test_ucb1_prefers_best_arm():
    b = SlidingWindowUCB1(4, window=15, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(300):
        a = b.select()
        b.update(a, float(rng.random() < (0.9 if a == 2 else 0.1)))
    assert np.argmax(b.pulls) == 2


def test_linucb_uses_context():
    b = LinUCB(2, dim=6, alpha=0.2, seed=0)
    ctx = [np.eye(6)[0], np.eye(6)[1]]
    rng = np.random.default_rng(1)
    for _ in range(400):
        k = rng.integers(2)
        b.set_context(ctx[k])
        a = b.select()
        b.update(a, 1.0 if a == k else 0.0)   # best arm depends on context
    for k in range(2):
        b.set_context(ctx[k])
        assert b.select() == k


def test_context_normalised():
    s = build_context(500, 0.3, 10, 5, 25, 0.4, 250.0)
    assert s.shape == (6,) and s.min() >= 0 and s.max() <= 1
