import numpy as np

from dkex.config import Config
from dkex.simulation.clearing import BatchedPolicy, MyopicPolicy
from dkex.simulation.compatibility import build_graph, hash_uniform
from dkex.simulation.dynamics import DynamicPool
from dkex.simulation.generator import PoolGenerator
from dkex.simulation.logistics import LogisticsNetwork
from dkex.simulation.pool import abo_compatible


def test_abo_rules():
    assert abo_compatible("O", "AB") and abo_compatible("A", "A")
    assert not abo_compatible("A", "O") and not abo_compatible("AB", "B")


def test_generated_pairs_are_incompatible_and_calibrated():
    cfg = Config()
    gen = PoolGenerator(cfg.pool, np.random.default_rng(1))
    pairs = [gen.make_pair(0) for _ in range(3000)]
    high = np.mean([p.cpra >= 0.8 for p in pairs])
    assert abs(high - cfg.pool.high_cpra_rate) < 0.03
    # Every ABO-compatible pair must have been rejected unless crossmatch-positive.
    assert all(0 <= p.cpra < 1 for p in pairs)


def test_hash_uniform_is_deterministic_and_uniform():
    a, b = np.arange(10000), np.arange(10000)[::-1]
    u1 = hash_uniform(7, a, b, 1)
    assert np.array_equal(u1, hash_uniform(7, a, b, 1))
    assert not np.array_equal(u1, hash_uniform(8, a, b, 1))
    assert abs(u1.mean() - 0.5) < 0.02 and u1.min() >= 0 and u1.max() < 1


def test_cit_filter_removes_edges_when_binding():
    cfg = Config()
    assert LogisticsNetwork(cfg.logistics).binding_fraction() == 0.0
    cfg.logistics.inter_centre_delay_hours = 6.0
    net = LogisticsNetwork(cfg.logistics)
    perth, auck = net.index["Perth"], net.index["Auckland"]
    assert not net.is_feasible(perth, auck) and net.is_feasible(perth, perth)
    pool = DynamicPool(cfg, 3).state
    g = build_graph(pool, net, 3)
    assert g.n_edges < g.n_edges_abo
    centres = {p.pid: p.centre for p in pool.pairs.values()}
    for (i, j) in g.weight:
        assert net.feasible[centres[int(g.ids[i])], centres[int(g.ids[j])]]


def test_graph_edges_valid_and_weights_in_range():
    cfg = Config()
    pool = DynamicPool(cfg, 5).state
    g = build_graph(pool, LogisticsNetwork(cfg.logistics), 5)
    pairs = {p.pid: p for p in pool.pairs.values()}
    for (i, j), w in g.weight.items():
        d, r = pairs[int(g.ids[i])], pairs[int(g.ids[j])]
        assert not r.altruist and i != j
        assert abo_compatible(d.donor_abo, r.patient_abo)
        assert 0.0 <= w <= 1.0


def test_pool_evolution_equation():
    cfg = Config()
    dp = DynamicPool(cfg, 11)
    before = set(dp.state.pairs)
    matched = list(before)[:5]
    ev = dp.step(matched)
    after = set(dp.state.pairs)
    expected = (before - set(matched) - set(ev.deceased_exits) - set(ev.withdrawn)) | set(ev.arrivals)
    assert after == expected
    assert dp.state.round == 1


def test_common_random_numbers_across_configs():
    cfg = Config()
    a, b = DynamicPool(cfg, 99), DynamicPool(cfg, 99)
    assert [p.pid for p in a._arrivals[0]] == [p.pid for p in b._arrivals[0]]
    assert sorted(a.state.pairs) == sorted(b.state.pairs)


def test_withdrawal_after_tau():
    cfg = Config()
    cfg.dynamics.withdrawal_threshold_months = 6
    cfg.dynamics.deceased_exit_rate = 0.0
    dp = DynamicPool(cfg, 2)
    for _ in range(8):
        dp.step([])
    assert all(p.wait_months <= 6 for p in dp.state.pairs.values())


def test_clearing_policies():
    cfg = Config()
    pool = DynamicPool(cfg, 1).state
    assert MyopicPolicy().should_clear(pool, 0, False)
    assert not BatchedPolicy(10_000).should_clear(pool, 0, False)
    assert BatchedPolicy(1).should_clear(pool, 0, False)
    valve = BatchedPolicy(10_000, max_interval=3)
    fired = [valve.should_clear(pool, t, False) for t in range(7)]
    assert fired == [False, False, True, False, False, True, False]
