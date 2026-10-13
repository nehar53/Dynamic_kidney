

from dkex import pipeline
from dkex.data.preflib import load_wmd
from dkex.experiments import BASELINES, simulate


def test_every_baseline_runs(small_cfg):
    for name in BASELINES:
        rows, summary = simulate(small_cfg, name, 12.0, 0)
        assert len(rows) == small_cfg.dynamics.rounds
        for r in rows:
            if r["cleared"]:
                assert r["F1"] <= r["F1_ilp"] + 1e-9
                if name == "myopic_ilp":
                    assert r["PoF"] == 0.0


def test_paired_runs_share_initial_pool(small_cfg):
    a, _ = simulate(small_cfg, "myopic_ilp", 12.0, 0)
    b, _ = simulate(small_cfg, "dickerson_lex", 12.0, 0)
    assert a[0]["pool_size"] == b[0]["pool_size"] and a[0]["F1_ilp"] == b[0]["F1_ilp"]


def test_stages_end_to_end(small_cfg, tmp_path):
    small_cfg.experiment.output_dir = str(tmp_path)
    small_cfg.experiment.taus = (12.0,)
    small_cfg.experiment.baselines = ("myopic_ilp", "dyn_nsga3_ucb1_myopic", "dyn_nsga3_linucb_myopic")
    rep = pipeline.calibrate(small_cfg, n_samples=1)
    assert rep["warnings"]  # default logistics never binds -> warning expected
    rounds, runs = pipeline.simulate_all(small_cfg, progress=False)
    assert len(runs) == 3 * small_cfg.experiment.runs
    out = pipeline.analyse(small_cfg)
    assert not out["trajectories"].empty and not out["wilcoxon"].empty
    assert pipeline.plot(small_cfg)
    # Resumable: a second call runs nothing new.
    r2, _ = pipeline.simulate_all(small_cfg, progress=False)
    assert len(r2) == len(rounds)


def test_preflib_modern_and_legacy(tmp_path, small_cfg):
    modern = tmp_path / "m.wmd"
    lines = ["# FILE NAME: m.wmd", "# NUMBER ALTERNATIVES: 4", "# NUMBER EDGES: 4",
             "# ALTERNATIVE NAME 1: Pair 1", "# ALTERNATIVE NAME 2: Pair 2",
             "# ALTERNATIVE NAME 3: Pair 3", "# ALTERNATIVE NAME 4: Altruist 1",
             "1,2,1.0", "2,1,1.0", "4,3,1.0", "3,1,1.0"]
    modern.write_text("\n".join(lines))
    g = load_wmd(modern)
    assert g.n == 4 and g.altruists == {4} and (1, 2) in g.edges and len(g.edges) == 4

    legacy = tmp_path / "l.wmd"
    legacy.write_text("3\n1,Pair 1\n2,Pair 2\n3,Pair 3\n3,2\n1,2,1\n2,1,1\n")
    g2 = load_wmd(legacy)
    assert g2.n == 3 and g2.edges == {(1, 2), (2, 1)}

    small_cfg.pool.initial_pool_size = 4
    small_cfg.pool.arrival_rate = 0
    rows, _ = simulate(small_cfg, "myopic_ilp", 12.0, 0, preflib_graph=g)
    assert rows[0]["F1_ilp"] >= 2   # the 1<->2 two-cycle (or chain 4->3->1) is available
