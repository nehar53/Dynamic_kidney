import pytest

from dkex.config import Config


@pytest.fixture
def small_cfg():
    cfg = Config()
    cfg.dynamics.rounds = 3
    cfg.pool.initial_pool_size = 40
    cfg.moea.pop_size = 12
    cfg.moea.generations = 4
    cfg.moea.hv_samples = 500
    cfg.structures.max_chains = 2000
    cfg.experiment.runs = 2
    return cfg
