import numpy as np
from n_body.config import Config
from n_body.initialization import set_states
from n_body.simulation import Simulation


def test_reproducibility():
    cfg = Config(dt=0.01, tf=0.1, seed=123)
    s1 = set_states(init_conds=5, mass_equal=False, init_vel=True, max_pts=6, seed=cfg.seed, scale=10.0)
    s2 = set_states(init_conds=5, mass_equal=False, init_vel=True, max_pts=6, seed=cfg.seed, scale=10.0)
    sim1 = Simulation(cfg, s1)
    sim2 = Simulation(cfg, s2)
    r1 = sim1.run()
    r2 = sim2.run()
    assert np.allclose(r1.state_history, r2.state_history)


def test_momentum_conservation():
    # Initialize simple symmetric system to ensure COM velocity ~ 0 over steps
    cfg = Config(dt=0.01, tf=0.1, seed=42)
    s = set_states(init_conds=4, mass_equal=True, init_vel=False, max_pts=6, seed=cfg.seed, scale=1.0)
    sim = Simulation(cfg, s)
    r = sim.run()
    masses = s.mass
    # momentum at each step = sum(m_i * v_i)
    momentum = (r.state_history[:, :, 3:6] * masses.reshape(1, -1, 1)).sum(axis=1)
    # Should be near zero for all timesteps
    assert np.allclose(momentum, 0.0, atol=1e-6)
