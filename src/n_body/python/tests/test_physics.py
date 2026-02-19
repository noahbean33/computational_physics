import numpy as np
from n_body.config import Config, State
from n_body.physics import accelerations, energy


def test_action_reaction_symmetry():
    # Two bodies along x-axis with equal masses
    mass = np.array([[50.0], [50.0]], dtype=np.float64)
    pos = np.array([[ -1.0, 0.0, 0.0],
                    [  1.0, 0.0, 0.0]], dtype=np.float64)
    vel = np.zeros((2,3), dtype=np.float64)
    state = State(mass=mass, pos=pos, vel=vel)

    acc = accelerations(state, G=1.0, S=0.1)
    # Net acceleration weighted by mass should be ~0 (Newton's third law)
    net = (acc * mass).sum(axis=0)
    assert np.allclose(net, np.zeros(3), atol=1e-10)


def test_energy_finite():
    mass = np.array([[60.0], [40.0]], dtype=np.float64)
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    vel = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)
    state = State(mass=mass, pos=pos, vel=vel)
    KE, PE = energy(state, G=1.0)
    assert np.isfinite(KE) and np.isfinite(PE)
