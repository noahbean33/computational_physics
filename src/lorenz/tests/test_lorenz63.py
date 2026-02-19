import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lorenz_systems.lorenz63 import lorenz63, solve_lorenz63

def test_lorenz63_derivatives():
    """
    Tests the Lorenz 63 derivative function with known values.
    """
    xyz = np.array([0, 1, 20])
    sigma, rho, beta = 10, 28, 8/3
    derivatives = lorenz63(0, xyz, sigma, rho, beta)
    expected = np.array([10, -1, -160/3])
    assert np.allclose(derivatives, expected)

def test_solve_lorenz63():
    """
    Tests the Lorenz 63 solver to ensure it runs and returns expected shapes.
    """
    xyz0 = np.array([0, 1, 1.05])
    t_span = (0, 1)
    t_eval = np.linspace(t_span[0], t_span[1], 10)
    sol = solve_lorenz63(xyz0, t_span, t_eval)
    assert sol.y.shape == (3, len(t_eval))
    assert sol.t.shape == (len(t_eval),)