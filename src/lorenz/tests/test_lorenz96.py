import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lorenz_systems.lorenz96 import lorenz96, solve_lorenz96

def test_lorenz96_derivatives():
    """
    Tests the Lorenz 96 derivative function with a simple case.
    """
    x = np.array([1, 2, 3, 4, 5])
    F = 8.0
    derivatives = lorenz96(0, x, F)
    expected = np.array([ (x[1] - x[-2]) * x[-1] - x[0] + F, # (2 - 4) * 5 - 1 + 8 = -10 - 1 + 8 = -3
                           (x[2] - x[-1]) * x[0] - x[1] + F, # (3 - 5) * 1 - 2 + 8 = -2 - 2 + 8 = 4
                           (x[3] - x[0]) * x[1] - x[2] + F,  # (4 - 1) * 2 - 3 + 8 = 6 - 3 + 8 = 11
                           (x[4] - x[1]) * x[2] - x[3] + F,  # (5 - 2) * 3 - 4 + 8 = 9 - 4 + 8 = 13
                           (x[0] - x[2]) * x[3] - x[4] + F])# (1 - 3) * 4 - 5 + 8 = -8 - 5 + 8 = -5
    assert np.allclose(derivatives, expected)

def test_solve_lorenz96():
    """
    Tests the Lorenz 96 solver to ensure it runs and returns expected shapes.
    """
    N = 5
    x0 = np.random.rand(N)
    t_span = (0, 1)
    t_eval = np.linspace(t_span[0], t_span[1], 10)
    sol = solve_lorenz96(x0, t_span, t_eval)
    assert sol.y.shape == (N, len(t_eval))
    assert sol.t.shape == (len(t_eval),)