import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lorenz_systems.enkf import EnsembleKalmanFilter

# A simple linear model for testing
def linear_model(x, t_span, t_eval, A):
    # This is a mock solution object to match the expected input for EnKF
    class MockSol:
        def __init__(self, y):
            self.y = y

    # Propagate the state using the matrix A
    y_end = A @ x
    return MockSol(np.array([y_end]).T)

def test_enkf_linear_model():
    """
    Tests the EnKF with a simple linear model.
    """
    dim_x = 2
    N = 10
    A = np.array([[1.1, 0.1], [0.1, 1.1]])
    R = np.eye(dim_x) * 0.1
    H = np.eye(dim_x)

    # Create a mock model function for the EnKF
    model_func = lambda x, t_span, t_eval: linear_model(x, t_span, t_eval, A)

    enkf = EnsembleKalmanFilter(model=model_func, R=R, N=N)

    # Initial ensemble
    X = np.random.randn(dim_x, N)

    # Forecast step
    X_f = enkf.forecast(X, t_span=(0, 1), t_eval=np.array([0, 1]))
    assert X_f.shape == (dim_x, N)

    # Analysis step
    y = np.array([2.0, 2.0])
    X_a = enkf.analysis(X_f, y, H)
    assert X_a.shape == (dim_x, N)

    # Check if the analysis mean is closer to the observation than the forecast mean
    mean_f = np.mean(X_f, axis=1)
    mean_a = np.mean(X_a, axis=1)
    dist_f = np.linalg.norm(mean_f - y)
    dist_a = np.linalg.norm(mean_a - y)
    assert dist_a < dist_f
