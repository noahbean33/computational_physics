import numpy as np
from scipy.integrate import solve_ivp

def lorenz96(t, x, F):
    """
    Computes the derivatives of the Lorenz 96 system.

    Args:
        t (float): Time (not used, but required for solve_ivp).
        x (np.ndarray): State vector of the system.
        F (float): Forcing constant.

    Returns:
        np.ndarray: Array of derivatives.
    """
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return dxdt

def solve_lorenz96(x0, t_span, t_eval, F=8.0):
    """
    Solves the Lorenz 96 system for a given initial condition and time span.

    Args:
        x0 (np.ndarray): Initial state vector.
        t_span (tuple): Time interval for integration (t_start, t_end).
        t_eval (np.ndarray): Time points at which to store the solution.
        F (float, optional): Forcing constant. Defaults to 8.0.

    Returns:
        OdeResult: An object containing the solution.
    """
    sol = solve_ivp(
        fun=lorenz96,
        t_span=t_span,
        y0=x0,
        t_eval=t_eval,
        args=(F,),
        dense_output=True
    )
    return sol