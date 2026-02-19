import numpy as np
from scipy.integrate import solve_ivp

def lorenz63(t, xyz, sigma, rho, beta):
    """
    Computes the derivatives of the Lorenz 63 system.

    Args:
        t (float): Time (not used, but required for solve_ivp).
        xyz (array-like): State vector [x, y, z].
        sigma (float): Lorenz parameter sigma.
        rho (float): Lorenz parameter rho.
        beta (float): Lorenz parameter beta.

    Returns:
        np.ndarray: Array of derivatives [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

def solve_lorenz63(xyz0, t_span, t_eval, sigma=10.0, rho=28.0, beta=8/3):
    """
    Solves the Lorenz 63 system for a given initial condition and time span.

    Args:
        xyz0 (array-like): Initial state vector [x0, y0, z0].
        t_span (tuple): Time interval for integration (t_start, t_end).
        t_eval (array-like): Time points at which to store the solution.
        sigma (float, optional): Lorenz parameter sigma. Defaults to 10.0.
        rho (float, optional): Lorenz parameter rho. Defaults to 28.0.
        beta (float, optional): Lorenz parameter beta. Defaults to 8/3.

    Returns:
        OdeResult: An object containing the solution.
    """
    sol = solve_ivp(
        fun=lorenz63,
        t_span=t_span,
        y0=xyz0,
        t_eval=t_eval,
        args=(sigma, rho, beta),
        dense_output=True
    )
    return sol