"""
ODE solving tutorials (scalar exponential decay and Lorenz system) with CLI.

Refactored from a notebook export into a reusable, import-safe script.
"""

import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from src.visualization.plot_utils import maybe_save_or_show

# --- 1. Define the Ordinary Differential Equation (ODE) ---
# We'll use a simple first-order ODE for demonstration purposes:
# dy/dt = -y
# The analytical solution is y(t) = y0 * exp(-t)
# This allows us to easily check the accuracy of our numerical methods.

def model(t, y):
    """
    Defines the ODE dy/dt = f(t, y).
    Args:
        t (float): The current time.
        y (float): The current value of the dependent variable.
    Returns:
        float: The derivative of y with respect to t.
    """
    return -y

def _maybe_save_or_show(path: str | None):
    # Backward-compatible wrapper for older calls in this file
    maybe_save_or_show(path)

# --- 3. Implementation of Numerical Methods ---

def euler_method(f, y0, t_points):
    """
    Solves a first-order ODE using Euler's method.

    Formula: y_{n+1} = y_n + h * f(t_n, y_n)

    Args:
        f (callable): The function defining the ODE, f(t, y).
        y0 (list or np.array): The initial condition for y.
        t_points (np.array): The array of time points to solve for.
    Returns:
        np.array: The solution y(t) at each time point.
    """
    print("Running Euler's Method...")
    y = np.zeros(len(t_points))
    y[0] = y0[0]
    h = t_points[1] - t_points[0] # Calculate step size
    for i in range(len(t_points) - 1):
        y[i+1] = y[i] + h * f(t_points[i], y[i])
    return y

def runge_kutta_2(f, y0, t_points):
    """
    Solves a first-order ODE using the 2nd-order Runge-Kutta method (Midpoint Method).

    Formula:
    k1 = h * f(t_n, y_n)
    k2 = h * f(t_n + h/2, y_n + k1/2)
    y_{n+1} = y_n + k2

    Args:
        f (callable): The function defining the ODE, f(t, y).
        y0 (list or np.array): The initial condition for y.
        t_points (np.array): The array of time points to solve for.
    Returns:
        np.array: The solution y(t) at each time point.
    """
    print("Running 2nd-Order Runge-Kutta (Midpoint Method)...")
    y = np.zeros(len(t_points))
    y[0] = y0[0]
    h = t_points[1] - t_points[0]
    for i in range(len(t_points) - 1):
        k1 = h * f(t_points[i], y[i])
        k2 = h * f(t_points[i] + h/2, y[i] + k1/2)
        y[i+1] = y[i] + k2
    return y

def runge_kutta_4(f, y0, t_points):
    """
    Solves a first-order ODE using the classic 4th-order Runge-Kutta method.

    Formula:
    k1 = h * f(t_n, y_n)
    k2 = h * f(t_n + h/2, y_n + k1/2)
    k3 = h * f(t_n + h/2, y_n + k2/2)
    k4 = h * f(t_n + h, y_n + k3)
    y_{n+1} = y_n + (k1 + 2*k2 + 2*k3 + k4) / 6

    Args:
        f (callable): The function defining the ODE, f(t, y).
        y0 (list or np.array): The initial condition for y.
        t_points (np.array): The array of time points to solve for.
    Returns:
        np.array: The solution y(t) at each time point.
    """
    print("Running 4th-Order Runge-Kutta...")
    y = np.zeros(len(t_points))
    y[0] = y0[0]
    h = t_points[1] - t_points[0]
    for i in range(len(t_points) - 1):
        k1 = h * f(t_points[i], y[i])
        k2 = h * f(t_points[i] + h/2, y[i] + k1/2)
        k3 = h * f(t_points[i] + h/2, y[i] + k2/2)
        k4 = h * f(t_points[i] + h, y[i] + k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return y

def demo_simple_decay(y0=1.0, t0=0.0, t1=10.0, dt=1.0, save_path: str | None = None):
    """Run scalar decay dy/dt=-y with multiple methods and plot."""
    t_points = np.arange(t0, t1 + dt, dt)
    y0_vec = [y0]

    y_euler = euler_method(model, y0_vec, t_points)
    y_rk2   = runge_kutta_2(model, y0_vec, t_points)
    y_rk4   = runge_kutta_4(model, y0_vec, t_points)

    sol_scipy = solve_ivp(model, (t0, t1), y0_vec, dense_output=True, t_eval=t_points)
    y_analytical = y0 * np.exp(-t_points)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(t_points, y_analytical, 'k-', label='Analytical Solution', linewidth=4, alpha=0.7)
    ax.plot(t_points, y_euler, 'o--', label="Euler's Method", markersize=8)
    ax.plot(t_points, y_rk2, 's--', label='RK2 (Midpoint)', markersize=8)
    ax.plot(t_points, y_rk4, '^-', label='RK4', markersize=8)
    ax.plot(sol_scipy.t, sol_scipy.y[0], 'x-', label='SciPy solve_ivp', markersize=8, color='purple')
    ax.set_title(f'Comparison of ODE Solvers for dy/dt = -y (h={dt})', fontsize=16)
    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel('y(t)', fontsize=12)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    _maybe_save_or_show(save_path)
    return dict(t=t_points, y_euler=y_euler, y_rk2=y_rk2, y_rk4=y_rk4, y_ref=sol_scipy.y[0])

## Imports consolidated above

# --- 1. Define the Ordinary Differential Equation (ODE) System ---
# We will use the Lorenz system, a classic example of a chaotic system.
# dx/dt = sigma * (y - x)
# dy/dt = x * (rho - z) - y
# dz/dt = x * y - beta * z

def lorenz_system(t, y, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    Defines the Lorenz system of ODEs.
    Args:
        t (float): The current time (not used in this autonomous system, but required by solvers).
        y (list or np.array): A list or array [x, y, z] of the current state.
        sigma, rho, beta (float): Parameters of the Lorenz system.
    Returns:
        np.array: The derivatives [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = y
    dydt = [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]
    return np.array(dydt)

def _lorenz_timepoints(t0: float, t1: float, dt: float):
    return np.arange(t0, t1 + dt, dt)

# --- 3. Update Numerical Methods to Handle Systems (Vectors) ---
# The logic is identical, but now we use NumPy for vector arithmetic.

def euler_method_system(f, y0, t_points):
    """Solves a system of ODEs using Euler's method."""
    y = np.zeros((len(t_points), len(y0)))
    y[0, :] = y0
    h = t_points[1] - t_points[0]
    for i in range(len(t_points) - 1):
        y[i+1, :] = y[i, :] + h * f(t_points[i], y[i, :])
    return y

def runge_kutta_2_system(f, y0, t_points):
    """Solves a system of ODEs using the 2nd-order Runge-Kutta method."""
    y = np.zeros((len(t_points), len(y0)))
    y[0, :] = y0
    h = t_points[1] - t_points[0]
    for i in range(len(t_points) - 1):
        k1 = h * f(t_points[i], y[i, :])
        k2 = h * f(t_points[i] + h/2, y[i, :] + k1/2)
        y[i+1, :] = y[i, :] + k2
    return y

def runge_kutta_4_system(f, y0, t_points):
    """Solves a system of ODEs using the 4th-order Runge-Kutta method."""
    y = np.zeros((len(t_points), len(y0)))
    y[0, :] = y0
    h = t_points[1] - t_points[0]
    for i in range(len(t_points) - 1):
        k1 = h * f(t_points[i], y[i, :])
        k2 = h * f(t_points[i] + h/2, y[i, :] + k1/2)
        k3 = h * f(t_points[i] + h/2, y[i, :] + k2/2)
        k4 = h * f(t_points[i] + h, y[i, :] + k3)
        y[i+1, :] = y[i, :] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return y

def demo_lorenz_benchmark(t0=0.0, t1=25.0, dt=0.01, y0=(1.0, 1.0, 1.0), save_prefix: str | None = None):
    """Benchmark Euler, RK2, RK4 against high-accuracy Lorenz solution and plot."""
    t_points = _lorenz_timepoints(t0, t1, dt)
    y0 = list(y0)

    ref_sol = solve_ivp(lorenz_system, (t0, t1), y0, dense_output=True, t_eval=t_points, rtol=1e-8, atol=1e-8)
    y_ref = ref_sol.y.T

    results = {}

    start = time.time(); y_euler = euler_method_system(lorenz_system, y0, t_points); te = time.time()-start
    results['Euler'] = {'time': te, 'mse': np.mean((y_euler - y_ref)**2)}

    start = time.time(); y_rk2 = runge_kutta_2_system(lorenz_system, y0, t_points); tr2 = time.time()-start
    results['RK2'] = {'time': tr2, 'mse': np.mean((y_rk2 - y_ref)**2)}

    start = time.time(); y_rk4 = runge_kutta_4_system(lorenz_system, y0, t_points); tr4 = time.time()-start
    results['RK4'] = {'time': tr4, 'mse': np.mean((y_rk4 - y_ref)**2)}

    start = time.time(); solve_ivp(lorenz_system, (t0, t1), y0, t_eval=t_points); ts = time.time()-start
    results['SciPy solve_ivp'] = {'time': ts, 'mse': 0}

    # Plots
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'ODE Solver Benchmark: Lorenz System (dt={dt})', fontsize=20)

    ax1 = fig.add_subplot(2, 2, (1, 3), projection='3d')
    ax1.plot(y_ref[:, 0], y_ref[:, 1], y_ref[:, 2], label='Reference Solution', color='black', alpha=0.7)
    ax1.plot(y_rk4[:, 0], y_rk4[:, 1], y_rk4[:, 2], 'r--', label='RK4 Solution', alpha=0.8)
    ax1.set_title('Lorenz Attractor', fontsize=16)
    ax1.set_xlabel('X Axis'); ax1.set_ylabel('Y Axis'); ax1.set_zlabel('Z Axis')
    ax1.legend(); ax1.view_init(elev=20, azim=-120)

    methods = list(results.keys()); times = [d['time'] for d in results.values()]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.bar(methods, times, color=colors)
    ax2.set_title('Execution Time Comparison', fontsize=16)
    ax2.set_ylabel('Time (seconds) - Log Scale')
    ax2.set_yscale('log')

    methods_mse = methods[:-1]
    mses = [results[m]['mse'] for m in methods_mse]
    colors_mse = colors[:-1]
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.bar(methods_mse, mses, color=colors_mse)
    ax3.set_title('Precision Comparison (Lower is Better)', fontsize=16)
    ax3.set_ylabel('Mean Squared Error (MSE) - Log Scale')
    ax3.set_yscale('log')

    if save_prefix:
        os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{save_prefix}_lorenz_benchmark.png", dpi=150)
        plt.close()
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    return dict(results=results, t=t_points, y_ref=y_ref, y_rk4=y_rk4)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='ODE tutorials (decay + Lorenz)')
    # Decay demo
    p.add_argument('--decay', action='store_true', help='Run scalar decay demo')
    p.add_argument('--y0', type=float, default=1.0, help='Initial condition for decay')
    p.add_argument('--t0', type=float, default=0.0, help='Start time')
    p.add_argument('--t1', type=float, default=10.0, help='End time')
    p.add_argument('--dt', type=float, default=1.0, help='Step size for decay demo')
    # Lorenz demo
    p.add_argument('--lorenz', action='store_true', help='Run Lorenz benchmark demo')
    p.add_argument('--lx0', type=float, default=1.0, help='Lorenz x0')
    p.add_argument('--ly0', type=float, default=1.0, help='Lorenz y0')
    p.add_argument('--lz0', type=float, default=1.0, help='Lorenz z0')
    p.add_argument('--ldt', type=float, default=0.01, help='Lorenz step size')
    p.add_argument('--lt1', type=float, default=25.0, help='Lorenz end time')
    # Output
    p.add_argument('--outdir', type=str, default='', help='If set, save figures here; otherwise show')
    p.add_argument('--prefix', type=str, default='ode_tutorial', help='Filename prefix for saved figures')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    save = bool(args.outdir)
    def out(name: str) -> str | None:
        return os.path.join(args.outdir, f"{args.prefix}_{name}.png") if save else None

    run_all = not (args.decay or args.lorenz)

    if args.decay or run_all:
        demo_simple_decay(y0=args.y0, t0=args.t0, t1=args.t1, dt=args.dt, save_path=out('decay'))

    if args.lorenz or run_all:
        save_prefix = os.path.join(args.outdir, args.prefix) if save else None
        demo_lorenz_benchmark(t0=args.t0, t1=args.lt1, dt=args.ldt, y0=(args.lx0, args.ly0, args.lz0), save_prefix=save_prefix)


if __name__ == '__main__':
    main()