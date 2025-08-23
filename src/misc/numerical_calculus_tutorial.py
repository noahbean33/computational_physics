"""
Numerical calculus tutorials (differentiation and integration) with CLI.

Refactored from a notebook export into a reusable, import-safe script.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.visualization.plot_utils import maybe_save_or_show
from scipy.interpolate import BarycentricInterpolator

def f(x):
  """
  The example function for differentiation.
  In this case, f(x) = sin(x).
  """
  return np.sin(x)

def analytical_derivative(x):
  """
  The analytical derivative of the function f(x).
  For f(x) = sin(x), the derivative is f'(x) = cos(x).
  """
  return np.cos(x)

def forward_difference(f, x, h):
  """
  Calculates the forward difference approximation of the derivative.
  """
  return (f(x + h) - f(x)) / h

def backward_difference(f, x, h):
  """
  Calculates the backward difference approximation of the derivative.
  """
  return (f(x) - f(x - h)) / h

def central_difference(f, x, h):
  """
  Calculates the central difference approximation of the derivative.
  """
  return (f(x + h) - f(x - h)) / (2 * h)

def higher_order_central_difference(f, x, h):
  """
  Calculates a higher-order central difference approximation of the derivative.
  """
  return (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12 * h)

def plot_derivatives_and_errors(h: float = 0.1, show: bool = True, save_path: str | None = None):
  """
  Generates plots for the function, its derivatives, and the errors of the
  numerical methods.
  """
  x = np.linspace(0, 2 * np.pi, 100)

  # Calculate the derivatives
  y = f(x)
  y_prime_analytical = analytical_derivative(x)
  y_prime_forward = forward_difference(f, x, h)
  y_prime_backward = backward_difference(f, x, h)
  y_prime_central = central_difference(f, x, h)
  y_prime_higher_central = higher_order_central_difference(f, x, h)

  # Calculate the errors
  error_forward = np.abs(y_prime_forward - y_prime_analytical)
  error_backward = np.abs(y_prime_backward - y_prime_analytical)
  error_central = np.abs(y_prime_central - y_prime_analytical)
  error_higher_central = np.abs(y_prime_higher_central - y_prime_analytical)

  # Plotting the function and its derivatives
  plt.figure(figsize=(12, 10))

  plt.subplot(2, 1, 1)
  plt.plot(x, y, label='f(x) = sin(x)', color='blue')
  plt.plot(x, y_prime_analytical, label="f'(x) = cos(x) (Analytical)", color='black', linestyle='--')
  plt.plot(x, y_prime_forward, label='Forward Difference', linestyle=':', marker='.')
  plt.plot(x, y_prime_backward, label='Backward Difference', linestyle=':', marker='.')
  plt.plot(x, y_prime_central, label='Central Difference', linestyle='-.', marker='.')
  plt.plot(x, y_prime_higher_central, label='Higher-Order Central Difference', linestyle='-', marker='.')
  plt.title('Numerical Differentiation of f(x) = sin(x)')
  plt.xlabel('x')
  plt.ylabel("f(x) and f'(x)")
  plt.legend()
  plt.grid(True)

  # Plotting the errors
  plt.subplot(2, 1, 2)
  plt.plot(x, error_forward, label='Forward Difference Error', linestyle=':', marker='.')
  plt.plot(x, error_backward, label='Backward Difference Error', linestyle=':', marker='.')
  plt.plot(x, error_central, label='Central Difference Error', linestyle='-.', marker='.')
  plt.plot(x, error_higher_central, label='Higher-Order Central Difference Error', linestyle='-', marker='.')
  plt.title('Error of Numerical Differentiation Methods')
  plt.xlabel('x')
  plt.ylabel('Absolute Error')
  plt.yscale('log')
  plt.legend()
  plt.grid(True)

  plt.tight_layout()
  maybe_save_or_show(save_path, show)



def analytical_integral(a, b):
    """
    The analytical definite integral of f(x) from a to b.
    For f(x) = sin(x), the integral is [-cos(x)] from a to b.
    """
    return -np.cos(b) + np.cos(a)

# --- Numerical Integration Functions ---
def left_riemann_sum(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b - h, n)
    return h * np.sum(f(x))

def right_riemann_sum(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a + h, b, n)
    return h * np.sum(f(x))

def midpoint_riemann_sum(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a + h / 2, b - h / 2, n)
    return h * np.sum(f(x))

def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    return (h / 2) * (f(x[0]) + 2 * np.sum(f(x[1:-1])) + f(x[-1]))

def simpsons_rule(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("Number of subintervals (n) must be even for Simpson's rule.")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    return (h / 3) * (f(x[0]) + 4 * np.sum(f(x[1:-1:2])) + 2 * np.sum(f(x[2:-2:2])) + f(x[-1]))

def plot_error_comparison(show: bool = True, save_path: str | None = None):
    """
    Generates a plot comparing the absolute error of each integration method
    as the number of subintervals increases. This recreates the user's plot.
    """
    a, b = 0, np.pi
    analytical_solution = analytical_integral(a, b)

    n_values = np.arange(2, 101, 2)
    methods = {
        "Left Riemann Sum": left_riemann_sum,
        "Right Riemann Sum": right_riemann_sum,
        "Midpoint Riemann Sum": midpoint_riemann_sum,
        "Trapezoidal Rule": trapezoidal_rule,
        "Simpson's Rule": simpsons_rule
    }
    errors = {name: [] for name in methods}

    for n in n_values:
        for name, func in methods.items():
            error = np.abs(func(f, a, b, n) - analytical_solution)
            errors[name].append(error)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8))
    for name, error_list in errors.items():
        plt.plot(n_values, error_list, label=name, marker='.', markersize=5)

    plt.title(r'Error of Numerical Integration Methods for $\int_0^{\pi} \sin(x)dx$')
    plt.xlabel('Number of Subintervals (n)')
    plt.ylabel('Absolute Error')
    plt.yscale('log')
    plt.legend()
    _maybe_save_or_show(save_path, show)

def plot_method_visualizations(n_vis: int = 10, show: bool = True, save_path: str | None = None):
    """
    Generates a plot visualizing the area approximated by each method for a
    fixed number of subintervals.
    """
    a, b = 0, np.pi
    x_curve = np.linspace(a, b, 200)
    y_curve = f(x_curve)

    fig, axs = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle('Visualization of Numerical Integration Methods (n=10)', fontsize=20, y=0.95)
    axs = axs.ravel()

    # --- 1. Original Function ---
    axs[0].plot(x_curve, y_curve, 'b', label='f(x) = sin(x)')
    axs[0].fill_between(x_curve, 0, y_curve, color='blue', alpha=0.1)
    axs[0].set_title(f'Original Area\nAnalytical Value ≈ {analytical_integral(a, b):.4f}')

    # --- 2. Left Riemann Sum ---
    x_steps = np.linspace(a, b, n_vis + 1)
    y_steps = f(x_steps)
    axs[1].plot(x_curve, y_curve, 'b')
    for i in range(n_vis):
        rect = patches.Rectangle((x_steps[i], 0), x_steps[i+1] - x_steps[i], y_steps[i], facecolor='red', alpha=0.5, edgecolor='black')
        axs[1].add_patch(rect)
    axs[1].set_title(f'Left Riemann Sum\nApprox. Value ≈ {left_riemann_sum(f, a, b, n_vis):.4f}')

    # --- 3. Right Riemann Sum ---
    axs[2].plot(x_curve, y_curve, 'b')
    for i in range(n_vis):
        rect = patches.Rectangle((x_steps[i], 0), x_steps[i+1] - x_steps[i], y_steps[i+1], facecolor='darkorange', alpha=0.5, edgecolor='black')
        axs[2].add_patch(rect)
    axs[2].set_title(f'Right Riemann Sum\nApprox. Value ≈ {right_riemann_sum(f, a, b, n_vis):.4f}')

    # --- 4. Midpoint Riemann Sum ---
    axs[3].plot(x_curve, y_curve, 'b')
    for i in range(n_vis):
        mid_point = (x_steps[i] + x_steps[i+1]) / 2
        height = f(mid_point)
        rect = patches.Rectangle((x_steps[i], 0), x_steps[i+1] - x_steps[i], height, facecolor='green', alpha=0.5, edgecolor='black')
        axs[3].add_patch(rect)
    axs[3].set_title(f'Midpoint Riemann Sum\nApprox. Value ≈ {midpoint_riemann_sum(f, a, b, n_vis):.4f}')

    # --- 5. Trapezoidal Rule ---
    axs[4].plot(x_curve, y_curve, 'b')
    for i in range(n_vis):
        poly = patches.Polygon([(x_steps[i], 0), (x_steps[i], y_steps[i]), (x_steps[i+1], y_steps[i+1]), (x_steps[i+1], 0)], facecolor='red', alpha=0.5, edgecolor='black')
        axs[4].add_patch(poly)
    axs[4].set_title(f'Trapezoidal Rule\nApprox. Value ≈ {trapezoidal_rule(f, a, b, n_vis):.4f}')

    # --- 6. Simpson's Rule ---
    axs[5].plot(x_curve, y_curve, 'b')
    for i in range(0, n_vis, 2):
        x_segment = x_steps[i:i+3]
        y_segment = y_steps[i:i+3]
        # Fit a parabola (quadratic) to the three points
        interp = BarycentricInterpolator(x_segment, y_segment)
        x_interp_curve = np.linspace(x_segment[0], x_segment[-1], 50)
        y_interp_curve = interp(x_interp_curve)
        axs[5].plot(x_interp_curve, y_interp_curve, 'm-')
        axs[5].fill_between(x_interp_curve, 0, y_interp_curve, color='purple', alpha=0.5)
    axs[5].set_title(f"Simpson's Rule\nApprox. Value ≈ {simpsons_rule(f, a, b, n_vis):.4f}")

    for ax in axs:
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend(['f(x) = sin(x)'], loc='upper right')
        ax.set_ylim(0, 1.2) # Give some space at the top

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    maybe_save_or_show(save_path, show)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Numerical calculus demos")
    p.add_argument("--diff", action="store_true", help="Run differentiation error demo")
    p.add_argument("--integ-error", action="store_true", help="Run integration error comparison")
    p.add_argument("--integ-visual", action="store_true", help="Run integration method visualization")
    p.add_argument("--h", type=float, default=0.1, help="Step size for differentiation demo")
    p.add_argument("--n-vis", type=int, default=10, help="n intervals for visualization")
    p.add_argument("--outdir", type=str, default="", help="If set, save figures here; otherwise show interactively")
    p.add_argument("--prefix", type=str, default="numerical_calc", help="Filename prefix for saved figures")
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)
    save = bool(args.outdir)
    def out(name: str) -> str | None:
        return os.path.join(args.outdir, f"{args.prefix}_{name}.png") if save else None

    run_all = not (args.diff or args.integ_error or args.integ_visual)

    if args.diff or run_all:
        plot_derivatives_and_errors(h=args.h, show=not save, save_path=out("diff_errors"))

    if args.integ_error or run_all:
        plot_error_comparison(show=not save, save_path=out("integration_errors"))

    if args.integ_visual or run_all:
        plot_method_visualizations(n_vis=args.n_vis, show=not save, save_path=out("integration_visual"))

if __name__ == "__main__":
    main()