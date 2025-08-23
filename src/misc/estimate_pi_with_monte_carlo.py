"""
Monte Carlo estimation of Pi with CLI and optional plotting to disk.

Refactored from a notebook export into a reusable, import-safe script.
"""

import argparse
import os
import random
import time
from typing import Optional

import matplotlib.pyplot as plt
from src.visualization.plot_utils import maybe_save_or_show

def estimate_pi_python(num_points: int, seed: Optional[int] = None, plot_cap: int = 10000) -> tuple[float, list[float], list[float], list[float], list[float]]:
    """
    Estimates Pi using the Monte Carlo method and returns the estimate
    along with points for plotting.

    Args:
        num_points: The total number of random points to generate.

    Returns:
        A tuple containing:
            - The estimated value of Pi.
            - A list of x-coordinates for points inside the circle.
            - A list of y-coordinates for points inside the circle.
            - A list of x-coordinates for points outside the circle.
            - A list of y-coordinates for points outside the circle.
    """
    if seed is not None:
        random.seed(seed)
    circle_points = 0
    # For plotting (optional to keep all, can be sampled for large num_points)
    inside_x: list[float] = []
    inside_y: list[float] = []
    outside_x: list[float] = []
    outside_y: list[float] = []

    if num_points == 0:
        return 0.0, inside_x, inside_y, outside_x, outside_y

    for _ in range(num_points):
        # Generate random x and y values from a uniform distribution
        # Range of x and y values is -1 to 1 (for a circle with radius 1 centered at 0,0)
        rand_x = random.uniform(-1, 1)
        rand_y = random.uniform(-1, 1)

        # Distance between (x, y) from the origin squared
        origin_dist_sq = rand_x**2 + rand_y**2

        # Checking if (x, y) lies inside the unit circle (x^2 + y^2 <= r^2, where r=1)
        if origin_dist_sq <= 1:
            circle_points += 1
            # Store points for plotting (can be demanding for very large N)
            if len(inside_x) < plot_cap:  # Limit points for plotting to avoid memory issues
                inside_x.append(rand_x)
                inside_y.append(rand_y)
        else:
            if len(outside_x) < plot_cap:  # Limit points for plotting
                outside_x.append(rand_x)
                outside_y.append(rand_y)

    # Pi = 4 * (number of points generated inside the circle) / (total number of points generated)
    pi_estimate = 4 * circle_points / num_points
    return pi_estimate, inside_x, inside_y, outside_x, outside_y

def _maybe_save_or_show(path: Optional[str] = None):
    # Delegate to shared helper
    maybe_save_or_show(path)


def plot_monte_carlo(pi_estimate: float, inside_x: list[float], inside_y: list[float], outside_x: list[float], outside_y: list[float], num_points: int, save_path: Optional[str] = None):
    """
    Plots the results of the Monte Carlo Pi estimation.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot points
    ax.scatter(inside_x, inside_y, color='blue', s=1, label='Inside Circle')
    ax.scatter(outside_x, outside_y, color='red', s=1, label='Outside Circle')

    # Draw the circle and the square
    circle = plt.Circle((0, 0), 1, color='lightgray', fill=False, linewidth=2)
    square = plt.Rectangle((-1, -1), 2, 2, color='gray', fill=False, linewidth=2)
    ax.add_artist(circle)
    ax.add_artist(square)

    # Set plot properties
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Monte Carlo Pi Estimation\nEstimate: {pi_estimate:.6f} (Points: {num_points:,})')
    ax.legend(loc='upper right')
    plt.xlabel("x")
    plt.ylabel("y")
    _maybe_save_or_show(save_path)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Estimate Pi with Monte Carlo")
    p.add_argument('-n', '--num-points', type=int, default=1_000_000, help='Number of random points')
    p.add_argument('--seed', type=int, default=None, help='Random seed')
    p.add_argument('--plot-cap', type=int, default=10000, help='Max points to keep for plotting')
    p.add_argument('--no-plot', action='store_true', help='Skip plotting')
    p.add_argument('--outdir', type=str, default='', help='If set, save plot(s) here; otherwise show')
    p.add_argument('--prefix', type=str, default='monte_carlo_pi', help='Filename prefix for saved figures')
    return p


def main(argv=None):
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    if args.num_points <= 0:
        print("No points requested. Pi estimation skipped.")
        return

    print(f"\nEstimating Pi using {args.num_points:,} points (Python implementation)...")

    start_time = time.perf_counter()
    pi_estimate, inside_x, inside_y, outside_x, outside_y = estimate_pi_python(args.num_points, seed=args.seed, plot_cap=args.plot_cap)
    end_time = time.perf_counter()

    print(f"Final Estimation of Pi = {pi_estimate:.8f}")
    print(f"Calculation took: {end_time - start_time:.4f} seconds")
    print(f"Actual value of Pi (approx) = {3.1415926535}")
    print(f"Difference = {abs(pi_estimate - 3.1415926535):.8f}")

    if not args.no_plot:
        save_path = os.path.join(args.outdir, f"{args.prefix}.png") if args.outdir else None
        plot_monte_carlo(pi_estimate, inside_x, inside_y, outside_x, outside_y, args.num_points, save_path=save_path)

if __name__ == "__main__":
    main()