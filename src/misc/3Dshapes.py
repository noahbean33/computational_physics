"""
3Dshapes.py: Standalone Matplotlib 3D rendering of several shapes.

Converted from VPython to Matplotlib to remove external visual module dependency.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)


def plot_sphere(ax, center=(0, 0, 0), radius=1.0, color="green", alpha=0.8):
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


def plot_arrow(ax, pos=(0, 0, 0), axis=(1, 0, 0), color="cyan"):
    x0, y0, z0 = pos
    dx, dy, dz = axis
    ax.quiver(x0, y0, z0, dx, dy, dz, color=color, arrow_length_ratio=0.15, linewidth=2)


def plot_cylinder(ax, pos=(-3, -2, 3), axis=(6, -1, 5), radius=1.0, color="gold", alpha=0.6):
    # Parameterize cylinder around the axis using local frame via SVD
    p0 = np.array(pos, dtype=float)
    v = np.array(axis, dtype=float)
    L = np.linalg.norm(v)
    if L == 0:
        return
    v = v / L
    # Find two orthonormal vectors perpendicular to v
    a = np.array([1.0, 0.0, 0.0])
    if np.allclose(np.abs(np.dot(a, v)), 1.0):
        a = np.array([0.0, 1.0, 0.0])
    n1 = a - np.dot(a, v) * v
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(v, n1)
    theta = np.linspace(0, 2 * np.pi, 40)
    h = np.linspace(0, L, 2)
    theta, h = np.meshgrid(theta, h)
    X = p0[0] + h * v[0] + radius * (np.cos(theta) * n1[0] + np.sin(theta) * n2[0])
    Y = p0[1] + h * v[1] + radius * (np.cos(theta) * n1[1] + np.sin(theta) * n2[1])
    Z = p0[2] + h * v[2] + radius * (np.cos(theta) * n1[2] + np.sin(theta) * n2[2])
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)


def plot_cone(ax, pos=(-6, -6, 0), axis=(-2, 1, -0.5), radius=2.0, color="magenta", alpha=0.6):
    p0 = np.array(pos, dtype=float)
    v = np.array(axis, dtype=float)
    L = np.linalg.norm(v)
    if L == 0:
        return
    v = v / L
    a = np.array([1.0, 0.0, 0.0])
    if np.allclose(np.abs(np.dot(a, v)), 1.0):
        a = np.array([0.0, 1.0, 0.0])
    n1 = a - np.dot(a, v) * v
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(v, n1)
    theta = np.linspace(0, 2 * np.pi, 40)
    r = np.linspace(0, radius, 20)
    theta, r = np.meshgrid(theta, r)
    # Build along the axis direction
    X = p0[0] + (r / radius) * axis[0] + r * (np.cos(theta) * n1[0] + np.sin(theta) * n2[0])
    Y = p0[1] + (r / radius) * axis[1] + r * (np.cos(theta) * n1[1] + np.sin(theta) * n2[1])
    Z = p0[2] + (r / radius) * axis[2] + r * (np.cos(theta) * n1[2] + np.sin(theta) * n2[2])
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)


def plot_helix(ax, pos=(-5, 5, -2), axis=(5, 0, 0), radius=2.0, turns=2.0, thickness=1.5, color="orange"):
    p0 = np.array(pos, dtype=float)
    v = np.array(axis, dtype=float)
    L = np.linalg.norm(v)
    if L == 0:
        return
    v = v / L
    a = np.array([1.0, 0.0, 0.0])
    if np.allclose(np.abs(np.dot(a, v)), 1.0):
        a = np.array([0.0, 1.0, 0.0])
    n1 = a - np.dot(a, v) * v
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(v, n1)
    t = np.linspace(0, 2 * np.pi * turns, 400)
    center = p0 + (L / (2 * np.pi * turns)) * t[:, None] * v
    x = center[:, 0] + radius * np.cos(t) * n1[0] + radius * np.sin(t) * n2[0]
    y = center[:, 1] + radius * np.cos(t) * n1[1] + radius * np.sin(t) * n2[1]
    z = center[:, 2] + radius * np.cos(t) * n1[2] + radius * np.sin(t) * n2[2]
    ax.plot(x, y, z, color=color, linewidth=thickness)


def plot_ring(ax, pos=(-6, 1, 0), axis=(1, 1, 1), radius=2.0, color=(0.3, 0.4, 0.6)):
    p0 = np.array(pos, dtype=float)
    v = np.array(axis, dtype=float)
    v = v / (np.linalg.norm(v) or 1.0)
    a = np.array([1.0, 0.0, 0.0])
    if np.allclose(np.abs(np.dot(a, v)), 1.0):
        a = np.array([0.0, 1.0, 0.0])
    n1 = a - np.dot(a, v) * v
    n1 /= (np.linalg.norm(n1) or 1.0)
    n2 = np.cross(v, n1)
    t = np.linspace(0, 2 * np.pi, 200)
    x = p0[0] + radius * np.cos(t) * n1[0] + radius * np.sin(t) * n2[0]
    y = p0[1] + radius * np.cos(t) * n1[1] + radius * np.sin(t) * n2[1]
    z = p0[2] + radius * np.cos(t) * n1[2] + radius * np.sin(t) * n2[2]
    ax.plot(x, y, z, color=color)


def plot_box(ax, pos=(5, -2, 2), size=(5, 5, 0.4), color=(0.4, 0.8, 0.2), alpha=0.4):
    from itertools import product, combinations

    lx, ly, lz = size
    cx, cy, cz = pos
    # Compute 8 corners
    x = np.array([cx - lx / 2, cx + lx / 2])
    y = np.array([cy - ly / 2, cy + ly / 2])
    z = np.array([cz - lz / 2, cz + lz / 2])
    corners = np.array(list(product(x, y, z)))
    # Draw edges
    for s, e in combinations(corners, 2):
        if np.sum(np.abs(s - e) > 1e-9) == 1:
            ax.plot3D(*zip(s, e), color=color, alpha=alpha)


def plot_pyramid(ax, pos=(2, 5, 2), size=(4, 3, 2), color=(0.7, 0.7, 0.2)):
    # Rectangular base pyramid, base center at pos
    lx, ly, h = size
    cx, cy, cz = pos
    # Base corners
    base = np.array([
        [cx - lx / 2, cy - ly / 2, cz],
        [cx + lx / 2, cy - ly / 2, cz],
        [cx + lx / 2, cy + ly / 2, cz],
        [cx - lx / 2, cy + ly / 2, cz],
    ])
    apex = np.array([cx, cy, cz + h])
    # Draw base
    for i in range(4):
        s = base[i]
        e = base[(i + 1) % 4]
        ax.plot3D(*zip(s, e), color=color)
    # Draw sides
    for i in range(4):
        ax.plot3D(*zip(base[i], apex), color=color)


def plot_ellipsoid(ax, pos=(-1, -7, 1), axes=(2, 1, 3), color=(0.1, 0.9, 0.8), alpha=0.7):
    a, b, c = axes
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x = pos[0] + a * np.outer(np.cos(u), np.sin(v))
    y = pos[1] + b * np.outer(np.sin(u), np.sin(v))
    z = pos[2] + c * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


def render(out_path: str | None = None, show: bool = True):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Matplotlib 3D Shapes")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)

    plot_sphere(ax, center=(0, 0, 0), radius=1.0, color="green")
    plot_sphere(ax, center=(0, 1, -3), radius=1.5, color="red")
    plot_arrow(ax, pos=(3, 2, 2), axis=(3, 1, 1), color="cyan")
    plot_cylinder(ax, pos=(-3, -2, 3), axis=(6, -1, 5), radius=1.0, color="gold")
    plot_cone(ax, pos=(-6, -6, 0), axis=(-2, 1, -0.5), radius=2.0, color="magenta")
    plot_helix(ax, pos=(-5, 5, -2), axis=(5, 0, 0), radius=2.0, turns=2.0, thickness=2.0, color="orange")
    plot_ring(ax, pos=(-6, 1, 0), axis=(1, 1, 1), radius=2.0, color=(0.3, 0.4, 0.6))
    plot_box(ax, pos=(5, -2, 2), size=(5, 5, 0.4), color=(0.4, 0.8, 0.2))
    plot_pyramid(ax, pos=(2, 5, 2), size=(4, 3, 2), color=(0.7, 0.7, 0.2))
    plot_ellipsoid(ax, pos=(-1, -7, 1), axes=(2, 1, 3), color=(0.1, 0.9, 0.8))

    plt.tight_layout()
    if out_path:
        import os

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
    elif show:
        plt.show()


def build_parser():
    p = argparse.ArgumentParser(description="Render a set of 3D shapes using Matplotlib")
    p.add_argument("--out", type=str, default="", help="If set, save figure to this path; otherwise show interactively")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    render(out_path=args.out or None, show=True)