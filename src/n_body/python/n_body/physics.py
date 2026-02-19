from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import State

FloatArray = NDArray[np.float64]


def accelerations(state: State, G: float, S: float) -> FloatArray:
    x = state.pos[:, 0:1]
    y = state.pos[:, 1:2]
    z = state.pos[:, 2:3]

    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    r3_inv = (dx**2 + dy**2 + dz**2 + S**2)
    mask = r3_inv > 0
    r3_inv[mask] = r3_inv[mask] ** (-1.5)

    ax = G * (dx * r3_inv) @ state.mass
    ay = G * (dy * r3_inv) @ state.mass
    az = G * (dz * r3_inv) @ state.mass

    return np.hstack((ax, ay, az)).astype(np.float64)


def energy(state: State, G: float) -> tuple[float, float]:
    # Kinetic: 0.5 * m * v^2
    KE = float(np.sum(state.mass * (state.vel ** 2)) / 2.0)

    # Potential: sum over pairs -G m_i m_j / r_ij
    x = state.pos[:, 0:1]
    y = state.pos[:, 1:2]
    z = state.pos[:, 2:3]

    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    r = np.sqrt(dx**2 + dy**2 + dz**2)
    with np.errstate(divide="ignore"):
        r_inv = np.zeros_like(r)
        mask = r > 0
        r_inv[mask] = 1.0 / r[mask]
    PE = float(G * np.sum(np.sum(np.triu(-(state.mass @ state.mass.T) * r_inv, 1))))

    return KE, PE
