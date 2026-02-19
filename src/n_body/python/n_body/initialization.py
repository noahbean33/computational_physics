from __future__ import annotations

from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray

from .config import State

FloatArray = NDArray[np.float64]


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed) if seed is not None else np.random.default_rng()


def get_mass_vec(n: int, equal: bool, rng: np.random.Generator) -> FloatArray:
    if equal:
        return (100.0 * np.ones((n, 1)) / n).astype(np.float64)
    m = rng.random((n, 1)).astype(np.float64)
    return (100.0 * m / np.sum(m)).astype(np.float64)


def _rand_points_in_ball(n: int, rng: np.random.Generator) -> FloatArray:
    d = 3
    u = rng.normal(0.0, 1.0, (d, n))
    norm = np.sum(u**2, axis=0) ** 0.5
    r = rng.random(n) ** (1.0 / d)
    return (r * u / norm).T.astype(np.float64)


def get_rand_states(n: int, mass_vec: FloatArray, init_vel: bool, scale: float, rng: np.random.Generator) -> State:
    pos = (scale * _rand_points_in_ball(n, rng)).astype(np.float64)
    vel = (init_vel * scale * (2 * rng.random((n, 3)) - 1)).astype(np.float64)
    return State(mass=mass_vec.astype(np.float64), pos=pos, vel=vel)


InitType = Union[int, float, list, FloatArray]


def set_states(
    init_conds: InitType,
    mass_equal: bool,
    init_vel: bool,
    max_pts: int,
    seed: Optional[int] = 42,
    scale: float = 10.0,
) -> State:
    rng = _rng(seed)

    if isinstance(init_conds, float):
        init_conds = int(init_conds)
    if isinstance(init_conds, list):
        init_conds = np.array(init_conds, dtype=np.float64)

    if isinstance(init_conds, int):
        init_conds = abs(init_conds)
        n = init_conds if init_conds > 1 else int(rng.integers(2, max_pts))
        mass_vec = get_mass_vec(n, mass_equal, rng)
        return get_rand_states(n, mass_vec, init_vel, scale, rng)

    if isinstance(init_conds, np.ndarray):
        arr = init_conds.astype(np.float64)
        n = arr.shape[0]
        # Try [n x 7]: mass, pos(3), vel(3)
        if arr.ndim == 2 and arr.shape[1] == 7:
            mass = arr[:, 0:1]
            pos = arr[:, 1:4]
            vel = arr[:, 4:7]
            return State(mass=mass, pos=pos, vel=vel)
        # Try [n x 6]: pos(3), vel(3), random masses
        if arr.ndim == 2 and arr.shape[1] == 6:
            mass = get_mass_vec(n, mass_equal, rng)
            pos = arr[:, 0:3]
            vel = arr[:, 3:6]
            return State(mass=mass, pos=pos, vel=vel)
        # Try [n x 4]: mass, pos(3), random vel
        if arr.ndim == 2 and arr.shape[1] == 4:
            mass = arr[:, 0:1]
            pos = arr[:, 1:4]
            vel = (init_vel * (2 * rng.random((n, 3)) - 1)).astype(np.float64)
            return State(mass=mass, pos=pos, vel=vel)
        # Try [n x 3]: pos(3), random mass and vel
        if arr.ndim == 2 and arr.shape[1] == 3:
            mass = get_mass_vec(n, mass_equal, rng)
            pos = arr
            vel = (init_vel * (2 * rng.random((n, 3)) - 1)).astype(np.float64)
            return State(mass=mass, pos=pos, vel=vel)
        # Try [n x 1]: mass, random pos and vel
        if arr.ndim == 2 and arr.shape[1] == 1:
            mass = arr
            return get_rand_states(n, mass, init_vel, scale, rng)

    # Fallback: random generation
    print("\nWARNING: INVALID INITIAL CONDITIONS\ninitial conditions randomized... done.\n")
    n = int(rng.integers(2, max_pts))
    mass_vec = get_mass_vec(n, mass_equal, rng)
    return get_rand_states(n, mass_vec, init_vel, scale, rng)
