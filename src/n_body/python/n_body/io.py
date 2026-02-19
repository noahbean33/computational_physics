from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from .config import Result


def write_csv(result: Result, path: str | Path = "simulation_output.csv") -> None:
    """
    Write simulation trajectory to CSV similar to the Rust output style.

    Columns: t,index,x,y,z,vx,vy,vz
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = result.state_history  # shape (T+1, n, 6)
    T_plus_1, n, _ = state.shape

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "index", "x", "y", "z", "vx", "vy", "vz"])  # header
        for t in range(T_plus_1):
            for i in range(n):
                x, y, z, vx, vy, vz = state[t, i]
                w.writerow([t, i, float(x), float(y), float(z), float(vx), float(vy), float(vz)])
