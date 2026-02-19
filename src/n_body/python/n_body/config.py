from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass
class Config:
    G: float = 1.0
    S: float = 0.1
    dt: float = 0.01
    tf: float = 10.0
    seed: Optional[int] = 42


@dataclass
class State:
    mass: FloatArray  # shape (n, 1)
    pos: FloatArray   # shape (n, 3)
    vel: FloatArray   # shape (n, 3)

    @property
    def n(self) -> int:
        return int(self.mass.shape[0])


@dataclass
class Result:
    state_history: FloatArray  # shape (T+1, n, 6)
    acc_history: FloatArray    # shape (T+1, n, 3)
    ke: FloatArray             # shape (T+1,)
    pe: FloatArray             # shape (T+1,)
    dt: float = 0.01

    def as_numpy(self) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        return self.state_history, self.acc_history, self.ke, self.pe
