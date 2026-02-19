from __future__ import annotations

import numpy as np

from .config import Config, State, Result
from .physics import accelerations, energy


class Simulation:
    def __init__(self, config: Config, state: State) -> None:
        self.cfg = config
        self.state = self._to_com_frame(state)
        self.T = int(np.ceil(self.cfg.tf / self.cfg.dt))

    @staticmethod
    def _to_com_frame(state: State) -> State:
        # Transform velocities into the Center-of-Mass (COM) frame
        total_mass = np.sum(state.mass)
        v_com = np.sum(state.mass * state.vel, axis=0, keepdims=True) / total_mass
        vel = state.vel - v_com
        return State(mass=state.mass, pos=state.pos, vel=vel)

    def run(self) -> Result:
        n = self.state.mass.shape[0]
        dt = self.cfg.dt

        acc = accelerations(self.state, self.cfg.G, self.cfg.S)
        KE, PE = energy(self.state, self.cfg.G)

        acc_hist = np.zeros((self.T + 1, n, 3), dtype=np.float64)
        ke_hist = np.zeros((self.T + 1,), dtype=np.float64)
        pe_hist = np.zeros((self.T + 1,), dtype=np.float64)
        state_hist = np.zeros((self.T + 1, n, 6), dtype=np.float64)

        acc_hist[0] = acc
        ke_hist[0] = KE
        pe_hist[0] = PE
        state_hist[0, :, 0:3] = self.state.pos
        state_hist[0, :, 3:6] = self.state.vel

        for i in range(self.T):
            # Kick
            self.state.vel += acc * dt / 2.0
            # Drift
            self.state.pos += self.state.vel * dt
            # Update accelerations
            acc = accelerations(self.state, self.cfg.G, self.cfg.S)
            # Kick
            self.state.vel += acc * dt / 2.0

            KE, PE = energy(self.state, self.cfg.G)

            acc_hist[i + 1] = acc
            ke_hist[i + 1] = KE
            pe_hist[i + 1] = PE
            state_hist[i + 1, :, 0:3] = self.state.pos
            state_hist[i + 1, :, 3:6] = self.state.vel

        return Result(
            state_history=state_hist,
            acc_history=acc_hist,
            ke=ke_hist,
            pe=pe_hist,
            dt=dt,
        )
