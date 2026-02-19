from __future__ import annotations

import argparse
import numpy as np

from .config import Config
from .initialization import set_states
from .simulation import Simulation
from .viz import animate
from .io import write_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Python N-body simulator")
    p.add_argument("-n", type=int, default=0, help="number of bodies (0 -> random 2..max_pts)")
    p.add_argument("-t", "--time", dest="tf", type=float, default=10.0, help="total simulation time")
    p.add_argument("--dt", type=float, default=0.01, help="time step")
    p.add_argument("--equal-mass", action="store_true", help="use equal masses")
    p.add_argument("--init-vel", action="store_true", help="use non-zero initial velocities")
    p.add_argument("--scale", type=float, default=10.0, help="position/velocity scale for random init")
    p.add_argument("--seed", type=int, default=42, help="random seed")

    p.add_argument("--visualize", action="store_true", help="render matplotlib animation")
    p.add_argument("--save-fps", type=int, default=100, help="animation fps if saving (not yet saving)")
    p.add_argument("--view-lim", type=int, default=20, help="plot view limit magnitude")
    p.add_argument("--autoscroll", action="store_true", help="auto-scale plot to keep all points in view")
    p.add_argument(
        "--csv-out",
        nargs="?",
        const="simulation_output.csv",
        default=None,
        help="write trajectory CSV to optional PATH (default: simulation_output.csv)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    max_pts = max(args.n + 1, 6)
    init_conds = int(args.n) if args.n > 0 else 0

    state = set_states(
        init_conds=init_conds,
        mass_equal=args.equal_mass,
        init_vel=args.init_vel,
        max_pts=max_pts,
        seed=args.seed,
        scale=args.scale,
    )

    cfg = Config(G=1.0, S=0.1, dt=args.dt, tf=args.tf, seed=args.seed)
    sim = Simulation(cfg, state)
    result = sim.run()

    print("N-body simulation finished (Python). Final state sample:")
    print(result.state_history[-1, :min(10, state.n), :])

    if args.csv_out is not None:
        write_csv(result, path=args.csv_out)
        print(f"Wrote CSV to {args.csv_out}")

    if args.visualize:
        animate(result, masses=state.mass, fps=args.save_fps, view_lim=args.view_lim, autoscroll=args.autoscroll, replay=True)


if __name__ == "__main__":
    main()
