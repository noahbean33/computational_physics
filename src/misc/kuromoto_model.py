"""
Kuramoto model simulation utilities with a CLI demo.

Refactored from a notebook export into a reusable, import-safe module.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from src.visualization.plot_utils import maybe_save_or_show

def kuramoto_rhs(t, theta, omega, K, A):
    phase_diff = theta[None, :] - theta[:, None]          # θ_j − θ_i
    coupling   = np.sum(A * np.sin(phase_diff), axis=1)   # Σ_j …
    return omega + (K / theta.size) * coupling


# ─────────────────────────────────────────────────────────────────────────────
# 2.  One-stop simulation helper with rich diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def kuramoto_simulate(N=32,
                      K=1.0,
                      t_span=(0, 50),
                      dt=0.05,
                      omega_std=1.0,
                      A=None,
                      seed=None):
    """
    Returns {t, theta, r, psi, sigma, omega} for plotting / analysis.
    """
    rng   = np.random.default_rng(seed)
    omega = rng.normal(0.0, omega_std, N)                 # natural freqs
    A     = np.ones((N, N)) - np.eye(N) if A is None else np.asarray(A)
    np.fill_diagonal(A, 0.0)                              # no self-loops
    theta0 = rng.uniform(0, 2*np.pi, N)                   # phases @ t0

    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
    sol    = solve_ivp(kuramoto_rhs, t_span, theta0,
                       args=(omega, K, A),
                       t_eval=t_eval, atol=1e-9, rtol=1e-9)

    t, theta = sol.t, sol.y                               # (M,), (N,M)
    z        = np.exp(1j*theta)                           # e^{iθ}
    r, psi   = np.abs(z.mean(0)), np.angle(z.mean(0))     # order parameter
    ω_eff    = np.diff(theta)/np.diff(t)                  # inst. freqs
    sigma    = np.std(ω_eff, axis=0)                      # spread σ(t)

    return dict(t=t, theta=theta, r=r, psi=psi,
                sigma=sigma, omega=omega)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Quick demo run  (adjust N, K, omega_std, t_span, seed ⬇)
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Kuramoto model demo")
    p.add_argument("--N", type=int, default=50, help="Number of oscillators")
    p.add_argument("--K", type=float, default=2.0, help="Coupling strength")
    p.add_argument("--omega-std", type=float, default=1.0, help="Std dev of natural frequencies")
    p.add_argument("--t0", type=float, default=0.0, help="Start time")
    p.add_argument("--t1", type=float, default=40.0, help="End time")
    p.add_argument("--dt", type=float, default=0.05, help="Time step for sampling")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--outdir", type=str, default="", help="If set, save figures here; otherwise show interactively")
    p.add_argument("--prefix", type=str, default="kuramoto", help="Filename prefix for saved figures")
    p.add_argument("--no-hist", action="store_true", help="Skip natural frequency histogram")
    p.add_argument("--no-r", action="store_true", help="Skip r(t) plot")
    p.add_argument("--no-sigma", action="store_true", help="Skip sigma(t) plot")
    p.add_argument("--no-phases", action="store_true", help="Skip phase trajectories plot")
    return p


def _maybe_save_or_show(path: str | None):
    # Delegate to shared plotting helper for consistency across scripts
    maybe_save_or_show(path)


def main(argv=None):
    args = build_parser().parse_args(argv)
    results = kuramoto_simulate(
        N=args.N, K=args.K, omega_std=args.omega_std,
        t_span=(args.t0, args.t1), dt=args.dt, seed=args.seed
    )

    t = results["t"]; theta = results["theta"]; r = results["r"]; sigma = results["sigma"]; omega = results["omega"]

    save = bool(args.outdir)
    def out(name: str) -> str | None:
        return os.path.join(args.outdir, f"{args.prefix}_{name}.png") if save else None

    if not args.no_hist:
        plt.figure(); plt.hist(omega, bins=12, edgecolor="black")
        plt.xlabel("Natural frequency ωᵢ"); plt.ylabel("count"); plt.grid(True)
        plt.title("Distribution of natural frequencies")
        _maybe_save_or_show(out("omega_hist"))

    if not args.no_r:
        plt.figure(); plt.plot(t, r)
        plt.xlabel("Time"); plt.ylabel("r(t)"); plt.grid(True)
        plt.title(f"Kuramoto synchronisation  (N={args.N}, K={args.K})")
        _maybe_save_or_show(out("r_t"))

    if not args.no_sigma:
        plt.figure(); plt.plot(t[1:], sigma)
        plt.xlabel("Time"); plt.ylabel("σ(ω_eff)"); plt.grid(True)
        plt.title("Evolution of effective frequency dispersion")
        _maybe_save_or_show(out("sigma_t"))

    if not args.no_phases:
        plt.figure()
        for i in range(min(args.N, 12)):
            plt.plot(t, np.mod(theta[i], 2*np.pi), lw=0.8)
        plt.xlabel("Time"); plt.ylabel("Phase θᵢ (mod 2π)"); plt.grid(True)
        plt.title("Sample oscillator phases")
        _maybe_save_or_show(out("phases"))

    # Console summary
    tail = slice(-100, None)
    print("\n=== Diagnostics ===")
    print(f"⟨r⟩ (last 100 pts)  = {r[tail].mean():.3f}")
    print(f"σ(ω_eff) tail      = {sigma[tail].mean():.3f}  rad s⁻¹")
    print(f"ωᵢ  st.dev.        = {args.omega_std}")


if __name__ == "__main__":
    main()