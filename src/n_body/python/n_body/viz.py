from __future__ import annotations

from typing import Optional

try:
    import matplotlib.pyplot as plt
    from matplotlib import animation
except Exception:  # pragma: no cover - optional dependency
    plt = None
    animation = None

import numpy as np

from .config import Result


def _require_matplotlib() -> None:
    if plt is None or animation is None:
        raise RuntimeError("matplotlib is required for visualization. Install with `pip install .[viz]`.")


def energy_bounds(ke: np.ndarray, pe: np.ndarray) -> int:
    b = int(max(np.max(np.abs(ke)), np.max(np.abs(pe)))) + 1
    if b < 10:
        return b
    if b < 100:
        return (b + 19) // 20 * 20
    if b < 1000:
        return (b + 49) // 50 * 50
    return (b + 99) // 100 * 100


def normalize_mass(mass: np.ndarray) -> np.ndarray:
    m = mass
    m_avg = np.ones_like(m) * (np.sum(np.sqrt(m**2)) / m.shape[0])
    m_std = float(np.std(m))
    if m_std < 1e-4:
        return np.full((m.shape[0],), 225.0)
    m_score = (m - m_avg) / m_std
    m_score[np.sign(m_score) * (10 * m_score) ** 2 < -144] = -144
    return (225 + np.sign(m_score) * (10 * m_score) ** 2).reshape(-1)


def animate(result: Result, masses: np.ndarray, fps: int = 100, view_lim: int = 20, autoscroll: bool = False, replay: bool = True) -> None:
    _require_matplotlib()
    Tp1, n, _ = result.state_history.shape
    T = Tp1 - 1

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 10), dpi=80)
    fig.suptitle('N-Body Dynamics', fontsize=26)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)

    ax1 = plt.subplot(grid[0:2, 0])
    ax1.set(xlim=(-view_lim, view_lim), ylim=(-view_lim, view_lim))
    ax1.set_axis_off()

    yl = energy_bounds(result.ke, result.pe)

    ax2 = plt.subplot(grid[2, 0])
    ax2.set(xlim=(0, T), ylim=(-yl, yl))
    ax2.set_xticks([0, T/4, T/2, 3*T/4, T])
    ax2.set_yticks([-yl, -yl/2, 0, yl/2, yl])
    ax2.set_xlabel("Time [dt]")
    ax2.set_ylabel("Energy")

    ti = 100
    cs = ['tab:blue','tab:red','tab:green','tab:purple','tab:orange','tab:pink','tab:cyan']
    cs_pts = [cs[j % len(cs)] for j in range(n)]
    cs_tail = [cs_pts[j % len(cs_pts)] for j in range(n) for _ in range(ti)]

    s = np.linspace(8, 16, ti).reshape((ti,))
    s_tail = np.vstack(tuple([s for _ in range(n)]))
    s_pts = normalize_mass(masses)

    trails = ax1.scatter([], [], c=cs_tail, s=s_tail, zorder=1)
    pts = ax1.scatter([], [], c=cs_pts, s=s_pts, zorder=2)

    (KE_line,) = ax2.plot([], [], 'b', lw=2)
    (PE_line,) = ax2.plot([], [], 'darkorange', lw=2)
    (E_line,) = ax2.plot([], [], 'lawngreen', lw=2)

    state = result.state_history

    def init():
        KE_line.set_data([], [])
        PE_line.set_data([], [])
        E_line.set_data([], [])
        return pts, trails, KE_line, PE_line, E_line

    KE_step = []
    PE_step = []
    E_step = []

    def animate_i(i):
        nonlocal KE_step, PE_step, E_step
        # tails
        ti_loc = ti
        if i - ti_loc < 0:
            trail_i = np.vstack(
                tuple([
                    np.vstack((state[0, j, 0:2] * np.ones((ti_loc - i, 2)), state[max(i - ti_loc, 0):i, j, 0:2]))
                    for j in range(n)
                ])
            )
        else:
            trail_i = np.vstack(tuple([state[max(i - ti_loc, 0):i, j, 0:2] for j in range(n)]))
        trails.set_offsets(trail_i)

        # points
        pts_i = state[i, :, 0:2]
        pts.set_offsets(pts_i)

        KE = float(result.ke[i])
        PE = float(result.pe[i])
        E = KE + PE
        KE_step.append(KE)
        PE_step.append(PE)
        E_step.append(E)
        t_steps = np.linspace(0, i, len(E_step))

        KE_line.set_data(t_steps, KE_step)
        PE_line.set_data(t_steps, PE_step)
        E_line.set_data(t_steps, E_step)

        if autoscroll:
            xy = pts_i
            xymax = max(np.max(xy[:, 1]), np.max(xy[:, 0]))
            xymax_lim = max(view_lim, max(1.1 * xymax, xymax + 5))
            xymin = min(np.min(xy[:, 1]), np.min(xy[:, 0]))
            xymin_lim = min(-view_lim, min(-1.1 * xymin, xymin - 5))
            ax1.set(xlim=(xymin_lim, xymax_lim), ylim=(xymin_lim, xymax_lim))

        return pts, trails, KE_line, PE_line, E_line

    ani = animation.FuncAnimation(fig, animate_i, frames=T+1, interval=1, blit=True, init_func=init, repeat=False)
    ax2.legend(("KE", "PE", "E"))

    if replay:
        plt.show()

    return None
