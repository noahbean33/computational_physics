"""
Visualization utilities for physics simulations and experiments.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib import cm

def setup_plot(title: str = None, xlabel: str = None, ylabel: str = None, 
              grid: bool = True, figsize: Tuple[float, float] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """Set up a matplotlib plot with common settings.
    
    Args:
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        grid: Whether to show grid
        figsize: Figure size (width, height)
        
    Returns:
        Tuple of (figure, axes) objects
    """
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=figsize)
    
    if title:
        ax.set_title(title, fontsize=14, pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig, ax


def maybe_save_or_show(path: Optional[str], show: bool = True, tight: bool = True, dpi: int = 150, close: bool = True) -> None:
    """Save current matplotlib figure to path or show it.

    Args:
        path: If provided, save figure to this path (directories created as needed).
        show: If True and path is None, show the figure interactively.
        tight: If True, apply tight_layout before save/show.
        dpi: DPI used when saving.
        close: If True and saving, close the figure to free memory.
    """
    if tight:
        try:
            plt.tight_layout()
        except Exception:
            pass
    if path:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=dpi)
        if close:
            plt.close()
    elif show:
        plt.show()

def plot_phase_space(trajectories: List[np.ndarray], labels: List[str] = None, 
                    xlabel: str = 'Position', ylabel: str = 'Momentum',
                    title: str = 'Phase Space Trajectory') -> Tuple[plt.Figure, plt.Axes]:
    """Plot phase space trajectories.
    
    Args:
        trajectories: List of 2D arrays where each row is [position, momentum]
        labels: Optional list of legend labels for each trajectory
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title: Plot title
        
    Returns:
        Tuple of (figure, axes) objects
    """
    fig, ax = setup_plot(title=title, xlabel=xlabel, ylabel=ylabel)
    
    for i, traj in enumerate(trajectories):
        if len(traj.shape) == 1:
            traj = traj.reshape(-1, 1)
        label = labels[i] if labels and i < len(labels) else f'Trajectory {i+1}'
        ax.plot(traj[:, 0], traj[:, 1], '.-', markersize=4, label=label)
    
    if labels:
        ax.legend()
    
    return fig, ax

def plot_3d_trajectory(points: np.ndarray, color_map: str = 'viridis', 
                       title: str = '3D Trajectory') -> Tuple[plt.Figure, Axes3D]:
    """Plot a 3D trajectory with color indicating time/progression.
    
    Args:
        points: Nx3 array of (x, y, z) coordinates
        color_map: Colormap for the trajectory
        title: Plot title
        
    Returns:
        Tuple of (figure, axes) objects
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create color map based on time/progression
    n_points = len(points)
    colors = cm.get_cmap(color_map)(np.linspace(0, 1, n_points))
    
    # Plot each segment with appropriate color
    for i in range(n_points - 1):
        ax.plot(points[i:i+2, 0], points[i:i+2, 1], points[i:i+2, 2], 
                color=colors[i], linewidth=1.5)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=color_map, 
                             norm=plt.Normalize(vmin=0, vmax=n_points-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('Time Step')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    return fig, ax

def animate_2d_trajectory(x: np.ndarray, y: np.ndarray, 
                         interval: int = 50, frames: int = None,
                         title: str = 'Animated Trajectory') -> FuncAnimation:
    """Create an animation of a 2D trajectory.
    
    Args:
        x: X-coordinates over time
        y: Y-coordinates over time
        interval: Delay between frames in milliseconds
        frames: Number of frames to show (default: all)
        title: Plot title
        
    Returns:
        Animation object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    line, = ax.plot([], [], 'b-', lw=2)
    point, = ax.plot([], [], 'ro')
    
    # Set up the plot
    ax.set_xlim(min(x) - 0.1, max(x) + 0.1)
    ax.set_ylim(min(y) - 0.1, max(y) + 0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.grid(True)
    
    # Initialize the animation
    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point
    
    # Animation update function
    def update(frame):
        line.set_data(x[:frame+1], y[:frame+1])
        point.set_data(x[frame], y[frame])
        return line, point
    
    # Create the animation
    frames = frames or len(x)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init,
                        blit=True, interval=interval, repeat=True)
    
    return ani

def plot_heatmap(data: np.ndarray, x: np.ndarray = None, y: np.ndarray = None,
                xlabel: str = 'X', ylabel: str = 'Y', title: str = 'Heatmap',
                cmap: str = 'viridis', colorbar_label: str = 'Intensity') -> Tuple[plt.Figure, plt.Axes]:
    """Create a heatmap from 2D data.
    
    Args:
        data: 2D array of values to plot
        x: X-axis values (optional)
        y: Y-axis values (optional)
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title: Plot title
        cmap: Colormap to use
        colorbar_label: Label for the colorbar
        
    Returns:
        Tuple of (figure, axes) objects
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create the heatmap
    im = ax.imshow(data, cmap=cmap, aspect='auto', 
                  extent=(x[0] if x is not None else 0, 
                          x[-1] if x is not None else data.shape[1],
                          y[-1] if y is not None else data.shape[0],
                          y[0] if y is not None else 0) if x is not None and y is not None else None)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    return fig, ax
