"""
Erdos-Renyi tutorial: phase transitions and graph properties, with CLI.

Refactored from a notebook export into a reusable, import-safe script.
"""

import argparse
import os
import math
from typing import Iterable, Optional

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from src.visualization.plot_utils import maybe_save_or_show


def _maybe_save_or_show(path: Optional[str] = None):
    # Backward-compatible wrapper; delegate to shared util
    maybe_save_or_show(path)

def generate_and_analyze_erdos_renyi(n_values: Iterable[int], p_values: Iterable[float], save_prefix: Optional[str] = None, seed: int = 42):
    """
    Generates and analyzes Erdős-Rényi graphs for various n and p values.

    Args:
        n_values (list): A list of integers for the number of nodes.
        p_values (list): A list of floats for the edge probability.
        save_prefix (str|None): If set, save figures using this prefix.
        seed (int): Random seed for layout/graphs.
    """
    for n in n_values:
        for p in p_values:
            # Generate the Erdős-Rényi graph
            g = nx.erdos_renyi_graph(n, p, seed=seed)

            # --- Visualization ---
            plt.figure(figsize=(8, 6))
            pos = nx.spring_layout(g, seed=seed)  # for consistent layout
            nx.draw(g, pos, with_labels=True, node_color='lightblue',
                    node_size=500, font_size=10, edge_color='gray')
            plt.title(f"Erdős-Rényi Graph: n={n}, p={p:.3f}")
            _maybe_save_or_show(None if not save_prefix else f"{save_prefix}_n{n}_p{p:.3f}.png")

            # --- Property Analysis ---
            print(f"\n--- Analysis for n={n}, p={p:.3f} ---")

            # Connected Components
            num_components = nx.number_connected_components(g)
            print(f"Number of Connected Components: {num_components}")

            if num_components > 0:
                largest_component = max(nx.connected_components(g), key=len)
                largest_component_size = len(largest_component)
                print(f"Size of Largest Connected Component: {largest_component_size}")
            else:
                print("Size of Largest Connected Component: 0")

            # Average Clustering Coefficient
            avg_clustering = nx.average_clustering(g)
            print(f"Average Clustering Coefficient: {avg_clustering:.4f}")

            # Density
            density = nx.density(g)
            print(f"Density: {density:.4f}")


## Top-level execution replaced by CLI below

## Imports consolidated above

# --- Helper Functions ---

def analyze_and_plot(g, n, p, title, highlight_nodes=None, save_path: Optional[str] = None):
    """A helper function to plot graphs and print basic properties."""
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(g, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(g, pos, node_color='lightblue', node_size=500)

    # Draw highlighted nodes if any
    if highlight_nodes:
        nx.draw_networkx_nodes(g, pos, nodelist=highlight_nodes, node_color='tomato', node_size=500)

    # Draw edges and labels
    nx.draw_networkx_edges(g, pos, edge_color='gray')
    nx.draw_networkx_labels(g, pos, font_size=10)

    plt.title(f"{title}\nGraph: n={n}, p={p:.4f}", fontsize=14)
    _maybe_save_or_show(save_path)

    print(f"Properties for n={n}, p={p:.4f}:")
    print(f"  - Is connected? {nx.is_connected(g)}")
    print(f"  - Number of components: {nx.number_connected_components(g)}")
    if g.number_of_nodes() > 0:
        min_deg = min(d for _, d in g.degree())
        print(f"  - Minimum degree: {min_deg}")
    if g.number_of_edges() > 0:
        is_acyclic = nx.is_forest(g)
        print(f"  - Is acyclic (a forest)? {is_acyclic}")
    print("-" * 30)

# --- Feature Demonstrations ---

## 1. Connectivity Threshold 📈
def demo_connectivity_threshold(n=50, seed=42, save_prefix: Optional[str] = None):
    """Demonstrates the sharp transition to a connected graph."""
    print("## 1. Connectivity Threshold Demonstration ##")
    p_crit = math.log(n) / n

    print(f"For n={n}, the connectivity threshold is p ≈ {p_crit:.4f}\n")

    # Below threshold
    p_below = p_crit - 0.04
    g_below = nx.erdos_renyi_graph(n, p_below, seed=seed)
    analyze_and_plot(g_below, n, p_below, "Below Connectivity Threshold", save_path=None if not save_prefix else f"{save_prefix}_connectivity_below.png")

    # Above threshold
    p_above = p_crit + 0.04
    g_above = nx.erdos_renyi_graph(n, p_above, seed=seed)
    analyze_and_plot(g_above, n, p_above, "Above Connectivity Threshold", save_path=None if not save_prefix else f"{save_prefix}_connectivity_above.png")

## 2. Giant Component Emergence 🧱
def demo_giant_component(n=100, seed=42, save_prefix: Optional[str] = None):
    """Shows the emergence of a giant component."""
    print("\n## 2. Giant Component Emergence Demonstration ##")
    p_crit = 1 / n

    print(f"For n={n}, the giant component threshold is p ≈ {p_crit:.4f}\n")

    # Below threshold
    p_below = p_crit / 2
    g_below = nx.erdos_renyi_graph(n, p_below, seed=seed)
    largest_comp_nodes = max(nx.connected_components(g_below), key=len)
    analyze_and_plot(g_below, n, p_below, "Below Giant Component Threshold", highlight_nodes=list(largest_comp_nodes), save_path=None if not save_prefix else f"{save_prefix}_giant_below.png")
    print(f"Size of largest component: {len(largest_comp_nodes)} ({len(largest_comp_nodes)/n:.1%})")

    # Above threshold
    p_above = p_crit * 2
    g_above = nx.erdos_renyi_graph(n, p_above, seed=seed)
    largest_comp_nodes = max(nx.connected_components(g_above), key=len)
    analyze_and_plot(g_above, n, p_above, "Above Giant Component Threshold", highlight_nodes=list(largest_comp_nodes), save_path=None if not save_prefix else f"{save_prefix}_giant_above.png")
    print(f"Size of largest component: {len(largest_comp_nodes)} ({len(largest_comp_nodes)/n:.1%})")

## 3. Appearance of Cycles 🌀
def demo_cycles(n=100, seed=42, save_prefix: Optional[str] = None):
    """Shows when cycles start to appear."""
    print("\n## 3. Appearance of Cycles Demonstration ##")
    p_crit = 1 / n

    print(f"For n={n}, the cycle appearance threshold is p ≈ {p_crit:.4f}\n")

    # Below threshold (likely a forest)
    p_below = p_crit * 0.5
    g_below = nx.erdos_renyi_graph(n, p_below, seed=seed)
    analyze_and_plot(g_below, n, p_below, "Below Cycle Threshold (Likely Acyclic)", save_path=None if not save_prefix else f"{save_prefix}_cycles_below.png")

    # Above threshold (cycles appear)
    p_above = p_crit * 1.5
    g_above = nx.erdos_renyi_graph(n, p_above, seed=seed)
    try:
        cycle = nx.find_cycle(g_above)
        cycle_nodes = {node for edge in cycle for node in edge}
        analyze_and_plot(g_above, n, p_above, "Above Cycle Threshold", highlight_nodes=list(cycle_nodes), save_path=None if not save_prefix else f"{save_prefix}_cycles_above.png")
        print(f"Found a cycle: {cycle}")
    except nx.NetworkXNoCycle:
        analyze_and_plot(g_above, n, p_above, "Above Cycle Threshold (No cycle found in this instance)", save_path=None if not save_prefix else f"{save_prefix}_cycles_above.png")

## 4. Minimum Degree ≥ 1 🔗
def demo_min_degree_one(n=50, seed=42, save_prefix: Optional[str] = None):
    """Shows when all nodes become connected (no isolates)."""
    print("\n## 4. Minimum Degree ≥ 1 Demonstration ##")
    # This happens at the same threshold as connectivity.
    p_crit = math.log(n) / n

    print(f"For n={n}, the threshold for min degree ≥ 1 is p ≈ {p_crit:.4f}\n")

    # Below threshold (isolates likely)
    p_below = p_crit - 0.05
    g_below = nx.erdos_renyi_graph(n, p_below, seed=seed)
    isolates = list(nx.isolates(g_below))
    analyze_and_plot(g_below, n, p_below, "Below Min Degree ≥ 1 Threshold", highlight_nodes=isolates, save_path=None if not save_prefix else f"{save_prefix}_mindeg1_below.png")

    # Above threshold (no isolates likely)
    p_above = p_crit + 0.05
    g_above = nx.erdos_renyi_graph(n, p_above, seed=seed)
    analyze_and_plot(g_above, n, p_above, "Above Min Degree ≥ 1 Threshold", save_path=None if not save_prefix else f"{save_prefix}_mindeg1_above.png")

## 5. Minimum Degree ≥ k 🔒
def demo_min_degree_k(n=100, k=3, seed=42, save_prefix: Optional[str] = None):
    """Shows when minimum degree reaches k."""
    print(f"\n## 5. Minimum Degree ≥ {k} Demonstration ##")
    p_crit = (math.log(n) + (k - 1) * math.log(math.log(n))) / n

    print(f"For n={n}, k={k}, the threshold for min degree ≥ {k} is p ≈ {p_crit:.4f}\n")

    # Below threshold
    p_below = p_crit - 0.015
    g_below = nx.erdos_renyi_graph(n, p_below, seed=seed)
    analyze_and_plot(g_below, n, p_below, f"Below Min Degree ≥ {k} Threshold", save_path=None if not save_prefix else f"{save_prefix}_mindeg{k}_below.png")

    # Above threshold
    p_above = p_crit + 0.015
    g_above = nx.erdos_renyi_graph(n, p_above, seed=seed)
    analyze_and_plot(g_above, n, p_above, f"Above Min Degree ≥ {k} Threshold", save_path=None if not save_prefix else f"{save_prefix}_mindeg{k}_above.png")

## 6. k-Core Emergence 🧠
def demo_k_core(n=100, k=3, seed=42, save_prefix: Optional[str] = None):
    """Shows the emergence of the k-core."""
    print(f"\n## 6. {k}-Core Emergence Demonstration ##")
    # Threshold for k=3 is approx p = c_k/n where c_k ~ 3.35
    p_core = 3.35 / n

    print(f"For n={n}, the {k}-core emerges at a higher p value.")

    g = nx.erdos_renyi_graph(n, p_core + 0.02, seed=seed)
    core = nx.k_core(g, k=k)

    print(f"Generated a graph with p={p_core + 0.02:.4f}")
    analyze_and_plot(g, n, p_core + 0.02, "Original Graph for k-Core Analysis", save_path=None if not save_prefix else f"{save_prefix}_kcore_graph.png")
    analyze_and_plot(core, core.number_of_nodes(), p_core + 0.02, f"Resulting {k}-Core Subgraph", save_path=None if not save_prefix else f"{save_prefix}_kcore_subgraph.png")
    print(f"The k-core is a concept of 'freezing' or 'rigidity', where a stable, highly interconnected subgraph emerges.")

## 7. Chromatic Number (Graph Coloring) 🎨
def demo_chromatic_number(n=100, p=0.15, seed=42, save_prefix: Optional[str] = None):
    """Estimates the chromatic number."""
    print("\n## 7. Chromatic Number Demonstration ##")
    g = nx.erdos_renyi_graph(n, p, seed=seed)

    # Greedy coloring provides an upper bound on the chromatic number
    coloring = nx.greedy_color(g, strategy="largest_first")
    chromatic_num_estimate = max(coloring.values()) + 1

    # Theoretical approximation
    b = 1 / (1 - p)
    theory_chi = n / (2 * math.log(n, b))

    print(f"For n={n}, p={p}:")
    print(f"  - Greedy coloring estimate (χ(G)): {chromatic_num_estimate}")
    print(f"  - Theoretical approximation: {theory_chi:.2f}")

    # Plot with colors
    node_colors = [coloring[node] for node in g.nodes()]
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(g, seed=42)
    nx.draw(g, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.jet, font_size=8)
    plt.title(f"Greedy Coloring (Used {chromatic_num_estimate} colors)\nn={n}, p={p:.4f}")
    _maybe_save_or_show(None if not save_prefix else f"{save_prefix}_chromatic.png")

## 8. Clique Number (Maximum Clique Size) 🏰
def demo_clique_number(n=100, k=5, seed=42, save_prefix: Optional[str] = None):
    """Shows the threshold for the appearance of a k-clique."""
    print(f"\n## 8. K_{k} (Clique of size {k}) Emergence Demonstration ##")
    p_crit = n**(-2 / (k - 1))

    print(f"For n={n}, the threshold for a K_{k} to appear is p ≈ {p_crit:.4f}\n")

    # Below threshold
    p_below = p_crit / 2
    g_below = nx.erdos_renyi_graph(n, p_below, seed=42)
    cliques_below = list(nx.find_cliques(g_below))
    max_clique_below = max(len(c) for c in cliques_below) if cliques_below else 0
    print(f"At p={p_below:.4f}, max clique found: {max_clique_below}")

    # Above threshold
    p_above = p_crit * 2
    g_above = nx.erdos_renyi_graph(n, p_above, seed=42)
    cliques_above = list(nx.find_cliques(g_above))
    max_clique_above = max(len(c) for c in cliques_above) if cliques_above else 0
    print(f"At p={p_above:.4f}, max clique found: {max_clique_above}")

    if max_clique_above >= k:
        k_clique_nodes = [c for c in cliques_above if len(c) == max_clique_above][0]
        analyze_and_plot(g_above, n, p_above, f"Above K_{k} Threshold (Found a {max_clique_above}-clique)", highlight_nodes=k_clique_nodes, save_path=None if not save_prefix else f"{save_prefix}_clique_above.png")
    else:
        print(f"A K_{k} was not found in this specific random instance, but is now probable.")

## 9. Hamiltonian Cycle Emergence 🧱
def demo_hamiltonian_cycle(n=50, seed=42, save_prefix: Optional[str] = None):
    """Checks a necessary condition for Hamiltonicity."""
    print("\n## 9. Hamiltonian Cycle Emergence Demonstration ##")
    # Threshold is the same as connectivity
    p_crit = (math.log(n) + math.log(math.log(n))) / n # A slightly stronger version
    print(f"For n={n}, Hamiltonicity threshold is p ≈ {p_crit:.4f}\n")

    print("A necessary (but not sufficient) condition for a Hamiltonian cycle is a minimum degree of at least 2.")

    # Below
    p_below = p_crit - 0.05
    g_below = nx.erdos_renyi_graph(n, p_below, seed=seed)
    analyze_and_plot(g_below, n, p_below, "Below Hamiltonian Threshold", save_path=None if not save_prefix else f"{save_prefix}_ham_below.png")

    # Above
    p_above = p_crit + 0.05
    g_above = nx.erdos_renyi_graph(n, p_above, seed=seed)
    analyze_and_plot(g_above, n, p_above, "Above Hamiltonian Threshold", save_path=None if not save_prefix else f"{save_prefix}_ham_above.png")
    print("Above this threshold, the minimum degree is almost surely ≥ 2, and the graph is almost surely Hamiltonian.")

## 10. Planarity Disappears 🧮
def demo_planarity(n=50, seed=42, save_prefix: Optional[str] = None):
    """Shows when the graph becomes non-planar."""
    print("\n## 10. Planarity Disappearance Demonstration ##")
    p_crit = 1 / n
    print(f"For n={n}, planarity disappears around p ≈ {p_crit:.4f}\n")

    # Below
    p_below = p_crit * 0.8
    g_below = nx.erdos_renyi_graph(n, p_below, seed=seed)
    is_planar_below, _ = nx.check_planarity(g_below)
    analyze_and_plot(g_below, n, p_below, "Below Planarity Threshold", save_path=None if not save_prefix else f"{save_prefix}_planar_below.png")
    print(f"Is the graph planar? {is_planar_below}")

    # Above
    p_above = p_crit * 1.5
    g_above = nx.erdos_renyi_graph(n, p_above, seed=seed)
    is_planar_above, _ = nx.check_planarity(g_above)
    analyze_and_plot(g_above, n, p_above, "Above Planarity Threshold", save_path=None if not save_prefix else f"{save_prefix}_planar_above.png")
    print(f"Is the graph planar? {is_planar_above}")

## 12. Spectral Transitions 🔬
def demo_spectral_gap(n=50, seed=42, save_prefix: Optional[str] = None):
    """Shows the spectral gap opening as the graph connects."""
    print("\n## 12. Spectral Transition (Fiedler Value) Demonstration ##")
    p_values = np.linspace(0.01, 0.2, 25)
    fiedler_values = []

    for p in p_values:
        g = nx.erdos_renyi_graph(n, p, seed=seed)
        if nx.is_connected(g):
            # The Fiedler value (algebraic connectivity) is the 2nd smallest eigenvalue
            # of the Laplacian matrix. It's > 0 if and only if the graph is connected.
            laplacian_eigenvalues = nx.laplacian_spectrum(g)
            fiedler_values.append(laplacian_eigenvalues[1])
        else:
            fiedler_values.append(0)

    p_crit = math.log(n) / n

    plt.figure(figsize=(10, 6))
    plt.plot(p_values, fiedler_values, 'bo-', label='Fiedler Value (Algebraic Connectivity)')
    plt.axvline(x=p_crit, color='r', linestyle='--', label=f'Connectivity Threshold (p ≈ {p_crit:.3f})')
    plt.title(f'Spectral Gap vs. Edge Probability p (n={n})', fontsize=14)
    plt.xlabel('Edge Probability (p)')
    plt.ylabel('Fiedler Value (λ₂)')
    plt.legend()
    plt.grid(True)
    _maybe_save_or_show(None if not save_prefix else f"{save_prefix}_spectral_gap.png")
    print("The plot shows the 'spectral gap' opening up. The Fiedler value becomes non-zero precisely when the graph connects.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Erdos-Renyi tutorial demos")
    p.add_argument('--sweep', action='store_true', help='Run n,p sweep visualization')
    p.add_argument('--n_list', type=str, default='30', help='Comma-separated n values for sweep')
    p.add_argument('--p_list', type=str, default='0.01,0.05,0.1,0.2', help='Comma-separated p values for sweep')

    p.add_argument('--connectivity', action='store_true', help='Run connectivity threshold demo')
    p.add_argument('--giant', action='store_true', help='Run giant component demo')
    p.add_argument('--cycles', action='store_true', help='Run cycles demo')
    p.add_argument('--mindeg1', action='store_true', help='Run minimum degree ≥ 1 demo')
    p.add_argument('--mindegk', action='store_true', help='Run minimum degree ≥ k demo')
    p.add_argument('--kcore', action='store_true', help='Run k-core demo')
    p.add_argument('--chromatic', action='store_true', help='Run chromatic number demo')
    p.add_argument('--clique', action='store_true', help='Run clique number demo')
    p.add_argument('--hamiltonian', action='store_true', help='Run Hamiltonian cycle threshold demo')
    p.add_argument('--planarity', action='store_true', help='Run planarity demo')
    p.add_argument('--spectral', action='store_true', help='Run spectral gap demo')

    p.add_argument('--n', type=int, default=100, help='Generic n for demos')
    p.add_argument('--p', type=float, default=0.15, help='Generic p for demos')
    p.add_argument('--k', type=int, default=3, help='Generic k for demos')
    p.add_argument('--seed', type=int, default=42, help='Random seed')

    p.add_argument('--outdir', type=str, default='', help='If set, save figures here; otherwise show')
    p.add_argument('--prefix', type=str, default='erdos_renyi', help='Filename prefix for saved figures')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    save = bool(args.outdir)
    def sp(name: str) -> Optional[str]:
        return os.path.join(args.outdir, f"{args.prefix}_{name}.png") if save else None

    run_all = not any([
        args.sweep, args.connectivity, args.giant, args.cycles, args.mindeg1,
        args.mindegk, args.kcore, args.chromatic, args.clique, args.hamiltonian,
        args.planarity, args.spectral,
    ])

    if args.sweep or run_all:
        n_vals = [int(x) for x in args.n_list.split(',') if x.strip()]
        p_vals = [float(x) for x in args.p_list.split(',') if x.strip()]
        save_prefix = None if not save else os.path.join(args.outdir, f"{args.prefix}_sweep")
        generate_and_analyze_erdos_renyi(n_vals, p_vals, save_prefix=save_prefix, seed=args.seed)

    if args.connectivity or run_all:
        demo_connectivity_threshold(n=args.n, seed=args.seed, save_prefix=None if not save else os.path.join(args.outdir, f"{args.prefix}"))

    if args.giant or run_all:
        demo_giant_component(n=args.n, seed=args.seed, save_prefix=None if not save else os.path.join(args.outdir, f"{args.prefix}"))

    if args.cycles or run_all:
        demo_cycles(n=args.n, seed=args.seed, save_prefix=None if not save else os.path.join(args.outdir, f"{args.prefix}"))

    if args.mindeg1 or run_all:
        demo_min_degree_one(n=args.n, seed=args.seed, save_prefix=None if not save else os.path.join(args.outdir, f"{args.prefix}"))

    if args.mindegk or run_all:
        demo_min_degree_k(n=args.n, k=args.k, seed=args.seed, save_prefix=None if not save else os.path.join(args.outdir, f"{args.prefix}"))

    if args.kcore or run_all:
        demo_k_core(n=args.n, k=args.k, seed=args.seed, save_prefix=None if not save else os.path.join(args.outdir, f"{args.prefix}"))

    if args.chromatic or run_all:
        demo_chromatic_number(n=args.n, p=args.p, seed=args.seed, save_prefix=None if not save else os.path.join(args.outdir, f"{args.prefix}"))

    if args.clique or run_all:
        demo_clique_number(n=args.n, k=args.k, seed=args.seed, save_prefix=None if not save else os.path.join(args.outdir, f"{args.prefix}"))

    if args.hamiltonian or run_all:
        demo_hamiltonian_cycle(n=args.n, seed=args.seed, save_prefix=None if not save else os.path.join(args.outdir, f"{args.prefix}"))

    if args.planarity or run_all:
        demo_planarity(n=args.n, seed=args.seed, save_prefix=None if not save else os.path.join(args.outdir, f"{args.prefix}"))

    if args.spectral or run_all:
        demo_spectral_gap(n=args.n, seed=args.seed, save_prefix=None if not save else os.path.join(args.outdir, f"{args.prefix}"))


if __name__ == "__main__":
    main()