
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import LogLocator, FormatStrFormatter
from adjustText import adjust_text

def scale_sizes(values, min_size=50, max_size=500):
    val_array = np.array(values)
    # Log scale the values first since they span orders of magnitude
    log_vals = np.log10(val_array)
    min_val = np.min(log_vals)
    max_val = np.max(log_vals)
    
    if max_val == min_val:
        return np.full(val_array.shape, (min_size + max_size) / 2)
        
    normalized = (log_vals - min_val) / (max_val - min_val)
    return normalized * (max_size - min_size) + min_size

def main():
    # Load data
    json_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'benchmark_results.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    names = [d['dataset'] for d in data]
    nodes = [d['nodes'] for d in data]
    edges = [d['edges'] for d in data]
    runtimes = [d['time'] for d in data]
    
    node_sizes = scale_sizes(nodes)
    edge_sizes = scale_sizes(edges)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    # Plot 1: Runtime vs Edges
    scatter_edges = axes[0].scatter(
        edges,
        runtimes,
        s=node_sizes,
        c=np.log10(nodes),
        cmap='viridis',
        alpha=0.85,
        edgecolors='k',
        linewidths=0.4,
        label='Datasets'
    )
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    # Use LogLocator to show more ticks (1, 2, 5)
    axes[0].yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0]))
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    axes[0].set_xlabel('Number of Edges |E|', fontsize=30)
    axes[0].set_ylabel('Time per Subgraph (s)', fontsize=30)
    # axes[0].set_title('Runtime vs Edges (Node Count Encoded)')
    axes[0].grid(True, which="major", ls="--", alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    cbar_edges = fig.colorbar(scatter_edges, ax=axes[0])
    cbar_edges.set_label('log10(|V|)', fontsize=30)
    cbar_edges.ax.tick_params(labelsize=30) 
    cbar_edges.outline.set_visible(False)
    axes[0].tick_params(axis='both', which='major', labelsize=30)

    # Plot 2: Runtime vs Nodes
    scatter_nodes = axes[1].scatter(
        nodes,
        runtimes,
        s=edge_sizes,
        c=np.log10(edges),
        cmap='plasma',
        alpha=0.85,
        edgecolors='k',
        linewidths=0.4,
        label='Datasets'
    )
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0]))
    axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    axes[1].set_xlabel('Number of Nodes |V|', fontsize=30)
    # axes[1].set_ylabel('Time per Subgraph (s)', fontsize=20)
    # axes[1].set_title('Runtime vs Nodes (Edge Count Encoded)', fontsize=20)
    axes[1].grid(True, which="major", ls="--", alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].tick_params(axis='both', which='major', labelsize=30)
    cbar_nodes = fig.colorbar(scatter_nodes, ax=axes[1])
    cbar_nodes.set_label('log10(|E|)', fontsize=30)
    cbar_nodes.ax.tick_params(labelsize=30)
    cbar_nodes.outline.set_visible(False)

    # Annotations
    # for i, txt in enumerate(names):
    #     xytext = (0, 10) if i % 2 == 0 else (0, -15)
    #     axes[0].annotate(txt, (edges[i], runtimes[i]), textcoords="offset points", xytext=xytext, ha='center', fontsize=20)
    #     axes[1].annotate(txt, (nodes[i], runtimes[i]), textcoords="offset points", xytext=xytext, ha='center', fontsize=20)
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'runtime_scaling_edges_nodes.pdf'), dpi=300)
    print(f"Plot saved to {os.path.join(output_dir, 'runtime_scaling_edges_nodes.pdf')}")

if __name__ == "__main__":
    main()