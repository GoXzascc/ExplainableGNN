
import torch
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian
from torch_geometric.nn import GCNConv
import numpy as np
import matplotlib.pyplot as plt
import os
from src.corasen import spectral_graph_coarsening
from src.benchmark_runtime import get_dataset, to_undirected

def compute_dirichlet_energy(x, edge_index, num_nodes):
    """
    Computes Dirichlet Energy E(X) = 1/N * Trace(X^T L X)
    Using normalized Laplacian for consistency.
    """
    device = x.device
    edge_index_L, edge_weight_L = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)
    L = torch.sparse_coo_tensor(edge_index_L, edge_weight_L, (num_nodes, num_nodes)).to(device)
    
    # Trace(X^T L X) = sum( (L @ X) * X )
    LX = torch.sparse.mm(L, x)
    energy = torch.sum(LX * x) / num_nodes
    return energy.item()

def run_metrics_benchmark():
    datasets = [
        'Cora', 'CiteSeer', 'PubMed',
        'Amazon-Photo', 'Amazon-Computers',
        'Coauthor-CS', 'Coauthor-Physics'
    ]
    
    num_runs = 10
    layers = 5  # Deep enough to see oversmoothing
    hidden = 64
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = {}
    
    for name in datasets:
        print(f"\nProcessing {name}...")
        try:
            data, _ = get_dataset(name)
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            continue
            
        data = data.to(device)
        if not data.is_undirected():
            data.edge_index = to_undirected(data.edge_index)
            
        print(f"  Nodes: {data.num_nodes}, Edges: {data.edge_index.size(1)}")
        
        # Pre-compute Coarsened Graph ONCE (structure doesn't change per run)
        # Using standard settings from paper/benchmark
        print("  Coarsening...")
        try:
            coarse_data, _, _ = spectral_graph_coarsening(data, k=20, alpha=0.5)
            print(f"  Coarse Graph: {coarse_data.num_nodes} nodes")
        except Exception as e:
            print(f"  Coarsening failed: {e}")
            continue
            
        # Storage for runs
        # Shape: (num_runs, layers + 1)
        energy_orig_matrix = np.zeros((num_runs, layers + 1))
        energy_coarse_matrix = np.zeros((num_runs, layers + 1))
        
        for run in range(num_runs):
            # print(f"    Run {run+1}/{num_runs}")
            
            # Re-init model for each run to average out weight initialization effects
            # We want to measure the ARCHITECTURE/GRAPH effect, not specific weight luck.
            model = torch.nn.Sequential(
                GCNConv(data.num_features, hidden),
                torch.nn.ReLU(),
            )
            for _ in range(layers - 1):
                model.add_module(f"conv_{_}", GCNConv(hidden, hidden))
                model.add_module(f"act_{_}", torch.nn.ReLU())
            
            model.to(device)
            conv_layers = [m for m in model if isinstance(m, GCNConv)]
            
            # --- Original ---
            curr_x = data.x
            energy_orig_matrix[run, 0] = compute_dirichlet_energy(curr_x, data.edge_index, data.num_nodes)
            
            with torch.no_grad():
                for idx, conv in enumerate(conv_layers):
                    curr_x = conv(curr_x, data.edge_index)
                    curr_x = F.relu(curr_x)
                    energy_orig_matrix[run, idx+1] = compute_dirichlet_energy(curr_x, data.edge_index, data.num_nodes)

            # --- Coarse ---
            # Reset model? Ideally we compare "A GCN on Orig" vs "A GCN on Coarse". 
            # We can use the SAME weights (if dimensions match) or different random weights.
            # Since Coarse features X' are aggregated from X, dimensions match (if X' is same dim).
            # Yes, spectral coarsening preserves feature dim.
            # BUT, to be fair, usually we train/eval models independently.
            # Using SAME weights isolates graph structure effect from weight initialization diffs.
            # Let's use SAME weights.
            
            curr_x_c = coarse_data.x
            energy_coarse_matrix[run, 0] = compute_dirichlet_energy(curr_x_c, coarse_data.edge_index, coarse_data.num_nodes)
            
            with torch.no_grad():
                for idx, conv in enumerate(conv_layers):
                    curr_x_c = conv(curr_x_c, coarse_data.edge_index)
                    curr_x_c = F.relu(curr_x_c)
                    energy_coarse_matrix[run, idx+1] = compute_dirichlet_energy(curr_x_c, coarse_data.edge_index, coarse_data.num_nodes)
        
        # Aggregate
        results[name] = {
            'orig_mean': np.mean(energy_orig_matrix, axis=0),
            'orig_std': np.std(energy_orig_matrix, axis=0),
            'coarse_mean': np.mean(energy_coarse_matrix, axis=0),
            'coarse_std': np.std(energy_coarse_matrix, axis=0)
        }
        
    # --- Plotting ---
    print("\nPlotting combined results...")
    # 7 datasets. Grid 2x4.
    num_plots = len(datasets)
    cols = 4
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()
    
    # Global range for y-axis could be useful, but energy scales might differ.
    # Usually Dirichlet energy decays exponentially. Log scale y often helps.
    
    for i, name in enumerate(datasets):
        if name not in results:
            continue
        
        ax = axes[i]
        res = results[name]
        
        x_axis = np.arange(layers + 1)
        
        # Original
        ax.plot(x_axis, res['orig_mean'], label='Original', color='blue', linewidth=2)
        ax.fill_between(x_axis, 
                        res['orig_mean'] - res['orig_std'], 
                        res['orig_mean'] + res['orig_std'], 
                        color='blue', alpha=0.2)
        
        # Coarse
        ax.plot(x_axis, res['coarse_mean'], label='Coarsened', color='orange', linewidth=2, linestyle='--')
        ax.fill_between(x_axis, 
                        res['coarse_mean'] - res['coarse_std'], 
                        res['coarse_mean'] + res['coarse_std'], 
                        color='orange', alpha=0.2)
        
        ax.set_title(name)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Dirichlet Energy")
        ax.set_yscale('log') # Log scale to see decay clearly
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/oversmoothing_combined.pdf')
    print("Saved plots/oversmoothing_combined.pdf")

if __name__ == "__main__":
    run_metrics_benchmark()
