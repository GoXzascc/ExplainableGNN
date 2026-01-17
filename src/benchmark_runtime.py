
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
import matplotlib.pyplot as plt
import numpy as np
import os
from src.corasen import spectral_graph_coarsening, link_wise_explanation

# Define a simple GCN (Random weights are sufficient for runtime benchmarking)
class SimpleGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x

# Monkeypatch torch.load to support OGB datasets with PyTorch 2.6+
_original_load = torch.load
def safe_load_wrapper(*args, **kwargs):
    if 'weights_only' not in kwargs:
         kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load_wrapper

def get_dataset(name, root='/tmp/BenchmarkDatasets'):
    print(f"Loading {name}...")
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root=root, name=name)
        data = dataset[0]
    elif name == 'Coauthor-CS':
        dataset = Coauthor(root=root, name='CS')
        data = dataset[0]
    elif name == 'Coauthor-Physics':
        dataset = Coauthor(root=root, name='Physics')
        data = dataset[0]
    elif name == 'Amazon-Computers':
        dataset = Amazon(root=root, name='Computers')
        data = dataset[0]
    elif name == 'Amazon-Photo':
        dataset = Amazon(root=root, name='Photo')
        data = dataset[0]
    elif name.startswith('ogbl-'):
        dataset = PygLinkPropPredDataset(name=name, root=root)
        data = dataset[0]
        # OGB Link Prop datasets might not have 'x' or 'num_classes' in the way we expect
        if not hasattr(data, 'x') or data.x is None:
            # Create random features if missing (e.g. ddi)
            # Use a small dim for efficiency (e.g. 128)
            print(f"  {name} has no features. specificing random.")
            data.x = torch.randn(data.num_nodes, 128)
        
        # PPA/Collab have features but might be large.
        if name == 'ogbl-ppa':
             # PPA node features are one-hot species ID if I recall, but PyG loader might handle it.
             # If x exists, keep it.
             pass
    else:
        raise ValueError(f"Unknown dataset {name}")

    return data, dataset

def run_benchmark():
    # Ordered roughly by size/complexity
    dataset_names = [
        'Cora', 'CiteSeer', 'PubMed',
        'Amazon-Photo', 'Amazon-Computers',
        'Coauthor-CS', 'Coauthor-Physics',
        'ogbl-ddi', 'ogbl-collab', 'ogbl-ppa'
    ]
    
    results = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for name in dataset_names:
        print(f"\nProcessing {name}...")
        try:
            data, dataset_obj = get_dataset(name)
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            continue
            
        data = data.to(device)
        
        # Ensure undirected
        if not data.is_undirected():
             data.edge_index = to_undirected(data.edge_index)
        
        num_edges = data.edge_index.size(1)
        print(f"  Nodes: {data.num_nodes}, Edges: {num_edges}")

        # 1. Global Coarsening
        # Use a timeout or skip if too large for this specific POC script?
        # PPA is 500k nodes. Eig decomp might be slow on dense matrix if not careful ("to_dense_adj" in corasen.py).
        # WAITING: src/corasen.py uses to_dense_adj! This will OOM on Collab/PPA/Physics!
        # I MUST fix corasen.py to use sparse spectral methods or limit to cpu/sparse if possible.
        # But 'torch.linalg.eigh' is for dense. 
        # For this benchmark, if the graph is too large, 'to_dense_adj' -> OOM.
        # Nodes:
        # PPA: 576k -> 576k^2 floats -> Huge. 
        # Collab: 235k -> Huge.
        # Physics: ~34k -> 34000^2 * 4 bytes ~= 4GB. Might fit?
        # Computers: 13k -> Fits.
        
        # FIX: We need sparse eigen solver for large graphs.
        # Scipy operates on sparse. 'torch_geometric.utils.get_laplacian' + 'scipy.sparse.linalg.eigsh'?
        # Or 'lobpcg' in torch?
        
        # For the purpose of this request, I should probably switch to sparse implementation in `corasen.py` 
        # OR just acknowledge the limit. But user asked for PPA/Collab.
        # I will try to update `corasen.py` to use `scipy` sparse solver if `num_nodes` is large, or always.
        
        # Let's try running. If it crashes, I will fix `corasen.py`.
        # Actually, being proactive: `to_dense_adj` on 576k nodes is impossible (Terabytes).
        # I MUST update `corasen.py` first.
        
        print("  Running Global Coarsening...")
        
        # Setup Model
        num_features = data.x.size(1)
        # Random logic for num_classes if not classification
        num_classes = dataset_obj.num_classes if hasattr(dataset_obj, 'num_classes') else 16
        
        model = SimpleGCN(num_features, 16, num_classes).to(device)
        model.eval()
        
        try:
            t0 = time.time()
            # Reducing k/alpha optionally for speed on huge graphs if needed, but keeping standard for now.
            coarse_data, cluster_assignment, P = spectral_graph_coarsening(data, k=20, alpha=0.5)
            t_global = time.time() - t0
            print(f"  Global Coarsening Time: {t_global:.4f}s")
        except RuntimeError as e: # OOM
            print(f"  Global Coarsening Failed (likely OOM): {e}")
            # If global fails, we can't run local.
            continue
        
        # 2. Measure Per-Subgraph Time
        num_samples = 20 # Reduced samples for speed on large sets
        indices = torch.randint(0, data.edge_index.size(1), (num_samples,))
        sampled_edges = data.edge_index[:, indices].t()
        
        times = []
        try:
            with torch.no_grad():
                for i in range(num_samples):
                    u, v = sampled_edges[i].tolist()
                    start = time.time()
                    
                    subgraph_data = link_wise_explanation(data, cluster_assignment, (u, v))
                    _ = model(subgraph_data.x, subgraph_data.edge_index, subgraph_data.edge_attr)
                    
                    end = time.time()
                    times.append(end - start)
        except Exception as e:
             print(f"  Inference Failed: {e}")
             continue
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"  Avg Per-Subgraph Time: {avg_time:.6f}s (+/- {std_time:.6f})")
        
        results.append({
            'dataset': name,
            'edges': num_edges,
            'time': avg_time,
            'std': std_time
        })

    # Plotting
    print("\nPlotting results...")
    # Sort by connections
    results.sort(key=lambda x: x['edges'])
    
    edges = [r['edges'] for r in results]
    runtimes = [r['time'] for r in results]
    names = [r['dataset'] for r in results]
    
    plt.figure(figsize=(10, 8))
    plt.loglog(edges, runtimes, 'o-', label='Ours (Coarsening)')
    
    for i, txt in enumerate(names):
        # Alternate label positions to avoid overlap
        xytext = (0, 10) if i % 2 == 0 else (0, -15)
        plt.annotate(txt, (edges[i], runtimes[i]), textcoords="offset points", xytext=xytext, ha='center', fontsize=8)

    plt.xlabel('Number of Edges |E|')
    plt.ylabel('Time per Subgraph (s)')
    plt.title('Runtime Scaling vs Edge Count')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/runtime_scaling_extended.pdf')
    print("Saved plot to plots/runtime_scaling_extended.pdf")

if __name__ == "__main__":
    run_benchmark()
