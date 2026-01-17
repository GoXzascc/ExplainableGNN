
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_scipy_sparse_matrix, remove_self_loops
from torch_geometric.data import Data
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx
import matplotlib.pyplot as plt
import os
from src.corasen import spectral_graph_coarsening
from src.benchmark_runtime import get_dataset, to_undirected

def extract_lcc(data):
    """
    Extracts the Largest Connected Component from the PyG Data object.
    """
    import scipy.sparse as sp
    from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
    
    num_nodes = data.num_nodes
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=num_nodes)
    num_components, component = sp.csgraph.connected_components(adj, connection='weak')
    
    if num_components == 1:
        return data
        
    # Find largest
    counts = np.bincount(component)
    largest_comp_idx = np.argmax(counts)
    
    # Mask
    node_mask = (component == largest_comp_idx)
    node_mask_torch = torch.from_numpy(node_mask).to(data.edge_index.device)
    
    # Subgraph
    # We need to re-index.
    # torch_geometric.utils.subgraph keeps original indices? No, it filters.
    # But we need compact indices for coarsening usually.
    # Let's use masking and re-indexing.
    
    nodes_to_keep = np.where(node_mask)[0]
    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(nodes_to_keep)}
    
    # Filter edges
    row, col = data.edge_index
    mask_edges = node_mask_torch[row] & node_mask_torch[col]
    new_edge_index = data.edge_index[:, mask_edges]
    
    # Remap
    # Very slow in python for loops? 
    # Use torch searchsorted or map.
    # Faster: 
    # new_id_map = -1 * torch.ones(num_nodes, dtype=torch.long, device=data.edge_index.device)
    # new_id_map[nodes_to_keep] = torch.arange(len(nodes_to_keep), device...)
    
    new_id_map = torch.full((num_nodes,), -1, dtype=torch.long, device=data.edge_index.device)
    new_id_map[torch.from_numpy(nodes_to_keep).to(data.edge_index.device)] = torch.arange(len(nodes_to_keep), device=data.edge_index.device)
    
    new_row = new_id_map[new_edge_index[0]]
    new_col = new_id_map[new_edge_index[1]]
    final_edge_index = torch.stack([new_row, new_col], dim=0)
    
    if data.x is not None:
        new_x = data.x[node_mask_torch]
    else:
        new_x = None
        
    return Data(x=new_x, edge_index=final_edge_index, num_nodes=len(nodes_to_keep))

def get_effective_resistance_pairs(adj_matrix, pairs, num_nodes, solver_tol=1e-5):
    """
    Computes R_eff for specific pairs.
    """
    # L = D - A
    adj_matrix = adj_matrix.astype(np.float64) # Ensure float
    deg = np.array(adj_matrix.sum(1)).flatten()
    D = sp.diags(deg)
    L = D - adj_matrix
    
    eff_res = []
    
    # Use dense pinv for small graphs (< 2000 nodes)
    use_dense = num_nodes < 2000
    
    if use_dense:
        try:
            L_dense = L.toarray()
            L_pinv = np.linalg.pinv(L_dense)
            for u, v in pairs:
                if u == v:
                    eff_res.append(0.0)
                else:
                    r = L_pinv[u, u] + L_pinv[v, v] - 2 * L_pinv[u, v]
                    eff_res.append(max(0, r)) # numerical noise
        except Exception as e:
            print(f"    Dense solver failed: {e}. Switching to iterative.")
            use_dense = False
            
    if not use_dense:
        # Iterative solver (CG)
        # Solve Lz = e_u - e_v
        # Since we use LCC, system is consistent.
        for i, (u, v) in enumerate(pairs):
             if u == v:
                 eff_res.append(0.0)
                 continue
                 
             rhs = np.zeros(num_nodes)
             rhs[u] = 1
             rhs[v] = -1
             
             # CG is good for Symmetric Positive Semi-Definite
             z, info = spla.cg(L, rhs, rtol=solver_tol)
             if info != 0:
                 print(f"      CG solver failed info={info}")
                 eff_res.append(0.0)
             else:
                 r = np.dot(rhs, z) 
                 eff_res.append(max(0, r))
             
    return np.array(eff_res)

def get_shortest_path_pairs(G, pairs):
    dists = []
    for u, v in pairs:
        if u == v:
            dists.append(0)
            continue
        try:
            d = nx.shortest_path_length(G, source=int(u), target=int(v))
            dists.append(d)
        except nx.NetworkXNoPath:
            dists.append(float('inf'))
    return np.array(dists)

def run_oversquashing_benchmark():
    datasets = [
        'Cora', 'CiteSeer', 'PubMed',
        'Amazon-Photo', 'Amazon-Computers',
        'Coauthor-CS', 'Coauthor-Physics'
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = {}
    
    num_pairs = 100 
    
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

        # Extract LCC
        print(f"  Orig Nodes: {data.num_nodes}")
        data = extract_lcc(data)
        print(f"  LCC Nodes: {data.num_nodes}")
        
        # Coarsen
        print("  Coarsening...")
        try:
            coarse_data, cluster_assignment, _ = spectral_graph_coarsening(data, k=20, alpha=0.5)
            print(f"  Coarse Graph: {coarse_data.num_nodes} nodes")
        except Exception as e:
            print(f"  Coarsening failed: {e}")
            continue
            
        # Sample Pairs
        # Ensure we pick pairs that map to different clusters (mostly), 
        # or just random and handle same-cluster (R=0, D=0).
        # Random pairs.
        indices = np.random.choice(data.num_nodes, (num_pairs, 2), replace=True)
        pairs_orig = [(u, v) for u, v in indices]
        
        # --- Original Metrics ---
        print("  Computing Original Metrics (Reff, SPD)...")
        edge_index_np = data.edge_index.cpu()
        adj_orig = to_scipy_sparse_matrix(edge_index_np, num_nodes=data.num_nodes)
        G_orig = nx.from_scipy_sparse_array(adj_orig)
        
        reff_orig = get_effective_resistance_pairs(adj_orig, pairs_orig, data.num_nodes)
        spd_orig = get_shortest_path_pairs(G_orig, pairs_orig)
        
        # --- Coarse Metrics ---
        print("  Computing Coarse Metrics...")
        cluster_cpu = cluster_assignment.cpu().numpy()
        pairs_coarse = [(cluster_cpu[u], cluster_cpu[v]) for u, v in pairs_orig]
        
        edge_index_c_np = coarse_data.edge_index.cpu()
        adj_coarse = to_scipy_sparse_matrix(edge_index_c_np, num_nodes=coarse_data.num_nodes)
        G_coarse = nx.from_scipy_sparse_array(adj_coarse)
        
        reff_coarse = get_effective_resistance_pairs(adj_coarse, pairs_coarse, coarse_data.num_nodes)
        spd_coarse = get_shortest_path_pairs(G_coarse, pairs_coarse)
        
        results[name] = {
            'reff_orig': reff_orig,
            'reff_coarse': reff_coarse,
            'spd_orig': spd_orig,
            'spd_coarse': spd_coarse
        }
        
    # --- Plotting ---
    print("\nPlotting combined results...")
    os.makedirs('plots', exist_ok=True)
    
    # 1. Effective Resistance Scatter (Combined)
    plt.figure(figsize=(8, 8))
    
    # Colors or markers for datasets
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    
    all_max_reff = 0
    
    for i, name in enumerate(datasets):
        if name not in results: continue
        res = results[name]
        r_orig = res['reff_orig']
        r_coarse = res['reff_coarse']
        
        plt.scatter(r_orig, r_coarse, label=name, alpha=0.6, s=15, marker=markers[i % len(markers)])
        
        curr_max = max(r_orig.max(), r_coarse.max())
        if curr_max > all_max_reff: all_max_reff = curr_max
        
    # y=x line
    plt.plot([0, all_max_reff], [0, all_max_reff], 'k--', alpha=0.5, label='Preserved')
    
    plt.title("Effective Resistance Preservation")
    plt.xlabel("Original R_eff")
    plt.ylabel("Coarse R_eff")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('plots/oversquashing_reff.pdf')
    print("Saved plots/oversquashing_reff.pdf")
    
    # 2. Hop Distance (Box plot or Jitter)
    # Since SPD is discrete, maybe average change?
    # Or scatter with heavy jitter.
    
    plt.figure(figsize=(10, 6))
    
    # We want to show Orig vs Coarse SPD.
    # Maybe aggregate: for each dataset, show mean SPD_orig vs mean SPD_coarse?
    # Or distributions.
    
    # Let's do violin plot of SPD values? 
    # Or scatter plot like Reff but discrete.
    
    for i, name in enumerate(datasets):
        if name not in results: continue
        res = results[name]
        s_orig = res['spd_orig']
        s_coarse = res['spd_coarse']
        
        # Filter infs
        mask = (s_orig != float('inf')) & (s_coarse != float('inf'))
        s_orig = s_orig[mask]
        s_coarse = s_coarse[mask]
        
        # Jitter
        plt.scatter(s_orig + np.random.uniform(-0.2, 0.2, len(s_orig)), 
                    s_coarse + np.random.uniform(-0.2, 0.2, len(s_coarse)), 
                    label=name, alpha=0.5, s=10)
                    
    max_dist = 20 # reasonable cap
    plt.plot([0, max_dist], [0, max_dist], 'k--', alpha=0.5, label='Identity')
    plt.plot([0, max_dist], [0, max_dist/2], 'r--', alpha=0.5, label='Contraction (x0.5)')
    
    plt.title("Hop Distance Contraction")
    plt.xlabel("Original Shortest Path Distance")
    plt.ylabel("Coarse Shortest Path Distance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, max_dist)
    plt.ylim(0, max_dist)
    
    plt.savefig('plots/oversquashing_spd.pdf')
    print("Saved plots/oversquashing_spd.pdf")

if __name__ == "__main__":
    run_oversquashing_benchmark()
