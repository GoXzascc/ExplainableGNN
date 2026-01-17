
import torch
from torch_geometric.utils import to_dense_adj, get_laplacian, add_self_loops, remove_self_loops
from torch_geometric.data import Data
import numpy as np

import torch
import scipy.sparse as sp
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix, remove_self_loops
from torch_geometric.data import Data

def get_normalized_adjacency_sparse(edge_index, num_nodes):
    """
    Computes \hat{A} as a scipy sparse matrix.
    """
    # Remove loops first to be safe, then add back
    edge_index, _ = remove_self_loops(edge_index)
    
    # Construct adjacency
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    
    # +I
    adj = adj + sp.eye(num_nodes)
    
    # D^{-1/2}
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    
    # D^{-1/2} (A + I) D^{-1/2} = D^{-1/2} adj D^{-1/2}
    norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
    return norm_adj

def compute_top_k_eigen_sparse(edge_index, num_nodes, k):
    """
    Computes top-k eigen decomposition using scipy sparse solver.
    """
    norm_adj = get_normalized_adjacency_sparse(edge_index, num_nodes)
    
    # Use eigsh for symmetric matrices
    # k+1 because sometimes the largest is just 1 (stationary), we might want variations
    # But theorem says top-k of \hat{A}.
    # \hat{A} eigenvalues are in [-1, 1]. Top means largest magnitude or largest positive?
    # Usually largest positive = low frequency in Laplacian L = I - \hat{A}.
    # Largest \lambda of \hat{A}  <-> Smallest \lambda of L.
    # We want "Dominant spectral components". 
    # Paper Section 2.2: "Top-k spectral components... correspond to most influential ways information spreads".
    # Usually this means largest algebraic values.
    
    # Check if k >= num_nodes
    if k >= num_nodes:
        vals, vecs = np.linalg.eigh(norm_adj.toarray())
    else:
        # 'LA' = Largest Algebraic
        vals, vecs = sp.linalg.eigsh(norm_adj, k=k, which='LA')
    
    # Sort descending
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    return torch.from_numpy(vals.copy()).float(), torch.from_numpy(vecs.copy()).float()

def compute_top_k_eigen_dense_torch(edge_index, num_nodes, k, device):
    """
    Computes top-k eigen decomposition using dense torch ops (GPU-friendly).
    """
    # Build dense adjacency on target device
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).to(device)

    # Add self-loops
    adj = adj + torch.eye(num_nodes, device=device)

    # Normalized adjacency: D^{-1/2} (A + I) D^{-1/2}
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
    norm_adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

    # Full eigen decomposition, then take top-k
    vals, vecs = torch.linalg.eigh(norm_adj)
    if k >= num_nodes:
        return vals.float(), vecs.float()

    topk_vals, topk_idx = torch.topk(vals, k, largest=True)
    topk_vecs = vecs[:, topk_idx]
    return topk_vals.float(), topk_vecs.float()

def compute_top_k_eigen_lobpcg(edge_index, num_nodes, k, device, max_iter=100, tol=1e-4):
    """
    Computes top-k eigen decomposition using LOBPCG on a sparse normalized adjacency (GPU-friendly).
    """
    # Build sparse normalized adjacency on device
    edge_index, _ = remove_self_loops(edge_index)
    self_loops = torch.arange(num_nodes, device=device)
    self_loops = torch.stack([self_loops, self_loops], dim=0)
    full_edge_index = torch.cat([edge_index.to(device), self_loops], dim=1)

    row, col = full_edge_index
    values = torch.ones(full_edge_index.size(1), device=device)

    deg = torch.zeros(num_nodes, device=device)
    deg.index_add_(0, row, values)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0

    norm_values = values * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    norm_adj = torch.sparse_coo_tensor(full_edge_index, norm_values, (num_nodes, num_nodes), device=device)

    # Free memory of intermediate tensors
    del row, col, values, deg, deg_inv_sqrt, norm_values, full_edge_index, self_loops
    torch.cuda.empty_cache()

    # LOBPCG expects a symmetric matrix; normalized adjacency is symmetric for undirected graphs.
    # Use a random initial guess.
    init = torch.randn(num_nodes, k, device=device)
    try:
        vals, vecs = torch.lobpcg(norm_adj, k=k, largest=True, niter=max_iter, tol=tol, X=init)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"  LOBPCG OOM on GPU. Converting to CPU for eigen decomp of {num_nodes} nodes...")
            torch.cuda.empty_cache()
            # Move to CPU
            norm_adj_cpu = norm_adj.cpu()
            # If norm_adj was coalesced, fine. If not, coalesce.
            if not norm_adj_cpu.is_coalesced():
                norm_adj_cpu = norm_adj_cpu.coalesce()
            
            # Scipy eigsh on CPU is usually more robust for memory
            # Convert to scipy
            import scipy.sparse as sp
            row = norm_adj_cpu.indices()[0].numpy()
            col = norm_adj_cpu.indices()[1].numpy()
            data = norm_adj_cpu.values().numpy()
            scipy_adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
            
            vals_np, vecs_np = sp.linalg.eigsh(scipy_adj, k=k, which='LA')
            
            # Sort descending
            idx = np.argsort(vals_np)[::-1]
            vals_np = vals_np[idx]
            vecs_np = vecs_np[:, idx]
            
            vals = torch.from_numpy(vals_np.copy()).float().to(device)
            vecs = torch.from_numpy(vecs_np.copy()).float().to(device)
        else:
            raise e
            
    return vals.float(), vecs.float()

def compute_perturbation_scores(eigenvalues, eigenvectors, edge_index, k):
    """
    Computes the perturbation score \Delta_{(a,b)}.
    """
    device = edge_index.device
    num_edges = edge_index.size(1)
    
    # Move to device if not
    vals = eigenvalues.to(device)
    vecs = eigenvectors.to(device)
    
    # To vectorize score computation:
    # \Delta = \sum_i | \nu_i / \eta_i |
    
    row, col = edge_index
    
    # Gather u, v vectors: (num_edges, k)
    u_vecs = vecs[row] 
    v_vecs = vecs[col]
    
    # eta = 1 - (u^2 + v^2)  (assuming unit norm)
    # (num_edges, k)
    eta = 1.0 - (u_vecs.pow(2) + v_vecs.pow(2))
    
    # nu terms
    # term1 = -u^2
    # term2 = (6 - lambda) * u * v
    # term3 = (lambda - 1) * v^2
    
    # vals: (k,) -> (1, k)
    vals_expanded = vals.unsqueeze(0)
    
    term1 = -u_vecs.pow(2)
    term2 = (6.0 - vals_expanded) * u_vecs * v_vecs
    term3 = (vals_expanded - 1.0) * v_vecs.pow(2)
    
    nu = term1 + term2 + term3
    
    # Prevent div by zero
    eta_safe = eta.clone()
    eta_safe[eta_safe.abs() < 1e-6] = 1e-6
    
    ratio = (nu / eta_safe).abs()
    
    # Sum over k -> (num_edges,)
    scores = ratio.sum(dim=1)
    
    # Handle self-loops (result in weird scores, set to inf)
    mask = (row == col)
    scores[mask] = float('inf')
    
    return scores

def partition_nodes(edge_index, scores, num_nodes, alpha, approx_gpu=True, edge_factor=2, max_lp_iters=10):
    """
    Implements Algorithm 2: Node Partition Rule.
    Uses Union-Find to efficiently merge nodes based on lowest perturbation scores.
    Merged sets (clusters) are formed until alpha * N merges are performed.
    """
    if edge_index.is_cuda and approx_gpu and hasattr(torch.Tensor, "scatter_reduce_"):
        num_edges = edge_index.size(1)
        target_edges = min(num_edges, int(alpha * num_nodes * edge_factor))
        if target_edges <= 0:
            cluster_assignment = torch.arange(num_nodes, device=edge_index.device, dtype=torch.long)
            return cluster_assignment, num_nodes

        _, top_idx = torch.topk(scores, k=target_edges, largest=False)
        sub_edges = edge_index[:, top_idx]
        row, col = sub_edges

        labels = torch.arange(num_nodes, device=edge_index.device, dtype=torch.long)
        for _ in range(max_lp_iters):
            new_labels = labels.clone()
            new_labels.scatter_reduce_(0, row, labels[col], reduce='amin', include_self=True)
            new_labels.scatter_reduce_(0, col, labels[row], reduce='amin', include_self=True)
            if torch.equal(new_labels, labels):
                break
            labels = new_labels

        unique_labels, inverse = torch.unique(labels, return_inverse=True)
        return inverse, unique_labels.numel()

    # Sort edges by score
    sorted_idx = torch.argsort(scores)
    sorted_edges = edge_index[:, sorted_idx]
    
    # Move to CPU / Numpy for sequential loop performance
    # iterating over tensor items one by one is slow in Python.
    edges_np = sorted_edges.cpu().numpy()
    row, col = edges_np[0], edges_np[1]
    
    # Union-Find initialization
    parent = np.arange(num_nodes)
    
    def find(i):
        # Iterative path compression
        path = []
        root = i
        while root != parent[root]:
            path.append(root)
            root = parent[root]
        
        for node in path:
            parent[node] = root
        return root

    num_merges = 0
    target_merges = int(alpha * num_nodes)
    
    # Iterate and merge
    num_edges = len(row)
    for i in range(num_edges):
        if num_merges >= target_merges:
            break
            
        u, v = row[i], col[i]
        root_u = find(u)
        root_v = find(v)
        
        if root_u != root_v:
            parent[root_u] = root_v
            num_merges += 1
            
    # Assign final cluster IDs
    # Flatten structure
    final_clusters = np.zeros(num_nodes, dtype=np.int64)
    for i in range(num_nodes):
        final_clusters[i] = find(i)
        
    # Remap unique roots to contiguous IDs Range(0, num_clusters)
    # unique returns sorted unique elements
    unique_roots, inverse = np.unique(final_clusters, return_inverse=True)
    
    # To Tensor
    cluster_assignment = torch.from_numpy(inverse).long()
    if edge_index.is_cuda:
        cluster_assignment = cluster_assignment.to(edge_index.device)
        
    num_clusters = len(unique_roots)
    
    return cluster_assignment, num_clusters

def build_coarse_graph(data, cluster_assignment, num_clusters):
    """
    Constructs the coarse graph G' based on partition.
    P matrix: N x N' (num_nodes x num_clusters)
    Uses sparse matrix operations to avoid OOM on large graphs.
    """
    device = data.x.device
    num_nodes = data.num_nodes
    
    # Construct Sparse P matrix (PyTorch Sparse or SciPy)
    # Using torch.sparse_coo_tensor for GPU compatibility in future,
    # but for simple algebra, sparse matrix multiplication in torch is 'spspmm'.
    
    indices = torch.stack([
        torch.arange(num_nodes, device=device),
        cluster_assignment.to(device)
    ])
    values = torch.ones(num_nodes, device=device)
    
    # P_hat (unnormalized) as sparse tensor
    P_hat = torch.sparse_coo_tensor(indices, values, (num_nodes, num_clusters), device=device)
    
    # Compute cluster sizes for normalization
    # P_hat is N x K. 
    # sum(0) via dense vector
    cluster_assignment = cluster_assignment.to(device)
    cluster_sizes = torch.zeros(num_clusters, device=device)
    cluster_sizes.index_add_(0, cluster_assignment, torch.ones(num_nodes, device=device))
    
    inv_sqrt_sizes = cluster_sizes.pow(-0.5)
    inv_sqrt_sizes[inv_sqrt_sizes == float('inf')] = 0
    
    # P = P_hat @ M^{-1/2}
    # This just scales the columns.
    # We can scale the values in P_hat directly.
    # P_vals = 1 * M^{-1/2}[cluster_idx]
    P_values_norm = inv_sqrt_sizes[cluster_assignment]
    P = torch.sparse_coo_tensor(indices, P_values_norm, (num_nodes, num_clusters), device=device)
    
    # Coarse Adjacency W = P^T A P
    # A is sparse. P is sparse. 3 Sparse matmuls?
    # Torch sparse support is limited. P.t() @ A @ P.
    # Convert to standard adjacency sparse tensor.
    
    A_indices = data.edge_index
    A_values = torch.ones(A_indices.size(1), device=device)
    A = torch.sparse_coo_tensor(A_indices, A_values, (num_nodes, num_nodes), device=device)
    
    # W = P.t() @ (A @ P)
    # A @ P: (N x N) @ (N x K) -> (N x K)
    # Note: torch.sparse.mm(A, P) (if P is dense) or sparse @ sparse
    # Torch sparse @ sparse is `torch.sparse.mm` in recent versions or `sspaddmm`.
    # Let's try `torch.sparse.mm` assuming 2D.
    # Actually P is sparse. A is sparse.
    # PyTorch sparse matmul is strict.
    # Better to just use indices logic for W construction?
    # W_uv = sum_{i in Ci, j in Cj} A_ij * P_ik * P_jk
    # Roughly: transform edge_index (u,v) -> (cluster[u], cluster[v])
    # Then sum duplicates.
    
    row, col = data.edge_index
    c_row = cluster_assignment[row]
    c_col = cluster_assignment[col]
    
    new_indices = torch.stack([c_row, c_col])
    
    # For Unnormalized W (as in Definition 3): \hat{P}^T A \hat{P}
    # Just sum weights of edges between clusters.
    # For Normalized W (P^T A P): weighted by 1/sqrt(mi*mj).
    # Paper Section 2.3 Eq 5 uses \hat{P} (unnormalized).
    # Then says "We will use normalized P for subsequent steps" (referring to spectral stuff).
    # But for "Weight and Degree of Coarse Graph", it explicitly uses \hat{P}.
    # Let's stick to Eq 5 for W: \hat{P}^T A \hat{P} -> Sum of edges.
    
    # Coalesce indices
    # We can use torch_geometric.utils.coalesce
    from torch_geometric.utils import coalesce
    edge_index_prime, edge_attr_prime = coalesce(new_indices, torch.ones(new_indices.size(1), device=device), num_nodes=num_clusters, reduce='add')
        
    # Coarse Features
    # X' = P^T X (weighted mean) or LogSumExp.
    # Global coarsening usually uses standard projection: X' = P^T X.
    # X is N x F (Dense). P is sparse.
    # X' = (P^T @ X) = (X^T @ P)^T
    # X^T @ P: (F x N) @ (N x K) -> (F x K).
    # torch.sparse.mm works for (Sparse, Dense). P is Sparse.
    # P.t() @ X -> Sparse @ Dense -> supported.
    
    # Note: If P is sparse (K, N) after transpose.
    # But transpose of sparse tensor in pytorch might not be coalesced or supported in mm easily.
    # P is (N, K). P.t() is (K, N).
    # Let's compute X' = P^T X.
    # P values are 'P_values_norm' (normalized).
    
    X_prime = torch.sparse.mm(P.t(), data.x) # (K, N) @ (N, F) -> (K, F)
    
    # Result
    data_prime = Data(x=X_prime, edge_index=edge_index_prime, edge_attr=edge_attr_prime)
    data_prime.num_nodes = num_clusters
    
    return data_prime, P

def dense_to_sparse_with_attr(adj): # Unused now
    indices = torch.nonzero(adj)
    rows = indices[:, 0]
    cols = indices[:, 1]
    values = adj[rows, cols]
    edge_index = torch.stack([rows, cols])
    return edge_index, values


def spectral_graph_coarsening(
    data,
    k=10,
    alpha=0.5,
    dense_threshold=5000,
    eigen_method="auto",
    lobpcg_max_nodes=2000000
):
    """
    Main pipeline for global coarsening.
    """
    num_nodes = data.num_nodes
    
    # 1. Compute Top-k Eigen
    # Use GPU dense path for smaller graphs; CPU sparse for large graphs.
    use_dense_gpu = data.x.is_cuda and num_nodes <= dense_threshold
    use_lobpcg = data.x.is_cuda and num_nodes <= lobpcg_max_nodes

    if eigen_method == "dense" and use_dense_gpu:
        vals, vecs = compute_top_k_eigen_dense_torch(data.edge_index, num_nodes, k, data.x.device)
        edge_index_for_scores = data.edge_index
    elif eigen_method == "lobpcg" and use_lobpcg:
        vals, vecs = compute_top_k_eigen_lobpcg(data.edge_index, num_nodes, k, data.x.device)
        edge_index_for_scores = data.edge_index
    elif eigen_method == "auto":
        if use_dense_gpu:
            vals, vecs = compute_top_k_eigen_dense_torch(data.edge_index, num_nodes, k, data.x.device)
            edge_index_for_scores = data.edge_index
        elif use_lobpcg:
            vals, vecs = compute_top_k_eigen_lobpcg(data.edge_index, num_nodes, k, data.x.device)
            edge_index_for_scores = data.edge_index
        else:
            edge_index_cpu = data.edge_index.cpu()
            vals, vecs = compute_top_k_eigen_sparse(edge_index_cpu, num_nodes, k)
            edge_index_for_scores = edge_index_cpu
    else:
        edge_index_cpu = data.edge_index.cpu()
        vals, vecs = compute_top_k_eigen_sparse(edge_index_cpu, num_nodes, k)
        edge_index_for_scores = edge_index_cpu
    
    # 2. Compute Scores
    edge_index_no_loop, _ = remove_self_loops(edge_index_for_scores) # Clean loops for scoring
    scores = compute_perturbation_scores(vals, vecs, edge_index_no_loop, k)
    
    # 3. Partition
    cluster_assignment, num_clusters = partition_nodes(edge_index_no_loop, scores, num_nodes, alpha)
    if data.x.is_cuda:
        cluster_assignment = cluster_assignment.to(data.x.device)
    
    # 4. Build Coarse Graph
    coarse_data, P = build_coarse_graph(data, cluster_assignment, num_clusters)
    
    return coarse_data, cluster_assignment, P

def link_wise_explanation(data, cluster_assignment, target_edge, agg='mean'):
    """
    Generates the link-wise explanatory subgraph G'_{(a,b)}.
    Splits the clusters containing a and b.
    """
    u, v = target_edge
    c_u = cluster_assignment[u].item()
    c_v = cluster_assignment[v].item()
    
    device = data.x.device
    
    # Identify nodes in these clusters
    u_nodes = (cluster_assignment == c_u).nonzero(as_tuple=True)[0]
    v_nodes = (cluster_assignment == c_v).nonzero(as_tuple=True)[0]
    
    split_nodes = torch.cat([u_nodes, v_nodes]) # These nodes become singletons
    
    # Create new mapping
    # Old clusters k \notin {c_u, c_v} remain as supernodes.
    # Nodes in c_u and c_v become individual supernodes (size 1).
    
    # Unique clusters to keep
    unique_clusters = torch.unique(cluster_assignment)
    kept_clusters = unique_clusters[(unique_clusters != c_u) & (unique_clusters != c_v)]
    
    # New supernodes map:
    # 0..len(kept)-1 : kept clusters
    # len(kept)... : split nodes
    
    # We need to build P_{(a,b)}
    # Let's map old_cluster_id -> new_supernode_idx
    # But split nodes don't follow old_cluster_id.
    
    num_kept = len(kept_clusters)
    num_split = len(split_nodes)
    total_new_nodes = num_kept + num_split
    
    # Map from kept cluster ID to new index (0 to num_kept-1)
    # We can use a dictionary or lookup tensor
    cluster_to_new_idx = {c.item(): i for i, c in enumerate(kept_clusters)}
    
    # Construct P_{(a,b)} indices
    # Each original node i:
    #   if cluster[i] in kept: assign to new_idx(cluster[i])
    #   if i in split_nodes: assign to num_kept + (index in split_nodes)
    
    new_assignment = torch.zeros(data.num_nodes, dtype=torch.long)
    
    # For split nodes, we need to know their order in split_nodes to assign unique IDs
    # Let's verify we don't duplicate if c_u == c_v
    if c_u == c_v:
        # split_nodes is just u_nodes (which equals v_nodes)
        split_nodes = u_nodes # Distinct sorted
    else:
        # Ensure unique if any overlap (shouldn't be, partition is disjoint)
        split_nodes = torch.unique(split_nodes)

    num_split = len(split_nodes)
    total_new_nodes = num_kept + num_split
    
    # Assign new IDs
    # Fast assignment mask
    is_split = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
    is_split[split_nodes] = True
    
    # 1. Assign Kept Clusters
    # We can iterate nodes or just use the cluster_assignment
    # If not split, new_id = cluster_to_new_idx[old_cluster]
    
    # This is slightly slow to map with dict in loop. 
    # Use a lookup tensor for clusters?
    # max_cluster = cluster_assignment.max().item()
    # map_tensor = torch.full((max_cluster+1,), -1, dtype=torch.long)
    # for k, v in cluster_to_new_idx.items(): map_tensor[k] = v
    # This works.
    
    max_cluster = cluster_assignment.max().item()
    cluster_map = torch.full((max_cluster + 1,), -1, dtype=torch.long, device=data.x.device)
    if len(kept_clusters) > 0:
        values = torch.arange(len(kept_clusters), device=data.x.device)
        cluster_map[kept_clusters] = values

    # Apply map
    node_cluster_ids = cluster_assignment
    mapped_ids = cluster_map[node_cluster_ids]
    
    new_assignment = mapped_ids.clone()
    
    # 2. Assign Split Nodes
    # They need range [num_kept, num_kept + num_split - 1]
    # We can just assign them incrementally
    split_indices = torch.arange(num_split, device=data.x.device) + num_kept
    new_assignment[split_nodes] = split_indices
    
    # Now build P and Coarse Graph
    # This logic is same as build_coarse_graph but with specific feature aggregation
    
    # unused P_hat removed


    # Features: mean aggregation (GPU-friendly) or LogSumExp (original, slower)
    X_prime = torch.zeros(total_new_nodes, data.x.size(1), device=data.x.device)
    if agg == 'lse':
        X_prime[num_kept:] = data.x[split_nodes]
        for i, c_id in enumerate(kept_clusters):
            mask = (cluster_assignment == c_id)
            nodes_in_c = data.x[mask]
            lse = torch.logsumexp(nodes_in_c, dim=0)
            X_prime[i] = lse
    else:
        X_prime.index_add_(0, new_assignment, data.x)
        counts = torch.zeros(total_new_nodes, device=data.x.device)
        counts.index_add_(0, new_assignment, torch.ones(data.num_nodes, device=data.x.device))
        X_prime = X_prime / counts.clamp_min(1).unsqueeze(1)

    # Weights and Edges
    # W = P_hat^T A P_hat. 
    # Since we want sparse, we map edges and coalesce.
    
    row, col = data.edge_index
    new_row = new_assignment[row] # Use correct variable
    new_col = new_assignment[col]
    
    new_indices = torch.stack([new_row, new_col])
    
    # Coalesce to sum weights
    from torch_geometric.utils import coalesce
    # Ensure weights exist. If data has edge_attr, use it? 
    # For OGB datasets without weights (weight=1), edge_attr is None.
    
    if data.edge_attr is not None and data.edge_attr.size(0) == row.size(0):
        # Assuming scalar weights or handling accumulation
        weights = data.edge_attr
        if weights.dim() > 1 and weights.size(1) > 1:
             # Link prediction edge features? Summing them might not be right but for coarsening usually yes.
             pass
    else:
        weights = torch.ones(row.size(0), device=device)

    # Note: coalesce signature changes across PyG versions. 
    # Trying reduce='add'.
    edge_index_prime, edge_attr_prime = coalesce(new_indices, weights, num_nodes=total_new_nodes, reduce='add')
        
    data_link = Data(x=X_prime, edge_index=edge_index_prime, edge_attr=edge_attr_prime)
    
    return data_link
