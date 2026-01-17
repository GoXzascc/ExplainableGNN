
import torch
from torch_geometric.utils import to_dense_adj, get_laplacian, add_self_loops, remove_self_loops
from torch_geometric.data import Data
import numpy as np

def get_normalized_adjacency(edge_index, num_nodes):
    """
    Computes the symmetrically normalized adjacency matrix \hat{A}.
    \hat{A} = D^{-1/2} (A + I) D^{-1/2}
    """
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    
    # Compute degree
    row, col = edge_index
    deg = torch.zeros(num_nodes, dtype=torch.float, device=edge_index.device)
    deg.scatter_add_(0, row, torch.ones(edge_index.size(1), device=edge_index.device))
    
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    # We can use dense matrix for spectral decomposition as it is often faster for 
    # small/medium graphs which we likely handle here, or we need sparse eig decomp.
    # For stability and simplicity in this POC, let's use dense.
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    
    # D^{-1/2} A D^{-1/2}
    d_mat_inv_sqrt = torch.diag(deg_inv_sqrt)
    norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    
    return norm_adj

def compute_top_k_eigen(norm_adj, k):
    """
    Computes the top-k eigenvectors and eigenvalues of the normalized adjacency matrix.
    """
    # Since norm_adj is symmetric
    eigenvalues, eigenvectors = torch.linalg.eigh(norm_adj)
    
    # Sort in descending order (eigh returns ascending)
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues[:k], eigenvectors[:, :k]

def compute_perturbation_scores(eigenvalues, eigenvectors, edge_index, k):
    """
    Computes the perturbation score \Delta_{(a,b)} for each edge (a,b).
    Uses the approximation from Theorem 2.
    """
    num_edges = edge_index.size(1)
    scores = torch.zeros(num_edges, device=edge_index.device)
    
    # Pre-compute terms to speed up
    # However, the formula is per-edge.
    # \Delta_{(a,b)} \approx \sum_{i=1}^k | \nu_i / \eta_i |
    
    # Loop over edges is inevitable without complex vectorization, 
    # but for prototype we can loop.
    # Let's try to vectorize over 'k' for each edge.
    
    row, col = edge_index
    
    for e in range(num_edges):
        u, v = row[e].item(), col[e].item()
        if u == v:
            scores[e] = float('inf') # Don't merge self-loops if any
            continue
            
        # Get u and v components of all k eigenvectors: shape (k,)
        u_vec = eigenvectors[u, :] 
        v_vec = eigenvectors[v, :] 
        
        # \eta_i = v_i^T u_i - (u_{ia}v_{ia} + u_{ib}v_{ib})
        # Since symmetric, u_i = v_i.
        # \eta_i = 1 - (u_{ia}^2 + u_{ib}^2)  (Assuming normalized eigenvectors v_i^T v_i = 1)
        eta = 1.0 - (u_vec**2 + v_vec**2)
        
        # \nu_i ... formula in Theorem 2
        # u_{ia} corresponds to 'a' component of eigenvector i
        # u_{ib} corresponds to 'b' component
        # v_{ia} = u_{ia}, v_{ib} = u_{ib}
        
        # \nu_i = -u_{ia}^2 + 3 u_{ia} u_{ib} + (3 - \lambda_i) u_{ib} u_{ia} + (\lambda_i - 1) u_{ib}^2
        #       = -u_{ia}^2 + 3 u_{ia} u_{ib} + 3 u_{ia} u_{ib} - \lambda_i u_{ia} u_{ib} + \lambda_i u_{ib}^2 - u_{ib}^2
        #       = -u_{ia}^2 - u_{ib}^2 + (6 - \lambda_i) u_{ia} u_{ib} + \lambda_i u_{ib}^2 ?
        # Wait, let's re-read the theorem carefully. 
        # \nu_i = -v_{ia}u_{ia} + 3v_{ia}u_{ib} + (3-\lambda_i)v_{ib}u_{ia} + (\lambda_i - 1)v_{ib}u_{ib}
        # substitutions:
        # = -u_{ia}^2 + 3 u_{ia} u_{ib} + (3 - \lambda_i) u_{ib} u_{ia} + (\lambda_i - 1) u_{ib}^2
        # = -u_{ia}^2 + (3 + 3 - \lambda_i) u_{ia} u_{ib} + (\lambda_i - 1) u_{ib}^2
        # = -u_{ia}^2 + (6 - \lambda_i) u_{ia} u_{ib} + (\lambda_i - 1) u_{ib}^2
        
        term1 = -u_vec**2
        term2 = (6.0 - eigenvalues) * u_vec * v_vec
        term3 = (eigenvalues - 1.0) * v_vec**2
        
        nu = term1 + term2 + term3
        
        # Avoid division by zero
        eta[eta.abs() < 1e-6] = 1e-6
        
        ratio = (nu / eta).abs()
        scores[e] = ratio.sum()
        
    return scores

def partition_nodes(edge_index, scores, num_nodes, alpha):
    """
    Implements Algorithm 2: Node Partition Rule.
    Greedily merges nodes with lowest perturbation scores.
    """
    # Sort edges by score
    sorted_idx = torch.argsort(scores)
    sorted_edges = edge_index[:, sorted_idx]
    
    # Partition map: node_id -> cluster_id
    # Initially everyone in their own cluster?
    # No, Algo says: "Init partition with empty classes... Place both a and b into an empty class Ci"
    # This implies we are building the partition P.
    
    # Let's track which cluster each node belongs to.
    # -1 means unassigned.
    cluster_assignment = torch.full((num_nodes,), -1, dtype=torch.long)
    next_cluster_id = 0
    
    num_merges = 0
    target_merges = int(alpha * num_nodes) 
    # Note: alpha |V| merges might be too many if alpha is close to 1 ? 
    # If we merge pairs, we reduce count.
    # Algo loop: while m < alpha |V|
    
    row, col = sorted_edges
    
    for i in range(sorted_edges.size(1)):
        if num_merges >= target_merges:
            break
            
        u, v = row[i].item(), col[i].item()
        
        # Check current assignments
        c_u = cluster_assignment[u].item()
        c_v = cluster_assignment[v].item()
        
        if c_u == -1 and c_v == -1:
            # New cluster
            cluster_assignment[u] = next_cluster_id
            cluster_assignment[v] = next_cluster_id
            next_cluster_id += 1
            num_merges += 1
        elif c_u != -1 and c_v == -1:
            # Add v to u's cluster
            cluster_assignment[v] = c_u
            num_merges += 1
        elif c_u == -1 and c_v != -1:
            # Add u to v's cluster
            cluster_assignment[u] = c_v
            num_merges += 1
        else:
            # Both assigned. 
            # If different clusters, do we merge clusters?
            # Algo says: 
            # If neither in C -> new class
            # If only one in C -> add other
            # If both in C -> Do nothing (implied by missing else)?
            # Wait, standard coarsening usually merges clusters.
            # But the algorithm purely says:
            # "If neither... Place both"
            # "Else if only one... Place the other"
            # It DOES NOT handle the case where both are already in different clusters.
            # This implies we ONLY grow existing clusters or create new ones from unassigned nodes.
            pass
            
    # Handle remaining unassigned nodes
    # "Send each node to an empty class" -> Assign unique new cluster IDs
    for n in range(num_nodes):
        if cluster_assignment[n] == -1:
            cluster_assignment[n] = next_cluster_id
            next_cluster_id += 1
            
    return cluster_assignment, next_cluster_id

def build_coarse_graph(data, cluster_assignment, num_clusters):
    """
    Constructs the coarse graph G' based on partition.
    P matrix: N x N' (num_nodes x num_clusters)
    """
    device = data.x.device
    num_nodes = data.num_nodes
    
    # Construct P matrix (sparse)
    # indices: [node_idx, cluster_idx]
    indices = torch.stack([
        torch.arange(num_nodes, device=device),
        cluster_assignment.to(device)
    ])
    values = torch.ones(num_nodes, device=device)
    
    # P_hat: N x N'
    P_hat = torch.sparse_coo_tensor(indices, values, (num_nodes, num_clusters)).to_dense()
    
    # Normalize P to get orthonormal columns
    # M = diag(m1, ..., mk)
    cluster_sizes = P_hat.sum(dim=0)
    # Avoid div by zero (shouldn't happen if every cluster has at least 1 node)
    inv_sqrt_sizes = cluster_sizes.pow(-0.5)
    inv_sqrt_sizes[inv_sqrt_sizes == float('inf')] = 0
    
    M_inv_sqrt = torch.diag(inv_sqrt_sizes)
    P = P_hat @ M_inv_sqrt
    
    # Coarse Adjacency W = P^T A P
    adj = to_dense_adj(data.edge_index, max_num_nodes=num_nodes)[0]
    W = P.t() @ adj @ P
    
    # Coarse Features
    # Typically aggregated. The paper mentions P^T X is one way, 
    # but for link-wise it mentions LogSumExp.
    # For global coarse graph, let's follow the standard projection P^T X?
    # Or aggregation? Definition 3 says "node features X' are aggregated from X".
    # Usually X' = P^T X corresponds to weighted sum normalized by size (mean pooling roughly).
    X_prime = P.t() @ data.x
    
    # Create coarse edge_index and edge_weight from W
    # W is dense N' x N'
    # extracting edges
    edge_index_prime, edge_attr_prime = dense_to_sparse_with_attr(W)
    
    # Create new Data object
    # We might need to store W as edge_weight
    data_prime = Data(x=X_prime, edge_index=edge_index_prime, edge_attr=edge_attr_prime)
    data_prime.num_nodes = num_clusters
    
    return data_prime, P

def dense_to_sparse_with_attr(adj):
    """
    Converts dense adjacency to sparse edge_index and edge_weight.
    """
    indices = torch.nonzero(adj)
    rows = indices[:, 0]
    cols = indices[:, 1]
    values = adj[rows, cols]
    edge_index = torch.stack([rows, cols])
    return edge_index, values

def spectral_graph_coarsening(data, k=10, alpha=0.5):
    """
    Main pipeline for global coarsening.
    """
    num_nodes = data.num_nodes
    
    # 1. Compute Top-k Eigen
    norm_adj = get_normalized_adjacency(data.edge_index, num_nodes)
    vals, vecs = compute_top_k_eigen(norm_adj, k)
    
    # 2. Compute Scores
    edge_index_no_loop, _ = remove_self_loops(data.edge_index) # Clean loops for scoring
    scores = compute_perturbation_scores(vals, vecs, edge_index_no_loop, k)
    
    # 3. Partition
    cluster_assignment, num_clusters = partition_nodes(edge_index_no_loop, scores, num_nodes, alpha)
    
    # 4. Build Coarse Graph
    coarse_data, P = build_coarse_graph(data, cluster_assignment, num_clusters)
    
    return coarse_data, cluster_assignment, P

def link_wise_explanation(data, cluster_assignment, target_edge):
    """
    Generates the link-wise explanatory subgraph G'_{(a,b)}.
    Splits the clusters containing a and b.
    """
    u, v = target_edge
    c_u = cluster_assignment[u].item()
    c_v = cluster_assignment[v].item()
    
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
    
    # Construct P_hat
    indices = torch.stack([
        torch.arange(data.num_nodes, device=data.x.device),
        new_assignment
    ])
    values = torch.ones(data.num_nodes, device=data.x.device)
    P_hat = torch.sparse_coo_tensor(indices, values, (data.num_nodes, total_new_nodes)).to_dense()

    # Features: LogSumExp for kept clusters, Identity for split nodes
    # X'_{C_i} = log sum exp (X_u)
    
    X_prime = torch.zeros(total_new_nodes, data.x.size(1), device=data.x.device)
    
    # For split nodes (identity)
    # The split nodes are mapped to indices >= num_kept
    # The original indices are split_nodes
    X_prime[num_kept:] = data.x[split_nodes]
    
    # For kept clusters
    # We iterate kept clusters to apply LSE
    # (Vectorized LSE might be tricky with variable sizes, loop is safer for now)
    for i, c_id in enumerate(kept_clusters):
        # finding original nodes in this cluster
        mask = (cluster_assignment == c_id)
        nodes_in_c = data.x[mask]
        # LSE along node dimension (0)
        # LSE(x) = log(sum(exp(x)))
        lse = torch.logsumexp(nodes_in_c, dim=0)
        X_prime[i] = lse

    # Weights and Edges
    # W = P_hat^T A P_hat (Do we normalize P for link-wise? 
    # Paper says: W = P^T A P. Usually P is normalized if it's spectral approximation. 
    # But for interpretation, usually we want summed weights?
    # Text says: "Edges ... aggregated into weighted super-edges... W = P^T A P... D' = P^T D P"
    # And Definition 3 uses \hat{P}. Wait.
    # Theorem 3 says "Given partition matrix \hat{P}... W = \hat{P}^T A \hat{P} ... degree ... D' = \hat{P}^T D \hat{P}".
    # Then it says "Normalization ... P = \hat{P} M^{-1/2} ... we will use this normalized P for subsequent steps".
    # BUT Link-wise section says "Similar aggregation rule ... W = P^T A P".
    # And "Node features ... X' = log sum exp".
    # It implies we use UN-normalized P (\hat{P}) for W if we want counts, or normalized P if we want spectral.
    # However, "X' = log sum exp" is explicitly for features.
    # Let's stick to the definition: W_{(a,b)} = \hat{P}^T A \hat{P}. 
    # This implies \hat{P} (0/1). 
    # Let's check text again: "Weighted adjacency ... W = \hat{P}^T A \hat{P}". Yes, hat is used.
    
    # So we use binary P_hat for W calculation.
    
    adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
    W_prime_link = P_hat.t() @ adj @ P_hat
    
    edge_index_prime, edge_attr_prime = dense_to_sparse_with_attr(W_prime_link)
    
    data_link = Data(x=X_prime, edge_index=edge_index_prime, edge_attr=edge_attr_prime)
    
    return data_link

