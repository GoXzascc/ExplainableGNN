import torch
import argparse
import os
import mlflow
import mlflow.pytorch

import torch_geometric as pyg
import torch.nn.functional as F

from loguru import logger
from datetime import datetime
from functools import cache
from tqdm import tqdm
import hashlib


def tensor_cache(maxsize=128):
    """
    Custom cache decorator that works with PyTorch tensors.
    
    Args:
        maxsize: Maximum size of the cache
    """
    def decorator(func):
        cache_dict = {}
        cache_order = []
        
        def tensor_to_key(tensor):
            """Convert tensor to a hashable key."""
            if hasattr(tensor, 'detach'):
                # PyTorch tensor
                tensor_bytes = tensor.detach().cpu().numpy().tobytes()
                tensor_shape = tuple(tensor.shape)
                tensor_dtype = str(tensor.dtype)
                # Create hash from bytes, shape, and dtype
                hash_input = tensor_bytes + str(tensor_shape).encode() + tensor_dtype.encode()
                return hashlib.md5(hash_input).hexdigest()
            else:
                # Regular Python object
                return hash(tensor)
        
        def make_key(*args, **kwargs):
            """Create a cache key from function arguments."""
            key_parts = []
            
            # Handle positional arguments
            for arg in args:
                if hasattr(arg, 'detach'):  # PyTorch tensor
                    key_parts.append(tensor_to_key(arg))
                else:
                    key_parts.append(str(arg))
            
            # Handle keyword arguments
            for k, v in sorted(kwargs.items()):
                if hasattr(v, 'detach'):  # PyTorch tensor
                    key_parts.append(f"{k}:{tensor_to_key(v)}")
                else:
                    key_parts.append(f"{k}:{v}")
            
            return tuple(key_parts)
        
        def wrapper(*args, **kwargs):
            # Create cache key
            key = make_key(*args, **kwargs)
            
            # Check if result is cached
            if key in cache_dict:
                return cache_dict[key]
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache with LRU eviction
            if len(cache_dict) >= maxsize:
                # Remove oldest entry
                oldest_key = cache_order.pop(0)
                del cache_dict[oldest_key]
            
            cache_dict[key] = result
            cache_order.append(key)
            
            return result
        
        # Add cache info method
        def cache_info():
            return f"Cache size: {len(cache_dict)}/{maxsize}"
        
        def cache_clear():
            cache_dict.clear()
            cache_order.clear()
        
        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        
        return wrapper
    return decorator


def remove_edges_randomly(
    edge_index, edge_removal_rate, preserve_mask=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly remove edges from the graph.

    Args:
        edge_index: Edge index tensor of shape [2, num_edges]
        edge_removal_rate: Fraction of edges to remove (0.0 to 1.0)
        preserve_mask: Boolean mask indicating which edges to preserve (optional)

    Returns:
        Modified edge_index with some edges removed
        Indices of removed edges
    """
    num_edges = edge_index.size(1)
    num_edges_to_remove = int(num_edges * edge_removal_rate)

    if num_edges_to_remove == 0:
        return edge_index, torch.tensor([])

    # Create a mask for all edges
    edge_mask = torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)

    # If preserve_mask is provided, don't remove preserved edges
    if preserve_mask is not None:
        available_edges = torch.where(~preserve_mask)[0]
    else:
        available_edges = torch.arange(num_edges, device=edge_index.device)

    # Randomly select edges to remove from available edges
    if len(available_edges) > 0:
        num_available = len(available_edges)
        num_to_remove = min(num_edges_to_remove, num_available)

        # Randomly select indices from available edges
        perm = torch.randperm(num_available, device=edge_index.device)
        edges_to_remove = available_edges[perm[:num_to_remove]]

        # Update the mask
        edge_mask[edges_to_remove] = False

        # Return filtered edge_index and removed edge indices
        return edge_index[:, edge_mask], edges_to_remove
    else:
        return edge_index, torch.tensor([])


def remove_edges_structure(edge_index, edge_removal_rate, preserve_mask=None):
    """
    Remove edges based on structural importance scores.

    Args:
        edge_index: Edge index tensor of shape [2, num_edges]
        edge_removal_rate: Fraction of edges to remove (0.0 to 1.0)
        preserve_mask: Boolean mask indicating which edges to preserve (optional)

    Returns:
        Modified edge_index with some edges removed
        Indices of removed edges
    """
    num_edges = edge_index.size(1)
    num_edges_to_remove = int(num_edges * edge_removal_rate)
    
    if num_edges_to_remove == 0:
        return edge_index, torch.tensor([])
    
    # Compute structural importance scores for all edges
    edge_scores = compute_edge_structural_scores(edge_index)
    
    # Create a mask for all edges
    edge_mask = torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)
    
    # If preserve_mask is provided, don't remove preserved edges
    if preserve_mask is not None:
        available_edges = torch.where(~preserve_mask)[0]
        available_scores = edge_scores[available_edges]
    else:
        available_edges = torch.arange(num_edges, device=edge_index.device)
        available_scores = edge_scores
    
    # Remove edges with lowest structural importance scores
    if len(available_edges) > 0:
        num_available = len(available_edges)
        num_to_remove = min(num_edges_to_remove, num_available)
        
        # Sort by scores (ascending) and select lowest scoring edges
        _, sorted_indices = torch.sort(available_scores, descending=False)
        edges_to_remove = available_edges[sorted_indices[:num_to_remove]]
        
        # Update the mask
        edge_mask[edges_to_remove] = False
        
        # Return filtered edge_index and removed edge indices
        return edge_index[:, edge_mask], edges_to_remove
    else:
        return edge_index, torch.tensor([])

@tensor_cache(maxsize=64)
def compute_edge_structural_scores(edge_index):
    """
    Compute structural importance scores for all edges based on multiple metrics.
    
    Args:
        edge_index: Edge index tensor of shape [2, num_edges]
        
    Returns:
        Tensor of shape [num_edges] with structural importance scores
    """
    device = edge_index.device
    num_edges = edge_index.size(1)
    num_nodes = edge_index.max().item() + 1
    
    # Convert to undirected for structural analysis
    edge_index_undirected = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index_undirected = torch.unique(edge_index_undirected, dim=1)
    
    # 1. Degree centrality of connected nodes
    degree = torch.zeros(num_nodes, device=device)
    degree.scatter_add_(0, edge_index_undirected[0], torch.ones(edge_index_undirected.size(1), device=device))
    degree.scatter_add_(0, edge_index_undirected[1], torch.ones(edge_index_undirected.size(1), device=device))
    
    # 2. Edge betweenness centrality (approximated)
    edge_betweenness = compute_edge_betweenness_approx(edge_index, num_nodes)
    
    # 3. Common neighbors (structural embeddedness)
    common_neighbors = compute_common_neighbors(edge_index, num_nodes)
    
    # 4. Clustering coefficient contribution
    clustering_contribution = compute_clustering_contribution(edge_index, num_nodes)
    
    # 5. Bridge score (is the edge a bridge?)
    bridge_scores = compute_bridge_scores(edge_index, num_nodes)
    
    # Combine all metrics into a single importance score
    # Higher score = more important edge
    src_nodes = edge_index[0]
    tgt_nodes = edge_index[1]
    
    # Normalize each component to [0, 1]
    degree_scores = (degree[src_nodes] + degree[tgt_nodes]) / 2
    degree_scores = (degree_scores - degree_scores.min()) / (degree_scores.max() - degree_scores.min() + 1e-8)
    
    edge_betweenness = (edge_betweenness - edge_betweenness.min()) / (edge_betweenness.max() - edge_betweenness.min() + 1e-8)
    common_neighbors = (common_neighbors - common_neighbors.min()) / (common_neighbors.max() - common_neighbors.min() + 1e-8)
    clustering_contribution = (clustering_contribution - clustering_contribution.min()) / (clustering_contribution.max() - clustering_contribution.min() + 1e-8)
    bridge_scores = (bridge_scores - bridge_scores.min()) / (bridge_scores.max() - bridge_scores.min() + 1e-8)
    
    # Weighted combination of metrics
    importance_scores = (
        0.3 * degree_scores +           # Higher degree = more important
        0.25 * edge_betweenness +       # Higher betweenness = more important  
        0.2 * common_neighbors +        # More common neighbors = more important
        0.15 * clustering_contribution + # Higher clustering = more important
        0.1 * bridge_scores             # Bridge edges = more important
    )
    
    return importance_scores


@tensor_cache(maxsize=32)
def compute_edge_betweenness_approx(edge_index, num_nodes):
    """
    Approximate edge betweenness centrality using random sampling.
    """
    device = edge_index.device
    num_edges = edge_index.size(1)
    
    # Simple approximation: edges connecting high-degree nodes have higher betweenness
    degree = torch.zeros(num_nodes, device=device)
    degree.scatter_add_(0, edge_index[0], torch.ones(num_edges, device=device))
    degree.scatter_add_(0, edge_index[1], torch.ones(num_edges, device=device))
    
    src_degree = degree[edge_index[0]]
    tgt_degree = degree[edge_index[1]]
    
    # Edges between high-degree nodes are likely to be on shortest paths
    betweenness_approx = src_degree * tgt_degree
    
    return betweenness_approx.float()


@tensor_cache(maxsize=32)
def compute_common_neighbors(edge_index, num_nodes):
    """
    Compute the number of common neighbors for each edge.
    """
    device = edge_index.device
    num_edges = edge_index.size(1)
    
    # More efficient approach: use sparse operations
    # Create adjacency lists instead of dense matrix
    adj_list = [[] for _ in range(num_nodes)]
    for i in range(num_edges):
        src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
        adj_list[src].append(tgt)
        adj_list[tgt].append(src)  # Undirected
    
    common_neighbors = torch.zeros(num_edges, device=device)
    
    for i in range(num_edges):
        src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
        
        # Convert to sets for efficient intersection
        src_neighbors = set(adj_list[src])
        tgt_neighbors = set(adj_list[tgt])
        
        # Count common neighbors
        common_count = len(src_neighbors.intersection(tgt_neighbors))
        common_neighbors[i] = float(common_count)
    
    return common_neighbors


@tensor_cache(maxsize=32)
def compute_clustering_contribution(edge_index, num_nodes):
    """
    Compute how much each edge contributes to local clustering.
    """
    device = edge_index.device
    num_edges = edge_index.size(1)
    
    # Use adjacency lists for efficiency
    adj_list = [[] for _ in range(num_nodes)]
    for i in range(num_edges):
        src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
        adj_list[src].append(tgt)
        adj_list[tgt].append(src)  # Undirected
    
    clustering_contribution = torch.zeros(num_edges, device=device)
    
    for i in range(num_edges):
        src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
        
        # Get neighbors of both nodes
        src_neighbors = set(adj_list[src])
        tgt_neighbors = set(adj_list[tgt])
        
        # Count triangles this edge participates in
        common_count = len(src_neighbors.intersection(tgt_neighbors))
        clustering_contribution[i] = float(common_count)
    
    return clustering_contribution


@tensor_cache(maxsize=32)
def compute_bridge_scores(edge_index, num_nodes):
    """
    Compute bridge scores (simplified - edges with few alternative paths).
    """
    device = edge_index.device
    num_edges = edge_index.size(1)
    
    # Use adjacency lists for efficiency
    adj_list = [[] for _ in range(num_nodes)]
    for i in range(num_edges):
        src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
        adj_list[src].append(tgt)
        adj_list[tgt].append(src)  # Undirected
    
    bridge_scores = torch.zeros(num_edges, device=device)
    
    for i in range(num_edges):
        src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
        
        # Simple heuristic: if removing this edge significantly reduces connectivity
        # Count the number of length-2 paths between src and tgt
        paths_via_others = 0
        src_neighbors = set(adj_list[src])
        tgt_neighbors = set(adj_list[tgt])
        
        # Count common neighbors (alternative paths of length 2)
        for neighbor in src_neighbors:
            if neighbor != tgt and neighbor in tgt_neighbors:
                paths_via_others += 1
        
        # Bridge score is inversely related to alternative paths
        bridge_scores[i] = 1.0 / (1.0 + paths_via_others)
    
    return bridge_scores


def get_preserve_edges_mask(edge_index, val_mask, test_mask):
    """
    Create a mask to preserve edges connected to validation or test nodes.

    Args:
        edge_index: Edge index tensor of shape [2, num_edges]
        val_mask: Boolean mask for validation nodes
        test_mask: Boolean mask for test nodes

    Returns:
        Boolean mask indicating which edges to preserve
    """
    # Get indices of validation and test nodes
    val_nodes = torch.where(val_mask)[0]
    test_nodes = torch.where(test_mask)[0]

    val_test_nodes = torch.cat([val_nodes, test_nodes])

    # Create a mask for edges that connect to val/test nodes
    preserve_mask = torch.zeros(
        edge_index.size(1), dtype=torch.bool, device=edge_index.device
    )

    # Check if either source or target node is in val/test set
    for i, (src, tgt) in enumerate(edge_index.t()):
        if src in val_test_nodes or tgt in val_test_nodes:
            preserve_mask[i] = True

    return preserve_mask


def extract_k_hop_subgraph(data, k: int = 2, max_nodes: int = 5000, seed: int = 42) -> pyg.data.Data:
    """
    Extract a k-hop subgraph from a large graph.
    
    Args:
        data: PyG Data object
        k: Number of hops
        max_nodes: Maximum number of nodes in subgraph
        seed: Random seed for reproducible sampling
    
    Returns:
        Subgraph as PyG Data object
    """
    torch.manual_seed(seed)
    
    num_nodes = data.x.size(0) if hasattr(data, 'x') and data.x is not None else data.edge_index.max().item() + 1
    
    if num_nodes <= max_nodes:
        logger.info(f"Graph has {num_nodes} nodes, no subgraph extraction needed")
        return data
    
    logger.info(f"Extracting {k}-hop subgraph from {num_nodes} nodes")
    
    # Start with random seed nodes
    num_seed_nodes = min(max_nodes // (k + 1), 100)
    seed_nodes = torch.randperm(num_nodes)[:num_seed_nodes]
    
    # Perform k-hop sampling
    subgraph_nodes = set(seed_nodes.tolist())
    current_nodes = seed_nodes
    
    for hop in range(k):
        # Find neighbors of current nodes
        neighbors = []
        for node in current_nodes:
            # Find all neighbors of this node
            mask = (data.edge_index[0] == node) | (data.edge_index[1] == node)
            node_edges = data.edge_index[:, mask]
            node_neighbors = torch.cat([node_edges[0], node_edges[1]]).unique()
            neighbors.extend(node_neighbors.tolist())
        
        # Add new neighbors to subgraph
        new_neighbors = [n for n in neighbors if n not in subgraph_nodes]
        
        # Limit the number of new nodes to prevent explosion
        if len(subgraph_nodes) + len(new_neighbors) > max_nodes:
            remaining_slots = max_nodes - len(subgraph_nodes)
            new_neighbors = new_neighbors[:remaining_slots]
        
        subgraph_nodes.update(new_neighbors)
        current_nodes = torch.tensor(new_neighbors)
        
        logger.info(f"Hop {hop + 1}: Added {len(new_neighbors)} nodes, total: {len(subgraph_nodes)}")
        
        if len(subgraph_nodes) >= max_nodes:
            break
    
    # Convert to tensor and create node mapping
    subgraph_node_list = list(subgraph_nodes)
    subgraph_node_tensor = torch.tensor(subgraph_node_list)
    
    # Create mapping from original to new node indices
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(subgraph_node_list)}
    
    # Extract subgraph edges
    edge_mask = torch.isin(data.edge_index[0], subgraph_node_tensor) & torch.isin(data.edge_index[1], subgraph_node_tensor)
    subgraph_edge_index = data.edge_index[:, edge_mask]
    
    # Remap edge indices
    subgraph_edge_index[0] = torch.tensor([old_to_new[idx.item()] for idx in subgraph_edge_index[0]])
    subgraph_edge_index[1] = torch.tensor([old_to_new[idx.item()] for idx in subgraph_edge_index[1]])
    
    # Create subgraph data object
    subgraph_data = pyg.data.Data()
    subgraph_data.edge_index = subgraph_edge_index
    subgraph_data.num_nodes = len(subgraph_nodes)
    
    # Extract node features if available
    if hasattr(data, 'x') and data.x is not None:
        subgraph_data.x = data.x[subgraph_node_tensor]
        subgraph_data.num_features = data.x.size(1)
    else:
        # Create dummy features
        subgraph_data.x = torch.eye(subgraph_data.num_nodes)
        subgraph_data.num_features = subgraph_data.num_nodes
    
    # Create train/val/test masks for the subgraph
    subgraph_size = len(subgraph_nodes)
    train_ratio, val_ratio = 0.6, 0.2
    
    indices = torch.randperm(subgraph_size)
    train_size = int(train_ratio * subgraph_size)
    val_size = int(val_ratio * subgraph_size)
    
    subgraph_data.train_mask = torch.zeros(subgraph_size, dtype=torch.bool)
    subgraph_data.val_mask = torch.zeros(subgraph_size, dtype=torch.bool)
    subgraph_data.test_mask = torch.zeros(subgraph_size, dtype=torch.bool)
    
    subgraph_data.train_mask[indices[:train_size]] = True
    subgraph_data.val_mask[indices[train_size:train_size + val_size]] = True
    subgraph_data.test_mask[indices[train_size + val_size:]] = True
    
    # Create dummy labels for node classification
    subgraph_data.y = torch.randint(0, 10, (subgraph_size,))  # 10 classes
    subgraph_data.num_classes = 10
    
    logger.info(f"Created subgraph with {subgraph_data.num_nodes} nodes and {subgraph_data.edge_index.size(1)} edges")
    
    return subgraph_data


def config() -> argparse.Namespace:
    """
    Config for structure performance curve.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="PubMed", choices=["Cora", "CiteSeer", "PubMed", "ogbl-collab"]
    )
    parser.add_argument("--model", type=str, default="GCN")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--log_dir", type=str, default="logs/structure_performance_curve"
    )
    parser.add_argument("--seed", type=int, default=432)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument(
        "--epochs", type=int, default=300
    )  # Increase epochs for better convergence
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--hidden_channels", type=int, default=64
    )  # Reduce to 64 for Cora
    parser.add_argument(
        "--num_layers", type=int, default=2
    )  # 2 layers is often optimal for Cora
    parser.add_argument("--dropout", type=float, default=0.5)  # Add dropout parameter
    parser.add_argument(
        "--activation", type=str, default="relu"
    )  # Add activation parameter
    parser.add_argument(
        "--edge_removal_rate", type=float, default=0.9999
    )  # Percentage of edges to remove
    parser.add_argument(
        "--remove_edges_in_training",
        action="store_true",
        help="Remove edges during training (default: only during val/test)",
    )
    parser.add_argument(
        "--edge_removal_method",
        type=str,
        default="structural",
        choices=["random", "structural"],
        help="Method for edge removal: random or structural",
    )
    parser.add_argument(
        "--subgraph_hops",
        type=int,
        default=2,
        help="Number of hops for subgraph extraction (for large datasets)",
    )
    parser.add_argument(
        "--max_nodes",
        type=int,
        default=5000,
        help="Maximum number of nodes in subgraph",
    )

    return parser.parse_args()


def structure_performance_curve() -> None:
    """
    Structure performance curve.
    """

    # Config
    args = config()
    logger.add(
        os.path.join(args.log_dir, "structure_performance_curve.log"),
        level="INFO",
        rotation="100 MB",
        retention="10 days",
    )
    logger.info(f"args: {args}")

    # Set seed
    pyg.seed_everything(args.seed)

    # Load dataset
    dataset = pyg.datasets.Planetoid(root=args.log_dir, name=args.dataset)
    data = dataset[0]

    data = data.to(args.device)

    # Store original edge statistics
    original_num_edges = data.edge_index.size(1)
    logger.info(f"Original number of edges: {original_num_edges}")

    # Load model
    model = pyg.nn.models.GCN(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        out_channels=dataset.num_classes,
        dropout=args.dropout,  # Add dropout for regularization
        act=args.activation,  # Ensure ReLU activation
    ).to(args.device)

    # Set mlflow
    mlflow.set_experiment("structure_performance_curve")
    mlflow.start_run(
        run_name=f"structure_performance_curve_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    mlflow.pytorch.autolog()

    # Log hyperparameters
    mlflow.log_params(
        {
            "dataset": args.dataset,
            "model": args.model,
            "device": args.device,
            "seed": args.seed,
            "num_runs": args.num_runs,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_channels": args.hidden_channels,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "activation": args.activation,
            "edge_removal_rate": args.edge_removal_rate,
            "remove_edges_in_training": args.remove_edges_in_training,
            "edge_removal_method": args.edge_removal_method,
            "subgraph_hops": args.subgraph_hops,
            "max_nodes": args.max_nodes,
            "original_num_edges": original_num_edges,
        }
    )

    # Train model
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_acc = 0
    best_test_acc = 0
    best_epoch = 0  # Initialize best_epoch
    for run in range(args.num_runs):
        pyg.seed_everything(args.seed + run)
        for epoch in range(args.epochs):
            # Training - use original full graph
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)  # Use original edges for training
            loss = F.cross_entropy(
                out[data.train_mask], data.y[data.train_mask]
            )  # Use cross_entropy instead of nll_loss
            loss.backward()
            optimizer.step()

            # Validation and Testing - use graph with removed edges
            model.eval()
            with torch.no_grad():

                out = model(
                    data.x, data.edge_index
                )  # Use modified edges for validation/testing

                # Validation metrics
                val_pred = out[data.val_mask].argmax(dim=1)
                val_correct = (val_pred == data.y[data.val_mask]).sum()
                val_acc = int(val_correct) / int(data.val_mask.sum())

                # Test metrics
                test_pred = out[data.test_mask].argmax(dim=1)
                test_correct = (test_pred == data.y[data.test_mask]).sum()
                test_acc = int(test_correct) / int(data.test_mask.sum())

                # Training metrics (use original graph for consistency)
                train_out = model(data.x, data.edge_index)
                train_pred = train_out[data.train_mask].argmax(dim=1)
                train_correct = (train_pred == data.y[data.train_mask]).sum()
                train_acc = int(train_correct) / int(data.train_mask.sum())

                # Keep track of best performance
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_epoch = epoch

                    # save model
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.log_dir, f"best_model_{best_epoch:03d}.pth"),
                    )

            # Logging
            mlflow.log_metrics(
                {
                    "train_loss": loss.item(),
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                },
                step=epoch,
            )

            if epoch % 10 == 0 or epoch == args.epochs - 1:
                logger.info(
                    f"Run {run:3d}, Epoch {epoch:3d}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Best Epoch: {best_epoch:3d}"
                )

    # Final results
    logger.info(f"Run {run:3d}, Best Validation Accuracy: {best_val_acc:.4f}")
    logger.info(f"Run {run:3d}, Best Test Accuracy: {best_test_acc:.4f}")
    mlflow.log_metrics(
        {
            "best_val_acc": best_val_acc,
            "best_test_acc": best_test_acc,
            "final_test_acc": test_acc,
        }
    )
    # save model
    torch.save(
        model.state_dict(),
        os.path.join(args.log_dir, f"final_model_{args.epochs:03d}.pth"),
    )
    logger.info(
        f"Saved final model to {os.path.join(args.log_dir, f'final_model_{args.epochs:03d}.pth')}"
    )

    # Test the best model with different edge removal rates
    logger.info("Testing best model with different edge removal rates...")
    original_edge_index = data.edge_index.clone()  # Preserve original
    preserve_mask = get_preserve_edges_mask(
        original_edge_index, data.val_mask, data.test_mask
    )

    # Load best model
    model.load_state_dict(
        torch.load(os.path.join(args.log_dir, f"best_model_{best_epoch:03d}.pth"))
    )
    model.eval()

    # Test with different removal rates
    removal_rates = [i * 0.00005 for i in range(18000, int(args.edge_removal_rate / 0.00005) + 1)]
    
    logger.info(f"Starting edge removal testing with {len(removal_rates)} different rates")
    logger.info(f"Cache info before testing - Structural scores: {compute_edge_structural_scores.cache_info()}")
    
    for remove_edges_rate in tqdm(removal_rates):
        # Remove edges from original (don't modify in place)
        if args.edge_removal_method == "structural":
            test_edge_index, _ = remove_edges_structure(
                original_edge_index, remove_edges_rate, preserve_mask=None
            )
        else:  # random
            test_edge_index, _ = remove_edges_randomly(
                original_edge_index, remove_edges_rate, preserve_mask=None
            )

        # Test with modified graph
        with torch.no_grad():
            out = model(data.x, test_edge_index)
            # Test metrics
            test_pred = out[data.test_mask].argmax(dim=1)
            test_correct = (test_pred == data.y[data.test_mask]).sum()
            test_acc = int(test_correct) / int(data.test_mask.sum())

            logger.info(
                f"Remove edges rate: {remove_edges_rate:.4f}, Method: {args.edge_removal_method}, Test Acc: {test_acc:.4f}"
            )
            mlflow.log_metric(f"test_acc_removal_{remove_edges_rate:.4f}_{args.edge_removal_method}", test_acc)

    # Log final cache statistics
    logger.info(f"Cache info after testing - Structural scores: {compute_edge_structural_scores.cache_info()}")
    logger.info(f"Cache info - Betweenness: {compute_edge_betweenness_approx.cache_info()}")
    logger.info(f"Cache info - Common neighbors: {compute_common_neighbors.cache_info()}")
    logger.info(f"Cache info - Clustering: {compute_clustering_contribution.cache_info()}")
    logger.info(f"Cache info - Bridge scores: {compute_bridge_scores.cache_info()}")

    # End mlflow run
    mlflow.end_run()


if __name__ == "__main__":
    structure_performance_curve()
