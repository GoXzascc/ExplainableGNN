import torch
import argparse
import os
import mlflow
import mlflow.pytorch

import torch_geometric as pyg
import torch.nn.functional as F

from loguru import logger
from datetime import datetime


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


def config() -> argparse.Namespace:
    """
    Config for structure performance curve.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="PubMed", choices=["Cora", "CiteSeer", "PubMed"]
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
        "--edge_removal_rate", type=float, default=0.9
    )  # Percentage of edges to remove
    parser.add_argument(
        "--remove_edges_in_training",
        action="store_true",
        help="Remove edges during training (default: only during val/test)",
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
    data = dataset[0].to(args.device)

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
    removal_rates = [i * 0.05 for i in range(int(args.edge_removal_rate / 0.05) + 1)]
    for remove_edges_rate in removal_rates:
        # Remove edges from original (don't modify in place)
        test_edge_index, _ = remove_edges_randomly(
            original_edge_index, remove_edges_rate, preserve_mask=preserve_mask
        )

        # Test with modified graph
        with torch.no_grad():
            out = model(data.x, test_edge_index)
            # Test metrics
            test_pred = out[data.test_mask].argmax(dim=1)
            test_correct = (test_pred == data.y[data.test_mask]).sum()
            test_acc = int(test_correct) / int(data.test_mask.sum())

            logger.info(
                f"Remove edges rate: {remove_edges_rate:.2f}, Test Acc: {test_acc:.4f}"
            )
            mlflow.log_metric(f"test_acc_removal_{remove_edges_rate:.2f}", test_acc)

    # End mlflow run
    mlflow.end_run()


if __name__ == "__main__":
    structure_performance_curve()
