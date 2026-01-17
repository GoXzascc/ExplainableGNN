
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
import matplotlib.pyplot as plt
import numpy as np
import os
from src.corasen import spectral_graph_coarsening

# --- Model ---
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# --- Training ---
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = pred[mask] == data.y[mask]
    acc = int(correct.sum()) / int(mask.sum())
    return acc

# --- Fidelity ---
def compute_node_fidelity(model, data, device, k_ratio=0.5):
    """
    Computes Node Fidelity on Test Mask.
    Fidelity = P( Pred_orig[v] == Pred_coarse[ Cluster[v] ] ) for v in Test Set.
    """
    model.eval()
    
    # 1. Original Predictions on Test Nodes
    with torch.no_grad():
        out_orig = model(data.x, data.edge_index)
        pred_orig = out_orig.argmax(dim=1)
        
    # 2. Coarsen
    num_nodes = data.num_nodes
    k = max(2, int(num_nodes * k_ratio))
    
    try:
        # Coarsen
        coarse_data, cluster_assignment, _ = spectral_graph_coarsening(data, k=k, alpha=0.5)
        
        # 3. Predict on Coarse Graph
        # Note: We reuse the same GCN. 
        # Feature dimension must be preserved. (Spectral coarsening does weighted avg of features usually)
        with torch.no_grad():
            out_coarse = model(coarse_data.x, coarse_data.edge_index)
            pred_coarse = out_coarse.argmax(dim=1) # [k_nodes]
            
        # 4. Map back to original nodes
        # cluster_assignment[v] gives the supernode index for node v
        # So we want pred_coarse[cluster_assignment]
        
        mapped_pred_coarse = pred_coarse[cluster_assignment] # [num_nodes]
        
        # 5. Measure consistency on TEST mask
        mask = data.test_mask
        
        # Consistency: Did the supernode predict the same thing as the original node?
        # Note: We compare prediction vs prediction (Fidelity), not vs Ground Truth.
        matches = (pred_orig[mask] == mapped_pred_coarse[mask])
        fidelity = matches.sum().item() / mask.sum().item()
        
        # Also interesting: Accuracy of the supernodes?
        # mapped_pred_coarse == data.y
        coarse_acc = (mapped_pred_coarse[mask] == data.y[mask]).sum().item() / mask.sum().item()
        
        return fidelity, coarse_acc
        
    except Exception as e:
        print(f"  Coarsening failed at ratio {k_ratio}: {e}")
        return 0.0, 0.0

def run_node_explanation_experiment():
    print("Loading Cora dataset...")
    dataset = Planetoid(root='data/Planetoid', name='Cora')
    data = dataset[0]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data = data.to(device)
    if not data.is_undirected():
        data.edge_index = to_undirected(data.edge_index)
        
    print(f"Nodes: {data.num_nodes}, Edges: {data.edge_index.size(1)}")
    
    # Init Model
    model = GCN(num_node_features=dataset.num_node_features, hidden_channels=16, num_classes=dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Train
    print("Training GCN on Cora...")
    for epoch in range(1, 201):
        loss = train(model, data, optimizer, criterion)
        if epoch % 20 == 0:
            test_acc = test(model, data, data.test_mask)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {test_acc:.4f}')
            
    print("Training Complete.")
    
    # Fidelity Benchmark
    print("\nEvaluating Node Fidelity...")
    ratios = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1]
    fidelities = []
    coarse_accs = []
    
    for r in ratios:
        fid, acc = compute_node_fidelity(model, data, device, k_ratio=r)
        print(f"  Ratio {r:.1f} (Keep {int(r*100)}%): Fidelity={fid:.4f}, Coarse Acc={acc:.4f}")
        fidelities.append(fid)
        coarse_accs.append(acc)
        
    # Plot
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(ratios, fidelities, marker='o', label='Fidelity (Consistency)', linewidth=2)
    plt.plot(ratios, coarse_accs, marker='s', label='Coarse Test Accuracy', linestyle='--', alpha=0.7)
    
    plt.xlabel("Node Retention Ratio (k/N)")
    plt.ylabel("Score")
    plt.title("Node Explanation Fidelity (Cora)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()
    
    plt.savefig('plots/explanation_node_fidelity.pdf')
    print("Saved plots/explanation_node_fidelity.pdf")

if __name__ == "__main__":
    run_node_explanation_experiment()
