
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx, to_undirected
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from src.corasen import spectral_graph_coarsening

# --- Visualization ---
def visualize_graph_clustering(data, cluster_assignment, title, filename):
    """
    Visualizes the graph with nodes colored by cluster assignment.
    """
    G = to_networkx(data, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    pos = nx.kamada_kawai_layout(G)
    clusters = cluster_assignment.cpu().numpy()
    
    if data.x is not None:
        atom_types = data.x.argmax(dim=1).cpu().numpy()
        labels = {i: str(atom_types[i]) for i in range(len(G))}
    else:
        labels = None

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_color=clusters, cmap='tab20', node_size=300)
    if labels:
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='white')
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

# --- Model ---
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch=None):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

# --- Training & Eval ---
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

# --- Fidelity ---
def compute_fidelity(model, dataset, device, k_ratio=0.5):
    """
    Computes Fidelity: Fraction of samples where Model(Original) == Model(Coarsened).
    Also returns accuracy on coarsened graphs.
    """
    model.eval()
    matches = 0
    coarse_correct = 0
    total = len(dataset)
    
    # Process one by one for coarsening
    for i in range(total):
        data = dataset[i]
        data = data.to(device)
        if not data.is_undirected():
            data.edge_index = to_undirected(data.edge_index)
            
        # Original Pred
        with torch.no_grad():
            pred_orig = model(data.x, data.edge_index).argmax(dim=1).item()
            label = data.y.item()
            
        # Coarsen
        num_nodes = data.num_nodes
        k = max(2, int(num_nodes * k_ratio))
        
        try:
            coarse_data, _, _ = spectral_graph_coarsening(data, k=k, alpha=0.5)
            # Predict Coarse
            # Note: GCN needs batch index if doing single inference usually handled by checking batch=None
            with torch.no_grad():
                pred_coarse = model(coarse_data.x, coarse_data.edge_index).argmax(dim=1).item()
                
            if pred_orig == pred_coarse:
                matches += 1
            if pred_coarse == label:
                coarse_correct += 1
                
        except Exception as e:
            print(f"Coarsening failed for graph {i}: {e}")
            # If fail, count as mismatch? or skip? Let's skip to be safe, or just ignore.
            total -= 1
            
    fidelity = matches / total if total > 0 else 0
    acc_coarse = coarse_correct / total if total > 0 else 0
    return fidelity, acc_coarse

def run_experiment():
    dataset = TUDataset(root='data/TUDataset', name='MUTAG')
    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    train_dataset = dataset[:150]
    test_dataset = dataset[150:]
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = GCN(hidden_channels=64, num_node_features=dataset.num_node_features, num_classes=dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    print("Training GCN on MUTAG...")
    for epoch in range(1, 101):
        loss = train(model, train_loader, optimizer, criterion, device)
        if epoch % 20 == 0:
            train_acc = test(model, train_loader, device)
            test_acc = test(model, test_loader, device)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
            
    print("Training Complete.")
    
    # --- Fidelity Analysis ---
    print("\nEvaluating Fidelity across compression ratios...")
    ratios = [0.9, 0.7, 0.5, 0.3, 0.2] # Keep 90% nodes ... down to 20%
    fidelities = []
    coarse_accs = []
    
    for r in ratios:
        fid, acc = compute_fidelity(model, test_dataset, device, k_ratio=r)
        print(f"  Ratio {r:.1f} (Keep {int(r*100)}%): Fidelity={fid:.4f}, Coarse Acc={acc:.4f}")
        fidelities.append(fid)
        coarse_accs.append(acc)
        
    # Plot
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(ratios, fidelities, marker='o', label='Fidelity (Consistency)', linewidth=2)
    plt.plot(ratios, coarse_accs, marker='s', label='Coarse Accuracy', linestyle='--', alpha=0.7)
    
    plt.xlabel("Node Retention Ratio (k/N)")
    plt.ylabel("Score")
    plt.title("Explanation Fidelity vs Coarsening Ratio")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis() # Show high ratio (low compression) on left? Or standard axis?
    # Usually X-axis: 0.2 -> 0.9.
    # If we invert, left is 0.9 (original-like).
    plt.gca().invert_xaxis()
    
    plt.savefig('plots/explanation_fidelity.pdf')
    print("Saved plots/explanation_fidelity.pdf")
    
    # Run Qualitative visualizer on one sample just to be sure
    run_vis_demo(dataset, device)

def run_vis_demo(dataset, device):
    # Visualize index 0 with k=6
    data = dataset[0].to(device)
    if not data.is_undirected(): data.edge_index = to_undirected(data.edge_index)
    try:
        coarse_data, cluster_assignment, _ = spectral_graph_coarsening(data, k=6, alpha=0.5)
        visualize_graph_clustering(data, cluster_assignment, "Sample Visualization (k=6)", "plots/explanation_mutag_sample.pdf")
    except Exception as e:
        print(f"Vis failed: {e}")

if __name__ == "__main__":
    run_experiment()
