    node_sizes = scale_sizes(nodes)
    edge_sizes = scale_sizes(edges)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

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
    axes[0].set_xlabel('Number of Edges |E|')
    axes[0].set_ylabel('Time per Subgraph (s)')
    axes[0].set_title('Runtime vs Edges (Node Count Encoded)')
    axes[0].grid(True, which="both", ls="-", alpha=0.4)
    fig.colorbar(scatter_edges, ax=axes[0], label='log10(|V|)')

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
    axes[1].set_xlabel('Number of Nodes |V|')
    axes[1].set_ylabel('Time per Subgraph (s)')
    axes[1].set_title('Runtime vs Nodes (Edge Count Encoded)')
    axes[1].grid(True, which="both", ls="-", alpha=0.4)
    fig.colorbar(scatter_nodes, ax=axes[1], label='log10(|E|)')

    for i, txt in enumerate(names):
        xytext = (0, 10) if i % 2 == 0 else (0, -15)
        axes[0].annotate(txt, (edges[i], runtimes[i]), textcoords="offset points", xytext=xytext, ha='center', fontsize=8)
        axes[1].annotate(txt, (nodes[i], runtimes[i]), textcoords="offset points", xytext=xytext, ha='center', fontsize=8)
    
    os.makedirs('plots', exist_ok=True)