
import matplotlib.pyplot as plt
import os

# Data collected from benchmark logs
results = [
    {'dataset': 'Cora', 'edges': 10556, 'time': 0.060265},
    {'dataset': 'CiteSeer', 'edges': 9104, 'time': 0.070394},
    {'dataset': 'PubMed', 'edges': 88648, 'time': 0.567532},
    {'dataset': 'Amazon-Photo', 'edges': 238162, 'time': 0.179596},
    {'dataset': 'Amazon-Computers', 'edges': 491722, 'time': 0.354250},
    {'dataset': 'Coauthor-CS', 'edges': 163788, 'time': 0.535559},
    {'dataset': 'Coauthor-Physics', 'edges': 495924, 'time': 1.441850},
    {'dataset': 'ogbl-ddi', 'edges': 2135822, 'time': 0.090220},
    {'dataset': 'ogbl-collab', 'edges': 2358104, 'time': 12.114132},
]

# Sort
results.sort(key=lambda x: x['edges'])

edges = [r['edges'] for r in results]
runtimes = [r['time'] for r in results]
names = [r['dataset'] for r in results]

plt.figure(figsize=(10, 8))
plt.loglog(edges, runtimes, 'o-', label='Ours (Coarsening)')


texts = []
for i, txt in enumerate(names):
    # plt.annotate(txt, (edges[i], runtimes[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    texts.append(plt.text(edges[i], runtimes[i], txt, fontsize=9))

# Try auto-adjust if library exists, else just annotate
try:
    from adjustText import adjust_text
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
except ImportError:
    pass

plt.xlabel('Number of Edges |E|')
plt.ylabel('Time per Subgraph (s)')
plt.title('Runtime Scaling vs Edge Count')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

os.makedirs('plots', exist_ok=True)
plt.savefig('plots/runtime_scaling_extended.pdf')
print("Saved plot to plots/runtime_scaling_extended.pdf")
