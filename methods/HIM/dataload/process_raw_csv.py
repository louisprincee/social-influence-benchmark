import networkx as nx
import pickle
from utils import MyTool as MT
import os
from tqdm import tqdm
import pandas as pd


raw_path = os.path.join(MT.get_data_path(), 'raw', 'SNAP_CSV')

data_names = {
    1: 'musae_git_edges.csv',
}

save_names = {
    data_names[1]: 'github',
}

CHECK_DIRECTED = {
    data_names[1]: False,
}

data_name = data_names[1]
save_name = save_names[data_name]

edge_file = os.path.join(raw_path, data_name)

# Dictionary to map node names to indices
node_to_index = {}
edges = []

df = pd.read_csv(edge_file)
all_edges = df.values

for edge in tqdm(all_edges):
    src, tgt = edge

    if src not in node_to_index:
        node_to_index[src] = len(node_to_index)

    if tgt not in node_to_index:
        node_to_index[tgt] = len(node_to_index)

    src = node_to_index[src]
    tgt = node_to_index[tgt]

    edges.append((src, tgt))

if CHECK_DIRECTED[data_name]:
    print(f"construct a directed graph")
    G = nx.DiGraph()
else:
    print(f"construct an undirected graph")
    G = nx.Graph()

G.add_edges_from(edges)
n_node = G.number_of_nodes()
n_edge = G.number_of_edges()

save_path = os.path.join(MT.get_data_path(), save_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_path = os.path.join(save_path, f'{save_name}.G')
with open(save_path, 'wb') as f:
    pickle.dump(G, f)

print(f"Graph {save_name} has been built with [{n_node}] nodes and [{n_edge}] edges")

