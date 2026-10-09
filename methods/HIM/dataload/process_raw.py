import networkx as nx
import pickle
from utils import MyTool as MT
import os
from tqdm import tqdm


raw_path = os.path.join(MT.get_data_path(), 'raw', 'SNAP')

data_names = {
    1: 'p2p-Gnutella08.txt',
    2: 'CA-GrQc.txt',
    3: 'soc-Epinions1.txt',
    4: 'com-youtube.ungraph.txt',
    5: 'roadNet-TX.txt',
    6: 'WikiTalk.txt'
}

save_names = {
    data_names[1]: 'Gnutella08',
    data_names[2]: 'GrQc',
    data_names[3]: 'epi',
    data_names[4]: 'youtube',
    data_names[5]: 'texas',
    data_names[6]: 'wiki'
}

CHECK_DIRECTED = {
    data_names[1]: True,
    data_names[2]: False,
    data_names[3]: False,
    data_names[4]: False,
    data_names[5]: False,
    data_names[6]: False,
}

data_name = data_names[6]
save_name = save_names[data_name]

edge_file = os.path.join(raw_path, data_name)

# Dictionary to map node names to indices
node_to_index = {}
edges = []

with open(edge_file, 'r', encoding="utf-8") as f:
    for line in tqdm(f, desc=f"Loading edges for [{data_name}]"):
        line = line.strip()
        if not line or line.startswith('#'):  # Skip empty lines and comments
            continue

        src, tgt = line.split('\t')  # Tab-separated

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

