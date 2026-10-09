import os
import networkx as nx
from utils import MyTool as MT
import pickle

def dfs_paths_to_end(G, node, path, all_paths):
    """Recursive DFS to find all paths from node to an end node (sink)."""
    path.append(node)

    if G.out_degree(node) == 0:  # If node has no outgoing edges, it's an end node
        all_paths.append(list(path))  # Store the full path

    else:
        for neighbor in G.successors(node):
            dfs_paths_to_end(G, neighbor, path, all_paths)

    path.pop()  # Backtrack


def _get_all_paths_to_end(G, start_node):
    """Finds all paths from start_node that reach an end node."""
    all_paths = []
    dfs_paths_to_end(G, start_node, [], all_paths)
    return all_paths

class DataTool:

    @staticmethod
    def load_random_graph(n, p):
        graph = nx.erdos_renyi_graph(n=n, p=p)
        return graph

    @staticmethod
    def load_raw_graph(data_name='pokec'):
        # TODO: not finished yet
        raw_path = os.path.join(MT.get_data_path(), 'raw')
        graph_path = os.path.join(MT.get_data_path(), data_name)
        os.makedirs(graph_path, exist_ok=True)

        G = nx.Graph()

        graph_file = os.path.join(graph_path, f"{data_name}.G")
        with open(graph_file, 'rb') as f:
            pickle.dump(G, f)

    @staticmethod
    def sort_nodes_by_degree(G):
        """ Return sorted nodes by degree. """
        if G.is_directed():
            return [node for node, _ in sorted(G.out_degree(), key=lambda x: x[1], reverse=True)]
        else:
            return [node for node, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)]

    @staticmethod
    def get_all_paths_to_end(G, start_node):
        return _get_all_paths_to_end(G, start_node)