import random
from datetime import datetime
from tqdm import tqdm
import numpy as np

def _generate_prob_Id():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def generate_diffusion_probability(G, diffusion='IC', random_type='degree',
                                   low_prob=0.1, high_prob=0.3):

    degree_query = None
    if random_type != 'random':
        if G.is_directed():
            degree_query = G.in_degree
        else:
            degree_query = G.degree

    pbar_desc = f"Generating probs for [{diffusion}] in type [{random_type}]"

    probs = {}

    if diffusion == 'IC':
        with tqdm(total=G.number_of_edges(), desc=pbar_desc) as pbar:
            for edge in G.edges():

                if random_type == 'random':
                    prob = random.uniform(low_prob, high_prob)
                else:
                    degree_prob = degree_query(edge[1])
                    if degree_prob == 0:
                        degree_prob = 1e-9
                    else:
                        degree_prob = 1 / degree_prob
                    prob = degree_prob
                    if random_type == 'degree_random':
                        prob = prob * random.random()

                probs[edge] = prob
                pbar.set_postfix(prob=prob)
                pbar.update(1)

    # elif diffusion == 'LT' or diffusion == 'PLT':
    #     with tqdm(total=G.number_of_nodes(), desc=pbar_desc) as pbar:
    #         for node in G.nodes():
    #             probs[node] = random.uniform(0.5, 0.9)
    #             pbar.set_postfix(prob=probs[node])
    #             pbar.update(1)


    if diffusion == 'PLT':
        PLT_probs = {
            'node': probs
        }

        edge_probs = {}

        if G.is_directed():
            incoming_query = G.predecessors
        else:
            incoming_query = G.neighbors

        with tqdm(total=G.number_of_nodes(), desc=pbar_desc) as pbar:
            for node in G.nodes():

                node_neighbors = list(incoming_query(node))
                n_neighbors = len(node_neighbors)
                if n_neighbors == 0:
                    continue

                neighbor_probs = np.random.rand(n_neighbors) + 1e-9
                # Select n unique random indices
                n = random.randint(0, n_neighbors)
                indices = np.random.choice(n_neighbors, n, replace=False)
                neighbor_probs[indices] += 1 + np.random.rand(n)
                neighbor_probs /= np.sum(neighbor_probs)

                for  i, v in enumerate(node_neighbors):
                    edge = (v, node)
                    edge_probs[edge] = neighbor_probs[i]

                pbar.set_postfix(prob=edge_probs[edge])
                pbar.update(1)

        PLT_probs['edge'] = edge_probs

        return PLT_probs

    return probs


if __name__ == '__main__':

    import networkx as nx
    from dataload import DataTool as DT
    G = DT.load_random_graph(20, 0.1)

    probs = generate_diffusion_probability(
        G, diffusion='PLT', random_type='degree',
        low_prob=0.1, high_prob=0.3)

    print("done.")

    print(probs['node'])

    print(probs['edge'])
