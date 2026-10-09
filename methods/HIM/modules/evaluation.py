from modules.diffusion.IndependentCascadesModel import IndependentCascadesModel
from modules.diffusion.ThresholdModel import ThresholdModel
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import networkx as nx
import numpy as np
from tqdm import tqdm

def evaluate_influence_spread(G, seed, diffusion='IC', R=10):
    total_infect = []

    for _ in tqdm(range(R), desc='evaluating influence...'):

        if diffusion == 'LT':
            model = ThresholdModel(G)
            config = mc.Configuration()
            for n in G.nodes():
                p = np.random.uniform(0.3, 0.6)
                config.add_node_configuration("threshold", n, p)

        elif diffusion == 'IC':
            model = IndependentCascadesModel(G)
            config = mc.Configuration()
            if G.is_directed():
                for e in G.edges():
                    # p = np.random.uniform(0.1, 0.3)
                    # config.add_edge_configuration("threshold", e, p)
                    config.add_edge_configuration("threshold", e, 1 / G.in_degree(e[1]))
            else:
                for e in G.edges():
                    # p = np.random.uniform(0.1, 0.3)
                    # config.add_edge_configuration("threshold", e, p)
                    config.add_edge_configuration("threshold", e, 1 / nx.degree(G)[e[1]])

        elif diffusion == 'SIS':
            model = ep.SISModel(G)
            config = mc.Configuration()
            config.add_model_parameter('beta', 0.001)
            config.add_model_parameter('lambda', 0.001)
        else:
            raise ValueError('Only IC, LT and SIS are supported.')

        config.add_model_initial_configuration("Infected", seed)
        model.set_initial_status(config)
        iterations = model.iteration_bunch(100)
        node_status = iterations[0]['status']

        for j in range(1, len(iterations)):
            node_status.update(iterations[j]['status'])

        # final status of all nodes
        inf_vec = np.array(list(node_status.values()))
        inf_vec[inf_vec == 2] = 1
        total_infect.append(inf_vec.sum())

    total_infect = np.array(total_infect)
    mean = np.mean(total_infect)
    std = np.std(total_infect)

    return mean, std
