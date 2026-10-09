from modules.diffusion.IndependentCascadesModel import IndependentCascadesModel
from modules.diffusion.ThresholdModel import ThresholdModel
from modules.diffusion.ParameterLTModel import ParameterThresholdModel
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import networkx as nx
import numpy as np
from tqdm import tqdm
import copy
import random
import os
from utils import MyTool as MT
import pickle
import logging
from modules.diffusion.components import _generate_prob_Id, generate_diffusion_probability

class DiffusionTools:

    prob_pair = {
        'default': [(0.0, 0.2), (0.1, 0.8), (0.5, 0.9)],
        'cora_ml': [(0.1, 0.2), (0.1, 0.8), (0.5, 0.9)],
        'power_grid': [(0.1, 0.2), (0.0, 0.99), (0.5, 0.9)],
        'politician': [(0, 0.1), (0, 1), (0.5, 0.9)],
        'github': [(0.001, 0.09), (0.1, 0.99), (0.5, 0.9)],
        'texas': [(0, 0.2), (0.1, 0.99), (0.3, 0.5)],
        'youtube': [(0, 0.2), (0.1, 0.99), (0.7, 0.9)],
        'flixster': [(0, 0.05), (0.1, 0.99), (0.98, 1.0)],
    }

    def __init__(self,
                 data_name,
                 graph,
                 diffusion='IC',
                 num_bunch=100,
                 random_type='degree'
                 ):
        self.data_name = data_name
        self.G = graph
        self.diffusion = diffusion
        self.num_bunch = num_bunch

        # type includes: ['random', 'degree_random', 'degree']
        self.random_type = random_type

        self.config_path = os.path.join(MT.get_root_path(), 'configs', 'diffusion')
        os.makedirs(self.config_path, exist_ok=True)

        if data_name not in DiffusionTools.prob_pair.keys():
            data_name = 'default'
        self.true_low, self.true_high = DiffusionTools.prob_pair[data_name][0]
        self.test_low, self.test_high = DiffusionTools.prob_pair[data_name][1]
        self.node_low, self.node_high = DiffusionTools.prob_pair[data_name][2]

        self._load_diffusion_probs()
        self.config = self._config_diffusion_model()

    def reset_config(self):
        logging.info("reset config as random probs")
        self.true_probs = generate_diffusion_probability(
            self.G,
            self.diffusion,
            random_type='random'
        )
        self._config_diffusion_model()

    def _load_diffusion_probs(self):
        file_name = f"{self.data_name}_{self.diffusion}_{self.random_type}_probs.pkl"
        file_path = os.path.join(self.config_path, file_name)

        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                prob_id = pickle.load(f)
                true_probs = pickle.load(f)
                test_probs = pickle.load(f)
            logging.info(f"load {file_name} with Id: {prob_id}")

        else:
            prob_id = _generate_prob_Id()
            true_probs = generate_diffusion_probability(
                self.G, self.diffusion, self.random_type,
                low_prob=self.true_low, high_prob=self.true_high,
            )

            test_probs = true_probs

            if self.random_type != 'degree':
                logging.info(f"save {file_name} with Id: {prob_id}")
                with open(file_path, 'wb') as f:
                    pickle.dump(prob_id, f)
                    pickle.dump(true_probs, f)
                    pickle.dump(test_probs, f)
            else:
                logging.info(f"load degree probability settings")
        self.prob_id = prob_id
        self.true_probs = true_probs
        self.test_probs = test_probs

    def load_probs(self):
        return self.true_probs

    def _config_diffusion_model(self):
        config = mc.Configuration()
        if self.diffusion == 'LT':
            for n in self.G.nodes():
                p = self.true_probs[n]
                config.add_node_configuration("threshold", n, p)
        elif self.diffusion == 'WIC':
            for e in self.G.edges():
                p = self.true_probs[e]
                config.add_edge_configuration("threshold", e, p)

        elif self.diffusion == 'WLT':
            for n in self.G.nodes():
                p = random.uniform(self.node_low, self.node_high)
                # p = self.true_probs['node'][n]
                config.add_node_configuration("threshold", n, p)

        else:
            raise ValueError('Only WIC and WLT are supported.')
        logging.info(f"config true diffusion state with Id: {self.prob_id}")
        return config

    def _diffusion_simulation(self, seed):
        config = copy.deepcopy(self.config)
        if self.diffusion == 'LT':
            model = ThresholdModel(self.G)
        elif self.diffusion == 'WIC':
            model = IndependentCascadesModel(self.G)
        elif self.diffusion == 'WLT':
            config = mc.Configuration()
            for n in self.G.nodes():
                p = random.uniform(self.node_low, self.node_high)
                # p = self.true_probs['node'][n]
                config.add_node_configuration("threshold", n, p)
            model = ParameterThresholdModel(self.G)
            model.edge_probs = self.true_probs['edge']
        else:
            raise ValueError('Only WIC and WLT are supported.')
        config.add_model_initial_configuration("Infected", seed)
        model.set_initial_status(config)
        iterations = model.iteration_bunch(self.num_bunch)
        return model, iterations

    # One function is used to evaluation and another is used to generate samples

    def spread_evaluation(self, seed, num_rounds):

        total_infect = []
        for i in range(num_rounds):
            _, iterations = self._diffusion_simulation(seed)
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

    # used to generate diffusion data

    def sampling_instance(self, seed):
        model, _ = self._diffusion_simulation(seed)
        return model.cascades

    def sampling_influence_vector(self, seed):
        """ Generating training data for DeepIM (ICML 2023). """
        model, iterations = self._diffusion_simulation(seed)
        node_status = iterations[0]['status']
        for j in range(1, len(iterations)):
            node_status.update(iterations[j]['status'])
        inf_vec = np.array(list(node_status.values()))
        inf_vec[inf_vec == 2] = 1
        return inf_vec


if __name__ == '__main__':
    from dataload.dataset import DataLoad

    methods = {
        1: 'GCNM',
        2: 'PIANO',
        3: 'ToupleGDD',
        4: 'SGNN',
        5: 'IMINFECTOR',
        6: 'DeepIM'
    }

    datasets = {
        1: 'power_grid',
        2: 'cora_ml',
        3: 'jazz',
        4: 'epinions',
        5: 'pokec'
    }

    method = methods[1]
    data_name = datasets[1]

    MT.set_easy_logging("TEST")

    diffusion_model = 'IC'
    Data = DataLoad(data_name, diffusion_model)
    G = Data.social_G

    diffusion_model = DiffusionTools(
        data_name=data_name,
        graph=G,
        diffusion='PLT',
        num_bunch=100,
        random_type='degree'
    )

    seeds = random.sample(range(G.number_of_nodes()), 10)

    res, res_std = diffusion_model.spread_evaluation(seeds, 2)

    print(res)
    print(res_std)

