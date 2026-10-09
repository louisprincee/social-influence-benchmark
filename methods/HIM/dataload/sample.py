import logging
import os
import pickle
import random

from tqdm import tqdm

from dataload.datatool import DataTool as DT
from modules.diffusion import DiffusionTools
from utils.tools import MyTool as MT


def find_all_paths(edges, start_node):
    """Finds all paths from a given start node using DFS on edge list."""
    all_paths = []

    def dfs(current_path):
        node = current_path[-1]

        # Check if we reached a dead-end (no further movement possible)
        has_next = False
        for src, tgt in edges:
            if src == node and tgt not in current_path:  # Ensure forward direction & no cycles
                has_next = True
                dfs(current_path + [tgt])  # Continue DFS with new path

        # If no new nodes were added, it's a complete path
        if not has_next:
            all_paths.append(current_path)

    # Start DFS from the given start node
    dfs([start_node])

    return all_paths


# Example usage
edges = [(0, 1), (1, 2), (2, 3), (0, 3), (3, 4)]  # Edge list

all_paths = find_all_paths(edges, 0)

class Sampler:

    def __init__(self,
                 data_name,
                 G,
                 diffusion,
                 num_seed=10,
                 num_bunch=100,
                 num_instances=5,
                 seed_sample_mode='random',
                 diffusion_tool=None,
                 ):
        self.data_name = data_name
        self.G = G
        self.num_instances = num_instances
        self.num_node = G.number_of_nodes()
        self.is_directed = G.is_directed()
        self.diffusion = diffusion
        if diffusion_tool is None:
            self.DiffModel = DiffusionTools(data_name, G, diffusion=diffusion, num_bunch=num_bunch)
        else:
            self.DiffModel = diffusion_tool
        self.num_seed = num_seed
        self.mode = seed_sample_mode

        self.data_path = os.path.join(MT.get_root_path(), 'data', data_name)
        self.sorted_degree_idx = DT.sort_nodes_by_degree(self.G)


    def _seed_generation(self, seed_ratio=0.2):
        if self.mode == 'random':  # random generation of num_seed seeds
            seed = random.sample(range(self.num_node), self.num_seed)
        elif self.mode == 'uneven':  # uneven sampling
            pass
        elif self.mode == 'degree':
            num_tgt = int(self.num_node * seed_ratio)
            sorted_indices = self.sorted_degree_idx[:num_tgt]
            seed = random.sample(sorted_indices, self.num_seed)

        return seed

    def _sampling_cascades(self, seed_group):
        """ Return a list of lists containing diffusion instances. """
        num_instances = len(seed_group)
        file_name = MT.get_diffusion_file_name(self.data_name, self.diffusion, self.mode, num_instances)
        instance_file = os.path.join(self.data_path, file_name)

        if os.path.exists(instance_file):
            with open(instance_file, 'rb') as f:
                seed_group = pickle.load(f)
                all_cascades = pickle.load(f)

            logging.info(f"load {len(all_cascades)} propagation instances")

        else:
            all_cascades = []
            for seed in seed_group:
                # return a list of edges
                cascade = self.DiffModel.sampling_instance(seed)
                all_cascades.append(cascade)
            with open(instance_file, 'wb') as f:
                pickle.dump(seed_group, f)
                pickle.dump(all_cascades, f)

            logging.info(f"save {num_instances} propagation instances in file: {instance_file}")

        return seed_group, all_cascades

    def sampling_for_HIM(self, seed_ratio=0.2):
        save_file = f"{self.data_name}_{self.diffusion}.cas"
        save_path = os.path.join(MT.get_data_path(), self.data_name, save_file)

        all_cascades = []
        logging.info(f"sampling in [{self.mode}] mode")
        for i in tqdm(range(self.num_instances), desc="sampling cascades for HIM"):
            seed = self._seed_generation(seed_ratio=seed_ratio)
            cascade = self.DiffModel.sampling_instance(seed)
            all_cascades.append(cascade)
        return all_cascades