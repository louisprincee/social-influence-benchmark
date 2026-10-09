import os
import pickle

from utils.tools import MyTool as MT


class DataLoad:

    def __init__(self,
                 data_name='power_grid',
                 diffusion_model='IC',
                 walk_length=3,
                 num_walks=1,
                 ):
        self.data_name = data_name
        self.diffusion_model = diffusion_model
        self.data_path = os.path.join(MT.get_root_path(), 'data', data_name)

        self.walk_length = walk_length
        self.num_walks = num_walks

        # load social network
        self.social_G = self._load_social_network()
        self.num_node = self.social_G.number_of_nodes()
        self.num_edge = self.social_G.number_of_edges()

    def _load_social_network(self):
        social_path = os.path.join(self.data_path, self.data_name + '.G')
        with open(social_path, 'rb') as f:
            social_network = pickle.load(f)
        return social_network

class HIMData(DataLoad):

    def __init__(self, data_name='power_grid', diffusion_model='IC'):
        super(HIMData, self).__init__(data_name, diffusion_model)

        self.diffusion_G = self._load_cascade_samples()

    def _load_cascade_samples(self):
        cas_file = self.data_name + '_' + self.diffusion_model + '.cas'
        cascades_path = os.path.join(self.data_path, cas_file)
        with open(cascades_path, 'rb') as f:
            diffusion_edges = pickle.load(f)
        return diffusion_edges


