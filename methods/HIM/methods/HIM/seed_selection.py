import heapq
from tqdm import tqdm
import torch
from modules.hyperbolic.lorentz_function import Lorentzian_vector, squared_lorentzian_distance


def adaptive_HBO_selection(G, embs, seed_num):
    gamma = 1.0
    embs = embs.detach().cpu()
    embs = Lorentzian_vector(embs, gamma)
    n_node = embs.shape[0]
    origin = torch.zeros(embs.shape)
    origin[:, 0] = 1  # Lorentz origin
    HBO = squared_lorentzian_distance(embs, origin)
    remove_tag = 2 ** 10

    adaptive_HBO = {}
    for idx in range(n_node):
        adaptive_HBO[idx] = HBO[idx].item()

    seed_set = set()
    for i in tqdm(range(seed_num), unit='ex'):
        c_node, c_HBO = min(adaptive_HBO.items(), key=lambda u: u[1])
        seed_set.add(c_node)
        adaptive_HBO[c_node] = remove_tag
        c_out_degree = G.out_degree(c_node) if G.is_directed() else G.degree(c_node)
        c_p = 1 / c_out_degree if c_out_degree != 0 else 0
        for n_node in list(G.neighbors(c_node)):
            if adaptive_HBO[n_node] != remove_tag:
                n_HBO = adaptive_HBO[n_node]
                dis = squared_lorentzian_distance(embs[c_node].unsqueeze(0), embs[n_node].unsqueeze(0), gamma=gamma)
                adaptive_HBO[n_node] = c_p * dis + n_HBO
    assert len(seed_set) == seed_num
    seed_set = list(seed_set)
    return seed_set


class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.entry_finder = {}
        self.REMOVED = -1
        self.counter = 0

    def add_or_update(self, key, value):
        if key in self.entry_finder:
            self.remove_key(key)
        count = self.counter
        self.counter += 1
        entry = [value, count, key]
        self.entry_finder[key] = entry
        heapq.heappush(self.heap, entry)

    def remove_key(self, key):
        entry = self.entry_finder.pop(key)
        entry[-1] = self.REMOVED

    def pop_key(self):
        while self.heap:
            value, count, key = heapq.heappop(self.heap)
            if key is not self.REMOVED:
                del self.entry_finder[key]
                return key, value
        raise KeyError('pop from an empty priority queue')

    def current_keys(self):
        keys = torch.tensor(self.heap)[:, -1]
        valid_keys = torch.nonzero(keys != self.REMOVED).flatten()
        keys = keys[valid_keys].to(torch.long)
        return keys

    def current_values(self):
        values = torch.tensor(self.heap)[:, 0]
        return values

    def is_empty(self):
        return not self.entry_finder


def adaptive_HBO_window_selection(G, embs, seed_num, w_size=10):
    gamma = 1.0
    embs = embs.detach().cpu()
    embs = Lorentzian_vector(embs, gamma)
    origin = torch.zeros(embs.shape)
    origin[:, 0] = 1  # Lorentz origin
    HBO = squared_lorentzian_distance(embs, origin)
    # base_score = HBO.mean()
    sorted_index = torch.argsort(HBO, descending=False)
    # candidates = sorted_index[1: w_size + 1]  #
    candidates_queue = PriorityQueue()
    for i in range(w_size):
        key = sorted_index[i + 1].item()  # Key is node index
        value = HBO[key].item()  # Value is HBO_score
        candidates_queue.add_or_update(key, value)

    window_position = w_size + 1
    c_node = sorted_index[0].item()
    seed_set = set()
    for i in tqdm(range(seed_num), unit='ex'):

        seed_set.add(c_node)
        c_neighbors = list(G.neighbors(c_node))
        current_targets = candidates_queue.current_keys().tolist()
        common_nodes = find_common_nodes(c_neighbors, current_targets)
        if len(common_nodes) == 0:
            c_node = candidates_queue.pop_key()[0]
            key = sorted_index[window_position].item()
            value = HBO[key].item()
            candidates_queue.add_or_update(key, value)
            window_position += 1
        else:
            tgt_embs = embs[common_nodes]
            tgt_distance = squared_lorentzian_distance(embs[c_node], tgt_embs)
            tgt_weights = torch.softmax(1 / tgt_distance, dim=-1)
            c_HBO_score = HBO[c_node]
            # c_HBO_score = base_score
            c_degree = G.out_degree(c_node) if G.is_directed() else G.degree(c_node)
            c_weights = 1 / c_degree if c_degree != 0 else 0
            update_socres = c_HBO_score * c_weights * tgt_weights
            update_socres = update_socres.tolist()
            for key, value in zip(common_nodes, update_socres):
                value = candidates_queue.entry_finder[key][0] + value
                candidates_queue.add_or_update(key, value)
            c_node = candidates_queue.pop_key()[0]
            key = sorted_index[window_position].item()
            value = HBO[key].item()
            candidates_queue.add_or_update(key, value)
            window_position += 1

    return seed_set


def find_common_nodes(node_set_1, node_set_2):
    node_set_1 = set(node_set_1)
    node_set_2 = set(node_set_2)
    common_nodes = node_set_1.intersection(node_set_2)
    return list(common_nodes)
