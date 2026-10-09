import torch
from tqdm import tqdm
from modules.hyperbolic.lorentz_function import Lorentzian_vector, squared_lorentzian_distance
from methods.HIM.seed_selection import adaptive_HBO_selection, adaptive_HBO_window_selection


class Runner:

    def __init__(self, model, optimizer, epochs, device):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.num_epoch = epochs

    def learn_influence_embeds(self, edge_loader, epoch, mode="SOC"):

        with tqdm(total=len(edge_loader), desc=f"[{epoch:>03d}/{self.num_epoch:>03d}] learn {mode} edges") as pbar:

            for i, batch in enumerate(edge_loader, start=1):
                edges = batch[0].to(self.device)
                num_batch = len(edges)
                if mode == "SOC":
                    loss = self.model(edges)
                else:
                    loss = self.model.train_dynamic_embeds(edges)
                self.optimizer.zero_grad()
                loss = loss / num_batch
                loss.backward()
                self.optimizer.step()

                if i % 20 == 0:
                    pbar.set_postfix({
                        'iter': f'{i}/{len(edge_loader)}',
                        'loss': loss.item()
                    })

                pbar.update(1)

    def seed_set_selection(self, seed_num, method='MD',
                           window_weight=1.0, G=None):

        emb = self.model.embs.weight
        if method == 'MD':
            # By HBO
            emb = Lorentzian_vector(emb)
            origin = torch.zeros(emb.shape).to(emb.device)
            origin[:, 0] = 1
            emb_norm = squared_lorentzian_distance(emb, origin)
            sorted_norm, indices = torch.sort(emb_norm)
            node_idx = indices.tolist()
            seeds = node_idx[:seed_num]
            return seeds

        elif method == 'AW':
            return adaptive_HBO_selection(G, emb, seed_num)

        elif method == 'ASW':
            num_nodes = G.number_of_nodes()
            # w_size = int(0.2 * num_nodes)
            w_size = int(window_weight * seed_num)
            return adaptive_HBO_window_selection(G, emb, seed_num, w_size=w_size)

