import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.hyperbolic.lorentz_function import Lorentzian_vector, squared_lorentzian_distance

class HIM(nn.Module):
    def __init__(self,
                 num_node,
                 degrees,
                 c,
                 dim,
                 init_size,
                 num_negs,
                 device
                 ):
        super(HIM, self).__init__()
        self.device = device
        # for HGCN
        if c is not None:
            self.c = torch.tensor([c])
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.c = self.c.to(self.device)
        self.n_nodes = num_node
        self.n_dim = dim
        self.init_size = init_size
        self.num_negs = num_negs

        self.directed = False
        self.degrees = torch.tensor(degrees, dtype=torch.float).to(self.device)  # out degree if directed
        self.max_degree = self.degrees.max()

        self.embs = torch.nn.Embedding(self.n_nodes, self.n_dim)
        # self.embs.weight.data = init_size * torch.randn(
        #     (self.n_nodes, dim))
        self.embs_init()

        self.bias_fr = nn.Embedding(self.n_nodes, 1)
        self.bias_fr.weight.data = torch.zeros((self.n_nodes, 1))
        self.bias_to = nn.Embedding(self.n_nodes, 1)
        self.bias_to.weight.data = torch.zeros((self.n_nodes, 1))

        self.social_rot = nn.Embedding(2, self.n_dim)
        self.social_rot.weight.data[:, ::2] = 1.0
        self.social_rot.weight.data[:, 1::2] = 0.0

        self.diffusion_rot = nn.Embedding(2, self.n_dim)
        self.diffusion_rot.weight.data[:, ::2] = 1.0
        self.diffusion_rot.weight.data[:, 1::2] = 0.0

    def embs_init(self):
        init_degree_emb = self.degrees.unsqueeze(1).expand(self.n_nodes, self.n_dim) + 1
        init_degree_emb = 1.0 / init_degree_emb * torch.randn(init_degree_emb.size()).to(self.device)
        self.embs.weight.data = init_degree_emb * self.init_size

    def forward(self, edges):

        frs = edges[:, 0:1]
        tos = edges[:, 1:2]
        to_negs = self.get_neg_samples(tos)
        positive_score = self.score(frs, tos, rot_type='social')
        if self.directed:
            pos_degrees = F.embedding(tos, self.in_deg.unsqueeze(1)) + 1
        else:
            pos_degrees = F.embedding(tos, self.degrees.unsqueeze(1))
        positive_score = positive_score * 1 / pos_degrees
        negative_score = self.score(frs, to_negs, rot_type='social')
        positive_loss = F.logsigmoid(positive_score).sum()  # b*L*1 -> num
        negative_loss = F.logsigmoid(-negative_score).sum()  # b*L*neg -> num
        loss = -(positive_loss + negative_loss)
        return loss

    def train_dynamic_embeds(self, graph):

        frs = graph[:, 0:1]
        tos = graph[:, 1:2]
        to_negs = self.get_neg_samples(tos)
        positive_score = self.score(frs, tos, rot_type='diffusion')

        # With Weight Adjustment
        if self.directed:
            pos_degrees = F.embedding(tos, self.in_deg.unsqueeze(1)) + 1
        else:
            pos_degrees = F.embedding(tos, self.degrees.unsqueeze(1))
        positive_score = positive_score * 1 / pos_degrees

        negative_score = self.score(frs, to_negs, rot_type='diffusion')

        positive_loss = F.logsigmoid(positive_score).sum()  # b*L*1 -> num
        negative_loss = F.logsigmoid(-negative_score).sum()  # b*L*neg -> num
        edge_loss = -(positive_loss + negative_loss)

        # Active Adjustment
        active_loss, active_sum_loss = self.activate_adjustment(frs)
        # batch_loss = edge_loss - active_loss
        batch_loss = edge_loss + active_sum_loss

        return batch_loss

    def activate_adjustment(self, frs):
        active_embs = self.embs(frs)
        frs_weights = F.embedding(frs.squeeze(1), self.degrees.unsqueeze(1)) / self.max_degree
        frs_weights = frs_weights
        active_embs = Lorentzian_vector(active_embs, self.c)
        Lorentzian_origin = torch.zeros(active_embs.shape).to(active_embs)
        Lorentzian_origin[:, :, 0] = torch.sqrt(self.c)
        score = squared_lorentzian_distance(active_embs, Lorentzian_origin, self.c)
        score = score * frs_weights
        sum_loss = score.sum()
        score = score.neg()
        loss = F.logsigmoid(score).sum()
        return loss, sum_loss

    def get_neg_samples(self, data):
        neg_samples = torch.randint(self.n_nodes, (data.shape[0], self.num_negs)).to(self.device)

        return neg_samples

    def score(self, frs, tos, rot_type=None):
        if rot_type == None:
            raise ValueError("Please set the rotation type.")
        context_vecs = self.get_embs(frs, rot_type=rot_type, direct=0)  # b*l
        target_gold_vecs = self.get_embs(tos, rot_type=rot_type, direct=1)  # b*t
        dist_score = self.similarity_score(context_vecs, target_gold_vecs)  # b*l*t
        bias_frs = self.bias_fr(frs)  # b*l*1
        bias_tos = self.bias_to(tos).permute(0, 2, 1)  # b*t*1 ->b*1*t
        score = dist_score + bias_frs + bias_tos

        return score

    def similarity_score(self, context_vecs, target_vecs):
        c = self.c
        x = context_vecs  # b*L*d
        y = target_vecs  # b*t*d
        x2_srt = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + c)  # b*L*1
        y2_srt = -torch.sqrt(torch.sum(y * y, dim=-1, keepdim=True) + c)  # b*t*1
        u = torch.cat((x, x2_srt), dim=-1)  # (b,L, d+1)
        v = torch.cat((y, y2_srt), dim=-1)  # (b,t, d+1)
        vt = v.permute(0, 2, 1)  # (b,d+1,t)
        uv = torch.bmm(u, vt)  # (b,L,t)
        result = - 2 * c - 2 * uv
        simi_score = result.neg()

        return simi_score

    def get_embs(self, node_idx, rot_type, direct):
        embs = self.embs(node_idx)
        if node_idx is None:
            raise ValueError("[node_idx] is None!")
        if rot_type == 'social':
            if direct == 0:
                rot_idx = torch.zeros(node_idx.shape, dtype=torch.long).to(node_idx.device)
            elif direct == 1:
                rot_idx = torch.ones(node_idx.shape, dtype=torch.long).to(node_idx.device)
            rot_mat = self.social_rot(rot_idx)
        elif rot_type == 'diffusion':
            if direct == 0:
                rot_idx = torch.zeros(node_idx.shape, dtype=torch.long).to(node_idx.device)
            elif direct == 1:
                rot_idx = torch.ones(node_idx.shape, dtype=torch.long).to(node_idx.device)
            rot_mat = self.diffusion_rot(rot_idx)
        else:
            raise ValueError("[rot_type] should be set as one of ('social', 'diffusion').")
        rot_embs = givens_rotations(rot_mat.view(-1, self.n_dim), embs.view(-1, self.n_dim)).view(embs.size())
        return rot_embs


def givens_rotations(r, x):
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    x = x.view((r.shape[0], -1, 2))
    x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)

    return x_rot.view((r.shape[0], -1))
