import argparse
import hashlib
import json
import logging
import os.path
import random

import geoopt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataload import Sampler
from dataload.dataset import DataLoad
from methods.HIM.HIM import HIM
from modules.diffusion import DiffusionTools
from utils.runtime import Runner
from utils.tools import MyTool as MT

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="HIM")
    parser.add_argument('--data_name', choices=['cora_ml', 'power_grid', 'facebook', 'github', 'youtube', 'ciao'],
                        default='cora_ml')
    parser.add_argument('--diffusion_model', default='WLT')
    parser.add_argument('--seed_ratio', type=float, default=10)
    parser.add_argument('--seed_num', type=int, default=None)
    parser.add_argument('--random_type', default='random')
    parser.add_argument('--num_instances', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--R', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=None)

    # model parameters
    parser.add_argument('--pre_spread', type=bool, default=False)
    parser.add_argument('--method', default='ASW')
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--save_emb', default=True)

    args = parser.parse_args()

    model_name = HIM.__name__

    data_name = args.data_name
    args.data_name = data_name
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    save_path = MT.set_save_path(
        data_name=args.data_name,  model_name=model_name, diffusion=args.diffusion_model, seed_rate=args.seed_ratio)
    MT.set_logging(save_path, model_name)

    m_args, m_cfgs = MT.load_model_params(f"HIM_{args.diffusion_model}", data_name)

    Data = DataLoad(data_name, args.diffusion_model)

    # social network
    graph = Data.social_G
    degrees = dict(graph.degree())
    degrees = list(degrees.values())

    num_node = graph.number_of_nodes()
    num_edge = graph.number_of_edges()

    seed_ratio = args.seed_ratio
    seed_num = args.seed_num if args.seed_num is not None else int(num_node * args.seed_ratio * 0.01)
    if not 0 < seed_num <= num_node:
        parser.error(f"seed count must be in [1, {num_node}]")

    logging.info(f"Basic Settings:")
    logging.info(f" data name             : {data_name}")
    logging.info(f" number of nodes       : {num_node}")
    logging.info(f" number of edges       : {num_edge}")
    logging.info(f" seed ratio (k)        : {seed_ratio}%")
    logging.info(f" number of seed nodes  : {seed_num}")
    logging.info(f" diffusion model       : {args.diffusion_model}")
    logging.info(f" number of eval rounds : {args.R}")
    logging.info(f" device                : {device}")
    logging.info(f" epochs                : {args.epochs}")
    logging.info(f" learning rate         : {m_args.lr}")
    logging.info(f" initial size:         : {m_cfgs.init_size}")
    logging.info(f" number of neg nodes   : {m_cfgs.num_negs}")

    evaluator = DiffusionTools(
        data_name=data_name,
        graph=graph,
        diffusion=args.diffusion_model,
        random_type=args.random_type,
    )

    sampler = Sampler(
        data_name=data_name,
        G=graph,
        diffusion=args.diffusion_model,
        num_seed=seed_num,
        num_instances=args.num_instances,
        seed_sample_mode='degree',
        diffusion_tool=evaluator,
    )

    social_edges = torch.LongTensor(list(graph.edges()))

    diff_path = os.path.join(MT.get_data_path(), 'diffusion')
    if not os.path.exists(diff_path): os.makedirs(diff_path)
    sample_ratio = m_args.sample_ratio
    graph_path = os.path.join(Data.data_path, f"{data_name}.G")
    graph_hash = hashlib.sha256()
    with open(graph_path, 'rb') as graph_file:
        for chunk in iter(lambda: graph_file.read(1024 * 1024), b''):
            graph_hash.update(chunk)
    seed_label = args.seed if args.seed is not None else 'unseeded'
    diff_file = (
        f"{data_name}_{args.diffusion_model}_k{seed_ratio:g}_b{seed_num}_n{args.num_instances}"
        f"_s{seed_label}_sr{sample_ratio:g}_{evaluator.prob_id}_{graph_hash.hexdigest()[:12]}.pkl"
    )
    diff_file = os.path.join(diff_path, diff_file)
    import pickle
    if not os.path.exists(diff_file):
        diffusion_edges = sampler.sampling_for_HIM(seed_ratio=sample_ratio)
        diffusion_edges = [torch.LongTensor(edges) for edges in diffusion_edges]
        with open(diff_file, 'wb') as f:
            pickle.dump(diffusion_edges, f)
        logging.info(f"save diffusion data at {diff_file}")
    else:
        with open(diff_file, 'rb') as f:
            diffusion_edges = pickle.load(f)
        if len(diffusion_edges) != args.num_instances:
            raise ValueError(
                f"cached cascade count {len(diffusion_edges)} does not match "
                f"--num_instances {args.num_instances}: {diff_file}"
            )
        logging.info(f"load diffusion data from {diff_file}")

    diffusion_cache_hash = hashlib.sha256()
    with open(diff_file, 'rb') as diffusion_file:
        for chunk in iter(lambda: diffusion_file.read(1024 * 1024), b''):
            diffusion_cache_hash.update(chunk)

    social_loader = DataLoader(
        TensorDataset(social_edges), batch_size=m_args.batch_size, shuffle=True, num_workers=0)

    # loader_mode = 'all'
    loader_mode = 'single'
    if loader_mode == 'all':
        diffusion_edges = torch.cat(diffusion_edges, dim=0)
        diffusion_loader = DataLoader(
            TensorDataset(diffusion_edges), batch_size=m_args.batch_size, shuffle=True, num_workers=0)
    else:
        diffusion_loader = DataLoader(diffusion_edges, batch_size=1, shuffle=True, num_workers=0)

    model = HIM(
        num_node=num_node,
        degrees=degrees,
        c=m_cfgs.c,
        dim=m_cfgs.num_dim,
        init_size=m_cfgs.init_size,
        num_negs=m_cfgs.num_negs,
        device=device,
    )
    model = model.to(device)
    optim = geoopt.optim.RiemannianAdam(params=model.parameters(), lr=float(m_args.lr), stabilize=0.99,
                                        weight_decay=0.0)
    run = Runner(model, optim, args.epochs, device)

    if args.pre_spread:
        seeds = run.seed_set_selection(seed_num, method=args.method, G=graph)
        inf_mean, inf_std = evaluator.spread_evaluation(seed=seeds, num_rounds=args.R)
        spread_ratio = (inf_mean / num_node) * 100
        spread_ratio_std = (inf_std / num_node) * 100
        logging.info(f" before training: {spread_ratio:.2f}% (std: {spread_ratio_std:.3f}%)")

    logging.info(f"start main process...")

    best_res = 0.0
    best_std = 0.0
    best_epoch = 0
    best_seeds = []
    patience_count = 0
    for epoch in range(1, args.epochs+1):

        run.learn_influence_embeds(social_loader, epoch, mode='SOC')

        run.learn_influence_embeds(diffusion_loader, epoch, mode='DIF')

        seeds = run.seed_set_selection(seed_num, method=args.method, window_weight=m_args.window_weight, G=graph)

        inf_mean, inf_std = evaluator.spread_evaluation(seed=seeds, num_rounds=args.R)
        spread_ratio = (inf_mean/num_node) * 100
        spread_ratio_std = (inf_std/num_node) * 100

        if best_res < spread_ratio:
            best_res = spread_ratio
            best_std = spread_ratio_std
            best_epoch = epoch
            best_seeds = sorted(seeds)
            torch.save(model.state_dict(), os.path.join(save_path, 'model_best_epoch.pt'))
            with open(os.path.join(save_path, 'seeds_best_epoch.json'), 'w', encoding='utf-8') as output:
                json.dump({
                    'data_name': data_name,
                    'diffusion_model': args.diffusion_model,
                    'seed_ratio_percent': args.seed_ratio,
                    'seed_count': seed_num,
                    'training_seed': args.seed,
                    'training_instances': args.num_instances,
                    'training_sample_ratio': sample_ratio,
                    'max_epochs': args.epochs,
                    'validation_rounds': args.R,
                    'early_stopping_patience': args.patience,
                    'selection_method': args.method,
                    'probability_id': evaluator.prob_id,
                    'graph_sha256': graph_hash.hexdigest(),
                    'training_cache': os.path.relpath(diff_file, MT.get_root_path()),
                    'training_cache_sha256': diffusion_cache_hash.hexdigest(),
                    'epoch': best_epoch,
                    'seeds': best_seeds,
                }, output, indent=2)
                output.write('\n')
            patience_count = 0
        else:
            patience_count += 1
            if patience_count > args.patience:
                logging.info(f"early stopping at epoch {epoch}!!!")
                break

        logging.info(f"epoch {epoch:03d} results (patience={patience_count}):")
        logging.info(f" influence spread : {inf_mean} (std: {inf_std:.3f})")
        logging.info(f" spread ratio     : {spread_ratio:.2f}% (std: {spread_ratio_std:.3f}%)")

    logging.info(f"best results at epoch {best_epoch} with {best_res:.2f}% + {best_std:.3f}%")
