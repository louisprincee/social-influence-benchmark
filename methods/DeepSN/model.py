import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import community
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from diffusion_models import independent_cascade_model, linear_threshold_model, sis_model
from main.model.diff import DeepSN, LearnableSparseCOO
from policy import NodeScoringNN

METHOD_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = METHOD_ROOT.parents[1]
DEFAULT_RESULTS = REPOSITORY_ROOT / "results" / "DeepSN"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="DeepSN influence maximization")
    datasets = ["jazz", "cora_ml", "power_grid", "netscience", "random5", "ciao"]
    parser.add_argument("-d", "--dataset", default="jazz", type=str, help="one of: {}".format(", ".join(sorted(datasets))))
    diffusion = ["IC", "LT", "SIS"]
    parser.add_argument("-dm", "--diffusion_model", default="SIS", type=str, help="one of: {}".format(", ".join(sorted(diffusion))))
    parser.add_argument("-sp", "--seed_rate", default=20, type=int, help="seed budget as a percentage of nodes")
    parser.add_argument("-m", "--mode", default="normal", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--estimation_epochs", default=50, type=int)
    parser.add_argument("--score_epochs", default=200, type=int)
    parser.add_argument("--eval_rounds", default=100, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--data_dir", default=str(METHOD_ROOT / "data"), type=str)
    parser.add_argument("--output", default=None, type=str, help="JSON result path; default is results/DeepSN/")
    parser.add_argument("--tag", default=None, type=str, help="optional label stored in the result JSON")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def graph_from_sparse(adj_matrix):
    if hasattr(nx, "from_scipy_sparse_array"):
        graph = nx.from_scipy_sparse_array(adj_matrix)
    else:
        graph = nx.from_scipy_sparse_matrix(adj_matrix)
    return graph.to_directed()


def diffusion_evaluation(adj_matrix, seed, diffusion="LT", rounds=100):
    total_infect = 0
    graph = graph_from_sparse(adj_matrix)
    values = []
    for _ in range(rounds):
        if diffusion == "LT":
            count = linear_threshold_model(graph, seed)
        elif diffusion == "IC":
            count = independent_cascade_model(graph, seed)
        elif diffusion == "SIS":
            count = sis_model(graph, seed)
        else:
            raise ValueError(f"unsupported diffusion {diffusion}")
        value = count * 100 / graph.number_of_nodes()
        values.append(value)
        total_infect += value
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return total_infect / rounds, stdev, values


def csr_to_sparse_tensor(csr):
    coo = csr.tocoo()
    indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float)
    size = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, size)


def sparse_mul(tensor1, tensor2):
    assert tensor1._indices().equal(tensor2._indices()), "Sparsity patterns do not match"
    result_values = tensor1._values() * tensor2._values()
    result_indices = tensor1._indices()
    return torch.sparse_coo_tensor(result_indices, result_values, tensor1.size())


def estimation_loss(y, y_hat):
    return F.mse_loss(y_hat.squeeze(), y, reduction="sum")


def maximization_loss(x_hat, x_i, adj, forward_model, n):
    y_hat = forward_model(x_hat.squeeze(0), adj, x_i, n)
    return 1 - (torch.sum(y_hat)).float() / y_hat.shape[0]


def load_graph_file(data_dir: Path, dataset: str, diffusion_model: str, seed_rate: int):
    path = data_dir / f"{dataset}_mean_{diffusion_model}{10 * seed_rate}.SG"
    if not path.is_file():
        raise FileNotFoundError(f"missing DeepSN graph file: {path}")
    with path.open("rb") as handle:
        graph = pickle_load(handle)
    return path, graph


def pickle_load(handle):
    import pickle

    return pickle.load(handle)


def build_feature_matrix(adj):
    adjacency_matrix = torch.sparse_coo_tensor(
        torch.tensor([adj.tocoo().row, adj.tocoo().col], dtype=torch.long),
        torch.tensor(adj.tocoo().data, dtype=torch.float32),
        torch.Size(adj.tocoo().shape),
    )
    degree = torch.sparse.sum(torch.sparse.mm(adjacency_matrix, adjacency_matrix), dim=1).to_dense()
    unique_degrees = torch.unique(degree)
    one_hot_encoder = {deg.item(): i for i, deg in enumerate(unique_degrees)}
    feature_matrix = torch.zeros((adjacency_matrix.size(0), len(unique_degrees)))
    for i, deg in enumerate(degree):
        feature_matrix[i, one_hot_encoder[deg.item()]] = 1.0
    return feature_matrix


def write_result(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {path}")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    n = 2
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Is GPU available? {}\n".format(torch.cuda.is_available()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (Path.cwd() / data_dir).resolve()
    graph_path, graph = load_graph_file(data_dir, args.dataset, args.diffusion_model, args.seed_rate)
    graph_path = graph_path.resolve()
    adj, inverse_pairs = graph["adj"], graph["inverse_pairs"]
    feature_matrix = build_feature_matrix(adj)
    num_nodes = int(adj.shape[0])
    adj = csr_to_sparse_tensor(adj)

    batch_size = min(args.batch_size, len(inverse_pairs))
    if len(inverse_pairs) > batch_size:
        train_set, test_set = torch.utils.data.random_split(
            inverse_pairs, [len(inverse_pairs) - batch_size, batch_size]
        )
    else:
        train_set = inverse_pairs
        test_set = inverse_pairs
    train_loader = DataLoader(dataset=inverse_pairs, batch_size=batch_size, shuffle=True, drop_last=False)

    forward_model = DeepSN(
        ninput=feature_matrix.shape[1],
        nfeat=8,
        nhid=32,
        nclass=1,
        dropout=0.5,
        edge_index=adj,
    )
    attention = LearnableSparseCOO(adj.coalesce().indices(), (adj.shape[0], adj.shape[1]))
    optimizer = Adam(
        [{"params": forward_model.parameters()}, {"params": attention.parameters()}],
        lr=2e-3,
        weight_decay=1e-4,
    )
    adj = adj.to(device)
    feature_matrix = feature_matrix.to(device)
    forward_model = forward_model.to(device)
    attention = attention.to(device)
    forward_model.train()

    estimation_times = []
    started = time.time()
    for epoch in range(args.estimation_epochs):
        begin = time.time()
        total_overall = 0
        for data_pair in train_loader:
            x = data_pair[:, :, 0].float().to(device)
            y = data_pair[:, :, 1].float().to(device)
            optimizer.zero_grad()
            loss = 0
            for i, x_i in enumerate(x):
                y_i = y[i]
                y_hat = forward_model(feature_matrix.squeeze(0), sparse_mul(adj, attention()), x_i, n)
                loss = loss + estimation_loss(y_i, y_hat)
            total_overall += loss.item()
            loss = loss / x.size(0)
            loss.backward()
            optimizer.step()
        elapsed = time.time() - begin
        estimation_times.append(elapsed)
        print(
            "Epoch: {}".format(epoch + 1),
            "\tTotal: {:.4f}".format(total_overall / max(len(train_set), 1)),
            "\tTime: {:.4f}".format(elapsed),
        )

    att_adj = sparse_mul(adj, attention()).coalesce()
    data = att_adj.values().detach().cpu().numpy()
    row = att_adj.indices()[0].detach().cpu().numpy()
    col = att_adj.indices()[1].detach().cpu().numpy()
    scipy_sparse_coo = sp.coo_matrix((data, (row, col)), shape=tuple(att_adj.size()))
    community_graph = nx.Graph()
    community_graph.add_weighted_edges_from(zip(scipy_sparse_coo.row, scipy_sparse_coo.col, scipy_sparse_coo.data))
    partition = community.best_partition(community_graph, weight="weight", resolution=1)
    community_tensor = torch.zeros(num_nodes, dtype=torch.long)
    for node, comm_id in partition.items():
        community_tensor[int(node)] = comm_id

    score_model = NodeScoringNN(input_dim=feature_matrix.shape[1], hidden_dim=128, output_dim=1)
    score_optimizer = Adam(score_model.parameters(), lr=1e-4)
    budget = math.ceil(adj.shape[0] * args.seed_rate / 100)
    score_times = []
    for epoch in range(args.score_epochs):
        begin = time.time()
        score_optimizer.zero_grad()
        x_i = torch.squeeze(score_model(feature_matrix.cpu(), community_tensor, budget))
        loss = maximization_loss(feature_matrix, x_i.to(device), adj, forward_model, n)
        loss.backward()
        score_optimizer.step()
        elapsed = time.time() - begin
        score_times.append(elapsed)
        print("Epoch: {}".format(epoch + 1), "\tTotal: {:.4f}".format(loss), "\tTime: {:.4f}".format(elapsed))

    seed_mask = x_i.cpu().detach().numpy()
    seed = [index for index, value in enumerate(seed_mask) if value != 0]
    if args.eval_rounds > 0:
        influence, std, values = diffusion_evaluation(
            graph["adj"], seed, diffusion=args.diffusion_model, rounds=args.eval_rounds
        )
        print(influence)
        mean_activated = influence / 100.0 * num_nodes
    else:
        influence, std, values, mean_activated = None, None, [], None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = args.tag or "run"
    output = Path(args.output) if args.output else DEFAULT_RESULTS / f"{args.dataset}_{args.diffusion_model}_k{args.seed_rate}_{tag}_{stamp}.json"
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    payload = {
        "method": "DeepSN",
        "dataset": args.dataset,
        "diffusion_model": args.diffusion_model,
        "seed_rate_percent": args.seed_rate,
        "seed_count": len(seed),
        "seed_budget_rule": "ceil(|V| * seed_rate / 100)",
        "training_seed": args.seed,
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]] if argv is None else ["model.py", *argv],
        "device": str(device),
        "gpu_available": bool(torch.cuda.is_available()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "graph_file": str(graph_path.relative_to(REPOSITORY_ROOT)),
        "graph_file_sha256": sha256_file(graph_path),
        "nodes": num_nodes,
        "inverse_pairs": int(inverse_pairs.shape[0]),
        "estimation_epochs": args.estimation_epochs,
        "score_epochs": args.score_epochs,
        "eval_rounds": args.eval_rounds,
        "mean_coverage_percent": influence,
        "stdev_coverage_percent": std,
        "mean_activated": mean_activated,
        "seeds": seed,
        "eval_coverages_percent": values,
        "runtime_seconds": time.time() - started,
        "mean_estimation_epoch_seconds": float(np.mean(estimation_times)) if estimation_times else 0.0,
        "mean_score_epoch_seconds": float(np.mean(score_times)) if score_times else 0.0,
        "tag": args.tag,
        "interpretation": (
            "Native DeepSN training and the author diffusion evaluator. "
            "This is not a frozen cross-method ranking unless the team evaluator and graph are the same."
        ),
    }
    write_result(output, payload)
    print(f"Wrote {output}")
    return payload


if __name__ == "__main__":
    main()
