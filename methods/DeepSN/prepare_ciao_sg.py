"""Build a small DeepSN .SG file from the shared Ciao undirected trust graph.

Training cascades use the same synthetic WIC rule as HIM:
p(u,v)=p(v,u)=1/max(degree(u), degree(v)). Training random streams are
separate from later independent evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import random
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from scipy.sparse import csr_matrix

METHOD_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = METHOD_ROOT.parents[1]
SHARED_DATA = REPOSITORY_ROOT / "data" / "processed" / "ciao"
DEFAULT_OUTPUT = METHOD_ROOT / "data" / "ciao_mean_IC50.SG"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_shared_graph() -> tuple[nx.Graph, str, str]:
    manifest_path = SHARED_DATA / "manifest.json"
    user_map_path = SHARED_DATA / "user_map.csv"
    edge_path = SHARED_DATA / "trust_edges_undirected.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Shared Ciao tables are missing. Download the official MAT archive, "
            "then run: python data/process_ciao.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hashes = manifest["generated_csv_sha256"]
    for name, path in (("user_map.csv", user_map_path), ("trust_edges_undirected.csv", edge_path)):
        actual = sha256_file(path)
        if actual != expected_hashes[name]:
            raise ValueError(f"Shared Ciao export hash mismatch for {name}")

    graph = nx.Graph()
    with user_map_path.open(newline="", encoding="utf-8") as source:
        for expected_index, row in enumerate(csv.DictReader(source)):
            node = int(row["user_index"])
            if node != expected_index:
                raise ValueError("Ciao user indices must be contiguous from zero")
            graph.add_node(node)
    with edge_path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            left = int(row["user_index_a"])
            right = int(row["user_index_b"])
            if left >= right:
                raise ValueError(f"Invalid undirected edge ({left}, {right})")
            graph.add_edge(left, right)
    return graph, expected_hashes["trust_edges_undirected.csv"], expected_hashes["user_map.csv"]


def edge_probability(graph: nx.Graph, source: int, target: int) -> float:
    return 1.0 / max(graph.degree(source), graph.degree(target))


def weighted_cascade(graph: nx.Graph, seeds: list[int]) -> list[int]:
    active = set(seeds)
    frontier = list(seeds)
    while frontier:
        node = frontier.pop()
        for neighbor in graph.neighbors(node):
            if neighbor in active:
                continue
            if random.random() <= edge_probability(graph, node, neighbor):
                active.add(neighbor)
                frontier.append(neighbor)
    return [1 if node in active else 0 for node in range(graph.number_of_nodes())]


def generate_pairs(graph: nx.Graph, seed_rate: int, n_pairs: int, repeats: int) -> torch.Tensor:
    n = graph.number_of_nodes()
    k = int(round(n * seed_rate / 100.0))
    xs = []
    ys = []
    for _ in range(n_pairs):
        seeds = random.sample(range(n), k)
        indicator = [0] * n
        for node in seeds:
            indicator[node] = 1
        outcomes = [weighted_cascade(graph, seeds) for _ in range(repeats)]
        mean_outcome = np.mean(np.array(outcomes), axis=0).tolist()
        xs.append(indicator)
        ys.append(mean_outcome)
    return torch.stack((torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-rate", type=int, default=5)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    graph, edge_hash, user_hash = load_shared_graph()
    pairs = generate_pairs(graph, args.seed_rate, args.pairs, args.repeats)
    adj = nx.to_scipy_sparse_array(graph, nodelist=list(range(graph.number_of_nodes())), dtype=np.float32)
    payload = {
        "adj": csr_matrix(adj),
        "inverse_pairs": pairs,
        "meta": {
            "dataset": "ciao",
            "diffusion": "IC",
            "probability": "p(u,v)=1/max(degree(u), degree(v))",
            "probability_id": "ciao_symmetric_weighted_cascade_v1_f2fc09305d5f",
            "seed_rate_percent": args.seed_rate,
            "training_pairs": args.pairs,
            "repeats_per_pair": args.repeats,
            "generation_seed": args.seed,
            "undirected_edge_csv_sha256": edge_hash,
            "user_map_sha256": user_hash,
            "interpretation": "Synthetic WIC training examples on the Ciao trust topology; not real adoption labels.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    with args.output.open("wb") as handle:
        pickle.dump(payload, handle)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "pairs": args.pairs,
                "sha256": sha256_file(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
