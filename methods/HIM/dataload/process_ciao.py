"""Adapt the shared Ciao export to HIM's graph and WIC-probability formats."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import networkx as nx


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SHARED_DATA = REPOSITORY_ROOT / "data" / "processed" / "ciao"
HIM_ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = HIM_ROOT / "data" / "ciao" / "ciao.G"
PROBABILITY_PATH = HIM_ROOT / "configs" / "diffusion" / "ciao_WIC_random_probs.pkl"
MANIFEST_PATH = HIM_ROOT / "data" / "ciao" / "ciao_adapter_manifest.json"
PROBABILITY_PROTOCOL = "symmetric_weighted_cascade_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_graph() -> tuple[nx.Graph, str, str]:
    manifest_path = SHARED_DATA / "manifest.json"
    user_map_path = SHARED_DATA / "user_map.csv"
    edge_path = SHARED_DATA / "trust_edges_undirected.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hashes = manifest["generated_csv_sha256"]
    for name, path in (("user_map.csv", user_map_path), ("trust_edges_undirected.csv", edge_path)):
        actual_hash = sha256_file(path)
        if actual_hash != expected_hashes[name]:
            raise ValueError(f"Shared Ciao export hash mismatch for {name}")

    graph = nx.Graph()
    with user_map_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        for expected_index, row in enumerate(reader):
            node = int(row["user_index"])
            if node != expected_index:
                raise ValueError("Ciao user indices must be sorted and contiguous from zero")
            graph.add_node(node)

    with edge_path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            left = int(row["user_index_a"])
            right = int(row["user_index_b"])
            if left >= right or left not in graph or right not in graph:
                raise ValueError(f"Invalid Ciao undirected edge ({left}, {right})")
            graph.add_edge(left, right)

    source_graph_hash = expected_hashes["trust_edges_undirected.csv"]
    user_map_hash = expected_hashes["user_map.csv"]
    return graph, source_graph_hash, user_map_hash


def make_probability_bank(graph: nx.Graph, source_graph_hash: str) -> tuple[str, dict[tuple[int, int], float]]:
    degrees = dict(graph.degree())
    probabilities = {
        edge: 1.0 / max(degrees[edge[0]], degrees[edge[1]])
        for edge in graph.edges()
    }
    probability_id = (
        f"ciao_{PROBABILITY_PROTOCOL}_"
        f"{hashlib.sha256(source_graph_hash.encode('ascii')).hexdigest()[:12]}"
    )
    return probability_id, probabilities


def graph_matches(path: Path, expected: nx.Graph) -> bool:
    with path.open("rb") as source:
        actual = pickle.load(source)
    return (
        isinstance(actual, nx.Graph)
        and not actual.is_directed()
        and list(actual.nodes) == list(expected.nodes)
        and {tuple(sorted(edge)) for edge in actual.edges}
        == {tuple(sorted(edge)) for edge in expected.edges}
    )


def probability_bank_matches(
    path: Path,
    probability_id: str,
    probabilities: dict[tuple[int, int], float],
) -> bool:
    with path.open("rb") as source:
        stored_id = pickle.load(source)
        stored_true = pickle.load(source)
        stored_test = pickle.load(source)
        trailing_data = source.read(1)
    return (
        not trailing_data
        and stored_id == probability_id
        and stored_true == probabilities
        and stored_test == probabilities
    )


def write_pickle(path: Path, values: tuple[Any, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        for value in values:
            pickle.dump(value, output)


def prepare() -> dict[str, Any]:
    graph, source_graph_hash, user_map_hash = read_graph()
    probability_id, probabilities = make_probability_bank(graph, source_graph_hash)
    expected_edges = graph.number_of_edges()
    if len(probabilities) != expected_edges:
        raise ValueError("The WIC probability bank does not cover every Ciao edge")

    if GRAPH_PATH.exists():
        if not graph_matches(GRAPH_PATH, graph):
            raise FileExistsError(f"Existing HIM graph does not match the shared Ciao export: {GRAPH_PATH}")
    else:
        write_pickle(GRAPH_PATH, (graph,))

    if PROBABILITY_PATH.exists():
        if not probability_bank_matches(PROBABILITY_PATH, probability_id, probabilities):
            raise FileExistsError(
                f"Existing Ciao WIC probabilities do not match the selected protocol: {PROBABILITY_PATH}"
            )
    else:
        write_pickle(PROBABILITY_PATH, (probability_id, probabilities, probabilities))

    result: dict[str, Any] = {
        "dataset": "ciao",
        "graph_path": str(GRAPH_PATH.relative_to(HIM_ROOT)),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": expected_edges,
        "graph_directed": graph.is_directed(),
        "source_graph_sha256": source_graph_hash,
        "source_user_map_sha256": user_map_hash,
        "probability_protocol": PROBABILITY_PROTOCOL,
        "probability_definition": "For every undirected edge {u,v}, p(u,v)=p(v,u)=1/max(degree(u), degree(v)).",
        "probability_id": probability_id,
        "probability_entries": len(probabilities),
        "probability_min": min(probabilities.values()),
        "probability_max": max(probabilities.values()),
        "probability_file_sha256": sha256_file(PROBABILITY_PATH),
    }
    MANIFEST_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
