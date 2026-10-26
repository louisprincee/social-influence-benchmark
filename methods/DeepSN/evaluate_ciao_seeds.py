"""Independent WIC evaluation of a saved DeepSN seed set on shared Ciao."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from prepare_ciao_sg import edge_probability, load_shared_graph, sha256_file

METHOD_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = METHOD_ROOT.parents[1]


def simulate_activated(graph, seeds: list[int]) -> int:
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
    return len(active)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True, help="DeepSN run JSON containing a seeds list")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--evaluation-seed", type=int, default=1042)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.result.is_absolute():
        result_path = args.result
    elif args.result.exists():
        result_path = args.result.resolve()
    elif (REPOSITORY_ROOT / args.result).exists():
        result_path = (REPOSITORY_ROOT / args.result).resolve()
    else:
        result_path = args.result.resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    seeds = [int(node) for node in result["seeds"]]
    graph, edge_hash, user_hash = load_shared_graph()
    if any(node < 0 or node >= graph.number_of_nodes() for node in seeds):
        raise ValueError("seed ids are outside the shared Ciao graph")

    activated = []
    for replicate in range(args.rounds):
        random.seed(args.evaluation_seed + replicate)
        np.random.seed(args.evaluation_seed + replicate)
        activated.append(simulate_activated(graph, seeds))

    payload = {
        "method": "DeepSN",
        "dataset": "ciao",
        "diffusion_model": "WIC",
        "probability": "p(u,v)=1/max(degree(u), degree(v))",
        "probability_id": "ciao_symmetric_weighted_cascade_v1_f2fc09305d5f",
        "source_result": str(result_path.relative_to(REPOSITORY_ROOT)),
        "source_result_sha256": sha256_file(result_path),
        "undirected_edge_csv_sha256": edge_hash,
        "user_map_sha256": user_hash,
        "seed_count": len(seeds),
        "seeds": seeds,
        "evaluation_seed_base": args.evaluation_seed,
        "rounds": args.rounds,
        "mean_activated": float(np.mean(activated)),
        "stdev_activated": float(np.std(activated, ddof=1)) if len(activated) > 1 else 0.0,
        "mean_coverage_percent": float(np.mean(activated)) * 100.0 / graph.number_of_nodes(),
        "activated_counts": activated,
        "interpretation": (
            "Independent synthetic WIC evaluation on the shared Ciao trust graph. "
            "One DeepSN training seed; not a cross-method ranking until the team freezes the evaluator."
        ),
    }
    output = args.output
    if output is None:
        output = REPOSITORY_ROOT / "results" / "DeepSN" / f"ciao_wic_independent_eval_{result_path.stem}.json"
    elif not output.is_absolute():
        output = REPOSITORY_ROOT / output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("mean_activated", "mean_coverage_percent", "rounds", "seed_count")}, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
