"""Evaluate ASW/MD seed sets from a saved HIM checkpoint on one fixed simulator."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch
from scipy.stats import t

from dataload.dataset import DataLoad
from modules.diffusion import DiffusionTools
from methods.HIM.HIM import HIM
from utils.runtime import Runner
from utils.tools import MyTool as MT


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_name", default="ciao")
    parser.add_argument("--diffusion_model", default="WIC")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed_metadata", type=Path, required=True)
    parser.add_argument("--evaluation_seed", type=int, default=1042)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--training_instances", type=int)
    parser.add_argument("--methods", nargs="+", choices=["ASW", "MD"], default=["ASW", "MD"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds <= 0 or not args.methods:
        parser.error("rounds and methods must be non-empty and positive")

    checkpoint = args.checkpoint.resolve()
    seed_metadata_path = args.seed_metadata.resolve()
    output_path = args.output if args.output.is_absolute() else REPOSITORY_ROOT / args.output
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    if not seed_metadata_path.is_file():
        parser.error(f"seed metadata does not exist: {seed_metadata_path}")
    if output_path.exists():
        parser.error(f"refusing to overwrite existing result: {output_path}")

    seed_metadata = json.loads(seed_metadata_path.read_text(encoding="utf-8"))
    if seed_metadata["data_name"] != args.data_name:
        parser.error("checkpoint seed metadata dataset does not match --data_name")
    if seed_metadata["diffusion_model"] != args.diffusion_model:
        parser.error("checkpoint seed metadata diffusion does not match --diffusion_model")
    seed_count = int(seed_metadata["seed_count"])

    data = DataLoad(args.data_name, args.diffusion_model)
    graph = data.social_G
    if list(graph.nodes) != list(range(graph.number_of_nodes())):
        parser.error("evaluation graph nodes must be contiguous integers starting at zero")
    if seed_count > graph.number_of_nodes():
        parser.error("seed budget exceeds the evaluation graph size")

    train_config, model_config = MT.load_model_params(
        f"HIM_{args.diffusion_model}", args.data_name
    )
    device = torch.device("cpu")
    model = HIM(
        graph.number_of_nodes(),
        list(dict(graph.degree()).values()),
        model_config.c,
        model_config.num_dim,
        model_config.init_size,
        model_config.num_negs,
        device,
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    runner = Runner(model, None, 0, device)
    selected_seeds: dict[str, list[int]] = {}
    selection_seconds: dict[str, float] = {}
    for method in args.methods:
        started = time.perf_counter()
        seeds = sorted(
            map(
                int,
                runner.seed_set_selection(
                    seed_count,
                    method=method,
                    window_weight=train_config.window_weight,
                    G=graph,
                ),
            )
        )
        selection_seconds[method] = time.perf_counter() - started
        if len(seeds) != seed_count or len(set(seeds)) != seed_count:
            raise ValueError(f"{method} returned an invalid seed set of size {len(seeds)}")
        if seeds[0] < 0 or seeds[-1] >= graph.number_of_nodes():
            raise ValueError(f"{method} returned a node outside the graph")
        selected_seeds[method] = seeds

    evaluator = DiffusionTools(
        args.data_name,
        graph,
        diffusion=args.diffusion_model,
        num_bunch=100,
        random_type="random",
    )
    outcomes = {method: [] for method in args.methods}
    evaluation_seconds: dict[str, float] = {}
    for method in args.methods:
        started = time.perf_counter()
        for replicate in range(args.rounds):
            replicate_seed = args.evaluation_seed + replicate
            random.seed(replicate_seed)
            np.random.seed(replicate_seed)
            mean, _ = evaluator.spread_evaluation(
                selected_seeds[method], num_rounds=1
            )
            outcomes[method].append(int(mean))
        evaluation_seconds[method] = time.perf_counter() - started

    summaries: dict[str, dict[str, float]] = {}
    for method, values in outcomes.items():
        results = np.asarray(values, dtype=np.float64)
        summaries[method] = {
            "mean_activated": float(results.mean()),
            "std_activated_population": float(results.std(ddof=0)),
            "mean_coverage_percent": float(100 * results.mean() / graph.number_of_nodes()),
            "std_coverage_percentage_points": float(
                100 * results.std(ddof=0) / graph.number_of_nodes()
            ),
        }

    paired: dict[str, Any] | None = None
    if "ASW" in outcomes and "MD" in outcomes:
        difference = np.asarray(outcomes["ASW"]) - np.asarray(outcomes["MD"])
        mean_difference = float(difference.mean())
        sem = float(difference.std(ddof=1) / np.sqrt(args.rounds)) if args.rounds > 1 else 0.0
        critical_value = float(t.ppf(0.975, args.rounds - 1)) if args.rounds > 1 else 0.0
        paired = {
            "mean_activated_difference": mean_difference,
            "ci95_t_activated": [
                mean_difference - critical_value * sem,
                mean_difference + critical_value * sem,
            ],
            "mean_coverage_difference_percentage_points": (
                100 * mean_difference / graph.number_of_nodes()
            ),
            "ASW_higher": int((difference > 0).sum()),
            "equal": int((difference == 0).sum()),
            "MD_higher": int((difference < 0).sum()),
        }

    probability_path = (
        REPOSITORY_ROOT
        / "methods"
        / "HIM"
        / "configs"
        / "diffusion"
        / f"{args.data_name}_{args.diffusion_model}_random_probs.pkl"
    )
    graph_path = REPOSITORY_ROOT / "methods" / "HIM" / "data" / args.data_name / f"{args.data_name}.G"
    training_cache = seed_metadata.get("training_cache")
    if training_cache is not None:
        training_cache = (Path("methods/HIM") / training_cache).as_posix()
    result: dict[str, Any] = {
        "dataset": args.data_name,
        "diffusion_model": args.diffusion_model,
        "probability_id": evaluator.prob_id,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "training_seed": seed_metadata.get("training_seed"),
        "training_epoch": seed_metadata.get("epoch"),
        "training_instances": (
            args.training_instances
            if args.training_instances is not None
            else seed_metadata.get("training_instances")
        ),
        "training_sample_ratio": seed_metadata.get("training_sample_ratio"),
        "training_configuration": {
            "max_epochs": seed_metadata.get("max_epochs"),
            "validation_rounds_per_epoch": seed_metadata.get("validation_rounds"),
            "early_stopping_patience": seed_metadata.get("early_stopping_patience"),
            "selection_method": seed_metadata.get("selection_method"),
        },
        "training_cache": training_cache,
        "training_cache_sha256": seed_metadata.get("training_cache_sha256"),
        "seed_budget": seed_count,
        "checkpoint": str(checkpoint.relative_to(REPOSITORY_ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "seed_metadata_sha256": sha256_file(seed_metadata_path),
        "graph_sha256": sha256_file(graph_path),
        "probability_file_sha256": sha256_file(probability_path),
        "evaluation_seed_base": args.evaluation_seed,
        "evaluation_rounds": args.rounds,
        "evaluation_protocol": (
            "For each replicate, reset Python random and NumPy to the same replicate-specific "
            "seed before evaluating each method against the same fixed edge-probability bank."
        ),
        "methods": {
            method: {
                "seeds": selected_seeds[method],
                "selection_seconds": selection_seconds[method],
                "evaluation_seconds": evaluation_seconds[method],
                "activated_per_replicate": outcomes[method],
            }
            for method in args.methods
        },
        "summaries": summaries,
        "paired_comparison": paired,
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "interpretation": (
            "Monte Carlo evaluation conditional on one fixed checkpoint, graph, and probability "
            "bank; it does not measure variation across model-training runs or real-world adoption."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "methods"}, indent=2))
    print(f"Saved evaluation to {output_path}")


if __name__ == "__main__":
    main()
