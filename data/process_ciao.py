"""Create deterministic, shared Ciao tables from the locally extracted MAT release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "data" / "raw" / "ciao" / "ciao"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "ciao"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_integer_table(path: Path, variable: str, columns: int) -> np.ndarray:
    table = loadmat(path, variable_names=[variable]).get(variable)
    if table is None or table.ndim != 2 or table.shape[1] != columns:
        raise ValueError(
            f"{path} must contain a {variable!r} matrix with {columns} columns"
        )
    if not np.issubdtype(table.dtype, np.number) or not np.isfinite(table).all():
        raise ValueError(f"{path}:{variable} contains non-finite or non-numeric data")
    if not np.equal(table, np.floor(table)).all():
        raise ValueError(f"{path}:{variable} contains non-integer values")
    return table.astype(np.int64, copy=False)


def write_csv(path: Path, header: list[str], rows: Iterable[Iterable[int]]) -> int:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        count = 0
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def connected_component_sizes(node_count: int, edges: list[tuple[int, int]]) -> list[int]:
    adjacency = [set() for _ in range(node_count)]
    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)

    seen: set[int] = set()
    sizes: list[int] = []
    for start in range(node_count):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def process(source_dir: Path, output_dir: Path) -> dict[str, object]:
    rating_path = source_dir / "rating.mat"
    trust_path = source_dir / "trustnetwork.mat"
    missing = [str(path) for path in (rating_path, trust_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Ciao source file(s): {', '.join(missing)}")

    rating = load_integer_table(rating_path, "rating", 5)
    trust = load_integer_table(trust_path, "trustnetwork", 2)
    if (rating[:, 0] <= 0).any() or (rating[:, 1] <= 0).any():
        raise ValueError("Ciao user and product IDs must be positive")
    if (trust <= 0).any():
        raise ValueError("Ciao trust-network user IDs must be positive")
    if ((rating[:, 3] < 0) | (rating[:, 3] > 5)).any():
        raise ValueError("Ciao ratings must be in the documented 0--5 range")

    source_users = sorted(map(int, np.unique(trust)))
    user_to_index = {user_id: index for index, user_id in enumerate(source_users)}
    in_scope = np.isin(rating[:, 0], source_users)
    scoped_ratings = rating[in_scope]
    product_ids = sorted(map(int, np.unique(scoped_ratings[:, 1])))
    product_to_index = {product_id: index for index, product_id in enumerate(product_ids)}

    directed_edges = sorted({(int(source), int(target)) for source, target in trust})
    undirected_edges = sorted(
        {(min(user_to_index[source], user_to_index[target]),
          max(user_to_index[source], user_to_index[target]))
         for source, target in directed_edges
         if source != target}
    )
    component_sizes = connected_component_sizes(len(source_users), undirected_edges)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        "user_map.csv",
        "item_map.csv",
        "trust_edges_directed.csv",
        "trust_edges_undirected.csv",
        "user_item_ratings.csv",
        "manifest.json",
    ]
    existing = [name for name in outputs if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite existing generated file(s): {', '.join(existing)}"
        )

    row_counts = {
        "user_map.csv": write_csv(
            output_dir / "user_map.csv",
            ["user_index", "source_user_id"],
            ((index, user_id) for index, user_id in enumerate(source_users)),
        ),
        "item_map.csv": write_csv(
            output_dir / "item_map.csv",
            ["item_index", "source_product_id"],
            ((index, product_id) for index, product_id in enumerate(product_ids)),
        ),
        "trust_edges_directed.csv": write_csv(
            output_dir / "trust_edges_directed.csv",
            ["source_user_index", "target_user_index", "source_user_id", "target_user_id"],
            ((user_to_index[source], user_to_index[target], source, target)
             for source, target in directed_edges),
        ),
        "trust_edges_undirected.csv": write_csv(
            output_dir / "trust_edges_undirected.csv",
            ["user_index_a", "user_index_b"],
            undirected_edges,
        ),
        "user_item_ratings.csv": write_csv(
            output_dir / "user_item_ratings.csv",
            [
                "source_row_index",
                "user_index",
                "item_index",
                "source_user_id",
                "source_product_id",
                "category_id",
                "rating",
                "helpfulness",
            ],
            (
                (
                    source_row_index,
                    user_to_index[int(row[0])],
                    product_to_index[int(row[1])],
                    int(row[0]),
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                    int(row[4]),
                )
                for source_row_index, row in enumerate(rating)
                if in_scope[source_row_index]
            ),
        ),
    }

    deduplicated_pairs = np.unique(scoped_ratings[:, :2], axis=0).shape[0]
    manifest: dict[str, object] = {
        "dataset": "Ciao",
        "processing_script": "data/process_ciao.py",
        "source_files": {
            "rating.mat": {
                "sha256": sha256_file(rating_path),
                "rows": int(rating.shape[0]),
                "columns": ["user_id", "product_id", "category_id", "rating", "helpfulness"],
            },
            "trustnetwork.mat": {
                "sha256": sha256_file(trust_path),
                "directed_arcs": int(trust.shape[0]),
                "columns": ["trustor_user_id", "trusted_user_id"],
            },
        },
        "policies": {
            "users": "All distinct users appearing in trustnetwork.mat; sorted source IDs mapped to contiguous zero-based indices.",
            "directed_edges": "Preserved as trustor -> trusted, with repeated directed arcs collapsed.",
            "undirected_edges": "Self-loops excluded and reciprocal/direct duplicates collapsed for user-level graph methods.",
            "ratings": "Keep source order, duplicate user-product rows, zero ratings, and all source fields for users present in the trust graph.",
            "ratings_outside_trust_graph": "Excluded from the shared user-item export; retained only in the unchanged source MAT file.",
        },
        "counts": {
            "source_rating_rows": int(rating.shape[0]),
            "shared_rating_rows": int(scoped_ratings.shape[0]),
            "rating_rows_excluded_outside_trust_graph": int((~in_scope).sum()),
            "source_duplicate_user_product_rows": int(rating.shape[0] - np.unique(rating[:, :2], axis=0).shape[0]),
            "shared_unique_user_product_pairs": int(deduplicated_pairs),
            "trust_users": len(source_users),
            "directed_trust_arcs": len(directed_edges),
            "undirected_trust_edges": len(undirected_edges),
            "connected_components": len(component_sizes),
            "largest_component_nodes": component_sizes[0] if component_sizes else 0,
            "isolated_users": sum(size == 1 for size in component_sizes),
            "products_in_shared_ratings": len(product_ids),
        },
        "generated_csv_data_rows": row_counts,
        "generated_csv_sha256": {
            name: sha256_file(output_dir / name)
            for name in outputs
            if name != "manifest.json"
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = process(args.source_dir, args.output_dir)
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Wrote processed Ciao tables to {args.output_dir}")


if __name__ == "__main__":
    main()
