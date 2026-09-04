"""Compact, fairness-gated summaries for repeated benchmark results."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .provenance import atomic_write_json, digest_file, utc_now

METRICS = ("accuracy", "auroc", "average_precision")


def _rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"result must be a non-empty JSON row list: {path}")
    return value


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize_property_results(*, result_paths: list[Path], output_path: Path) -> dict[str, Any]:
    all_rows = [row for path in result_paths for row in _rows(path)]
    fairness = (
        "normalization",
        "gene_universe_sha256",
        "data_receipt_sha256",
        "genept_seed_version",
        "numpy_version",
        "scikit_learn_version",
    )
    for field in fairness:
        values = {json.dumps(row.get(field), sort_keys=True) for row in all_rows}
        if len(values) != 1:
            raise ValueError(f"property fairness field differs: {field}")
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[(str(row["embedding"]), str(row["task"]), str(row["model"]))].append(row)
    expected_keys = {(str(row["task"]), str(row["model"])) for row in all_rows}
    by_embedding: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for embedding, task, model in grouped:
        by_embedding[embedding].add((task, model))
    if any(keys != expected_keys for keys in by_embedding.values()):
        raise ValueError("property embeddings do not contain the same task/model grid")
    conditions = []
    for (embedding, task, model), rows in sorted(grouped.items()):
        seeds = sorted({int(row["random_state"]) for row in rows})
        folds = sorted({int(row["fold"]) for row in rows})
        if len(rows) != len(seeds) * len(folds):
            raise ValueError(f"incomplete repeated folds for {embedding}/{task}/{model}")
        conditions.append(
            {
                "embedding": embedding,
                "task": task,
                "model": model,
                "seeds": seeds,
                "folds": folds,
                "n_genes": sorted({int(row["n"]) for row in rows}),
                **{metric: _summary([float(row[metric]) for row in rows]) for metric in METRICS},
            }
        )
    receipt = {
        "schema_version": "genept-seed-property-comparison-v1",
        "created_at": utc_now(),
        "fairness_fields_identical": True,
        "fairness_contract": {field: all_rows[0].get(field) for field in fairness},
        "result_sha256": {path.name: digest_file(path) for path in result_paths},
        "conditions": conditions,
    }
    atomic_write_json(output_path, receipt)
    return receipt


def summarize_gene_disjoint_results(*, result_paths: list[Path], output_path: Path) -> dict[str, Any]:
    result_sets = [(path, _rows(path)) for path in result_paths]
    reference = result_sets[0][1]
    reference_splits = {
        int(row["random_state"]): json.dumps(row["split_receipt"], sort_keys=True) for row in reference
    }
    reference_universe_filter = json.dumps(reference[0].get("universe_filter"), sort_keys=True)
    fairness = (
        "normalization",
        "gene_universe_sha256",
        "data_receipt_sha256",
        "model",
        "pair_operator",
        "genept_seed_version",
        "numpy_version",
        "scikit_learn_version",
    )
    conditions = []
    for path, rows in result_sets:
        current_splits = {
            int(row["random_state"]): json.dumps(row["split_receipt"], sort_keys=True)
            for row in rows
        }
        if current_splits != reference_splits:
            raise ValueError(f"gene-disjoint split receipts differ: {path}")
        if any(
            json.dumps(row.get("universe_filter"), sort_keys=True) != reference_universe_filter
            for row in rows
        ):
            raise ValueError(f"gene-disjoint universe filters differ: {path}")
        for field in fairness:
            if len({json.dumps(row.get(field), sort_keys=True) for row in rows + reference}) != 1:
                raise ValueError(f"gene-disjoint fairness field differs: {field}")
        conditions.append(
            {
                "embedding": rows[0]["embedding"],
                "vectors_sha256": rows[0]["vectors_sha256"],
                "seeds": sorted(reference_splits),
                **{metric: _summary([float(row[metric]) for row in rows]) for metric in METRICS},
            }
        )
    receipt = {
        "schema_version": "genept-seed-gene-disjoint-comparison-v1",
        "created_at": utc_now(),
        "fairness_fields_identical": True,
        "split_receipts_identical": True,
        "universe_filters_identical": True,
        "universe_filter": reference[0].get("universe_filter"),
        "result_sha256": {path.name: digest_file(path) for path in result_paths},
        "conditions": conditions,
    }
    atomic_write_json(output_path, receipt)
    return receipt
