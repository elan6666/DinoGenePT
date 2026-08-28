"""Audit matched GGI result receipts and build a compact comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .provenance import atomic_write_json, digest_file, utc_now

FAIRNESS_FIELDS = (
    "task",
    "model",
    "normalization",
    "pair_operator",
    "random_state",
    "train_n",
    "test_n",
    "train_coverage",
    "test_coverage",
    "data_receipt_sha256",
    "gene_universe",
    "gene_universe_sha256",
    "genept_seed_version",
    "numpy_version",
    "scikit_learn_version",
)
METRICS = ("accuracy", "auroc", "average_precision")


def _load_single_result(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict):
        raise ValueError(f"GGI result must contain exactly one result row: {path}")
    row = dict(raw[0])
    required = {"embedding", "vectors_sha256", *FAIRNESS_FIELDS, *METRICS}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"GGI result is missing fields {missing}: {path}")
    return row


def build_ggi_comparison(
    *, result_paths: list[Path], baseline: str, output_path: Path
) -> dict[str, Any]:
    if len(result_paths) < 2:
        raise ValueError("GGI comparison requires at least two result receipts")
    rows = [_load_single_result(path) for path in result_paths]
    names = [str(row["embedding"]) for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("GGI comparison contains duplicate embedding names")
    if baseline not in names:
        raise ValueError(f"baseline embedding is absent: {baseline}")
    reference = rows[0]
    mismatches: dict[str, dict[str, Any]] = {}
    for row in rows[1:]:
        differences = {
            field: {"expected": reference[field], "actual": row[field]}
            for field in FAIRNESS_FIELDS
            if row[field] != reference[field]
        }
        if differences:
            mismatches[str(row["embedding"])] = differences
    if mismatches:
        raise ValueError(f"GGI fairness fields differ: {mismatches}")
    baseline_row = rows[names.index(baseline)]
    conditions = []
    for path, row in zip(result_paths, rows, strict=True):
        conditions.append(
            {
                "embedding": row["embedding"],
                "vectors_sha256": row["vectors_sha256"],
                "result_sha256": digest_file(path),
                **{metric: row[metric] for metric in METRICS},
                "delta_vs_baseline": {
                    metric: float(row[metric]) - float(baseline_row[metric])
                    for metric in METRICS
                },
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": "genept-seed-progressive-ggi-comparison-v1",
        "created_at": utc_now(),
        "baseline": baseline,
        "fairness_fields_identical": True,
        "fairness_contract": {field: reference[field] for field in FAIRNESS_FIELDS},
        "conditions": conditions,
    }
    atomic_write_json(output_path, receipt)
    return receipt
