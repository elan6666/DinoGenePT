"""Resumable multi-dataset, multi-seed benchmark orchestration."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from dinogenept.config import load_config
from dinogenept.priors import PriorStore
from genept_seed.provenance import atomic_write_json

from .pipeline import _metric_summary, _run_or_reuse


def run_benchmark(
    benchmark_path: str | Path,
    *,
    only: tuple[str, ...] = (),
    only_seeds: tuple[int, ...] = (),
    output_root: Path | None = None,
) -> dict[str, Any]:
    source = Path(benchmark_path).resolve(strict=True)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "dinogenept-benchmark-v1":
        raise ValueError("invalid DinoGenePT benchmark schema")
    full_seeds = payload.get("seeds")
    if full_seeds != [1, 2, 3, 4]:
        raise ValueError("formal GraD-Pert comparison requires run seeds [1, 2, 3, 4]")
    selected_seed_set = set(only_seeds)
    unknown_seeds = selected_seed_set - set(full_seeds)
    if unknown_seeds:
        raise ValueError(f"benchmark selection names unknown seeds: {sorted(unknown_seeds)}")
    seeds = tuple(
        seed for seed in full_seeds if not selected_seed_set or seed in selected_seed_set
    )
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("benchmark rows must be a non-empty list")
    resolved_rows: list[tuple[dict[str, Any], Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("config"), str):
            raise ValueError("benchmark row requires a config path")
        config = load_config(source.parent / row["config"], overrides=row.get("overrides", []))
        resolved_rows.append((row, config))
    dataset_ids = [config.identity["dataset_id"] for _, config in resolved_rows]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("benchmark dataset identities must be unique")
    selected = set(only)
    unknown = selected - set(dataset_ids)
    if unknown:
        raise ValueError(f"benchmark selection names unknown datasets: {sorted(unknown)}")
    selected_ids = tuple(
        dataset_id for dataset_id in dataset_ids if not selected or dataset_id in selected
    )
    receipt_path = Path(str(payload.get("receipt", f"{source.stem}.result.json")))
    if not receipt_path.is_absolute():
        receipt_path = Path(output_root or "results/dinogenept").resolve() / receipt_path
    if selected or selected_seed_set:
        selection_hash = hashlib.sha256(
            json.dumps(
                {"datasets": selected_ids, "seeds": seeds},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:8]
        receipt_path = receipt_path.with_name(
            f"{receipt_path.stem}.partial-{selection_hash}{receipt_path.suffix}"
        )
    results: list[dict[str, Any]] = []
    for row, preflight_config in resolved_rows:
        if preflight_config.identity["dataset_id"] not in selected_ids:
            continue
        dataset_cache: dict[str, Any] = {}
        prior_cache: dict[str, PriorStore] = {}
        for seed in seeds:
            overrides = [*row.get("overrides", []), f"training.seed={seed}"]
            if output_root is not None:
                overrides.append(f"runtime.output_root={output_root}")
            config = load_config(source.parent / row["config"], overrides=overrides)
            if (
                config.identity["model_id"] != "scouter"
                or config.payload["priors"].get("optional")
                or config.payload["evaluation"].get("name") != "gradpert_exact"
                or int(config.payload["evaluation"].get("seed", -1)) != 20260824
            ):
                raise ValueError("Scouter benchmark must be Base-only with frozen GraD-Pert evaluation")
            receipt, reused = _run_or_reuse(
                config, dataset_cache=dataset_cache, prior_cache=prior_cache
            )
            results.append(
                {
                    "dataset_id": config.identity["dataset_id"],
                    "model_id": "scouter",
                    "prior": "genept-seed-base-ncbi-uniprot-only",
                    "seed": seed,
                    "status": receipt["status"],
                    "reused": reused,
                    "run_root": receipt["run_root"],
                    "dataset_fingerprint": receipt["dataset_fingerprint"],
                    "split_sha256": receipt["input_manifest"]["dataset"]["split_sha256"],
                    "fairness_contract_sha256": receipt["fairness_contract"][
                        "sha256"
                    ],
                    "metrics": _metric_summary(receipt),
                    "parameters": receipt["parameters"],
                }
            )
            atomic_write_json(
                receipt_path,
                {
                    "schema_version": "dinogenept-benchmark-result-v1",
                    "status": "running",
                    "seeds": list(seeds),
                    "grid": {
                        "expected_datasets": dataset_ids,
                        "selected_datasets": list(selected_ids),
                        "expected_seeds": full_seeds,
                        "selected_seeds": list(seeds),
                        "expected_runs": len(dataset_ids) * len(full_seeds),
                        "selected_runs": len(selected_ids) * len(seeds),
                        "completed_runs": len(results),
                    },
                    "rows": results,
                },
            )
        dataset_cache.clear()
        prior_cache.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ModuleNotFoundError:
            pass
    expected_runs = len(selected_ids) * len(seeds)
    if len(results) != expected_runs:
        raise RuntimeError("benchmark completed row count differs from selected grid")
    result = {
        "schema_version": "dinogenept-benchmark-result-v1",
        "status": (
            "complete"
            if set(selected_ids) == set(dataset_ids) and set(seeds) == set(full_seeds)
            else "partial_complete"
        ),
        "seeds": list(seeds),
        "grid": {
            "expected_datasets": dataset_ids,
            "selected_datasets": list(selected_ids),
            "expected_seeds": full_seeds,
            "selected_seeds": list(seeds),
            "expected_runs": len(dataset_ids) * len(full_seeds),
            "selected_runs": expected_runs,
            "completed_runs": len(results),
        },
        "row_count": len(results),
        "rows": results,
    }
    atomic_write_json(receipt_path, result)
    return result
