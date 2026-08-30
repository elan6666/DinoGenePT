"""Sequential ablation runner that reuses datasets and never launches parallel GPU jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dinogenept.config import ResolvedConfig, load_config
from dinogenept.priors import PriorStore
from dinogenept.registry import DATASETS
from genept_seed.provenance import atomic_write_json


def _paired_comparison(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    reference_rows = reference["metrics"]["per_condition"]
    candidate_rows = candidate["metrics"]["per_condition"]
    if set(reference_rows) != set(candidate_rows):
        raise ValueError("paired matrix comparison requires identical condition sets")
    conditions = tuple(sorted(reference_rows))
    metric_names = sorted(
        {name for condition in conditions for name in reference_rows[condition]}
    )
    result: dict[str, Any] = {}
    for offset, metric in enumerate(metric_names):
        differences = np.asarray(
            [
                candidate_rows[condition].get(metric, np.nan)
                - reference_rows[condition].get(metric, np.nan)
                if candidate_rows[condition].get(metric) is not None
                and reference_rows[condition].get(metric) is not None
                else np.nan
                for condition in conditions
            ],
            dtype=np.float64,
        )
        finite = differences[np.isfinite(differences)]
        row: dict[str, Any] = {
            "candidate_minus_reference": float(finite.mean()) if len(finite) else None,
            "finite_conditions": int(len(finite)),
            "total_conditions": len(conditions),
        }
        if resamples > 0 and len(finite) >= 2:
            rng = np.random.default_rng(seed + offset)
            means = np.asarray(
                [rng.choice(finite, len(finite), replace=True).mean() for _ in range(resamples)]
            )
            row["paired_condition_bootstrap_95ci"] = [
                float(np.quantile(means, 0.025)),
                float(np.quantile(means, 0.975)),
            ]
        result[metric] = row
    return result


def run_matrix(
    matrix_path: str | Path,
    *,
    device: str | None,
    output_root: Path | None,
) -> dict[str, Any]:
    source = Path(matrix_path).resolve(strict=True)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "dinogenept-matrix-v1":
        raise ValueError("invalid DinoGenePT matrix schema")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("matrix rows must be a non-empty list")
    resolved: list[ResolvedConfig] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("config"), str):
            raise ValueError("every matrix row requires a config path")
        overrides = list(row.get("overrides", []))
        if device:
            overrides.append(f"runtime.device={device}")
        if output_root:
            overrides.append(f"runtime.output_root={str(output_root)}")
        config = load_config(source.parent / row["config"], overrides=overrides)
        if bool(payload.get("smoke", False)) and (
            not config.payload["runtime"].get("smoke")
            or config.payload["training"]["epochs"] != 1
        ):
            raise ValueError("smoke matrix rows must explicitly use smoke mode and one epoch")
        resolved.append(config)
    identities = {
        (config.identity["model_id"], config.identity["dataset_id"])
        for config in resolved
    }
    if len(identities) != 1:
        raise ValueError("one matrix must keep a common model and dataset identity")
    experiment_ids = [config.identity["experiment_id"] for config in resolved]
    if len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError("matrix experiment_id values must be unique")
    dataset_contracts = {
        json.dumps(config.payload["dataset"], sort_keys=True, separators=(",", ":"))
        for config in resolved
    }
    evaluation_contracts = {
        json.dumps(config.payload["evaluation"], sort_keys=True, separators=(",", ":"))
        for config in resolved
    }
    seeds = {int(config.payload["training"]["seed"]) for config in resolved}
    if len(dataset_contracts) != 1 or len(evaluation_contracts) != 1 or len(seeds) != 1:
        raise ValueError("matrix rows must share one dataset, evaluation contract, and seed")

    cache: dict[str, Any] = {}
    prior_cache: dict[str, PriorStore] = {}
    receipts = []
    fingerprints = set()
    from .runner import (
        _dataset_cache_key,
        _prior_cache_key,
        implementation_sha256,
        run_experiment,
        run_root,
        validate_completed_run,
    )

    implementation_hash = implementation_sha256()
    first = resolved[0]
    dataset_key = _dataset_cache_key(first.payload)
    cache[dataset_key] = DATASETS.resolve(first.identity["dataset_id"])(first.payload)
    data = cache[dataset_key]
    targets = tuple(sorted({target for item in data.targets for target in item}))
    full_receipts: dict[str, dict[str, Any]] = {}

    for config in resolved:
        prior_key = _prior_cache_key(config.payload, targets)
        if prior_key not in prior_cache:
            prior_cache[prior_key] = PriorStore.from_config(config.payload, targets)
        priors = prior_cache[prior_key]
        completed_path = run_root(config) / "run.json"
        reused = False
        if completed_path.exists():
            receipt = json.loads(completed_path.read_text(encoding="utf-8"))
            validate_completed_run(
                config=config,
                receipt=receipt,
                data=data,
                priors=priors,
                implementation_hash=implementation_hash,
            )
            reused = True
        else:
            receipt = run_experiment(
                config,
                dataset_cache=cache,
                prior_cache=prior_cache,
                retry_failed=True,
            )
        full_receipts[config.identity["experiment_id"]] = receipt
        fingerprints.add(receipt["dataset_fingerprint"])
        metric_summary = {
            name: values.get("macro_mean")
            for name, values in receipt["metrics"]["summary"].items()
        }
        receipts.append(
            {
                "experiment_id": config.identity["experiment_id"],
                "status": receipt["status"],
                "reused": reused,
                "config_sha256": config.sha256,
                "dataset_fingerprint": receipt["dataset_fingerprint"],
                "run_root": receipt["run_root"],
                "elapsed_seconds": receipt["elapsed_seconds"],
                "peak_allocated_mb": receipt["gpu"].get("peak_allocated_mb"),
                "metrics": metric_summary,
            }
        )
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ModuleNotFoundError:
            pass
    if len(fingerprints) != 1:
        raise RuntimeError("matrix rows resolved to different dataset fingerprints")
    reference_id = str(payload.get("reference", experiment_ids[0]))
    if reference_id not in full_receipts:
        raise ValueError(f"matrix reference experiment is absent: {reference_id}")
    resamples = int(payload.get("paired_bootstrap_resamples", 0))
    paired = {
        experiment_id: _paired_comparison(
            full_receipts[reference_id],
            receipt,
            resamples=resamples,
            seed=next(iter(seeds)),
        )
        for experiment_id, receipt in full_receipts.items()
    }
    result = {
        "schema_version": "dinogenept-matrix-result-v1",
        "status": "complete",
        "row_count": len(receipts),
        "reused_count": sum(bool(row["reused"]) for row in receipts),
        "common_identity": list(next(iter(identities))),
        "dataset_fingerprint": next(iter(fingerprints)),
        "implementation_sha256": implementation_hash,
        "seed": next(iter(seeds)),
        "reference_experiment_id": reference_id,
        "paired_comparisons": paired,
        "rows": receipts,
    }
    common_output_root = Path(resolved[0].payload["runtime"]["output_root"])
    receipt_path = Path(str(payload.get("receipt", f"{source.stem}.result.json")))
    if not receipt_path.is_absolute():
        receipt_path = (common_output_root / receipt_path).resolve()
    atomic_write_json(receipt_path, result)
    return result
