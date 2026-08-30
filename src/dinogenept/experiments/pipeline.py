"""Resumable pretrain-to-finetune orchestration for real perturbation datasets."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from dinogenept.config import ResolvedConfig, load_config
from dinogenept.priors import PriorStore
from dinogenept.registry import DATASETS
from genept_seed.provenance import atomic_write_json

SCGPT_SIGNATURE = {
    "d_model": 512,
    "layers": 12,
    "heads": 8,
    "ff_dim": 512,
    "max_seq_len": 1200,
}


def _assert_scgpt_alignment(pretrain: ResolvedConfig, finetune: ResolvedConfig) -> None:
    for config in (pretrain, finetune):
        observed = {key: int(config.payload["model"][key]) for key in SCGPT_SIGNATURE}
        if observed != SCGPT_SIGNATURE:
            raise ValueError(f"scGPT architecture signature mismatch: {observed}")
        training = config.payload["training"]
        if (
            str(training.get("optimizer")) != "adam"
            or float(training["learning_rate"]) != 0.0001
            or float(training["lr_gamma"]) != 0.9
        ):
            raise ValueError("scGPT optimizer schedule mismatch")
    if pretrain.payload["training"].get("phase") != "pretrain" or int(
        pretrain.payload["training"]["epochs"]
    ) != 6:
        raise ValueError("formal pretraining must use six epochs")
    if finetune.payload["training"].get("phase") != "finetune" or int(
        finetune.payload["training"]["epochs"]
    ) != 15:
        raise ValueError("formal fine-tuning must use fifteen epochs")
    if (
        not finetune.payload["model"].get("pretrained_strict", False)
        or float(finetune.payload["model"].get("pretrained_minimum_match_fraction", 0.0))
        != 1.0
    ):
        raise ValueError("formal fine-tuning requires an exact checkpoint schema")
    if pretrain.payload["dataset"] != finetune.payload["dataset"]:
        raise ValueError("pretraining and fine-tuning must share one frozen dataset contract")


def _run_or_reuse(
    config: ResolvedConfig,
    *,
    dataset_cache: dict[str, Any],
    prior_cache: dict[str, PriorStore],
) -> tuple[dict[str, Any], bool]:
    from .runner import (
        _dataset_cache_key,
        _prior_cache_key,
        implementation_sha256,
        run_experiment,
        run_root,
        validate_completed_run,
    )

    completed_path = run_root(config) / "run.json"
    if not completed_path.exists():
        return (
            run_experiment(
                config,
                dataset_cache=dataset_cache,
                prior_cache=prior_cache,
                retry_failed=True,
            ),
            False,
        )
    dataset_key = _dataset_cache_key(config.payload)
    if dataset_key not in dataset_cache:
        dataset_cache[dataset_key] = DATASETS.resolve(config.identity["dataset_id"])(
            config.payload
        )
    data = dataset_cache[dataset_key]
    targets = tuple(sorted({target for item in data.targets for target in item}))
    prior_key = _prior_cache_key(config.payload, targets)
    if prior_key not in prior_cache:
        prior_cache[prior_key] = PriorStore.from_config(config.payload, targets)
    receipt = json.loads(completed_path.read_text(encoding="utf-8"))
    validate_completed_run(
        config=config,
        receipt=receipt,
        data=data,
        priors=prior_cache[prior_key],
        implementation_hash=implementation_sha256(),
    )
    return receipt, True


def _metric_summary(receipt: dict[str, Any]) -> dict[str, float | None]:
    summary = receipt.get("metrics", {}).get("summary", {})
    return {name: values.get("macro_mean") for name, values in summary.items()}


def run_pipeline(
    pipeline_path: str | Path,
    *,
    only: tuple[str, ...] = (),
    only_seeds: tuple[int, ...] = (),
    output_root: Path | None = None,
) -> dict[str, Any]:
    source = Path(pipeline_path).resolve(strict=True)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "dinogenept-pipeline-v1":
        raise ValueError("invalid DinoGenePT pipeline schema")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("pipeline rows must be a non-empty list")
    for row in rows:
        if not isinstance(row, dict) or not all(
            isinstance(row.get(key), str)
            for key in ("dataset_id", "pretrain", "finetune")
        ):
            raise ValueError(
                "each pipeline row requires dataset_id, pretrain, and finetune"
            )
    common_pretrain_overrides = payload.get("pretrain_overrides", [])
    common_finetune_overrides = payload.get("finetune_overrides", [])
    if not isinstance(common_pretrain_overrides, list) or not all(
        isinstance(item, str) for item in common_pretrain_overrides
    ):
        raise ValueError("pipeline pretrain_overrides must be a list of strings")
    if not isinstance(common_finetune_overrides, list) or not all(
        isinstance(item, str) for item in common_finetune_overrides
    ):
        raise ValueError("pipeline finetune_overrides must be a list of strings")
    full_seeds = payload.get("seeds")
    if full_seeds != [1, 2, 3, 4]:
        raise ValueError("formal GraD-Pert comparison requires run seeds [1, 2, 3, 4]")
    selected_seed_set = set(only_seeds)
    unknown_seeds = selected_seed_set - set(full_seeds)
    if unknown_seeds:
        raise ValueError(f"pipeline selection names unknown seeds: {sorted(unknown_seeds)}")
    seeds = tuple(seed for seed in full_seeds if not selected_seed_set or seed in selected_seed_set)
    selected = set(only)
    dataset_ids = [str(row.get("dataset_id")) for row in rows]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("pipeline dataset_id values must be unique")
    unknown = selected - set(dataset_ids)
    if unknown:
        raise ValueError(f"pipeline selection names unknown datasets: {sorted(unknown)}")
    selected_ids = tuple(
        dataset_id for dataset_id in dataset_ids if not selected or dataset_id in selected
    )
    results: list[dict[str, Any]] = []
    receipt_path = Path(str(payload.get("receipt", "pipeline-result.json")))
    if not receipt_path.is_absolute():
        receipt_path = (Path(output_root or "results/dinogenept") / receipt_path).resolve()
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

    # Resolve and validate the entire selected config grid before the first GPU
    # step. This catches an invalid late row without wasting earlier training.
    for row in rows:
        dataset_id = str(row.get("dataset_id"))
        if dataset_id not in selected_ids:
            continue
        for seed in seeds:
            common_overrides = [*row.get("overrides", []), f"training.seed={seed}"]
            if output_root is not None:
                common_overrides.append(f"runtime.output_root={output_root}")
            preflight_pretrain = load_config(
                source.parent / str(row["pretrain"]),
                overrides=[
                    *common_overrides,
                    *common_pretrain_overrides,
                    *row.get("pretrain_overrides", []),
                ],
            )
            preflight_finetune = load_config(
                source.parent / str(row["finetune"]),
                overrides=[
                    *common_overrides,
                    *common_finetune_overrides,
                    *row.get("finetune_overrides", []),
                ],
            )
            if preflight_pretrain.identity["dataset_id"] != dataset_id:
                raise ValueError("pipeline dataset_id differs from pretraining config")
            if preflight_finetune.identity["dataset_id"] != dataset_id:
                raise ValueError("pipeline dataset_id differs from fine-tuning config")
            _assert_scgpt_alignment(preflight_pretrain, preflight_finetune)

    for row in rows:
        dataset_id = str(row["dataset_id"])
        if dataset_id not in selected_ids:
            continue
        dataset_cache: dict[str, Any] = {}
        prior_cache: dict[str, PriorStore] = {}
        for seed in seeds:
            common_overrides = [*row.get("overrides", []), f"training.seed={seed}"]
            if output_root is not None:
                common_overrides.append(f"runtime.output_root={output_root}")
            pretrain = load_config(
                source.parent / str(row["pretrain"]),
                overrides=[
                    *common_overrides,
                    *common_pretrain_overrides,
                    *row.get("pretrain_overrides", []),
                ],
            )
            if pretrain.identity["dataset_id"] != dataset_id:
                raise ValueError("pipeline dataset_id differs from pretraining config")
            pretrain_receipt, pretrain_reused = _run_or_reuse(
                pretrain, dataset_cache=dataset_cache, prior_cache=prior_cache
            )
            checkpoint = Path(pretrain_receipt["run_root"]) / "checkpoint.pt"
            finetune = load_config(
                source.parent / str(row["finetune"]),
                overrides=[
                    *common_overrides,
                    *common_finetune_overrides,
                    *row.get("finetune_overrides", []),
                    f"model.pretrained_checkpoint={checkpoint}",
                ],
            )
            _assert_scgpt_alignment(pretrain, finetune)
            finetune_receipt, finetune_reused = _run_or_reuse(
                finetune, dataset_cache=dataset_cache, prior_cache=prior_cache
            )
            results.append(
                {
                    "dataset_id": dataset_id,
                    "seed": seed,
                    "pretrain": {
                        "status": pretrain_receipt["status"],
                        "reused": pretrain_reused,
                        "epochs": pretrain_receipt["training"]["epochs"],
                        "run_root": pretrain_receipt["run_root"],
                    },
                    "finetune": {
                        "status": finetune_receipt["status"],
                        "reused": finetune_reused,
                        "epochs": finetune_receipt["training"]["epochs"],
                        "best_epoch": finetune_receipt["training"].get("best_epoch"),
                        "metrics": _metric_summary(finetune_receipt),
                        "run_root": finetune_receipt["run_root"],
                    },
                    "dataset_fingerprint": finetune_receipt["dataset_fingerprint"],
                    "split_sha256": finetune_receipt["input_manifest"]["dataset"][
                        "split_sha256"
                    ],
                    "fairness_contract_sha256": finetune_receipt[
                        "fairness_contract"
                    ]["sha256"],
                    "parameters": finetune_receipt["parameters"],
                }
            )
            progressive = {
                "schema_version": "dinogenept-pipeline-result-v1",
                "status": "running",
                "grid": {
                    "expected_datasets": dataset_ids,
                    "selected_datasets": list(selected_ids),
                    "expected_seeds": full_seeds,
                    "selected_seeds": list(seeds),
                    "expected_stage_runs": len(dataset_ids) * len(full_seeds) * 2,
                    "selected_stage_runs": len(selected_ids) * len(seeds) * 2,
                    "completed_stage_runs": len(results) * 2,
                },
                "seeds": list(seeds),
                "scgpt_alignment": {
                    "architecture": SCGPT_SIGNATURE,
                    "pretrain_epochs": 6,
                    "finetune_epochs": 15,
                    "optimizer": "adam",
                    "learning_rate": 0.0001,
                    "lr_gamma": 0.9,
                },
                "rows": results,
            }
            atomic_write_json(receipt_path, progressive)
        dataset_cache.clear()
        prior_cache.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ModuleNotFoundError:
            pass
    if not results:
        raise ValueError("pipeline selection matched no datasets")
    full_grid = set(selected_ids) == set(dataset_ids) and set(seeds) == set(full_seeds)
    expected_rows = len(selected_ids) * len(seeds)
    if len(results) != expected_rows:
        raise RuntimeError("pipeline completed row count differs from selected grid")
    result = {
        "schema_version": "dinogenept-pipeline-result-v1",
        "status": "complete" if full_grid else "partial_complete",
        "grid": {
            "expected_datasets": dataset_ids,
            "selected_datasets": list(selected_ids),
            "expected_seeds": full_seeds,
            "selected_seeds": list(seeds),
            "expected_stage_runs": len(dataset_ids) * len(full_seeds) * 2,
            "selected_stage_runs": len(selected_ids) * len(seeds) * 2,
            "completed_stage_runs": len(results) * 2,
        },
        "seeds": list(seeds),
        "scgpt_alignment": {
            "architecture": SCGPT_SIGNATURE,
            "pretrain_epochs": 6,
            "finetune_epochs": 15,
            "optimizer": "adam",
            "learning_rate": 0.0001,
            "lr_gamma": 0.9,
        },
        "row_count": len(results),
        "rows": results,
    }
    atomic_write_json(receipt_path, result)
    return result
