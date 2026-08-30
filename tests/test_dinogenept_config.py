from pathlib import Path

import numpy as np
import pytest
import yaml

from dinogenept.config import load_config, validate_config
from dinogenept.experiments.matrix import run_matrix
from dinogenept.experiments.runner import (
    build_input_manifest,
    implementation_sha256,
    run_root,
)
from dinogenept.priors import PriorStore
from dinogenept.registry import DATASETS, EVALUATORS, MODELS
from genept_seed.provenance import atomic_write_json, digest_file

ROOT = Path(__file__).resolve().parents[1]


def test_composed_smoke_config_is_one_model_and_one_dataset():
    config = load_config(
        ROOT
        / "configs/experiments/dinogenept/ablations/07-dynamic-locals.yaml"
    )
    assert config.identity == {
        "model_id": "dinogenept",
        "dataset_id": "adamson",
        "experiment_id": "dynamic-locals",
    }
    assert config.payload["runtime"]["smoke"] is True
    assert config.payload["training"]["epochs"] == 1
    assert config.payload["priors"]["optional"]["go"]["source_only"] is True


def test_invalid_dynamic_local_without_dino_fails_closed():
    config = load_config(
        ROOT / "configs/experiments/dinogenept/ablations/00-supervised.yaml"
    )
    payload = config.payload
    payload["ablation"]["dynamic_locals"] = True
    with pytest.raises(ValueError, match="require the DINO"):
        validate_config(payload)


def test_registries_expose_model_dataset_and_common_evaluator():
    assert MODELS.names() == ("dinogenept",)
    assert {"adamson", "norman", "replogle_k562", "replogle_rpe1"}.issubset(
        DATASETS.names()
    )
    assert EVALUATORS.names() == ("perturbation",)


def test_matrix_reuses_matching_completed_receipt(tmp_path):
    source_config = (
        ROOT / "configs/experiments/dinogenept/ablations/00-supervised.yaml"
    )
    dataset_path = tmp_path / "dataset.npz"
    np.savez(
        dataset_path,
        expression=np.arange(28, dtype=np.float32).reshape(7, 4),
        genes=np.asarray(["A", "B", "C", "D"]),
        conditions=np.asarray(
            ["ctrl", "ctrl", "A+ctrl", "B+ctrl", "C+ctrl", "D+ctrl", "E+ctrl"]
        ),
    )
    output_root = tmp_path / "outputs"
    row_overrides = [
        f"dataset.path={dataset_path}",
        "runtime.enforce_server=false",
        "runtime.device=cpu",
    ]
    config = load_config(
        source_config,
        overrides=[*row_overrides, f"runtime.output_root={output_root}"],
    )
    data = DATASETS.resolve("adamson")(config.payload)
    targets = tuple(sorted({target for item in data.targets for target in item}))
    priors = PriorStore.from_config(config.payload, targets)
    completed = run_root(config) / "run.json"
    completed.parent.mkdir(parents=True)
    metrics = {
        "summary": {"mse": {"macro_mean": 0.5}},
        "per_condition": {
            condition: {"mse": 0.5} for condition in data.split_conditions("test")
        },
    }
    resolved_path = completed.parent / "resolved-config.json"
    metrics_path = completed.parent / "metrics.json"
    checkpoint_path = completed.parent / "checkpoint.pt"
    atomic_write_json(resolved_path, config.payload)
    atomic_write_json(metrics_path, metrics)
    checkpoint_path.write_bytes(b"deterministic-test-checkpoint")
    atomic_write_json(
        completed,
        {
            "status": "complete",
            "config_sha256": config.sha256,
            "implementation_sha256": implementation_sha256(),
            "input_manifest": build_input_manifest(config.payload, data, priors),
            "dataset_fingerprint": data.fingerprint,
            "run_root": str(completed.parent),
            "elapsed_seconds": 1.0,
            "gpu": {"peak_allocated_mb": 1},
            "metrics": metrics,
            "artifacts": {
                resolved_path.name: digest_file(resolved_path),
                metrics_path.name: digest_file(metrics_path),
                checkpoint_path.name: digest_file(checkpoint_path),
            },
        },
    )
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "schema_version": "dinogenept-matrix-v1",
                "smoke": True,
                "receipt": "matrix.json",
                "rows": [{"config": str(source_config), "overrides": row_overrides}],
            }
        ),
        encoding="utf-8",
    )
    result = run_matrix(matrix, device=None, output_root=output_root)
    assert result["reused_count"] == 1
    assert result["rows"][0]["metrics"]["mse"] == 0.5
    metrics_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        run_matrix(matrix, device=None, output_root=output_root)
