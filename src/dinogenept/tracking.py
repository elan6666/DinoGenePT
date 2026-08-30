"""Optional Hugging Face Trackio mirror for experiment observability.

Scientific receipts under ``results/`` remain authoritative. Trackio is a
comparison and visualization surface, never an input to training or reuse.
"""

from __future__ import annotations

import importlib
import os
import site
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _tracking_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("runtime", {}).get("tracking", {})
    if not isinstance(value, dict):
        raise ValueError("runtime.tracking must be a mapping")
    return value


def _run_name(config: dict[str, Any], config_sha256: str) -> str:
    identity = config["identity"]
    seed = int(config["training"]["seed"])
    return (
        f"{identity['model_id']}--{identity['dataset_id']}--"
        f"{identity['experiment_id']}--seed-{seed}--{config_sha256[:8]}"
    )


def _comparison_config(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    identity = config["identity"]
    training = config["training"]
    ablation = config["ablation"]
    return {
        "config_sha256": config_sha256,
        "model_id": identity["model_id"],
        "dataset_id": identity["dataset_id"],
        "experiment_id": identity["experiment_id"],
        "seed": int(training["seed"]),
        "phase": str(training.get("phase", "finetune")),
        "epochs": int(training["epochs"]),
        "batch_size": int(training["batch_size"]),
        "bag_size": int(training.get("bag_size", 0)),
        "learning_rate": float(training["learning_rate"]),
        "dino": bool(ablation.get("dino", False)),
        "delta_ibot": bool(ablation.get("delta_ibot", False)),
        "dynamic_locals": bool(ablation.get("dynamic_locals", False)),
        "local_sources": ",".join(
            str(item) for item in ablation.get("local_sources", [])
        ),
        "adapter": str(ablation.get("adapter", {}).get("type", "none")),
        "koleo_weight": float(ablation.get("koleo_weight", 0.0)),
    }


def _epoch_metrics(row: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {"epoch": float(row["epoch"])}
    for key, value in row.items():
        if key == "epoch" or value is None or not isinstance(value, (int, float)):
            continue
        if key == "train_loss":
            name = "train/loss"
        elif key == "validation_loss":
            name = "validation/loss"
        elif key == "validation_prediction_mse":
            name = "validation/prediction_mse"
        elif key == "learning_rate":
            name = "optimizer/learning_rate"
        elif key in {"batches", "steps"}:
            name = f"progress/{key}"
        else:
            name = f"train/{key}"
        metrics[name] = float(value)
    if "total" in row:
        metrics["train/loss"] = float(row["total"])
    return metrics


def _final_metrics(metrics: dict[str, Any], elapsed_seconds: float) -> dict[str, float]:
    result = {"run/elapsed_seconds": float(elapsed_seconds)}
    summary = metrics.get("summary", {})
    if isinstance(summary, dict):
        for name, values in summary.items():
            if not isinstance(values, dict):
                continue
            for statistic in ("macro_mean", "macro_median", "micro"):
                value = values.get(statistic)
                if isinstance(value, (int, float)):
                    result[f"evaluation/{name}/{statistic}"] = float(value)
    return result


@dataclass
class TrackioRun:
    """Small fail-fast adapter around Trackio's process-global API."""

    enabled: bool
    project: str | None = None
    run_name: str | None = None
    space_id: str | None = None
    sync_space_id: str | None = None
    _trackio: Any = field(default=None, repr=False)
    epoch_rows: int = 0
    finished: bool = False

    @classmethod
    def start(cls, config: dict[str, Any], config_sha256: str) -> TrackioRun:
        tracking = _tracking_config(config)
        enabled = bool(tracking.get("enabled", False))
        if not enabled:
            return cls(enabled=False)
        provider = str(tracking.get("provider", "trackio"))
        if provider != "trackio":
            raise ValueError(f"unsupported experiment tracker: {provider!r}")
        project = str(tracking.get("project", "dinogenept-ablation")).strip()
        if not project:
            raise ValueError("runtime.tracking.project must be non-empty")
        raw_space_id = tracking.get("space_id")
        space_id = str(raw_space_id).strip() if raw_space_id else None
        raw_sync_space_id = tracking.get("sync_space_id")
        sync_space_id = str(raw_sync_space_id).strip() if raw_sync_space_id else None
        site_path = os.environ.get("DINOGENEPT_TRACKIO_SITE", "").strip()
        if site_path:
            site.addsitedir(str(Path(site_path).expanduser().resolve(strict=True)))
        local_dir = tracking.get("local_dir")
        if local_dir:
            os.environ.setdefault(
                "TRACKIO_DIR", str(Path(str(local_dir)).expanduser().resolve())
            )
        try:
            trackio = importlib.import_module("trackio")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Trackio tracking requires `pip install -e '.[tracking]'`"
            ) from error
        run_name = _run_name(config, config_sha256)
        kwargs: dict[str, Any] = {
            "project": project,
            "name": run_name,
            "group": (
                f"{config['identity']['model_id']}/"
                f"{config['identity']['experiment_id']}"
            ),
            "config": _comparison_config(config, config_sha256),
            "resume": "allow",
        }
        if space_id:
            kwargs["space_id"] = space_id
            kwargs["private"] = bool(tracking.get("private", True))
        kwargs["auto_log_gpu"] = bool(tracking.get("auto_log_gpu", True))
        kwargs["auto_log_cpu"] = bool(tracking.get("auto_log_cpu", True))
        trackio.init(**kwargs)
        return cls(
            enabled=True,
            project=project,
            run_name=run_name,
            space_id=space_id,
            sync_space_id=sync_space_id,
            _trackio=trackio,
        )

    def log_epoch(self, row: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._trackio.log(_epoch_metrics(row))
        self.epoch_rows += 1

    def log_final(self, metrics: dict[str, Any], elapsed_seconds: float) -> None:
        if self.enabled:
            self._trackio.log(_final_metrics(metrics, elapsed_seconds))

    def finish(self) -> None:
        if self.enabled and not self.finished:
            self._trackio.finish()
            self.finished = True

    def receipt(self) -> dict[str, Any]:
        return {
            "provider": "trackio",
            "enabled": self.enabled,
            "project": self.project,
            "run_name": self.run_name,
            "space_id": self.space_id,
            "sync_space_id": self.sync_space_id,
            "epoch_rows": self.epoch_rows,
            "finished": self.finished,
            "scientific_source_of_truth": False,
        }
