"""Strict, composable experiment configuration loading."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "dinogenept-experiment-v1"
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?}")
_TOP_LEVEL = {
    "schema_version",
    "identity",
    "dataset",
    "priors",
    "model",
    "training",
    "evaluation",
    "runtime",
    "ablation",
}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _interpolate_env(text: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        variable, default = match.group(1), match.group(2)
        if variable in os.environ:
            return os.environ[variable]
        if default is not None:
            return default
        raise ValueError(f"configuration requires environment variable {variable}")

    return _ENV_PATTERN.sub(replacement, text)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_mapping(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    source = path.resolve(strict=True)
    if source in stack:
        chain = " -> ".join(str(item) for item in (*stack, source))
        raise ValueError(f"cyclic configuration include: {chain}")
    payload = yaml.safe_load(_interpolate_env(source.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping: {source}")
    includes = payload.pop("includes", [])
    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, list) or not all(isinstance(item, str) for item in includes):
        raise ValueError(f"includes must be a list of paths: {source}")
    resolved: dict[str, Any] = {}
    for item in includes:
        included = (source.parent / item).resolve()
        resolved = _deep_merge(resolved, _load_mapping(included, (*stack, source)))
    return _deep_merge(resolved, payload)


def _parse_override(raw: str) -> tuple[list[str], Any]:
    if "=" not in raw:
        raise ValueError(f"override must be KEY=VALUE: {raw}")
    key, value = raw.split("=", 1)
    parts = [part for part in key.split(".") if part]
    if not parts:
        raise ValueError(f"override key is empty: {raw}")
    return parts, yaml.safe_load(value)


def _apply_override(payload: dict[str, Any], raw: str) -> None:
    parts, value = _parse_override(raw)
    cursor: dict[str, Any] = payload
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"override crosses a scalar at {part}: {raw}")
        cursor = child
    cursor[parts[-1]] = value


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"configuration section {key!r} must be a mapping")
    return value


def validate_config(payload: dict[str, Any]) -> None:
    unknown = sorted(set(payload) - _TOP_LEVEL)
    if unknown:
        raise ValueError(f"unknown top-level configuration keys: {unknown}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    identity = _require_mapping(payload, "identity")
    for key in ("model_id", "dataset_id", "experiment_id"):
        if not isinstance(identity.get(key), str) or not identity[key].strip():
            raise ValueError(f"identity.{key} must be a non-empty string")
        if not _IDENTIFIER.fullmatch(identity[key]):
            raise ValueError(f"identity.{key} is not a safe path component: {identity[key]!r}")
    dataset = _require_mapping(payload, "dataset")
    if dataset.get("adapter") != identity["dataset_id"]:
        raise ValueError("dataset adapter and identity.dataset_id differ")
    model = _require_mapping(payload, "model")
    if model.get("name") != identity["model_id"]:
        raise ValueError("model name and identity.model_id differ")
    model_id = identity["model_id"]
    if model_id == "dinogenept":
        for key in ("d_model", "layers", "heads", "ff_dim", "max_seq_len"):
            if int(model.get(key, 0)) <= 0:
                raise ValueError(f"model.{key} must be positive")
        if int(model["d_model"]) % int(model["heads"]):
            raise ValueError("model.heads must divide model.d_model")
        if not isinstance(model.get("pretrained_strict", False), bool):
            raise ValueError("model.pretrained_strict must be boolean")
        dino_head = model.get("dino_head")
        if not isinstance(dino_head, dict) or any(
            int(dino_head.get(key, 0)) <= 0
            for key in ("hidden_dim", "bottleneck_dim", "prototypes")
        ):
            raise ValueError("model.dino_head dimensions must be positive")
        mask = model.get("mask")
        if not isinstance(mask, dict) or mask.get("gene_selection") not in {
            "train_variance",
            "post_topk",
        }:
            raise ValueError("model.mask.gene_selection must be train_variance or post_topk")
    elif model_id == "scouter":
        architecture = model.get("architecture")
        if not isinstance(architecture, dict):
            raise ValueError("Scouter requires model.architecture")
        if tuple(architecture.get("encoder", ())) != (2048, 512):
            raise ValueError("Scouter encoder must preserve the official (2048, 512) default")
        if int(architecture.get("encoder_output", 0)) != 64:
            raise ValueError("Scouter encoder_output must preserve the official default 64")
        if tuple(architecture.get("decoder", ())) != (2048,):
            raise ValueError("Scouter decoder must preserve the official (2048,) default")
    else:
        raise ValueError(f"unsupported model identity: {model_id}")
    training = _require_mapping(payload, "training")
    if int(training.get("epochs", 0)) <= 0 or int(training.get("batch_size", 0)) <= 0:
        raise ValueError("training epochs and batch_size must be positive")
    if model_id == "dinogenept" and int(training.get("bag_size", 0)) <= 0:
        raise ValueError("training.bag_size must be positive")
    if float(training.get("learning_rate", 0.0)) <= 0:
        raise ValueError("training.learning_rate must be positive")
    allowed_phases = {"pretrain", "finetune"} if model_id == "dinogenept" else {"finetune"}
    if training.get("phase", "finetune") not in allowed_phases:
        raise ValueError("training.phase must be pretrain or finetune")
    if training.get("optimizer", "adamw") not in {"adam", "adamw"}:
        raise ValueError("training.optimizer must be adam or adamw")
    if training.get("selection_metric", "none") not in {
        "none",
        "validation_prediction_mse",
    }:
        raise ValueError("unsupported training.selection_metric")
    runtime = _require_mapping(payload, "runtime")
    if not isinstance(runtime.get("output_root"), str) or not runtime["output_root"]:
        raise ValueError("runtime.output_root must be a non-empty path")
    if runtime.get("enforce_server", False):
        if not isinstance(runtime.get("required_work_root"), str) or not runtime[
            "required_work_root"
        ]:
            raise ValueError("server enforcement requires runtime.required_work_root")
        if not isinstance(runtime.get("required_cuda_visible_devices"), str):
            raise ValueError(
                "server enforcement requires runtime.required_cuda_visible_devices"
            )
    tracking = runtime.get("tracking", {})
    if not isinstance(tracking, dict):
        raise ValueError("runtime.tracking must be a mapping")
    if tracking.get("enabled", False):
        if tracking.get("provider", "trackio") != "trackio":
            raise ValueError("runtime.tracking.provider must be trackio")
        if not isinstance(tracking.get("project"), str) or not tracking["project"].strip():
            raise ValueError("enabled Trackio requires runtime.tracking.project")
        for key in ("space_id", "sync_space_id"):
            value = tracking.get(key)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"runtime.tracking.{key} must be a string or null")
        local_dir = tracking.get("local_dir")
        if local_dir is not None and (
            not isinstance(local_dir, str) or not local_dir.strip()
        ):
            raise ValueError("runtime.tracking.local_dir must be a non-empty string")
        for key in ("private", "auto_log_gpu", "auto_log_cpu"):
            if key in tracking and not isinstance(tracking[key], bool):
                raise ValueError(f"runtime.tracking.{key} must be boolean")
    priors = _require_mapping(payload, "priors")
    base = priors.get("base")
    if not isinstance(base, dict):
        raise ValueError("priors.base must be configured")
    smoke = bool(runtime.get("smoke", False))
    if not smoke and base.get("fixture"):
        raise ValueError("fixture priors are forbidden outside smoke mode")
    formal_prior_fields = (
        "manifest",
        "profile",
        "source_manifest",
        "expected_source_manifest_sha256",
        "expected_corpus_sha256",
        "expected_gene_universe_sha256",
        "expected_text_fingerprint_sha256",
        "expected_embedding_model",
        "expected_gene_case",
    )
    if not smoke and any(
        not isinstance(base.get(key), str) or not base[key]
        for key in formal_prior_fields
    ):
        raise ValueError("formal Base prior lacks preregistered corpus lineage")
    optional = priors.get("optional", {})
    if not isinstance(optional, dict):
        raise ValueError("priors.optional must be a mapping")
    for source, source_config in optional.items():
        if not isinstance(source_config, dict):
            raise ValueError(f"optional prior {source!r} must be a mapping")
        if not smoke and not source_config.get("source_only", False):
            raise ValueError(f"formal optional prior {source!r} must declare source_only: true")
        if not smoke and source_config.get("fixture"):
            raise ValueError("fixture optional priors are forbidden outside smoke mode")
        if not smoke and any(
            not isinstance(source_config.get(key), str) or not source_config[key]
            for key in formal_prior_fields
        ):
            raise ValueError(
                f"formal optional prior {source!r} lacks preregistered corpus lineage"
            )
        if source == "base":
            raise ValueError("base cannot also be configured as an optional prior")
    if model_id == "scouter" and any(
        source_config.get("enabled", True) for source_config in optional.values()
    ):
        raise ValueError("Scouter baseline is Base-only; optional priors are forbidden")
    evaluation = _require_mapping(payload, "evaluation")
    if evaluation.get("name") == "gradpert_exact" and evaluation.get(
        "normalization"
    ) != "none":
        raise ValueError("GraD-Pert exact evaluation requires normalization=none")
    split = dataset.get("split")
    if not isinstance(split, dict):
        raise ValueError("dataset.split must be configured")
    if not smoke and split.get("strategy") != "manifest":
        raise ValueError("formal runs require a frozen split manifest")
    ablation = _require_mapping(payload, "ablation")
    if model_id == "scouter":
        if any(
            (
                ablation.get("dino", False),
                ablation.get("delta_ibot", False),
                ablation.get("dynamic_locals", False),
                bool(ablation.get("local_sources", [])),
            )
        ):
            raise ValueError("Scouter must not enable DinoGenePT ablations")
        return
    if float(ablation.get("koleo_weight", 0.0)) < 0:
        raise ValueError("ablation.koleo_weight must be non-negative")
    if ablation.get("kda", {}).get("enabled") and model.get("positional_embedding", False):
        raise ValueError("KDA experiments must not add an absolute positional embedding")
    if ablation.get("dynamic_locals") and not ablation.get("dino"):
        raise ValueError("dynamic local views require the DINO objective")
    local_sources = ablation.get("local_sources", [])
    if not isinstance(local_sources, list) or not all(
        isinstance(source, str) and source in optional for source in local_sources
    ):
        raise ValueError("ablation.local_sources must name configured optional priors")
    if ablation.get("source_control", "none") not in {
        "none",
        "shuffled",
        "availability_only",
    }:
        raise ValueError("ablation.source_control is unsupported")
    if ablation.get("ibot_target_control", "none") not in {
        "none",
        "shuffled_expression",
    }:
        raise ValueError("ablation.ibot_target_control is unsupported")
    kda = ablation.get("kda", {})
    if kda.get("enabled") and int(kda.get("global_every", 0)) <= 0:
        raise ValueError("ablation.kda.global_every must be positive")
    adapter = ablation.get("adapter", {})
    if adapter.get("type", "none") not in {"none", "dense", "moe"}:
        raise ValueError("ablation.adapter.type must be none, dense, or moe")
    if adapter.get("type") in {"dense", "moe"} and (
        int(adapter.get("top_layers", 0)) <= 0
        or int(adapter.get("latent_dim", 0)) <= 0
    ):
        raise ValueError("active adapters require positive top_layers and latent_dim")
    if adapter.get("type") == "moe":
        experts = int(adapter.get("experts", 0))
        top_k = int(adapter.get("top_k", 0))
        if experts < 2 or not 0 < top_k <= experts:
            raise ValueError("MoE experts/top_k are invalid")
        if adapter.get("balance", "auxiliary") not in {"auxiliary", "quantile", "none"}:
            raise ValueError("unsupported MoE balancing mode")


@dataclass(frozen=True)
class ResolvedConfig:
    source: Path
    payload: dict[str, Any]
    sha256: str

    @property
    def identity(self) -> dict[str, str]:
        return self.payload["identity"]


def load_config(path: str | Path, *, overrides: list[str] | None = None) -> ResolvedConfig:
    source = Path(path).resolve(strict=True)
    payload = _load_mapping(source, ())
    for raw in overrides or []:
        _apply_override(payload, raw)
    validate_config(payload)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return ResolvedConfig(
        source=source,
        payload=payload,
        sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )
