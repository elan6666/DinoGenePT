"""One-run orchestration with strict output identity and shared evaluation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from dinogenept import __version__
from dinogenept.config import ResolvedConfig
from dinogenept.priors import PriorStore
from dinogenept.registry import DATASETS, EVALUATORS, MODELS
from genept_seed.provenance import atomic_write_json, digest_file, environment_manifest


def _torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("training requires `pip install -e '.[train]'`") from error
    return torch


def _dataset_cache_key(config: dict[str, Any]) -> str:
    identity = {
        "dataset": config["dataset"],
        "model_max_seq_len": config["model"]["max_seq_len"],
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _prior_cache_key(
    config: dict[str, Any], required_targets: tuple[str, ...]
) -> str:
    identity = {"priors": config["priors"], "required_targets": required_targets}
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def implementation_sha256() -> str:
    """Hash the executable DinoGenePT source, independent of Git state."""
    source_root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for package_name in ("dinogenept", "genept_seed"):
        for source in sorted((source_root / package_name).rglob("*.py")):
            digest.update(source.relative_to(source_root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(source.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _validate_execution_site(config: dict[str, Any]) -> None:
    runtime = config["runtime"]
    if not runtime.get("enforce_server", False):
        return
    required_root = Path(str(runtime["required_work_root"])).resolve()
    if Path.cwd().resolve() != required_root:
        raise RuntimeError(
            f"training is restricted to {required_root}; current={Path.cwd().resolve()}"
        )
    required_visible = str(runtime["required_cuda_visible_devices"])
    actual_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if actual_visible != required_visible:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES does not match the server execution contract: "
            f"expected={required_visible!r}, actual={actual_visible!r}"
        )
    if not str(runtime.get("device", "")).startswith("cuda"):
        raise RuntimeError("server-enforced runs require a CUDA device")


def _physical_gpu_identity(visible_devices: str | None) -> dict[str, str | None]:
    result: dict[str, str | None] = {
        "cuda_visible_devices": visible_devices,
        "physical_index": None,
        "physical_uuid": None,
    }
    if not visible_devices or not visible_devices.isdigit():
        return result
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return result
    if completed.returncode:
        return result
    for line in completed.stdout.splitlines():
        index, separator, uuid = line.partition(",")
        if separator and index.strip() == visible_devices:
            result["physical_index"] = index.strip()
            result["physical_uuid"] = uuid.strip()
            break
    return result


def _runtime_environment(torch: Any) -> dict[str, Any]:
    environment: dict[str, Any] = environment_manifest()
    dependencies: dict[str, str | None] = {}
    for distribution in ("numpy", "scipy", "pandas", "anndata", "torch", "PyYAML"):
        try:
            dependencies[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependencies[distribution] = None
    environment["dependencies"] = dependencies
    environment["cuda_runtime"] = torch.version.cuda
    environment["cudnn"] = (
        str(torch.backends.cudnn.version())
        if torch.backends.cudnn.is_available()
        else None
    )
    return environment


def _device(config: dict[str, Any]):
    torch = _torch()
    requested = str(config["runtime"].get("device", "cpu"))
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device(requested)
        torch.cuda.set_device(device)
        fraction = float(config["runtime"].get("max_gpu_memory_fraction", 0.0))
        if fraction:
            torch.cuda.set_per_process_memory_fraction(fraction, device=device)
        free, total = torch.cuda.mem_get_info(device)
        minimum = int(config["runtime"].get("min_free_gpu_mb", 0)) * 1024 * 1024
        if free < minimum:
            raise RuntimeError(
                f"refusing GPU launch: free={free // 2**20}MiB, required={minimum // 2**20}MiB"
            )
        gpu = {
            "free_before_mb": free // 2**20,
            "total_mb": total // 2**20,
            "visible_device_index": device.index if device.index is not None else 0,
            "device_name": torch.cuda.get_device_name(device),
            **_physical_gpu_identity(os.environ.get("CUDA_VISIBLE_DEVICES")),
        }
        return device, gpu
    return torch.device("cpu"), {
        "free_before_mb": None,
        "total_mb": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def condition_targets(data) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for condition, targets in zip(data.conditions, data.targets, strict=True):
        if condition != "ctrl":
            result.setdefault(condition, targets)
    return result


def run_root(config: ResolvedConfig) -> Path:
    identity = config.identity
    root = Path(config.payload["runtime"]["output_root"])
    return (
        root
        / identity["dataset_id"]
        / identity["model_id"]
        / identity["experiment_id"]
        / f"seed-{int(config.payload['training']['seed'])}"
    ).resolve()


def build_input_manifest(
    config: dict[str, Any], data: Any, priors: PriorStore
) -> dict[str, Any]:
    checkpoint = config["model"].get("pretrained_checkpoint")
    return {
        "dataset": {
            "fingerprint": data.fingerprint,
            "source_sha256": data.source_sha256,
            "split_sha256": data.split_sha256,
        },
        "priors": priors.input_manifest,
        "pretrained_checkpoint_sha256": (
            digest_file(Path(str(checkpoint)).resolve(strict=True)) if checkpoint else None
        ),
    }


def _archive_incomplete_run(output: Path) -> None:
    attempts = output / "attempts"
    attempt = 1
    while (attempts / f"attempt-{attempt}").exists():
        attempt += 1
    destination = attempts / f"attempt-{attempt}"
    destination.mkdir(parents=True)
    for item in tuple(output.iterdir()):
        if item != attempts:
            item.rename(destination / item.name)


def _atomic_torch_save(torch: Any, payload: Any, target: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def validate_completed_run(
    *,
    config: ResolvedConfig,
    receipt: dict[str, Any],
    data: Any,
    priors: PriorStore,
    implementation_hash: str,
) -> None:
    output = run_root(config)
    expected_input = build_input_manifest(config.payload, data, priors)
    if (
        receipt.get("status") != "complete"
        or receipt.get("config_sha256") != config.sha256
        or receipt.get("implementation_sha256") != implementation_hash
        or receipt.get("input_manifest") != expected_input
        or receipt.get("dataset_fingerprint") != data.fingerprint
        or receipt.get("run_root") != str(output)
    ):
        raise ValueError(f"completed run receipt identity mismatch: {output / 'run.json'}")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"completed run has no artifact manifest: {output / 'run.json'}")
    expected_names = {"resolved-config.json", "metrics.json"}
    if config.payload["runtime"].get("save_checkpoint", False):
        expected_names.add("checkpoint.pt")
    if set(artifacts) != expected_names:
        raise ValueError(f"completed run artifact set mismatch: {output / 'run.json'}")
    for name, expected_sha256 in artifacts.items():
        artifact = output / name
        if not artifact.is_file() or digest_file(artifact) != expected_sha256:
            raise ValueError(f"completed run artifact hash mismatch: {artifact}")


def run_experiment(
    config: ResolvedConfig,
    *,
    dataset_cache: dict[str, Any] | None = None,
    prior_cache: dict[str, PriorStore] | None = None,
    retry_failed: bool = False,
) -> dict[str, Any]:
    torch = _torch()
    from dinogenept.experiments.training import set_reproducible_seed

    payload = config.payload
    _validate_execution_site(payload)
    output = run_root(config)
    if output.exists() and any(output.iterdir()):
        if not retry_failed or (output / "run.json").exists():
            raise ValueError(f"refusing to overwrite an existing run: {output}")
        _archive_incomplete_run(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    implementation_hash = implementation_sha256()
    try:
        seed = int(payload["training"]["seed"])
        set_reproducible_seed(seed)
        cache = dataset_cache if dataset_cache is not None else {}
        cache_key = _dataset_cache_key(payload)
        if cache_key not in cache:
            loader = DATASETS.resolve(config.identity["dataset_id"])
            cache[cache_key] = loader(payload)
        data = cache[cache_key]
        targets = tuple(sorted({target for item in data.targets for target in item}))
        priors_by_key = prior_cache if prior_cache is not None else {}
        prior_key = _prior_cache_key(payload, targets)
        if prior_key not in priors_by_key:
            priors_by_key[prior_key] = PriorStore.from_config(payload, targets)
        priors = priors_by_key[prior_key]
        input_manifest = build_input_manifest(payload, data, priors)
        device, gpu = _device(payload)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        plugin = MODELS.resolve(config.identity["model_id"])
        model = plugin.build_model(
            payload, n_genes=data.n_genes, prior_dimensions=priors.dimensions
        )
        pretrained_receipt = None
        checkpoint = payload["model"].get("pretrained_checkpoint")
        if checkpoint:
            pretrained_receipt = model.load_pretrained(
                checkpoint,
                minimum_match_fraction=float(
                    payload["model"].get("pretrained_minimum_match_fraction", 0.9)
                ),
            )
        model.to(device)
        trainer = plugin.build_trainer(
            model=model,
            data=data,
            priors=priors,
            config=payload,
            device=device,
        )
        training_receipt = trainer.train()
        predictions, truths, controls = plugin.predict_split(
            model=model,
            data=data,
            priors=priors,
            config=payload,
            device=device,
            split=str(payload["evaluation"].get("split", "test")),
        )
        evaluator = EVALUATORS.resolve(str(payload["evaluation"].get("name", "perturbation")))(
            payload
        )
        evaluated_conditions = data.split_conditions(
            str(payload["evaluation"].get("split", "test"))
        )
        targets_by_condition = condition_targets(data)
        availability = {
            condition: priors.available_optional(targets_by_condition[condition])
            for condition in evaluated_conditions
        }
        metrics = evaluator.evaluate(
            predictions=predictions,
            truths=truths,
            controls=controls,
            top_de_indices=data.top_de_indices,
            strata={
                "optional_source_count": {
                    condition: str(len(sources))
                    for condition, sources in availability.items()
                },
                "optional_source_pattern": {
                    condition: "+".join(sources) if sources else "base_only"
                    for condition, sources in availability.items()
                },
            },
        )
        if device.type == "cuda":
            gpu["peak_allocated_mb"] = int(torch.cuda.max_memory_allocated(device) // 2**20)
            gpu["peak_reserved_mb"] = int(torch.cuda.max_memory_reserved(device) // 2**20)
        receipt = {
            "schema_version": "dinogenept-run-v1",
            "status": "complete",
            "identity": config.identity,
            "config_sha256": config.sha256,
            "implementation_sha256": implementation_hash,
            "input_manifest": input_manifest,
            "dataset_fingerprint": data.fingerprint,
            "package_version": __version__,
            "environment": _runtime_environment(torch),
            "device": str(device),
            "gpu": gpu,
            "parameters": model.parameter_receipt(),
            "pretrained": pretrained_receipt,
            "prior_audit": priors.audit(condition_targets(data)),
            "training": training_receipt,
            "metrics": metrics,
            "elapsed_seconds": time.time() - started,
            "run_root": str(output),
        }
        resolved_path = output / "resolved-config.json"
        metrics_path = output / "metrics.json"
        atomic_write_json(resolved_path, payload)
        atomic_write_json(metrics_path, metrics)
        artifacts = {
            resolved_path.name: digest_file(resolved_path),
            metrics_path.name: digest_file(metrics_path),
        }
        if payload["runtime"].get("save_checkpoint", False):
            checkpoint_path = output / "checkpoint.pt"
            _atomic_torch_save(
                torch,
                {"model": model.state_dict(), "config_sha256": config.sha256},
                checkpoint_path,
            )
            artifacts[checkpoint_path.name] = digest_file(checkpoint_path)
        receipt["artifacts"] = artifacts
        atomic_write_json(output / "run.json", receipt)
        return receipt
    except BaseException as error:
        atomic_write_json(
            output / "failure.json",
            {
                "schema_version": "dinogenept-run-failure-v1",
                "status": "failed",
                "identity": config.identity,
                "config_sha256": config.sha256,
                "implementation_sha256": implementation_hash,
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.time() - started,
            },
        )
        raise


def run_permutation_check(config: ResolvedConfig) -> dict[str, Any]:
    _torch()
    from dinogenept.experiments.training import set_reproducible_seed

    payload = config.payload
    _validate_execution_site(payload)
    set_reproducible_seed(int(payload["training"]["seed"]))
    data = DATASETS.resolve(config.identity["dataset_id"])(payload)
    targets_by_condition = condition_targets(data)
    targets = tuple(sorted({target for item in targets_by_condition.values() for target in item}))
    priors = PriorStore.from_config(payload, targets)
    device, _ = _device(payload)
    plugin = MODELS.resolve(config.identity["model_id"])
    model = plugin.build_model(
        payload, n_genes=data.n_genes, prior_dimensions=priors.dimensions
    ).to(device)
    model.eval()
    audit = plugin.permutation_check(
        model=model,
        data=data,
        priors=priors,
        config=payload,
        device=device,
    )
    return {
        "schema_version": "dinogenept-permutation-check-v1",
        "identity": config.identity,
        "config_sha256": config.sha256,
        **audit,
    }
