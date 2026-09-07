"""Atomic weights-only-loadable checkpoints with explicit resume contracts."""

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def config_hash(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def rng_state():
    state = np.random.get_state()
    # Do not initialize contexts on another GPU merely to serialize its RNG.
    # Each DDP rank / single-GPU task only draws randomness on its current GPU.
    devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
        "cuda": [torch.cuda.get_rng_state(device) for device in devices],
        "cuda_device_indices": devices,
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    name, keys, position, has_gauss, cached_gauss = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gauss))
    if state["cuda"]:
        devices = state.get("cuda_device_indices", list(range(len(state["cuda"]))))
        if len(devices) != len(state["cuda"]) or any(i < 0 or i >= torch.cuda.device_count() for i in devices):
            raise ValueError("Checkpoint CUDA RNG device mapping differs")
        for device, item in zip(devices, state["cuda"], strict=True):
            torch.cuda.set_rng_state(item.cpu(), device=device)


def save_checkpoint(path: Path, model, optimizer, *, config: dict, progress: dict, rank_rng: list[dict]):
    """Only save at optimizer boundaries; no pending gradients or raw predictions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "dinogenept.training.v1",
        "config": config,
        "config_sha256": config_hash(config),
        "progress": progress,
        "rank_rng": rank_rng,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def load_checkpoint(path: Path, model, optimizer, *, config: dict, rank: int = 0):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "dinogenept.training.v1" or payload.get("config_sha256") != config_hash(config):
        raise ValueError("Checkpoint schema/config mismatch; do not resume a different experiment")
    if config_hash(payload["config"]) != payload["config_sha256"]:
        raise ValueError("Stored checkpoint config checksum mismatch")
    if not 0 <= rank < len(payload["rank_rng"]):
        raise ValueError("Missing rank-specific RNG state")
    # Check all model axes before any partial load changes live parameters.
    current = model.state_dict()
    if current.keys() != payload["model"].keys() or any(
        value.shape != payload["model"][key].shape for key, value in current.items()
    ):
        raise ValueError("Checkpoint model structure mismatch")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload["rank_rng"][rank])
    return payload["progress"]
