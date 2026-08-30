"""Audited identity for the external Scouter implementation.

This module deliberately avoids importing torch so run-reuse validation can bind
the installed Scouter wheel without materializing a training model.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
from pathlib import Path
from typing import Any

SCOUTER_VERSION = "0.1.10"
SCOUTER_SOURCE_SHA256 = (
    "d704dff48bf08dff0392c234c0c2c54456a9717e0e13a307833c1d6842d2dd6d"
)
SCOUTER_BALANCED_DATASET_SEED = 24


def scouter_source_sha256() -> str:
    files = importlib.metadata.files("scouter-learn")
    if files is None:
        raise RuntimeError("scouter-learn distribution has no file manifest")
    digest = hashlib.sha256()
    selected = [
        item
        for item in files
        if str(item).startswith("scouter/") and item.suffix == ".py"
    ]
    if not selected:
        raise RuntimeError("scouter-learn distribution has no auditable Python sources")
    for item in sorted(selected, key=str):
        path = item.locate()
        digest.update(str(item).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def upstream_identity() -> dict[str, Any]:
    try:
        version = importlib.metadata.version("scouter-learn")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("Scouter baseline requires scouter-learn==0.1.10") from error
    if version != SCOUTER_VERSION:
        raise RuntimeError(
            f"Scouter baseline requires scouter-learn=={SCOUTER_VERSION}; found {version}"
        )
    observed_source = scouter_source_sha256()
    if observed_source != SCOUTER_SOURCE_SHA256:
        raise RuntimeError(
            "scouter-learn source differs from the audited 0.1.10 wheel: "
            f"expected={SCOUTER_SOURCE_SHA256}, observed={observed_source}"
        )
    try:
        from scouter import Scouter, ScouterData
        from scouter._datasets import BalancedDataset
    except ModuleNotFoundError as error:
        raise RuntimeError("Scouter baseline requires scouter-learn==0.1.10") from error
    pairing_seed = inspect.signature(BalancedDataset.__init__).parameters[
        "seed"
    ].default
    if pairing_seed != SCOUTER_BALANCED_DATASET_SEED:
        raise RuntimeError(
            "Scouter BalancedDataset default seed differs from the audited package"
        )
    return {
        "package": "scouter-learn",
        "version": version,
        "source_sha256": observed_source,
        "balanced_dataset_seed": pairing_seed,
        "module_paths": {
            "Scouter": str(Path(inspect.getfile(Scouter)).resolve(strict=True)),
            "ScouterData": str(Path(inspect.getfile(ScouterData)).resolve(strict=True)),
            "BalancedDataset": str(
                Path(inspect.getfile(BalancedDataset)).resolve(strict=True)
            ),
        },
    }


def input_manifest(config: dict[str, Any]) -> dict[str, Any]:
    del config
    return upstream_identity()
