"""Stateless continuous-expression crops; IDs/epoch determine all randomness.

CellFM source supplies weighted input capping and 20%/80% target/hidden masks.
Independent fractional crops, per-cell deterministic seeds and minimum-one
mask guards are declared project adaptations. No bins or rank-valued tokens.
"""

import hashlib
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CropConfig:
    cap: int = 2048
    local_count: int = 2
    global_scale: tuple[float, float] = (0.4, 1.0)
    local_scale: tuple[float, float] = (0.05, 0.4)
    target_fraction: float = 0.2
    hidden_fraction: float = 0.8
    seed: int = 42

    def __post_init__(self):
        if self.cap < 1 or self.local_count < 0:
            raise ValueError("Invalid crop counts")
        for low, high in (self.global_scale, self.local_scale):
            if not 0 < low <= high <= 1:
                raise ValueError("Crop scales must lie in (0,1]")
        if not 0 < self.target_fraction <= 1 or not 0 < self.hidden_fraction <= 1:
            raise ValueError("Invalid reconstruction/hidden fraction")


def cell_rng(seed: int, epoch: int, cell_id: str) -> np.random.Generator:
    if epoch < 0:
        raise ValueError("Negative epoch")
    digest = hashlib.sha256(f"{seed}\0{epoch}\0{cell_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "little"))


def sample_crops(gene_ids, counts, library_size: float, *, cell_id: str, epoch: int, config: CropConfig):
    ids = np.asarray(gene_ids)
    counts = np.asarray(counts, dtype=np.float64)
    if ids.ndim != 1 or counts.shape != ids.shape or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Expected matching flat integer gene IDs and counts")
    if ids.size == 0 or np.any(ids <= 0) or np.unique(ids).size != ids.size:
        raise ValueError("Measured genes must have positive unique IDs")
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("Counts must be finite and nonnegative")
    if not math.isfinite(library_size) or library_size <= 0 or counts.sum() > library_size * (1 + 1e-5):
        raise ValueError("Invalid full-cell library size")
    indices = np.flatnonzero(counts > 0)
    if not indices.size:
        raise ValueError("Pretraining cell has no positive mapped counts")
    rng = cell_rng(config.seed, epoch, cell_id)
    if indices.size > config.cap:
        weights = np.log1p(counts[indices])
        indices = rng.choice(indices, size=config.cap, replace=False, p=weights / weights.sum())
    indices = indices[np.argsort(ids[indices])]
    expression = np.log1p(1e4 * counts / library_size).astype(np.float32)
    views = []
    for view_id in range(2 + config.local_count):
        scale = config.global_scale if view_id < 2 else config.local_scale
        ratio = rng.uniform(*scale)
        size = min(indices.size, max(1, math.ceil(ratio * indices.size)))
        selected = rng.choice(indices, size=size, replace=False)
        selected = selected[np.argsort(ids[selected])]
        target, hidden = np.zeros(size, dtype=bool), np.zeros(size, dtype=bool)
        if view_id < 2:
            # Tiny crops must not inherit upstream's l==0/all-target edge case.
            n_target = max(1, int(config.target_fraction * size))
            chosen = rng.permutation(size)[:n_target]
            target[chosen] = True
            hidden[chosen[: max(1, int(config.hidden_fraction * n_target))]] = True
        views.append(
            {
                "gene_ids": ids[selected].astype(np.int64),
                "expression": expression[selected],
                "targets": target,
                "hidden": hidden,
                "ratio": float(ratio),
            }
        )
    return {"cell_id": str(cell_id), "capped_gene_ids": ids[indices].astype(np.int64), "views": views}


def collate_crops(rows):
    """Pad within a view, not to the corpus maximum; no cell duplication."""
    import torch

    if len(rows) < 2 or len({row["cell_id"] for row in rows}) != len(rows):
        raise ValueError("Each per-rank KoLeo batch needs >=2 distinct cell IDs")
    n_views = len(rows[0]["views"])
    if any(len(row["views"]) != n_views for row in rows):
        raise ValueError("Mixed view configurations")
    batches = []
    for view_id in range(n_views):
        size = max(len(row["views"][view_id]["gene_ids"]) for row in rows)
        arrays = {
            key: np.zeros((len(rows), size), dtype=dtype)
            for key, dtype in (
                ("gene_ids", np.int64),
                ("expression", np.float32),
                ("targets", bool),
                ("hidden", bool),
                ("valid", bool),
            )
        }
        for i, row in enumerate(rows):
            view = row["views"][view_id]
            length = len(view["gene_ids"])
            for key in ("gene_ids", "expression", "targets", "hidden"):
                arrays[key][i, :length] = view[key]
            arrays["valid"][i, :length] = True
        batches.append({key: torch.from_numpy(array) for key, array in arrays.items()})
    return {"cell_ids": [row["cell_id"] for row in rows], "views": batches}


def epoch_batches(count: int, per_rank: int, world_size: int, rank: int, *, seed: int, epoch: int):
    """Identical step counts across ranks without padding/repeating/dropping cells."""
    if world_size < 1 or not 0 <= rank < world_size or per_rank < 2 or count < 2 * world_size:
        raise ValueError("Insufficient cells or invalid distributed batch settings")
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(count)
    n_batches = math.ceil(count / (per_rank * world_size))
    if count // n_batches < 2 * world_size:
        raise ValueError("Batch size cannot cover all cells with >=2 cells per rank; increase per-rank batch")
    for global_batch in np.array_split(order, n_batches):
        yield np.array_split(global_batch, world_size)[rank].tolist()
