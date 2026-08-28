"""Embedding storage and compatibility helpers."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EmbeddingSet:
    genes: np.ndarray
    vectors: np.ndarray
    model: str

    def as_dict(self) -> dict[str, np.ndarray]:
        return {str(gene).upper(): self.vectors[i] for i, gene in enumerate(self.genes)}


def _validate(genes: np.ndarray, vectors: np.ndarray) -> None:
    if genes.ndim != 1 or vectors.ndim != 2:
        raise ValueError("genes must be 1-D and vectors must be 2-D")
    if len(genes) != len(vectors):
        raise ValueError("gene and vector counts differ")
    if vectors.size and not np.isfinite(vectors).all():
        raise ValueError("vectors contain non-finite values")


def save_npz(
    path: Path,
    vectors: Mapping[str, np.ndarray],
    model: str,
    *,
    uppercase_genes: bool = True,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {
        (str(gene).upper() if uppercase_genes else str(gene)): np.asarray(vector, dtype=np.float32)
        for gene, vector in vectors.items()
    }
    if not normalized:
        raise ValueError("cannot save an empty embedding set")
    genes = np.asarray(sorted(normalized), dtype=str)
    matrix = np.stack([normalized[gene] for gene in genes])
    _validate(genes, matrix)
    np.savez_compressed(path, genes=genes, vectors=matrix, model=np.asarray(model))
    return path


def load_npz(path: Path) -> EmbeddingSet:
    with np.load(path, allow_pickle=False) as data:
        genes = np.asarray(data["genes"], dtype=str)
        vectors = np.asarray(data["vectors"], dtype=np.float32)
        model = str(data["model"].item())
    _validate(genes, vectors)
    return EmbeddingSet(genes=genes, vectors=vectors, model=model)


def load_official_pickle(path: Path, *, trusted: bool = False) -> EmbeddingSet:
    """Load an official GenePT pickle only after explicit trust acknowledgement."""
    if not trusted:
        raise ValueError("pickle loading requires trusted=True")
    with path.open("rb") as handle:
        raw = pickle.load(handle)  # noqa: S301 - explicit trusted-source gate above
    if not isinstance(raw, dict) or not raw:
        raise ValueError("expected a non-empty gene-to-vector dictionary")
    normalized = {str(g).upper(): np.asarray(v, dtype=np.float32) for g, v in raw.items()}
    return EmbeddingSet(
        genes=np.asarray(sorted(normalized), dtype=str),
        vectors=np.stack([normalized[g] for g in sorted(normalized)]),
        model="official-genept",
    )


def l2_normalize(embedding_set: EmbeddingSet) -> EmbeddingSet:
    norms = np.linalg.norm(embedding_set.vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot normalize zero-length vectors")
    return EmbeddingSet(
        genes=embedding_set.genes,
        vectors=(embedding_set.vectors / norms).astype(np.float32),
        model=embedding_set.model,
    )


def coverage(embedding_set: EmbeddingSet, requested_genes: set[str]) -> dict[str, object]:
    available = {str(g).upper() for g in embedding_set.genes}
    requested = {str(g).upper() for g in requested_genes}
    found = requested & available
    return {
        "requested": len(requested),
        "found": len(found),
        "coverage": len(found) / len(requested) if requested else 1.0,
        "missing": sorted(requested - available),
    }
