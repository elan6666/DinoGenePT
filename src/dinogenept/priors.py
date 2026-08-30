"""Frozen Base and sparse source-only perturbation prior loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from genept_seed.provenance import digest_file
from genept_seed.vectors import load_npz


@dataclass(frozen=True)
class PriorTable:
    source: str
    genes: tuple[str, ...]
    vectors: np.ndarray
    model: str
    source_only: bool
    artifact_sha256: str

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])

    def mapping(self) -> dict[str, int]:
        return {gene: index for index, gene in enumerate(self.genes)}


def _fixture_vector(source: str, gene: str, dimension: int) -> np.ndarray:
    digest = hashlib.sha256(f"{source}\0{gene}".encode()).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    vector = np.random.default_rng(seed).standard_normal(dimension).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _fixture_table(
    source: str, genes: tuple[str, ...], config: dict[str, Any], *, source_only: bool
) -> PriorTable:
    dimension = int(config.get("dimension", 64))
    coverage = float(config.get("coverage", 1.0))
    if dimension <= 0 or not 0 < coverage <= 1:
        raise ValueError("fixture prior dimension and coverage are invalid")
    selected: list[str] = []
    vectors: list[np.ndarray] = []
    for gene in genes:
        gate = int.from_bytes(hashlib.sha256(f"coverage\0{source}\0{gene}".encode()).digest()[:8], "little")
        if source == "base" or gate / (2**64 - 1) < coverage:
            selected.append(gene)
            vectors.append(_fixture_vector(source, gene, dimension))
    artifact_identity = {
        "schema": "dinogenept-deterministic-prior-v1",
        "source": source,
        "dimension": dimension,
        "coverage": coverage,
        "source_only": source_only,
        "genes": selected,
    }
    artifact_sha256 = hashlib.sha256(
        json.dumps(artifact_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return PriorTable(
        source=source,
        genes=tuple(selected),
        vectors=(
            np.stack(vectors)
            if vectors
            else np.empty((0, dimension), dtype=np.float32)
        ),
        model="deterministic-smoke-fixture",
        source_only=source_only,
        artifact_sha256=artifact_sha256,
    )


def _load_table(source: str, config: dict[str, Any], genes: tuple[str, ...], *, source_only: bool) -> PriorTable:
    if config.get("fixture") == "deterministic":
        return _fixture_table(source, genes, config, source_only=source_only)
    path = Path(str(config["path"])).resolve(strict=True)
    loaded = load_npz(path)
    normalized = tuple(str(gene) for gene in loaded.genes)
    return PriorTable(
        source=source,
        genes=normalized,
        vectors=loaded.vectors,
        model=loaded.model,
        source_only=source_only,
        artifact_sha256=digest_file(path),
    )


class PriorStore:
    def __init__(self, tables: dict[str, PriorTable], required_targets: tuple[str, ...]) -> None:
        if "base" not in tables:
            raise ValueError("PriorStore requires a Base prior")
        self.tables = tables
        self._indices = {source: table.mapping() for source, table in tables.items()}
        missing = sorted(set(required_targets) - set(self._indices["base"]))
        if missing:
            raise ValueError(f"Base prior is missing perturbation targets: {missing[:20]}")

    @classmethod
    def from_config(cls, config: dict[str, Any], required_targets: tuple[str, ...]) -> PriorStore:
        prior_config = config["priors"]
        gene_universe = tuple(sorted(set(required_targets)))
        tables = {
            "base": _load_table(
                "base", prior_config["base"], gene_universe, source_only=False
            )
        }
        for source, source_config in sorted(prior_config.get("optional", {}).items()):
            if source_config.get("enabled", True):
                tables[source] = _load_table(
                    source,
                    source_config,
                    gene_universe,
                    source_only=bool(source_config.get("source_only", False)),
                )
        return cls(tables, gene_universe)

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(sorted(self.tables))

    @property
    def optional_sources(self) -> tuple[str, ...]:
        return tuple(source for source in self.sources if source != "base")

    @property
    def dimensions(self) -> dict[str, int]:
        return {source: table.dimension for source, table in self.tables.items()}

    @property
    def input_manifest(self) -> dict[str, dict[str, Any]]:
        return {
            source: {
                "artifact_sha256": table.artifact_sha256,
                "genes": len(table.genes),
                "dimension": table.dimension,
                "model": table.model,
                "source_only": table.source_only,
            }
            for source, table in sorted(self.tables.items())
        }

    def source_available(self, source: str, targets: tuple[str, ...]) -> bool:
        if source not in self._indices:
            raise ValueError(f"unknown prior source {source!r}; available={self.sources}")
        return bool(targets) and all(target in self._indices[source] for target in targets)

    def available_optional(self, targets: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            source for source in self.optional_sources if self.source_available(source, targets)
        )

    def batch(
        self,
        source: str,
        targets: tuple[tuple[str, ...], ...],
        *,
        control: str = "none",
        seed: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        table = self.tables[source]
        index = self._indices[source]
        width = max((len(item) for item in targets), default=1)
        vectors = np.zeros((len(targets), width, table.dimension), dtype=np.float32)
        token_mask = np.zeros((len(targets), width), dtype=bool)
        condition_mask = np.ones(len(targets), dtype=bool)
        permutation: dict[str, str] | None = None
        if control == "shuffled":
            genes = np.asarray(table.genes, dtype=str)
            shuffled = genes.copy()
            np.random.default_rng(seed).shuffle(shuffled)
            permutation = dict(zip(genes.tolist(), shuffled.tolist(), strict=True))
        constant = table.vectors.mean(axis=0) if control == "availability_only" else None
        for row, condition_targets in enumerate(targets):
            if not self.source_available(source, condition_targets):
                condition_mask[row] = False
                continue
            for column, gene in enumerate(condition_targets):
                selected = permutation[gene] if permutation is not None else gene
                vectors[row, column] = constant if constant is not None else table.vectors[index[selected]]
                token_mask[row, column] = True
        return vectors, token_mask, condition_mask

    def audit(self, conditions: dict[str, tuple[str, ...]]) -> dict[str, Any]:
        rows: dict[str, Any] = {}
        for source in self.sources:
            available = [
                condition
                for condition, targets in conditions.items()
                if self.source_available(source, targets)
            ]
            rows[source] = {
                "dimension": self.tables[source].dimension,
                "model": self.tables[source].model,
                "source_only": self.tables[source].source_only,
                "artifact_sha256": self.tables[source].artifact_sha256,
                "genes": len(self.tables[source].genes),
                "available_conditions": len(available),
                "total_conditions": len(conditions),
                "coverage": len(available) / len(conditions) if conditions else 1.0,
            }
        return rows
