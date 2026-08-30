"""Shared leakage-aware loading for GEARS-style AnnData datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from genept_seed.provenance import digest_file

ConditionParser = Callable[[str], tuple[str, ...]]


@dataclass(frozen=True)
class PerturbationData:
    name: str
    expression: np.ndarray
    genes: tuple[str, ...]
    conditions: tuple[str, ...]
    targets: tuple[tuple[str, ...], ...]
    control_mask: np.ndarray
    split_by_condition: dict[str, str]
    top_de_indices: dict[str, tuple[int, ...]]
    ibot_gene_indices: tuple[int, ...]
    source_sha256: str
    split_sha256: str
    fingerprint: str

    @property
    def n_genes(self) -> int:
        return len(self.genes)

    @property
    def noncontrol_conditions(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.conditions) - {"ctrl"}))

    def indices_for_condition(self, condition: str) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.conditions, dtype=str) == condition)

    def split_conditions(self, split: str) -> tuple[str, ...]:
        return tuple(
            condition
            for condition in sorted(self.split_by_condition)
            if self.split_by_condition[condition] == split
        )


def plus_condition_parser(condition: str) -> tuple[str, ...]:
    if condition == "ctrl":
        return ()
    return tuple(part for part in condition.split("+") if part and part.lower() != "ctrl")


def _condition_split(
    conditions: tuple[str, ...], split_config: dict[str, Any]
) -> dict[str, str]:
    strategy = split_config.get("strategy")
    if strategy == "manifest":
        path = Path(str(split_config["path"])).resolve(strict=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        result: dict[str, str] = {}
        for split in ("train", "validation", "test"):
            values = payload.get(split)
            if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
                raise ValueError(f"split manifest {split!r} must be a list of conditions")
            for condition in values:
                if condition in result:
                    raise ValueError(f"condition appears in multiple splits: {condition}")
                result[condition] = split
        if set(result) != set(conditions):
            missing = sorted(set(conditions) - set(result))
            extra = sorted(set(result) - set(conditions))
            raise ValueError(f"split manifest condition mismatch: missing={missing}, extra={extra}")
        return result
    if strategy != "deterministic_condition":
        raise ValueError(f"unsupported split strategy: {strategy!r}")
    seed = int(split_config.get("seed", 1))
    values = np.asarray(sorted(conditions), dtype=str)
    np.random.default_rng(seed).shuffle(values)
    n = len(values)
    if n < 3:
        raise ValueError("condition split requires at least three non-control conditions")
    test_fraction = float(split_config.get("test_fraction", 0.25))
    validation_fraction = float(split_config.get("validation_fraction", 0.1))
    n_test = max(1, min(n - 2, int(round(n * test_fraction))))
    n_validation = max(1, min(n - n_test - 1, int(round(n * validation_fraction))))
    result = {condition: "test" for condition in values[:n_test]}
    result.update(
        {condition: "validation" for condition in values[n_test : n_test + n_validation]}
    )
    result.update({condition: "train" for condition in values[n_test + n_validation :]})
    return result


def _dense_rows(matrix: Any, rows: np.ndarray, columns: np.ndarray | slice) -> np.ndarray:
    # NumPy interprets ``matrix[row_array, column_array]`` as paired advanced
    # indexing.  Slice the two axes separately so a cell-by-gene matrix stays
    # rank two for dense arrays, scipy sparse matrices, and backed AnnData.
    selected = matrix[rows, :]
    selected = selected[:, columns]
    if hasattr(selected, "toarray"):
        selected = selected.toarray()
    return np.asarray(selected, dtype=np.float32)


def _column_variance(matrix: Any, rows: np.ndarray, *, chunk_rows: int) -> np.ndarray:
    """Compute train-only gene variances without densifying the full matrix."""
    if chunk_rows <= 0:
        raise ValueError("variance_chunk_rows must be positive")
    sums = np.zeros(matrix.shape[1], dtype=np.float64)
    squared_sums = np.zeros(matrix.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(rows), chunk_rows):
        selected = matrix[rows[start : start + chunk_rows], :]
        if hasattr(selected, "multiply"):
            sums += np.asarray(selected.sum(axis=0)).reshape(-1)
            squared_sums += np.asarray(selected.multiply(selected).sum(axis=0)).reshape(-1)
            count += selected.shape[0]
            continue
        dense = np.asarray(selected, dtype=np.float64)
        sums += dense.sum(axis=0)
        squared_sums += np.square(dense).sum(axis=0)
        count += dense.shape[0]
    if count == 0:
        raise ValueError("train-only variance selection received zero cells")
    mean = sums / count
    return np.maximum(squared_sums / count - np.square(mean), 0.0)


def _read_npz(path: Path, config: dict[str, Any]) -> tuple[np.ndarray, list[str], list[str], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        expression = np.asarray(payload["expression"], dtype=np.float32)
        genes = [str(item) for item in payload["genes"]]
        conditions = [str(item) for item in payload["conditions"]]
    return expression, genes, conditions, {}


def _read_h5ad(
    path: Path, config: dict[str, Any]
) -> tuple[Any, list[str], list[str], dict[str, Any]]:
    try:
        import anndata
    except ModuleNotFoundError as error:
        raise RuntimeError("AnnData loading requires the 'train' optional dependencies") from error
    adata = anndata.read_h5ad(path, backed="r")
    condition_key = str(config.get("condition_key", "condition"))
    gene_key = str(config.get("gene_key", "gene_name"))
    if condition_key not in adata.obs:
        raise ValueError(f"AnnData is missing obs[{condition_key!r}]")
    conditions = [str(item) for item in adata.obs[condition_key].tolist()]
    if gene_key in adata.var:
        genes = [str(item) for item in adata.var[gene_key].tolist()]
    else:
        genes = [str(item) for item in adata.var_names.tolist()]
    return adata, genes, conditions, dict(adata.uns)


def _extract_top_de(
    uns: dict[str, Any],
    key: str | None,
    genes: tuple[str, ...],
    original_columns: np.ndarray,
) -> dict[str, tuple[int, ...]]:
    if not key or key not in uns or not isinstance(uns[key], dict):
        return {}
    lookup = {gene: index for index, gene in enumerate(genes)}
    original_index_lookup = {
        int(original): selected for selected, original in enumerate(original_columns)
    }
    result: dict[str, tuple[int, ...]] = {}
    for condition, values in uns[key].items():
        indices: list[int] = []
        for value in values:
            if isinstance(value, (int, np.integer)) and int(value) in original_index_lookup:
                indices.append(original_index_lookup[int(value)])
            elif str(value) in lookup:
                indices.append(lookup[str(value)])
        if indices:
            result[str(condition)] = tuple(dict.fromkeys(indices[:20]))
    return result


def load_perturbation_dataset(
    config: dict[str, Any], *, name: str, parse_condition: ConditionParser
) -> PerturbationData:
    dataset = config["dataset"]
    path = Path(str(dataset["path"])).resolve(strict=True)
    if path.suffix == ".npz":
        matrix, genes, raw_conditions, uns = _read_npz(path, dataset)
        row_count = matrix.shape[0]
    else:
        matrix, genes, raw_conditions, uns = _read_h5ad(path, dataset)
        row_count = len(raw_conditions)
    if len(set(genes)) != len(genes):
        raise ValueError("dataset gene symbols must be unique")
    control_label = str(dataset.get("control_label", "ctrl"))
    normalized_conditions = ["ctrl" if item == control_label else item for item in raw_conditions]
    unique_conditions = tuple(sorted(set(normalized_conditions) - {"ctrl"}))
    split = _condition_split(unique_conditions, dataset["split"])
    limits = dataset.get("limits", {})
    max_conditions = int(limits.get("max_conditions", 0))
    if max_conditions > 0 and len(unique_conditions) > max_conditions:
        selected_conditions = set(unique_conditions[:max_conditions])
        unique_conditions = tuple(sorted(selected_conditions))
        split = _condition_split(unique_conditions, dataset["split"])
    rng = np.random.default_rng(int(dataset["split"].get("seed", 1)))
    max_per_condition = int(limits.get("max_cells_per_condition", 0))
    max_controls = int(limits.get("max_control_cells", 0))
    selected_rows: list[int] = []
    conditions_array = np.asarray(normalized_conditions, dtype=str)
    control_rows = np.flatnonzero(conditions_array == "ctrl")
    if max_controls > 0 and len(control_rows) > max_controls:
        control_rows = np.sort(rng.choice(control_rows, max_controls, replace=False))
    selected_rows.extend(int(item) for item in control_rows)
    for condition in unique_conditions:
        rows = np.flatnonzero(conditions_array == condition)
        if max_per_condition > 0 and len(rows) > max_per_condition:
            rows = np.sort(rng.choice(rows, max_per_condition, replace=False))
        selected_rows.extend(int(item) for item in rows)
    rows = np.asarray(sorted(set(selected_rows)), dtype=np.int64)
    if not len(rows) or int(rows.max()) >= row_count:
        raise ValueError("selected dataset rows are invalid")

    train_conditions = {condition for condition, value in split.items() if value == "train"}
    train_rows = np.asarray(
        [
            row
            for row in rows
            if normalized_conditions[row] in train_conditions
            or normalized_conditions[row] == "ctrl"
        ],
        dtype=np.int64,
    )
    max_genes = int(limits.get("max_genes", 0))
    columns = np.arange(len(genes), dtype=np.int64)
    variances: np.ndarray | None = None
    expression_matrix = matrix.X if hasattr(matrix, "X") else matrix
    variances = _column_variance(
        expression_matrix,
        train_rows,
        chunk_rows=int(limits.get("variance_chunk_rows", 1024)),
    )
    if max_genes > 0 and len(columns) > max_genes:
        if variances is None:
            raise RuntimeError("train-only variances were not computed")
        columns = np.sort(np.argpartition(variances, -max_genes)[-max_genes:])
    expression = _dense_rows(matrix.X if hasattr(matrix, "X") else matrix, rows, columns)
    if hasattr(matrix, "file") and hasattr(matrix.file, "close"):
        matrix.file.close()
    selected_genes = tuple(genes[int(index)] for index in columns)
    selected_conditions = tuple(normalized_conditions[int(index)] for index in rows)
    targets = tuple(parse_condition(condition) for condition in selected_conditions)
    control_mask = np.asarray([condition == "ctrl" for condition in selected_conditions], dtype=bool)
    if not control_mask.any():
        raise ValueError("dataset contains no selected control cells")
    if not (~control_mask).any():
        raise ValueError("dataset contains no selected perturbed cells")
    top_de = _extract_top_de(
        uns, dataset.get("top_de_key"), selected_genes, columns
    )
    count = min(int(config.get("model", {}).get("max_seq_len", len(columns))), len(columns))
    selected_variances = variances[columns]
    ranked = np.argsort(-selected_variances, kind="stable")[:count]
    ibot_gene_indices = tuple(int(index) for index in ranked)
    source_sha256 = digest_file(path)
    if dataset["split"].get("strategy") == "manifest":
        split_sha256 = digest_file(Path(str(dataset["split"]["path"])).resolve(strict=True))
    else:
        split_sha256 = hashlib.sha256(
            json.dumps(split, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    identity = {
        "name": name,
        "path": str(path),
        "shape": list(expression.shape),
        "genes": selected_genes,
        "conditions": sorted(set(selected_conditions)),
        "split": split,
        "source_sha256": source_sha256,
        "split_sha256": split_sha256,
        "ibot_gene_indices": ibot_gene_indices,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return PerturbationData(
        name=name,
        expression=expression,
        genes=selected_genes,
        conditions=selected_conditions,
        targets=targets,
        control_mask=control_mask,
        split_by_condition=split,
        top_de_indices=top_de,
        ibot_gene_indices=ibot_gene_indices,
        source_sha256=source_sha256,
        split_sha256=split_sha256,
        fingerprint=fingerprint,
    )
