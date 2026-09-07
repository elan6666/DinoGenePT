"""Atomic sparse array artifacts for efficient server-side random row access."""

import os
import tempfile
from pathlib import Path

import numpy as np

from dinogenept.provenance import digest_file


def validate_measured_coordinates(matrix, presence, row_sources, source_ids):
    """Support SciPy sparse matrices AND sparse arrays without 1-D COO slicing."""
    sources = np.asarray(row_sources, dtype=str)
    if matrix.shape[0] != len(sources) or matrix.shape[1] != presence.shape[1]:
        raise ValueError("Expression and measurement presence axes differ")
    positions = {source: i for i, source in enumerate(source_ids)}
    for source in np.unique(sources):
        if source not in positions:
            raise ValueError("Unknown source dataset for measurement audit")
        rows = np.flatnonzero(sources == source)
        # Integer indexing a modern csr_array yields a 1-D coo_array. Retain
        # both dimensions so the CSR index contract is explicit and stable.
        measured = presence[[positions[source]], :].tocsr().indices
        expressed = matrix[rows].tocsr().indices
        if np.setdiff1d(np.unique(expressed), measured).size:
            raise ValueError("Expression stored for a gene marked unmeasured")


def write_csr_shard(root: Path, name: str, matrix, row_ids, library_sum, *, split: str):
    """Write five read-only-loadable .npy arrays; manifest is published by caller.

    A new directory prevents accidental overwrite of a previous materialization.
    CSR counts stay sparse; .npy memory maps avoid decompressing an entire NPZ
    every time global shuffle jumps to another shard. No object arrays/pickle.
    """
    root = Path(root).resolve()
    directory = (root / name).resolve()
    if directory == root or not directory.is_relative_to(root):
        raise ValueError("Shard name must name a child of the corpus root")
    if split not in {"train", "validation"}:
        raise ValueError("Unknown pretraining split")
    matrix = matrix.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    row_ids, library_sum = np.asarray(row_ids), np.asarray(library_sum, dtype=np.float64)
    if matrix.shape[0] < 1 or row_ids.shape != (matrix.shape[0],) or library_sum.shape != row_ids.shape:
        raise ValueError("CSR row metadata does not match matrix")
    if row_ids.dtype.hasobject or np.unique(row_ids).size != row_ids.size:
        raise ValueError("Cell IDs must be unique and non-object")
    if np.any(np.diff(matrix.indptr) <= 0) or not np.isfinite(matrix.data).all() or np.any(matrix.data <= 0):
        raise ValueError("Nonempty raw positive count rows required")
    if not np.equal(matrix.data, np.floor(matrix.data)).all():
        raise ValueError("Fractional normalized values violate raw-count contract")
    sums = np.asarray(matrix.sum(axis=1, dtype=np.float64)).ravel()
    if not np.isfinite(library_sum).all() or np.any(sums > library_sum * (1 + 1e-5)):
        raise ValueError("Full-library totals smaller than stored counts")
    directory.mkdir(parents=True, exist_ok=False)
    values = {
        "data": matrix.data,
        "indices": matrix.indices,
        "indptr": matrix.indptr,
        "row_ids": row_ids,
        "library_sum": library_sum,
    }
    arrays = {}
    for key, value in values.items():
        target = directory / f"{key}.npy"
        descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=f".{key}.", suffix=".part")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.save(handle, value, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        arrays[key] = {
            "path": str(target.relative_to(root)),
            "sha256": digest_file(target),
            "bytes": target.stat().st_size,
        }
    return {"format": "csr_npy", "split": split, "cells": matrix.shape[0], "arrays": arrays}
