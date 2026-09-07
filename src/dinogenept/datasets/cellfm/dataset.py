"""Frozen CellFM expression and population access for native LoRA training.

Read only X/obs/var, never upstream uns/DE annotations. Keep X sparse in host
memory; densify only requested bags. Predictions and training receive separate
interfaces so evaluation truth cannot accidentally be forwarded to the model.
"""

import json
from pathlib import Path

import numpy as np

from dinogenept.datasets.populations import PopulationIndex, continuous_view
from dinogenept.provenance import digest_file

from .split import targets


class CellFMDataset:
    def __init__(self, directory, vocabulary, *, audit_sha256, mapping_sha256, source=None):
        import h5py
        import pandas as pd
        from anndata.io import read_elem
        from scipy import sparse

        directory, vocabulary = Path(directory), Path(vocabulary)
        audit_path, mapping_path = directory / "audit.json", directory / "gene_mapping.json"
        if digest_file(audit_path) != audit_sha256 or digest_file(mapping_path) != mapping_sha256:
            raise ValueError("Pinned downstream audit/mapping changed")
        audit, mapping = (json.loads(path.read_text()) for path in (audit_path, mapping_path))
        if mapping["downstream_audit_sha256"] != audit_sha256:
            raise ValueError("Mapping belongs to another downstream audit")
        if digest_file(vocabulary) != mapping["vocabulary_sha256"]:
            raise ValueError("Pretraining vocabulary differs from downstream mapping")
        for name in ("axis_symbols.json", "target_symbols.json", "split.json", "observations.parquet"):
            if digest_file(directory / name) != audit["files_sha256"][name]:
                raise ValueError(f"Frozen downstream file changed: {name}")
        symbols = json.loads((directory / "axis_symbols.json").read_text())
        target_symbols = json.loads((directory / "target_symbols.json").read_text())
        genes = json.loads(vocabulary.read_text())
        axis = np.asarray(mapping["model_gene_ids"])
        if (
            mapping["symbols"] != symbols
            or axis.shape != (len(symbols),)
            or not np.issubdtype(axis.dtype, np.integer)
            or not len(axis)
            or len(set(symbols)) != len(axis)
            or len(set(genes)) != len(genes)
            or len(np.unique(axis)) != len(axis)
            or np.any(axis < 1)
            or np.any(axis > len(genes))
            or mapping["vocabulary_size"] != len(genes)
        ):
            raise ValueError("Invalid frozen gene axes")
        if [genes[int(i) - 1] for i in axis] != mapping["ensembl_ids"]:
            raise ValueError("Native IDs do not match frozen Ensembl identity")
        lookup = dict(zip(symbols, axis.tolist(), strict=True))
        if set(target_symbols) - set(lookup) or mapping["target_model_ids"] != {
            gene: lookup[gene] for gene in target_symbols
        }:
            raise ValueError("Target mapping differs from the prediction axis")
        source = Path(audit["source"] if source is None else source)
        if digest_file(source) != audit["source_sha256"]:
            raise ValueError("Official H5AD changed")
        # Public AnnData reader handles encoded CSR/dataframes, but does not read
        # uns (which contains evaluation-only DE) or create a whole dense matrix.
        with h5py.File(source, "r") as handle:
            matrix, obs, var = (read_elem(handle[key]) for key in ("X", "obs", "var"))
        frozen_obs = pd.read_parquet(directory / "observations.parquet")
        if not sparse.issparse(matrix) or matrix.shape != (audit["cells"], audit["genes"]):
            raise ValueError("Official expression must be sparse on the frozen axes")
        matrix = sparse.csr_matrix(matrix, dtype=np.float32)
        if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
            raise ValueError("Invalid continuous expression")
        if var.index.astype(str).tolist() != symbols or var.gene_name.astype(str).tolist() != symbols:
            raise ValueError("Official expression gene order differs")
        if obs.index.astype(str).tolist() != frozen_obs.row_id.astype(str).tolist():
            raise ValueError("Official expression cell order differs")
        if not np.array_equal(frozen_obs.source_row.to_numpy(), np.arange(len(obs))):
            raise ValueError("Frozen observations are not in source row order")
        for field in ("condition", "cell_type", "condition_name"):
            if obs[field].astype(str).tolist() != frozen_obs[field].astype(str).tolist():
                raise ValueError(f"Frozen observation metadata differs: {field}")
        split = json.loads((directory / "split.json").read_text())
        self.split_conditions = split["conditions"]
        self.index = PopulationIndex(frozen_obs.row_id, frozen_obs.condition, frozen_obs.cell_type, split["conditions"])
        actual_targets = {gene for condition in self.index.conditions for gene in targets(condition)}
        if actual_targets != set(target_symbols):
            raise ValueError("Perturbation conditions disagree with target manifest")
        if audit["expression"] != "official_provided_log1p_matrix_no_renormalization":
            raise ValueError("Unsupported expression preprocessing contract")
        self.axis, self.symbols, self.mapping = axis.astype(np.int64), tuple(symbols), lookup
        self._matrix = matrix
        self.identity = {
            "audit_sha256": audit_sha256,
            "mapping_sha256": mapping_sha256,
            "vocabulary_sha256": mapping["vocabulary_sha256"],
            "source_sha256": audit["source_sha256"],
            "split_sha256": audit["files_sha256"]["split.json"],
            "cells": matrix.shape[0],
            "genes": matrix.shape[1],
            "expression": audit["expression"],
        }

    def _rows(self, rows, *, condition, context):
        rows = np.asarray(rows)
        if (
            rows.ndim != 1
            or not len(rows)
            or not np.issubdtype(rows.dtype, np.integer)
            or np.any(rows < 0)
            or np.any(rows >= len(self.index.row_ids))
        ):
            raise ValueError("Invalid population row indices")
        if not np.all(self.index.conditions[rows] == condition) or not np.all(self.index.contexts[rows] == context):
            raise ValueError("Rows do not belong to the requested condition/context")
        return rows

    def _expression(self, rows):
        # Advanced CSR row indexing preserves ordering and repeated controls.
        return self._matrix[rows].toarray()

    def _view(self, rows, *, seed, epoch, role, cap, masked=False):
        values = self._expression(rows)
        return continuous_view(
            self.axis,
            values,
            self.index.row_ids[rows],
            seed=seed,
            epoch=epoch,
            role=role,
            cap=cap,
            mask_fraction=0.2 if masked else 0.0,
        ), values

    def condition_targets(self, condition):
        symbols = tuple(sorted(targets(condition), key=self.mapping.__getitem__))
        if not symbols:
            raise ValueError("A perturbation condition is required")
        return symbols, np.asarray([self.mapping[gene] for gene in symbols], dtype=np.int64)

    def training_inputs(self, bag, *, epoch, seed=42, cap=2048):
        if bag.condition == "ctrl" or bag.condition not in self.index.splits["train"]:
            raise ValueError("Training post-treatment bags must be in train")
        primary = self._rows(bag.primary_post, condition=bag.condition, context=bag.context)
        observed = self._rows(bag.observed_post, condition=bag.condition, context=bag.context)
        teacher = self._rows(bag.teacher_post, condition=bag.condition, context=bag.context)
        controls = self._rows(bag.controls, condition="ctrl", context=bag.context)
        kwargs = {"seed": seed, "epoch": epoch, "cap": cap}
        view, full_control = self._view(controls, role=f"control:{bag.key}", **kwargs)
        observed_view, _ = self._view(observed, role=f"observed:{bag.key}", masked=True, **kwargs)
        teacher_view, _ = self._view(teacher, role=f"teacher:{bag.key}", **kwargs)
        _, model_targets = self.condition_targets(bag.condition)
        return {
            "control_view": view,
            "observed_view": observed_view,
            "teacher_view": teacher_view,
            "full_control": full_control,
            "train_post": self._expression(primary),
            "axis": self.axis.copy(),
            "targets": model_targets,
        }

    def prediction_inputs(self, condition, context, control_rows, *, seed=42, cap=2048):
        # Deliberately no truth rows or post-expression argument in this API.
        rows = self._rows(control_rows, condition="ctrl", context=context)
        view, full_control = self._view(rows, seed=seed, epoch=0, role="inference", cap=cap)
        _, model_targets = self.condition_targets(condition)
        return {"control_view": view, "full_control": full_control, "axis": self.axis.copy(), "targets": model_targets}

    def evaluation_truth(self, split, condition, context):
        """Called only by the evaluator after prediction, never the training API."""
        if split not in {"val", "test"} or condition not in self.index.splits[split]:
            raise ValueError("Evaluation truth must belong to the requested held-out split")
        return self._expression(self.index.groups[(condition, context)])
