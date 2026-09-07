"""Freeze published CellFM axes/splits without refitting HVGs or expression."""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np

from dinogenept.datasets.cellfm.split import simulation_split, targets
from dinogenept.provenance import atomic_write_json, digest_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.source / "range_receipt.json").read_text())
    if args.output.exists():
        raise FileExistsError("Preserve previously frozen dataset axes and splits")
    reports = []
    for name in ("adamson", "norman"):
        source = args.source / f"{name}.h5ad"
        sha = digest_file(source)
        if sha != receipt["files"][source.name]["sha256"]:
            raise ValueError("Source H5AD differs from the extracted artifact")
        data = ad.read_h5ad(source)
        matrix = data.X.tocsr()
        axis = data.var.gene_name.astype(str).tolist()
        if axis != data.var_names.tolist() or len(set(axis)) != len(axis) or not data.obs_names.is_unique:
            raise ValueError("Nonunique/mismatched official gene/cell identity")
        if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
            raise ValueError("Invalid official expression values")
        conditions = data.obs.condition.astype(str).unique().tolist()
        target_genes = sorted({g for condition in conditions for g in targets(condition)})
        if set(target_genes) - set(axis):
            raise ValueError("Official target absent from official gene axis")
        de = {str(key): [str(g) for g in values] for key, values in data.uns["rank_genes_groups_cov_all"].items()}
        if any(set(values) != set(axis) or len(values) != len(axis) for values in de.values()):
            raise ValueError("Official DE rankings do not exactly span the delivered axis")
        split = simulation_split(conditions, seed=3)
        output = args.output / name
        output.mkdir(parents=True)
        atomic_write_json(output / "axis_symbols.json", axis)
        atomic_write_json(output / "target_symbols.json", target_genes)
        atomic_write_json(output / "split.json", split)
        atomic_write_json(output / "de_evaluation_only.json", de)
        obs = data.obs[["condition", "cell_type", "condition_name"]].copy()
        obs.insert(0, "row_id", data.obs_names.astype(str))
        obs.insert(1, "source_row", np.arange(len(obs)))
        obs.to_parquet(output / "observations.parquet", index=False)
        report = {
            "dataset": name,
            "status": "source_axis_split_frozen_mapping_and_training_pending",
            "source": str(source.resolve()),
            "source_sha256": sha,
            "cells": data.n_obs,
            "genes": data.n_vars,
            "conditions_including_control": len(conditions),
            "target_genes": len(target_genes),
            "targets_missing_from_axis": [],
            "highly_variable_genes": int(data.var.highly_variable.sum()),
            "expression": "official_provided_log1p_matrix_no_renormalization",
            "scale_caveat": (
                "Original target library sum cannot be recovered from the reduced axis; do not infer or renormalize it"
            ),
            "hvg_caveat": "Published preselected axis retained; no claim of train-only upstream HVG fitting",
            "de_scope": "evaluation_only_never_model_or_training_input",
            "contexts": data.obs.cell_type.astype(str).value_counts().to_dict(),
            "split_conditions_including_control": {key: len(values) for key, values in split["conditions"].items()},
            "split_cells": {
                key: int(data.obs.condition.isin(values).sum()) for key, values in split["conditions"].items()
            },
            "source_protocol": "CellFM_released_reduced_axis_simulation_seed3",
            "notebook_identity": (
                "norman-1000 name is consistent with reduced axis, but filename identity is not proven"
            ),
            "files_sha256": {p.name: digest_file(p) for p in output.iterdir() if p.is_file()},
        }
        atomic_write_json(output / "audit.json", report)
        reports.append(report)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
