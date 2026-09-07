"""Resume pinned raw-count CSR shards; mark ready only after the complete audit."""

import argparse
import fcntl
import json
import shutil
from pathlib import Path

import cellxgene_census
import numpy as np
import pandas as pd
from scipy import sparse

from dinogenept.cell.dataset import PretrainingDataset
from dinogenept.cell.sampling import CropConfig
from dinogenept.cell.sparse import validate_measured_coordinates, write_csr_shard
from dinogenept.provenance import atomic_write_json, digest_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-cells", type=int, default=2048)
    args = parser.parse_args()
    selection_path = args.selection / "selection.json"
    selection = json.loads(selection_path.read_text())
    if selection["status"] != "selection_frozen_raw_expression_pending" or args.shard_cells < 2:
        raise ValueError("Unfrozen selection or invalid shard size")
    for name, expected in selection["files_sha256"].items():
        if digest_file(args.selection / name) != expected:
            raise ValueError("Selection artifact changed")
    identity = {
        "selection_sha256": digest_file(selection_path),
        "inventory_sha256": digest_file(args.inventory),
        "shard_cells": args.shard_cells,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    guard = (args.output / ".materialization.lock").open("a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = args.output / "materialization.json"
    if state.exists():
        if json.loads(state.read_text())["identity"] != identity:
            raise ValueError("Cannot resume a different materialization in this directory")
    else:
        atomic_write_json(state, {"identity": identity, "status": "running"})
        for name in (*selection["files_sha256"], "selection.json"):
            shutil.copyfile(args.selection / name, args.output / name)
    var = pd.read_parquet(args.selection / "gene_metadata.parquet").sort_values("soma_joinid")
    frames = {split: pd.read_parquet(args.selection / f"{split}_obs.parquet") for split in ("train", "validation")}
    inventory = json.loads(args.inventory.read_text())
    if inventory["census_release"] != selection["census_release"]:
        raise ValueError("Census source release mismatch")
    dataset_map = {row["dataset_id"]: row["soma_joinid"] for row in inventory["datasets"]}
    source_ids = sorted(set.union(*(set(frame.dataset_id) for frame in frames.values())))
    dataset_coords = np.array([dataset_map[x] for x in source_ids])
    presence_file = args.output / "measurement_presence.npz"
    presence_receipt = args.output / "measurement_presence.json"
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        PretrainingDataset(manifest_path, "train", CropConfig()).verify()
        print(json.dumps({"status": "already_verified", "manifest": str(manifest_path)}), flush=True)
        return
    with cellxgene_census.open_soma(
        census_version=selection["census_release"],
        tiledb_config={"sm.compute_concurrency_level": 2, "sm.io_concurrency_level": 4},
    ) as census:
        if presence_receipt.exists():
            record = json.loads(presence_receipt.read_text())
            if digest_file(presence_file) != record["sha256"] or record["datasets"] != source_ids:
                raise ValueError("Measurement presence cache changed")
            presence = sparse.load_npz(presence_file).tocsr()
        else:
            print(json.dumps({"status": "fetching_measurement_presence", "datasets": len(source_ids)}), flush=True)
            presence = cellxgene_census.get_presence_matrix(census, "Homo sapiens")[dataset_coords][
                :, var.soma_joinid
            ].tocsr()
            sparse.save_npz(presence_file, presence)
            atomic_write_json(
                presence_receipt,
                {
                    "datasets": source_ids,
                    "sha256": digest_file(presence_file),
                    "gene_metadata_sha256": selection["files_sha256"]["gene_metadata.parquet"],
                },
            )
        measured_counts = np.asarray(presence.sum(axis=1)).ravel()
        source_pos = {source: i for i, source in enumerate(source_ids)}
        for frame in frames.values():
            actual_measured = np.array([measured_counts[source_pos[x]] for x in frame.dataset_id])
            if not np.array_equal(actual_measured, frame.n_measured_vars.to_numpy()):
                raise ValueError("Dataset presence and per-cell measured-gene metadata disagree")
        shards = []
        for split, frame in frames.items():
            for chunk, start in enumerate(range(0, len(frame), args.shard_cells)):
                selected = frame.iloc[start : start + args.shard_cells].sort_values("soma_joinid")
                name = f"{split}-{chunk:04d}"
                receipt_path = args.output / f"{name}.json"
                if receipt_path.exists():
                    record = json.loads(receipt_path.read_text())
                    if record["row_ids"] != selected.soma_joinid.tolist():
                        raise ValueError("Shard row selection differs")
                    for entry in record["shard"]["arrays"].values():
                        if digest_file(args.output / entry["path"]) != entry["sha256"]:
                            raise ValueError("Previously materialized shard changed")
                    shards.append(record["shard"])
                    continue
                if (args.output / name).exists():
                    raise FileExistsError(f"Unreceipted shard needs inspection: {name}")
                print(
                    json.dumps(
                        {
                            "status": "fetching_raw",
                            "split": split,
                            "shard": chunk,
                            "cells": len(selected),
                            "completed_cells": sum(x["cells"] for x in shards),
                        }
                    ),
                    flush=True,
                )
                data = cellxgene_census.get_anndata(
                    census,
                    "Homo sapiens",
                    X_name="raw",
                    obs_coords=selected.soma_joinid.to_numpy(),
                    obs_column_names=["soma_joinid", "dataset_id", "raw_sum"],
                    var_column_names=["soma_joinid", "feature_id"],
                )
                if (
                    data.obs.soma_joinid.tolist() != selected.soma_joinid.tolist()
                    or data.var.feature_id.tolist() != var.feature_id.tolist()
                ):
                    raise ValueError("Census returned unexpected cell/gene ordering")
                if data.obs.dataset_id.tolist() != selected.dataset_id.tolist():
                    raise ValueError("Census source dataset differs from frozen selection")
                matrix = data.X.tocsr()
                sums = np.asarray(matrix.sum(axis=1, dtype=np.float64)).ravel()
                if not np.allclose(sums, selected.raw_sum.to_numpy(), rtol=1e-6, atol=1e-3):
                    raise ValueError("Raw expression library sum differs from metadata")
                validate_measured_coordinates(matrix, presence, selected.dataset_id.to_numpy(), source_ids)
                shard = write_csr_shard(args.output, name, matrix, selected.soma_joinid.to_numpy(), sums, split=split)
                atomic_write_json(receipt_path, {"row_ids": selected.soma_joinid.tolist(), "shard": shard})
                shards.append(shard)
                print(
                    json.dumps(
                        {"status": "shard_verified", "name": name, "completed_cells": sum(x["cells"] for x in shards)}
                    ),
                    flush=True,
                )
                del data, matrix
    if sum(x["cells"] for x in shards) != selection["training_cells"] + selection["validation_cells"]:
        raise RuntimeError("Incomplete raw corpus")
    manifest = {
        "schema": "dinogenept.pretraining.csr.v1",
        "status": "training_ready",
        "purpose": "formal_pretraining",
        "data_id": "census-two-atlas-2023-12-15-500k-v1",
        "expression": "raw_counts",
        "vocabulary": {"path": "genes.json", "sha256": digest_file(args.output / "genes.json"), "genes": len(var)},
        "shards": shards,
        "leakage_audit": {
            "status": "passed",
            "scope": "published_source_and_donor_cell_metadata",
            "selection_sha256": identity["selection_sha256"],
            "donor_overlap": 0,
            "cell_overlap": 0,
            "limitations": ["normal-to-K562 domain shift", "published downstream HVG-axis provenance inherited"],
        },
        "measurement_presence": {"path": presence_file.name, "sha256": digest_file(presence_file)},
        "materialization_identity": identity,
    }
    pending = args.output / "manifest.pending.json"
    atomic_write_json(pending, manifest)
    PretrainingDataset(pending, "train", CropConfig()).verify()
    pending.replace(manifest_path)
    atomic_write_json(
        state, {"identity": identity, "status": "completed", "manifest_sha256": digest_file(manifest_path)}
    )
    print(json.dumps({"status": "completed", "manifest": str(manifest_path)}), flush=True)


if __name__ == "__main__":
    main()
