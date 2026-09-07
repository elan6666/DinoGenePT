"""Inventory candidate cell metadata on the server; does not approve training."""

import argparse
import json
from pathlib import Path

import cellxgene_census
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--collection", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    inventory = json.loads(args.inventory.read_text())
    datasets = [x for x in inventory["datasets"] if x["collection_id"] in args.collection]
    if {x["collection_id"] for x in datasets} != set(args.collection):
        raise ValueError("Missing requested collection")
    # IDs come from the pinned public inventory, not untrusted filter snippets.
    dataset_ids = [x["dataset_id"] for x in datasets]
    if any(not all(c in "0123456789abcdef-" for c in x) for x in dataset_ids):
        raise ValueError("Invalid dataset UUID")
    filter_text = f"is_primary_data == True and dataset_id in {dataset_ids!r} and nnz >= 200 and raw_sum > 0"
    columns = [
        "soma_joinid", "dataset_id", "donor_id", "assay", "assay_ontology_term_id",
        "tissue_general", "disease", "suspension_type", "nnz", "raw_sum", "n_measured_vars",
    ]
    with cellxgene_census.open_soma(
        census_version=inventory["census_release"],
        tiledb_config={"sm.compute_concurrency_level": 2, "sm.io_concurrency_level": 4},
    ) as census:
        obs = census["census_data"]["homo_sapiens"].obs.read(
            value_filter=filter_text, column_names=columns
        ).concat().to_pandas()
    lookup = pd.DataFrame(datasets)[["dataset_id", "collection_id"]]
    obs = obs.merge(lookup, how="left", on="dataset_id", validate="many_to_one")
    if obs.soma_joinid.duplicated().any() or obs.collection_id.isna().any():
        raise ValueError("Invalid candidate identity mapping")
    report = {
        "status": "candidate_metadata_not_frozen_training_split",
        "census_release": inventory["census_release"],
        "filter": filter_text,
        "cells": len(obs),
        "collections": obs.collection_id.value_counts().to_dict(),
        "assays": obs.assay.value_counts().to_dict(),
        "diseases": obs.disease.value_counts().to_dict(),
        "tissues": obs.tissue_general.value_counts().to_dict(),
        "donors_per_collection": obs.groupby("collection_id").donor_id.nunique().to_dict(),
        "datasets": datasets,
    }
    args.output_dir.mkdir(parents=True)
    obs.to_parquet(args.output_dir / "candidate_obs.parquet", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in {"filter", "datasets"}}, indent=2))


if __name__ == "__main__":
    main()
