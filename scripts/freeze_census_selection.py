"""Freeze audited-source, donor-heldout candidate rows before raw extraction."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cellxgene_census
import numpy as np
import pandas as pd

from dinogenept.provenance import atomic_write_json, digest_file

COLLECTIONS = {
    "e5f58829-1a66-40b5-a624-9046778e74f5": "10.1126/science.abl4896",
    "62ef75e4-cbea-454e-a0ce-998ec40223d3": "10.1126/science.abl5197",
}
ASSAYS = {"EFO:0009922", "EFO:0030004", "EFO:0011025", "EFO:0009900"}


def score(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def subsample(frame, count, seed):
    """Proportional largest-remainder quotas, without replacing any cell."""
    if len(frame) < count:
        raise ValueError(f"Only {len(frame)} eligible cells for requested {count}")
    grouped = list(
        frame.groupby(
            ["collection_id", "donor_id", "tissue_general", "assay_ontology_term_id"], observed=True, sort=True
        )
    )
    exact = np.array([len(group) * count / len(frame) for _, group in grouped])
    quotas = np.floor(exact).astype(int)
    remainder = count - quotas.sum()
    priorities = sorted(range(len(grouped)), key=lambda i: (-(exact[i] - quotas[i]), score(seed, grouped[i][0])))
    quotas[priorities[:remainder]] += 1
    parts = []
    for i, (key, group) in enumerate(grouped):
        if not quotas[i]:
            continue
        group = group.sort_values("soma_joinid")
        rng = np.random.default_rng(int(score(seed, key)[:16], 16))
        parts.append(group.iloc[rng.choice(len(group), quotas[i], replace=False)])
    return pd.concat(parts).sort_values("soma_joinid").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-cells", type=int, default=500000)
    parser.add_argument("--validation-cells", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Selection already exists; inspect its receipt rather than overwrite")
    summary_path = args.candidate_dir / "summary.json"
    obs_path = args.candidate_dir / "candidate_obs.parquet"
    summary = json.loads(summary_path.read_text())
    obs = pd.read_parquet(obs_path)
    if summary["census_release"] != "2023-12-15" or set(obs.collection_id) != set(COLLECTIONS):
        raise ValueError("Unexpected source/release")
    for source in summary["datasets"]:
        if source["collection_doi"] != COLLECTIONS[source["collection_id"]]:
            raise ValueError("Source publication differs from the audited allowlist")
    eligible = obs.loc[obs.assay_ontology_term_id.isin(ASSAYS)].copy()
    if not eligible.is_primary_data.all() or not (eligible.disease == "normal").all():
        raise ValueError("Allowlist metadata must contain primary normal human cells")
    if eligible.soma_joinid.duplicated().any() or (eligible.nnz < 200).any() or (eligible.raw_sum <= 0).any():
        raise ValueError("Duplicate or invalid-QC candidate row")
    if eligible.donor_id.isna().any() or eligible.donor_id.astype(str).str.lower().isin({"unknown", "na", ""}).any():
        raise ValueError("Missing donor metadata prevents heldout-donor validation")
    eligible["donor_key"] = eligible.collection_id.astype(str) + ":" + eligible.donor_id.astype(str)
    heldout = []
    for _collection, frame in eligible.groupby("collection_id", observed=True):
        donors = sorted(frame.donor_key.unique(), key=lambda x: score(args.seed, x))
        heldout.extend(donors[: math.ceil(0.1 * len(donors))])
    is_validation = eligible.donor_key.isin(heldout)
    train = subsample(eligible.loc[~is_validation], args.training_cells, args.seed)
    validation = subsample(eligible.loc[is_validation], args.validation_cells, args.seed + 1)
    if set(train.donor_key) & set(validation.donor_key) or set(train.soma_joinid) & set(validation.soma_joinid):
        raise ValueError("Training/validation identity overlap")
    if train.collection_id.value_counts(normalize=True).max() > 0.65:
        raise ValueError("Selected training set violates the 65% per-study cap")
    print(
        json.dumps(
            {
                "status": "selected_rows_fetching_gene_metadata",
                "train": len(train),
                "validation": len(validation),
                "heldout_donors": heldout,
            }
        ),
        flush=True,
    )
    with cellxgene_census.open_soma(
        census_version="2023-12-15", tiledb_config={"sm.compute_concurrency_level": 2, "sm.io_concurrency_level": 4}
    ) as census:
        var = census["census_data"]["homo_sapiens"].ms["RNA"].var.read().concat().to_pandas().sort_values("soma_joinid")
    if var.feature_id.duplicated().any() or var.soma_joinid.duplicated().any():
        raise ValueError("Census gene identifiers are not unique")
    args.output.mkdir(parents=True)
    train.to_parquet(args.output / "train_obs.parquet", index=False)
    validation.to_parquet(args.output / "validation_obs.parquet", index=False)
    var.to_parquet(args.output / "gene_metadata.parquet", index=False)
    atomic_write_json(args.output / "genes.json", var.feature_id.tolist())
    files = {
        name: digest_file(args.output / name)
        for name in ("train_obs.parquet", "validation_obs.parquet", "gene_metadata.parquet", "genes.json")
    }
    report = {
        "status": "selection_frozen_raw_expression_pending",
        "census_release": "2023-12-15",
        "seed": args.seed,
        "candidate_metadata_sha256": digest_file(obs_path),
        "candidate_summary_sha256": digest_file(summary_path),
        "eligible_droplet_cells": len(eligible),
        "training_cells": len(train),
        "validation_cells": len(validation),
        "genes": len(var),
        "heldout_donors": heldout,
        "donor_overlap": 0,
        "cell_overlap": 0,
        "training_collections": train.collection_id.value_counts().to_dict(),
        "training_tissues": train.tissue_general.value_counts().to_dict(),
        "files_sha256": files,
        "source_audit": {
            "scope": "published_study_and_population_metadata",
            "allowlist_publications": COLLECTIONS,
            "excluded_assays": ["Smart-seq2"],
            "rationale": (
                "Normal donor tissue atlases are distinct studies/populations from CellFM K562 CRISPR experiments"
            ),
            "remaining": ["raw-count validity", "measurement metadata", "downstream gene mapping", "GPU capacity"],
        },
        "expression": "raw_counts_then_log1p_10000_full_library_no_bins",
        "sampling": "10_percent_donors_per_collection_held_out_then_proportional_stratified_cells_without_replacement",
    }
    atomic_write_json(args.output / "selection.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
