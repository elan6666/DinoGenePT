"""Bind frozen downstream axes to the selected pretraining vocabulary."""

import argparse
import json

import pandas as pd

from dinogenept.datasets.cellfm.mapping import map_axis
from dinogenept.provenance import atomic_write_json, digest_file


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--cellfm", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads((args.selection / "selection.json").read_text())
    for name in ("genes.json", "gene_metadata.parquet"):
        if digest_file(args.selection / name) != selection["files_sha256"][name]:
            raise ValueError("Pretraining vocabulary metadata changed")
    vocabulary = json.loads((args.selection / "genes.json").read_text())
    metadata = pd.read_parquet(args.selection / "gene_metadata.parquet").to_dict(orient="records")
    for dataset in ("adamson", "norman"):
        directory = args.cellfm / dataset
        destination = directory / "gene_mapping.json"
        if destination.exists():
            raise FileExistsError("Preserve existing gene mapping")
        audit = json.loads((directory / "audit.json").read_text())
        for name in ("axis_symbols.json", "target_symbols.json"):
            if digest_file(directory / name) != audit["files_sha256"][name]:
                raise ValueError("Frozen downstream identity changed")
        result = map_axis(
            json.loads((directory / "axis_symbols.json").read_text()),
            json.loads((directory / "target_symbols.json").read_text()),
            vocabulary,
            metadata,
        )
        result.update(
            vocabulary_sha256=selection["files_sha256"]["genes.json"],
            gene_metadata_sha256=selection["files_sha256"]["gene_metadata.parquet"],
            downstream_audit_sha256=digest_file(directory / "audit.json"),
        )
        atomic_write_json(destination, result)
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "mapped_axis": len(result["symbols"]),
                    "mapped_targets": len(result["target_model_ids"]),
                    "sha256": digest_file(destination),
                }
            )
        )


if __name__ == "__main__":
    main()
