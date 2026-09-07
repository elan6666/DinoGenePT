"""Freeze evaluator-only DE/references/control draws before model comparison."""

import argparse
import json
from pathlib import Path

from dinogenept.datasets.cellfm.dataset import CellFMDataset
from dinogenept.evaluation.cellfm import load_evaluation, prepare_evaluation
from dinogenept.provenance import digest_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cellfm", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("adamson", "norman"):
        directory = args.cellfm / name
        data = CellFMDataset(
            directory,
            args.vocabulary,
            audit_sha256=digest_file(directory / "audit.json"),
            mapping_sha256=digest_file(directory / "gene_mapping.json"),
        )
        path = args.output / f"{name}.json"
        if path.exists():
            state = load_evaluation(data, path, sha256=digest_file(path))
            status = "existing_verified_not_regenerated"
        else:
            prepare_evaluation(data, path, dataset_id=f"cellfm_{name}")
            state = load_evaluation(data, path, sha256=digest_file(path))
            status = "created_and_verified"
        print(
            json.dumps(
                {
                    "dataset": name,
                    "status": status,
                    "path": str(path),
                    "sha256": digest_file(path),
                    "evaluation_conditions": len(state["conditions"]),
                    "scanpy_version": state["scanpy_version"],
                    "de_available": sum(row["de_unavailable_reason"] is None for row in state["conditions"].values()),
                    "de_gene_count_range": [
                        min(len(row["de_indices"]) for row in state["conditions"].values()),
                        max(len(row["de_indices"]) for row in state["conditions"].values()),
                    ],
                    "controls_per_condition": 300,
                    "with_replacement": True,
                    "seed": state["evaluation_seed"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
