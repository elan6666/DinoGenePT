"""Exercise frozen real CellFM training/evaluation IO without running a model."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dinogenept.datasets.cellfm.dataset import CellFMDataset
from dinogenept.provenance import atomic_write_json, digest_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cellfm", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the previous data IO receipt")
    reports = []
    for name in ("adamson", "norman"):
        started = time.monotonic()
        directory = args.cellfm / name
        data = CellFMDataset(
            directory,
            args.vocabulary,
            audit_sha256=digest_file(directory / "audit.json"),
            mapping_sha256=digest_file(directory / "gene_mapping.json"),
        )
        bags = data.index.training_bags(epoch=0)
        primary = [i for bag in bags for i in bag.primary_post]
        expected = [i for i, c in enumerate(data.index.conditions) if c != "ctrl" and c in data.index.splits["train"]]
        if sorted(primary) != expected:
            raise ValueError("A full training epoch must visit every train post cell exactly once")
        # Materialize a real bag from every training condition/context; reject
        # any invalid populations before investing in model/API work.
        tested_groups = set()
        for bag in bags:
            key = (bag.condition, bag.context)
            if key not in tested_groups:
                inputs = data.training_inputs(bag, epoch=0)
                if not inputs["control_view"]["valid"].all():
                    raise ValueError("Reduced CellFM measured axes unexpectedly padded")
                tested_groups.add(key)
        evaluation = {}
        for split in ("val", "test"):
            groups = data.index.evaluation_groups(split)
            truth_cells = 0
            for group in groups:
                inputs = data.prediction_inputs(group["condition"], group["context"], group["control_rows"])
                truth = data.evaluation_truth(split, group["condition"], group["context"])
                if inputs["full_control"].shape != (300, len(data.axis)) or not np.isfinite(truth).all():
                    raise ValueError("Invalid evaluation population")
                truth_cells += len(truth)
            evaluation[split] = {"condition_context_groups": len(groups), "truth_cells": truth_cells}
        reports.append(
            {
                "dataset": name,
                "identity": data.identity,
                "training_bags": len(bags),
                "training_post_cells_once_per_epoch": len(primary),
                "train_groups_materialized": len(tested_groups),
                "evaluation": evaluation,
                "input_genes": len(data.axis),
                "true_zeros_eligible": True,
                "expression_renormalized": False,
                "DE_read_as_training_input": False,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        print(json.dumps(reports[-1]), flush=True)
    atomic_write_json(
        args.output,
        {
            "schema": "dinogenept.cellfm.training_io.v1",
            "status": "passed",
            "scope": "real_sparse_data_and_population_IO_not_model_training_or_performance",
            "datasets": reports,
        },
    )


if __name__ == "__main__":
    main()
