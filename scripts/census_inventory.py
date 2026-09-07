"""Server-only inventory: read public metadata, never expression or credentials."""

import argparse
import json
from pathlib import Path

import cellxgene_census


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="2023-12-15")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with cellxgene_census.open_soma(
        census_version=args.release,
        tiledb_config={"sm.compute_concurrency_level": 2, "sm.io_concurrency_level": 4},
    ) as census:
        table = census["census_info"]["datasets"].read().concat().to_pandas()
        experiment = census["census_data"]["homo_sapiens"]
        report = {
            "census_release": args.release,
            "expression_layer": "raw",
            "purpose": "metadata_inventory_not_approved_training_allowlist",
            "obs_columns": list(experiment.obs.schema.names),
            "var_columns": list(experiment.ms["RNA"].var.schema.names),
            "datasets": json.loads(table.to_json(orient="records")),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"datasets": len(report["datasets"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
