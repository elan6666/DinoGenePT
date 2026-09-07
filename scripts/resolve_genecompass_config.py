"""Freeze the user-approved all500k, one-epoch, no-validation configuration."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from dinogenept.cell.backbone import BackboneConfig
from dinogenept.cell.genecompass_data import GeneCompassDataset
from dinogenept.cell.pretraining import HeadConfig
from dinogenept.cell.sampling import CropConfig
from dinogenept.provenance import atomic_write_json, digest_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(Path("configs/cell/census500k_default_recipe.json").read_text())
    crops = CropConfig(**reference["crops"])
    data = GeneCompassDataset(args.manifest, crops)
    if len(data) != 500000 or data.gene_count != 23113:
        raise ValueError("Unexpected human corpus/vocabulary size")
    data.verify()
    backbone = {**reference["backbone"], "genes": data.gene_count}
    config = dict(
        purpose="formal_pretraining", protocol=data.manifest["protocol"],
        data_manifest=str(args.manifest.resolve()), data_manifest_sha256=digest_file(args.manifest),
        output=str(args.run_output.resolve()), backbone=asdict(BackboneConfig(**backbone)),
        heads=asdict(HeadConfig(**reference["heads"])), crops=asdict(crops),
        training={**reference["training"], "epochs": 1, "learning_rate": 5e-5,
                  "lr_scheduler": "sclong_epoch_restarts"},
        downstream_overlap="unknown", validation_policy="none_all_cells_train",
    )
    # New paths for smoke and formal runs; never replace an existing frozen config.
    if args.config.exists():
        raise FileExistsError(args.config)
    atomic_write_json(args.config, config)
    print(json.dumps(dict(status="resolved_not_launched", config=str(args.config),
                          manifest_sha256=config["data_manifest_sha256"])))


if __name__ == "__main__":
    main()
