"""Freeze a formal recipe only after full corpus and matching CUDA gates pass."""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from dinogenept.cell.backbone import BackboneConfig
from dinogenept.cell.dataset import PretrainingDataset
from dinogenept.cell.pretraining import HeadConfig
from dinogenept.cell.sampling import CropConfig
from dinogenept.provenance import atomic_write_json, digest_file


def validate_capacity(recipe, capacity, numerical):
    """A single-card or small-backbone probe cannot authorize the full DDP run."""
    identity = capacity.get("identity", {})
    if (
        capacity.get("status") != "capacity_passed"
        or identity.get("purpose") != "capacity_probe_not_formal_training"
        or identity.get("world_size") != recipe["training"]["world_size"]
        or identity.get("microbatch") != recipe["training"]["microbatch"]
        or identity.get("backbone") != recipe["backbone"]
        or identity.get("heads") != recipe["heads"]
        or identity.get("vocabulary_sha256") != recipe["expected_vocabulary_sha256"]
        or len(capacity.get("steps", [])) < 2
    ):
        raise ValueError("Matching full-default multi-rank CUDA capacity evidence is required")
    cap = recipe["crops"]["cap"]
    local = math.ceil(cap * recipe["crops"]["local_scale"][1])
    expected_lengths = [cap] * 2 + [local] * recipe["crops"]["local_count"]
    if identity.get("view_lengths") != expected_lengths:
        raise ValueError("Capacity probe must cover the full declared crop bounds")
    if (
        numerical.get("status") != "passed"
        or numerical.get("candidate_kernel") != recipe["backbone"]["kda_implementation"]
    ):
        raise ValueError("Selected execution backend has not passed numerical acceptance")
    if numerical.get("comparison_errors"):
        raise ValueError("Numerical failures cannot be overridden by a status label")
    if numerical.get("backbone_tokens", 0) < cap:
        raise ValueError("Whole-backbone numerical audit must cover the full token cap")
    observed = {**numerical.get("backbone", {}), "kda_implementation": numerical.get("candidate_kernel")}
    if observed != recipe["backbone"]:
        raise ValueError("Numerical audit must cover the full requested backbone/vocabulary")
    expected_kernel_shape = [1, cap + 1, recipe["backbone"]["kda_heads"], recipe["backbone"]["kda_head_dim"]]
    if (
        numerical.get("long_kernel_shape") != expected_kernel_shape
        or len(numerical.get("kernel_values_and_gradients", [])) != 7
    ):
        raise ValueError("Numerical receipt lacks long-sequence value and gradient evidence")
    for precision, limit in (("float32", 2e-4), ("bfloat16", 0.02)):
        for field in ("cls", "genes"):
            measured = numerical.get("backbone_comparison", {}).get(precision, {}).get(field, {})
            error, absolute = measured.get("relative_l2", math.inf), measured.get("max_absolute", math.inf)
            if not math.isfinite(error) or not math.isfinite(absolute) or error < 0 or error > limit or absolute < 0:
                raise ValueError("Missing or unacceptable whole-backbone numerical measurements")
    for step in capacity["steps"]:
        values = [
            step.get("seconds", math.nan),
            step.get("gradient_norm_rank0", math.nan),
            *step.get("losses_rank0", {}).values(),
        ]
        expected_losses = {"expression", "cell_expression", "dino", "ibot", "koleo", "total"}
        if not all(math.isfinite(value) for value in values) or set(step.get("losses_rank0", {})) != expected_losses:
            raise ValueError("Capacity steps need all five finite losses, total and finite gradients")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--numerical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    if recipe.get("schema") != "dinogenept.pretraining.recipe.v1" or recipe["training"]["epochs"] != 2:
        raise ValueError("Not the full two-epoch campaign recipe")
    capacity, numerical = [json.loads(path.read_text()) for path in (args.capacity, args.numerical)]
    validate_capacity(recipe, capacity, numerical)
    for receipt, required in (
        (capacity["identity"], ("backbone", "kda", "pretraining", "distillation")),
        (numerical, ("backbone", "kda")),
    ):
        for module in required:
            name = f"src/dinogenept/cell/{module}.py"
            if receipt.get("source_sha256", {}).get(name) != digest_file(name):
                raise ValueError("Audited model source differs from this checkout; revalidate before launch")
    manifest = Path(recipe["data_manifest"]).resolve()
    crops = CropConfig(**recipe["crops"])
    train = PretrainingDataset(manifest, "train", crops)
    validation = PretrainingDataset(manifest, "validation", crops)
    if (
        train.manifest.get("purpose") != "formal_pretraining"
        or train.manifest["data_id"] != recipe["expected_data_id"]
        or len(train) != recipe["expected_training_cells"]
        or len(validation) != recipe["expected_validation_cells"]
        or train.gene_count != recipe["expected_vocabulary_genes"]
        or train.manifest["vocabulary"]["sha256"] != recipe["expected_vocabulary_sha256"]
    ):
        raise ValueError("Full frozen 500k/20k corpus identity is not satisfied")
    train.verify()
    config = {
        "purpose": "formal_pretraining",
        "data_manifest": str(manifest),
        "data_manifest_sha256": digest_file(manifest),
        "output": str(Path(recipe["output"]).resolve()),
        "backbone": asdict(BackboneConfig(**recipe["backbone"])),
        "heads": asdict(HeadConfig(**recipe["heads"])),
        "crops": asdict(crops),
        "training": recipe["training"],
        "readiness_evidence": {
            name: {"path": str(path.resolve()), "sha256": digest_file(path)}
            for name, path in (("recipe", args.recipe), ("capacity", args.capacity), ("numerical", args.numerical))
        },
    }
    config = json.loads(json.dumps(config))
    if args.output.exists():
        if json.loads(args.output.read_text()) != config:
            raise FileExistsError("Existing resolved config differs; do not overwrite a frozen run")
    else:
        atomic_write_json(args.output, config)
    print(json.dumps({"status": "resolved_not_launched", "path": str(args.output), "sha256": digest_file(args.output)}))


if __name__ == "__main__":
    main()
