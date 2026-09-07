import copy
import json
import runpy
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
validate_capacity = runpy.run_path(str(ROOT / "scripts/resolve_pretraining_config.py"))["validate_capacity"]


def evidence():
    recipe = json.loads((ROOT / "configs/cell/census500k_default_recipe.json").read_text())
    capacity = {
        "status": "capacity_passed",
        "identity": {
            "purpose": "capacity_probe_not_formal_training",
            "world_size": 2,
            "microbatch": 4,
            "backbone": copy.deepcopy(recipe["backbone"]),
            "heads": recipe["heads"],
            "vocabulary_sha256": recipe["expected_vocabulary_sha256"],
            "view_lengths": [2048, 2048, 820, 820],
        },
        "steps": [
            {
                "seconds": 1.0,
                "gradient_norm_rank0": 1.0,
                "losses_rank0": {
                    name: 1.0 for name in ("expression", "cell_expression", "dino", "ibot", "koleo", "total")
                },
            }
        ]
        * 2,
    }
    numerical = {
        "status": "passed",
        "candidate_kernel": "batched_chunk",
        "backbone_tokens": 2048,
        "backbone": {**recipe["backbone"], "kda_implementation": "chunk"},
        "comparison_errors": [],
        "long_kernel_shape": [1, 2049, 6, 128],
        "kernel_values_and_gradients": [{"max_absolute": 0, "relative_l2": 0}] * 7,
        "backbone_comparison": {
            precision: {field: {"relative_l2": 0, "max_absolute": 0} for field in ("cls", "genes")}
            for precision in ("float32", "bfloat16")
        },
    }
    return recipe, capacity, numerical


def test_matching_capacity_contract_and_inadequate_probes():
    recipe, capacity, numerical = evidence()
    assert validate_capacity(recipe, capacity, numerical)
    for field, wrong in (("world_size", 1), ("microbatch", 2), ("view_lengths", [128] * 4)):
        changed = copy.deepcopy(capacity)
        changed["identity"][field] = wrong
        with pytest.raises(ValueError):
            validate_capacity(recipe, changed, numerical)
    for field, wrong in (("backbone_tokens", 256), ("status", "failed"), ("comparison_errors", ["failed gate"])):
        changed = copy.deepcopy(numerical)
        changed[field] = wrong
        with pytest.raises(ValueError):
            validate_capacity(recipe, capacity, changed)
    changed = copy.deepcopy(numerical)
    changed["backbone"]["genes"] = 256
    with pytest.raises(ValueError, match="full requested backbone"):
        validate_capacity(recipe, capacity, changed)
    changed = copy.deepcopy(capacity)
    changed["steps"][0]["gradient_norm_rank0"] = float("nan")
    with pytest.raises(ValueError, match="finite gradients"):
        validate_capacity(recipe, changed, numerical)
    changed = copy.deepcopy(numerical)
    changed.pop("backbone_comparison")
    with pytest.raises(ValueError, match="numerical measurements"):
        validate_capacity(recipe, capacity, changed)


def test_recipe_preserves_scope_and_effective_batch():
    recipe, _, _ = evidence()
    assert recipe["expected_training_cells"] == 500000 and recipe["expected_validation_cells"] == 20000
    assert recipe["training"]["epochs"] == 2
    assert recipe["backbone"]["depth"] == 12 and recipe["backbone"]["width"] == 768
    assert recipe["crops"]["local_count"] == 2
    assert (
        recipe["training"]["world_size"] * recipe["training"]["microbatch"] * recipe["training"]["accumulation"] == 128
    )
