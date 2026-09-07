import json
from pathlib import Path

import pytest

from dinogenept.cell.schedule import learning_rate, sclong_learning_rate


def test_warmup_restart_boundaries_and_epoch_units():
    assert sclong_learning_rate(0) == 1e-6
    assert sclong_learning_rate(1) == pytest.approx(1.08e-5)
    assert sclong_learning_rate(5) == pytest.approx(5e-5)
    for start, peak in [(15, 4.5e-5), (40, 4.05e-5), (85, 3.645e-5)]:
        assert sclong_learning_rate(start) == 1e-6
        assert sclong_learning_rate(start + 5) == pytest.approx(peak)
    assert 1e-6 < sclong_learning_rate(14) < sclong_learning_rate(13)
    assert learning_rate("sclong_epoch_restarts", 5e-5, epoch=1, step=0, total_steps=10) == learning_rate(
        "sclong_epoch_restarts", 5e-5, epoch=1, step=999, total_steps=1000
    )


def test_native_formula_matches_sequential_cycle_recurrence():
    import math

    cycle, position, length = 0, 0, 15
    for epoch in range(300):
        peak = 5e-5 * 0.9**cycle
        factor = position / 5 if position < 5 else (1 + math.cos(math.pi * (position - 5) / (length - 5))) / 2
        assert sclong_learning_rate(epoch) == pytest.approx(1e-6 + (peak - 1e-6) * factor, abs=1e-18)
        position += 1
        if position >= length:
            cycle += 1
            position -= length
            length = (length - 5) * 2 + 5


def test_policy_and_bad_input():
    policy = json.loads((Path(__file__).parents[1] / "configs/cell/lr_sclong.json").read_text())
    assert policy["learning_rate"] == 5e-5 and policy["interval"] == "epoch"
    for value in (-1, 1.2, True):
        with pytest.raises(ValueError):
            sclong_learning_rate(value)
    with pytest.raises(ValueError):
        learning_rate("unknown", 5e-5, epoch=0, step=0, total_steps=10)


def test_current_campaign_is_one_epoch_with_explicit_data_protocol():
    selection = json.loads((Path(__file__).parents[1] / "configs/cell/default_data_selection.json").read_text())
    assert selection["pretraining_epochs"] == 1
    assert selection["protocol"] == "genecompass_all_cells_one_epoch_no_validation"
    assert selection["training_split"] == "all_500000_published_cells"
    assert selection["validation_split"] == "none"
    assert selection["downstream_overlap"] == "unknown_no_source_cell_metadata"
    assert selection["finetuning_method"] == "lora"
    assert selection["launch_allowed"] is False
    assert selection["current_execution_scope"] == "human500k_full_one_epoch_pretraining_only"
