import hashlib

import numpy as np
import pytest

from dinogenept.evaluation.cellfm import control_draws, select_de


def test_draw_policy_is_context_resampling_then_replacement_not_uniform_without_replacement():
    truth_contexts = ["K562", "K562", "RPE1"]
    pools = {"K562": np.arange(4), "RPE1": np.arange(10, 15)}
    rows, contexts = control_draws("norman", "test", "A+B", truth_contexts, pools)
    # Independently spelled frozen source algorithm (Grad-Pert data/controls.py).
    seed = int.from_bytes(hashlib.sha256(b"20260824::norman::test::A+B").digest()[:16], byteorder="big")
    rng = np.random.Generator(np.random.PCG64(seed))
    expected_contexts = [str(x) for x in rng.choice(truth_contexts, size=300, replace=True)]
    expected_rows = [int(rng.choice(pools[c])) for c in expected_contexts]
    assert rows == expected_rows and contexts == expected_contexts
    assert len(set(rows)) < 300
    assert control_draws("norman", "val", "A+B", truth_contexts, pools)[0] != rows


def test_de_order_non_dropout_then_twenty_then_target_exclusion_without_refill():
    symbols = ["A"] + [f"G{i}" for i in range(24)]
    perturbed, control = np.ones(25), np.ones(25)
    perturbed[1] = 0  # Dropout: exclude before selecting top20.
    perturbed[2] = control[2] = 0  # Both-zero gene retained as in upstream.
    selected = select_de(symbols, symbols, "A+ctrl", perturbed, control)
    assert selected == list(range(2, 21)) and len(selected) == 19
    with pytest.raises(ValueError, match="No DE"):
        select_de(["A"], ["A"], "A+ctrl", np.ones(1), np.ones(1))
