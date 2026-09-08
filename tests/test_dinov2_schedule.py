import math

import numpy as np
import pytest

from dinogenept.cell.schedule import dinov2_learning_rate, learning_rate


@pytest.mark.parametrize('batch', [8, 16, 128, 3072])
@pytest.mark.parametrize('total', [1, 6, 7, 391, 1000])
def test_official_numpy_schedule_parity(batch, total):
    peak = 2e-4 * math.sqrt(batch / 1024)
    warmup = int(.16 * total)
    iters = np.arange(total - warmup)
    reference = np.concatenate((np.linspace(0, peak, warmup),
                                1e-6 + .5 * (peak - 1e-6) * (1 + np.cos(np.pi * iters / len(iters)))))
    actual = [learning_rate('dinov2_step_cosine', 2e-4, epoch=0, step=s,
                             total_steps=total, batch=batch) for s in range(total)]
    np.testing.assert_allclose(actual, reference, rtol=1e-13, atol=1e-18)
    assert dinov2_learning_rate(2e-4, batch=batch, step=total, total_steps=total) == 1e-6


def test_batch_scaling_and_resume_are_stateless():
    assert dinov2_learning_rate(2e-4, batch=3072, step=100000, total_steps=625000) == pytest.approx(3.464101615e-4)
    for step in [0, 61, 62, 63, 200, 390]:
        assert learning_rate('dinov2_step_cosine', 2e-4, epoch=0, step=step, total_steps=391, batch=128) == (
            learning_rate('dinov2_step_cosine', 2e-4, epoch=999, step=step, total_steps=391, batch=128))
