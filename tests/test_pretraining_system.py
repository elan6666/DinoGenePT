import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dinogenept.cell.backbone import BackboneConfig  # noqa: E402
from dinogenept.cell.pretraining import HeadConfig, PretrainingSystem  # noqa: E402
from dinogenept.cell.sampling import CropConfig, collate_crops, sample_crops  # noqa: E402


def system_and_batch():
    torch.manual_seed(16)
    config = BackboneConfig(
        genes=25,
        width=16,
        depth=4,
        kda_heads=1,
        kda_head_dim=8,
        mla_heads=2,
        mla_head_dim=4,
        mla_shared_dim=2,
        query_rank=8,
        kv_rank=8,
        gradient_checkpointing=True,
    )
    model = PretrainingSystem(config, HeadConfig(hidden=32, bottleneck=8, cell_prototypes=16, gene_prototypes=12))
    rows = [
        sample_crops(
            np.arange(1, 26),
            np.arange(1, 26) + i,
            500,
            cell_id=f"cell-{i}",
            epoch=0,
            config=CropConfig(cap=20),
        )
        for i in range(3)
    ]
    return model, collate_crops(rows)


def test_five_losses_backward_teacher_stopgrad_and_ema():
    model, batch = system_and_batch()
    model.train()
    assert model.student.training and not model.teacher.training
    before = {k: v.clone() for k, v in model.teacher.state_dict().items()}
    losses = model(batch, step=1, total_steps=10)
    assert set(losses) == {"total", "expression", "cell_expression", "dino", "ibot", "koleo"}
    assert all(torch.isfinite(x) for x in losses.values())
    losses["total"].backward()
    assert not any(p.grad is not None for p in model.teacher.parameters())
    assert all(p.grad is not None for p in model.student.parameters() if p.requires_grad)
    for name in ("backbone", "cell_head", "gene_head", "gene_expression", "cell_gene_query"):
        params = list(getattr(model.student, name).parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params), name
        assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
    for key, value in model.teacher.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    optimizer = torch.optim.SGD(model.student.parameters(), lr=0.001)
    optimizer.step()
    model.update_ema(1, 10)
    assert any(not torch.equal(value, before[key]) for key, value in model.teacher.state_dict().items())


def test_validation_does_not_update_centers_or_teacher():
    model, batch = system_and_batch()
    model.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model(batch, step=0, total_steps=10)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_collator_rejects_duplicate_cells():
    row = sample_crops([1, 2], [2, 4], 6, cell_id="duplicate", epoch=0, config=CropConfig())
    with pytest.raises(ValueError):
        collate_crops([row, row])
