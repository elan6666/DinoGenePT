import pytest

torch = pytest.importorskip('torch')
from torch import nn  # noqa: E402

from dinogenept.cell.finetune import FineTuneOptions  # noqa: E402
from dinogenept.cell.optimization import optimizer_groups  # noqa: E402
from dinogenept.cell.schedule import dinov2_learning_rate, teacher_momentum  # noqa: E402


def test_unseen_embedding_unchanged_with_other_gene_loss():
    model = nn.Sequential(nn.Embedding(5, 8), nn.Linear(8, 1))
    before = model[0].weight.detach().clone()
    opt = torch.optim.AdamW(optimizer_groups(model, 0.01), lr=0.01)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        model(torch.tensor([1, 2])).square().sum().backward()
        assert torch.count_nonzero(model[0].weight.grad[3]) == 0
        opt.step()
    torch.testing.assert_close(before[3], model[0].weight[3], rtol=0, atol=0)
    assert not torch.equal(before[1], model[0].weight[1])
    assert sorted(g['weight_decay'] for g in opt.param_groups) == [0., 0.01]


def test_shared_finetune_schedule_defaults():
    cfg = FineTuneOptions()
    assert cfg.lr_scheduler == 'dinov2_step_cosine'
    assert cfg.betas == (0.9, 0.95)
    assert teacher_momentum(1, 100) == pytest.approx(0.994)
    assert 0.999 < teacher_momentum(100, 100) < 1
    assert dinov2_learning_rate(cfg.learning_rate, batch=108, step=15, total_steps=100) == pytest.approx(6.495190528e-5)
    with pytest.raises(ValueError):
        teacher_momentum(0, 100)
