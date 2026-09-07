"""Small CPU numerical fixtures, never research training or materialized data."""

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from dinogenept.cell.distillation import (  # noqa: E402
    TeacherCenter,
    dino_loss,
    ibot_loss,
    koleo_loss,
    pretraining_loss,
    reconstruction_loss,
    update_teacher,
)
from dinogenept.cell.lora import LoRALinear, attach_lora  # noqa: E402


def test_lora_initial_identity_and_frozen_base():
    torch.manual_seed(2)
    base = nn.Linear(7, 5)
    x = torch.randn(4, 7)
    expected = base(x).detach()
    layer = LoRALinear(base, rank=2, alpha=4, dropout=0)
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
    layer(x).square().sum().backward()
    assert layer.base.weight.grad is None and layer.base.bias.grad is None
    assert layer.b.grad.abs().sum() > 0
    assert layer.a.grad is not None
    with torch.no_grad():
        layer.b.fill_(0.1)
    actual = layer(x)
    layer.eval()
    layer.train()
    torch.testing.assert_close(layer(x), actual)
    torch.testing.assert_close(base(x), expected, rtol=0, atol=0)


def test_attach_exact_targets_and_validate_before_freeze():
    model = nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5))
    with pytest.raises(AttributeError):
        attach_lora(model, ["0", "missing"], rank=2)
    assert all(p.requires_grad for p in model.parameters())
    attach_lora(model, ["0"], rank=2)
    assert isinstance(model[0], LoRALinear)
    assert not model[1].weight.requires_grad
    assert {n for n, p in model.named_parameters() if p.requires_grad} == {"0.a", "0.b"}
    with pytest.raises(ValueError):
        attach_lora(model, ["1"], rank=2)


def test_dino_excludes_matching_globals_and_detaches_teacher():
    student = [torch.tensor([[0.1, 0.3]], requires_grad=True) for _ in range(4)]
    teacher = [torch.tensor([[0.2, 0.8]], requires_grad=True), torch.tensor([[0.9, 0.1]], requires_grad=True)]
    loss = dino_loss(student, teacher)
    pairs = [(t, s) for t in range(2) for s in range(4) if t != s]
    expected = sum(-(teacher[t] * (student[s] / 0.1).log_softmax(-1)).sum() for t, s in pairs) / 6
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert all(x.grad is None for x in teacher)
    assert all(x.grad is not None for x in student)


def test_masked_reconstruction_equal_cell_not_equal_token():
    prediction = torch.tensor([[1.0, 100.0], [2.0, 2.0]], requires_grad=True)
    targets = torch.tensor([[True, False], [True, True]])
    loss = reconstruction_loss(prediction, torch.zeros_like(prediction), targets)
    torch.testing.assert_close(loss, torch.tensor(2.5))
    loss.backward()
    assert prediction.grad[0, 1] == 0


def test_ibot_missing_mask_is_zero_and_teacher_detached():
    student = torch.randn(2, 3, 4, requires_grad=True)
    teacher = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    loss = ibot_loss(student, teacher.softmax(-1), mask)
    assert loss.item() == 0
    loss.backward()
    assert teacher.grad is None


def test_koleo_matches_source_and_does_not_select_self():
    torch.manual_seed(4)
    x = torch.randn(5, 8, requires_grad=True)
    z = nn.functional.normalize(x, dim=-1, eps=1e-8)
    distances = torch.cdist(z, z).detach()
    distances.fill_diagonal_(torch.inf)
    nearest = distances.argmin(-1)
    oracle = -(nn.PairwiseDistance(p=2, eps=1e-8)(z, z[nearest]) + 1e-8).log().mean()
    torch.testing.assert_close(koleo_loss(x), oracle)
    koleo_loss(x).backward()
    assert torch.isfinite(x.grad).all()
    with pytest.raises(ValueError):
        koleo_loss(x[:1])


def test_centers_valid_counts_and_no_teacher_gradient():
    center = TeacherCenter(2, momentum=0.5)
    logits = torch.tensor([[[2.0, 4.0], [100.0, 100.0]]], requires_grad=True)
    center.update(logits, torch.tensor([[True, False]]))
    torch.testing.assert_close(center.center, torch.tensor([1.0, 2.0]))
    assert not center.targets(logits, 0.1).requires_grad
    before = center.center.clone()
    center.update(logits, torch.zeros(1, 2, dtype=torch.bool))
    torch.testing.assert_close(before, center.center)


def test_teacher_ema_and_fixed_loss_schedule():
    student = nn.Linear(2, 2)
    teacher = deepcopy(student)
    before = teacher.weight.detach().clone()
    with torch.no_grad():
        student.weight.add_(1)
    update_teacher(student, teacher, 0.75)
    torch.testing.assert_close(teacher.weight, before + 0.25)
    assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
    assert pretraining_loss(1, 2, 3, 4, 5, step=0, total_steps=100) == 3
    assert pretraining_loss(1, 2, 3, 4, 5, step=10, total_steps=100) == 10.5
