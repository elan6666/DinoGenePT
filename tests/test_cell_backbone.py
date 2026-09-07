from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")

from dinogenept.cell.backbone import BackboneConfig, CellBackbone, GatedMLA  # noqa: E402
from dinogenept.cell.lora import attach_lora  # noqa: E402


def small_config(**kwargs):
    return BackboneConfig(
        genes=20, width=32, depth=4, kda_heads=2, kda_head_dim=16, mla_heads=2,
        mla_head_dim=8, mla_shared_dim=4, query_rank=16, kv_rank=16,
        gradient_checkpointing=False, **kwargs,
    )


def fixture():
    ids = torch.tensor([[1, 2, 3, 4, 0], [2, 4, 5, 6, 7]])
    values = torch.tensor([[0.0, 2.0, 3.0, 1.0, 0.0], [0.0, 0.2, 3.0, 1.0, 0.1]])
    return ids, values, ids > 0


def test_backbone_padding_and_zero_expression_are_distinct():
    torch.manual_seed(5)
    model = CellBackbone(small_config())
    ids, values, valid = fixture()
    result = model(ids, values, valid)
    assert result["cls"].shape == (2, 32)
    assert result["genes"].shape == (2, 5, 32)
    assert result["genes"][0, -1].abs().sum() == 0
    assert result["genes"][0, 0].abs().sum() > 0
    (result["cls"].square().mean() + result["genes"].square().mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_hidden_expression_never_enters_backbone():
    model = CellBackbone(small_config()).eval()
    ids, values, valid = fixture()
    hidden = torch.zeros_like(valid)
    hidden[:, 1] = True
    expected = model(ids, values, valid, hidden)
    values[:, 1] = torch.nan
    actual = model(ids, values, valid, hidden)
    torch.testing.assert_close(actual["cls"], expected["cls"], atol=0, rtol=0)
    torch.testing.assert_close(actual["genes"], expected["genes"], atol=0, rtol=0)


def test_lora_backbone_base_frozen_and_initial_function_preserved():
    torch.manual_seed(10)
    model = CellBackbone(small_config()).eval()
    original = deepcopy(model)
    inputs = fixture()
    attach_lora(model, model.lora_targets(), rank=2, alpha=4, dropout=0)
    expected, actual = original(*inputs), model(*inputs)
    torch.testing.assert_close(actual["cls"], expected["cls"], atol=0, rtol=0)
    actual["cls"].sum().backward()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.endswith((".a", ".b")) for n, _ in trainable)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for _, p in trainable)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None, name


def test_global_mla_permutation_equivariance():
    torch.manual_seed(12)
    layer = GatedMLA(small_config())
    x = torch.randn(2, 6, 32)
    valid = torch.ones(2, 6, dtype=torch.bool)
    write = valid.clone()
    write[:, 2] = False
    perm = torch.tensor([3, 0, 4, 1, 5, 2])
    expected = layer(x, valid, write)[:, perm]
    actual = layer(x[:, perm], valid[:, perm], write[:, perm])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
