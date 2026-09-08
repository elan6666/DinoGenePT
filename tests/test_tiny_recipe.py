import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
from dinogenept.cell.backbone import BackboneConfig, CellBackbone  # noqa: E402
from dinogenept.cell.pretraining import HeadConfig, PretrainingSystem  # noqa: E402


def test_tiny_recipe_parameter_budget_and_preserved_objectives():
    cfg = json.loads((Path(__file__).resolve().parents[1] /
                      'configs/cell/genecompass50k_500k_recipe.json').read_text())
    with torch.device('meta'):
        model = PretrainingSystem(BackboneConfig(**cfg['backbone']), HeadConfig(**cfg['heads']))
    count = sum(p.numel() for p in model.student.parameters())
    assert abs(count - 500000) / 500000 < .01
    assert cfg['backbone']['genes'] == 23113 and cfg['backbone']['depth'] == 12
    assert cfg['training']['accumulation'] == 1
    assert cfg['heads']['cell_prototypes'] == 8192
    assert cfg['crops']['cap'] == 2048


def test_expression_basis_is_configurable_and_legacy_default_unchanged():
    assert BackboneConfig(genes=10).expression_basis == 256
    cfg = BackboneConfig(genes=10, width=16, depth=4, kda_heads=1, kda_head_dim=16,
                         mla_heads=1, mla_head_dim=16, mla_shared_dim=8, query_rank=8,
                         kv_rank=8, expression_basis=128)
    model = CellBackbone(cfg)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = model(ids, torch.ones(2, 3), torch.ones(2, 3, dtype=torch.bool))
    out['cls'].square().sum().backward()
    assert out['genes'].shape == (2, 3, 16)
    assert model.value.mix.weight.shape == (128, 128)
    assert model.value.mix.weight.grad is not None
