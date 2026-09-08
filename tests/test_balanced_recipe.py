import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
from dinogenept.cell.backbone import BackboneConfig  # noqa: E402
from dinogenept.cell.pretraining import HeadConfig, PretrainingSystem  # noqa: E402


def test_balanced_preserves_gene_width_and_reduces_depth():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / 'configs/cell/genecompass50k_balanced_recipe.json').read_text())
    with torch.device('meta'):
        model = PretrainingSystem(BackboneConfig(**cfg['backbone']), HeadConfig(**cfg['heads']))
    assert model.student.backbone.gene.weight.shape == (23114, 768)
    assert len(model.student.backbone.blocks) == 4
    assert cfg['backbone']['kda_heads'] == 6
    assert cfg['backbone']['kda_head_dim'] == 128
    assert cfg['backbone']['expression_basis'] == 256
    assert cfg['heads']['hidden'] == 1024 and cfg['heads']['bottleneck'] == 128
    assert cfg['training']['accumulation'] == 1
    assert cfg['crops']['cap'] == 2048 and cfg['crops']['local_count'] == 2
    assert model.student.cell_head is model.student.gene_head
    count = sum(p.numel() for p in model.student.parameters())
    assert count == 40639827


def test_width256_recipe_and_user_selected_global_batch108():
    root = Path(__file__).resolve().parents[1] / 'configs/cell'
    reference = json.loads((root / 'genecompass50k_balanced_recipe.json').read_text())
    cfg = json.loads((root / 'genecompass50k_width256_recipe.json').read_text())
    expected = reference.copy()
    expected['backbone'] = {**reference['backbone'], 'width': 256,
                            'kda_heads': 2, 'mla_heads': 4}
    expected['training'] = {**reference['training'], 'microbatch': 54}
    assert cfg == expected
    assert cfg['training']['microbatch'] * cfg['training']['world_size'] == 108
    assert cfg['training']['accumulation'] == 1
    with torch.device('meta'):
        model = PretrainingSystem(BackboneConfig(**cfg['backbone']), HeadConfig(**cfg['heads']))
    assert model.student.backbone.gene.weight.shape == (23114, 256)
    assert model.student.cell_head is model.student.gene_head
    assert sum(p.numel() for p in model.student.parameters()) == 11017031
