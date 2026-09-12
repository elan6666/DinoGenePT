"""Tiny native fixtures: these establish semantics, not model quality."""

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")

from test_pretraining_system import system_and_batch  # noqa: E402

from dinogenept.cell.muon import (  # noqa: E402
    apply_qk_clip,
    build_optimizer,
    head_layout,
    orthogonalize,
    sinkhorn_update,
)
from dinogenept.cell.pretraining import HeadConfig, PretrainingSystem  # noqa: E402
from dinogenept.cell.schedule import learning_rate, teacher_momentum  # noqa: E402


@pytest.mark.parametrize("schedule,warm,floor", [("glm5_cosine", 5, .2), ("kimi3_cosine", 1, .01),
                                                ("ds41_plateau_cosine", 5, .1)])
def test_recipe_boundaries(schedule, warm, floor):
    values = [learning_rate(schedule, 1e-4, epoch=0, step=s, total_steps=100, batch=1024) for s in range(101)]
    assert values[0] == 0
    assert values[warm] == pytest.approx(1e-4)
    assert values[-1] == pytest.approx(1e-4 * floor)
    assert all(v >= 0 for v in values)
    if schedule == "ds41_plateau_cosine":
        assert values[61] == values[62] == pytest.approx(1e-4)
        assert values[89] == values[100]


@pytest.mark.parametrize("chunk", [1, 3, 256])
@pytest.mark.parametrize("separate,empty", [(False, False), (True, False), (True, True)])
def test_chunk_full_update_parity(chunk, separate, empty):
    torch.set_num_threads(1)
    original, batch = system_and_batch()
    heads = HeadConfig(hidden=32, bottleneck=8, cell_prototypes=16, gene_prototypes=12,
                       ibot_separate_head=separate)
    reference = PretrainingSystem(original.student.backbone.config, heads)
    candidate = PretrainingSystem(original.student.backbone.config, HeadConfig(**{**heads.__dict__,
                                                                                "ibot_chunk_size": chunk}))
    candidate.load_state_dict(reference.state_dict())
    if empty:
        for view in batch["views"]:
            view["hidden"].zero_()
            view["targets"].zero_()
    outputs = []
    for model in (reference, candidate):
        optim = torch.optim.AdamW(model.student.parameters(), lr=1e-4)
        losses = model(batch, step=2, total_steps=10)
        losses["total"].backward()
        outputs.append(({k: v.detach() for k, v in losses.items()},
                        {k: p.grad.clone() for k, p in model.student.named_parameters()}))
        optim.step()
        model.update_ema(3, 10)
    for k in outputs[0][0]:
        torch.testing.assert_close(outputs[0][0][k], outputs[1][0][k], atol=1e-6, rtol=1e-4)
    for k in outputs[0][1]:
        torch.testing.assert_close(outputs[0][1][k], outputs[1][1][k], atol=1e-6, rtol=1e-4)
    for k, value in reference.state_dict().items():
        # Adam may magnify tiny cancellation errors at near-zero gradients.
        torch.testing.assert_close(value, candidate.state_dict()[k], atol=2e-6, rtol=1e-4)


@pytest.mark.parametrize("mode", ["adamw", "muon", "glm5_muon", "kimi3_muon", "ds41_muon"])
def test_optimizer_partition_update_and_resume(mode):
    torch.set_num_threads(1)
    model, batch = system_and_batch()
    settings = dict(optimizer=mode, learning_rate=1e-4, weight_decay=.01, betas=(.9, .95))
    opt = build_optimizer(model.student, settings)
    pad = model.student.backbone.gene.weight[0].detach().clone()
    loss = model(batch, step=2, total_steps=10)["total"]
    loss.backward()
    opt.step()
    torch.testing.assert_close(pad, model.student.backbone.gene.weight[0], atol=0, rtol=0)
    twin, _ = system_and_batch()
    twin.load_state_dict(model.state_dict())
    resumed = build_optimizer(twin.student, settings)
    resumed.load_state_dict(deepcopy(opt.state_dict()))
    for system, optimizer in ((model, opt), (twin, resumed)):
        optimizer.zero_grad(set_to_none=True)
        system(batch, step=3, total_steps=10)["total"].backward()
        optimizer.step()
    for p, q in zip(model.student.parameters(), twin.student.parameters(), strict=True):
        torch.testing.assert_close(p, q, atol=0, rtol=0)


def test_layout_complete_disjoint_and_value_difference():
    model, _ = system_and_batch()
    mla = model.student.backbone.blocks[-1].mixer
    for mode in ("glm5_muon", "kimi3_muon", "ds41_muon"):
        layout = head_layout(mla, "kv_b_proj", mode)
        flat = sum(layout, [])
        assert sorted(flat) == list(range(mla.kv_b_proj.weight.shape[0]))
        assert len(flat) == len(set(flat))
    assert len(head_layout(mla, "kv_b_proj", "ds41_muon")) == mla.heads + 1


def test_qk_clip_does_not_touch_value_or_shared_key():
    model, batch = system_and_batch()
    build_optimizer(model.student, dict(optimizer="kimi3_muon", learning_rate=1e-4, weight_decay=.01,
                                        betas=(.9, .95), qk_clip_threshold=1.))
    mla = model.student.backbone.blocks[-1].mixer
    before = mla.kv_b_proj.weight.detach().clone().view(mla.heads, 2 * mla.dim, -1)
    shared = mla.kv_a_proj.weight.detach().clone()
    mla.qk_max.fill_(4)
    assert apply_qk_clip(model.student) == mla.heads
    after = mla.kv_b_proj.weight.view_as(before)
    torch.testing.assert_close(after[:, :mla.dim], before[:, :mla.dim] * .5)
    torch.testing.assert_close(after[:, mla.dim:], before[:, mla.dim:], atol=0, rtol=0)
    torch.testing.assert_close(shared, mla.kv_a_proj.weight, atol=0, rtol=0)
    assert mla.qk_max.count_nonzero() == 0
    model(batch, step=1, total_steps=10)["total"].backward()
    assert torch.isfinite(mla.qk_max).all()


def test_ns_zero_rectangular_and_ema_candidates():
    assert orthogonalize(torch.zeros(3, 7)).count_nonzero() == 0
    torch.manual_seed(1)
    x = torch.randn(3, 7)
    torch.testing.assert_close(orthogonalize(x).T, orthogonalize(x.T))
    heads = torch.randn(4, 3, 7)
    torch.testing.assert_close(orthogonalize(heads), torch.stack([orthogonalize(h) for h in heads]))
    for initial in (.990, .994, .998):
        assert teacher_momentum(1, 10, initial) == pytest.approx(initial)
        assert initial < teacher_momentum(10, 10, initial) < 1


def test_sinkhorn_rows_and_absent_gene_semantics():
    torch.manual_seed(1)
    raw = torch.randn(5, 7)
    raw[0] = 0
    raw[1] *= 1e-10
    out = sinkhorn_update(raw)
    assert out[:2].count_nonzero() == 0
    torch.testing.assert_close(out[2:].square().mean(-1), torch.ones(3), rtol=1e-6, atol=1e-6)
    model, _ = system_and_batch()
    opt = build_optimizer(model.student, dict(optimizer="ds41_muon", embedding_optimizer="sinkhorn",
                                              learning_rate=1e-3, betas=(.9, .95), weight_decay=.01))
    table = model.student.backbone.gene.weight
    before = table.detach().clone()
    table.grad = torch.zeros_like(table)
    table.grad[1] = 1
    opt.step()
    torch.testing.assert_close(table[2:], before[2:], rtol=0, atol=0)
    torch.testing.assert_close(table[0], before[0], rtol=0, atol=0)
    previous = table.detach().clone()
    table.grad.zero_()
    opt.step()
    assert not torch.equal(table[1], previous[1])  # historical momentum, not current activity
    torch.testing.assert_close(table[2:], previous[2:], rtol=0, atol=0)


@pytest.mark.parametrize("overlay", [
    {"mixer_type": "eretnet"}, {"mixer_type": "full_mla"}, {"residual_type": "standard"},
    {"residual_type": "deepnorm", "ffn_type": "sglu", "mixer_type": "eretnet"},
    {"kda_direction": "bidirectional"}, {"short_convolution": 4}, {"ffn_type": "swiglu"},
    {"delta_rule": "scalar"}, {"global_attention": "gqa"},
])
def test_architecture_small_backward(overlay):
    from dataclasses import replace

    torch.set_num_threads(1)
    original, batch = system_and_batch()
    model = PretrainingSystem(replace(original.student.backbone.config, **overlay), original.head_config)
    loss = model(batch, step=2, total_steps=10)["total"]
    loss.backward()
    for name, p in model.student.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_eret_algebra_matches_explicit_attention():
    from dinogenept.cell.backbone import ERetMixer

    model, _ = system_and_batch()
    mixer = ERetMixer(model.student.backbone.config)
    x = torch.randn(2, 5, 16)
    valid = torch.ones(2, 5, dtype=torch.bool)
    writable = valid.clone()
    writable[:, 2] = False
    shape = (2, 5, mixer.heads, mixer.dim)
    q, k, v, u = [getattr(mixer, n)(x).reshape(shape).transpose(1, 2)
                  for n in ("q_proj", "k_proj", "v_proj", "u_proj")]
    weights = (q.relu() @ k.relu().transpose(-1, -2)) / mixer.dim
    weights = weights * writable[:, None, None, :]
    expected = weights @ v
    expected = expected / (expected.norm(dim=-1, keepdim=True) / mixer.dim ** .5).clamp_min(1e-12)
    expected = mixer.o_proj((expected * torch.nn.functional.silu(u)).transpose(1, 2).flatten(-2))
    torch.testing.assert_close(mixer(x, valid, writable), expected, atol=1e-6, rtol=1e-4)


def test_registry_resolves_without_changing_lineage():
    import json
    from pathlib import Path

    from dinogenept.cell.ablations import ablation_registry, resolve_ablation

    base = json.loads(Path("configs/cell/genecompass50k_width256_recipe.json").read_text())
    base["lineage_fixture"] = "unchanged"
    base["crops"]["hvg_gene_ids"] = list(range(1, 21))  # synthetic config only
    for name in ablation_registry():
        result = resolve_ablation(base, name)
        assert result["lineage_fixture"] == "unchanged"
        assert result["ablation"]["id"] == name
    with pytest.raises(ValueError):
        resolve_ablation(base, "invented")


@pytest.mark.parametrize("sources,anchor,expected", [([], "TextBase", 0), (["GO"], "TextBase", 1),
                                                   (["HPA"], "TextBase", 0), (["TextBase", "GO"], "CellGene", 2)])
def test_knowledge_selection_and_anchor(sources, anchor, expected):
    from dataclasses import replace

    from test_perturbation_model import fixture

    model, batch = fixture()
    config = replace(model.config, local_sources=tuple(sources), main_anchor=anchor)
    model.config = config
    model.student.config = config
    losses = model(**batch)
    assert losses["local_count"] == expected
    losses["total"].backward()
    assert torch.isfinite(losses["total"])


@pytest.mark.parametrize("source", ["Protein", "Pathway", "HPA"])
def test_each_optional_present_knowledge_source(source):
    from dataclasses import replace

    from test_perturbation_model import fixture

    model, batch = fixture()
    model.config = replace(model.config, local_sources=(source,))
    batch["source_vectors"][source] = torch.randn(2, 6)
    losses = model(**batch)
    assert losses["local_count"] == 1
    losses["total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.student.adapters[source].parameters())


def test_hvg_locals_require_frozen_ids_and_keep_continuous_values():
    import numpy as np

    from dinogenept.cell.sampling import CropConfig, sample_crops

    with pytest.raises(ValueError, match="HVG"):
        CropConfig(local_sampling="hvg")
    cfg = CropConfig(local_sampling="hvg", hvg_gene_ids=(5, 3, 1), local_hvg_k=2)
    row = sample_crops(np.arange(1, 7), np.arange(1, 7) * .1, None, cell_id="test", epoch=0,
                       config=cfg, expression_scale="genecompass_published_continuous")
    assert row["views"][2]["gene_ids"].tolist() == [3, 5]
    np.testing.assert_allclose(row["views"][2]["expression"], [.3, .5])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires server CUDA")
@pytest.mark.parametrize("chunk", [256, 512, 1024])
def test_bf16_head_chunk_gradient_parity(chunk):
    from dinogenept.cell.distillation import ProjectionHead
    from dinogenept.cell.ibot_chunk import projected_ibot

    torch.manual_seed(42)
    student = ProjectionHead(32, 128, 64, 16).cuda()
    teacher = deepcopy(student).requires_grad_(False)
    x = torch.randn(1031, 32, device="cuda", requires_grad=True)
    target = torch.randn_like(x)
    weights = x.new_full((1031,), 1 / 1031)
    params = tuple(student.parameters())
    results = []
    for size in (0, chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, total, count = projected_ibot(student, teacher, x, target, weights, x.new_zeros(128), .07, .1, size)
        gradients = torch.autograd.grad(loss, (x, *params))
        results.append((loss.detach(), total, count, gradients))
    for a, b in zip(results[0][:3], results[1][:3], strict=True):
        torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-2)
    a = torch.cat([g.flatten() for g in results[0][3]])
    b = torch.cat([g.flatten() for g in results[1][3]])
    assert ((a - b).norm() / a.norm().clamp_min(1e-8)) < .01
    assert torch.isfinite(b).all()


@pytest.mark.parametrize("mode", ["glm5_muon", "kimi3_muon", "ds41_muon"])
def test_runner_one_step_optimizer_integration(tmp_path, mode):
    from test_pretraining_runner import fixture_config

    from dinogenept.cell.train import run_pretraining

    cfg = fixture_config(tmp_path / "data")
    cfg["training"].update(optimizer=mode, lr_scheduler="glm5_cosine")
    cfg["heads"]["ibot_chunk_size"] = 3
    assert run_pretraining(cfg, smoke_one_step=True)["completed_steps"] == 1
