import pytest

torch = pytest.importorskip("torch")

from dinogenept.cell.backbone import BackboneConfig  # noqa: E402
from dinogenept.cell.perturbation import PerturbationConfig, PerturbationSystem  # noqa: E402
from dinogenept.cell.pretraining import HeadConfig, PretrainingNetwork  # noqa: E402


def fixture():
    torch.manual_seed(4)
    torch.set_num_threads(1)
    config = BackboneConfig(
        genes=12,
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
    pretrained = PretrainingNetwork(config, HeadConfig(hidden=16, bottleneck=8, cell_prototypes=8, gene_prototypes=8))
    model = PerturbationSystem(
        pretrained, PerturbationConfig(vector_width=6, decoder_width=8, lora_rank=2, lora_alpha=4, lora_dropout=0)
    )
    axis = torch.arange(1, 13)
    controls = torch.rand(3, 12)
    controls[:, 0] = 0  # Real zeros are valid genes, not pads.
    view = {"gene_ids": axis.expand(3, -1), "expression": controls, "valid": torch.ones(3, 12, dtype=torch.bool)}
    post = controls + torch.rand_like(controls)
    post_view = {**view, "expression": post}
    batch = dict(
        control_view=view,
        observed_view=post_view,
        teacher_view=post_view,
        full_control=controls,
        train_post=post,
        axis=axis,
        targets=torch.tensor([2, 5]),
        source_vectors={"TextBase": torch.randn(2, 6), "GO": torch.randn(2, 6), "HPA": None},
    )
    return model, batch


def test_lora_only_backbone_gradients_and_missing_local_omission():
    model, batch = fixture()
    model.train()
    frozen = {n: p.clone() for n, p in model.student.backbone.named_parameters() if not p.requires_grad}
    before_teacher = {n: p.clone() for n, p in model.teacher.named_parameters()}
    result = model(**batch)
    assert result["local_count"] == 1
    result["total"].backward()
    assert all(p.grad is None for p in model.teacher.parameters())
    assert all(p.grad is None for p in model.student.adapters["HPA"].parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.student.adapters["GO"].parameters())
    optimizer = torch.optim.AdamW((p for p in model.student.parameters() if p.requires_grad), lr=1e-3)
    optimizer.step()
    for name, parameter in model.student.backbone.named_parameters():
        if name in frozen:
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
            assert parameter.grad is None
        else:
            assert name.endswith((".a", ".b"))
    for name, parameter in model.teacher.named_parameters():
        torch.testing.assert_close(parameter, before_teacher[name], rtol=0, atol=0)
    model.update_ema(1, 10)
    assert any(not torch.equal(p, before_teacher[n]) for n, p in model.teacher.named_parameters())


def test_main_only_inference_and_canonical_combo_order():
    model, batch = fixture()
    model.eval()
    args = [
        batch["control_view"],
        batch["full_control"],
        batch["axis"],
        batch["targets"],
        batch["source_vectors"]["TextBase"],
    ]
    with torch.no_grad():
        expected = model.predict(*args)
        args[-2] = args[-2].flip(0)
        args[-1] = args[-1].flip(0)
        actual = model.predict(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert expected.shape == (3, 12)
    with pytest.raises(RuntimeError, match="training-only"):
        model(**batch)


def test_absent_locals_are_not_zero_filled_and_partial_combos_fail():
    model, batch = fixture()
    batch["source_vectors"]["GO"] = None
    result = model(**batch)
    assert result["local_count"] == 0 and result["knowledge_dino"] == 0
    batch["source_vectors"]["GO"] = torch.zeros(2, 6)
    with pytest.raises(ValueError, match="Zero-filled"):
        model(**batch)
    batch["source_vectors"]["GO"] = torch.randn(1, 6)
    with pytest.raises(ValueError, match="EVERY"):
        model(**batch)
    batch["source_vectors"].pop("TextBase")
    with pytest.raises(ValueError, match="Missing required anchor"):
        model(**batch)
