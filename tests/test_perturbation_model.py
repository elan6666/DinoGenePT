import pytest

torch = pytest.importorskip("torch")

from dinogenept.cell.backbone import BackboneConfig  # noqa: E402
from dinogenept.cell.perturbation import PerturbationConfig, PerturbationSystem  # noqa: E402
from dinogenept.cell.pretraining import HeadConfig, PretrainingNetwork  # noqa: E402


def fixture(**overrides):
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
        pretrained, PerturbationConfig(vector_width=6, decoder_width=8, lora_rank=2, lora_alpha=4,
                                      lora_dropout=0, **overrides)
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
        observed_cell_ids=torch.arange(3),
        teacher_cell_ids=torch.arange(3, 6),
        observed_conditions=torch.zeros(3, dtype=torch.long),
        full_control=controls,
        train_post=post,
        axis=axis,
        targets=torch.tensor([2, 5]),
        source_vectors={"TextBase": torch.randn(2, 6), "GO": torch.randn(2, 6), "HPA": None},
    )
    return model, batch


@pytest.mark.parametrize("ibot,koleo", [(False, False), (True, False), (False, True), (True, True)])
def test_optional_losses_defaults_weighting_and_backward(ibot, koleo):
    model, batch = fixture(observed_ibot=ibot, observed_koleo=koleo, ibot_chunk_size=2)
    hidden = torch.zeros_like(batch["observed_view"]["valid"])
    hidden[0, :2] = True
    batch["observed_view"] = {**batch["observed_view"], "hidden": hidden}
    result = model(**batch)
    expected = (0.5 * result["reconstruction"] + result["primary_dino"] + result["knowledge_dino"]
                + result["observed_dino"] + (0.5 if ibot else 0) * result["observed_ibot"]
                + (0.1 if koleo else 0) * result["observed_koleo"])
    torch.testing.assert_close(result["total"], expected)
    assert result["koleo_eligible_cells"] == 0
    assert result["ibot_masked_cells"] == int(ibot)
    assert (result["observed_ibot"] > 0) == ibot
    if ibot:
        assert model.student.gene_head is model.student.cell_head
    result["total"].backward()
    assert all(p.grad is None for p in model.teacher.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.student.parameters() if p.grad is not None)


def test_singleton_skip_and_reject_self_pairing():
    model, batch = fixture(observed_ibot=True, observed_koleo=True)
    batch["observed_view"] = None
    result = model(**batch)
    assert result["observed_dino"] == result["observed_ibot"] == result["observed_koleo"] == 0
    result["total"].backward()
    model, batch = fixture()
    batch["teacher_cell_ids"] = batch["observed_cell_ids"]
    with pytest.raises(ValueError, match="disjoint"):
        model(**batch)


def test_condition_filtered_koleo_mixed_conditions():
    from dinogenept.cell.perturbation import condition_filtered_koleo
    cls = torch.randn(4, 8, requires_grad=True)
    condition = torch.tensor([0, 0, 1, 1])
    loss, n = condition_filtered_koleo(cls, condition)
    assert n == 4 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(cls.grad).all() and cls.grad.abs().sum() > 0
    zero, n = condition_filtered_koleo(cls, torch.zeros(4))
    assert n == 0 and zero == 0


def test_ibot_teacher_uses_same_b_and_clean_mask():
    model, batch = fixture(observed_ibot=True)
    hidden = torch.zeros_like(batch["observed_view"]["valid"])
    hidden[0, :2] = True
    batch["observed_view"] = {**batch["observed_view"], "hidden": hidden}
    seen = []
    original = model.teacher.encode
    def capture(view, *args, **kwargs):
        seen.append(view)
        return original(view, *args, **kwargs)
    model.teacher.encode = capture
    model(**batch)
    assert len(seen) == 2
    assert not seen[1]["hidden"].any()
    for key in ("gene_ids", "expression", "valid"):
        torch.testing.assert_close(seen[1][key], batch["observed_view"][key], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA smoke requires server GPU")
@pytest.mark.parametrize("ibot,koleo", [(False, False), (True, False), (False, True), (True, True)])
def test_cuda_bfloat16_two_optimizer_steps(ibot, koleo):
    from dinogenept.cell.finetune import to_device
    model, batch = fixture(observed_ibot=ibot, observed_koleo=koleo, ibot_chunk_size=2)
    hidden = torch.zeros_like(batch["observed_view"]["valid"])
    hidden[0, :2] = True
    batch["observed_view"] = {**batch["observed_view"], "hidden": hidden}
    model = model.cuda()
    def move(value):
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value.cuda() if torch.is_tensor(value) else value
    batch = move(to_device(batch, torch.device("cuda")))
    optimizer = torch.optim.AdamW((p for p in model.student.parameters() if p.requires_grad), lr=1e-4)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(**batch)
        assert torch.isfinite(result["total"])
        result["total"].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.student.parameters() if p.grad is not None)
        optimizer.step()
        model.update_ema(step + 1, 2)


def test_finetuning_loss_ablation_registry():
    from dinogenept.cell.ablations import ablation_registry
    rows = ablation_registry()
    for number, enabled in enumerate(((False, False), (True, False), (False, True), (True, True))):
        config = PerturbationConfig(**rows[f"F{number}-OBSERVED"]["perturbation"])
        assert (config.observed_ibot, config.observed_koleo) == enabled
        assert config.reconstruction_weight == 0.5
        assert config.ibot_weight == 0.5 and config.koleo_weight == 0.1


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
