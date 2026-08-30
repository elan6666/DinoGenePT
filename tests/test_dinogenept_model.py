from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dinogenept.config import load_config  # noqa: E402
from dinogenept.models.dinogenept import PLUGIN, build_model  # noqa: E402
from dinogenept.models.dinogenept.losses import koleo_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ABLATIONS = (
    "07-dynamic-locals.yaml",
    "12-attention-residual.yaml",
    "13-dense-adapter.yaml",
    "14-moe-aux.yaml",
    "15-moe-quantile.yaml",
    "16-moe-situ.yaml",
    "17-backbone-situ.yaml",
    "19-hybrid-kda.yaml",
)


@pytest.mark.parametrize("filename", ABLATIONS)
def test_architecture_ablations_forward_and_backward(filename):
    config = load_config(
        ROOT / "configs/experiments/dinogenept/ablations" / filename
    ).payload
    model = build_model(
        config,
        n_genes=20,
        prior_dimensions={"base": 8, "go": 8, "protein": 8, "pathway": 8, "hpa": 8},
    )
    expression = torch.randn(3, 20)
    vectors = torch.randn(3, 1, 8)
    mask = torch.ones(3, 1, dtype=torch.bool)
    view = model.conditional_view(
        expression,
        source="base",
        prior_vectors=vectors,
        prior_mask=mask,
    )
    prediction = model.predict_expression(expression, view)
    assert prediction.shape == expression.shape
    prediction.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("filename", ("14-moe-aux.yaml", "19-hybrid-kda.yaml"))
def test_sensitive_ablations_support_cuda_bfloat16(filename):
    config = load_config(
        ROOT / "configs/experiments/dinogenept/ablations" / filename
    ).payload
    model = build_model(
        config,
        n_genes=20,
        prior_dimensions={"base": 8, "go": 8, "protein": 8, "pathway": 8, "hpa": 8},
    ).cuda()
    expression = torch.randn(3, 20, device="cuda")
    vectors = torch.randn(3, 1, 8, device="cuda")
    mask = torch.ones(3, 1, dtype=torch.bool, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        view = model.conditional_view(
            expression,
            source="base",
            prior_vectors=vectors,
            prior_mask=mask,
        )
        loss = model.predict_expression(expression, view).square().mean()
    loss.backward()
    assert torch.isfinite(loss)


def test_koleo_loss_supports_backward_without_inplace_version_error():
    features = torch.randn(4, 16, requires_grad=True)
    loss = koleo_loss(features)
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_mask_replaces_expression_value_but_preserves_gene_identity():
    config = load_config(
        ROOT / "configs/experiments/dinogenept/ablations/02-dino-ibot.yaml"
    ).payload
    model = build_model(config, n_genes=5, prior_dimensions={"base": 8})
    indices = torch.tensor([[1, 3]])
    first = torch.tensor([[1.0, 2.0]])
    second = torch.tensor([[100.0, -50.0]])
    mask = torch.ones_like(first, dtype=torch.bool)
    first_tokens = model.backbone.embed_gene_tokens(indices, first, mask)
    second_tokens = model.backbone.embed_gene_tokens(indices, second, mask)
    assert torch.equal(first_tokens, second_tokens)
    assert not torch.equal(
        model.backbone.embed_gene_tokens(indices, first, None),
        model.backbone.embed_gene_tokens(indices, second, None),
    )


def test_source_view_ablations_keep_the_same_parameter_surface():
    dimensions = {"base": 8, "go": 8, "protein": 8, "pathway": 8, "hpa": 8}
    totals = []
    for filename in (
        "00-supervised.yaml",
        "03-local-go.yaml",
        "04-local-protein.yaml",
        "05-local-pathway.yaml",
        "06-local-hpa.yaml",
        "07-dynamic-locals.yaml",
    ):
        config = load_config(
            ROOT / "configs/experiments/dinogenept/ablations" / filename
        ).payload
        totals.append(build_model(config, n_genes=20, prior_dimensions=dimensions).parameter_receipt()["total"])
    assert len(set(totals)) == 1


def test_registered_plugin_owns_training_and_prediction_hooks():
    assert all(
        callable(getattr(PLUGIN, name))
        for name in ("build_model", "build_trainer", "predict_split", "permutation_check")
    )
