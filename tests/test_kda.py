import pytest

torch = pytest.importorskip("torch")

from dinogenept.cell.kda import chunk_kda, recurrent_kda  # noqa: E402


@pytest.mark.parametrize("size", [1, 4, 16])
def test_chunk_matches_recurrence_values_and_gradients(size):
    torch.manual_seed(7)
    shape = (2, 9, 2, 4)
    q = torch.nn.functional.normalize(torch.randn(shape), dim=-1).requires_grad_()
    k = torch.nn.functional.normalize(torch.randn(shape), dim=-1).requires_grad_()
    v = torch.randn(2, 9, 2, 3, requires_grad=True)
    g = (-torch.rand(shape) * 5).requires_grad_()
    beta = torch.rand(2, 9, 2, requires_grad=True)
    state = torch.randn(2, 2, 4, 3, requires_grad=True)
    inputs = (q, k, v, g, beta, state)
    expected, expected_state = recurrent_kda(*inputs)
    actual, actual_state = chunk_kda(*inputs, chunk_size=size)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-6, rtol=2e-5)
    reference_grad = torch.autograd.grad(expected.square().sum() + expected_state.square().sum(), inputs)
    actual_grad = torch.autograd.grad(actual.square().sum() + actual_state.square().sum(), inputs)
    for a, b in zip(actual_grad, reference_grad, strict=True):
        torch.testing.assert_close(a, b, atol=5e-6, rtol=5e-5)


def test_read_only_tokens_do_not_change_state_or_later_output():
    torch.manual_seed(12)
    q, k, v = [torch.randn(1, 5, 2, 3) for _ in range(3)]
    g, beta = -torch.rand_like(q), torch.rand(1, 5, 2)
    g[:, 2] = 0
    beta[:, 2] = 0
    full, state = chunk_kda(q, k, v, g, beta, chunk_size=4)
    keep = torch.tensor([0, 1, 3, 4])
    short, short_state = chunk_kda(q[:, keep], k[:, keep], v[:, keep], g[:, keep], beta[:, keep])
    torch.testing.assert_close(full[:, keep], short)
    torch.testing.assert_close(state, short_state)


def test_strong_decay_does_not_create_future_overflow():
    torch.manual_seed(20)
    q, k, v = [torch.randn(1, 64, 1, 4, requires_grad=True) for _ in range(3)]
    g = torch.full_like(q, -5)
    beta = torch.full((1, 64, 1), 0.1)
    output, state = chunk_kda(q, k, v, g, beta, chunk_size=64)
    (output.square().mean() + state.square().mean()).backward()
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(x.grad).all() for x in (q, k, v))
