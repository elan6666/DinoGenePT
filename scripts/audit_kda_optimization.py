"""Numerical audit of equivalent native KDA schedules, not a model ablation."""

import argparse
import os
from dataclasses import asdict, replace
from pathlib import Path

import torch

from dinogenept.cell.backbone import BackboneConfig, CellBackbone
from dinogenept.cell.kda import batched_chunk_kda, chunk_kda, parallel_chunk_kda
from dinogenept.cell.train import _gpu_guard
from dinogenept.provenance import atomic_write_json, digest_file


def difference(actual, expected):
    error = (actual.float() - expected.float()).flatten()
    denominator = expected.float().norm().clamp_min(1e-12)
    return {"max_absolute": float(error.abs().max()), "relative_l2": float(error.norm() / denominator)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel", choices=("parallel_chunk", "batched_chunk"), default="parallel_chunk")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--genes", type=int, default=256)
    args = parser.parse_args()
    if args.tokens < 2 or args.genes < args.tokens:
        raise ValueError("Numerical fixture needs distinct genes covering the token axis")
    if args.output.exists():
        raise FileExistsError("Preserve previous numerical evidence")
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if not visible.startswith("GPU-") or "," in visible:
        raise ValueError("Bind this single-card audit to one exact CUDA_VISIBLE_DEVICES GPU UUID")
    _gpu_guard(selected_uuid=visible)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    q, k = [torch.nn.functional.normalize(torch.randn(1, 2049, 6, 128, device=device), dim=-1) for _ in range(2)]
    v = torch.randn_like(q)
    g = -5 * torch.rand_like(q)
    beta = torch.rand(q.shape[:3], device=device)
    g[:, ::5] = 0
    beta[:, ::5] = 0  # read-only slots including the first and chunk-boundary tokens
    values, gradients = [], []
    selected_kernel = {"parallel_chunk": parallel_chunk_kda, "batched_chunk": batched_chunk_kda}[args.kernel]
    for kernel in (chunk_kda, selected_kernel):
        leaves = [x.detach().clone().requires_grad_() for x in (q, k, v, g, beta)]
        output, state = kernel(*leaves)
        gradients.append(
            tuple(x.detach() for x in torch.autograd.grad(output.square().sum() + state.square().sum(), leaves))
        )
        values.append((output.detach(), state.detach()))
    records = []
    for actual, expected in zip((*values[1], *gradients[1]), (*values[0], *gradients[0]), strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        records.append(difference(actual, expected))
    del leaves, output, state, gradients, values, q, k, v, g, beta
    config = BackboneConfig(genes=args.genes)
    torch.manual_seed(7)
    reference = CellBackbone(config).to(device).eval()
    candidate = CellBackbone(replace(config, kda_implementation=args.kernel)).to(device).eval()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    ids = torch.arange(1, args.tokens + 1, device=device).expand(2, -1)
    expression = torch.rand(2, args.tokens, device=device) * 4
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[0, -7:] = False
    hidden = torch.zeros_like(valid)
    hidden[:, ::5] = True
    backbone_records, comparison_errors, fp32_reference = {}, [], None
    for name, enabled, atol, rtol in (("float32", False, 2e-4, 2e-4), ("bfloat16", True, 0.06, 0.03)):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
            expected = reference(ids, expression, valid, hidden)
            actual = candidate(ids, expression, valid, hidden)
        if not enabled:
            fp32_reference = {field: expected[field].detach() for field in ("cls", "genes")}
        backbone_records[name] = {}
        for field in ("cls", "genes"):
            try:
                torch.testing.assert_close(actual[field], expected[field], atol=atol, rtol=rtol)
            except AssertionError as error:
                comparison_errors.append({"precision": name, "field": field, "error": str(error)})
            record = difference(actual[field], expected[field])
            # Separate normalized bound avoids hiding meaningful drift behind
            # elementwise tolerances near zero. BF16 is not bitwise equivalent.
            if record["relative_l2"] > (2e-4 if name == "float32" else 0.02):
                comparison_errors.append({"precision": name, "field": field, "error": "relative L2 bound exceeded"})
            if enabled:
                record["original_bf16_vs_original_fp32"] = difference(expected[field], fp32_reference[field])
                record["candidate_bf16_vs_original_fp32"] = difference(actual[field], fp32_reference[field])
            backbone_records[name][field] = record
    atomic_write_json(
        args.output,
        {
            "status": "failed" if comparison_errors else "passed",
            "comparison_errors": comparison_errors,
            "purpose": "numerical_fixture_not_training",
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(),
            "backbone": asdict(config),
            "candidate_kernel": args.kernel,
            "backbone_tokens": args.tokens,
            "long_kernel_shape": [1, 2049, 6, 128],
            "read_only_every": 5,
            "kernel_fp32_atol": 2e-5,
            "kernel_fp32_rtol": 2e-4,
            "kernel_values_and_gradients": records,
            "backbone_comparison": backbone_records,
            "backbone_elementwise_tolerances": {"float32": [2e-4, 2e-4], "bfloat16": [0.06, 0.03]},
            "backbone_relative_l2_limits": {"float32": 2e-4, "bfloat16": 0.02},
            "source_sha256": {
                name: digest_file(Path(name))
                for name in (
                    "scripts/audit_kda_optimization.py",
                    "src/dinogenept/cell/kda.py",
                    "src/dinogenept/cell/backbone.py",
                )
            },
        },
    )
    print(args.output)
    if comparison_errors:
        raise SystemExit("Numerical acceptance gate failed; original backend remains default. See frozen receipt.")


if __name__ == "__main__":
    main()
