"""Full-default CUDA capacity probe on a verified real shard, never an epoch.

Run with torchrun --nproc_per_node=2 for DDP, or python for a single free card.
Upper crop bounds deliberately stress the declared 2048-token cap. No formal
checkpoint is emitted and partial corpus readiness is not asserted.
"""

import argparse
import fcntl
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from dinogenept.cell.backbone import BackboneConfig
from dinogenept.cell.kda import batched_chunk_kda, chunk_kda, parallel_chunk_kda, recurrent_kda
from dinogenept.cell.pretraining import HeadConfig, PretrainingSystem
from dinogenept.cell.sampling import CropConfig, collate_crops, sample_crops
from dinogenept.cell.train import _gpu_guard, _move
from dinogenept.provenance import atomic_write_json, digest_file


def kernel_parity(device, implementation=chunk_kda):
    """FP32 CUDA recurrent/chunk values and derivatives on a small gate fixture."""
    generator = torch.Generator(device=device).manual_seed(42)
    q = torch.nn.functional.normalize(torch.randn(2, 33, 2, 16, device=device, generator=generator), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(2, 33, 2, 16, device=device, generator=generator), dim=-1)
    inputs = [
        q,
        k,
        torch.randn(2, 33, 2, 16, device=device, generator=generator),
        -torch.rand(2, 33, 2, 16, device=device, generator=generator),
        torch.rand(2, 33, 2, device=device, generator=generator),
    ]
    values, derivatives = [], []
    for kernel in (recurrent_kda, implementation):
        leaves = [item.detach().clone().requires_grad_() for item in inputs]
        output, state = kernel(*leaves)
        derivatives.append(torch.autograd.grad(output.square().sum() + state.square().sum(), leaves))
        values.append((output.detach(), state.detach()))
    errors = []
    for left, right in zip((*values[0], *derivatives[0]), (*values[1], *derivatives[1]), strict=True):
        torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-4)
        errors.append(float((left - right).abs().max()))
    return {"status": "passed", "max_absolute_errors": errors, "atol": 2e-5, "rtol": 2e-4}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--kernel", choices=("chunk", "parallel_chunk", "batched_chunk"), default="chunk")
    args = parser.parse_args()
    if args.microbatch < 2 or args.steps < 1:
        raise ValueError("Need >=2 distinct cells per rank and >=1 step")
    rank, world, local_rank = (
        int(os.getenv(key, default)) for key, default in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0"))
    )
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
    selected_uuid = visible if world == 1 and visible.startswith("GPU-") and "," not in visible else None
    _gpu_guard(selected_uuid=selected_uuid)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    guard = None
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        guard = (args.output / ".probe.lock").open("a")
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / "started.json").exists():
            raise FileExistsError("Use a fresh probe output; do not overwrite evidence")
    receipt = json.loads(args.shard.read_text())
    # Shard receipts contain the exact writer entry plus row/source audit data.
    entry = receipt["shard"]
    arrays = {}
    for key, item in entry["arrays"].items():
        path = args.shard.parent / item["path"]
        if digest_file(path) != item["sha256"]:
            raise ValueError("Real shard array changed")
        arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
    vocabulary = args.shard.parent / "genes.json"
    genes = json.loads(vocabulary.read_text())
    selected = np.argsort(-np.diff(arrays["indptr"]), kind="stable")[: args.microbatch * world]
    if len(selected) != args.microbatch * world:
        raise ValueError("Not enough distinct real cells for the probe")
    crops = CropConfig(global_scale=(1.0, 1.0), local_scale=(0.4, 0.4))
    rows = []
    for row in selected[rank::world]:
        start, stop = arrays["indptr"][row : row + 2]
        rows.append(
            sample_crops(
                arrays["indices"][start:stop] + 1,
                arrays["data"][start:stop],
                float(arrays["library_sum"][row]),
                cell_id=str(arrays["row_ids"][row]),
                epoch=0,
                config=crops,
            )
        )
    batch = _move(collate_crops(rows), device)
    parity = kernel_parity(
        device,
        {"chunk": chunk_kda, "parallel_chunk": parallel_chunk_kda, "batched_chunk": batched_chunk_kda}[args.kernel],
    )
    torch.manual_seed(42 + rank)
    config, heads = BackboneConfig(genes=len(genes), kda_implementation=args.kernel), HeadConfig()
    model = PretrainingSystem(config, heads).to(device).train()
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else model
    if world > 1:
        model.teacher.load_state_dict(model.student.state_dict())
    optimizer = torch.optim.AdamW(model.student.parameters(), lr=1e-5)
    identity = {
        "purpose": "capacity_probe_not_formal_training",
        "backbone": asdict(config),
        "heads": asdict(heads),
        "crops": asdict(crops),
        "world_size": world,
        "microbatch": args.microbatch,
        "shard_sha256": digest_file(args.shard),
        "vocabulary_sha256": digest_file(vocabulary),
        "precision": "autocast_bfloat16_float32_parameters_and_kda_state",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "kernel_parity": parity,
        "student_parameters": sum(p.numel() for p in model.student.parameters()),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "view_lengths": [view["gene_ids"].shape[1] for view in batch["views"]],
        "source_sha256": {
            str(path): digest_file(path) for path in [Path(__file__), *sorted(Path("src/dinogenept/cell").glob("*.py"))]
        },
    }
    if rank == 0:
        atomic_write_json(args.output / "started.json", identity)
        print(json.dumps({"event": "started", **identity}), flush=True)
    records = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = wrapped(batch, step=10 + step, total_steps=100 + args.steps)
        if not all(bool(torch.isfinite(value)) for value in losses.values()):
            raise FloatingPointError("Nonfinite probe loss")
        losses["total"].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.student.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        model.update_ema(11 + step, 100 + args.steps)
        torch.cuda.synchronize()
        numbers = torch.tensor(
            [time.monotonic() - started, torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
            device=device,
            dtype=torch.float64,
        )
        if world > 1:
            dist.all_reduce(numbers, op=dist.ReduceOp.MAX)
        seconds, allocated, reserved = numbers.tolist()
        record = {
            "step": step + 1,
            "seconds": seconds,
            "peak_allocated_bytes": int(allocated),
            "peak_reserved_bytes": int(reserved),
            "cells_per_second": args.microbatch * world / seconds,
            "gradient_norm_rank0": float(norm),
            "losses_rank0": {key: float(value.detach()) for key, value in losses.items()},
        }
        records.append(record)
        if rank == 0:
            print(json.dumps(record), flush=True)
            atomic_write_json(args.output / "progress.json", {"identity": identity, "steps": records})
    if rank == 0:
        atomic_write_json(
            args.output / "completion.json", {"status": "capacity_passed", "identity": identity, "steps": records}
        )
    if world > 1:
        dist.destroy_process_group()
    if guard is not None:
        guard.close()


if __name__ == "__main__":
    main()
