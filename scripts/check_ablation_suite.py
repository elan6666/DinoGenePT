#!/usr/bin/env python3
"""Server-only synthetic smoke/paired memory probes; never formal training.

--list/--resolve is read-only configuration inspection suitable on a laptop.
All reports clearly identify synthetic inputs and physical test dimensions.
"""

import argparse
import gc
import json
import os
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dinogenept.cell.ablations import ablation_registry, resolve_ablation
from dinogenept.cell.backbone import BackboneConfig
from dinogenept.cell.muon import apply_qk_clip, build_optimizer
from dinogenept.cell.pretraining import HeadConfig, PretrainingSystem
from dinogenept.cell.sampling import CropConfig, collate_crops, sample_crops
from dinogenept.cell.schedule import learning_rate
from dinogenept.cell.train import _gpu_guard, _move
from dinogenept.provenance import atomic_write_json


def tiny_config():
    return dict(backbone=asdict(BackboneConfig(
        genes=96, width=16, depth=4, kda_heads=1, kda_head_dim=8, mla_heads=2, mla_head_dim=8,
        mla_shared_dim=2, query_rank=8, kv_rank=8, expression_basis=16, kda_implementation="batched_chunk")),
        heads=asdict(HeadConfig(hidden=32, bottleneck=8, cell_prototypes=32, gene_prototypes=24)),
        crops=asdict(CropConfig(cap=24, hvg_gene_ids=tuple(range(1, 97)))),
        training=dict(optimizer="adamw", learning_rate=2e-4, betas=[.9, .95], weight_decay=.01,
                      lr_scheduler="dinov2_step_cosine", microbatch=2, world_size=1, accumulation=1))


def smoke(name, device, world, rank):
    cfg = resolve_ablation(tiny_config(), name)
    # Scale only the size sweeps; retain and disclose the requested axes.
    if name.startswith("S-WIDTH-"):
        cfg["backbone"]["width"] //= 16
        cfg["backbone"]["kda_heads"] = max(1, cfg["backbone"]["width"] // 8)
        cfg["backbone"]["mla_heads"] = max(1, cfg["backbone"]["width"] // 8)
    cfg["training"].update(microbatch=2, world_size=world, accumulation=1)
    if name in {"E-IBOT-256", "E-IBOT-512", "E-IBOT-1024"}:
        cfg["heads"]["ibot_chunk_size"] = {"E-IBOT-256": 1, "E-IBOT-512": 2, "E-IBOT-1024": 3}[name]
    torch.manual_seed(42)
    model = PretrainingSystem(BackboneConfig(**cfg["backbone"]), HeadConfig(**cfg["heads"])).to(device)
    optimizer = build_optimizer(model.student, cfg["training"])
    wrapped = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                  broadcast_buffers=False) if world > 1 else model
    crop = CropConfig(**cfg["crops"])
    batch = _move(collate_crops([sample_crops(np.arange(1, 97), np.arange(1, 97), 5000,
                                           cell_id=f"r{rank}-c{i}", epoch=0, config=crop) for i in range(2)]), device)
    if name.startswith("E-IBOT-") and world > 1 and rank == 1:
        for view in batch["views"][:2]:
            view["hidden"].zero_()
            view["targets"].zero_()
    for step in (2, 3):
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(cfg["training"]["lr_scheduler"], cfg["training"]["learning_rate"],
                           epoch=0, step=step, total_steps=10, batch=2 * world,
                           warmup_fraction=cfg["training"].get("lr_warmup_fraction"),
                           min_lr=cfg["training"].get("min_learning_rate"))
        for group in optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            losses = wrapped(batch, step=step, total_steps=10)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.student.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        apply_qk_clip(model.student)
        model.update_ema(step + 1, 10)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError(name)
    if world > 1:
        for parameter in model.student.parameters():
            reference = parameter.detach().clone()
            dist.broadcast(reference, 0)
            torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
    return dict(id=name, status="passed", physical_config=cfg, steps=2,
                losses={k: float(v.detach()) for k, v in losses.items()},
                student_parameters=sum(p.numel() for p in model.student.parameters()))


def memory_probe(device, tokens, steps, warmup):
    from dinogenept.cell.distillation import ProjectionHead
    from dinogenept.cell.ibot_chunk import projected_ibot

    if device.type != "cuda":
        raise ValueError("Memory probe requires CUDA")
    records = []
    for repeat in range(3):
        chunks = [0, 256, 512, 1024]
        chunks = chunks[repeat:] + chunks[:repeat]
        for chunk in chunks:
            gc.collect()
            torch.cuda.empty_cache()
            torch.manual_seed(16)
            student = ProjectionHead(256, 8192, 1024, 128).to(device)
            teacher = ProjectionHead(256, 8192, 1024, 128).to(device).requires_grad_(False)
            teacher.load_state_dict(student.state_dict())
            x = torch.randn(tokens, 256, device=device, requires_grad=True)
            target = torch.randn_like(x)
            center = x.new_zeros(8192)
            weights = x.new_full((tokens,), 1 / tokens)
            durations, allocated, reserved = [], [], []
            for step in range(warmup + steps):
                student.zero_grad(set_to_none=True)
                x.grad = None
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _, _ = projected_ibot(student, teacher, x, target, weights, center, .07, .1, chunk)
                loss.backward()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if step >= warmup:
                    durations.append(elapsed)
                    allocated.append(torch.cuda.max_memory_allocated())
                    reserved.append(torch.cuda.max_memory_reserved())
            records.append(dict(chunk=chunk, repeat=repeat, median_seconds=statistics.median(durations),
                                p95_seconds=float(np.quantile(durations, .95)), allocated_bytes=max(allocated),
                                reserved_bytes=max(reserved), loss=float(loss.detach())))
            del loss, student, teacher, x, target, center, weights
    return dict(scope="head+loss-only BF16 synthetic; NOT full-model speed or batch108 capacity",
                tokens=tokens, width=256, prototypes=8192, warmup=warmup, steps=steps, records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--resolve")
    parser.add_argument("--smoke", nargs="*", default=None)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.list:
        print(json.dumps(ablation_registry(), indent=2))
        return
    if args.resolve:
        if args.recipe is None:
            parser.error("--resolve requires --recipe")
        print(json.dumps(resolve_ablation(json.loads(args.recipe.read_text()), args.resolve), indent=2))
        return
    if not args.memory and args.smoke is None:
        parser.error("Choose --list, --resolve, --smoke or --memory")
    if args.tokens < 1 or args.steps < 1 or args.warmup < 0:
        parser.error("Invalid test budget")
    if args.output is None or args.output.exists():
        parser.error("A new --output receipt is required")
    world, rank = int(os.getenv("WORLD_SIZE", "1")), int(os.getenv("RANK", "0"))
    if args.memory and world != 1:
        parser.error("Head memory probe is single-rank; use --smoke for DDP")
    device = torch.device(f"cuda:{os.getenv('LOCAL_RANK', '0')}" if args.device == "cuda" else "cpu")
    torch.set_num_threads(1)
    if device.type == "cuda":
        _gpu_guard()
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        if args.memory:
            result = memory_probe(device, args.tokens, args.steps, args.warmup)
        else:
            names = args.smoke or [n for n in ablation_registry() if not n.startswith(("K-", "F"))]
            if any(n.startswith(("K-", "F")) for n in names):
                parser.error("Knowledge variants use perturbation tests, not pretraining smoke")
            result = []
            for name in names:
                result.append(smoke(name, device, world, rank))
                if rank == 0:
                    print(f"PASS {name}", flush=True)
        if rank == 0:
            atomic_write_json(args.output, dict(purpose="synthetic_smoke_only", torch=torch.__version__,
                                               device=str(device), world_size=world, result=result))
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
