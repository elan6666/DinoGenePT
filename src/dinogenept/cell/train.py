"""Native pretraining runner: complete epochs, DDP, resume and scalar receipts."""

import contextlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from dinogenept.provenance import atomic_write_json, digest_file

from .backbone import BackboneConfig
from .checkpoint import load_checkpoint, rng_state, save_checkpoint
from .dataset import PretrainingDataset
from .pretraining import HeadConfig, PretrainingSystem
from .sampling import CropConfig, collate_crops, epoch_batches
from .schedule import learning_rate


def _dataclass(cls, values):
    unknown = set(values) - {field.name for field in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def _same_torchrun_launcher(pid):
    """Elastic ranks have separate sessions; only a verified launcher owns peers.

    Do not whitelist siblings of an arbitrary shell/sshd or all user's jobs.
    Read only process identity, never credential-bearing process environments.
    """
    if not os.environ.get("TORCHELASTIC_RUN_ID") or int(os.getenv("LOCAL_WORLD_SIZE", "1")) <= 1:
        return False
    parent = os.getppid()
    try:
        arguments = (Path("/proc") / str(parent) / "cmdline").read_bytes().split(b"\0")
        named_launcher = any(Path(os.fsdecode(arg)).name == "torchrun" for arg in arguments if arg)
        module_launcher = any(
            arguments[i : i + 2] == [b"-m", b"torch.distributed.run"] for i in range(len(arguments) - 1)
        )
        if not (named_launcher or module_launcher):
            return False
        # Linux stat's comm may contain spaces or parentheses; split after its
        # final close parenthesis. Following fields are state, ppid, pgrp, ...
        stat = (Path("/proc") / str(pid) / "stat").read_text()
        fields = stat[stat.rfind(")") + 2 :].split()
        return len(fields) >= 2 and int(fields[1]) == parent
    except (OSError, ValueError):
        return False


def _authorized_shared_process(pid):
    """Opt-in sharing binds PID and Linux start ticks; PID reuse fails closed."""
    token = os.getenv("DINOGENEPT_SHARED_PROCESS", "")
    if not token:
        return False
    try:
        allowed_pid, start_ticks = map(int, token.split(":"))
        if pid != allowed_pid or allowed_pid <= 0 or start_ticks <= 0:
            return False
        stat = (Path("/proc") / str(pid) / "stat").read_text()
        return int(stat[stat.rfind(")") + 2:].split()[19]) == start_ticks
    except (OSError, ValueError, IndexError):
        return False


def _gpu_guard(*, selected_uuid=None):
    """Refuse unrelated compute; accept our group or verified torchrun peers."""
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True
    )
    occupied = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 2 or not fields[1].isdigit():
            raise RuntimeError("GPU ownership query returned an unrecognized row; allocation refused")
        if selected_uuid is not None and fields[0] != selected_uuid:
            continue
        if fields[1].isdigit():
            pid = int(fields[1])
            try:
                if (os.getpgid(pid) != os.getpgid(0) and not _same_torchrun_launcher(pid)
                        and not _authorized_shared_process(pid)):
                    occupied.append(pid)
            except ProcessLookupError:
                continue
            except PermissionError:
                occupied.append(pid)
    if occupied:
        raise RuntimeError(f"GPU allocation refused: unrelated active compute processes {occupied}")


def _move(batch, device):
    return {
        "cell_ids": batch["cell_ids"],
        "views": [
            {key: tensor.to(device, non_blocking=True) for key, tensor in view.items()} for view in batch["views"]
        ],
    }


def _reduce(values, device):
    keys = sorted(values)
    tensor = torch.tensor([values[k] for k in keys], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return dict(zip(keys, tensor.cpu().tolist(), strict=True))


def _checkpoint(path, model, optimizer, config, progress, rank, world):
    states = [None] * world if rank == 0 else None
    local = rng_state()
    if world > 1:
        dist.gather_object(local, states, dst=0)
    else:
        states = [local]
    if rank == 0:
        save_checkpoint(path, model, optimizer, config=config, progress=progress, rank_rng=states)
    if world > 1:
        dist.barrier()


def run_pretraining(config: dict, *, resume: Path | None = None, smoke_one_step: bool = False):
    """Config is resolved JSON; a unit_fixture run can never claim formal completion."""
    config = json.loads(json.dumps(config))
    if config.get("execution_mode", "formal") != "formal":
        raise ValueError("Smoke resolved configs cannot launch or resume formal training")
    if smoke_one_step and resume is not None:
        raise ValueError("One-step smoke requires a fresh run, not a resume checkpoint")
    if smoke_one_step:
        config["execution_mode"] = "smoke_one_step"
    rank, world, local_rank = (
        int(os.getenv("RANK", "0")),
        int(os.getenv("WORLD_SIZE", "1")),
        int(os.getenv("LOCAL_RANK", "0")),
    )
    training = config["training"]
    training.setdefault("lr_scheduler", "sclong_epoch_restarts")
    if world != training["world_size"] or training["epochs"] < 1 or training["accumulation"] < 1:
        raise ValueError("World size/epoch/accumulation mismatch")
    if training["device"] not in {"cpu", "cuda"} or training["checkpoint_steps"] < 1:
        raise ValueError("Invalid device or checkpoint interval")
    if training["microbatch"] < 2 or training.get("workers", 0) < 0:
        raise ValueError("KoLeo requires microbatch >=2 and nonnegative worker count")
    purpose = config.get("purpose", "formal_pretraining")
    if purpose not in {"formal_pretraining", "unit_fixture"}:
        raise ValueError("Unknown run purpose")
    device = torch.device(f"cuda:{local_rank}" if training["device"] == "cuda" else "cpu")
    if purpose == "formal_pretraining" and device.type != "cuda":
        raise ValueError("Formal training requires the audited server GPUs")
    published_protocol = config.get("protocol") == "genecompass_all_cells_one_epoch_no_validation"
    if published_protocol and training["epochs"] != 1:
        raise ValueError("Published all-cells protocol requires exactly one epoch")
    if purpose == "formal_pretraining" and not published_protocol and training["epochs"] != 30:
        raise ValueError("Formal pretraining configuration requires 30 complete epochs")
    if device.type == "cuda":
        _gpu_guard()
        torch.cuda.set_device(device)
        if os.getenv("DINOGENEPT_SHARED_PROCESS"):
            # Cap our allocator only; NCCL/context memory is additional. Leave
            # room for the authorized co-tenant, without touching its process.
            torch.cuda.set_per_process_memory_fraction(0.75, device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    seed = int(training["seed"])
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    crops = _dataclass(CropConfig, config["crops"])
    manifest = Path(config["data_manifest"])
    if published_protocol:
        from .genecompass_data import GeneCompassDataset

        dataset = GeneCompassDataset(manifest, crops)
        expected_cells = config.get("published_cells", 500000)
        if expected_cells not in (50000, 500000):
            raise ValueError("Published protocol supports explicitly selected50k/500k only")
        if purpose == "formal_pretraining" and len(dataset) != expected_cells:
            raise ValueError(f"Published protocol requires all{expected_cells} cells")
        validation = None
    else:
        dataset = PretrainingDataset(manifest, "train", crops)
        validation = PretrainingDataset(manifest, "validation", crops)
    if dataset.manifest.get("purpose") != purpose:
        raise ValueError("Data purpose and run purpose differ")
    actual_hash = digest_file(manifest)
    if actual_hash != config["data_manifest_sha256"]:
        raise ValueError("Pinned dataset manifest hash differs")
    # Every rank validates independently so a failure never strands peers at a barrier.
    dataset.verify()
    if config["backbone"]["genes"] != dataset.gene_count:
        raise ValueError("Backbone vocabulary differs from corpus")
    model = PretrainingSystem(
        _dataclass(BackboneConfig, config["backbone"]), _dataclass(HeadConfig, config["heads"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.student.parameters(),
        lr=training["learning_rate"],
        betas=tuple(training["betas"]),
        weight_decay=training["weight_decay"],
    )
    output = Path(config["output"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if resume is None and (output / "resolved_config.json").exists():
            raise FileExistsError("Existing run requires an explicit resume checkpoint or a new output directory")
        atomic_write_json(output / "resolved_config.json", config)
    if world > 1:
        dist.barrier()
    batches_per_epoch = len(list(epoch_batches(len(dataset), training["microbatch"], world, rank, seed=seed, epoch=0)))
    steps_per_epoch = math.ceil(batches_per_epoch / training["accumulation"])
    total_steps = training["epochs"] * steps_per_epoch
    progress = {"epoch": 0, "next_batch": 0, "completed_steps": 0, "cells_seen": 0, "best_validation": None}
    if resume is not None:
        progress = load_checkpoint(resume, model, optimizer, config=config, rank=rank)
    wrapped = (
        DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None, broadcast_buffers=False
        )
        if world > 1
        else model
    )
    # DDP broadcasts the Student, but frozen Teacher synchronization is explicit.
    if world > 1 and resume is None:
        model.teacher.load_state_dict(model.student.state_dict())
    autocast_dtype = torch.bfloat16
    train_start = time.monotonic()
    for epoch in range(progress["epoch"], training["epochs"]):
        dataset.epoch = epoch
        batches = list(epoch_batches(len(dataset), training["microbatch"], world, rank, seed=seed, epoch=epoch))
        start_batch = progress["next_batch"] if epoch == progress["epoch"] else 0
        if start_batch % training["accumulation"] and start_batch != len(batches):
            raise ValueError("Resume must be at an optimizer boundary")
        global_lengths = [len(dataset) // len(batches) + (i < len(dataset) % len(batches)) for i in range(len(batches))]
        loader = DataLoader(
            dataset,
            batch_sampler=batches[start_batch:],
            collate_fn=collate_crops,
            num_workers=training.get("workers", 0),
            # NCCL is not fork-safe. Workers reopen mmap shards through the
            # dataset's explicit pickle contract, never inherit CUDA contexts.
            multiprocessing_context="spawn" if training.get("workers", 0) else None,
            pin_memory=device.type == "cuda",
            # Loader iterator creation must not consume model RNG on resume.
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        iterator = iter(loader)
        model.train()
        for window_start in range(start_batch, len(batches), training["accumulation"]):
            window_end = min(len(batches), window_start + training["accumulation"])
            window_cells = sum(global_lengths[window_start:window_end])
            optimizer.zero_grad(set_to_none=True)
            stats = dict.fromkeys(
                ["expression", "cell_expression", "dino", "ibot", "koleo", "total", "cells", "tokens", "data_seconds"],
                0.0,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            step = progress["completed_steps"]
            current_lr = learning_rate(
                training["lr_scheduler"],
                training["learning_rate"],
                epoch=epoch,
                step=step,
                total_steps=total_steps,
                batch=training["microbatch"] * world * training["accumulation"],
                warmup_fraction=training.get("lr_warmup_fraction", 0.16),
                min_lr=training.get("min_learning_rate", 1e-6),
            )
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            for index in range(window_start, window_end):
                before_data = time.monotonic()
                batch = _move(next(iterator), device)
                stats["data_seconds"] += time.monotonic() - before_data
                n = len(batch["cell_ids"])
                synchronize = contextlib.nullcontext() if world == 1 or index == window_end - 1 else wrapped.no_sync()
                with synchronize:
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
                        losses = wrapped(batch, step=step, total_steps=total_steps)
                        scaled_loss = losses["total"] * (world * n / window_cells)
                    if not torch.isfinite(scaled_loss):
                        raise FloatingPointError("Nonfinite pretraining loss; refusing to step or overwrite checkpoint")
                    scaled_loss.backward()
                for key, value in losses.items():
                    stats[key] += float(value.detach()) * n
                stats["cells"] += n
                stats["tokens"] += sum(int(view["valid"].sum()) for view in batch["views"])
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.student.parameters(), training["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            progress.update(
                epoch=epoch,
                next_batch=window_end,
                completed_steps=step + 1,
                cells_seen=progress["cells_seen"] + window_cells,
            )
            momentum = model.update_ema(step + 1, total_steps)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            stats = _reduce(stats, device)
            performance = torch.tensor(
                [elapsed, torch.cuda.max_memory_allocated() if device.type == "cuda" else 0],
                dtype=torch.float64,
                device=device,
            )
            if world > 1:
                dist.all_reduce(performance, op=dist.ReduceOp.MAX)
            elapsed, peak_memory = performance.cpu().tolist()
            if rank == 0:
                ramp = min(1.0, step / max(1, math.ceil(total_steps * 0.1)))
                record = {
                    "event": "optimizer_step",
                    "epoch": epoch,
                    "step": step + 1,
                    **{key: stats[key] / stats["cells"] for key in losses},
                    "cells": int(stats["cells"]),
                    "student_gene_tokens": int(stats["tokens"]),
                    "data_seconds_rank_mean": stats["data_seconds"] / world,
                    "step_seconds": elapsed,
                    "cells_per_second": stats["cells"] / elapsed,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "ema_momentum": momentum,
                    "distillation_weight": ramp,
                    "weighted_losses": {
                        key: stats[key] / stats["cells"] * weight
                        for key, weight in {
                            "expression": 1,
                            "cell_expression": 1,
                            "dino": ramp,
                            "ibot": ramp,
                            "koleo": 0.1 * ramp,
                        }.items()
                    },
                    "gradient_norm": float(gradient_norm),
                    "peak_allocated_bytes_max_rank": int(peak_memory),
                }
                with (output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
            if smoke_one_step:
                _checkpoint(output / "smoke.pt", model, optimizer, config, progress, rank, world)
                if rank == 0:
                    atomic_write_json(
                        output / "smoke.json",
                        {
                            "status": "smoke_completed",
                            "purpose": "smoke_one_step",
                            "data_purpose": purpose,
                            "formal_transfer_eligible": False,
                            "full_epoch_completed": False,
                            "data_manifest_sha256": actual_hash,
                            "checkpoint_sha256": digest_file(output / "smoke.pt"),
                            "world_size": world,
                            **progress,
                        },
                    )
                if world > 1:
                    dist.barrier()
                    dist.destroy_process_group()
                return progress
            if (step + 1) % training["checkpoint_steps"] == 0:
                _checkpoint(output / "last.pt", model, optimizer, config, progress, rank, world)
        # Epoch validation never updates centers/teacher and uses held-out donors.
        model.eval()
        if validation is not None:
            validation.epoch = 0
        val_batches = (list(epoch_batches(len(validation), training["microbatch"], world, rank, seed=seed, epoch=0))
                       if validation is not None else [])
        val_loader = DataLoader(
            validation if validation is not None else [],
            batch_sampler=val_batches,
            collate_fn=collate_crops,
            num_workers=0,
            generator=torch.Generator().manual_seed(seed),
        )
        val_stats = {"expression": 0.0, "cell_expression": 0.0, "cells": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                batch = _move(batch, device)
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
                    result = model(
                        batch, step=min(progress["completed_steps"], total_steps - 1), total_steps=total_steps
                    )
                n = len(batch["cell_ids"])
                val_stats["cells"] += n
                for key in ("expression", "cell_expression"):
                    val_stats[key] += float(result[key]) * n
        val_stats = _reduce(val_stats, device)
        metric = ((val_stats["expression"] + val_stats["cell_expression"]) / val_stats["cells"]
                  if validation is not None else None)
        improved = metric is not None and (progress["best_validation"] is None or metric < progress["best_validation"])
        progress.update(epoch=epoch + 1, next_batch=0)
        if improved:
            progress["best_validation"] = metric
        _checkpoint(output / "last.pt", model, optimizer, config, progress, rank, world)
        if improved:
            _checkpoint(output / "best.pt", model, optimizer, config, progress, rank, world)
        if rank == 0:
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "event": "epoch_complete",
                            "epoch": epoch + 1,
                            "validation_reconstruction": metric,
                            "cells_seen": progress["cells_seen"],
                        }
                    )
                    + "\n"
                )
    if progress["cells_seen"] != len(dataset) * training["epochs"] or progress["completed_steps"] != total_steps:
        raise RuntimeError("Full-epoch coverage/step audit failed")
    if rank == 0:
        receipt = {
            "status": "completed",
            "purpose": purpose,
            "epochs": training["epochs"],
            "training_cells": len(dataset),
            "validation_cells": len(validation) if validation is not None else 0,
            "protocol": config.get("protocol", "heldout_validation"),
            "downstream_overlap": dataset.manifest.get("downstream_overlap", "see_leakage_audit"),
            **progress,
            "total_steps": total_steps,
            "elapsed_seconds_this_invocation": time.monotonic() - train_start,
            "data_manifest_sha256": actual_hash,
            "student_parameters": sum(p.numel() for p in model.student.parameters()),
            "world_size": world,
            "device": str(device),
            "torch": str(torch.__version__),
        }
        atomic_write_json(output / "completion.json", receipt)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return progress
