"""Native complete-epoch LoRA perturbation training and shared evaluation.

One process trains one dataset. Pretraining uses DDP; stage-2 condition bags have
variable source availability and do not silently inherit a static DDP graph.
Formal runs require CUDA, completed real pretraining and frozen evaluation IO.
"""

import fcntl
import json
import math
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from dinogenept.datasets.cellfm.dataset import CellFMDataset
from dinogenept.datasets.knowledge import KnowledgeBank
from dinogenept.evaluation.cellfm import load_evaluation
from dinogenept.evaluation.metrics import evaluate_condition, macro_average
from dinogenept.provenance import atomic_write_json, digest_file

from .checkpoint import load_checkpoint, rng_state, save_checkpoint
from .optimization import optimizer_groups
from .perturbation import PerturbationConfig, PerturbationSystem
from .schedule import learning_rate
from .train import _dataclass, _gpu_guard
from .transfer import load_pretrained_student


@dataclass(frozen=True)
class FineTuneOptions:
    epochs: int = 30
    accumulation: int = 4
    seed: int = 42
    device: str = "cuda"
    cuda_index: int = 0
    bag_size: int = 8
    gene_cap: int = 2048
    evaluation_batch: int = 8
    checkpoint_steps: int = 50
    learning_rate: float = 2e-4
    lr_scheduler: str = "dinov2_step_cosine"
    lr_warmup_fraction: float = 0.16
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    gradient_clip: float = 1.0

    def __post_init__(self):
        if self.lr_scheduler not in {"dinov2_step_cosine", "sclong_epoch_restarts", "legacy_step_cosine"}:
            raise ValueError("Unknown learning-rate schedule")
        if (not 0 <= self.lr_warmup_fraction < 1 or not math.isfinite(self.min_learning_rate)
                or self.min_learning_rate < 0):
            raise ValueError("Invalid LR warmup/minimum")
        if (
            min(
                self.epochs,
                self.accumulation,
                self.bag_size,
                self.gene_cap,
                self.evaluation_batch,
                self.checkpoint_steps,
            )
            < 1
        ):
            raise ValueError("Training batch/budget fields must be positive")
        if self.device not in {"cpu", "cuda"} or self.evaluation_batch > 300:
            raise ValueError("Invalid fine-tuning device/evaluation batch")
        if type(self.cuda_index) is not int or self.cuda_index < 0:
            raise ValueError("CUDA index must select a nonnegative visible-list entry or physical NVML index")
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
            or not math.isfinite(self.gradient_clip)
            or self.gradient_clip <= 0
            or len(self.betas) != 2
            or any(not 0 <= beta < 1 for beta in self.betas)
        ):
            raise ValueError("Invalid optimizer settings")


def to_device(value, device):
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(device, non_blocking=True)
    return value


@torch.no_grad()
def evaluate_model(model, data, bank, state, split, *, device, batch_size=8, gene_cap=2048):
    """Predict before reading truth. Keep at most one [300,G] condition in RAM."""
    if split not in {"val", "test"} or batch_size < 1:
        raise ValueError("Invalid evaluation split/batch")
    model.eval()
    lookup = {row: i for i, row in enumerate(data.index.row_ids)}
    result, diagnostics = {}, {}
    started = time.monotonic()
    for condition, row in state["conditions"].items():
        if row["split"] != split:
            continue
        controls = [lookup[r] for r in row["control_row_ids"]]
        prediction = np.empty((300, len(data.axis)), dtype=np.float32)
        full_control = np.empty_like(prediction)
        symbols, _ = data.condition_targets(condition)
        vectors = to_device(bank.for_targets(symbols)[model.config.main_anchor], device)
        for context in sorted(set(row["control_contexts"])):
            positions = [i for i, c in enumerate(row["control_contexts"]) if c == context]
            for start in range(0, len(positions), batch_size):
                selected = positions[start : start + batch_size]
                inputs = data.prediction_inputs(
                    condition, context, [controls[i] for i in selected], cap=gene_cap, seed=state["evaluation_seed"]
                )
                full_control[selected] = inputs["full_control"]
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    values = model.predict(**to_device(inputs, device), main_vectors=vectors)
                if not torch.isfinite(values).all():
                    raise FloatingPointError("Nonfinite prediction; no metric can be published")
                prediction[selected] = values.float().cpu().numpy()
        truth_indices = np.flatnonzero(data.index.conditions == condition)
        truth = data._matrix[truth_indices].toarray()  # evaluator-only, after prediction
        result[condition] = evaluate_condition(
            prediction,
            full_control,
            truth,
            np.asarray(row["metric_control_mean"]),
            np.asarray(state["systema_reference"]),
            row["de_indices"],
            row["top_de_indices"],
            de_unavailable_reason=row["de_unavailable_reason"],
        )
        diagnostics[condition] = {
            "mean_expression_mse": float(
                np.mean((prediction.mean(0, dtype=np.float64) - truth.mean(0, dtype=np.float64)) ** 2)
            ),
            "prediction_rows": 300,
            "truth_rows": len(truth),
        }
    summary = macro_average(result)
    summary.update(split=split, diagnostics=diagnostics, elapsed_seconds=time.monotonic() - started)
    return summary


def _save(path, model, optimizer, config, progress):
    save_checkpoint(path, model, optimizer, config=config, progress=progress, rank_rng=[rng_state()])


def _log(path, event):
    with path.open("a") as handle:
        handle.write(json.dumps(event, allow_nan=False) + "\n")


def run_finetuning(config, *, resume=None):
    config = json.loads(json.dumps(config))
    options = _dataclass(FineTuneOptions, config["training"])
    config["training"] = json.loads(json.dumps(asdict(options)))
    purpose = config.get("purpose", "formal_finetuning")
    if purpose not in {"formal_finetuning", "unit_fixture"}:
        raise ValueError("Unknown fine-tuning purpose")
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise ValueError("This conditional-LoRA runner is one process per dataset, not static-graph DDP")
    if purpose == "formal_finetuning" and (options.device != "cuda" or options.epochs != 30):
        raise ValueError("Formal LoRA campaign requires CUDA and 30 complete epochs")
    if options.device == "cuda":
        if torch.cuda.is_initialized():
            raise RuntimeError("Start each single-GPU fine-tuning job in a fresh process for explicit GPU isolation")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        identifiers = visible.split(",") if visible is not None else None
        if identifiers is not None and options.cuda_index >= len(identifiers):
            raise ValueError("CUDA logical index is outside the visible device list")
        identifier = identifiers[options.cuda_index].strip() if identifiers is not None else str(options.cuda_index)
        if not identifier or identifier == "-1":
            raise ValueError("Requested CUDA device is hidden")
        uuids = (
            subprocess.check_output(
                ["nvidia-smi", "-i", identifier, "--query-gpu=uuid", "--format=csv,noheader"], text=True
            )
            .strip()
            .splitlines()
        )
        if len(uuids) != 1 or not uuids[0].startswith("GPU-"):
            raise ValueError("Cannot resolve the exact requested GPU without initializing CUDA")
        _gpu_guard(selected_uuid=uuids[0])
        # Restrict this fresh process to the audited physical UUID. Never assume
        # CUDA's default enumeration order equals nvidia-smi's device indices.
        os.environ["CUDA_VISIBLE_DEVICES"] = uuids[0]
        config["runtime_gpu_uuid"] = uuids[0]
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    guard = (output / ".training.lock").open("a")
    try:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(config, options, purpose, output, device, resume)
    finally:
        guard.close()


def _run(config, options, purpose, output, device, resume):
    if resume is None and (output / "resolved_config.json").exists():
        raise FileExistsError("Existing run requires explicit resume, never overwrite or restart it")
    data = CellFMDataset(**config["data"])
    pretrained, transfer_identity = load_pretrained_student(
        **config["pretrained"],
        vocabulary=config["data"]["vocabulary"],
        purpose="formal_pretraining" if purpose == "formal_finetuning" else "unit_fixture",
    )
    bank = KnowledgeBank(required_genes=data.symbols, **config["knowledge"])
    state = load_evaluation(data, config["evaluation"]["path"], sha256=config["evaluation"]["sha256"])
    model = PerturbationSystem(pretrained, _dataclass(PerturbationConfig, config.get("perturbation", {}))).to(device)
    if model.config.main_anchor != "TextBase":
        raise ValueError("Only the declared default TextBase run is scheduled, not anchor ablations")
    del pretrained
    if bank.width != model.config.vector_width:
        raise ValueError("Knowledge-bank and model dimensions differ")
    trainable = [p for p in model.student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        optimizer_groups(model.student, options.weight_decay),
        lr=options.learning_rate,
        betas=tuple(options.betas),
        weight_decay=options.weight_decay,
    )
    initial_bags = data.index.training_bags(epoch=0, seed=options.seed, bag_size=options.bag_size)
    bag_count = len(initial_bags)
    post_count = sum(len(b.primary_post) for b in initial_bags)
    total_steps = math.ceil(bag_count / options.accumulation) * options.epochs
    progress = {
        "epoch": 0,
        "next_bag": 0,
        "completed_steps": 0,
        "post_cells_seen": 0,
        "bags_seen": 0,
        "best_validation": None,
        "best_epoch": None,
    }
    if resume is not None:
        progress = load_checkpoint(Path(resume), model, optimizer, config=config)
    atomic_write_json(output / "resolved_config.json", config)
    atomic_write_json(
        output / "input_identity.json",
        {
            "data": data.identity,
            "transfer": transfer_identity,
            "knowledge_coverage": bank.coverage,
            "evaluation_sha256": config["evaluation"]["sha256"],
        },
    )
    parameters = {
        "student_total": sum(p.numel() for p in model.student.parameters()),
        "student_trainable": sum(p.numel() for p in trainable),
        "teacher_trainable": sum(p.numel() for p in model.teacher.parameters() if p.requires_grad),
    }
    if parameters["teacher_trainable"] or any(
        p.requires_grad and not name.endswith((".a", ".b")) for name, p in model.student.backbone.named_parameters()
    ):
        raise RuntimeError("LoRA/frozen-backbone contract failed")
    log = output / "metrics.jsonl"
    _log(
        log,
        {
            "event": "invocation_start",
            "resumed": resume is not None,
            "from_step": progress["completed_steps"],
            "parameters": parameters,
            "torch": str(torch.__version__),
            "device": str(device),
            "hardware": torch.cuda.get_device_name() if device.type == "cuda" else "CPU fixture",
        },
    )
    began = time.monotonic()
    for epoch in range(progress["epoch"], options.epochs):
        bags = data.index.training_bags(epoch=epoch, seed=options.seed, bag_size=options.bag_size)
        if len(bags) != bag_count or sum(len(b.primary_post) for b in bags) != post_count:
            raise RuntimeError("Full-epoch bag/cell coverage changed")
        start = progress["next_bag"]
        if not 0 <= start <= bag_count or (start % options.accumulation and start != bag_count):
            raise ValueError("Resume must be at an optimizer boundary")
        model.train()
        for position in range(start, bag_count, options.accumulation):
            window = bags[position : position + options.accumulation]
            cells = sum(len(b.primary_post) for b in window)
            step = progress["completed_steps"]
            current_lr = learning_rate(
                options.lr_scheduler,
                options.learning_rate,
                epoch=epoch,
                step=step,
                total_steps=total_steps,
                batch=options.bag_size * options.accumulation,
                warmup_fraction=options.lr_warmup_fraction,
                min_lr=options.min_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            stats, data_seconds, tokens = {}, 0.0, 0
            for bag in window:
                before_data = time.monotonic()
                inputs = data.training_inputs(bag, epoch=epoch, seed=options.seed, cap=options.gene_cap)
                symbols, _ = data.condition_targets(bag.condition)
                inputs["source_vectors"] = bank.for_targets(symbols)
                active = sum(v is not None for v in inputs["source_vectors"].values())
                tokens += int(inputs["control_view"]["valid"].sum()) * active
                tokens += int(inputs["observed_view"]["valid"].sum() + inputs["teacher_view"]["valid"].sum())
                inputs = to_device(inputs, device)
                data_seconds += time.monotonic() - before_data
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    result = model(**inputs)
                    weighted = result["total"] * (len(bag.primary_post) / cells)
                if not torch.isfinite(weighted):
                    raise FloatingPointError("Nonfinite loss; optimizer and durable checkpoint not advanced")
                weighted.backward()
                for key, value in result.items():
                    stats[key] = stats.get(key, 0.0) + float(value.detach() if torch.is_tensor(value) else value) * (
                        len(bag.primary_post) / cells
                    )
            grad = torch.nn.utils.clip_grad_norm_(trainable, options.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            momentum = model.update_ema(step + 1, total_steps)
            progress.update(
                epoch=epoch,
                next_bag=position + len(window),
                completed_steps=step + 1,
                post_cells_seen=progress["post_cells_seen"] + cells,
                bags_seen=progress["bags_seen"] + len(window),
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            _log(
                log,
                {
                    "event": "optimizer_step",
                    "epoch": epoch,
                    "step": step + 1,
                    **stats,
                    "weighted_losses": {
                        k: stats[k] * (1 if k == "reconstruction" else 0.1)
                        for k in ("reconstruction", "primary_dino", "knowledge_dino", "observed_dino")
                    },
                    "primary_post_cells": cells,
                    "bags": len(window),
                    "valid_gene_tokens_all_views": tokens,
                    "data_seconds": data_seconds,
                    "step_seconds": elapsed,
                    "post_cells_per_second": cells / elapsed,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": float(grad),
                    "ema_momentum": momentum,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                },
            )
            if (step + 1) % options.checkpoint_steps == 0:
                _save(output / "last.pt", model, optimizer, config, progress)
        validation = evaluate_model(
            model,
            data,
            bank,
            state,
            "val",
            device=device,
            batch_size=options.evaluation_batch,
            gene_cap=options.gene_cap,
        )
        metric = validation["metrics"]["txpert_macro_pearson_delta"]["macro_mean"]
        if metric is None or not math.isfinite(metric):
            raise ValueError("No finite validation selection metric; no test-based fallback")
        improved = progress["best_validation"] is None or metric > progress["best_validation"]
        progress.update(epoch=epoch + 1, next_bag=0)
        if improved:
            progress.update(best_validation=metric, best_epoch=epoch + 1)
            _save(output / "best.pt", model, optimizer, config, progress)
        atomic_write_json(output / f"validation-epoch{epoch + 1:02d}.json", validation)
        _save(output / "last.pt", model, optimizer, config, progress)
        _log(log, {"event": "epoch_complete", **progress, "validation_metric": metric})
    if (
        progress["completed_steps"] != total_steps
        or progress["bags_seen"] != bag_count * options.epochs
        or progress["post_cells_seen"] != post_count * options.epochs
    ):
        raise RuntimeError("Complete-epoch/optimizer accounting failed")
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"], strict=True)
    test = evaluate_model(
        model,
        data,
        bank,
        state,
        "test",
        device=device,
        batch_size=options.evaluation_batch,
        gene_cap=options.gene_cap,
    )
    atomic_write_json(output / "test.json", test)
    atomic_write_json(
        output / "completion.json",
        {
            "status": "completed",
            "purpose": purpose,
            "epochs": options.epochs,
            **progress,
            "training_post_cells": post_count,
            "bags_per_epoch": bag_count,
            "total_steps": total_steps,
            "parameters": parameters,
            "best_checkpoint_sha256": digest_file(output / "best.pt"),
            "last_checkpoint_sha256": digest_file(output / "last.pt"),
            "test_sha256": digest_file(output / "test.json"),
            "evaluation_sha256": config["evaluation"]["sha256"],
            "pretraining": transfer_identity,
            "elapsed_seconds_this_invocation": time.monotonic() - began,
        },
    )
    return progress
