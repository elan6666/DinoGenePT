"""Condition-bag training loop for DinoGenePT."""

from __future__ import annotations

import hashlib
import math
import os
import random
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dinogenept.datasets import PerturbationData
from dinogenept.priors import PriorStore


def _torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("training requires `pip install -e '.[train]'`") from error
    return torch


def set_reproducible_seed(seed: int) -> None:
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class ConditionBatch:
    conditions: tuple[str, ...]
    targets: tuple[tuple[str, ...], ...]
    controls: np.ndarray
    posts: np.ndarray


class ConditionBagSampler:
    def __init__(
        self,
        data: PerturbationData,
        *,
        split: str,
        bag_size: int,
        conditions_per_batch: int,
        seed: int,
    ) -> None:
        self.data = data
        self.conditions = data.split_conditions(split)
        self.bag_size = bag_size
        self.conditions_per_batch = conditions_per_batch
        self.seed = seed
        self.control_indices = np.flatnonzero(data.control_mask)
        if not self.conditions:
            raise ValueError(f"dataset has no {split} conditions")
        if not len(self.control_indices):
            raise ValueError("dataset has no training control pool")
        first_target: dict[str, tuple[str, ...]] = {}
        for condition, targets in zip(data.conditions, data.targets, strict=True):
            if condition != "ctrl":
                first_target.setdefault(condition, targets)
        self.targets = first_target

    def batches(self, epoch: int, *, shuffle: bool) -> Iterator[ConditionBatch]:
        rng = np.random.default_rng(self.seed + epoch * 1009)
        conditions = np.asarray(self.conditions, dtype=str)
        if shuffle:
            rng.shuffle(conditions)
        for start in range(0, len(conditions), self.conditions_per_batch):
            selected = tuple(str(item) for item in conditions[start : start + self.conditions_per_batch])
            controls = []
            posts = []
            targets = []
            for condition in selected:
                post_indices = self.data.indices_for_condition(condition)
                controls.append(
                    self.data.expression[
                        rng.choice(self.control_indices, self.bag_size, replace=True)
                    ]
                )
                posts.append(
                    self.data.expression[
                        rng.choice(post_indices, self.bag_size, replace=len(post_indices) < self.bag_size)
                    ]
                )
                targets.append(self.targets[condition])
            yield ConditionBatch(
                conditions=selected,
                targets=tuple(targets),
                controls=np.stack(controls).astype(np.float32),
                posts=np.stack(posts).astype(np.float32),
            )


def _repeat_targets(targets: tuple[tuple[str, ...], ...], repeats: int) -> tuple[tuple[str, ...], ...]:
    return tuple(target for target in targets for _ in range(repeats))


def _source_for_condition(
    priors: PriorStore,
    condition: str,
    targets: tuple[str, ...],
    epoch: int,
    allowed_sources: tuple[str, ...],
) -> str | None:
    allowed = set(allowed_sources)
    available = tuple(
        source for source in priors.available_optional(targets) if source in allowed
    )
    if not available:
        return None
    offset = int.from_bytes(hashlib.sha256(condition.encode()).digest()[:4], "little")
    return available[(offset + epoch) % len(available)]


def _tensor_prior(
    priors: PriorStore,
    source: str,
    targets: tuple[tuple[str, ...], ...],
    *,
    device: Any,
    control: str,
    seed: int,
):
    torch = _torch()
    vectors, token_mask, condition_mask = priors.batch(
        source, targets, control=control, seed=seed
    )
    return (
        torch.as_tensor(vectors, device=device),
        torch.as_tensor(token_mask, device=device),
        torch.as_tensor(condition_mask, device=device),
    )


def _mask_expression(shape: tuple[int, int], config: dict[str, Any], device: Any):
    torch = _torch()
    probability = float(config.get("sample_probability", 0.5))
    low, high = (float(value) for value in config.get("ratio", [0.1, 0.5]))
    selected = torch.rand(shape[0], device=device) < probability
    ratios = torch.empty(shape[0], device=device).uniform_(low, high)
    mask = torch.rand(shape, device=device) < ratios[:, None]
    return mask & selected[:, None]


def _schedule(start: float, end: float, step: int, total: int) -> float:
    if total <= 1:
        return start
    progress = min(max(step / (total - 1), 0.0), 1.0)
    return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0


class Trainer:
    def __init__(
        self,
        *,
        model: Any,
        teacher: Any,
        data: PerturbationData,
        priors: PriorStore,
        config: dict[str, Any],
        device: Any,
        metric_logger: Any | None = None,
        training_state_path: Path | None = None,
        training_state_identity: dict[str, Any] | None = None,
    ) -> None:
        torch = _torch()
        self.torch = torch
        self.model = model
        self.teacher = teacher
        self.data = data
        self.priors = priors
        self.config = config
        self.device = device
        self.metric_logger = metric_logger
        self.training_state_path = training_state_path
        self.training_state_identity = training_state_identity
        training = config["training"]
        optimizer_name = str(training.get("optimizer", "adamw"))
        optimizer_class = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
        self.optimizer = optimizer_class(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training.get("weight_decay", 0.01)),
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=float(training.get("lr_gamma", 0.9))
        )
        prototypes = int(config["model"]["dino_head"]["prototypes"])
        self.center = torch.zeros(1, prototypes, device=device)

    def _save_training_state(self, payload: dict[str, Any]) -> None:
        if self.training_state_path is None:
            return
        self.training_state_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.training_state_path.name}.",
            dir=self.training_state_path.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            self.torch.save(payload, temporary)
            os.replace(temporary, self.training_state_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _load_training_state(self) -> dict[str, Any] | None:
        if self.training_state_path is None or not self.training_state_path.exists():
            return None
        state = self.torch.load(
            self.training_state_path,
            map_location=self.device,
            weights_only=False,
        )
        if not isinstance(state, dict) or state.get("identity") != self.training_state_identity:
            raise ValueError("training-state identity differs from current formal run")
        self.model.load_state_dict(state["model"])
        self.teacher.load_state_dict(state["teacher"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.center.copy_(state["center"].to(self.device))
        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_random_state"])
        self.torch.set_rng_state(state["torch_random_state"].cpu())
        cuda_states = state.get("cuda_random_states")
        if cuda_states is not None and self.torch.cuda.is_available():
            self.torch.cuda.set_rng_state_all(cuda_states)
        return state

    def _validation_prediction_mse(self, _epoch: int) -> float:
        """Condition-macro validation GEP used for scGPT-style checkpoint selection."""

        torch = self.torch
        training = self.config["training"]
        sampler = ConditionBagSampler(
            self.data,
            split="validation",
            bag_size=int(training["bag_size"]),
            conditions_per_batch=int(training["batch_size"]),
            seed=int(training["seed"]) + 50_000,
        )
        max_batches = int(training.get("max_validation_batches", 0))
        total = 0.0
        batches = 0
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            # Checkpoint selection must compare epochs on the same sampled
            # controls and post-perturbation cells. Training remains epoch-
            # randomized; validation deliberately uses the fixed epoch-0 draw.
            for batch_index, batch in enumerate(sampler.batches(0, shuffle=False)):
                if max_batches > 0 and batch_index >= max_batches:
                    break
                condition_count = len(batch.conditions)
                bag_size = int(training["bag_size"])
                controls = torch.as_tensor(
                    batch.controls.reshape(-1, self.data.n_genes), device=self.device
                )
                posts = torch.as_tensor(
                    batch.posts.reshape(-1, self.data.n_genes), device=self.device
                )
                repeated_targets = _repeat_targets(batch.targets, bag_size)
                vectors, token_mask, _ = _tensor_prior(
                    self.priors,
                    "base",
                    repeated_targets,
                    device=self.device,
                    control="none",
                    seed=int(training["seed"]),
                )
                view = self.model.conditional_view(
                    controls,
                    source="base",
                    prior_vectors=vectors,
                    prior_mask=token_mask,
                )
                prediction = self.model.predict_expression(controls, view).view(
                    condition_count, bag_size, -1
                ).mean(dim=1)
                truth = posts.view(condition_count, bag_size, -1).mean(dim=1)
                total += float(torch.nn.functional.mse_loss(prediction, truth).cpu())
                batches += 1
        self.model.train(was_training)
        if not batches:
            raise RuntimeError("validation checkpoint selection produced zero batches")
        return total / batches

    def train(self) -> dict[str, Any]:
        from dinogenept.models.dinogenept.losses import (
            condition_dino_loss,
            delta_ibot_loss,
            koleo_loss,
            prediction_loss,
            update_center,
        )

        torch = self.torch
        training = self.config["training"]
        ablation = self.config["ablation"]
        losses_config = training["losses"]
        epochs = int(training["epochs"])
        bag_size = int(training["bag_size"])
        sampler = ConditionBagSampler(
            self.data,
            split="train",
            bag_size=bag_size,
            conditions_per_batch=int(training["batch_size"]),
            seed=int(training["seed"]),
        )
        batches_per_epoch = math.ceil(len(sampler.conditions) / int(training["batch_size"]))
        max_batches = int(training.get("max_batches", 0))
        if max_batches > 0:
            batches_per_epoch = min(batches_per_epoch, max_batches)
        total_steps = max(1, epochs * batches_per_epoch)
        resume_state = self._load_training_state()
        history: list[dict[str, float]] = (
            list(resume_state["history"]) if resume_state is not None else []
        )
        global_step = int(resume_state["global_step"]) if resume_state is not None else 0
        best_epoch: int | None = (
            resume_state.get("best_epoch") if resume_state is not None else None
        )
        best_validation = (
            float(resume_state["best_validation"])
            if resume_state is not None
            else float("inf")
        )
        best_state: dict[str, Any] | None = (
            resume_state.get("best_state") if resume_state is not None else None
        )
        start_epoch = int(resume_state["completed_epochs"]) if resume_state is not None else 0
        if start_epoch > epochs or len(history) != start_epoch:
            raise ValueError("training-state epoch history differs from configuration")
        use_amp = bool(training.get("amp_bfloat16", True)) and self.device.type == "cuda"
        autocast = torch.autocast
        self.model.train()
        for epoch in range(start_epoch, epochs):
            aggregate: dict[str, float] = {}
            batches = 0
            for batch_index, batch in enumerate(sampler.batches(epoch, shuffle=True)):
                if max_batches > 0 and batch_index >= max_batches:
                    break
                condition_count = len(batch.conditions)
                controls = torch.as_tensor(
                    batch.controls.reshape(-1, self.data.n_genes), device=self.device
                )
                posts = torch.as_tensor(
                    batch.posts.reshape(-1, self.data.n_genes), device=self.device
                )
                repeated_targets = _repeat_targets(batch.targets, bag_size)
                base_vectors, base_mask, _ = _tensor_prior(
                    self.priors,
                    "base",
                    repeated_targets,
                    device=self.device,
                    control="none",
                    seed=int(training["seed"]),
                )
                self.optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp):
                    base_view = self.model.conditional_view(
                        controls,
                        source="base",
                        prior_vectors=base_vectors,
                        prior_mask=base_mask,
                    )
                    post_mean = posts.view(condition_count, bag_size, -1).mean(dim=1)
                    control_mean = controls.view(condition_count, bag_size, -1).mean(dim=1)
                    prediction_weight = float(losses_config.get("prediction_weight", 1.0))
                    if prediction_weight:
                        prediction = self.model.predict_expression(controls, base_view)
                        prediction_mean = prediction.view(
                            condition_count, bag_size, -1
                        ).mean(dim=1)
                        pred_loss = prediction_loss(
                            prediction_mean,
                            post_mean,
                            control_mean,
                            kind=str(losses_config.get("prediction", "mse")),
                            direction_weight=float(
                                losses_config.get("direction_weight", 0.5)
                            ),
                        )
                        total_loss = pred_loss * prediction_weight
                        components: dict[str, Any] = {"prediction": pred_loss}
                    else:
                        total_loss = base_view.cls.sum() * 0.0
                        components = {}

                    teacher_condition = None
                    if ablation.get("dino", True) or ablation.get("delta_ibot", True):
                        teacher_gene_indices = None
                        if self.config["model"]["mask"].get("gene_selection") == "train_variance":
                            fixed = torch.as_tensor(
                                self.data.ibot_gene_indices,
                                device=self.device,
                                dtype=torch.long,
                            )
                            teacher_gene_indices = fixed.unsqueeze(0).expand(
                                posts.shape[0], -1
                            )
                        with torch.no_grad():
                            clean_view, teacher_logits = self.teacher(
                                posts, gene_indices=teacher_gene_indices
                            )
                            teacher_condition = teacher_logits.view(
                                condition_count, bag_size, -1
                            ).mean(dim=1)
                        base_logits = self.model.dino_head(base_view.cls).view(
                            condition_count, bag_size, -1
                        ).mean(dim=1)
                        if ablation.get("dino", True):
                            dino_config = self.config["model"]["dino"]
                            teacher_temperature = _schedule(
                                float(dino_config["teacher_temperature_start"]),
                                float(dino_config["teacher_temperature_end"]),
                                global_step,
                                total_steps,
                            )
                            base_dino = condition_dino_loss(
                                base_logits,
                                teacher_condition,
                                center=self.center,
                                student_temperature=float(dino_config["student_temperature"]),
                                teacher_temperature=teacher_temperature,
                            )
                            total_loss = total_loss + float(
                                losses_config.get("dino_weight", 0.5)
                            ) * base_dino
                            components["dino_base"] = base_dino

                            if ablation.get("dynamic_locals", False):
                                local_losses = []
                                allowed_sources = tuple(
                                    str(source)
                                    for source in ablation.get(
                                        "local_sources", self.priors.optional_sources
                                    )
                                )
                                source_control = str(ablation.get("source_control", "none"))
                                for source in self.priors.optional_sources:
                                    rows = [
                                        index
                                        for index, (condition, targets) in enumerate(
                                            zip(batch.conditions, batch.targets, strict=True)
                                        )
                                        if _source_for_condition(
                                            self.priors,
                                            condition,
                                            targets,
                                            epoch,
                                            allowed_sources,
                                        )
                                        == source
                                    ]
                                    if not rows:
                                        continue
                                    flat_indices = [
                                        row * bag_size + offset
                                        for row in rows
                                        for offset in range(bag_size)
                                    ]
                                    flat = torch.as_tensor(flat_indices, device=self.device)
                                    local_targets = tuple(
                                        repeated_targets[index] for index in flat_indices
                                    )
                                    vectors, token_mask, condition_mask = _tensor_prior(
                                        self.priors,
                                        source,
                                        local_targets,
                                        device=self.device,
                                        control=source_control,
                                        seed=int(training["seed"]) + epoch,
                                    )
                                    if not condition_mask.all():
                                        raise RuntimeError("source scheduler selected an unavailable local")
                                    local_view = self.model.conditional_view(
                                        controls.index_select(0, flat),
                                        source=source,
                                        prior_vectors=vectors,
                                        prior_mask=token_mask,
                                        route_anchor=base_view.route_anchor.index_select(0, flat),
                                    )
                                    local_logits = self.model.dino_head(local_view.cls).view(
                                        len(rows), bag_size, -1
                                    ).mean(dim=1)
                                    local_losses.append(
                                        condition_dino_loss(
                                            local_logits,
                                            teacher_condition.index_select(
                                                0, torch.as_tensor(rows, device=self.device)
                                            ),
                                            center=self.center,
                                            student_temperature=float(
                                                dino_config["student_temperature"]
                                            ),
                                            teacher_temperature=teacher_temperature,
                                            reduction="none",
                                        )
                                    )
                                if local_losses:
                                    local_dino = torch.cat(local_losses).mean()
                                    total_loss = total_loss + float(
                                        losses_config.get("local_dino_weight", 0.5)
                                    ) * local_dino
                                    components["dino_local"] = local_dino

                        if ablation.get("delta_ibot", True):
                            gene_indices = clean_view.gene_indices
                            mask = _mask_expression(
                                tuple(gene_indices.shape), self.config["model"]["mask"], self.device
                            )
                            noisy_posts = posts
                            noise_std = float(self.config["model"]["mask"].get("noise_std", 0.0))
                            if noise_std:
                                noisy_posts = posts + noise_std * torch.randn_like(posts)
                            observed = self.model.observed_view(
                                noisy_posts,
                                gene_indices=gene_indices,
                                expression_mask=mask,
                            )
                            control_reference = control_mean[:, None, :].expand(
                                condition_count, bag_size, self.data.n_genes
                            ).reshape_as(posts)
                            delta_target = posts - control_reference
                            if ablation.get("ibot_target_control", "none") == "shuffled_expression":
                                permutation = torch.randperm(
                                    self.data.n_genes, device=self.device
                                )
                                delta_target = delta_target.index_select(1, permutation)
                            ibot = delta_ibot_loss(
                                observed.token_hidden,
                                observed.gene_indices,
                                mask,
                                delta_target,
                                self.model.token_delta_head,
                            )
                            total_loss = total_loss + float(
                                losses_config.get("ibot_weight", 0.5)
                            ) * ibot
                            components["delta_ibot"] = ibot

                    koleo_weight = float(ablation.get("koleo_weight", 0.0))
                    if koleo_weight:
                        condition_features = base_view.cls.view(
                            condition_count, bag_size, -1
                        ).mean(dim=1)
                        koleo = koleo_loss(condition_features)
                        total_loss = total_loss + koleo_weight * koleo
                        components["koleo"] = koleo

                    condition_features = base_view.cls.view(
                        condition_count, bag_size, -1
                    ).mean(dim=1)
                    components["representation_std"] = condition_features.float().std(
                        dim=0, unbiased=False
                    ).mean()

                    balance_losses = [
                        value
                        for key, value in base_view.diagnostics.items()
                        if key.endswith("moe_balance_loss")
                    ]
                    if balance_losses and ablation.get("adapter", {}).get("balance") == "auxiliary":
                        balance = torch.stack(balance_losses).mean()
                        total_loss = total_loss + float(
                            losses_config.get("moe_balance_weight", 0.01)
                        ) * balance
                        components["moe_balance"] = balance
                    for key, value in base_view.diagnostics.items():
                        if key.endswith("moe_entropy"):
                            components[f"{key}.mean"] = value.float().mean()
                        elif key.endswith("moe_load"):
                            load = value.float()
                            components[f"{key}.dead_fraction"] = (load == 0).float().mean()
                            components[f"{key}.load_cv"] = load.std(unbiased=False) / load.mean().clamp_min(1.0)
                    if koleo_weight:
                        components["koleo_active"] = total_loss.new_tensor(
                            float(condition_count >= 2)
                        )
                if not torch.isfinite(total_loss):
                    raise FloatingPointError("training loss became non-finite")
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(training.get("gradient_clip", 1.0))
                )
                self.optimizer.step()
                if teacher_condition is not None:
                    momentum = _schedule(
                        float(self.config["model"]["dino"]["ema_start"]),
                        1.0,
                        global_step,
                        total_steps,
                    )
                    self.teacher.update(self.model, momentum)
                    update_center(
                        self.center,
                        teacher_condition,
                        float(self.config["model"]["dino"].get("center_momentum", 0.9)),
                    )
                components["total"] = total_loss
                for key, value in components.items():
                    aggregate[key] = aggregate.get(key, 0.0) + float(value.detach().float().cpu())
                batches += 1
                global_step += 1
            if not batches:
                raise RuntimeError("training epoch produced zero batches")
            self.scheduler.step()
            row = {
                "epoch": float(epoch + 1),
                "batches": float(batches),
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                **{key: value / batches for key, value in aggregate.items()},
            }
            if training.get("selection_metric", "none") == "validation_prediction_mse":
                validation = self._validation_prediction_mse(epoch)
                row["validation_prediction_mse"] = validation
                if validation < best_validation:
                    best_validation = validation
                    best_epoch = epoch + 1
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in self.model.state_dict().items()
                    }
            history.append(row)
            self._save_training_state(
                {
                    "schema_version": "dinogenept-training-state-v1",
                    "identity": self.training_state_identity,
                    "completed_epochs": epoch + 1,
                    "global_step": global_step,
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_validation": best_validation,
                    "best_state": best_state,
                    "model": self.model.state_dict(),
                    "teacher": self.teacher.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "center": self.center.detach().cpu(),
                    "python_random_state": random.getstate(),
                    "numpy_random_state": np.random.get_state(),
                    "torch_random_state": torch.get_rng_state(),
                    "cuda_random_states": (
                        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                    ),
                }
            )
            if self.metric_logger is not None:
                self.metric_logger(row)
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {
            "phase": str(training.get("phase", "finetune")),
            "epochs": epochs,
            "steps": global_step,
            "effective_cell_batch_size": int(training["batch_size"])
            * int(training["bag_size"]),
            "selection_metric": str(training.get("selection_metric", "none")),
            "best_epoch": best_epoch,
            "best_validation_prediction_mse": (
                best_validation if best_epoch is not None else None
            ),
            "history": history,
            "resumed_from_epoch": start_epoch,
        }
