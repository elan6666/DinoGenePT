"""Base-only adapter around the official ``scouter-learn`` package."""

from __future__ import annotations

import importlib.metadata
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from .provenance import (
    SCOUTER_BALANCED_DATASET_SEED,
    input_manifest,
    scouter_source_sha256,
    upstream_identity,
)


def _torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("Scouter training requires `pip install -e '.[train,scouter]'`") from error
    return torch


def _upstream():
    upstream_identity()
    try:
        from scouter import Scouter, ScouterData
    except ModuleNotFoundError as error:
        raise RuntimeError("Scouter baseline requires scouter-learn==0.1.10") from error
    return Scouter, ScouterData


class ScouterAdapter(_torch().nn.Module):
    """Lifecycle shim; all predictive layers come from upstream ScouterModel."""

    def __init__(self, *, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.network: Any | None = None
        self.upstream: Any | None = None

    def attach(self, upstream: Any, device: Any) -> None:
        if self.network is not None:
            raise RuntimeError("Scouter network is already initialized")
        self.upstream = upstream
        self.network = upstream.network.to(device)

    def load_pretrained(self, *_: Any, **__: Any) -> dict[str, Any]:
        raise ValueError("Scouter does not accept a DinoGenePT pretraining checkpoint")

    def parameter_receipt(self) -> dict[str, Any]:
        if self.network is None:
            raise RuntimeError("Scouter network has not been initialized")
        total = sum(parameter.numel() for parameter in self.network.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.network.parameters() if parameter.requires_grad
        )
        try:
            version = importlib.metadata.version("scouter-learn")
        except importlib.metadata.PackageNotFoundError:
            version = None
        return {
            "total": total,
            "trainable": trainable,
            "upstream_package": "scouter-learn",
            "upstream_version": version,
            "upstream_source_sha256": scouter_source_sha256(),
            "upstream_balanced_dataset_seed": SCOUTER_BALANCED_DATASET_SEED,
            "prior_contract": "genept-seed-base-ncbi-uniprot-only",
        }


def _base_embedding_frame(priors: Any):
    import pandas as pd

    if priors.optional_sources:
        raise ValueError("Scouter baseline forbids optional GO/Protein/Pathway/HPA priors")
    table = priors.tables["base"]
    frame = pd.DataFrame(table.vectors, index=list(table.genes))
    if "ctrl" in frame.index:
        raise ValueError("Base embedding artifact must not overload the ctrl sentinel")
    frame.loc["ctrl"] = np.zeros(table.dimension, dtype=np.float32)
    return frame


def _nonzero_indices(data: Any, conditions: tuple[str, ...]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for condition in conditions:
        values = data.expression[data.indices_for_condition(condition)].mean(axis=0)
        result[condition] = np.flatnonzero(values != 0).astype(np.int64)
    return result


def _prepare_upstream(data: Any, priors: Any, config: dict[str, Any], device: Any):
    import anndata as ad
    import pandas as pd
    from scipy import sparse

    Scouter, ScouterData = _upstream()
    adata = ad.AnnData(
        X=sparse.csr_matrix(data.expression),
        obs=pd.DataFrame(
            {"condition": list(data.conditions)}, index=list(data.row_ids)
        ),
        var=pd.DataFrame({"gene_name": list(data.genes)}, index=list(data.genes)),
    )
    pertdata = ScouterData(adata, _base_embedding_frame(priors), "condition", "gene_name")
    pertdata.setup_ad("embd_index", slim=True)
    if len(pertdata.unmatched_genes):
        raise ValueError("Scouter setup attempted to drop uncovered perturbation conditions")

    conditions = np.asarray(pertdata.adata.obs["condition"], dtype=str)
    control = conditions == "ctrl"
    train_conditions = data.split_conditions("train")
    validation_conditions = data.split_conditions("validation")
    test_conditions = data.split_conditions("test")
    pertdata.train_conds = list(train_conditions)
    pertdata.val_conds = list(validation_conditions)
    pertdata.test_conds = list(test_conditions)
    pertdata.train_adata = pertdata.adata[control | np.isin(conditions, train_conditions)].copy()
    pertdata.val_adata = pertdata.adata[control | np.isin(conditions, validation_conditions)].copy()
    pertdata.test_adata = pertdata.adata[control | np.isin(conditions, test_conditions)].copy()

    # Upstream loss consumes only a per-condition non-zero-gene mask. Build it
    # from each train/validation condition's own population; never inspect test
    # outcomes while fitting or selecting a checkpoint.
    nonzero = _nonzero_indices(data, (*train_conditions, *validation_conditions))
    pertdata.train_adata.uns["gene_idx_non_zeros"] = nonzero
    pertdata.val_adata.uns["gene_idx_non_zeros"] = nonzero

    upstream = Scouter(pertdata, device=str(device))
    architecture = config["model"]["architecture"]
    upstream.model_init(
        n_encoder=tuple(int(value) for value in architecture["encoder"]),
        n_out_encoder=int(architecture["encoder_output"]),
        n_decoder=tuple(int(value) for value in architecture["decoder"]),
        use_batch_norm=bool(architecture["batch_norm"]),
        use_layer_norm=bool(architecture["layer_norm"]),
        dropout_rate=float(architecture["dropout"]),
    )
    return upstream


class OfficialScouterTrainer:
    def __init__(
        self,
        *,
        model: ScouterAdapter,
        upstream: Any,
        config: dict[str, Any],
        metric_logger: Any | None = None,
    ) -> None:
        self.model = model
        self.upstream = upstream
        self.config = config
        self.metric_logger = metric_logger

    def train(self) -> dict[str, Any]:
        training = self.config["training"]
        self.upstream.train(
            nonzero_idx_key="gene_idx_non_zeros",
            batch_size=int(training["batch_size"]),
            loss_gamma=float(training.get("loss_gamma", 0.0)),
            loss_lambda=float(training["loss_lambda"]),
            lr=float(training["learning_rate"]),
            sched_gamma=float(training.get("lr_gamma", 0.9)),
            n_epochs=int(training["epochs"]),
            patience=int(training.get("patience", 5)),
        )
        history = self.upstream.loss_history
        rows = [
            {
                "epoch": index + 1,
                "train_loss": float(train_loss),
                "validation_loss": (
                    float(history["val_loss"][index])
                    if index < len(history["val_loss"])
                    else None
                ),
            }
            for index, train_loss in enumerate(history["train_loss"])
        ]
        if self.metric_logger is not None:
            for row in rows:
                self.metric_logger(row)
        return {
            "phase": "finetune",
            "epochs_requested": int(training["epochs"]),
            "epochs": len(history["train_loss"]),
            "early_stopping_patience": int(training.get("patience", 5)),
            "implementation": "scouter-learn-0.1.10-train-api",
            "run_seed": int(training["seed"]),
            "upstream_balanced_dataset_seed": SCOUTER_BALANCED_DATASET_SEED,
            "comparison_label": (
                "Scouter v0.1.10 + GenePT-Seed Base under GraD-Pert protocol"
            ),
            "history": rows,
        }


class ScouterPlugin:
    def input_manifest(self, config: dict[str, Any]) -> dict[str, Any]:
        return input_manifest(config)

    def build_model(
        self,
        config: dict[str, Any],
        *,
        n_genes: int,
        prior_dimensions: dict[str, int],
    ) -> ScouterAdapter:
        del n_genes
        if set(prior_dimensions) != {"base"}:
            raise ValueError("Scouter baseline requires exactly one Base prior")
        return ScouterAdapter(config=config["model"])

    def build_trainer(
        self,
        *,
        model: ScouterAdapter,
        data: Any,
        priors: Any,
        config: dict[str, Any],
        device: Any,
        metric_logger: Any | None = None,
        training_state_path: Any | None = None,
        training_state_identity: dict[str, Any] | None = None,
    ) -> OfficialScouterTrainer:
        if training_state_path is not None and Path(training_state_path).exists():
            raise ValueError("Scouter does not support epoch-state resume; restart the run")
        del training_state_identity
        upstream = _prepare_upstream(data, priors, config, device)
        model.attach(upstream, device)
        return OfficialScouterTrainer(
            model=model,
            upstream=upstream,
            config=config,
            metric_logger=metric_logger,
        )

    def predict_split(
        self,
        *,
        model: ScouterAdapter,
        data: Any,
        priors: Any,
        config: dict[str, Any],
        device: Any,
        split: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        if model.network is None or model.upstream is None:
            raise RuntimeError("Scouter has not been trained")
        evaluation = config["evaluation"]
        if evaluation.get("name") == "gradpert_exact":
            from dinogenept.evaluation.gradpert import frozen_control_rows

            controls, _, _ = frozen_control_rows(data, evaluation, split)
        else:
            controls = {}
        conditions = data.split_conditions(split)
        samples = int(evaluation.get("control_samples", 300))
        batch_size = int(evaluation.get("batch_size", 64))
        rng = np.random.default_rng(int(evaluation.get("seed", 20260824)))
        control_indices = np.flatnonzero(data.control_mask)
        embedding_index = model.upstream.embd_idx_dict
        torch = _torch()
        use_amp = bool(config["training"].get("amp_bfloat16", False)) and device.type == "cuda"
        predictions: dict[str, np.ndarray] = {}
        truths: dict[str, np.ndarray] = {}
        selected_controls: dict[str, np.ndarray] = {}
        model.network.eval()
        with torch.no_grad():
            for condition in conditions:
                current_controls = (
                    controls[condition]
                    if controls
                    else data.expression[rng.choice(control_indices, samples, replace=True)]
                )
                targets = [part for part in condition.split("+") if part]
                try:
                    target_indices = [embedding_index[target] for target in targets]
                except KeyError as error:
                    raise ValueError(f"Base prior lacks Scouter target: {error.args[0]}") from error
                chunks = []
                for start in range(0, len(current_controls), batch_size):
                    current = current_controls[start : start + batch_size]
                    pert_idx = torch.as_tensor(
                        [target_indices] * len(current), dtype=torch.long, device=device
                    )
                    ctrl = torch.as_tensor(current, dtype=torch.float32, device=device)
                    context = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if use_amp
                        else nullcontext()
                    )
                    with context:
                        predicted = model.network(pert_idx, ctrl)
                    chunks.append(predicted.float().cpu().numpy())
                predictions[condition] = np.concatenate(chunks, axis=0)
                truths[condition] = data.expression[data.indices_for_condition(condition)]
                selected_controls[condition] = np.asarray(current_controls, dtype=np.float32)
        return predictions, truths, selected_controls

    def permutation_check(self, **_: Any) -> dict[str, float | bool]:
        return {
            "max_abs_difference": 0.0,
            "mean_abs_difference": 0.0,
            "tolerance": 0.0,
            "passed": True,
            "not_applicable": True,
        }


PLUGIN = ScouterPlugin()
