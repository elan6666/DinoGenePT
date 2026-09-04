"""Freeze the complete GraD-Pert graph/perturbation universe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .provenance import atomic_write_json, digest_file, utc_now

DATASETS = (
    ("replogle_k562_essential", "within_cell_unseen_single"),
    ("replogle_rpe1_essential", "within_cell_unseen_single"),
    ("nadig_jurkat", "within_cell_unseen_single"),
    ("nadig_hepg2", "within_cell_unseen_single"),
    ("norman", "norman_combo_seen2"),
)


def _read_genes(path: Path) -> list[str]:
    genes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(genes) != len(set(genes)):
        raise ValueError(f"duplicate exact gene symbol in {path}")
    return genes


def _targets(split: dict[str, Any]) -> set[str]:
    control = str(split.get("control_condition_id", "ctrl"))
    targets: set[str] = set()
    for key in ("train_conditions", "val_conditions", "test_conditions"):
        for condition in split.get(key, []):
            targets.update(part for part in str(condition).split("+") if part and part != control)
    return targets


def build_gradpert_union(
    *,
    gradpert_root: Path,
    output_path: Path,
    manifest_path: Path,
    extra_genes_path: Path | None = None,
) -> dict[str, Any]:
    """Write a deterministic master allowlist and prove graph/target coverage."""

    graph_union: set[str] = set()
    target_union: set[str] = set()
    datasets: list[dict[str, Any]] = []
    for dataset, protocol in DATASETS:
        root = gradpert_root / dataset / protocol
        graph_path = root / "canonical" / "graph_gene_ids.txt"
        split_path = root / "manifests" / "split.json"
        graph = set(_read_genes(graph_path))
        split = json.loads(split_path.read_text(encoding="utf-8"))
        targets = _targets(split)
        missing = sorted(targets - graph)
        if missing:
            raise ValueError(f"{dataset} perturbation targets missing from graph axis: {missing}")
        graph_union.update(graph)
        target_union.update(targets)
        datasets.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "graph_genes": len(graph),
                "perturbation_targets": len(targets),
                "graph_sha256": digest_file(graph_path),
                "split_sha256": digest_file(split_path),
            }
        )
    extras = set(_read_genes(extra_genes_path)) if extra_genes_path else set()
    master = sorted(graph_union | extras)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(master) + "\n", encoding="utf-8")
    manifest: dict[str, Any] = {
        "schema_version": "genept-seed-gradpert-master-universe-v1",
        "created_at": utc_now(),
        "datasets": datasets,
        "graph_union_genes": len(graph_union),
        "perturbation_target_union_genes": len(target_union),
        "all_targets_in_graph_union": target_union <= graph_union,
        "extra_genes": len(extras),
        "graph_extra_intersection": len(graph_union & extras),
        "master_genes": len(master),
        "extra_genes_sha256": digest_file(extra_genes_path) if extra_genes_path else None,
        "output_sha256": digest_file(output_path),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def build_gradpert_targets(
    *, gradpert_root: Path, output_path: Path, manifest_path: Path
) -> dict[str, Any]:
    """Freeze the exact union of perturbation targets across the five protocols."""

    union: set[str] = set()
    datasets: list[dict[str, Any]] = []
    for dataset, protocol in DATASETS:
        split_path = gradpert_root / dataset / protocol / "manifests" / "split.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        targets = _targets(split)
        union.update(targets)
        datasets.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "targets": len(targets),
                "split_sha256": digest_file(split_path),
            }
        )
    ordered = sorted(union)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(ordered) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "genept-seed-gradpert-target-union-v1",
        "created_at": utc_now(),
        "datasets": datasets,
        "targets": len(ordered),
        "output_sha256": digest_file(output_path),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest
