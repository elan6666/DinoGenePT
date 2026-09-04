"""Strict coverage audit for the three progressive knowledge embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .gradpert_union import DATASETS, _targets
from .provenance import atomic_write_json, digest_file, utc_now
from .vectors import load_npz


def audit_knowledge_vectors(
    *,
    genes_path: Path,
    ggi_genes_path: Path,
    gradpert_root: Path,
    protein_path: Path,
    pathway_path: Path,
    hpa_path: Path,
    output_path: Path,
    expected_dimension: int = 2048,
) -> dict[str, Any]:
    expected = [line.strip() for line in genes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_set = set(expected)
    if len(expected) != len(expected_set):
        raise ValueError("master vector allowlist contains duplicate exact labels")
    ggi = {
        line.strip()
        for line in ggi_genes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not ggi <= expected_set:
        raise ValueError(f"GGI genes are absent from master universe: {sorted(ggi - expected_set)[:20]}")
    paths = {
        "seed_go_protein": protein_path,
        "seed_go_protein_pathway": pathway_path,
        "seed_go_protein_pathway_hpa": hpa_path,
    }
    conditions: dict[str, Any] = {}
    models: set[str] = set()
    for name, path in paths.items():
        loaded = load_npz(path)
        actual = [str(gene) for gene in loaded.genes.tolist()]
        if actual != expected:
            missing = sorted(expected_set - set(actual))
            extra = sorted(set(actual) - expected_set)
            raise ValueError(f"{name} vectors differ from master; missing={missing[:20]}, extra={extra[:20]}")
        if loaded.vectors.shape != (len(expected), expected_dimension):
            raise ValueError(f"{name} vector shape is {loaded.vectors.shape}")
        zero_norms = int(np.count_nonzero(np.linalg.norm(loaded.vectors, axis=1) == 0))
        if zero_norms:
            raise ValueError(f"{name} contains {zero_norms} zero vectors")
        models.add(loaded.model)
        conditions[name] = {
            "genes": len(actual),
            "dimension": int(loaded.vectors.shape[1]),
            "model": loaded.model,
            "zero_vectors": 0,
            "ggi_exact_labels": len(ggi & set(actual)),
            "sha256": digest_file(path),
        }
    if len(models) != 1:
        raise ValueError(f"progressive embeddings use different models: {sorted(models)}")
    datasets: list[dict[str, Any]] = []
    for dataset, protocol in DATASETS:
        root = gradpert_root / dataset / protocol
        graph = {
            line.strip()
            for line in (root / "canonical" / "graph_gene_ids.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        split = json.loads((root / "manifests" / "split.json").read_text(encoding="utf-8"))
        targets = _targets(split)
        datasets.append(
            {
                "dataset": dataset,
                "graph_genes": len(graph),
                "perturbation_targets": len(targets),
                "graph_exact_coverage": len(graph & expected_set) / len(graph),
                "target_exact_coverage": len(targets & expected_set) / len(targets),
            }
        )
    normalized: dict[str, list[str]] = {}
    for gene in expected:
        normalized.setdefault(gene.upper(), []).append(gene)
    case_pairs = {key: values for key, values in normalized.items() if len(values) > 1}
    receipt: dict[str, Any] = {
        "schema_version": "genept-seed-progressive-vector-audit-v1",
        "created_at": utc_now(),
        "master_genes": len(expected),
        "master_sha256": digest_file(genes_path),
        "ggi_genes": len(ggi),
        "ggi_sha256": digest_file(ggi_genes_path),
        "ggi_selection_policy": "prefer exact fixed-universe label before unique case-fold fallback",
        "casefold_collision_groups": case_pairs,
        "conditions": conditions,
        "datasets": datasets,
        "all_graph_axes_exact_coverage": all(row["graph_exact_coverage"] == 1.0 for row in datasets),
        "all_targets_exact_coverage": all(row["target_exact_coverage"] == 1.0 for row in datasets),
    }
    atomic_write_json(output_path, receipt)
    return receipt
