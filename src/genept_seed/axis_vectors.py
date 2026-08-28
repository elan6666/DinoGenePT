"""Align a trusted official embedding artifact to an exact graph axis."""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np

from .axis_corpus import HGNCIndex, _load_ensembl_map
from .provenance import atomic_write_json, digest_file, utc_now


def materialize_axis_vectors(
    *,
    source_path: Path,
    genes_path: Path,
    hgnc_path: Path,
    output_path: Path,
    manifest_path: Path,
    model: str,
    trusted_pickle: bool,
    axis_mapping_path: Path | None = None,
) -> dict[str, object]:
    if not trusted_pickle:
        raise ValueError("official pickle alignment requires trusted_pickle=True")
    with source_path.open("rb") as handle:
        raw = pickle.load(handle)  # noqa: S301 - explicit trusted-source gate above
        if handle.read(1):
            raise ValueError("official embedding pickle contains trailing bytes")
    if type(raw) is not dict or not raw:
        raise ValueError("official embedding pickle must be a non-empty exact dictionary")
    embeddings = cast(dict[object, object], raw)
    widths = {
        int(np.asarray(vector).reshape(-1).shape[0])
        for vector in embeddings.values()
    }
    if len(widths) != 1:
        raise ValueError("official embedding artifact has inconsistent widths")
    width = widths.pop()
    genes = [line.strip() for line in genes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(genes) != len(set(genes)):
        raise ValueError("graph axis contains duplicate exact gene IDs")
    ensembl_by_gene = _load_ensembl_map(axis_mapping_path)
    hgnc = HGNCIndex(hgnc_path)
    matrix = np.zeros((len(genes), width), dtype=np.float32)
    source_key_by_gene: dict[str, str] = {}
    missing: list[str] = []
    stats: Counter[str] = Counter()
    for index, gene in enumerate(genes):
        source_key = gene if gene in embeddings else None
        if source_key is None:
            row, _ = hgnc.resolve(gene, ensembl_by_gene.get(gene))
            if row is not None and row.symbol in embeddings:
                source_key = row.symbol
        if source_key is None:
            missing.append(gene)
            stats["learned_id_fallback"] += 1
            continue
        vector = np.asarray(embeddings[source_key], dtype=np.float32).reshape(-1)
        if vector.shape != (width,) or not np.isfinite(vector).all():
            raise ValueError(f"invalid official embedding vector for {source_key}")
        matrix[index] = vector
        source_key_by_gene[gene] = source_key
        stats["exact_official" if source_key == gene else "hgnc_alias_official"] += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            genes=np.asarray(genes, dtype=str),
            vectors=matrix,
            model=np.asarray(model),
        )
    temporary.replace(output_path)
    manifest: dict[str, object] = {
        "schema_version": "genept-seed-axis-vectors-v1",
        "created_at": utc_now(),
        "model": model,
        "source_sha256": digest_file(source_path),
        "genes_sha256": digest_file(genes_path),
        "hgnc_sha256": digest_file(hgnc_path),
        "axis_mapping_sha256": digest_file(axis_mapping_path) if axis_mapping_path else None,
        "output_sha256": digest_file(output_path),
        "genes": len(genes),
        "dimension": width,
        "source_counts": dict(sorted(stats.items())),
        "missing_prior_gene_ids": missing,
        "missing_prior_policy": "zero_prior_plus_downstream_learned_id_residual",
        "source_key_by_gene": source_key_by_gene,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest
