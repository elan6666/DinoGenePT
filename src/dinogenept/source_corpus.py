"""Derive sparse source-only text from audited append-only corpora."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .provenance import atomic_write_json, digest_file, utc_now


def build_source_only_corpus(
    *,
    base_path: Path,
    enriched_path: Path,
    output_path: Path,
    manifest_path: Path,
    source: str,
    genes_path: Path | None = None,
) -> dict[str, Any]:
    base_raw = json.loads(base_path.read_text(encoding="utf-8"))
    enriched_raw = json.loads(enriched_path.read_text(encoding="utf-8"))
    if not isinstance(base_raw, dict) or not isinstance(enriched_raw, dict):
        raise ValueError("source-only inputs must be JSON objects")
    if set(base_raw) != set(enriched_raw):
        raise ValueError("append-only corpora have different gene universes")
    allowed = None
    if genes_path is not None:
        allowed = {
            line.strip()
            for line in genes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    output: dict[str, str] = {}
    unchanged = 0
    for gene, base_value in base_raw.items():
        if allowed is not None and gene not in allowed:
            continue
        base_text = str(base_value)
        enriched_text = str(enriched_raw[gene])
        if not enriched_text.startswith(base_text):
            raise ValueError(f"enriched corpus is not append-only for {gene}")
        suffix = enriched_text[len(base_text) :].strip()
        if suffix:
            output[str(gene)] = suffix
        else:
            unchanged += 1
    atomic_write_json(output_path, output)
    manifest: dict[str, Any] = {
        "schema_version": "genept-seed-source-only-corpus-v1",
        "created_at": utc_now(),
        "source": source,
        "requested_genes": len(allowed) if allowed is not None else len(base_raw),
        "available_genes": len(output),
        "missing_source_genes": unchanged,
        "base_sha256": digest_file(base_path),
        "enriched_sha256": digest_file(enriched_path),
        "genes_sha256": digest_file(genes_path) if genes_path else None,
        "output_sha256": digest_file(output_path),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest
