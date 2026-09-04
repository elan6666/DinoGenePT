"""Materialize an exact graph-axis GenePT corpus with HGNC-backed identity fallbacks."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .provenance import atomic_write_json, digest_file, digest_text, utc_now


@dataclass(frozen=True)
class HGNCRow:
    hgnc_id: str
    symbol: str
    name: str
    locus_group: str
    locus_type: str
    location: str
    entrez_id: str
    ensembl_gene_id: str
    uniprot_ids: str


def _split_symbols(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


class HGNCIndex:
    def __init__(self, path: Path) -> None:
        self.by_symbol: dict[str, list[HGNCRow]] = defaultdict(list)
        self.by_previous: dict[str, list[HGNCRow]] = defaultdict(list)
        self.by_alias: dict[str, list[HGNCRow]] = defaultdict(list)
        self.by_ensembl: dict[str, list[HGNCRow]] = defaultdict(list)
        with path.open(encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle, delimiter="\t"):
                if raw.get("status") != "Approved":
                    continue
                row = HGNCRow(
                    hgnc_id=raw.get("hgnc_id", ""),
                    symbol=raw.get("symbol", ""),
                    name=raw.get("name", ""),
                    locus_group=raw.get("locus_group", ""),
                    locus_type=raw.get("locus_type", ""),
                    location=raw.get("location", ""),
                    entrez_id=raw.get("entrez_id", ""),
                    ensembl_gene_id=raw.get("ensembl_gene_id", ""),
                    uniprot_ids=raw.get("uniprot_ids", ""),
                )
                self.by_symbol[row.symbol.upper()].append(row)
                for value in _split_symbols(raw.get("prev_symbol", "")):
                    self.by_previous[value.upper()].append(row)
                for value in _split_symbols(raw.get("alias_symbol", "")):
                    self.by_alias[value.upper()].append(row)
                for value in _split_symbols(row.ensembl_gene_id):
                    self.by_ensembl[value].append(row)

    def resolve(self, symbol: str, ensembl_gene_id: str | None) -> tuple[HGNCRow | None, str | None]:
        key = symbol.upper()
        for match_kind, index in (
            ("approved_symbol", self.by_symbol),
            ("previous_symbol", self.by_previous),
            ("alias_symbol", self.by_alias),
        ):
            rows = index.get(key, [])
            if len(rows) == 1:
                return rows[0], match_kind
            if len(rows) > 1 and ensembl_gene_id:
                exact = [row for row in rows if ensembl_gene_id in _split_symbols(row.ensembl_gene_id)]
                if len(exact) == 1:
                    return exact[0], f"{match_kind}+ensembl"
        if ensembl_gene_id:
            rows = self.by_ensembl.get(ensembl_gene_id, [])
            if len(rows) == 1:
                return rows[0], "ensembl_gene_id"
        return None, None


def compose_hgnc_identity_text(requested_symbol: str, row: HGNCRow) -> str:
    pieces = [f"Gene Symbol {requested_symbol}"]
    if row.symbol.upper() != requested_symbol.upper():
        pieces.append(f"Current HGNC symbol: {row.symbol}.")
    if row.name:
        pieces.append(row.name.rstrip(".") + ".")
    if row.locus_group or row.locus_type:
        label = row.locus_type or row.locus_group
        pieces.append(f"HGNC locus type: {label}.")
    if row.ensembl_gene_id:
        pieces.append(f"Ensembl gene ID: {row.ensembl_gene_id}.")
    if row.location:
        pieces.append(f"Chromosomal location: {row.location}.")
    return " ".join(pieces)


def _load_ensembl_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        columns = line.split()
        if len(columns) != 3:
            raise ValueError(f"axis mapping line {line_number} must contain SYMBOL NCBI_ID|- ENSEMBL_ID")
        if columns[0] in result:
            raise ValueError(f"duplicate axis mapping symbol: {columns[0]}")
        if columns[2].startswith("ENSG"):
            result[columns[0]] = columns[2]
    return result


def build_axis_corpus(
    *,
    source_path: Path,
    genes_path: Path,
    hgnc_path: Path,
    output_path: Path,
    manifest_path: Path,
    axis_mapping_path: Path | None = None,
    ensembl_archive_path: Path | None = None,
    allow_case_duplicates: bool = False,
    allow_identity_only: bool = False,
) -> dict[str, Any]:
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("source GenePT corpus must be a JSON object")
    source: dict[str, tuple[str, str]] = {}
    for raw_gene, raw_text in raw.items():
        gene, text = str(raw_gene), str(raw_text)
        normalized = gene.upper()
        if normalized in source:
            raise ValueError(f"source corpus has a case-insensitive collision: {source[normalized][0]}, {gene}")
        if text.strip():
            source[normalized] = (gene, text)
    genes = [line.strip() for line in genes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not allow_case_duplicates and len({gene.upper() for gene in genes}) != len(genes):
        raise ValueError("graph axis has a case-insensitive duplicate")
    ensembl_by_gene = _load_ensembl_map(axis_mapping_path)
    ensembl_archive = (
        json.loads(ensembl_archive_path.read_text(encoding="utf-8"))
        if ensembl_archive_path is not None
        else {}
    )
    hgnc = HGNCIndex(hgnc_path)
    output: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    unresolved: list[str] = []
    for gene in genes:
        normalized = gene.upper()
        if normalized in source:
            source_gene, text = source[normalized]
            source_kind = "exact_genept" if source_gene == gene else "case_normalized_genept"
            hgnc_row = None
            match_kind = None
        else:
            hgnc_row, match_kind = hgnc.resolve(gene, ensembl_by_gene.get(gene))
            if hgnc_row is None:
                archived = ensembl_archive.get(gene)
                if not isinstance(archived, dict):
                    if allow_identity_only:
                        text = f"Gene Symbol {gene}"
                        source_kind = "axis_identity_only"
                        output[gene] = text
                        stats[source_kind] += 1
                        records.append(
                            {
                                "gene": gene,
                                "source": source_kind,
                                "hgnc_id": None,
                                "hgnc_match_kind": None,
                                "hgnc_approved_symbol": None,
                                "text_sha256": digest_text(text),
                                "text_characters": len(text),
                            }
                        )
                        continue
                    unresolved.append(gene)
                    continue
                ensembl_id = str(archived.get("id", ensembl_by_gene.get(gene, "")))
                assembly = str(archived.get("assembly", ""))
                release = str(archived.get("release", ""))
                text = f"Gene Symbol {gene} Ensembl stable gene ID: {ensembl_id}."
                details = "This is an archived Ensembl gene"
                if assembly:
                    details += f" on {assembly}"
                if release:
                    details += f"; last recorded in release {release}"
                text += f" {details}."
                source_kind = "ensembl_archive_identity"
                match_kind = "archived_ensembl_id"
                output[gene] = text
                stats[source_kind] += 1
                records.append(
                    {
                        "gene": gene,
                        "source": source_kind,
                        "hgnc_id": None,
                        "hgnc_match_kind": match_kind,
                        "hgnc_approved_symbol": None,
                        "text_sha256": digest_text(text),
                        "text_characters": len(text),
                    }
                )
                continue
            approved = source.get(hgnc_row.symbol.upper())
            if approved is not None:
                text = (
                    f"Gene Symbol {gene} Current HGNC symbol: {hgnc_row.symbol}. "
                    f"{approved[1]}"
                )
                source_kind = "hgnc_alias_to_genept"
            else:
                text = compose_hgnc_identity_text(gene, hgnc_row)
                source_kind = "hgnc_identity"
        output[gene] = text
        stats[source_kind] += 1
        records.append(
            {
                "gene": gene,
                "source": source_kind,
                "hgnc_id": hgnc_row.hgnc_id if hgnc_row else None,
                "hgnc_match_kind": match_kind,
                "hgnc_approved_symbol": hgnc_row.symbol if hgnc_row else None,
                "text_sha256": digest_text(text),
                "text_characters": len(text),
            }
        )
    if unresolved:
        raise ValueError(f"unable to build source-grounded text for graph-axis genes: {unresolved}")
    atomic_write_json(output_path, output)
    manifest: dict[str, Any] = {
        "schema_version": "genept-seed-axis-corpus-v1",
        "created_at": utc_now(),
        "source_sha256": digest_file(source_path),
        "genes_sha256": digest_file(genes_path),
        "hgnc_sha256": digest_file(hgnc_path),
        "axis_mapping_sha256": digest_file(axis_mapping_path) if axis_mapping_path else None,
        "ensembl_archive_sha256": (
            digest_file(ensembl_archive_path) if ensembl_archive_path else None
        ),
        "output_sha256": digest_file(output_path),
        "genes": len(genes),
        "case_duplicate_aliases_allowed": allow_case_duplicates,
        "identity_only_fallback_allowed": allow_identity_only,
        "source_counts": dict(sorted(stats.items())),
        "records": records,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest
