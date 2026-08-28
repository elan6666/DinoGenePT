"""Deterministic, sparse knowledge sections for GenePT-Seed text."""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

from .gradpert_union import DATASETS, _targets
from .provenance import atomic_write_json, digest_file, digest_text, utc_now

PROFILES = ("protein", "protein-pathway", "protein-pathway-hpa")
_SPACE = re.compile(r"\s+")


def _clean(value: str) -> str:
    return _SPACE.sub(" ", value.replace("\n", " ").strip())


def _first(row: dict[str, str], *names: str) -> str:
    for name in names:
        if row.get(name, "").strip():
            return _clean(row[name])
    return ""


def _bounded(values: Iterable[str], limit: int) -> list[str]:
    return sorted({_clean(value) for value in values if _clean(value)})[:limit]


def _clip(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return value[:limit].rsplit(" ", 1)[0].rstrip(";,. ") + "…", True


def load_interpro_names(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        result = {
            row["ENTRY_AC"]: f"{_clean(row['ENTRY_TYPE'])}: {_clean(row['ENTRY_NAME'])}"
            for row in rows
            if row.get("ENTRY_AC") and row.get("ENTRY_NAME")
        }
    if not result:
        raise ValueError("InterPro entry table contains no named entries")
    return result


def load_uniprot(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    genes: dict[str, dict[str, Any]] = {}
    accession_to_gene: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            gene = _first(row, "Gene Names (primary)", "Gene Names (Primary)", "Gene Names")
            gene = gene.split()[0] if gene else ""
            accession = _first(row, "Entry", "accession")
            if not gene or not accession or gene in genes:
                continue
            interpro_raw = _first(row, "InterPro", "Cross-reference (InterPro)")
            genes[gene] = {
                "accession": accession,
                "protein_name": _first(row, "Protein names", "Protein Name"),
                "subcellular_location": _first(row, "Subcellular location [CC]"),
                "catalytic_activity": _first(row, "Catalytic activity"),
                "cofactor": _first(row, "Cofactor"),
                "ptm": _first(row, "Post-translational modification"),
                "activity_regulation": _first(row, "Activity regulation"),
                "domain": _first(row, "Domain [FT]", "Domain"),
                "active_site": _first(row, "Active site", "Active site [FT]"),
                "binding_site": _first(row, "Binding site", "Binding site [FT]"),
                "interpro_ids": tuple(x for x in re.split(r"[;, ]+", interpro_raw) if x.startswith("IPR")),
            }
            accession_to_gene[accession] = gene
    if not genes:
        raise ValueError("UniProt table contains no primary human genes")
    return genes, accession_to_gene


def load_reactome(path: Path, accession_to_gene: dict[str, str]) -> dict[str, list[str]]:
    values: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 6 or columns[5] != "Homo sapiens":
                continue
            gene = accession_to_gene.get(columns[0])
            if gene:
                values[gene].append(columns[3])
    return values


def load_signor(path: Path, targets: set[str]) -> dict[str, list[str]]:
    values: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if _first(row, "TAX_ID") not in {"9606", "taxon:9606"}:
                continue
            if _first(row, "DIRECT").upper() not in {"YES", "TRUE", "T", "1"}:
                continue
            if _first(row, "TYPEA").lower() != "protein" or _first(row, "TYPEB").lower() != "protein":
                continue
            a, b = _first(row, "ENTITYA"), _first(row, "ENTITYB")
            effect, mechanism = _first(row, "EFFECT"), _first(row, "MECHANISM")
            if not a or not b or not effect or effect.lower() in {"unknown", "unspecified"}:
                continue
            detail = f"{a} {effect} {b}"
            if mechanism and mechanism.lower() not in {"unknown", "unspecified"}:
                detail += f" via {mechanism}"
            if a in targets:
                values[a].append("outgoing: " + detail)
            if b in targets:
                values[b].append("incoming: " + detail)
    return values


def _hpa_handle(path: Path) -> TextIO:
    archive = zipfile.ZipFile(path)
    member = archive.open(archive.namelist()[0])
    handle = io.TextIOWrapper(member, encoding="utf-8")
    handle._genept_archive = archive  # type: ignore[attr-defined]
    return handle


def load_hpa(path: Path) -> dict[str, list[str]]:
    handle = _hpa_handle(path) if zipfile.is_zipfile(path) else path.open(encoding="utf-8")
    values: dict[str, list[str]] = defaultdict(list)
    try:
        for row in csv.DictReader(handle, delimiter="\t"):
            gene = _first(row, "Gene", "Gene name")
            if not gene:
                continue
            fields = []
            for label, names in (
                ("tissue specificity", ("RNA tissue specificity", "Tissue specificity")),
                ("tissue distribution", ("RNA tissue distribution", "Tissue distribution")),
                ("single-cell specificity", ("RNA single cell type specificity", "Single cell type specificity")),
                ("subcellular location", ("Subcellular location", "Main location")),
            ):
                value = _first(row, *names)
                if value and value.lower() not in {"not detected", "n/a", "na"}:
                    fields.append(f"{label}: {value}")
            values[gene].extend(fields)
    finally:
        handle.close()
    return values


def build_knowledge_corpus(
    *,
    base_path: Path,
    genes_path: Path,
    uniprot_path: Path,
    interpro_path: Path,
    output_path: Path,
    manifest_path: Path,
    profile: str,
    reactome_path: Path | None = None,
    signor_path: Path | None = None,
    hpa_path: Path | None = None,
    max_items_per_source: int = 8,
    max_characters_per_field: int = 2000,
) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    if max_items_per_source < 1 or max_characters_per_field < 1:
        raise ValueError("knowledge section bounds must be positive")
    if profile in {"protein-pathway", "protein-pathway-hpa"} and (reactome_path is None or signor_path is None):
        raise ValueError("pathway profiles require both Reactome and SIGNOR")
    if profile == "protein-pathway-hpa" and hpa_path is None:
        raise ValueError("protein-pathway-hpa profile requires HPA")
    base_raw = json.loads(base_path.read_text(encoding="utf-8"))
    base = {str(k).upper(): str(v) for k, v in base_raw.items() if str(v).strip()}
    genes = [x.strip() for x in genes_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    missing = [gene for gene in genes if gene.upper() not in base]
    if missing:
        raise ValueError(f"base corpus is missing requested genes: {missing[:20]}")
    interpro = load_interpro_names(interpro_path)
    uniprot, accession_to_gene = load_uniprot(uniprot_path)
    targets = {gene.upper() for gene in genes}
    uniprot = {gene.upper(): row for gene, row in uniprot.items()}
    reactome = load_reactome(reactome_path, accession_to_gene) if reactome_path else {}
    reactome = {gene.upper(): values for gene, values in reactome.items()}
    signor = load_signor(signor_path, targets) if signor_path else {}
    signor = {gene.upper(): values for gene, values in signor.items()}
    hpa = load_hpa(hpa_path) if hpa_path else {}
    hpa = {gene.upper(): values for gene, values in hpa.items()}
    if profile in {"protein-pathway", "protein-pathway-hpa"} and not reactome:
        raise ValueError("Reactome produced no gene-mapped pathways")
    if profile in {"protein-pathway", "protein-pathway-hpa"} and not signor:
        raise ValueError("SIGNOR produced no direct human causal relations for the universe")
    if profile == "protein-pathway-hpa" and not hpa:
        raise ValueError("HPA produced no gene annotations")
    output: dict[str, str] = {}
    stats: Counter[str] = Counter()
    fingerprints: dict[str, str] = {}
    for gene in genes:
        key, sections = gene.upper(), []
        record = uniprot.get(key)
        if record:
            fields = []
            for label in (
                "protein_name",
                "subcellular_location",
                "catalytic_activity",
                "cofactor",
                "ptm",
                "activity_regulation",
                "domain",
                "active_site",
                "binding_site",
            ):
                if record[label]:
                    value, clipped = _clip(record[label], max_characters_per_field)
                    fields.append(f"{label.replace('_', ' ')}: {value}")
                    stats["clipped_uniprot_fields"] += int(clipped)
            if fields:
                sections.append("UniProt structured knowledge: " + "; ".join(fields))
                stats["genes_with_uniprot_structured"] += 1
            names = _bounded((interpro.get(i, "") for i in record["interpro_ids"]), max_items_per_source)
            if names:
                sections.append("InterPro entries: " + "; ".join(names))
                stats["genes_with_interpro"] += 1
        if profile in {"protein-pathway", "protein-pathway-hpa"}:
            pathways = _bounded(reactome.get(key, []), max_items_per_source)
            if pathways:
                sections.append("Reactome pathways: " + "; ".join(pathways))
                stats["genes_with_reactome"] += 1
            causal = _bounded(signor.get(key, []), max_items_per_source)
            if causal:
                sections.append("SIGNOR direct causal relations: " + "; ".join(causal))
                stats["genes_with_signor"] += 1
        if profile == "protein-pathway-hpa":
            atlas = _bounded(hpa.get(key, []), max_items_per_source)
            if atlas:
                sections.append("Human Protein Atlas: " + "; ".join(atlas))
                stats["genes_with_hpa"] += 1
        suffix = "" if not sections else "\n" + "\n".join(sections)
        output[gene] = base[key] + suffix
        if suffix:
            stats["genes_enriched"] += 1
            stats["added_characters"] += len(suffix)
            fingerprints[gene] = digest_text(suffix)
    atomic_write_json(output_path, output)
    sources = {"base": base_path, "genes": genes_path, "uniprot": uniprot_path, "interpro": interpro_path}
    for name, path in (("reactome", reactome_path), ("signor", signor_path), ("hpa", hpa_path)):
        if path:
            sources[name] = path
    manifest: dict[str, Any] = {
        "schema_version": "genept-seed-progressive-knowledge-corpus-v1",
        "created_at": utc_now(),
        "profile": profile,
        "genes": len(genes),
        "all_genes_preserved": len(output) == len(genes),
        "max_items_per_source": max_items_per_source,
        "max_characters_per_field": max_characters_per_field,
        "source_sha256": {name: digest_file(path) for name, path in sources.items()},
        "selection_stats": dict(sorted(stats.items())),
        "enrichment_fingerprint_sha256": digest_text(
            "\n".join(f"{g}\t{fingerprints[g]}" for g in sorted(fingerprints))
        ),
        "output_sha256": digest_file(output_path),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def audit_knowledge_corpora(
    *,
    genes_path: Path,
    gradpert_root: Path,
    base_path: Path,
    protein_path: Path,
    pathway_path: Path,
    hpa_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Prove exact universe coverage and append-only progressive text."""

    genes = [line.strip() for line in genes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = set(genes)
    paths = {
        "seed_go": base_path,
        "seed_go_protein": protein_path,
        "seed_go_protein_pathway": pathway_path,
        "seed_go_protein_pathway_hpa": hpa_path,
    }
    corpora: dict[str, dict[str, str]] = {}
    conditions: dict[str, Any] = {}
    for name, path in paths.items():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{name} corpus must be a JSON object")
        corpus = {str(gene): str(text) for gene, text in raw.items()}
        if set(corpus) != expected or list(corpus) != genes:
            raise ValueError(f"{name} corpus does not match the exact master allowlist")
        empty = [gene for gene, text in corpus.items() if not text.strip()]
        if empty:
            raise ValueError(f"{name} corpus contains empty text: {empty[:20]}")
        corpora[name] = corpus
        conditions[name] = {
            "genes": len(corpus),
            "empty_texts": 0,
            "sha256": digest_file(path),
        }
    transitions: dict[str, Any] = {}
    ordered = list(corpora)
    for previous_name, current_name in zip(ordered[:-1], ordered[1:], strict=True):
        previous, current = corpora[previous_name], corpora[current_name]
        not_prefix = [gene for gene in genes if not current[gene].startswith(previous[gene])]
        if not_prefix:
            raise ValueError(f"{current_name} is not append-only for genes: {not_prefix[:20]}")
        changed = sum(current[gene] != previous[gene] for gene in genes)
        transitions[f"{previous_name}_to_{current_name}"] = {
            "changed_genes": changed,
            "unchanged_genes": len(genes) - changed,
            "all_previous_text_is_prefix": True,
        }
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
                "protocol": protocol,
                "graph_genes": len(graph),
                "perturbation_targets": len(targets),
                "graph_missing_from_master": sorted(graph - expected),
                "targets_missing_from_master": sorted(targets - expected),
                "all_conditions_cover_graph": all(graph <= set(corpus) for corpus in corpora.values()),
                "all_conditions_cover_targets": all(targets <= set(corpus) for corpus in corpora.values()),
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": "genept-seed-progressive-corpus-audit-v1",
        "created_at": utc_now(),
        "master_genes": len(genes),
        "master_sha256": digest_file(genes_path),
        "conditions": conditions,
        "transitions": transitions,
        "datasets": datasets,
        "all_graph_axes_covered": all(row["all_conditions_cover_graph"] for row in datasets),
        "all_perturbation_targets_covered": all(
            row["all_conditions_cover_targets"] for row in datasets
        ),
    }
    atomic_write_json(output_path, receipt)
    return receipt
