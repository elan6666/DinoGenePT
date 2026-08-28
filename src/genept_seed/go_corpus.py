"""Build a bounded GO experimental-annotation extension for GenePT text."""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .provenance import atomic_write_json, digest_file, digest_text, utc_now

SAFE_EXPERIMENTAL_EVIDENCE = frozenset({"EXP", "IDA", "IMP", "IEP", "HTP", "HDA", "HMP", "HEP"})
INTERACTION_EVIDENCE = frozenset({"IPI", "IGI", "HGI"})
ALL_EXPERIMENTAL_EVIDENCE = SAFE_EXPERIMENTAL_EVIDENCE | INTERACTION_EVIDENCE
EVIDENCE_PRIORITY = {
    "IDA": 0,
    "IMP": 1,
    "EXP": 2,
    "IEP": 3,
    "HDA": 4,
    "HMP": 5,
    "HTP": 6,
    "HEP": 7,
    "IPI": 8,
    "IGI": 9,
    "HGI": 10,
}
ASPECT_NAMES = {
    "F": "Molecular functions",
    "P": "Biological processes",
    "C": "Cellular components",
}
NAMESPACE_ASPECTS = {
    "molecular_function": "F",
    "biological_process": "P",
    "cellular_component": "C",
}
_QUOTED_DEFINITION = re.compile(r'^"((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class GOTerm:
    go_id: str
    name: str
    aspect: str
    definition: str
    parents: tuple[str, ...]


@dataclass(frozen=True)
class GOAnnotation:
    go_id: str
    aspect: str
    evidence: str


def _finish_obo_term(fields: dict[str, list[str]], terms: dict[str, GOTerm]) -> None:
    go_id = fields.get("id", [""])[0]
    name = fields.get("name", [""])[0]
    namespace = fields.get("namespace", [""])[0]
    if not go_id or not name or namespace not in NAMESPACE_ASPECTS:
        return
    if fields.get("is_obsolete", ["false"])[0] == "true":
        return
    raw_definition = fields.get("def", [""])[0]
    match = _QUOTED_DEFINITION.match(raw_definition)
    definition = bytes(match.group(1), "utf-8").decode("unicode_escape") if match else ""
    parents = tuple(value.split()[0] for value in fields.get("is_a", []))
    terms[go_id] = GOTerm(
        go_id=go_id,
        name=name,
        aspect=NAMESPACE_ASPECTS[namespace],
        definition=definition,
        parents=parents,
    )


def load_go_ontology(path: Path) -> dict[str, GOTerm]:
    terms: dict[str, GOTerm] = {}
    fields: dict[str, list[str]] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "[Term]":
            if fields is not None:
                _finish_obo_term(fields, terms)
            fields = defaultdict(list)
            continue
        if line.startswith("["):
            if fields is not None:
                _finish_obo_term(fields, terms)
            fields = None
            continue
        if fields is None or not line or line.startswith("!") or ": " not in line:
            continue
        key, value = line.split(": ", 1)
        fields[key].append(value)
    if fields is not None:
        _finish_obo_term(fields, terms)
    if not terms:
        raise ValueError("GO ontology contains no active terms")
    return terms


def load_go_annotations(
    path: Path,
    *,
    target_genes: set[str],
    evidence_codes: frozenset[str] = SAFE_EXPERIMENTAL_EVIDENCE,
) -> tuple[dict[str, list[GOAnnotation]], dict[str, Any]]:
    targets = {gene.upper() for gene in target_genes}
    annotations: dict[str, list[GOAnnotation]] = defaultdict(list)
    counts: Counter[str] = Counter()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("!"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 15:
                raise ValueError("GO GAF row has fewer than 15 columns")
            counts["rows"] += 1
            qualifiers = {value for value in columns[3].split("|") if value}
            if "NOT" in qualifiers:
                counts["excluded_not"] += 1
                continue
            evidence = columns[6]
            if evidence not in evidence_codes:
                if evidence in INTERACTION_EVIDENCE:
                    counts[f"excluded_{evidence}"] += 1
                continue
            primary = columns[2].upper()
            matched_gene: str | None = primary if primary in targets else None
            match_kind = "primary"
            if matched_gene is None:
                aliases = {alias.upper() for alias in columns[10].split("|") if alias}
                matches = sorted(aliases & targets)
                if len(matches) == 1:
                    matched_gene = matches[0]
                    match_kind = "synonym"
                elif len(matches) > 1:
                    counts["ambiguous_synonym_rows"] += 1
            if matched_gene is None:
                continue
            aspect = columns[8]
            if aspect not in ASPECT_NAMES:
                counts["invalid_aspect"] += 1
                continue
            annotations[matched_gene].append(GOAnnotation(columns[4], aspect, evidence))
            counts[f"evidence_{evidence}"] += 1
            counts[f"matched_{match_kind}"] += 1
    return dict(annotations), dict(sorted(counts.items()))


def _ancestor_function(terms: dict[str, GOTerm]):
    cache: dict[str, frozenset[str]] = {}

    def ancestors(go_id: str, trail: frozenset[str] = frozenset()) -> frozenset[str]:
        if go_id in cache:
            return cache[go_id]
        if go_id in trail or go_id not in terms:
            return frozenset()
        direct = {parent for parent in terms[go_id].parents if parent in terms}
        result = direct | set().union(
            *(ancestors(parent, trail | {go_id}) for parent in direct)
        ) if direct else set()
        cache[go_id] = frozenset(result)
        return cache[go_id]

    return ancestors


def select_go_terms(
    gene_annotations: list[GOAnnotation],
    terms: dict[str, GOTerm],
    *,
    max_terms_per_aspect: int,
) -> dict[str, list[GOTerm]]:
    if max_terms_per_aspect < 1:
        raise ValueError("max_terms_per_aspect must be positive")
    ancestors = _ancestor_function(terms)
    evidence_by_term: dict[str, set[str]] = defaultdict(set)
    aspect_by_term: dict[str, str] = {}
    for annotation in gene_annotations:
        term = terms.get(annotation.go_id)
        if term is None or term.aspect != annotation.aspect:
            continue
        evidence_by_term[annotation.go_id].add(annotation.evidence)
        aspect_by_term[annotation.go_id] = annotation.aspect
    selected: dict[str, list[GOTerm]] = {}
    for aspect in ASPECT_NAMES:
        candidates = {go_id for go_id, value in aspect_by_term.items() if value == aspect}
        redundant = {
            ancestor
            for descendant in candidates
            for ancestor in ancestors(descendant)
            if ancestor in candidates
        }
        specific = candidates - redundant
        ranked = sorted(
            specific,
            key=lambda go_id: (
                -len(ancestors(go_id)),
                min(EVIDENCE_PRIORITY[code] for code in evidence_by_term[go_id]),
                go_id,
            ),
        )
        selected[aspect] = [terms[go_id] for go_id in ranked[:max_terms_per_aspect]]
    return selected


def append_go_text(base_text: str, selected: dict[str, list[GOTerm]]) -> str:
    sections = []
    for aspect in ("F", "P", "C"):
        descriptions = []
        for term in selected.get(aspect, []):
            description = f"{term.name}: {term.definition}" if term.definition else term.name
            descriptions.append(description.rstrip("."))
        if descriptions:
            sections.append(f"{ASPECT_NAMES[aspect]}: " + "; ".join(descriptions) + ".")
    if not sections:
        return base_text
    prefix = "GO experimental annotations (interaction-derived evidence excluded)."
    return " ".join([base_text, prefix, *sections])


def build_go_exp_corpus(
    *,
    base_path: Path,
    genes_path: Path,
    gaf_path: Path,
    obo_path: Path,
    output_path: Path,
    manifest_path: Path,
    max_terms_per_aspect: int = 8,
    include_interaction_evidence: bool = False,
    allow_case_duplicates: bool = False,
) -> dict[str, Any]:
    raw = json.loads(base_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("base GenePT corpus must be a JSON object")
    base_by_normalized: dict[str, tuple[str, str]] = {}
    for raw_gene, raw_text in raw.items():
        gene = str(raw_gene)
        text = str(raw_text)
        if not text.strip():
            continue
        normalized = gene.upper()
        if normalized in base_by_normalized:
            previous = base_by_normalized[normalized][0]
            previous_text = base_by_normalized[normalized][1]
            if not allow_case_duplicates or previous_text != text:
                raise ValueError(f"base GenePT corpus has a case-insensitive collision: {previous}, {gene}")
            continue
        base_by_normalized[normalized] = (gene, text)
    genes_in_order = [
        line.strip()
        for line in genes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    normalized_genes = [gene.upper() for gene in genes_in_order]
    if not allow_case_duplicates and len(normalized_genes) != len(set(normalized_genes)):
        raise ValueError("gene allowlist contains a case-insensitive duplicate")
    missing = sorted(set(normalized_genes) - set(base_by_normalized))
    if missing:
        raise ValueError(f"base GenePT corpus is missing requested genes: {missing[:20]}")
    evidence_codes = ALL_EXPERIMENTAL_EVIDENCE if include_interaction_evidence else SAFE_EXPERIMENTAL_EVIDENCE
    ontology = load_go_ontology(obo_path)
    annotations, annotation_stats = load_go_annotations(
        gaf_path,
        target_genes=set(normalized_genes),
        evidence_codes=evidence_codes,
    )
    output: dict[str, str] = {}
    selected_term_counts: Counter[str] = Counter()
    selected_fingerprints: dict[str, str] = {}
    for gene, normalized in zip(genes_in_order, normalized_genes, strict=True):
        base_text = base_by_normalized[normalized][1]
        selected = select_go_terms(
            annotations.get(normalized, []),
            ontology,
            max_terms_per_aspect=max_terms_per_aspect,
        )
        output[gene] = append_go_text(base_text, selected)
        count = sum(len(values) for values in selected.values())
        if count:
            selected_term_counts["genes_with_go"] += 1
            selected_term_counts["terms"] += count
            for aspect, values in selected.items():
                selected_term_counts[f"terms_{aspect}"] += len(values)
            selected_fingerprints[gene] = digest_text(output[gene][len(base_text) :])
    atomic_write_json(output_path, output)
    lengths_added = [
        len(output[gene]) - len(base_by_normalized[gene.upper()][1])
        for gene in genes_in_order
    ]
    manifest: dict[str, Any] = {
        "schema_version": "genept-seed-go-exp-corpus-v1",
        "created_at": utc_now(),
        "condition": "GenePT-Seed-GOEXP",
        "interaction_evidence_included": include_interaction_evidence,
        "included_evidence_codes": sorted(evidence_codes),
        "excluded_interaction_evidence_codes": (
            [] if include_interaction_evidence else sorted(INTERACTION_EVIDENCE)
        ),
        "max_terms_per_aspect": max_terms_per_aspect,
        "base_sha256": digest_file(base_path),
        "genes_sha256": digest_file(genes_path),
        "gaf_sha256": digest_file(gaf_path),
        "ontology_sha256": digest_file(obo_path),
        "output_sha256": digest_file(output_path),
        "genes": len(genes_in_order),
        "case_duplicate_aliases_allowed": allow_case_duplicates,
        "ontology_terms": len(ontology),
        "annotation_stats": annotation_stats,
        "selection_stats": dict(sorted(selected_term_counts.items())),
        "added_characters_total": sum(lengths_added),
        "added_characters_max": max(lengths_added),
        "go_text_fingerprint_sha256": digest_text(
            "\n".join(f"{gene}\t{selected_fingerprints[gene]}" for gene in sorted(selected_fingerprints))
        ),
        "output_file": output_path.name,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest
