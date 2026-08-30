"""Leakage controls and deterministic gene-disjoint GGI evaluation."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .benchmarks import evaluate_ggi
from .provenance import atomic_write_json, digest_file, digest_text
from .tasks import GGIDataset


def filter_ggi_universe(dataset: GGIDataset, universe: set[str]) -> tuple[GGIDataset, dict]:
    requested = {gene.upper() for gene in universe}

    def selected(
        pairs: list[tuple[str, str]], labels: list[int]
    ) -> tuple[list[tuple[str, str]], list[int]]:
        rows = [
            (pair, label)
            for pair, label in zip(pairs, labels, strict=True)
            if pair[0] in requested and pair[1] in requested
        ]
        return [pair for pair, _ in rows], [label for _, label in rows]

    train_pairs, train_labels = selected(dataset.train_pairs, dataset.train_labels)
    test_pairs, test_labels = selected(dataset.test_pairs, dataset.test_labels)
    observed = {gene for pair in train_pairs + test_pairs for gene in pair}
    isolated = sorted(requested - observed)
    filtered = GGIDataset(train_pairs, train_labels, test_pairs, test_labels)
    receipt = {
        "requested_genes": len(requested),
        "observed_genes": len(observed),
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "dropped_train_pairs": len(dataset.train_pairs) - len(train_pairs),
        "dropped_test_pairs": len(dataset.test_pairs) - len(test_pairs),
        "isolated_genes": len(isolated),
        "isolated_gene_examples": isolated[:20],
        "all_pairs_within_universe": observed.issubset(requested),
    }
    return filtered, receipt


def gene_disjoint_split(
    dataset: GGIDataset,
    *,
    seed: int,
    test_fraction: float = 0.2,
    universe: set[str] | None = None,
) -> tuple[GGIDataset, dict]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    rows = list(zip(dataset.train_pairs + dataset.test_pairs, dataset.train_labels + dataset.test_labels, strict=True))
    observed = {gene for pair, _ in rows for gene in pair}
    genes = sorted({gene.upper() for gene in universe} if universe is not None else observed)
    if not observed.issubset(genes):
        raise ValueError("GGI rows contain genes outside the requested partition universe")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(genes, dtype=object)
    rng.shuffle(shuffled)
    test_n = max(1, round(len(genes) * test_fraction))
    test_genes = set(shuffled[:test_n].tolist())
    train_genes = set(genes) - test_genes
    train_rows = [(pair, label) for pair, label in rows if pair[0] in train_genes and pair[1] in train_genes]
    test_rows = [(pair, label) for pair, label in rows if pair[0] in test_genes and pair[1] in test_genes]
    dropped = len(rows) - len(train_rows) - len(test_rows)
    if len({label for _, label in train_rows}) != 2 or len({label for _, label in test_rows}) != 2:
        raise ValueError(f"gene-disjoint split {seed} does not retain both labels")
    split = GGIDataset(
        [pair for pair, _ in train_rows], [label for _, label in train_rows],
        [pair for pair, _ in test_rows], [label for _, label in test_rows],
    )
    receipt = {
        "seed": seed,
        "test_fraction": test_fraction,
        "genes": len(genes),
        "train_genes": len(train_genes),
        "test_genes": len(test_genes),
        "train_gene_sha256": digest_text("\n".join(sorted(train_genes))),
        "test_gene_sha256": digest_text("\n".join(sorted(test_genes))),
        "train_pairs": len(train_rows),
        "test_pairs": len(test_rows),
        "cross_partition_pairs_dropped": dropped,
        "train_labels": dict(sorted(Counter(label for _, label in train_rows).items())),
        "test_labels": dict(sorted(Counter(label for _, label in test_rows).items())),
        "gene_overlap": len(train_genes & test_genes),
    }
    return split, receipt


def evaluate_gene_disjoint_ggi(
    dataset: GGIDataset,
    vectors: Mapping[str, np.ndarray],
    *,
    seeds: Sequence[int],
    test_fraction: float = 0.2,
    universe: set[str] | None = None,
) -> list[dict[str, object]]:
    rows = []
    for seed in seeds:
        split, receipt = gene_disjoint_split(
            dataset, seed=seed, test_fraction=test_fraction, universe=universe
        )
        result = evaluate_ggi(split, vectors, random_state=seed)
        rows.append({"split": "gene_disjoint", "split_receipt": receipt, "random_state": seed, **result})
    return rows


def _signor_pairs(path: Path) -> set[tuple[str, str]]:
    pairs = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            direct = row.get("DIRECT", "").strip().upper()
            taxon = row.get("TAX_ID", "").strip()
            if direct not in {"YES", "TRUE", "T", "1"} or taxon not in {"9606", "taxon:9606"}:
                continue
            if row.get("TYPEA", "").strip().lower() != "protein" or row.get("TYPEB", "").strip().lower() != "protein":
                continue
            a, b = row.get("ENTITYA", "").strip().upper(), row.get("ENTITYB", "").strip().upper()
            if a and b:
                pairs.add(tuple(sorted((a, b))))
    return pairs


def _signor_sections(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("knowledge corpus must be a JSON object")
    marker = "SIGNOR direct causal relations: "
    sections = {}
    for gene, text in raw.items():
        value = str(text)
        if marker not in value:
            continue
        sections[str(gene).upper()] = value.split(marker, 1)[1].split("\n", 1)[0]
    return sections


def _mentions(text: str, symbol: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def audit_signor_ggi_leakage(
    *,
    dataset: GGIDataset,
    signor_path: Path,
    output_path: Path,
    corpus_path: Path | None = None,
) -> dict:
    signor = _signor_pairs(signor_path)
    signor_self_pairs = {pair for pair in signor if pair[0] == pair[1]}
    signor_nonself_pairs = signor - signor_self_pairs
    corpus_sections = _signor_sections(corpus_path) if corpus_path else {}
    text_exposed_pairs = (
        {
            pair
            for pair in signor_nonself_pairs
            if _mentions(corpus_sections.get(pair[0], ""), pair[1])
            or _mentions(corpus_sections.get(pair[1], ""), pair[0])
        }
        if corpus_path
        else set()
    )
    sections = {}
    for name, pairs, labels in (
        ("train", dataset.train_pairs, dataset.train_labels),
        ("test", dataset.test_pairs, dataset.test_labels),
    ):
        by_label = {}
        for label in (0, 1):
            selected = [pair for pair, value in zip(pairs, labels, strict=True) if value == label]
            source_overlap = sum(tuple(sorted(pair)) in signor for pair in selected)
            text_overlap = None
            if corpus_path:
                text_overlap = sum(tuple(sorted(pair)) in text_exposed_pairs for pair in selected)
            by_label[str(label)] = {
                "pairs": len(selected),
                "source_signor_overlap": source_overlap,
                "source_rate": source_overlap / len(selected) if selected else None,
                "text_exposed_signor_overlap": text_overlap,
                "text_exposed_rate": (
                    text_overlap / len(selected) if text_overlap is not None and selected else None
                ),
            }
        sections[name] = by_label
    report = {
        "schema_version": "genept-seed-signor-ggi-leakage-v3",
        "signor_sha256": digest_file(signor_path),
        "corpus_sha256": digest_file(corpus_path) if corpus_path else None,
        "direct_human_protein_pairs": len(signor),
        "direct_self_pairs": len(signor_self_pairs),
        "text_exposure_definition": "non-self partner symbol in the bounded SIGNOR section",
        "text_exposed_direct_pairs": len(text_exposed_pairs) if corpus_path else None,
        "sections": sections,
    }
    atomic_write_json(output_path, report)
    return report
