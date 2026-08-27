"""Dataset parsers for selected GenePT paper tasks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PropertyTask:
    name: str
    genes: list[str]
    labels: list[int]


@dataclass(frozen=True)
class GGIDataset:
    train_pairs: list[tuple[str, str]]
    train_labels: list[int]
    test_pairs: list[tuple[str, str]]
    test_labels: list[int]


def load_property_tasks(path: Path) -> dict[str, PropertyTask]:
    grouped: dict[str, tuple[list[str], list[int]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"task", "gene", "label"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("property CSV requires task,gene,label columns")
        for row in reader:
            task = row["task"].strip()
            gene = row["gene"].strip().upper()
            label = int(row["label"])
            if label not in {0, 1}:
                raise ValueError("labels must be binary")
            genes, labels = grouped.setdefault(task, ([], []))
            genes.append(gene)
            labels.append(label)
    return {name: PropertyTask(name, genes, labels) for name, (genes, labels) in grouped.items()}


def _load_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.replace(",", " ").split()
        if len(fields) != 2:
            raise ValueError(f"expected two genes per line in {path.name}")
        pairs.append((fields[0].upper(), fields[1].upper()))
    return pairs


def _load_labels(path: Path) -> list[int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    labels = [int(line.strip()) for line in lines if line.strip()]
    if any(label not in {0, 1} for label in labels):
        raise ValueError(f"non-binary label in {path.name}")
    return labels


def load_ggi(directory: Path) -> GGIDataset:
    train_pairs = _load_pairs(directory / "train_text.txt")
    train_labels = _load_labels(directory / "train_label.txt")
    test_pairs = _load_pairs(directory / "test_text.txt")
    test_labels = _load_labels(directory / "test_label.txt")
    if len(train_pairs) != len(train_labels) or len(test_pairs) != len(test_labels):
        raise ValueError("GGI pair and label counts differ")
    return GGIDataset(train_pairs, train_labels, test_pairs, test_labels)


def ggi_genes(dataset: GGIDataset) -> set[str]:
    return {
        gene
        for pair in dataset.train_pairs + dataset.test_pairs
        for gene in pair
    }
