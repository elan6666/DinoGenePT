"""Pinned preparation of the four gene-property tasks used by GenePT."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from .data import download, verify_sha256
from .provenance import atomic_write_json, digest_file, utc_now

GENEPT_COMMIT = "3602699e7425a7be577771f8f07e218db6c79b9f"
GENECORPUS_REVISION = "eab40d350be70ca6dff2d42555bacc44c8a71cc7"
LONG_SHORT_URL = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-16106-x/"
    "MediaObjects/41467_2020_16106_MOESM4_ESM.csv"
)
HF_ROOT = f"https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/resolve/{GENECORPUS_REVISION}"
HF_MIRROR_ROOT = f"https://hf-mirror.com/datasets/ctheodoris/Genecorpus-30M/resolve/{GENECORPUS_REVISION}"
PROPERTY_FILES = {
    "long_short_tf.csv": LONG_SHORT_URL,
    "dosage_sensitive_tf.csv": (
        f"{HF_ROOT}/example_input_files/gene_classification/dosage_sensitive_tfs/"
        "dosage_sens_tf_labels.csv?download=true"
    ),
    "bivalent.txt": (
        f"{HF_ROOT}/example_input_files/gene_classification/bivalent_promoters/"
        "bivalent_gene_labels.txt?download=true"
    ),
    "lys4_only.txt": (
        f"{HF_ROOT}/example_input_files/gene_classification/bivalent_promoters/"
        "lys4_only_gene_labels.txt?download=true"
    ),
    "no_methylation.txt": (
        f"{HF_ROOT}/example_input_files/gene_classification/bivalent_promoters/"
        "no_methylation_gene_labels.txt?download=true"
    ),
}
PROPERTY_SHA256 = {
    "long_short_tf.csv": "d0f095bbdbe2e07b0e8c654831a531a8f40b4f4d63b74ad53b81efb583cda6ae",
    "dosage_sensitive_tf.csv": "6a3db1b9bd2c4bd1ddf9ed09152722f05cb1def24624d4badf7e559d9a4ba46d",
    "bivalent.txt": "c8c2c7a4b71317d70b00a3f0bcb41727912521076f0644b5c5855a28de930266",
    "lys4_only.txt": "8f65a830ce094f1c84d05273e390347bf69b90da9503d23e2e67362737022343",
    "no_methylation.txt": "c71a82b53e07539cbed78988cd8b78daa038661f15a98d46bc6b6c1a79433fe9",
}
_ENSG = re.compile(r"ENSG\d+(?:\.\d+)?")


def _hgnc_ensembl_map(path: Path) -> dict[str, str]:
    mapping = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            symbol = row.get("symbol", "").strip().upper()
            if not symbol:
                continue
            for ensembl in re.split(r"[|,; ]+", row.get("ensembl_gene_id", "")):
                if ensembl:
                    mapping[ensembl.split(".")[0]] = symbol
    if not mapping:
        raise ValueError("HGNC file contains no Ensembl-to-symbol mappings")
    return mapping


def _ensembl_ids(path: Path) -> list[str]:
    return [match.split(".")[0] for match in _ENSG.findall(path.read_text(encoding="utf-8"))]


def _long_short(path: Path) -> tuple[list[str], list[str]]:
    positive, negative = [], []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            gene = (row.get("") or row.get("Unnamed: 0") or next(iter(row.values())) or "").strip().upper()
            assignment = row.get("assignment", "").strip().lower()
            if gene and assignment == "long-range tf":
                positive.append(gene)
            elif gene and assignment == "short-range tf":
                negative.append(gene)
    return positive, negative


def _dosage(path: Path) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dosage-sensitive TF source is empty")
    names = {name.lower().strip(): name for name in (rows[0].keys() if rows else []) if name}
    positive_name = names.get("dosage_sensitive")
    negative_name = names.get("dosage_insensitive")
    if not positive_name or not negative_name:
        raise ValueError(f"unexpected dosage-sensitive TF columns: {sorted(names)}")
    positive = [row[positive_name].strip().upper() for row in rows if row.get(positive_name, "").strip()]
    negative = [row[negative_name].strip().upper() for row in rows if row.get(negative_name, "").strip()]
    return positive, negative


def _map_ensembl(values: list[str], mapping: dict[str, str]) -> list[str]:
    return [mapping.get(value.split(".")[0], value) for value in values]


def _clean_binary(positive: list[str], negative: list[str]) -> tuple[list[str], list[str], list[str]]:
    overlap = sorted(set(positive) & set(negative))
    blocked = set(overlap)
    return (
        sorted(set(positive) - blocked),
        sorted(set(negative) - blocked),
        overlap,
    )


def prepare_property_tasks(destination: Path, *, hgnc_path: Path, clean_overlaps: bool = True) -> dict:
    """Download pinned labels, map Ensembl IDs, and write one matched task table."""

    source_dir = destination / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_paths = {}
    download_endpoints = {}
    for name, url in PROPERTY_FILES.items():
        path = source_dir / name
        if not path.exists() or path.stat().st_size == 0:
            try:
                download(url, path, max_retries=0)
                download_endpoints[name] = url
            except Exception:
                mirror_url = url.replace(HF_ROOT, HF_MIRROR_ROOT)
                if mirror_url == url:
                    raise
                download(mirror_url, path, max_retries=2)
                download_endpoints[name] = mirror_url
        else:
            download_endpoints[name] = "pre-existing file; content identified by SHA-256"
        verify_sha256(path, PROPERTY_SHA256[name])
        source_paths[name] = path
    ensembl = _hgnc_ensembl_map(hgnc_path)
    bivalent_ids = _ensembl_ids(source_paths["bivalent.txt"])
    lys4_ids = _ensembl_ids(source_paths["lys4_only.txt"])
    no_methylation_ids = _ensembl_ids(source_paths["no_methylation.txt"])
    unresolved = sorted(
        set(bivalent_ids + lys4_ids + no_methylation_ids) - set(ensembl)
    )
    dosage_positive, dosage_negative = _dosage(source_paths["dosage_sensitive_tf.csv"])
    task_classes = {
        "long_range_vs_short_range_tf": _long_short(source_paths["long_short_tf.csv"]),
        "dosage_sensitive_vs_insensitive_tf": (
            _map_ensembl(dosage_positive, ensembl),
            _map_ensembl(dosage_negative, ensembl),
        ),
        "bivalent_vs_non_methylated": (
            [ensembl[x] for x in bivalent_ids if x in ensembl],
            [ensembl[x] for x in no_methylation_ids if x in ensembl],
        ),
        "bivalent_vs_lys4_only": (
            [ensembl[x] for x in bivalent_ids if x in ensembl],
            [ensembl[x] for x in lys4_ids if x in ensembl],
        ),
    }
    rows = []
    task_stats = {}
    for task, (positive, negative) in task_classes.items():
        overlap: list[str] = []
        if clean_overlaps:
            positive, negative, overlap = _clean_binary(positive, negative)
        else:
            positive, negative = sorted(set(positive)), sorted(set(negative))
        for gene in positive:
            rows.append((task, gene, 1))
        for gene in negative:
            rows.append((task, gene, 0))
        task_stats[task] = {
            "positive": len(positive),
            "negative": len(negative),
            "class_overlap_removed": len(overlap),
        }
    output = destination / "genept_property_tasks.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("task", "gene", "label"))
        writer.writerows(rows)
    manifest = {
        "schema_version": "genept-seed-property-tasks-v1",
        "prepared_at": utc_now(),
        "genept_reference_commit": GENEPT_COMMIT,
        "genecorpus_revision": GENECORPUS_REVISION,
        "clean_overlaps": clean_overlaps,
        "source_sha256": {name: digest_file(path) for name, path in source_paths.items()},
        "canonical_source_urls": PROPERTY_FILES,
        "download_endpoints": download_endpoints,
        "hgnc_sha256": digest_file(hgnc_path),
        "unresolved_ensembl": len(unresolved),
        "unresolved_examples": unresolved[:20],
        "tasks": task_stats,
        "label_counts": dict(sorted(Counter(label for _, _, label in rows).items())),
        "output_sha256": digest_file(output),
    }
    atomic_write_json(destination / "property_tasks_manifest.json", manifest)
    return manifest
