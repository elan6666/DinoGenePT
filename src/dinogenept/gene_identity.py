"""Frozen human gene identities and conservative, order-preserving source joins.

HGNC field semantics: https://www.genenames.org/help/statistics-and-downloads/
This is a new opt-in adapter; legacy artifacts and running models are unchanged.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .axis_corpus import HGNCIndex, _split_symbols
from .provenance import atomic_write_json, digest_file, digest_text

POLICY = "dinogenept.human-identity.v1"
ENSEMBL = re.compile(r"^(ENSG\d{11})(?:\.\d+)?$")


def read_identity_json(path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in identity input: {key}")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(), object_pairs_hook=unique_pairs)


def stable_ensembl(value):
    match = ENSEMBL.fullmatch(value.strip()) if isinstance(value, str) else None
    return match[1] if match else None


class GeneIdentityIndex:
    """One frozen HGNC snapshot shared by axes, vocabularies and all sources."""

    def __init__(self, path, *, expected_sha256):
        self.path = Path(path)
        self.sha256 = digest_file(self.path)
        if self.sha256 != expected_sha256:
            raise ValueError("HGNC snapshot checksum mismatch")
        legacy = HGNCIndex(self.path)
        self.rows = {}
        self.names = [defaultdict(set) for _ in range(3)]
        self.ensembl = defaultdict(set)
        for group in legacy.by_symbol.values():
            for row in group:
                if not row.hgnc_id or not row.symbol or row.hgnc_id in self.rows:
                    raise ValueError("HGNC approved rows need unique nonempty IDs and symbols")
                self.rows[row.hgnc_id] = row
                for value in _split_symbols(row.ensembl_gene_id):
                    stable = stable_ensembl(value)
                    if stable:
                        self.ensembl[stable].add(row.hgnc_id)
        if not self.rows:
            raise ValueError("HGNC snapshot contains no approved genes")
        for target, source in zip(self.names, [legacy.by_symbol, legacy.by_previous, legacy.by_alias], strict=True):
            for name, rows in source.items():
                target[name].update(row.hgnc_id for row in rows)

    def resolve(self, label, ensembl_id=None):
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Every axis record requires a nonempty original label")
        inferred = stable_ensembl(label)
        explicit = ensembl_id is not None and ensembl_id != ""
        supplied = ensembl_id if explicit else (label if inferred else None)
        stable = stable_ensembl(supplied)
        result = {
            "original_label": label, "original_ensembl_id": supplied,
            "ensembl_id": stable, "hgnc_id": None, "approved_symbol": None,
            "status": "unresolved", "match": None, "candidates": [],
        }
        if supplied is not None and stable is None:
            return {**result, "status": "invalid_ensembl_id"}
        if inferred and explicit and inferred != stable:
            return {**result, "status": "conflict"}
        named, kind = set(), None
        if not inferred:
            for index, field in zip(self.names, ["approved_symbol", "previous_symbol", "alias_symbol"], strict=True):
                if index.get(label.strip().upper()):
                    named, kind = index[label.strip().upper()], field
                    break  # Approved names take priority over another gene's historical alias.
        by_id = self.ensembl.get(stable, set())
        candidates = named | by_id
        result["candidates"] = sorted(candidates)
        if stable:
            if not by_id:
                return {**result, "status": "unverified_ensembl_id"}
            selected = by_id & named if named else by_id
            if named and not selected:
                return {**result, "status": "conflict"}
            kind = f"{kind}+ensembl" if named else "ensembl_id"
        else:
            selected = named
        if len(selected) != 1:
            return {**result, "status": "ambiguous" if selected else "unresolved"}
        row = self.rows[next(iter(selected))]
        return {
            **result, "status": "resolved", "match": kind,
            "hgnc_id": row.hgnc_id, "approved_symbol": row.symbol,
            "hgnc_ensembl_ids": _split_symbols(row.ensembl_gene_id),
        }

    def axis(self, records):
        rows, identities, labels = [], defaultdict(list), defaultdict(list)
        for position, item in enumerate(records):
            if isinstance(item, str):
                item = {"label": item}
            if not isinstance(item, dict) or "label" not in item:
                raise ValueError("Axis must contain labels or objects with label and optional ensembl_id")
            row = {**self.resolve(item["label"], item.get("ensembl_id")), "position": position}
            rows.append(row)
            labels[row["original_label"]].append(position)
            if row["hgnc_id"]:
                identities[row["hgnc_id"]].append(position)
        if not rows:
            raise ValueError("Empty gene axis")
        duplicates = {k: v for k, v in identities.items() if len(v) > 1}
        duplicate_labels = {k: v for k, v in labels.items() if len(v) > 1}
        for row in rows:
            row["duplicate_identity"] = row["hgnc_id"] in duplicates
        return {
            "schema": POLICY, "hgnc_sha256": self.sha256, "rows": rows,
            "counts": dict(Counter(r["status"] for r in rows)),
            "duplicate_identities": duplicates, "duplicate_labels": duplicate_labels,
            "safe_unique_axis": not duplicates and not duplicate_labels
            and all(r["status"] == "resolved" for r in rows),
        }


class SourceIdentityIndex:
    """Index unchanged source text; conflicting variants never pick a winner."""

    def __init__(self, identity, corpus):
        if not isinstance(corpus, dict):
            raise ValueError("Source corpus must be a gene-to-text JSON object")
        self.groups = defaultdict(list)
        self.uncertain = defaultdict(list)
        self.unresolved = []
        for key, text in corpus.items():
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty/non-string source text")
            row = identity.resolve(key)
            if row["status"] != "resolved":
                self.unresolved.append(row)
                for candidate in row["candidates"]:
                    self.uncertain[candidate].append(key)
            else:
                self.groups[row["hgnc_id"]].append({"key": key, "text_sha256": digest_text(text)})

    def join(self, row):
        result = {"status": "identity_unresolved", "source_key": None, "text_sha256": None}
        if row["status"] != "resolved":
            return result
        if self.uncertain.get(row["hgnc_id"]):
            return {**result, "status": "source_identity_ambiguous",
                    "unresolved_keys": sorted(self.uncertain[row["hgnc_id"]])}
        entries = sorted(self.groups.get(row["hgnc_id"], []), key=lambda e: e["key"])
        if not entries:
            return {**result, "status": "no_resolved_source_record"}
        if len({e["text_sha256"] for e in entries}) > 1:
            return {**result, "status": "source_conflict", "variants": entries}
        preferred = sorted(entries, key=lambda e: (e["key"] != row["approved_symbol"], e["key"]))[0]
        return {
            "status": "available", "source_key": preferred["key"],
            "text_sha256": preferred["text_sha256"], "equivalent_keys": [e["key"] for e in entries],
        }


def build_identity_report(identity, axis_records, corpora):
    report = identity.axis(axis_records)
    report["sources"] = {}
    for name, corpus in corpora.items():
        source = SourceIdentityIndex(identity, corpus)
        joined = [source.join(row) for row in report["rows"]]
        report["sources"][name] = {
            "counts": dict(Counter(x["status"] for x in joined)),
            "unresolved_source_keys": source.unresolved,
            "joins": joined,
        }
    report["note"] = "Source availability is not functional-annotation or vector coverage; no text/array was rewritten."
    return report


def audit_identity_files(*, hgnc, hgnc_sha256, axis, sources, output):
    """Audit only. Fresh JSON output, no API calls or tensor materialization."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh identity report path")
    index = GeneIdentityIndex(hgnc, expected_sha256=hgnc_sha256)
    paths = {name: Path(path) for name, path in sources.items()}
    report = build_identity_report(
        index, read_identity_json(axis),
        {name: read_identity_json(path) for name, path in paths.items()},
    )
    report["inputs"] = {
        "axis": {"path": str(axis), "sha256": digest_file(axis)},
        "hgnc": {"path": str(hgnc), "sha256": index.sha256},
        "sources": {name: {"path": str(p), "sha256": digest_file(p)} for name, p in paths.items()},
    }
    atomic_write_json(output, report)
    return {
        "output": str(output), "sha256": digest_file(output),
        "axis_genes": len(report["rows"]), "counts": report["counts"],
        "duplicate_identities": len(report["duplicate_identities"]),
        "safe_unique_axis": report["safe_unique_axis"],
        "sources": {name: value["counts"] for name, value in report["sources"].items()},
    }
