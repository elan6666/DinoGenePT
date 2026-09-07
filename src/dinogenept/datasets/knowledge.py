"""Exact, provenance-checked source-only vectors for perturbation locals.

This loader never calls an API or substitutes another gene/source. A combination
local exists only when every target has that source. Missing optional annotation
is represented by None, never a zero vector or a repeated TextBase vector.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from dinogenept.provenance import digest_file
from dinogenept.vectors import load_npz

KNOWLEDGE_SOURCES = ("TextBase", "GO", "Protein", "Pathway", "HPA")


def _read_pinned(entry):
    path = Path(entry["path"])
    if digest_file(path) != entry["sha256"]:
        raise ValueError(f"Knowledge artifact checksum mismatch: {path.name}")
    return path


def _fingerprint(corpus):
    digest = hashlib.sha256()
    for gene, text in sorted(corpus.items()):
        digest.update(gene.encode() + b"\0" + text.encode() + b"\0")
    return digest.hexdigest()


class KnowledgeBank:
    def __init__(self, sources, required_genes, *, model, width=2048):
        if set(sources) != set(KNOWLEDGE_SOURCES):
            raise ValueError("Explicit corpus entries required for TextBase and all four optional sources")
        required = set(required_genes)
        if not required or any(not isinstance(g, str) or not g for g in required) or width < 1 or not model:
            raise ValueError("Invalid knowledge universe/model/width")
        self.tables, self.coverage = {}, {}
        for source in KNOWLEDGE_SOURCES:
            entry = sources[source]
            corpus_path = _read_pinned(entry["corpus"])
            corpus = json.loads(corpus_path.read_text())
            if not isinstance(corpus, dict) or any(
                not gene or not isinstance(text, str) or not text.strip() for gene, text in corpus.items()
            ):
                raise ValueError("Source corpus must contain only nonempty exact gene/text entries")
            if source != "TextBase":
                proof = json.loads(_read_pinned(entry["source_manifest"]).read_text())
                if (
                    proof.get("schema_version") != "genept-seed-source-only-corpus-v1"
                    or proof.get("source") != source
                    or proof.get("output_sha256") != entry["corpus"]["sha256"]
                    or proof.get("available_genes") != len(corpus)
                ):
                    raise ValueError("Optional local needs matching source-only corpus provenance, not cumulative text")
            elif required - set(corpus):
                raise ValueError(f"TextBase lacks required genes: {sorted(required - set(corpus))[:20]}")
            if not corpus:
                if entry.get("vectors") is not None or entry.get("embedding_manifest") is not None:
                    raise ValueError("An empty source must not have fabricated vector artifacts")
                table = {}
            else:
                vector_path = _read_pinned(entry["vectors"])
                manifest = json.loads(_read_pinned(entry["embedding_manifest"]).read_text())
                if (
                    manifest.get("schema_version") != "genept-seed-embedding-v2"
                    or manifest.get("model") != model
                    or manifest.get("dimension") != width
                    or manifest.get("genes") != len(corpus)
                    or manifest.get("output_sha256") != entry["vectors"]["sha256"]
                    or manifest.get("corpus_sha256") != entry["corpus"]["sha256"]
                    or manifest.get("text_fingerprint_sha256") != _fingerprint(corpus)
                ):
                    raise ValueError("Embedding provenance differs from exact corpus/model/dimension")
                vectors = load_npz(vector_path)
                labels = vectors.genes.tolist()
                if (
                    len(set(labels)) != len(labels)
                    or set(labels) != set(corpus)
                    or vectors.vectors.shape != (len(corpus), width)
                    or vectors.model != model
                    or np.any(np.linalg.norm(vectors.vectors.astype(np.float64), axis=1) == 0)
                ):
                    raise ValueError("Knowledge vectors have missing/duplicate genes, wrong shape or zero rows")
                table = dict(zip(labels, vectors.vectors, strict=True))
            self.tables[source] = table
            self.coverage[source] = {
                "required_genes": len(required),
                "available": len(required & set(table)),
                "missing": sorted(required - set(table)),
                "corpus_sha256": entry["corpus"]["sha256"],
            }
        self.model, self.width = model, width

    def for_targets(self, targets):
        targets = tuple(targets)
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("Knowledge lookup requires distinct perturbation target symbols")
        if any(gene not in self.tables["TextBase"] for gene in targets):
            raise ValueError("Required TextBase missing a perturbation target")
        return {
            source: np.stack([table[gene] for gene in targets]) if all(gene in table for gene in targets) else None
            for source, table in self.tables.items()
        }
