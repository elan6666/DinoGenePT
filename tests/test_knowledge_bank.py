import json

import numpy as np
import pytest

from dinogenept.datasets.knowledge import KNOWLEDGE_SOURCES, KnowledgeBank
from dinogenept.embedding import generate_embeddings
from dinogenept.provenance import atomic_write_json, digest_file
from dinogenept.source_corpus import build_source_only_corpus


def pinned(path):
    return {"path": str(path), "sha256": digest_file(path)}


@pytest.fixture
def artifacts(tmp_path):
    return make_artifacts(tmp_path)


def make_artifacts(tmp_path, genes=("A", "B")):
    tmp_path.mkdir(exist_ok=True)
    base = {gene: f"NCBI and UniProt description {gene}" for gene in genes}
    base_path = tmp_path / "base.json"
    atomic_write_json(base_path, base)
    entries = {}
    for source in KNOWLEDGE_SOURCES:
        available = set(genes) if source in {"TextBase", "Protein"} else {"A"} if source == "GO" else set()
        corpus_path = tmp_path / f"{source}.json"
        source_manifest = tmp_path / f"{source}.proof.json"
        if source == "TextBase":
            atomic_write_json(corpus_path, base)
        else:
            enriched = tmp_path / f"{source}.enriched.json"
            atomic_write_json(
                enriched,
                {g: text + (f"\n{source} knowledge {g}" if g in available else "") for g, text in base.items()},
            )
            build_source_only_corpus(
                base_path=base_path,
                enriched_path=enriched,
                output_path=corpus_path,
                manifest_path=source_manifest,
                source=source,
            )
        entries[source] = {"corpus": pinned(corpus_path)}
        if source != "TextBase":
            entries[source]["source_manifest"] = pinned(source_manifest)
        if available:
            vector_path = tmp_path / f"{source}.npz"
            generate_embeddings(
                json.loads(corpus_path.read_text()),
                embed=lambda texts: [np.arange(1, 7, dtype=np.float32)] * len(texts),
                model="fixture-model",
                checkpoint_path=tmp_path / f"{source}.sqlite",
                output_path=vector_path,
                expected_dimension=6,
                corpus_path=corpus_path,
                uppercase_genes=False,
            )
            entries[source].update(
                vectors=pinned(vector_path),
                embedding_manifest=pinned(vector_path.with_suffix(".npz.manifest.json")),
            )
    return entries


def test_sparse_sources_and_combinations_keep_exact_order(artifacts):
    bank = KnowledgeBank(artifacts, ["A", "B"], model="fixture-model", width=6)
    assert bank.coverage["GO"]["missing"] == ["B"]
    assert bank.for_targets(["A"])["GO"].shape == (1, 6)
    pair = bank.for_targets(["B", "A"])
    assert pair["GO"] is None and pair["HPA"] is None and pair["Pathway"] is None
    assert pair["Protein"].shape == (2, 6)
    with pytest.raises(ValueError, match="distinct"):
        bank.for_targets(["A", "A"])
    with pytest.raises(ValueError, match="TextBase missing"):
        bank.for_targets(["a"])


def test_base_must_cover_full_requested_axis_and_optional_entries_are_explicit(artifacts):
    with pytest.raises(ValueError, match="TextBase lacks"):
        KnowledgeBank(artifacts, ["A", "B", "C"], model="fixture-model", width=6)
    artifacts.pop("HPA")
    with pytest.raises(ValueError, match="Explicit corpus"):
        KnowledgeBank(artifacts, ["A", "B"], model="fixture-model", width=6)


def test_cumulative_or_mislabeled_source_manifest_rejected(artifacts):
    from pathlib import Path

    proof = Path(artifacts["GO"]["source_manifest"]["path"])
    data = json.loads(proof.read_text())
    data["schema_version"] = "genept-seed-progressive-knowledge-corpus-v1"
    atomic_write_json(proof, data)
    artifacts["GO"]["source_manifest"] = pinned(proof)
    with pytest.raises(ValueError, match="source-only"):
        KnowledgeBank(artifacts, ["A", "B"], model="fixture-model", width=6)


def test_wrong_text_model_dimension_and_checksum_rejected(artifacts):
    from pathlib import Path

    for kwargs in ({"model": "other", "width": 6}, {"model": "fixture-model", "width": 7}):
        with pytest.raises(ValueError, match="Embedding provenance"):
            KnowledgeBank(artifacts, ["A", "B"], **kwargs)
    corpus = Path(artifacts["GO"]["corpus"]["path"])
    corpus.write_text(corpus.read_text() + " ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        KnowledgeBank(artifacts, ["A", "B"], model="fixture-model", width=6)


def test_source_only_builder_does_not_confuse_absent_gene_with_missing_annotation(tmp_path):
    base, enriched, genes = tmp_path / "base.json", tmp_path / "enriched.json", tmp_path / "genes.txt"
    atomic_write_json(base, {"A": "base"})
    atomic_write_json(enriched, {"A": "base"})
    genes.write_text("A\nB\n")
    with pytest.raises(ValueError, match="absent from source corpus universe"):
        build_source_only_corpus(
            base_path=base,
            enriched_path=enriched,
            genes_path=genes,
            source="GO",
            output_path=tmp_path / "out.json",
            manifest_path=tmp_path / "proof.json",
        )
