import json
import threading

import numpy as np
import pytest

from genept_seed.embedding import (
    DEFAULT_BASE_URL,
    audit_embedding_checkpoint,
    generate_embeddings,
    load_gene_texts,
    merge_exact_checkpoint_hits,
    select_gene_texts,
    text_statistics,
    validate_plan_base_url,
)
from genept_seed.vectors import load_npz


def test_plan_endpoint_guard():
    assert validate_plan_base_url(DEFAULT_BASE_URL + "/") == DEFAULT_BASE_URL
    with pytest.raises(ValueError, match="Agent Plan"):
        validate_plan_base_url("https://ark.cn-beijing.volces.com/api/v3")


def test_load_nested_gene_texts(tmp_path):
    path = tmp_path / "texts.json"
    path.write_text(json.dumps({"tp53": {"summary": "tumor", "function": ["repair"]}}))
    assert load_gene_texts(path) == {"TP53": "tumor repair"}
    assert load_gene_texts(path, uppercase_genes=False) == {"tp53": "tumor repair"}


def test_text_selection_and_statistics():
    selected = select_gene_texts({"TP53": "tumor repair", "BRCA1": "dna"}, {"tp53", "missing"})
    assert selected == {"TP53": "tumor repair"}
    assert text_statistics(selected)["characters_max"] == 12


def test_generation_resumes_without_reembedding(tmp_path):
    calls = []

    def fake_embed(texts):
        calls.append(list(texts))
        return [np.asarray([len(text), 1], dtype=np.float32) for text in texts]

    arguments = dict(
        gene_texts={"B": "beta", "A": "alpha"},
        embed=fake_embed,
        model="mock",
        checkpoint_path=tmp_path / "checkpoint.sqlite3",
        output_path=tmp_path / "vectors.npz",
        batch_size=1,
        expected_dimension=2,
    )
    generate_embeddings(**arguments)
    generate_embeddings(**arguments)
    assert calls == [["alpha"], ["beta"]]
    assert load_npz(arguments["output_path"]).vectors.shape == (2, 2)
    manifest = json.loads((tmp_path / "vectors.npz.manifest.json").read_text())
    assert manifest["dimension"] == 2
    assert manifest["genes"] == 2
    assert manifest["schema_version"] == "genept-seed-embedding-v2"
    assert len(manifest["output_sha256"]) == 64
    assert audit_embedding_checkpoint(
        arguments["gene_texts"],
        checkpoint_path=arguments["checkpoint_path"],
        model="mock",
    ) == {"requested": 2, "exact_cached": 2, "pending": 0, "dimensions": [2]}


def test_checkpoint_audit_reports_all_pending_without_materializing_database(tmp_path):
    missing = tmp_path / "missing.sqlite3"
    expected = {
        "requested": 2,
        "exact_cached": 0,
        "pending": 2,
        "dimensions": [],
    }
    assert audit_embedding_checkpoint(
        {"A": "alpha", "B": "beta"}, checkpoint_path=missing, model="mock"
    ) == expected
    assert not missing.exists()

    empty = tmp_path / "empty.sqlite3"
    empty.touch()
    assert audit_embedding_checkpoint(
        {"A": "alpha", "B": "beta"}, checkpoint_path=empty, model="mock"
    ) == expected


def test_generation_rejects_unexpected_dimension(tmp_path):
    with pytest.raises(ValueError, match="expected 2048"):
        generate_embeddings(
            {"A": "alpha"},
            embed=lambda _: [np.zeros(3, dtype=np.float32)],
            model="mock",
            checkpoint_path=tmp_path / "checkpoint.sqlite3",
            output_path=tmp_path / "vectors.npz",
            expected_dimension=2048,
        )


def test_generation_supports_parallel_requests_with_main_thread_checkpointing(tmp_path):
    worker_threads = set()

    def fake_embed(texts):
        worker_threads.add(threading.get_ident())
        return [np.asarray([len(text), 1], dtype=np.float32) for text in texts]

    output = tmp_path / "parallel.npz"
    vectors = generate_embeddings(
        {f"G{index}": f"text-{index}" for index in range(8)},
        embed=fake_embed,
        model="mock",
        checkpoint_path=tmp_path / "parallel.sqlite3",
        output_path=output,
        batch_size=1,
        max_workers=2,
        expected_dimension=2,
    )
    assert len(vectors) == 8
    assert len(worker_threads) >= 1
    manifest = json.loads((tmp_path / "parallel.npz.manifest.json").read_text())
    assert manifest["max_workers"] == 2
    assert manifest["request_interval"] == 0.0


def test_generation_rejects_negative_request_interval(tmp_path):
    with pytest.raises(ValueError, match="request_interval"):
        generate_embeddings(
            {"A": "alpha"},
            embed=lambda _: [np.zeros(2, dtype=np.float32)],
            model="mock",
            checkpoint_path=tmp_path / "checkpoint.sqlite3",
            output_path=tmp_path / "vectors.npz",
            request_interval=-1,
        )


def test_generation_can_preserve_gene_case(tmp_path):
    output = tmp_path / "case.npz"
    generate_embeddings(
        {"C12orf45": "text"},
        embed=lambda _: [np.ones(2, dtype=np.float32)],
        model="mock",
        checkpoint_path=tmp_path / "case.sqlite3",
        output_path=output,
        expected_dimension=2,
        uppercase_genes=False,
    )
    assert load_npz(output).genes.tolist() == ["C12orf45"]
    manifest = json.loads((tmp_path / "case.npz.manifest.json").read_text())
    assert manifest["gene_case"] == "preserved"


def test_merge_checkpoints_accepts_only_exact_nonconflicting_hits(tmp_path):
    texts = {"A": "alpha", "B": "beta", "C": "gamma"}
    sources = []
    for index, selected in enumerate(({"A": "alpha", "B": "beta"}, {"B": "beta", "C": "gamma"})):
        checkpoint = tmp_path / f"source-{index}.sqlite3"
        generate_embeddings(
            selected,
            embed=lambda values: [
                np.asarray([len(value), 1], dtype=np.float32) for value in values
            ],
            model="mock",
            checkpoint_path=checkpoint,
            output_path=tmp_path / f"source-{index}.npz",
            expected_dimension=2,
        )
        sources.append(checkpoint)
    destination = tmp_path / "merged.sqlite3"
    receipt = merge_exact_checkpoint_hits(
        texts,
        source_paths=sources,
        destination_path=destination,
        model="mock",
    )
    assert receipt["exact_merged"] == 3
    assert receipt["pending"] == 0
    assert audit_embedding_checkpoint(
        texts, checkpoint_path=destination, model="mock"
    )["exact_cached"] == 3
