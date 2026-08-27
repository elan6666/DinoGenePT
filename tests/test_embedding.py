import json

import numpy as np
import pytest

from genept_seed.embedding import (
    DEFAULT_BASE_URL,
    generate_embeddings,
    load_gene_texts,
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
