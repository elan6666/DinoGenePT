import json

import numpy as np
import pytest

from dinogenept.priors import PriorStore
from genept_seed.embedding import generate_embeddings
from genept_seed.provenance import digest_file


def _config(optional_coverage: float = 0.5):
    return {
        "priors": {
            "base": {"fixture": "deterministic", "dimension": 8, "coverage": 1.0},
            "optional": {
                "go": {
                    "fixture": "deterministic",
                    "dimension": 8,
                    "coverage": optional_coverage,
                    "source_only": True,
                }
            },
        }
    }


def test_optional_missingness_is_omission_not_a_fake_token():
    genes = tuple(f"G{index}" for index in range(40))
    store = PriorStore.from_config(_config(), genes)
    missing = next(gene for gene in genes if not store.source_available("go", (gene,)))
    vectors, token_mask, condition_mask = store.batch("go", ((missing,),))
    assert condition_mask.tolist() == [False]
    assert token_mask.tolist() == [[False]]
    assert np.count_nonzero(vectors) == 0


def test_combination_requires_every_target_and_controls_keep_availability():
    genes = tuple(f"G{index}" for index in range(40))
    store = PriorStore.from_config(_config(optional_coverage=1.0), genes)
    targets = ((genes[0], genes[1]),)
    original = store.batch("go", targets)
    shuffled = store.batch("go", targets, control="shuffled", seed=9)
    availability = store.batch("go", targets, control="availability_only")
    assert original[1].all() and shuffled[1].all() and availability[1].all()
    assert not np.allclose(original[0], shuffled[0])
    assert np.allclose(availability[0][0, 0], availability[0][0, 1])


def test_formal_prior_is_bound_to_profile_corpus_universe_and_artifact(tmp_path):
    corpus = tmp_path / "base.json"
    corpus.write_text('{"A":"alpha","B":"beta"}\n', encoding="utf-8")
    universe = tmp_path / "genes.txt"
    universe.write_text("A\nB\n", encoding="utf-8")
    source_manifest = tmp_path / "base.manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": "genept-seed-axis-corpus-v1",
                "output_sha256": digest_file(corpus),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "base.npz"
    generate_embeddings(
        {"A": "alpha", "B": "beta"},
        embed=lambda values: [
            np.asarray([len(value), 1], dtype=np.float32) for value in values
        ],
        model="mock",
        checkpoint_path=tmp_path / "checkpoint.sqlite3",
        output_path=output,
        expected_dimension=2,
        profile="ncbi-uniprot-base-v1",
        corpus_path=corpus,
        universe_path=universe,
    )
    embedding_manifest = json.loads(
        output.with_suffix(".npz.manifest.json").read_text(encoding="utf-8")
    )
    config = {
        "priors": {
            "base": {
                "path": str(output),
                "manifest": str(output.with_suffix(".npz.manifest.json")),
                "profile": "ncbi-uniprot-base-v1",
                "source_manifest": str(source_manifest),
                "expected_source_manifest_sha256": digest_file(source_manifest),
                "expected_corpus_sha256": digest_file(corpus),
                "expected_gene_universe_sha256": digest_file(universe),
                "expected_text_fingerprint_sha256": embedding_manifest[
                    "text_fingerprint_sha256"
                ],
                "expected_embedding_model": "mock",
                "expected_gene_case": "uppercase",
            },
            "optional": {},
        }
    }
    store = PriorStore.from_config(config, ("A", "B"))
    assert store.input_manifest["base"]["profile"] == "ncbi-uniprot-base-v1"
    assert store.input_manifest["base"]["source_manifest_sha256"] == digest_file(
        source_manifest
    )
    with output.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="manifest identity differs"):
        PriorStore.from_config(config, ("A", "B"))
