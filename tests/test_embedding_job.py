"""No-network fixture for the resumable multi-source API orchestration."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from test_knowledge_bank import make_artifacts

from dinogenept.embedding import EmbeddingCheckpoint
from dinogenept.provenance import atomic_write_json, digest_file


def test_source_job_reuses_completed_artifacts_and_never_injects_fake_missing_locals(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "embedding_job_fixture", Path(__file__).parents[1] / "scripts/embed_cellfm_knowledge.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    corpora = tmp_path / "corpora"
    artifacts = make_artifacts(corpora)
    (corpora / "genes.txt").write_text("A\nB\n")
    for source in ("GO", "Protein", "Pathway", "HPA"):
        shutil.copyfile(artifacts[source]["source_manifest"]["path"], corpora / f"{source}.source.json")
    atomic_write_json(
        corpora / "receipt.json", {"files_sha256": {p.name: digest_file(p) for p in corpora.iterdir() if p.is_file()}}
    )
    (tmp_path / "checkpoints").mkdir()
    for name in module.SEED_CACHES.values():
        EmbeddingCheckpoint(tmp_path / "checkpoints" / name).close()
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs == {"max_retries": 0}

        def embed(self, texts):
            calls.append(len(texts))
            return [np.arange(1, 2049, dtype=np.float32)] * len(texts)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "ArkEmbeddingClient", FakeClient)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "embed_cellfm_knowledge.py",
            "--corpora",
            str(corpora),
            "--output",
            str(tmp_path / "vectors"),
            "--checkpoints",
            str(tmp_path / "new_checkpoints"),
        ],
    )
    module.main()
    assert calls == [2, 1, 2]
    bundle = json.loads((tmp_path / "vectors/bundle.json").read_text())
    assert bundle["status"] == "verified" and bundle["coverage"]["HPA"]["available"] == 0
    assert not (tmp_path / "vectors/HPA.npz").exists()
    module.main()
    assert calls == [2, 1, 2]  # Completed sources are not regenerated.
