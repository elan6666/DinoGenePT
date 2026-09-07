"""Small CPU-only fixtures test orchestration, never stand in for formal data."""

import json
import os
import socket
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dinogenept.cell import train  # noqa: E402
from dinogenept.cell.backbone import BackboneConfig  # noqa: E402
from dinogenept.cell.dataset import PretrainingDataset  # noqa: E402
from dinogenept.cell.pretraining import HeadConfig  # noqa: E402
from dinogenept.cell.sampling import CropConfig  # noqa: E402
from dinogenept.cell.transfer import load_pretrained_student  # noqa: E402
from dinogenept.provenance import digest_file  # noqa: E402


def test_published_protocol_no_validation_epoch(tmp_path):
    from test_genecompass_dataset import corpus

    config = fixture_config(tmp_path / "legacy")
    root = tmp_path / "published"
    root.mkdir()
    manifest, record = corpus(root)
    record["purpose"] = "unit_fixture"
    manifest.write_text(json.dumps(record))
    config.update(protocol=record["protocol"], data_manifest=str(manifest),
                  data_manifest_sha256=digest_file(manifest), output=str(tmp_path / "run"))
    config["backbone"]["genes"] = 2
    config["training"]["epochs"] = 1
    config["training"]["microbatch"] = 2
    train.run_pretraining(config)
    receipt = json.loads((tmp_path / "run/completion.json").read_text())
    assert receipt["cells_seen"] == receipt["training_cells"] == 2
    assert receipt["epoch"] == receipt["epochs"] == 1
    assert receipt["validation_cells"] == 0
    assert receipt["best_validation"] is None
    assert receipt["downstream_overlap"] == "unknown"
    assert (tmp_path / "run/last.pt").exists()
    assert not (tmp_path / "run/best.pt").exists()


def fixture_config(root):
    root.mkdir()
    vocab = root / "genes.json"
    vocab.write_text(json.dumps([f"fixture-{i}" for i in range(12)]))
    shards = []
    for split, n, offset in [("train", 10, 0), ("validation", 4, 10)]:
        path = root / f"{split}.npz"
        counts = np.arange(1, n * 12 + 1, dtype=np.float32).reshape(n, 12)
        np.savez_compressed(
            path,
            data=counts.ravel(),
            indices=np.tile(np.arange(12), n),
            indptr=np.arange(n + 1) * 12,
            row_ids=np.arange(n) + offset,
            library_sum=counts.sum(1),
        )
        shards.append({"split": split, "cells": n, "path": path.name, "sha256": digest_file(path)})
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "dinogenept.pretraining.csr.v1",
                "status": "training_ready",
                "purpose": "unit_fixture",
                "leakage_audit": {"status": "passed", "scope": "synthetic_fixture_only"},
                "expression": "raw_counts",
                "data_id": "synthetic-test",
                "shards": shards,
                "vocabulary": {"genes": 12, "path": vocab.name, "sha256": digest_file(vocab)},
            }
        )
    )
    return {
        "purpose": "unit_fixture",
        "data_manifest": str(manifest),
        "data_manifest_sha256": digest_file(manifest),
        "output": str(root / "run"),
        "backbone": asdict(
            BackboneConfig(
                genes=12,
                width=16,
                depth=4,
                kda_heads=1,
                kda_head_dim=8,
                mla_heads=2,
                mla_head_dim=4,
                mla_shared_dim=2,
                query_rank=8,
                kv_rank=8,
                gradient_checkpointing=True,
            )
        ),
        "heads": asdict(HeadConfig(hidden=16, bottleneck=8, cell_prototypes=8, gene_prototypes=8)),
        "crops": asdict(CropConfig(cap=12)),
        "training": {
            "device": "cpu",
            "world_size": 1,
            "epochs": 2,
            "accumulation": 2,
            "microbatch": 3,
            "seed": 42,
            "learning_rate": 1e-4,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "checkpoint_steps": 1,
            "workers": 0,
        },
    }


def test_one_step_smoke_is_isolated_from_formal_training(tmp_path):
    torch.set_num_threads(1)
    config = fixture_config(tmp_path / "data")
    result = train.run_pretraining(config, smoke_one_step=True)
    output = tmp_path / "data/run"
    assert result["completed_steps"] == 1 and result["epoch"] == 0
    assert not (output / "completion.json").exists()
    assert not (output / "last.pt").exists()
    receipt = json.loads((output / "smoke.json").read_text())
    assert receipt["status"] == "smoke_completed"
    assert receipt["formal_transfer_eligible"] is False
    assert receipt["checkpoint_sha256"] == digest_file(output / "smoke.pt")
    payload = torch.load(output / "smoke.pt", weights_only=True)
    assert payload["config"]["execution_mode"] == "smoke_one_step"
    assert payload["progress"] == result
    with pytest.raises(ValueError, match="Smoke checkpoints"):
        load_pretrained_student(
            output / "smoke.pt",
            output / "smoke.json",
            tmp_path / "data/genes.json",
            checkpoint_sha256=digest_file(output / "smoke.pt"),
            completion_sha256=digest_file(output / "smoke.json"),
            purpose="unit_fixture",
        )
    with pytest.raises(ValueError, match="Smoke resolved configs"):
        train.run_pretraining(payload["config"])
    with pytest.raises(ValueError, match="fresh run"):
        train.run_pretraining(config, resume=output / "smoke.pt", smoke_one_step=True)


@pytest.mark.parametrize("backend,workers", [("chunk", 0), ("batched_chunk", 0), ("batched_chunk", 1)])
def test_two_full_epochs_and_exact_resume(tmp_path, monkeypatch, backend, workers):
    torch.set_num_threads(1)
    config = fixture_config(tmp_path / "data")
    config["backbone"]["kda_implementation"] = backend
    config["training"]["workers"] = workers
    train.run_pretraining(config)
    baseline = torch.load(tmp_path / "data/run/last.pt", weights_only=True)
    receipt = json.loads((tmp_path / "data/run/completion.json").read_text())
    assert receipt["purpose"] == "unit_fixture"
    assert receipt["cells_seen"] == 20 and receipt["epoch"] == 2 and receipt["completed_steps"] == 4
    config["output"] = str(tmp_path / "interrupted")
    original = train._checkpoint

    def interrupt_after_durable_checkpoint(*args, **kwargs):
        original(*args, **kwargs)
        raise InterruptedError("test interruption after successful optimizer checkpoint")

    monkeypatch.setattr(train, "_checkpoint", interrupt_after_durable_checkpoint)
    with pytest.raises(InterruptedError):
        train.run_pretraining(config)
    assert not (tmp_path / "interrupted/completion.json").exists()
    monkeypatch.setattr(train, "_checkpoint", original)
    train.run_pretraining(config, resume=tmp_path / "interrupted/last.pt")
    resumed = torch.load(tmp_path / "interrupted/last.pt", weights_only=True)
    assert resumed["progress"] == baseline["progress"]
    for key, value in baseline["model"].items():
        torch.testing.assert_close(value, resumed["model"][key], rtol=0, atol=0, msg=key)
    assert torch.equal(baseline["rank_rng"][0]["torch"], resumed["rank_rng"][0]["torch"])
    records = [json.loads(x) for x in (tmp_path / "interrupted/metrics.jsonl").read_text().splitlines()]
    steps = [x for x in records if x["event"] == "optimizer_step"]
    assert [x["step"] for x in steps] == [1, 2, 3, 4]
    for record in steps:
        assert sum(record["weighted_losses"].values()) == pytest.approx(record["total"], rel=1e-6)

    checkpoint, completion, vocabulary = (
        tmp_path / "data/run/best.pt",
        tmp_path / "data/run/completion.json",
        tmp_path / "data/genes.json",
    )
    kwargs = {"checkpoint_sha256": digest_file(checkpoint), "completion_sha256": digest_file(completion)}
    pretrained, identity = load_pretrained_student(checkpoint, completion, vocabulary, purpose="unit_fixture", **kwargs)
    selected = torch.load(checkpoint, weights_only=True)
    assert identity["source_branch"] == "student" and identity["completed_pretraining_epochs"] == 2
    for key, value in pretrained.state_dict().items():
        torch.testing.assert_close(value, selected["model"][f"student.{key}"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="formal campaign"):
        load_pretrained_student(checkpoint, completion, vocabulary, **kwargs)
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({**receipt, "cells_seen": 19}))
    with pytest.raises(ValueError, match="incomplete"):
        load_pretrained_student(
            checkpoint,
            incomplete,
            vocabulary,
            purpose="unit_fixture",
            checkpoint_sha256=digest_file(checkpoint),
            completion_sha256=digest_file(incomplete),
        )


def test_mismatched_manifest_and_formal_cpu_are_rejected(tmp_path):
    config = fixture_config(tmp_path / "data")
    config["data_manifest_sha256"] = "wrong"
    with pytest.raises(ValueError, match="manifest hash"):
        train.run_pretraining(config)
    config["purpose"] = "formal_pretraining"
    with pytest.raises(ValueError, match="audited server GPUs"):
        train.run_pretraining(config)
    config["training"]["device"] = "cuda"
    with pytest.raises(ValueError, match="30 complete epochs"):
        train.run_pretraining(config)


def _distributed_worker(rank, config, port):
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port)
    )
    torch.set_num_threads(1)
    train.run_pretraining(config)


@pytest.mark.parametrize("backend", ["chunk", "batched_chunk"])
def test_two_rank_cpu_accumulation_covers_every_cell(tmp_path, backend):
    config = fixture_config(tmp_path / "data")
    config["backbone"]["kda_implementation"] = backend
    config["training"]["world_size"] = 2
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    torch.multiprocessing.spawn(_distributed_worker, args=(config, port), nprocs=2, join=True)
    state = torch.load(tmp_path / "data/run/last.pt", weights_only=True)
    assert len(state["rank_rng"]) == 2
    assert state["progress"]["cells_seen"] == 20
    assert state["progress"]["completed_steps"] == 2


def test_corpus_verifier_rejects_cross_split_duplicates(tmp_path):
    fixture_config(tmp_path / "data")
    path = tmp_path / "data/validation.npz"
    with np.load(path, allow_pickle=False) as data:
        values = dict(data)
    values["row_ids"][0] = 0
    np.savez_compressed(path, **values)
    manifest = tmp_path / "data/manifest.json"
    content = json.loads(manifest.read_text())
    content["shards"][1]["sha256"] = digest_file(path)
    manifest.write_text(json.dumps(content))
    dataset = PretrainingDataset(manifest, "train", CropConfig())
    with pytest.raises(ValueError, match="across corpus shards/splits"):
        dataset.verify()


def test_corpus_verifier_rejects_fake_vocabulary_and_normalized_counts(tmp_path):
    fixture_config(tmp_path / "data")
    manifest = tmp_path / "data/manifest.json"
    content = json.loads(manifest.read_text())
    path = tmp_path / "data/train.npz"
    with np.load(path, allow_pickle=False) as data:
        values = dict(data)
    values["data"][0] = 0.5
    np.savez_compressed(path, **values)
    content["shards"][0]["sha256"] = digest_file(path)
    manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="fractional/normalized"):
        PretrainingDataset(manifest, "train", CropConfig()).verify()
    vocabulary = tmp_path / "data/genes.json"
    vocabulary.write_text(json.dumps(["duplicate"] * 12))
    content["vocabulary"]["sha256"] = digest_file(vocabulary)
    manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="unique ordered gene"):
        PretrainingDataset(manifest, "train", CropConfig()).verify()
