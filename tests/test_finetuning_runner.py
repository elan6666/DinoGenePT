import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scanpy")

from test_cellfm_dataset import frozen  # noqa: E402,F401 - shared fixture
from test_knowledge_bank import make_artifacts  # noqa: E402
from test_pretraining_runner import fixture_config  # noqa: E402

from dinogenept.cell import finetune  # noqa: E402
from dinogenept.cell.train import run_pretraining  # noqa: E402
from dinogenept.datasets.cellfm.dataset import CellFMDataset  # noqa: E402
from dinogenept.evaluation.cellfm import prepare_evaluation  # noqa: E402
from dinogenept.provenance import atomic_write_json, digest_file  # noqa: E402


@pytest.mark.parametrize("backend", ["chunk", "batched_chunk"])
def test_two_epoch_transfer_ten_epoch_lora_and_exact_resume(tmp_path, frozen, monkeypatch, backend):  # noqa: F811
    torch.set_num_threads(1)
    pre = fixture_config(tmp_path / "pretraining")
    pre["backbone"]["kda_implementation"] = backend
    run_pretraining(pre)
    data_config, _ = frozen
    vocab = tmp_path / "pretraining/genes.json"
    mapping_path = data_config["directory"] / "gene_mapping.json"
    mapping = json.loads(mapping_path.read_text())
    genes = json.loads(vocab.read_text())
    mapping.update(
        vocabulary_sha256=digest_file(vocab),
        vocabulary_size=len(genes),
        ensembl_ids=[genes[i - 1] for i in mapping["model_gene_ids"]],
    )
    atomic_write_json(mapping_path, mapping)
    data_config.update(vocabulary=vocab, mapping_sha256=digest_file(mapping_path))
    data = CellFMDataset(**data_config)
    evaluation = tmp_path / "evaluation.json"
    prepare_evaluation(data, evaluation, dataset_id="cellfm-fixture")
    best, completion = tmp_path / "pretraining/run/best.pt", tmp_path / "pretraining/run/completion.json"
    config = {
        "purpose": "unit_fixture",
        "output": str(tmp_path / "baseline"),
        "data": {key: str(value) for key, value in data_config.items()},
        "pretrained": {
            "checkpoint": str(best),
            "completion": str(completion),
            "checkpoint_sha256": digest_file(best),
            "completion_sha256": digest_file(completion),
        },
        "knowledge": {
            "sources": make_artifacts(tmp_path / "knowledge", genes=("A", "B", "C")),
            "model": "fixture-model",
            "width": 6,
        },
        "evaluation": {"path": str(evaluation), "sha256": digest_file(evaluation)},
        "perturbation": {"vector_width": 6, "decoder_width": 8, "lora_rank": 2, "lora_alpha": 4, "lora_dropout": 0.05},
        "training": {
            "device": "cpu",
            "epochs": 10,
            "bag_size": 2,
            "accumulation": 2,
            "gene_cap": 3,
            "evaluation_batch": 100,
            "checkpoint_steps": 1,
        },
    }
    seen_splits = []
    original_eval = finetune.evaluate_model

    def spy_eval(*args, **kwargs):
        seen_splits.append(args[4])
        return original_eval(*args, **kwargs)

    monkeypatch.setattr(finetune, "evaluate_model", spy_eval)
    progress = finetune.run_finetuning(config)
    assert progress["epoch"] == 10 and progress["post_cells_seen"] == 50
    assert progress["bags_seen"] == 30 and progress["completed_steps"] == 20
    assert seen_splits == ["val"] * 10 + ["test"]
    baseline = torch.load(tmp_path / "baseline/last.pt", weights_only=True)
    pretrained = torch.load(best, weights_only=True)["model"]
    for name, value in baseline["model"].items():
        if name.startswith("student.backbone.") and not name.endswith((".a", ".b")):
            original_name = name.replace(".base.", ".")
            torch.testing.assert_close(value, pretrained[original_name], rtol=0, atol=0)
    with pytest.raises(FileExistsError, match="explicit resume"):
        finetune.run_finetuning(config)
    config["output"] = str(tmp_path / "resumed")
    original_save = finetune._save

    def interrupt(*args, **kwargs):
        original_save(*args, **kwargs)
        raise InterruptedError("fixture interruption at successful checkpoint")

    monkeypatch.setattr(finetune, "_save", interrupt)
    with pytest.raises(InterruptedError):
        finetune.run_finetuning(config)
    assert not (tmp_path / "resumed/completion.json").exists()
    monkeypatch.setattr(finetune, "_save", original_save)
    finetune.run_finetuning(config, resume=tmp_path / "resumed/last.pt")
    resumed = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    assert resumed["progress"] == baseline["progress"]
    for key, value in baseline["model"].items():
        torch.testing.assert_close(value, resumed["model"][key], rtol=0, atol=0, msg=key)
    assert torch.equal(baseline["rank_rng"][0]["torch"], resumed["rank_rng"][0]["torch"])
    receipt = json.loads((tmp_path / "resumed/completion.json").read_text())
    assert receipt["purpose"] == "unit_fixture" and receipt["parameters"]["teacher_trainable"] == 0
    metrics = json.loads((tmp_path / "resumed/test.json").read_text())["metrics"]
    assert all(row["total_condition_count"] == 1 for row in metrics.values())


def test_formal_finetune_rejects_cpu_and_shortened_epochs():
    for training in ({"device": "cpu"}, {"epochs": 1}):
        with pytest.raises(ValueError, match="CUDA and ten"):
            finetune.run_finetuning({"purpose": "formal_finetuning", "training": training})
