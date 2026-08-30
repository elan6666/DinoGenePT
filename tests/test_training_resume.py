import random
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dinogenept.config import load_config  # noqa: E402
from dinogenept.experiments.runner import run_experiment, run_root  # noqa: E402
from dinogenept.registry import MODELS  # noqa: E402
from dinogenept.tracking import TrackioRun  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_dinogenept_resumes_from_atomic_epoch_state(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.npz"
    conditions = ["ctrl"] * 4
    for gene in ("A", "B", "C", "D", "E", "F"):
        conditions.extend([f"{gene}+ctrl"] * 2)
    expression = np.random.default_rng(7).normal(
        size=(len(conditions), 8)
    ).astype(np.float32)
    np.savez(
        dataset,
        expression=expression,
        genes=np.asarray([f"G{index}" for index in range(8)]),
        conditions=np.asarray(conditions),
        row_ids=np.asarray([f"row-{index}" for index in range(len(conditions))]),
    )
    config = load_config(
        ROOT / "configs/experiments/dinogenept/ablations/00-supervised.yaml",
        overrides=[
            f"dataset.path={dataset}",
            f"runtime.output_root={tmp_path / 'outputs'}",
            "runtime.enforce_server=false",
            "runtime.device=cpu",
            "training.epochs=2",
            "training.max_batches=1",
            "evaluation.run=false",
        ],
    )

    def interrupt_after_saved_epoch(self, row):
        del self, row
        raise RuntimeError("simulated interruption after epoch checkpoint")

    monkeypatch.setattr(TrackioRun, "log_epoch", interrupt_after_saved_epoch)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_experiment(config)
    state_path = run_root(config) / "training-state.pt"
    assert state_path.is_file()

    monkeypatch.undo()
    receipt = run_experiment(config, retry_failed=True)
    assert receipt["status"] == "complete"
    assert receipt["training"]["resumed_from_epoch"] == 1
    assert len(receipt["training"]["history"]) == 2
    assert not state_path.exists()
    assert len(list((run_root(config) / "attempts").glob("attempt-*/failure.json"))) == 1


def test_tracker_rng_consumption_cannot_change_seeded_checkpoint(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.npz"
    conditions = (
        ["ctrl"] * 4
        + ["A+ctrl"] * 2
        + ["B+ctrl"] * 2
        + ["C+ctrl"] * 2
    )
    np.savez(
        dataset,
        expression=np.random.default_rng(3).normal(size=(10, 8)).astype(np.float32),
        genes=np.asarray([f"G{index}" for index in range(8)]),
        conditions=np.asarray(conditions),
        row_ids=np.asarray([f"row-{index}" for index in range(10)]),
    )
    source = ROOT / "configs/experiments/dinogenept/ablations/00-supervised.yaml"

    def configured(output_root, tracking):
        return load_config(
            source,
            overrides=[
                f"dataset.path={dataset}",
                f"runtime.output_root={output_root}",
                "runtime.enforce_server=false",
                "runtime.device=cpu",
                "runtime.save_checkpoint=true",
                f"runtime.tracking.enabled={str(tracking).lower()}",
                "training.epochs=1",
                "training.max_batches=1",
                "evaluation.run=false",
            ],
        )

    baseline = configured(tmp_path / "baseline", False)
    run_experiment(baseline)

    def consuming_tracker(_config, _sha256):
        random.random()
        np.random.random()
        torch.rand(4)
        return TrackioRun(enabled=False)

    monkeypatch.setattr(TrackioRun, "start", consuming_tracker)
    tracked = configured(tmp_path / "tracked", True)
    run_experiment(tracked)

    first = torch.load(
        run_root(baseline) / "checkpoint.pt", map_location="cpu", weights_only=True
    )["model"]
    second = torch.load(
        run_root(tracked) / "checkpoint.pt", map_location="cpu", weights_only=True
    )["model"]
    assert first.keys() == second.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)


def test_formal_pretrained_checkpoint_requires_exact_parameter_schema(tmp_path):
    config = load_config(
        ROOT / "configs/experiments/dinogenept/ablations/00-supervised.yaml"
    )
    dimensions = {"base": 64, "go": 64, "protein": 64, "pathway": 64, "hpa": 64}
    model = MODELS.resolve("dinogenept").build_model(
        config.payload, n_genes=8, prior_dimensions=dimensions
    )
    checkpoint = tmp_path / "exact.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    receipt = model.load_pretrained(
        checkpoint, minimum_match_fraction=1.0, strict=True
    )
    assert receipt["strict"] is True
    assert (
        receipt["checkpoint_parameter_schema_sha256"]
        == receipt["current_parameter_schema_sha256"]
    )

    incomplete = dict(model.state_dict())
    incomplete.pop(next(iter(incomplete)))
    bad = tmp_path / "incomplete.pt"
    torch.save({"model": incomplete}, bad)
    with pytest.raises(ValueError, match="strict pretrained checkpoint schema mismatch"):
        model.load_pretrained(bad, minimum_match_fraction=1.0, strict=True)
