from pathlib import Path
from types import SimpleNamespace

from dinogenept.config import load_config
from dinogenept.tracking import TrackioRun

ROOT = Path(__file__).resolve().parents[1]


def test_disabled_trackio_has_no_import_or_side_effect():
    config = load_config(
        ROOT / "configs/experiments/dinogenept/ablations/00-supervised.yaml"
    )
    tracker = TrackioRun.start(config.payload, config.sha256)
    tracker.log_epoch({"epoch": 1, "total": 0.5})
    tracker.finish()
    assert tracker.receipt() == {
        "provider": "trackio",
        "enabled": False,
        "project": None,
        "run_name": None,
        "space_id": None,
        "sync_space_id": None,
        "epoch_rows": 0,
        "finished": False,
        "scientific_source_of_truth": False,
    }


def test_trackio_logs_loss_curves_and_final_metrics(monkeypatch):
    calls: dict[str, object] = {"logged": []}

    def init(**kwargs):
        calls["init"] = kwargs

    def log(values):
        calls["logged"].append(values)

    def finish():
        calls["finished"] = True

    fake = SimpleNamespace(init=init, log=log, finish=finish)
    monkeypatch.setattr("dinogenept.tracking.importlib.import_module", lambda _: fake)
    config = load_config(
        ROOT / "configs/experiments/dinogenept/ablations/00-supervised.yaml",
        overrides=[
            "runtime.tracking.enabled=true",
            "runtime.tracking.space_id=elan/dinogenept-trackio",
        ],
    )
    tracker = TrackioRun.start(config.payload, config.sha256)
    tracker.log_epoch(
        {
            "epoch": 1,
            "total": 0.75,
            "prediction": 0.5,
            "validation_prediction_mse": 0.25,
            "learning_rate": 0.0001,
        }
    )
    tracker.log_final(
        {"summary": {"pearson_delta": {"macro_mean": 0.42}}}, 12.5
    )
    tracker.finish()

    init_call = calls["init"]
    assert init_call["project"] == "dinogenept-ablation"
    assert init_call["space_id"] == "elan/dinogenept-trackio"
    assert init_call["private"] is True
    assert init_call["auto_log_gpu"] is True
    assert init_call["auto_log_cpu"] is True
    assert "seed-1" in init_call["name"]
    first = calls["logged"][0]
    assert first["train/loss"] == 0.75
    assert first["train/prediction"] == 0.5
    assert first["validation/prediction_mse"] == 0.25
    assert first["optimizer/learning_rate"] == 0.0001
    assert calls["logged"][1] == {
        "run/elapsed_seconds": 12.5,
        "evaluation/pearson_delta/macro_mean": 0.42,
    }
    assert calls["finished"] is True
    assert tracker.receipt()["epoch_rows"] == 1
    assert tracker.receipt()["sync_space_id"] == "elan68681/dinogenept-ablation"
    assert tracker.receipt()["scientific_source_of_truth"] is False
