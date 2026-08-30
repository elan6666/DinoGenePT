from pathlib import Path

import yaml

from dinogenept.config import load_config
from dinogenept.experiments import benchmark, pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_priority_ablation_prior_contracts_are_explicit():
    pipelines = ROOT / "configs/pipelines"
    for filename in (
        "ablation-00-supervised.yaml",
        "ablation-01-dino-base.yaml",
        "ablation-02-dino-ibot.yaml",
    ):
        source = pipelines / filename
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        row = payload["rows"][0]
        for phase in ("pretrain", "finetune"):
            config = load_config(
                source.parent / row[phase],
                overrides=payload[f"{phase}_overrides"],
            )
            assert not any(
                entry["enabled"] for entry in config.payload["priors"]["optional"].values()
            )

    source = pipelines / "ablation-07-dynamic-locals.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    row = payload["rows"][0]
    config = load_config(
        source.parent / row["pretrain"],
        overrides=payload["pretrain_overrides"],
    )
    assert config.payload["ablation"]["local_sources"] == [
        "go",
        "protein",
        "pathway",
        "hpa",
    ]
    assert all(
        config.payload["priors"]["optional"][source_name]["enabled"]
        for source_name in config.payload["ablation"]["local_sources"]
    )


def test_pipeline_only_uses_distinct_partial_receipt_and_cannot_claim_complete(
    tmp_path, monkeypatch
):
    def fake_run_or_reuse(config, **_):
        phase = config.payload["training"]["phase"]
        seed = int(config.payload["training"]["seed"])
        return (
            {
                "status": "complete",
                "run_root": str(tmp_path / phase / f"seed-{seed}"),
                "training": {
                    "epochs": int(config.payload["training"]["epochs"]),
                    "best_epoch": 1,
                },
                "metrics": {
                    "summary": {"pearson_delta": {"macro_mean": 0.5}}
                },
                "dataset_fingerprint": "f" * 64,
                "input_manifest": {"dataset": {"split_sha256": "s" * 64}},
                "fairness_contract": {"sha256": "c" * 64},
                "parameters": {"total": 1, "trainable": 1},
            },
            False,
        )

    monkeypatch.setattr(pipeline, "_run_or_reuse", fake_run_or_reuse)
    result = pipeline.run_pipeline(
        ROOT / "configs/pipelines/gears5-scgpt-aligned.yaml",
        only=("norman",),
        only_seeds=(1,),
        output_root=tmp_path,
    )
    assert result["status"] == "partial_complete"
    assert result["row_count"] == 1
    assert result["grid"]["expected_stage_runs"] == 40
    assert result["grid"]["selected_stage_runs"] == 2
    assert result["grid"]["completed_stage_runs"] == 2
    assert result["grid"]["selected_seeds"] == [1]
    assert not (tmp_path / "gears5-scgpt-aligned.json").exists()
    partials = list(tmp_path.glob("gears5-scgpt-aligned.partial-*.json"))
    assert len(partials) == 1


def test_scouter_benchmark_only_cannot_overwrite_full_receipt(tmp_path, monkeypatch):
    def fake_run_or_reuse(config, **_):
        seed = int(config.payload["training"]["seed"])
        return (
            {
                "status": "complete",
                "run_root": str(tmp_path / "scouter" / f"seed-{seed}"),
                "metrics": {
                    "summary": {"systema_pearson": {"macro_mean": 0.25}}
                },
                "dataset_fingerprint": "f" * 64,
                "input_manifest": {"dataset": {"split_sha256": "s" * 64}},
                "fairness_contract": {"sha256": "c" * 64},
                "parameters": {"total": 1, "trainable": 1},
            },
            False,
        )

    monkeypatch.setattr(benchmark, "_run_or_reuse", fake_run_or_reuse)
    result = benchmark.run_benchmark(
        ROOT / "configs/benchmarks/scouter-gradpert5-base.yaml",
        only=("norman",),
        only_seeds=(1,),
        output_root=tmp_path,
    )
    assert result["status"] == "partial_complete"
    assert result["row_count"] == 1
    assert result["grid"]["expected_runs"] == 20
    assert result["grid"]["selected_runs"] == 1
    assert result["grid"]["completed_runs"] == 1
    assert result["grid"]["selected_seeds"] == [1]
    assert not (tmp_path / "scouter-gradpert5-base.json").exists()
    assert len(list(tmp_path.glob("scouter-gradpert5-base.partial-*.json"))) == 1
