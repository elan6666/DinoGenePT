"""Formal-run gate rejects incomplete, stale or failed smoke evidence."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("launcher", ROOT / "scripts/launch_genecompass.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.fixture
def smoke(tmp_path):
    config = dict(backbone={}, heads={}, crops={}, training={}, data_manifest_sha256="abc", protocol="test")
    def save(name, obj):
        (tmp_path / name).write_text(json.dumps(obj))
    (tmp_path / "smoke.pt").write_bytes(b"synthetic gate fixture; not a real model")
    save("resolved_config.json", {**config, "execution_mode": "smoke_one_step"})
    save("exit.json", {"returncode": 0})
    save("smoke.json", dict(status="smoke_completed", completed_steps=1, epoch=0,
                            world_size=2, data_manifest_sha256="abc",
                            checkpoint_sha256=launcher.digest_file(tmp_path / "smoke.pt")))
    save("launch.json", {"source_sha256": {str(p): launcher.digest_file(p)
                                           for p in (ROOT / "src/dinogenept/cell").glob("*.py")}})
    save("metrics.jsonl", dict(event="optimizer_step", total=1, expression=1, cell_expression=1,
                              dino=1, ibot=1, koleo=1, gradient_norm=1))
    return config, tmp_path, save


def test_accept_matching_successful_smoke(smoke):
    config, root, _ = smoke
    launcher.validate_smoke(config, root)


def test_population_cannot_reuse_other_smoke(smoke):
    config, root, save = smoke
    config["published_cells"] = 50000
    with pytest.raises(ValueError, match="population"):
        launcher.validate_smoke(config, root)
    previous = json.loads((root / "resolved_config.json").read_text())
    save("resolved_config.json", {**previous, "published_cells": 50000})
    launcher.validate_smoke(config, root)


def test_shared_resource_policy_must_match(smoke):
    config, root, save = smoke
    with pytest.raises(ValueError, match="resource policy"):
        launcher.validate_smoke(config, root, "222:123")
    launch = json.loads((root / "launch.json").read_text())
    save("launch.json", {**launch, "shared_process": "222:123"})
    launcher.validate_smoke(config, root, "222:123")


@pytest.mark.parametrize("filename,patch", [
    ("exit.json", {"returncode": 1}),
    ("resolved_config.json", {"execution_mode": "formal"}),
    ("resolved_config.json", {"backbone": {"changed": True}}),
    ("smoke.json", {"completed_steps": 0}),
    ("smoke.json", {"world_size": 1}),
    ("smoke.json", {"checkpoint_sha256": "wrong"}),
    ("launch.json", {"source_sha256": {}}),
    ("metrics.jsonl", {"gradient_norm": float("nan")}),
    ("metrics.jsonl", {"event": "not_optimizer_step"}),
])
def test_reject_invalid_smoke(smoke, filename, patch):
    config, root, save = smoke
    save(filename, {**json.loads((root / filename).read_text()), **patch})
    with pytest.raises(ValueError):
        launcher.validate_smoke(config, root)


def test_reject_missing_exit(smoke):
    config, root, _ = smoke
    (root / "exit.json").unlink()
    with pytest.raises(FileNotFoundError):
        launcher.validate_smoke(config, root)


def test_reject_stale_source(smoke):
    config, root, save = smoke
    launch = json.loads((root / "launch.json").read_text())
    launch["source_sha256"][next(iter(launch["source_sha256"]))] = "changed"
    save("launch.json", launch)
    with pytest.raises(ValueError, match="Source changed"):
        launcher.validate_smoke(config, root)
