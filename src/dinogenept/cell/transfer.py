"""Audited pretraining Student transfer, without upstream model dependencies."""

import json
from pathlib import Path

import torch

from dinogenept.provenance import digest_file

from .backbone import BackboneConfig
from .checkpoint import config_hash
from .pretraining import HeadConfig, PretrainingNetwork


def load_pretrained_student(
    checkpoint, completion, vocabulary, *, checkpoint_sha256, completion_sha256, purpose="formal_pretraining"
):
    """A best checkpoint may precede the last epoch, but the run must be complete.

    Pin the chosen checkpoint and completion receipt separately. A smoke/fixture
    checkpoint cannot initialize a formal campaign. The vocabulary is checked by
    identity hash, not just by its size. Optimizer/RNG/teacher state is NOT reused
    for a new downstream task; downstream resume is a separate contract.
    """
    checkpoint, completion, vocabulary = Path(checkpoint), Path(completion), Path(vocabulary)
    if purpose not in {"formal_pretraining", "unit_fixture"}:
        raise ValueError("Unknown pretrained run purpose")
    if digest_file(checkpoint) != checkpoint_sha256 or digest_file(completion) != completion_sha256:
        raise ValueError("Pinned pretraining checkpoint/completion changed")
    receipt = json.loads(completion.read_text())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = payload.get("config", {})
    if payload.get("schema") != "dinogenept.training.v1" or payload.get("config_sha256") != config_hash(config):
        raise ValueError("Invalid pretraining checkpoint schema/config")
    epochs = config["training"]["epochs"]
    if (
        receipt.get("status") != "completed"
        or receipt.get("purpose") != purpose
        or config.get("purpose", "formal_pretraining") != purpose
        or epochs < 1
        or (purpose == "formal_pretraining" and epochs != 2)
        or receipt.get("epochs") != epochs
        or receipt.get("epoch") != epochs
        or receipt.get("next_batch") != 0
        or receipt.get("training_cells", 0) < 1
        or receipt.get("cells_seen") != receipt.get("training_cells", 0) * epochs
        or receipt.get("total_steps", 0) < 1
        or receipt.get("completed_steps") != receipt.get("total_steps")
    ):
        raise ValueError("Pretraining run is incomplete or not the requested two-epoch formal campaign")
    progress = payload["progress"]
    if (
        not 1 <= progress["epoch"] <= epochs
        or progress["next_batch"] != 0
        or progress["cells_seen"] != progress["epoch"] * receipt["training_cells"]
        or not 1 <= progress["completed_steps"] <= receipt["completed_steps"]
    ):
        raise ValueError("Transfer checkpoint must be from a completed pretraining epoch")
    manifest_path = Path(config["data_manifest"])
    if (
        digest_file(manifest_path) != config["data_manifest_sha256"]
        or receipt.get("data_manifest_sha256") != config["data_manifest_sha256"]
    ):
        raise ValueError("Pretraining data lineage differs")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("purpose") != purpose or digest_file(vocabulary) != manifest["vocabulary"]["sha256"]:
        raise ValueError("Pretrained vocabulary identity/purpose differs")
    genes = json.loads(vocabulary.read_text())
    if len(set(genes)) != len(genes) or len(genes) != config["backbone"]["genes"]:
        raise ValueError("Pretrained vocabulary is not unique or has wrong size")
    student = PretrainingNetwork(BackboneConfig(**config["backbone"]), HeadConfig(**config["heads"]))
    state = {
        name.removeprefix("student."): value for name, value in payload["model"].items() if name.startswith("student.")
    }
    student.load_state_dict(state, strict=True)
    if any(not torch.isfinite(value).all() for value in student.state_dict().values() if value.is_floating_point()):
        raise ValueError("Pretrained Student contains nonfinite parameters")
    student.eval()
    identity = {
        "checkpoint_sha256": checkpoint_sha256,
        "completion_sha256": completion_sha256,
        "pretraining_config_sha256": payload["config_sha256"],
        "vocabulary_sha256": digest_file(vocabulary),
        "source_branch": "student",
        "source_checkpoint_epoch": progress["epoch"],
        "completed_pretraining_epochs": epochs,
        "purpose": purpose,
    }
    return student, identity
