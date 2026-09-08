"""Guarded fresh-run launcher; formal training requires a matching GPU smoke."""

import argparse
import fcntl
import json
import math
import os
import subprocess
from pathlib import Path

from dinogenept.provenance import atomic_write_json, digest_file


def validate_smoke(config, smoke_root, shared_process=None):
    receipt = json.loads((smoke_root / "smoke.json").read_text())
    previous = json.loads((smoke_root / "resolved_config.json").read_text())
    if json.loads((smoke_root / "exit.json").read_text()).get("returncode") != 0:
        raise ValueError("Smoke process did not exit successfully")
    if previous.get("execution_mode") != "smoke_one_step":
        raise ValueError("Not an isolated smoke configuration")
    if previous.get("published_cells", 500000) != config.get("published_cells", 500000):
        raise ValueError("Smoke population differs")
    for key in ("backbone", "heads", "crops", "training", "data_manifest_sha256", "protocol"):
        if previous[key] != config[key]:
            raise ValueError(f"Smoke configuration differs: {key}")
    if (receipt.get("status") != "smoke_completed" or receipt.get("completed_steps") != 1
            or receipt.get("epoch") != 0 or receipt.get("world_size") != 2
            or receipt.get("data_manifest_sha256") != config["data_manifest_sha256"]
            or receipt.get("checkpoint_sha256") != digest_file(smoke_root / "smoke.pt")):
        raise ValueError("Invalid GPU smoke receipt")
    launch = json.loads((smoke_root / "launch.json").read_text())
    if launch.get("shared_process") != shared_process:
        raise ValueError("Smoke resource policy differs")
    expected_sources = {str(p) for p in (Path(__file__).resolve().parents[1] / "src/dinogenept/cell").glob("*.py")}
    if set(launch["source_sha256"]) != expected_sources:
        raise ValueError("Incomplete smoke source identity")
    for path, checksum in launch["source_sha256"].items():
        if digest_file(path) != checksum:
            raise ValueError("Source changed after smoke; revalidate")
    records = [json.loads(line) for line in (smoke_root / "metrics.jsonl").read_text().splitlines()]
    steps = [record for record in records if record.get("event") == "optimizer_step"]
    if len(steps) != 1:
        raise ValueError("Smoke must have exactly one optimizer record")
    for key in ("total", "expression", "cell_expression", "dino", "ibot", "koleo", "gradient_norm"):
        if not math.isfinite(steps[0].get(key, math.nan)):
            raise ValueError("Nonfinite or missing smoke metric")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--share-with-process", help="explicitly authorized PID:Linux_start_ticks")
    parser.add_argument("--cells", type=int, choices=(50000, 500000), default=500000)
    args = parser.parse_args()
    root = Path.cwd()
    if str(root) != "/data/yilangliu/DinoGenePT":
        raise ValueError("Run from the approved server project root")
    stem = "smoke" if args.mode == "smoke" else "one-epoch"
    label = f"genecompass{args.cells // 1000}k-mmap"
    path = root / f".runtime/{label}-{stem}-config.json"
    config = json.loads(path.read_text())
    if (config.get("protocol") != "genecompass_all_cells_one_epoch_no_validation"
            or config["training"]["epochs"] != 1 or config["training"]["world_size"] != 2):
        raise ValueError("Wrong campaign configuration")
    if digest_file(config["data_manifest"]) != config["data_manifest_sha256"]:
        raise ValueError("Manifest changed")
    manifest = json.loads(Path(config["data_manifest"]).read_text())
    if config.get("published_cells", 500000) != args.cells or manifest["training_cells"] != args.cells:
        raise ValueError("Launcher/config/data population differs")
    output = Path(config["output"]).resolve()
    if not output.is_relative_to(root / "results/pretraining") or output.exists():
        raise FileExistsError("Output must be a fresh project run directory")
    if args.mode == "train":
        validate_smoke(config, root / f"results/pretraining/{label}-smoke-v1",
                       args.share_with_process)
    command = [str(root / ".venv/bin/torchrun"), "--standalone", "--nproc_per_node=2",
               "--no-python", str(root / ".venv/bin/dinogenept"), "pretrain", "--config", str(path)]
    if args.mode == "smoke":
        command.append("--smoke-one-step")
    if args.check_only:
        print(json.dumps(dict(status="configuration_checked_not_gpu_authorized", command=command)))
        return
    with (root / ".runtime/genecompass-training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from dinogenept.cell.train import _authorized_shared_process, _gpu_guard

        os.environ.pop("DINOGENEPT_SHARED_PROCESS", None)
        if args.share_with_process:
            os.environ["DINOGENEPT_SHARED_PROCESS"] = args.share_with_process
            pid = int(args.share_with_process.split(":")[0])
            if not _authorized_shared_process(pid):
                # A terminated co-tenant does not require sharing, but cannot
                # silently reuse this authorization for a replacement process.
                raise ValueError("Authorized shared process identity is no longer valid")

        _gpu_guard()
        output.mkdir(parents=True, exist_ok=False)
        sources = sorted((root / "src/dinogenept/cell").glob("*.py"))
        atomic_write_json(output / "launch.json", dict(
            config_sha256=digest_file(path), command=command,
            shared_process=args.share_with_process,
            allocator_memory_fraction=0.75 if args.share_with_process else 1.0,
            source_sha256={str(p): digest_file(p) for p in sources},
        ))
        result = subprocess.run(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"})
        atomic_write_json(output / "exit.json", dict(returncode=result.returncode))
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
