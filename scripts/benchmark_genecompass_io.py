"""Compare identical random cells/crops across NPZ and mmap corpora on CPU."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from dinogenept.cell.genecompass_data import GeneCompassDataset
from dinogenept.cell.sampling import CropConfig
from dinogenept.provenance import atomic_write_json


def main():
    output = Path("results/genecompass-io-comparison-v1.json")
    if output.exists():
        raise FileExistsError(output)
    indices = np.random.default_rng(42).choice(500000, size=512, replace=False)
    results = {}
    for name in ("genecompass-human500k-v1", "genecompass-human500k-mmap-v1"):
        dataset = GeneCompassDataset(Path("data/pretraining") / name / "manifest.json", CropConfig())
        manifest_sha = dataset.verify()
        trials = []
        for _ in range(2):
            checksum = hashlib.sha256()
            start = time.perf_counter()
            for index in indices:
                for view in dataset[int(index)]["views"]:
                    for key in ("gene_ids", "expression", "targets", "hidden"):
                        checksum.update(view[key].tobytes())
            elapsed = time.perf_counter() - start
            trials.append(dict(seconds=elapsed, cells_per_second=512 / elapsed,
                               crop_sha256=checksum.hexdigest()))
        results[name] = dict(manifest_sha256=manifest_sha, trials=trials)
        print(json.dumps({name: results[name]}), flush=True)
    digests = {trial["crop_sha256"] for item in results.values() for trial in item["trials"]}
    if len(digests) != 1:
        raise ValueError("Crops differ between formats or trials")
    atomic_write_json(output, dict(status="passed", scope="CPU_random_access_not_training_throughput",
                                   sampled_cells=512, seed=42, cache_policy="post_verify_warm_cache",
                                   results=results))


if __name__ == "__main__":
    main()
