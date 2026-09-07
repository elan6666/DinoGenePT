"""Lossless NPZ -> memory-mapped NPY conversion in a new immutable corpus root."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from dinogenept.cell.genecompass_data import GeneCompassDataset
from dinogenept.cell.sampling import CropConfig
from dinogenept.provenance import atomic_write_json, digest_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = GeneCompassDataset(args.source, CropConfig())
    data.verify()
    args.output.mkdir(parents=True)
    manifest = json.loads(args.source.read_text())
    shutil.copyfile(data._path(manifest["vocabulary"]["path"]), args.output / "vocabulary.json")
    manifest["source_manifest_sha256"] = digest_file(args.source)
    shards = []
    for i, shard in enumerate(data.shards):
        ids, values = data._read(i)
        arrays = {}
        for key, array in (("gene_ids", ids), ("expression", values)):
            path = args.output / f"shard-{i:05d}-{key}.npy"
            np.save(path, array, allow_pickle=False)
            mapped = np.load(path, mmap_mode="r", allow_pickle=False)
            if not np.array_equal(array, mapped):
                raise ValueError("Conversion changed values")
            arrays[key] = dict(path=path.name, sha256=digest_file(path))
        shards.append(dict(format="npy", cells=shard["cells"], arrays=arrays))
    manifest["shards"] = shards
    atomic_write_json(args.output / "manifest.json", manifest)
    GeneCompassDataset(args.output / "manifest.json", CropConfig()).verify()
    print(json.dumps(dict(status="verified", cells=len(data), shards=len(shards))), flush=True)


if __name__ == "__main__":
    main()
