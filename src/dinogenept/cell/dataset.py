"""Sparse shard reader with strict provenance and no pickle/object arrays."""

import bisect
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from dinogenept.provenance import digest_file

from .sampling import CropConfig, sample_crops


class PretrainingDataset:
    """CSR NPZ shards contain raw data/indices/indptr, row_ids and library_sum.

    `indices` are zero-based positions in the pinned vocabulary; native model
    gene IDs are indices+1. `library_sum` retains the full source-cell library,
    including counts outside the selected vocabulary. Only positive entries
    need to be stored. The trainer independently verifies all corpus checksums.
    """

    def __init__(self, manifest: Path, split: str, crops: CropConfig, *, cache_shards: int = 2):
        self.path = Path(manifest).resolve()
        self.root = self.path.parent
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get("schema") != "dinogenept.pretraining.csr.v1":
            raise ValueError("Unknown pretraining shard schema")
        if (
            self.manifest.get("status") != "training_ready"
            or self.manifest.get("leakage_audit", {}).get("status") != "passed"
        ):
            raise ValueError("Pretraining corpus is not audited/ready")
        if self.manifest.get("expression") != "raw_counts":
            raise ValueError("This loader requires raw counts, not normalized/binned/ranked values")
        if cache_shards < 1 or split not in {"train", "validation"}:
            raise ValueError("Invalid cache or split")
        self.shards = [x for x in self.manifest["shards"] if x["split"] == split]
        if not self.shards or any(x["cells"] < 1 for x in self.shards):
            raise ValueError("Empty shard split")
        self.ends = np.cumsum([x["cells"] for x in self.shards]).tolist()
        self.cache_shards, self.crops, self.epoch = cache_shards, crops, 0
        self.cache = OrderedDict()
        self.gene_count = int(self.manifest["vocabulary"]["genes"])
        if self.gene_count < 1:
            raise ValueError("Empty vocabulary")

    def _path(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Artifact path escapes dataset root")
        return path

    def verify(self):
        entries = [self.manifest["vocabulary"], *self.manifest["shards"]]
        if len({entry["path"] for entry in entries}) != len(entries):
            raise ValueError("Repeated artifact path in corpus manifest")
        for entry in entries:
            if digest_file(self._path(entry["path"])) != entry["sha256"]:
                raise ValueError(f"Corpus checksum changed: {entry['path']}")
        vocabulary = json.loads(self._path(self.manifest["vocabulary"]["path"]).read_text())
        if (
            not isinstance(vocabulary, list)
            or len(vocabulary) != self.gene_count
            or any(not isinstance(gene, str) or not gene.strip() for gene in vocabulary)
            or len(set(vocabulary)) != self.gene_count
        ):
            raise ValueError("Vocabulary must contain exactly the unique ordered gene identifiers")
        seen = set()
        for split in ("train", "validation"):
            reader = PretrainingDataset(self.path, split, self.crops, cache_shards=1)
            for shard_id in range(len(reader.shards)):
                values = reader._read(shard_id)
                ids = set(values["row_ids"].astype(str).tolist())
                if seen & ids:
                    raise ValueError("Cell repeated across corpus shards/splits")
                seen.update(ids)
        return digest_file(self.path)

    def __len__(self):
        return self.ends[-1]

    def _read(self, shard_id):
        if shard_id in self.cache:
            self.cache.move_to_end(shard_id)
            return self.cache[shard_id]
        entry = self.shards[shard_id]
        with np.load(self._path(entry["path"]), allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in ("data", "indices", "indptr", "row_ids", "library_sum")}
        n = entry["cells"]
        if data["indptr"].shape != (n + 1,) or data["row_ids"].shape != (n,) or data["library_sum"].shape != (n,):
            raise ValueError("Shard cell metadata mismatch")
        if not np.issubdtype(data["indices"].dtype, np.integer) or not np.issubdtype(data["indptr"].dtype, np.integer):
            raise ValueError("CSR indices must be integers")
        if data["indptr"][0] != 0 or data["indptr"][-1] != len(data["data"]) or np.any(np.diff(data["indptr"]) <= 0):
            raise ValueError("Invalid CSR pointer or empty cell")
        if (
            data["indices"].shape != data["data"].shape
            or np.any(data["indices"] < 0)
            or np.any(data["indices"] >= self.gene_count)
        ):
            raise ValueError("CSR gene index mismatch")
        if not np.isfinite(data["data"]).all() or np.any(data["data"] <= 0):
            raise ValueError("CSR must contain finite positive raw counts")
        if not np.equal(data["data"], np.floor(data["data"])).all():
            raise ValueError("Raw-count contract rejects fractional/normalized values")
        sums = np.add.reduceat(data["data"].astype(np.float64), data["indptr"][:-1])
        if (
            not np.isfinite(data["library_sum"]).all()
            or np.any(data["library_sum"] <= 0)
            or np.any(sums > data["library_sum"] * (1 + 1e-5))
        ):
            raise ValueError("Full library totals do not cover stored counts")
        differences = np.diff(data["indices"])
        differences[data["indptr"][1:-1] - 1] = 1
        if np.any(differences <= 0):
            raise ValueError("Each CSR row must have unique ascending gene indices")
        if np.unique(data["row_ids"]).size != n:
            raise ValueError("Duplicate cell IDs within shard")
        self.cache[shard_id] = data
        while len(self.cache) > self.cache_shards:
            self.cache.popitem(last=False)
        return data

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_id = bisect.bisect_right(self.ends, index)
        row = index - (self.ends[shard_id - 1] if shard_id else 0)
        data = self._read(shard_id)
        start, stop = data["indptr"][row : row + 2]
        return sample_crops(
            data["indices"][start:stop] + 1,
            data["data"][start:stop],
            float(data["library_sum"][row]),
            cell_id=f"{self.manifest['data_id']}:{data['row_ids'][row]}",
            epoch=self.epoch,
            config=self.crops,
        )
