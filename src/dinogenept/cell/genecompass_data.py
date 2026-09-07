"""Native published-value loader; all cells train, no fabricated provenance."""

import bisect
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from dinogenept.provenance import digest_file

from .sampling import sample_crops


class GeneCompassDataset:
    def __init__(self, manifest, crops):
        self.path = Path(manifest).resolve()
        self.root = self.path.parent
        self.manifest = json.loads(self.path.read_text())
        m = self.manifest
        if (m.get("schema") != "dinogenept.genecompass.continuous.v1"
                or m.get("protocol") != "genecompass_all_cells_one_epoch_no_validation"
                or m.get("expression") != "genecompass_published_continuous"
                or m.get("downstream_overlap") != "unknown" or m.get("validation_cells") != 0):
            raise ValueError("Invalid published-data protocol")
        self.shards = m["shards"]
        if not self.shards or any(s["cells"] < 1 for s in self.shards):
            raise ValueError("Empty corpus")
        self.ends = np.cumsum([s["cells"] for s in self.shards]).tolist()
        if self.ends[-1] != m["training_cells"]:
            raise ValueError("Row total differs")
        self.gene_count = m["vocabulary"]["genes"]
        self.crops, self.epoch, self.cache = crops, 0, OrderedDict()
        self.validated = set()

    def _path(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Artifact escapes corpus root")
        return path

    def verify(self):
        entries = [self.manifest["vocabulary"]]
        for shard in self.shards:
            entries.extend(shard["arrays"].values() if shard.get("format") == "npy" else [shard])
        if len({e["path"] for e in entries}) != len(entries):
            raise ValueError("Duplicate corpus path")
        for entry in entries:
            if digest_file(self._path(entry["path"])) != entry["sha256"]:
                raise ValueError("Corpus checksum differs")
        genes = json.loads(self._path(self.manifest["vocabulary"]["path"]).read_text())
        if len(genes) != self.gene_count or len(set(genes)) != len(genes):
            raise ValueError("Invalid vocabulary")
        for i in range(len(self.shards)):
            self._read(i)
        return digest_file(self.path)

    def _read(self, index):
        if index not in self.cache:
            shard = self.shards[index]
            if shard.get("format") == "npy":
                if set(shard["arrays"]) != {"gene_ids", "expression"}:
                    raise ValueError("Missing mapped arrays")
                ids, values = [np.load(self._path(shard["arrays"][key]["path"]),
                                       mmap_mode="r", allow_pickle=False)
                               for key in ("gene_ids", "expression")]
            else:
                with np.load(self._path(shard["path"]), allow_pickle=False) as data:
                    ids, values = data["gene_ids"], data["expression"]
            if index not in self.validated and (
                    ids.shape != values.shape or ids.shape != (self.shards[index]["cells"], 2048)
                    or not np.issubdtype(ids.dtype, np.integer) or ids.min() < 0
                    or ids.max() > self.gene_count or not np.isfinite(values).all()
                    or (values[ids > 0] <= 0).any() or (values[ids == 0] != 0).any()):
                raise ValueError("Invalid native shard")
            self.validated.add(index)
            self.cache[index] = (ids, values)
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def __getstate__(self):
        return {**self.__dict__, "cache": OrderedDict()}

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect.bisect_right(self.ends, index)
        row = index - (self.ends[shard-1] if shard else 0)
        ids, values = self._read(shard)
        valid = ids[row] > 0
        return sample_crops(ids[row][valid], values[row][valid], None,
                            cell_id=f"{self.manifest['data_id']}:{index}", epoch=self.epoch,
                            config=self.crops, expression_scale="genecompass_published_continuous")
