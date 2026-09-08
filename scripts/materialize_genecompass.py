"""Materialize published continuous values without executing upstream models."""

import argparse
import base64
import hashlib
import json
import pickletools
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow as pa

from dinogenept.provenance import atomic_write_json, digest_file


def vocabulary():
    url = "https://api.github.com/repos/xCompass-AI/GeneCompass/git/blobs/24ffded13058b548e134bee3f7dac109f41299d1"
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = base64.b64decode(json.load(response)["content"])
    checksum = hashlib.sha256(raw).hexdigest()
    if checksum != "b86750520401bee48e777e65d68f4fd1ba1eade3cd5f7b391bb90d014e72e525":
        raise ValueError("Official token dictionary checksum differs")
    allowed = {"PROTO", "FRAME", "EMPTY_DICT", "MEMOIZE", "MARK", "SHORT_BINUNICODE",
               "BININT1", "BININT2", "SETITEMS", "STOP"}
    literals = []
    for op, arg, _ in pickletools.genops(raw):
        if op.name not in allowed:
            raise ValueError("Unexpected dictionary opcode")
        if op.name in {"SHORT_BINUNICODE", "BININT1", "BININT2"}:
            literals.append(arg)
    keys, values = literals[::2], literals[1::2]
    if len(keys) != 50558 or len(set(keys)) != 50558 or set(values) != set(range(50558)):
        raise ValueError("Invalid token dictionary")
    mapping = dict(zip(keys, values, strict=True))
    genes = sorted(gene for gene in mapping if gene.startswith("ENSG"))
    lookup = np.full(50558, -1, dtype=np.int32)
    lookup[0] = 0
    for index, gene in enumerate(genes, 1):
        lookup[mapping[gene]] = index
    return genes, lookup, checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cells", type=int, choices=(50000, 500000), default=500000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh output; never overwrite a materialization")
    receipt = json.loads(Path(str(args.archive) + ".receipt.json").read_text())
    if digest_file(args.archive) != receipt["sha256"]:
        raise ValueError("Archive changed")
    genes, lookup, token_sha = vocabulary()
    args.output.mkdir(parents=True)
    atomic_write_json(args.output / "vocabulary.json", genes)
    shards, rows, species = [], 0, set()
    with tarfile.open(args.archive, "r|gz") as archive:
        for member in archive:
            if not member.name.endswith(".arrow"):
                continue
            if not member.isfile():
                raise ValueError("Arrow member is not regular")
            for batch in pa.ipc.open_stream(archive.extractfile(member)):
                columns = batch.to_pydict()
                ids = np.asarray(columns["input_ids"], dtype=np.int32)
                values = np.asarray(columns["values"], dtype=np.float32)
                lengths = np.asarray(columns["length"]).reshape(-1)
                species.update(np.asarray(columns["species"]).reshape(-1).tolist())
                if ids.shape != (batch.num_rows, 2048) or values.shape != ids.shape:
                    raise ValueError("Expected aligned top2048 records")
                if ids.min() < 0 or ids.max() >= len(lookup) or (lookup[ids] < 0).any():
                    raise ValueError("Nonhuman or unknown gene token")
                native = lookup[ids]
                valid = native > 0
                if not np.isfinite(values).all() or (values[valid] <= 0).any() or (values[~valid] != 0).any():
                    raise ValueError("Invalid value/PAD alignment")
                if not np.array_equal(valid.sum(1), lengths) or (lengths == 0).any():
                    raise ValueError("Invalid length metadata")
                for row in native:
                    positive = row[row > 0]
                    if len(np.unique(positive)) != len(positive):
                        raise ValueError("Duplicate gene token in cell")
                name = f"shard-{len(shards):05d}.npz"
                np.savez(args.output / name, gene_ids=native, expression=values)
                shards.append(dict(path=name, sha256=digest_file(args.output / name), cells=batch.num_rows))
                rows += batch.num_rows
                print(json.dumps(dict(rows=rows, shards=len(shards))), flush=True)
    if rows != args.expected_cells:
        raise ValueError(f"Expected exactly{args.expected_cells} cells; got{rows}")
    atomic_write_json(args.output / "manifest.json", dict(
        schema="dinogenept.genecompass.continuous.v1", purpose="formal_pretraining",
        protocol="genecompass_all_cells_one_epoch_no_validation", downstream_overlap="unknown",
        expression="genecompass_published_continuous", data_id=f"genecompass-human{rows // 1000}k-v1",
        training_cells=rows, validation_cells=0, archive_sha256=receipt["sha256"],
        token_dictionary_sha256=token_sha, species_codes=sorted(species), shards=shards,
        vocabulary=dict(path="vocabulary.json", genes=len(genes), sha256=digest_file(args.output / "vocabulary.json")),
    ))


if __name__ == "__main__":
    main()
