"""Read-only streaming content audit; never extract or execute archive content."""

import argparse
import json
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow as pa

from dinogenept.provenance import atomic_write_json, digest_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = json.loads(Path(str(args.archive) + ".receipt.json").read_text())
    if digest_file(args.archive) != receipt["sha256"]:
        raise ValueError("Archive differs from download receipt")
    members, shards, metadata = set(), [], {}
    with tarfile.open(args.archive, mode="r|gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.name in members:
                raise ValueError("Unsafe or duplicate archive path")
            members.add(member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("Non-regular archive member")
            stream = archive.extractfile(member)
            if path.suffix == ".json" and member.size <= 1024 * 1024:
                metadata[member.name] = json.load(stream)
            elif path.suffix == ".arrow":
                rows, values_count, fractional = 0, 0, 0
                lo, hi = float("inf"), float("-inf")
                reader = pa.ipc.open_stream(stream)
                fields = reader.schema.names
                for batch in reader:
                    rows += batch.num_rows
                    values = batch.column(fields.index("values")).flatten().to_numpy()
                    if not np.isfinite(values).all() or (values < 0).any():
                        raise ValueError("Invalid expression values")
                    values_count += values.size
                    fractional += int(np.count_nonzero(values != np.floor(values)))
                    if values.size:
                        lo, hi = min(lo, float(values.min())), max(hi, float(values.max()))
                shards.append(dict(name=member.name, rows=rows, fields=fields,
                                   values_count=values_count, fractional_values=fractional,
                                   minimum=lo, maximum=hi))
                print(json.dumps(shards[-1]), flush=True)
            else:
                raise ValueError("Unexpected archive member")
    atomic_write_json(args.output, dict(
        archive_sha256=receipt["sha256"], shards=shards, metadata=metadata,
        rows=sum(s["rows"] for s in shards),
        provenance_audited=False, vocabulary_audited=False, formal_eligible=False,
    ))


if __name__ == "__main__":
    main()
