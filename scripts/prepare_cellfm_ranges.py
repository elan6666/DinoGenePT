"""Extract only two ZIP_STORED CellFM members via bounded HTTP ranges.

The ZIP central-directory CRC is verified by ZipExtFile and each output gets a
SHA-256 receipt. This path does NOT claim verification of the full archive MD5.
Stop the full-archive writer before passing its preserved prefix as a cache.
"""

import argparse
import json
import zipfile
from pathlib import Path

from inspect_cellfm_archive import SIZE, URL, RemoteDirectoryReader

from dinogenept.provenance import atomic_write_json, digest_file, utc_now

EXPECTED = {"CellFM/adamson.h5ad": (89486806, 2394048644), "CellFM/norman.h5ad": (106937332, 2470504895)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = args.output / "range_receipt.json"
    if receipt.exists():
        contents = json.loads(receipt.read_text())
        for name, entry in contents["files"].items():
            if digest_file(args.output / name) != entry["sha256"]:
                raise ValueError("Previously verified range extraction changed")
        print(json.dumps({"status": "already_verified", "receipt": str(receipt)}), flush=True)
        return
    remote = RemoteDirectoryReader(args.prefix)
    files = {}
    with zipfile.ZipFile(remote) as archive:
        for member, (size, crc) in EXPECTED.items():
            info = archive.getinfo(member)
            if info.compress_type != zipfile.ZIP_STORED or info.file_size != size or crc != info.CRC:
                raise ValueError("Official archive directory no longer matches the inspected source")
            name = Path(member).name
            target = args.output / name
            temporary = args.output / (name + ".part")
            if target.exists() or temporary.exists():
                raise FileExistsError("Unreceipted member requires inspection before resuming")
            print(json.dumps({"status": "extracting", "member": member, "bytes": size}), flush=True)
            with archive.open(info) as source, temporary.open("xb") as out:
                while chunk := source.read(1024 * 1024):
                    out.write(chunk)
            if temporary.stat().st_size != size:
                raise ValueError("Extracted member size differs")
            temporary.replace(target)
            files[name] = {"sha256": digest_file(target), "bytes": size, "zip_crc32": crc, "archive_member": member}
    atomic_write_json(
        receipt,
        {
            "status": "raw_downstream_data_not_training_ready",
            "source": URL,
            "archive_bytes": SIZE,
            "full_archive_md5_verified": False,
            "integrity": "HTTPS_range_headers+ZIP_entry_CRC32+local_SHA256",
            "network_bytes": remote.transferred,
            "cached_archive_prefix_bytes": remote.prefix_size,
            "files": files,
            "created_at": utc_now(),
        },
    )
    print(
        json.dumps({"status": "verified_extracted", "receipt": str(receipt), "network_bytes": remote.transferred}),
        flush=True,
    )


if __name__ == "__main__":
    main()
