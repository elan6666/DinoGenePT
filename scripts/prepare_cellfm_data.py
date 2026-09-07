"""Acquire only the two official genetic-perturbation H5ADs, with archive audit."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from dinogenept.data import download, verify_md5
from dinogenept.provenance import atomic_write_json, digest_file, utc_now

URL = "https://zenodo.org/api/records/15138665/files/CellFM_data.zip/content"
MD5 = "9d40d163b917419327dd94fcf3882923"
SIZE = 5296319440
MEMBERS = {"adamson.h5ad", "norman.h5ad"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    receipt = output / "archive_receipt.json"
    if receipt.exists():
        data = json.loads(receipt.read_text())
        for name, item in data["files"].items():
            if digest_file(output / name) != item["sha256"]:
                raise ValueError(f"Previously extracted file changed: {name}")
        print(json.dumps({"status": "already_verified", "receipt": str(receipt)}), flush=True)
        return
    archive = output / "CellFM_data.zip"
    if not archive.exists():
        print(json.dumps({"status": "downloading", "bytes": SIZE, "url": URL}), flush=True)
        download(URL, archive)
    if archive.stat().st_size != SIZE:
        raise ValueError("Archive size differs from the pinned official record")
    verify_md5(archive, MD5)
    files = {}
    with zipfile.ZipFile(archive) as handle:
        for name in sorted(MEMBERS):
            entries = [entry for entry in handle.infolist() if Path(entry.filename).name == name]
            if len(entries) != 1 or entries[0].is_dir():
                raise ValueError(f"Archive member missing/ambiguous: {name}")
            target = output / name
            if target.exists():
                raise FileExistsError(f"Unreceipted file requires inspection: {target}")
            temporary = output / (name + ".part")
            with handle.open(entries[0]) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            temporary.replace(target)
            files[name] = {
                "sha256": digest_file(target),
                "archive_member": entries[0].filename,
                "bytes": target.stat().st_size,
            }
    atomic_write_json(
        receipt,
        {
            "status": "raw_downstream_data_not_training_ready",
            "source": URL,
            "publisher_md5": MD5,
            "archive_sha256": digest_file(archive),
            "archive_bytes": SIZE,
            "files": files,
            "created_at": utc_now(),
        },
    )
    print(json.dumps({"status": "verified_extracted", "receipt": str(receipt)}), flush=True)


if __name__ == "__main__":
    main()
