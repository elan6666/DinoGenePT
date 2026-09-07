"""Download three user-selected public archives on the server; never extract code."""

import fcntl
import gzip
import json
import time
import urllib.request
from pathlib import Path

from dinogenept.provenance import atomic_write_json, digest_file

FILES = [
    ("randsel_50w_human.tar.gz", "d46129ce14de622c4a07a1d1574dddaa", 2566066308),
    ("randsel_5w_human.tar.gz", "be29663db8c11f4e59aaf2d572b699fe", 257095465),
    ("randsel_500w_human.tar.gz", "826380b58010377e3fa8724fcf4a5fcc", 25676724557),
]


def main():
    root = Path("data/official/genecompass-human")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, fid, size in FILES:
            final, part = root / name, root / (name + ".part")
            receipt = root / (name + ".receipt.json")
            if final.exists():
                record = json.loads(receipt.read_text())
                if final.stat().st_size != size or digest_file(final) != record["sha256"]:
                    raise ValueError("Existing archive changed")
                continue
            offset = part.stat().st_size if part.exists() else 0
            if offset > size:
                raise ValueError("Partial archive exceeds expected size")
            url = "https://china.scidb.cn/download?fileId=" + fid
            if offset < size:
                request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    if offset and (
                        response.status != 206
                        or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
                    ):
                        raise ValueError("Server did not honor resume offset")
                    if response.status not in (200, 206):
                        raise ValueError("Invalid download response")
                    print(
                        json.dumps({"file": name, "status": "downloading", "offset": offset, "total": size}), flush=True
                    )
                    reported = time.monotonic()
                    with part.open("ab") as stream:
                        while block := response.read(1024 * 1024):
                            if offset + len(block) > size:
                                raise ValueError("Response exceeds expected archive size")
                            stream.write(block)
                            offset += len(block)
                            if time.monotonic() - reported >= 30:
                                stream.flush()
                                print(json.dumps({"file": name, "bytes": offset, "total": size}), flush=True)
                                reported = time.monotonic()
            if part.stat().st_size != size:
                raise ValueError("Incomplete archive; preserve partial and resume explicitly")
            with gzip.open(part, "rb") as stream:
                while stream.read(8 * 1024 * 1024):
                    pass
            record = {
                "file": name,
                "source": url,
                "bytes": size,
                "sha256": digest_file(part),
                "gzip_crc": "passed",
                "contents_audited": False,
            }
            atomic_write_json(receipt, record)
            part.rename(final)
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
