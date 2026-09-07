"""Download three user-selected public archives on the server; never extract code."""

import argparse
import concurrent.futures
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


def range_block(url, start, end, total):
    """Reject ignored ranges or changed length before appending any bytes."""
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
            raise ValueError("Invalid bounded range response")
        content = response.read(end - start + 2)
        if len(content) != end - start + 1:
            raise ValueError("Incomplete bounded range response")
        return content


def parallel_resume(url, part, size, workers, request_interval=0):
    """Bound memory to workers*8MiB; persist only a contiguous completed prefix."""
    offset = part.stat().st_size if part.exists() else 0
    started, initial = time.monotonic(), offset
    chunk = 8 * 1024 * 1024
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool, part.open("ab") as stream:
        while offset < size:
            ranges = [(start, min(start + chunk, size) - 1)
                      for start in range(offset, min(offset + workers * chunk, size), chunk)]
            futures = [pool.submit(range_block, url, start, end, size) for start, end in ranges]
            for future in futures:
                content = future.result()
                stream.write(content)
                stream.flush()
                offset += len(content)
            print(json.dumps(dict(file=part.name, bytes=offset, total=size, workers=workers,
                                  bytes_per_second=(offset-initial)/(time.monotonic()-started))), flush=True)
            if offset < size and request_interval:
                time.sleep(request_interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--bounded", action="store_true", help="bounded ranges also for one worker")
    parser.add_argument("--request-interval", type=float, default=0, help="seconds between range batches")
    args = parser.parse_args()
    if not 0 <= args.request_interval <= 3600:
        parser.error("request interval must be finite and between 0 and 3600")
    if args.request_interval and not args.bounded:
        parser.error("request interval requires --bounded")
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
            if offset < size and (args.workers > 1 or args.bounded):
                parallel_resume(url, part, size, args.workers, args.request_interval)
                offset = part.stat().st_size
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
