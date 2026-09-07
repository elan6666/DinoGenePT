"""Read only the official ZIP directory via verified HTTP ranges, on server."""

import io
import json
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/api/records/15138665/files/CellFM_data.zip/content"
SIZE = 5296319440


class RemoteDirectoryReader(io.RawIOBase):
    def __init__(self, prefix=None):
        self.position = 0
        self.transferred = 0
        self.prefix = Path(prefix) if prefix is not None else None
        self.prefix_size = self.prefix.stat().st_size if self.prefix is not None else 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else SIZE)
        if not 0 <= self.position <= SIZE:
            raise ValueError("ZIP directory seek out of bounds")
        return self.position

    def read(self, size=-1):
        end = SIZE if size < 0 else min(SIZE, self.position + size)
        length = end - self.position
        if length == 0:
            return b""
        if length > 1024 * 1024:
            raise ValueError("Directory-only probe refuses >1MiB reads")
        if end <= self.prefix_size:
            with self.prefix.open("rb") as handle:
                handle.seek(self.position)
                data = handle.read(length)
            if len(data) != length:
                raise RuntimeError("Cached prefix changed during archive access")
            self.position = end
            return data
        request = urllib.request.Request(URL, headers={"Range": f"bytes={self.position}-{end - 1}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            expected = f"bytes {self.position}-{end - 1}/{SIZE}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise RuntimeError("Server returned an unexpected archive byte range")
            data = response.read(length + 1)
            if len(data) != length:
                raise RuntimeError("Truncated/oversized ZIP directory response")
        self.position = end
        self.transferred += length
        return data


def main():
    remote = RemoteDirectoryReader()
    with zipfile.ZipFile(remote) as archive:
        files = [
            {
                "name": x.filename,
                "compressed_bytes": x.compress_size,
                "bytes": x.file_size,
                "method": x.compress_type,
                "crc32": x.CRC,
                "header_offset": x.header_offset,
            }
            for x in archive.infolist()
            if not x.is_dir()
        ]
    print(
        json.dumps(
            {"source": URL, "archive_bytes": SIZE, "directory_bytes_read": remote.transferred, "files": files}, indent=2
        )
    )


if __name__ == "__main__":
    main()
