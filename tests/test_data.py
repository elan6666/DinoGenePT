import hashlib
import zipfile

import pytest

from dinogenept.data import download, extract_allowlisted, verify_md5, verify_sha256


def test_checksum_and_allowlisted_extraction(tmp_path):
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("nested/expected.json.", "{}")
        handle.writestr("ignored.txt", "no")
    extracted = extract_allowlisted(archive, tmp_path / "out", {"expected.json"})
    assert [path.name for path in extracted] == ["expected.json"]
    expected = hashlib.md5(archive.read_bytes()).hexdigest()
    verify_md5(archive, expected)
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_md5(archive, "0" * 32)
    verify_sha256(archive, hashlib.sha256(archive.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_sha256(archive, "0" * 64)


def test_missing_archive_member_fails(tmp_path):
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("other", "x")
    with pytest.raises(ValueError, match="missing expected"):
        extract_allowlisted(archive, tmp_path / "out", {"expected"})


def test_download_retries_and_replaces_atomically(tmp_path, monkeypatch):
    attempts = 0

    class Response:
        done = False

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self, _):
            if self.done:
                return b""
            self.done = True
            return b"payload"

    def fake_urlopen(*_, **__):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("temporary")
        return Response()

    monkeypatch.setattr("dinogenept.data.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("dinogenept.data.shutil.which", lambda _: None)
    monkeypatch.setattr("dinogenept.data.time.sleep", lambda _: None)
    output = download("https://example.invalid/file", tmp_path / "file", max_retries=1)
    assert output.read_bytes() == b"payload"
    assert attempts == 2
