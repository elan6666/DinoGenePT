import hashlib
import subprocess
import zipfile
from pathlib import Path

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


def test_curl_retry_resumes_partial_and_never_falls_back_to_truncation(tmp_path, monkeypatch):
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        target = Path(command[command.index("--output") + 1])
        assert command[command.index("--retry") + 1] == "0"
        assert command[command.index("--continue-at") + 1] == "-"
        if len(commands) == 1:
            target.write_bytes(b"prefix")
            raise subprocess.CalledProcessError(28, command)
        assert target.read_bytes() == b"prefix"
        with target.open("ab") as handle:
            handle.write(b"suffix")

    monkeypatch.setattr("dinogenept.data.shutil.which", lambda _: "/usr/bin/curl")
    monkeypatch.setattr("dinogenept.data.subprocess.run", execute)
    monkeypatch.setattr("dinogenept.data.time.sleep", lambda _: None)
    assert download("https://example.invalid/file", tmp_path / "file", max_retries=1).read_bytes() == b"prefixsuffix"


def test_failed_curl_preserves_partial(tmp_path, monkeypatch):
    partial = tmp_path / "file.part"
    partial.write_bytes(b"existing progress")

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(28, command)

    monkeypatch.setattr("dinogenept.data.shutil.which", lambda _: "/usr/bin/curl")
    monkeypatch.setattr("dinogenept.data.subprocess.run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        download("https://example.invalid/file", tmp_path / "file", max_retries=0)
    assert partial.read_bytes() == b"existing progress"
    assert not (tmp_path / "file").exists()


def test_urllib_refuses_ignored_range_without_overwriting(tmp_path, monkeypatch):
    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    partial = tmp_path / "file.part"
    partial.write_bytes(b"keep")
    monkeypatch.setattr("dinogenept.data.shutil.which", lambda _: None)
    monkeypatch.setattr("dinogenept.data.urllib.request.urlopen", lambda *a, **kw: Response())
    with pytest.raises(RuntimeError, match="partial download preserved"):
        download("https://example.invalid/file", tmp_path / "file")
    assert partial.read_bytes() == b"keep"
