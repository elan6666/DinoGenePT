import importlib.util
import json
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import pytest

spec = importlib.util.spec_from_file_location(
    "download", Path(__file__).resolve().parents[1] / "scripts/download_genecompass_human.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_paced_resume_preserves_prefix(tmp_path, monkeypatch):
    part = tmp_path / "partial"
    part.write_bytes(b"old")
    ranges, sleeps = [], []
    size = 8 * 1024 * 1024 + 4

    def block(url, start, end, total):
        ranges.append((start, end, total))
        return b"x" * (end - start + 1)

    monkeypatch.setattr(module, "range_block", block)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    module.parallel_resume("unused", part, size, 1, request_interval=10)
    assert ranges == [(3, size - 2, size), (size - 1, size - 1, size)]
    assert sleeps == [10]
    assert part.read_bytes() == b"old" + b"x" * (size - 3)


def test_failure_does_not_append_or_retry(tmp_path, monkeypatch):
    part = tmp_path / "partial"
    part.write_bytes(b"old")
    calls = []

    def fail(*args):
        calls.append(args)
        raise ValueError("Invalid bounded range response")

    monkeypatch.setattr(module, "range_block", fail)
    with pytest.raises(ValueError):
        module.parallel_resume("unused", part, 100, 1, request_interval=10)
    assert part.read_bytes() == b"old"
    assert len(calls) == 1


@pytest.mark.parametrize("retry_after", ["3600", "Wed, 09 Sep 2026 00:00:00 GMT", None])
def test_http_failure_reports_metadata_without_retry(monkeypatch, capsys, retry_after):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    calls = []

    def fail():
        calls.append(True)
        raise HTTPError("https://example.invalid", 429, "limited", headers, None)

    monkeypatch.setattr(module, "main", fail)
    with pytest.raises(HTTPError):
        module.cli()
    result = json.loads(capsys.readouterr().out)
    assert result["event"] == "download_http_error"
    assert result["status"] == 429
    assert result["retry_after"] == retry_after
    assert result["observed_at_unix"] > 0
    assert result["automatic_retry"] is False
    assert len(calls) == 1
