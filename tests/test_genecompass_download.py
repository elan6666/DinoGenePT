import runpy
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[1] / "scripts/download_genecompass_human.py"


def test_parallel_resume_preserves_prefix(tmp_path, monkeypatch):
    module = runpy.run_path(str(MODULE))
    resume = module["parallel_resume"]
    part = tmp_path / "archive.part"
    part.write_bytes(b"original")
    monkeypatch.setitem(resume.__globals__, "range_block", lambda url, start, end, total: b"x" * (end-start+1))
    resume("unused", part, 30, 4)
    assert part.read_bytes() == b"original" + b"x" * 22


def test_failed_range_does_not_append(tmp_path, monkeypatch):
    module = runpy.run_path(str(MODULE))
    resume = module["parallel_resume"]
    part = tmp_path / "archive.part"
    part.write_bytes(b"original")

    def fail(*args):
        raise ValueError("Bad range")

    monkeypatch.setitem(resume.__globals__, "range_block", fail)
    with pytest.raises(ValueError, match="Bad range"):
        resume("unused", part, 30, 4)
    assert part.read_bytes() == b"original"
