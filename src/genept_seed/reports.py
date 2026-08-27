"""Small, auditable result writers."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

from .provenance import atomic_write_json


def write_results(rows: Iterable[Mapping[str, object]], output: Path) -> Path:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot write an empty result table")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        atomic_write_json(output, materialized)
        return output
    fields = sorted({key for row in materialized for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)
    return output
