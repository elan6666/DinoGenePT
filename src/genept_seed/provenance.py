from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def digest_file(path: str | Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def environment_manifest() -> dict[str, str]:
    return {
        "created_at": utc_now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def credential_present(variable: str = "ARK_API_KEY") -> bool:
    return bool(os.environ.get(variable, "").strip())


def sanitized_config(
    *, model: str, base_url: str, dimensions: int | None, credential_variable: str
) -> dict[str, Any]:
    return {
        "model": model,
        "base_url": base_url,
        "dimensions": dimensions,
        "credential_variable": credential_variable,
        "credential_present": credential_present(credential_variable),
    }

