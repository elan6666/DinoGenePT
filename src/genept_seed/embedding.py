"""Ark embedding client and resumable gene embedding generation."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Lock

import numpy as np

from .provenance import atomic_write_json, utc_now
from .vectors import save_npz

DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/plan/v3"
DEFAULT_MODEL = "doubao-embedding-vision"


def validate_plan_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized != DEFAULT_BASE_URL:
        raise ValueError(f"base URL must be the Agent Plan endpoint: {DEFAULT_BASE_URL}")
    return normalized


class ArkEmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        dimensions: int | None = None,
        max_retries: int = 5,
    ) -> None:
        self.api_key = api_key or os.environ.get("ARK_API_KEY")
        if not self.api_key:
            raise RuntimeError("ARK_API_KEY is not set")
        self.base_url = validate_plan_base_url(base_url)
        self.model = model
        self.dimensions = dimensions
        self.max_retries = max_retries

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        payload: dict[str, object] = {"model": self.model, "input": texts}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = json.load(response)
                ordered = sorted(body["data"], key=lambda item: item["index"])
                return [np.asarray(item["embedding"], dtype=np.float32) for item in ordered]
            except urllib.error.HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == self.max_retries:
                    message = f"Ark embedding request failed with HTTP {error.code}"
                    raise RuntimeError(message) from error
            except urllib.error.URLError as error:
                if attempt == self.max_retries:
                    raise RuntimeError("Ark embedding request failed after retries") from error
            time.sleep(min(2**attempt, 30))
        raise AssertionError("unreachable")


def _flatten_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(filter(None, (_flatten_text(item) for item in value)))
    if isinstance(value, dict):
        preferred = ("summary", "description", "function", "text", "ncbi", "uniprot")
        pieces = [_flatten_text(value[key]) for key in preferred if key in value]
        if not pieces:
            pieces = [_flatten_text(value[key]) for key in sorted(value)]
        return " ".join(filter(None, pieces))
    return "" if value is None else str(value).strip()


def load_gene_texts(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = []
        for row in raw:
            if not isinstance(row, dict):
                continue
            gene = row.get("gene") or row.get("symbol") or row.get("gene_symbol")
            if gene:
                items.append((gene, row.get("text") or row))
    else:
        raise ValueError("gene text JSON must be an object or list")
    result = {str(gene).upper(): _flatten_text(text) for gene, text in items}
    return {gene: text for gene, text in result.items() if text}


def select_gene_texts(gene_texts: Mapping[str, str], genes: set[str]) -> dict[str, str]:
    requested = {gene.strip().upper() for gene in genes if gene.strip()}
    return {gene: text for gene, text in gene_texts.items() if gene in requested}


def text_statistics(gene_texts: Mapping[str, str]) -> dict[str, object]:
    lengths = np.asarray([len(text) for text in gene_texts.values()], dtype=np.int64)
    if not len(lengths):
        raise ValueError("no non-empty gene texts")
    return {
        "genes": len(lengths),
        "characters_total": int(lengths.sum()),
        "characters_min": int(lengths.min()),
        "characters_median": float(np.median(lengths)),
        "characters_p95": float(np.quantile(lengths, 0.95)),
        "characters_max": int(lengths.max()),
    }


class EmbeddingCheckpoint:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS embeddings (
                gene TEXT PRIMARY KEY, text_sha256 TEXT NOT NULL, model TEXT NOT NULL,
                dimension INTEGER NOT NULL, vector BLOB NOT NULL
            )"""
        )
        self.connection.commit()

    def get(self, gene: str, text_sha256: str, model: str) -> np.ndarray | None:
        row = self.connection.execute(
            "SELECT dimension, vector FROM embeddings WHERE gene=? AND text_sha256=? AND model=?",
            (gene, text_sha256, model),
        ).fetchone()
        if row is None:
            return None
        vector = np.frombuffer(row[1], dtype=np.float32).copy()
        return vector if len(vector) == row[0] else None

    def put(self, gene: str, text_sha256: str, model: str, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float32)
        self.connection.execute(
            "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?)",
            (gene, text_sha256, model, len(vector), vector.tobytes()),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def generate_embeddings(
    gene_texts: Mapping[str, str],
    *,
    embed: Callable[[list[str]], list[np.ndarray]],
    model: str,
    checkpoint_path: Path,
    output_path: Path,
    batch_size: int = 10,
    max_workers: int = 1,
    request_interval: float = 0.0,
    limit: int | None = None,
    expected_dimension: int | None = None,
) -> dict[str, np.ndarray]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if request_interval < 0:
        raise ValueError("request_interval must be non-negative")
    items = sorted((str(g).upper(), text) for g, text in gene_texts.items())
    if limit is not None:
        items = items[:limit]
    checkpoint = EmbeddingCheckpoint(checkpoint_path)
    collected: dict[str, np.ndarray] = {}
    pending: list[tuple[str, str, str]] = []
    try:
        for gene, text in items:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            cached = checkpoint.get(gene, digest, model)
            if cached is not None:
                collected[gene] = cached
            else:
                pending.append((gene, text, digest))
        batches = [pending[start : start + batch_size] for start in range(0, len(pending), batch_size)]

        request_lock = Lock()
        next_request_at = 0.0

        def request(batch: list[tuple[str, str, str]]) -> list[np.ndarray]:
            nonlocal next_request_at
            if request_interval:
                with request_lock:
                    now = time.monotonic()
                    delay = max(0.0, next_request_at - now)
                    next_request_at = max(now, next_request_at) + request_interval
                if delay:
                    time.sleep(delay)
            return embed([text for _, text, _ in batch])

        def store(batch: list[tuple[str, str, str]], vectors: list[np.ndarray]) -> None:
            if len(vectors) != len(batch):
                raise ValueError("embedding response count differs from request count")
            dimensions = {len(np.asarray(vector).reshape(-1)) for vector in vectors}
            if len(dimensions) != 1:
                raise ValueError("embedding response contains inconsistent dimensions")
            dimension = dimensions.pop()
            if expected_dimension is not None and dimension != expected_dimension:
                raise ValueError(
                    f"expected {expected_dimension}-dimensional embeddings, got {dimension}"
                )
            for (gene, _, digest), vector in zip(batch, vectors, strict=True):
                checkpoint.put(gene, digest, model, vector)
                collected[gene] = np.asarray(vector, dtype=np.float32)

        if max_workers == 1:
            for batch in batches:
                store(batch, request(batch))
        else:
            batch_iter = iter(batches)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                active: dict[Future[list[np.ndarray]], list[tuple[str, str, str]]] = {}
                for _ in range(max_workers):
                    try:
                        batch = next(batch_iter)
                    except StopIteration:
                        break
                    active[executor.submit(request, batch)] = batch
                while active:
                    done, _ = wait(active, return_when=FIRST_COMPLETED)
                    for future in done:
                        batch = active.pop(future)
                        store(batch, future.result())
                        try:
                            next_batch = next(batch_iter)
                        except StopIteration:
                            continue
                        active[executor.submit(request, next_batch)] = next_batch
    finally:
        checkpoint.close()
    dimensions = {len(vector) for vector in collected.values()}
    if len(dimensions) != 1:
        raise ValueError("checkpoint contains inconsistent embedding dimensions")
    final_dimension = next(iter(dimensions))
    if expected_dimension is not None and final_dimension != expected_dimension:
        raise ValueError(
            f"expected {expected_dimension}-dimensional embeddings, got {final_dimension}"
        )
    save_npz(output_path, collected, model)
    text_fingerprint = hashlib.sha256()
    for gene, text in items:
        text_fingerprint.update(gene.encode("utf-8"))
        text_fingerprint.update(b"\0")
        text_fingerprint.update(text.encode("utf-8"))
        text_fingerprint.update(b"\0")
    atomic_write_json(
        output_path.with_suffix(output_path.suffix + ".manifest.json"),
        {
            "created_at": utc_now(),
            "model": model,
            "genes": len(collected),
            "dimension": final_dimension,
            "batch_size": batch_size,
            "max_workers": max_workers,
            "request_interval": request_interval,
            "text_fingerprint_sha256": text_fingerprint.hexdigest(),
            "output": output_path.name,
        },
    )
    return collected
