# Technical Specification

## Architecture

- Python 3.11+ package under `src/genept_seed`.
- Standard-library HTTP client for the OpenAI-compatible plan endpoint to keep
  dependencies small; API behavior is isolated behind an embedding client.
- NumPy/pandas/scikit-learn for vectors and evaluation.
- CLI built with `argparse`.
- JSONL checkpoint plus compressed NPZ final vectors; pickle loading is
  read-only and restricted to trusted official artifacts.

## Data objects

- `GeneText`: normalized uppercase symbol, text, source status, text SHA-256.
- `EmbeddingSet`: model metadata, ordered genes, matrix, coverage, checksum.
- `BenchmarkDataset`: fixed samples, labels, task, source checksum.
- `RunManifest`: versions, inputs, parameters, environment, outputs.

## Integrations

- Zenodo record 10833191.
- GitHub repositories for pinned GenePT and Gene2vec inputs.
- Agent Plan endpoint with model alias `doubao-embedding-vision`.

## Safety and reproducibility

- API key is read only from `ARK_API_KEY`; never serialized or logged.
- Downloads are streamed and checksummed before extraction.
- Zip extraction allowlists expected members and blocks traversal.
- All random estimators receive an explicit seed.
- Pair/task intersections are frozen before comparing models.
- Server sync excludes `.git`, data, results, checkpoints, virtualenvs, caches,
  and credentials.

## Testing

- Unit tests for text parsing, key normalization, checkpoint resume, safe
  extraction, vector loading, pair construction, and deterministic metrics.
- Mock HTTP integration test for batching and retry behavior.
- CLI smoke tests.
- Server install/test/dry-run verification; real embedding run only after a
  rotated credential is privately available.

