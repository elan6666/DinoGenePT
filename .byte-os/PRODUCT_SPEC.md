# Product Specification

## Positioning

A minimal reproducibility layer for apples-to-apples GenePT embedding-backbone
comparisons.

## MVP

- Acquire and verify official GenePT Zenodo v2 inputs.
- Normalize and audit NCBI + UniProt gene text.
- Generate resumable Doubao embeddings through the Agent Plan endpoint.
- Load official Ada/model-3 and GenePT-Seed vectors through one interface.
- Run gene-property and GGI benchmarks with matched samples and seeds.
- Produce CSV/JSON results, coverage reports, and provenance manifests.
- Support local tests and server-only materialization/execution.

## Non-goals

- Rebuilding every GenePT paper figure.
- PPI, cell embedding, batch removal, or perturbation prediction in v0.
- Training an embedding model or using image input.
- Publishing raw data, generated embeddings, keys, or large logs to GitHub.

## Acceptance criteria

- CLI help and dry-run paths work without credentials.
- Secret values never appear in tracked files or command logs.
- Mock API generation resumes without duplicating completed genes.
- Benchmark comparisons use the same intersection of genes/pairs per run.
- Results record model, dimension, text source, split, seed, coverage, and input
  checksums.
- Local test suite and server smoke verification pass.

