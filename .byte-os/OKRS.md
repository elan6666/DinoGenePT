# Objective and Key Results

## Objective

Deliver a credible, minimal, reproducible comparison of GenePT-Seed against the
latest official GenePT while changing only the embedding backbone.

## Key Results

1. Preserve identical NCBI + UniProt text and report exact source checksums.
2. Produce embeddings for at least 99% of valid source genes, with resumable
   checkpoints and explicit failures.
3. Run two selected paper-aligned benchmark families with identical samples,
   splits, preprocessing, classifiers, seeds, and metrics across models.
4. Pass local tests and server verification with no credential or large-data
   artifact tracked by Git.
5. Publish a documented repository whose results distinguish reproduced,
   newly measured, and unavailable evidence.

Baseline: repository and GitHub remote were empty on 2026-08-28.

Evidence needed: test output, server receipts, checksums, coverage manifests,
benchmark CSV/JSON, and a current review verdict.

