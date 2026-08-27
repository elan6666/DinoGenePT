# GenePT-Seed

GenePT-Seed is a reproducible research toolkit that keeps GenePT's official
NCBI + UniProt text and benchmark protocol while replacing only the embedding
backbone with Doubao Embedding Vision.

- Target users: computational biology researchers evaluating gene priors.
- Core problem: embedding-model comparisons are confounded by changing text,
  mappings, splits, downstream models, or dimensions.
- Delivery format: Python package, command-line workflows, server runbooks, and
  machine-readable result manifests.
- Current stage: auto build.
- Success: one command can acquire official inputs, generate checkpointed
  GenePT-Seed embeddings, run selected matched benchmarks, and produce an
  auditable comparison without committing credentials or large artifacts.

