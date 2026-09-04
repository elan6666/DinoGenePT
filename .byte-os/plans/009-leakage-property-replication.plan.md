# Plan 009: Leakage controls and GenePT property replication

## Goal

Separate Reactome and SIGNOR effects, test partner-identity leakage under both
the released GGI split and repeated gene-disjoint splits, and reproduce four
GenePT property tasks under one matched vector/evaluation contract.

## Acceptance criteria

1. Reactome-only, SIGNOR-only, masked-SIGNOR, and shuffled-SIGNOR corpora retain
   the exact 17,730-label master order and emit source/text hashes.
2. Every new Doubao checkpoint has 17,730 exact text/model hits at width 2,048;
   every final NPZ covers the exact master and 10,870 GGI allowlist.
3. Static receipts quantify SIGNOR overlap separately for GGI train/test and
   positive/negative labels.
4. Fixed GGI comparisons share the existing 10,870 genes, pair rows, labels,
   L2 normalization, pair sum, logistic regression, and seed 42.
5. Gene-disjoint GGI uses seeds 42–51, zero train/test gene overlap, identical
   split receipts across embeddings, and the same estimator contract.
6. Four pinned GenePT property tasks use the same 519-gene allowlist, repeated
   stratified five-fold splits for seeds 42–51, and both logistic regression
   and 500-tree random forest.
7. Compact comparison receipts, source documentation, Byte OS iteration/review
   evidence, local/server tests, Ruff, build, CLI, secret scan, checksum sync,
   Git commit, and GitHub push all pass.
8. GraD-Pert is neither run nor modified; untracked `uv.lock` is preserved.

## Execution boundary

Local Git is source of truth. Downloads, embeddings, and benchmarks materialize
only under `/data/yilangliu/DinoGenePT`. Credentials remain process-only and
large data, vectors, checkpoints, and logs remain untracked.
