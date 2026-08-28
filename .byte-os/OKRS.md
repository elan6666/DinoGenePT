# Objective and Key Results

## Objective

Deliver a credible, minimal, reproducible comparison of GenePT-Seed against the
latest official GenePT while changing only the embedding backbone.

Extended objective: deliver progressive GO/protein/pathway/HPA text embeddings
that cover every graph-axis and perturbation-target label in the five frozen
GraD-Pert datasets, then compare them only on the fixed GGI protocol.

## Key Results

1. Preserve identical NCBI + UniProt text and report exact source checksums.
2. Produce embeddings for at least 99% of valid source genes, with resumable
   checkpoints and explicit failures.
3. Run one selected paper-aligned benchmark plus a preprocessing sensitivity
   check with identical samples, splits, classifiers, seeds, and metrics across
   models; the user explicitly allowed a subset rather than every experiment.
4. Pass local tests and server verification with no credential or large-data
   artifact tracked by Git.
5. Publish a documented repository whose results distinguish reproduced,
   newly measured, and unavailable evidence.
6. Prove 17,730-label exact corpus/vector coverage, including all five graph
   axes and targets, for three progressive knowledge conditions.
7. Compare Accuracy, AUROC, and AP on one identical 10,870-gene GGI contract
   without training or evaluating GraD-Pert.

Baseline: repository and GitHub remote were empty on 2026-08-28.

Evidence needed: test output, server receipts, checksums, coverage manifests,
benchmark CSV/JSON, and a current review verdict.

## Final Status

- KR1: achieved; GenePT v2 and GGI source receipts are pinned and verified.
- KR2: achieved; 10,870/10,870 vectors, 2,048 dimensions, resumable SQLite.
- KR3: achieved for selected GGI plus native-vector sensitivity.
- KR4: achieved; server lint, 18 tests, build, CLI, secret and sync checks pass.
- KR5: achieved; documented repository and evidence-bounded result table.
- KR6: achieved; all three corpora and vectors contain 17,730 exact labels,
  all 10,870 GGI labels, and 100% graph/target coverage for all five datasets.
- KR7: achieved; all six GGI rows share all 15 fairness fields and
  ProteinPathway is the best tested condition at 0.74968/0.83615/0.82859.
