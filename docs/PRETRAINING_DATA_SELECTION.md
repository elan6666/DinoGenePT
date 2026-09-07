# Default pretraining data decision

2026-09-07. User delegates source/size choice for 2xRTX5090, requires continuous
expression, excludes scGPT, and requests two complete epochs (no ablations).

## Preferred source

CELLxGENE Census human raw counts, proposed fixed LTS release `2023-12-15`.
This is the public data source/release identified by scPRINT, not scGPT input
tokens and not a claim of reproducing scPRINT's full multispecies training set.
Official dataset metadata and source access must be checked before selection.

- Start with a target **500,000 training cells**, subject to actual eligible
  counts and measured dual-GPU runtime before the final manifest is frozen.
- Human, primary data, raw RNA counts, preferably droplet UMI assays; no
  imputed/integrated matrices or binned/rank-token expression.
- Diverse tissues, donors and collections, with explicit study-level caps;
  do not take the first 500k rows, a single brain atlas or only PBMC.
- Choose an audited collection allowlist, not only keyword-based exclusion.
  Exclude perturbation studies and downstream Adamson/Norman source accessions
  and duplicates. Reserve pretraining validation by donor/study, not random
  duplicate cells. Audit actual eligibility and matching coverage.
- Normalize full library counts before input cropping: log1p(1e4*c/sum(c));
  no integer bins. Raw counts are naturally integers, but their magnitudes are
  retained and transformed to continuous values, not category IDs.
- Preserve gene measurement metadata: absent measurement is not biological zero.
  Keep raw data provenance separate from tokenization and source-only knowledge.

## Suitability and limits

The source offers broad human expression states for reconstruction and
self-distillation; donor/study IDs make leakage checks practical. However,
normal-tissue expression does not supply causal perturbation labels and differs
from K562 perturbation cell populations. The two downstream datasets are needed
to test transfer; diverse pretraining alone is not evidence of perturbation
accuracy. Token/gene coverage, complexity and tissue composition must be audited.

GPU count alone cannot determine optimal cell count or wall time. Measure the
full model, two teacher globals, student globals+locals, and backward pass with
the intended gene-length distribution. For N cells, 2 epochs and measured joint
throughput v cells/s, compute 2*N/v plus validation/checkpoint overhead. Gradient
accumulation affects optimizer batch, not KoLeo neighbor-pool size. No training
or memory fit has been verified yet; both GPUs have unrelated active jobs.

## Alternatives checked

- CellFM: full processed 100M corpus access not verified; downstream ZIP is
  not sufficient evidence. Do not wait indefinitely or use that ZIP as full
  pretraining data.
- scFoundation: official source provides download/preprocess examples and
  points to supplementary accession lists, not a verified monolithic 50M release.
- scPRINT/Census: versioned public raw-count interface with explicit cell,
  donor, assay and dataset metadata; preferred for a traceable bounded subset.

## Sources and readiness

- [scPRINT paper](https://www.nature.com/articles/s41467-025-58699-1)
- [Census raw expression extraction](https://chanzuckerberg.github.io/cellxgene-census/_autosummary/cellxgene_census.get_anndata.html)
- [Census dataset inventory and deduplication](https://chanzuckerberg.github.io/cellxgene-census/notebooks/api_demo/census_datasets.html)
- [scFoundation preprocessing](https://github.com/biomap-research/scFoundation/tree/main/preprocessing)

## Frozen selection and live extraction (2026-09-07)

Candidate metadata returned 812,914 primary normal-human cells. Excluding
27,051 Smart-seq2 cells leaves 785,863 droplet cells (four explicit 10x assay
ontology IDs). The two source publications are [Tabula Sapiens](https://doi.org/10.1126/science.abl4896)
and the [cross-tissue immune atlas](https://doi.org/10.1126/science.abl5197), both
published 13 May 2022. These donor-tissue studies are distinct from the K562
CRISPR studies [Adamson GSE90546](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546)
and [Norman GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344).
This is a study/population provenance argument, not a genome-level identity test.

- Frozen train: 500,000 cells, including 289,059 Tabula Sapiens and 210,941 immune
  atlas cells across 26 tissue labels. Largest-remainder proportional quotas
  within study/donor/tissue/assay; without replacement, <=65% from either study.
- Validation: 20,000 cells from held-out donors A29, 637C, TSP7 and TSP12.
  Donor and Census cell-ID overlaps with training are both zero. Donors were
  chosen using seed42 keyed hashes, before reading expression or model results.
- Frozen Census vocabulary: 60,664 Ensembl IDs, PAD reserved as native ID0.
  Adamson 1,069/1,069 and Norman 1,049/1,049 output symbols map uniquely and
  exactly; all 78 and 100 perturbation targets map. No aliasing/row removal.
- Raw extraction is now running via the official data client, in 2,048-cell
  read-only-memory-mapped CSR shards. Dataset-specific measurement presence has
  been retrieved and agrees with selected cells' measured-gene counts. Per-shard
  checks enforce raw integer counts, full-library sums and measured coordinates.
- The selection is not a completed raw corpus or proof of CUDA capacity. Only
  the final checksum/row/donor/measurement audit may publish training_ready.

Server artifacts: `data/pretraining/census-two-atlas-500k-selection-v1/` and
`data/pretraining/census-two-atlas-500k-csr-v1/`. Tools are
`freeze_census_selection.py` and `materialize_census.py`; the latter has an
exclusive file lock and per-shard receipts for exact resume.
