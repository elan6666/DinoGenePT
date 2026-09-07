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

Current status: preferred source and target size chosen; collection allowlist,
download, leakage/coverage audit, final counts and capacity receipt pending.
`scripts/census_inventory.py` produces metadata only, not an approved manifest.
