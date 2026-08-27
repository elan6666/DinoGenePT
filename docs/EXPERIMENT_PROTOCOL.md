# Experiment protocol

## Fixed scientific contract

- Text source: the same `NCBI_UniProt_summary_of_genes.json` from GenePT v2.
- Primary benchmark: Gene2vec's released fixed GGI train/test split.
- Pair representation: sum of the two gene vectors, matching the official
  GenePT Figure 2 notebook.
- Classifier: logistic regression with a fixed seed and convergence budget.
- Primary preprocessing: per-gene L2 normalization for every embedding model.
- Sensitivity preprocessing: native vectors with no normalization.
- Common universe: only genes that have NCBI+UniProt text; every compared model
  is evaluated on the exact same pair rows.
- Metrics: accuracy, AUROC, average precision, sample count, and coverage.

The official notebook adds two gene vectors and uses the released Gene2vec
split. GenePT-Seed keeps those choices but records coverage and convergence-safe
settings explicitly.

## Conditions

1. Paper reference: NCBI text with `text-embedding-ada-002`.
2. Latest official GenePT: NCBI+UniProt text with `text-embedding-3-large`.
3. GenePT-Seed: identical NCBI+UniProt text with `doubao-embedding-vision`.

The third condition must not be reported until a rotated Ark key has been
privately provisioned on the server and the resulting embedding manifest has
passed shape, coverage, and checkpoint audits.
