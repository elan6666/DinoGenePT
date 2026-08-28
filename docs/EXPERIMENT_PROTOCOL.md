# Experiment protocol

## Fixed scientific contract

- Base text source: the immutable GenePT v2
  `NCBI_UniProt_summary_of_genes.json`, plus an explicitly versioned extension
  for missing current or historical symbols. Existing GenePT texts are never
  overwritten.
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
4. GenePT-Seed+GO-EXP: the completed GenePT-Seed corpus plus bounded GO terms
   supported by EXP, IDA, IMP, IEP, HTP, HDA, HMP, or HEP evidence, encoded by
   the same Doubao model. `NOT`, IPI, IGI, and HGI annotations are excluded.

For the GGI comparison, both Seed conditions select the exact same 10,870
genes. The added missing-symbol records lie outside this released benchmark,
so they improve corpus coverage without changing its pair rows. GO text is
present for 9,597 of the 10,870 selected genes.

## Verified Ark execution settings

- Endpoint: Agent Plan `/api/plan/v3`, not separately billed `/api/v3`.
- Output: dense float vectors with an asserted dimension of 2,048.
- Batch size: 10; live tests showed that 16 and 32 return HTTP 400.
- Throughput: three workers with at least four seconds between request starts.
- Recovery: SQLite records every completed gene and skips it on restart.
- Credential handling: server process environment only; never source, args, or
  logs.

The final vector audit must show 10,870/10,870 coverage, one dimension, finite
values, and a manifest fingerprint for the selected source text.
