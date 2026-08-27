# GenePT-Seed Discussion

Date: 2026-08-28

## User request

Build GenePT-Seed in this repository, run experiments on the yilangliu server,
reuse the official GenePT implementation wherever possible, replace only the
embedding model, keep the NCBI + UniProt text input, and compare GenePT-Seed
against the latest official GenePT on selected experiments from the paper.

## Current understanding

- Local project root: `/Users/elan/code/GenePT-Seed`.
- GitHub repository: `https://github.com/elan6666/GenePT-Seed`.
- Remote experiment root should default to `/data/yilangliu/GenePT-Seed`.
- Official text and baseline artifacts come from GenePT Zenodo v2 record
  `10833191`, especially `NCBI_UniProt_summary_of_genes.json` and
  `GenePT_gene_protein_embedding_model_3_text.pickle`.
- GenePT-Seed changes only the embedding backbone to
  `doubao-embedding-vision`; gene identifiers, source text, task data, splits,
  classifiers, metrics, and random seeds remain matched.
- Official GenePT is a notebook repository rather than an installable library.
  Reuse therefore means pinning its commit/data and invoking or adapting its
  public evaluation logic through a thin compatibility layer, not rewriting
  the scientific method.

## Open questions

1. Confirm that a thin compatibility layer is acceptable where the official
   notebooks contain local absolute paths and cannot be imported directly.
2. Provision the Agent Plan API key privately on the server as `ARK_API_KEY`;
   it is currently unset. The key must not enter Git, logs, chat, or sync jobs.
3. Confirm the first experiment set. Suggested MVP: Table 1 gene-property
   classification, Figure 2 gene-gene interaction prediction, and the Figure 2
   protein-interaction benchmarks for which source data can be reproduced.

## Suggested defaults

- Pin official GenePT commit `3602699e7425a7be577771f8f07e218db6c79b9f`.
- Verify Zenodo archive checksum `md5:3f6ce4317e3a0091978ae5cb8fbf05a3`.
- Compare three named conditions:
  1. paper reference: Ada + NCBI text;
  2. latest official GenePT: text-embedding-3-large + NCBI/UniProt text;
  3. GenePT-Seed: doubao-embedding-vision + identical NCBI/UniProt text.
- Use native dimensions as the primary comparison and a train-fold-only common
  dimensionality sensitivity analysis to separate embedding quality from
  dimensionality effects.
- L2-normalize vectors consistently and report gene coverage/missing mappings.
- Run all downloads, embedding generation, and experiments on the server.
- Keep raw datasets, API keys, generated embeddings, checkpoints, and logs out
  of Git; commit manifests, checksums, configs, code, and compact result tables.

## Confirmed decisions

- This is an embedding-backbone replacement study, not a new GenePT method.
- NCBI + UniProt text is the fixed input for the latest-GenePT versus
  GenePT-Seed comparison.
- The latest official GenePT embedding is the direct baseline.
- Experiments run on `yilangliu@10.24.1.91`, not on the Mac.
- Repository code should avoid duplicating capabilities already available from
  official GenePT, subject to the lack of a packaged API.

## Non-goals

- Reimplementing GenePT cell aggregation or benchmark methods from scratch.
- Changing source text, gene mappings, labels, data splits, metrics, or
  classifier tuning only for GenePT-Seed.
- Claiming GenePT-Seed is an official GenePT release.
- Copying unlicensed official notebook code wholesale into this repository.
- Running perturbation-prediction experiments in the first benchmark tranche.

## Risks

- The official GenePT repository has no LICENSE file and exposes notebooks with
  machine-specific paths, so direct package import is unavailable and wholesale
  code copying is not appropriate.
- Different vector dimensions can confound comparisons unless native and
  dimension-controlled results are separated.
- The GenePT Zenodo description contains a wording inconsistency for the
  gene-protein embedding input; the actual JSON/embedding key coverage and
  archive contents must be audited before experiments.
- The Agent Plan API key is not yet configured on the server.
- Reproducing some paper tasks may require external datasets not bundled in the
  official repository.

## Recommended next command

`byte-start` to initialize the empty repository and convert these confirmed
decisions into project state. Use `byte-shape` after the project skeleton exists.
