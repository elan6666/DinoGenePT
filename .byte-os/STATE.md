# Active project state

## Outcome contract (2026-09-07, goal active)

- Native DinoGenePT: 12/768 single-direction hybrid KDA, no short convolution;
  DINO+iBOT+KoLeo+CellFM two reconstruction losses, independent knowledge locals.
- Choose audited real continuous-expression pretraining data for two RTX5090s;
  scGPT data route excluded. Complete 2 full pretraining epochs, then CellFM
  Adamson and Norman each 10 LoRA fine-tuning epochs. No ablation runs now.
- Do not preempt/share unrelated GPU jobs. Dataset selection and model building
  precede supplemental GenePT embedding API work. Ark Agent Plan existing
  Keychain helper/model/config must be reused secret-safely; no API calls yet.
- AGENTS.md holds durable engineering standards, not this campaign sequence.
- Local /Users/elan/code/DinoGenePT; remote /data/yilangliu/DinoGenePT;
  branch codex/dinogenept-cleanup, GitHub elan6666/DinoGenePT. Preserve uv.lock.

## Implemented and verified, not formal experimental results

- GenePT corpus/vector/benchmark subsystem preserved. New cell/ native
  backbone, LoRA, distillation, sampling, pretraining orchestration, checkpoint,
  dataset and runner; dinogenept pretrain --config ... [--resume ...].
- KDA chunk/reference values and gradients agree on CPU. Five pretraining
  losses, Teacher stop-gradient/EMA, masked-value exclusion, real-zero vs PAD,
  deterministic crops, all-cell epoch coverage tested.
- Runner fixture: two complete small CPU epochs; interrupted/resumed parameters
  AND RNG exactly match uninterrupted run. Two-rank Gloo accumulation passes.
  Removed dead first-block AttnRes query/norm after DDP exposed missing gradients.
- Corpus verifier rejects modified hashes, invalid/duplicate vocabulary,
  fractional normalized values masquerading as raw counts, invalid CSR/library
  totals and duplicate cell IDs across train/validation shards.
- Knowledge-local perturbation model CPU tests pass: frozen backbone stays
  unchanged through LoRA optimizer step; missing locals omitted; partial-combo
  vectors rejected; canonical target order; main-only inference API.
  Default TextBase main, GO/Protein/Pathway/HPA optional independent locals;
  rank16/alpha32/dropout.05; full-axis rank128 delta decoder. See model document.
- Native CellFM simulation splitter matches fixed official source for single
  and combo fixtures, seeds1/3/42, including subgroups and original row order.
  Raw reference files stored remote .runtime/references and hash-checked by
  test-only AST execution. No upstream research-model runtime imports.
- Own server .venv: torch2.13.0+cu130 and Census/data dependencies. Local torch
  absent; CPU integration tests run on server, never borrow GraD-Pert env.
- Current verification: server 97 tests passed in 17.14 seconds, Ruff, native
  pretrain CLI help, wheel and sdist pass. Local suite/build also pass with
  torch/reference-dependent tests skipped. No CUDA correctness claim yet.

## Live materialization (inspect exact handles before restarting)

Snapshot 2026-09-07 09:59 UTC: both jobs confirmed live, not failed.

- Census metadata PID448508 / original exec81061. Candidate collections:
  Tabula Sapiens e5f58829-1a66-40b5-a624-9046778e74f5 and cross-tissue immune
  atlas 62ef75e4-cbea-454e-a0ce-998ec40223d3. Query primary human cells in
  Census LTS2023-12-15, nnz>=200, raw_sum>0. Metadata output expected at
  data/pretraining/census-two-atlas-candidates-v1; not created until query ends.
  Process RSS grew to ~1.2GB; no raw expression shards downloaded.
- Inventory already exists: data/pretraining/census-2023-12-15-inventory.json,
  651 source datasets. Candidate target 500k stratified training cells, subject
  to actual eligibility, donor/study overlap audit and real GPU throughput.
- CellFM archive PID457997 / original exec7856: scripts/prepare_cellfm_data.py
  --output data/official/cellfm-15138665. CellFM_data.zip.part ~1.4GB of
  5,296,319,440 bytes at snapshot. Publisher MD5 pinned; will extract ONLY
  adamson.h5ad and norman.h5ad after full checksum and write archive_receipt.json.
- Both GPUs still occupied by unrelated PIDs391010/391014 (~6GB each).
  No DinoGenePT CUDA work launched; formal pretraining/fine-tuning epochs = 0.

## Remaining critical path

1. Poll live jobs; freeze source/donor/assay/overlap audit, vocabulary and
   actual raw-count pretraining shards. Manifest labels alone do not prove audit.
   Ensure efficient random sparse access rather than repeated NPZ decompression.
2. Inspect real CellFM artifacts, axis, scale, stored DE/splits and notebook
   norman-1000 discrepancy. Fixture split parity is not actual-data parity.
3. Finish condition/context bag data pipeline and 10-epoch LoRA runner,
   pretrain checkpoint/vocabulary transfer, fixed validation/test evaluation.
   Enforce no held-out post observations in train/teacher/HVG/prototypes.
4. After dataset/model ready, supplement independent source-only GenePT vectors
   on server through approved Ark Keychain helper; preserve exact caches/rate cap.
5. When GPUs truly free, full-default forward/backward/DDP capacity and BF16
   parity/throughput audit; finish actual 2+10+10 epochs and result/performance
   receipts. CPU fixture completion must never count as formal training.
6. Continue tests/lint/build/secret scan, code checksum sync, commit/push and
   final requirement-by-requirement outcome audit. Goal remains active.
