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
- Current verification: server 111 tests passed in 16.08 seconds, Ruff, native
  pretrain CLI help, wheel and sdist pass. Local suite/build also pass with
  torch/reference-dependent tests skipped. No CUDA correctness claim yet.

## Data progress and current live handle (2026-09-07)

- Census candidate metadata completed: 812,914 normal primary human cells,
  785,863 after excluding Smart-seq2. Frozen 500,000 train +20,000 validation
  cells, 60,664 Ensembl-ID vocabulary, 26 tissue labels. Donor/cell overlaps zero.
  Train contributions 289,059 Tabula Sapiens +210,941 cross-tissue immune atlas;
  heldout donors A29/637C/TSP7/TSP12. Selection artifact:
  data/pretraining/census-two-atlas-500k-selection-v1/selection.json.
- CellFM target H5ADs are fully extracted and CRC/SHA256 verified; full archive
  MD5 NOT verified. A curl retry truncation bug was fixed and tested. The two
  files occupy only the first ~196MB of the stored ZIP; reused the preserved
  ~209MB prefix +4,135 remote directory bytes, then stopped our full downloader.
  Old download PIDs457997/457998 and471872/471873 are stopped; do NOT restart them.
  Preserve prefix at data/official/cellfm-15138665/CellFM_data.zip.part.
  Receipt: same folder/range_receipt.json.
- Adamson: 47,795 cells x1,069 genes, 78 targets; simulation seed3 condition
  splits including train ctrl =53/6/20, cell splits34731/2689/10375.
  Norman:80,506x1,049,100 targets; condition splits104/24/97, cells42095/10552/27859.
  Original continuous log1p X retained without second normalization. Provided
  HVG axes/DE rankings inherited, not claimed train-only upstream HVG fitting.
  Artifacts data/perturbation/cellfm-v1/{adamson,norman}/ contain audit/split/
  axis/observations/evaluation-only DE/exact gene_mapping.json. Every output
  and target symbol maps uniquely to the pretrained IDs; no aliases or drops.
- Raw Census materializer is LIVE: PID482929, exec38004, script
  scripts/materialize_census.py --selection data/pretraining/census-two-atlas-500k-selection-v1
  --inventory data/pretraining/census-2023-12-15-inventory.json
  --output data/pretraining/census-two-atlas-500k-csr-v1.
  It holds .materialization.lock, reuses measurement presence for 2 source
  datasets, and creates 2,048-cell memory-mapped CSR shards with per-shard receipts.
  Last emitted state: fetching train shard0. No verified raw shards yet at snapshot.
  Prior PID479096/exec32196 terminated on modern SciPy 1D COO slicing; fixed to
  explicit 2D CSR row selection and added matrix/array tests before restarting.
  Do not restart a live materializer; inspect handle and shard receipts first.
- Both GPUs still occupied by unrelated PIDs391010/391014 (~6GB each).
  No DinoGenePT CUDA work launched; formal pretraining/fine-tuning epochs = 0.

## Remaining critical path

1. Finish live raw-count extraction and independently verify frozen selection,
   donor/cell membership, counts and measurement metadata. Memory-mapped sparse
   storage implemented; whole-corpus readiness requires all 520,000 cells.
2. Full CellFM notebook-name equivalence remains unproven; report this run as
   released CellFM reduced-axis data plus native source-matched simulation split,
   not full GEARS/GraD-Pert data or a proven author checkpoint reproduction.
3. Condition/context bag sampler is implemented/tested; finish 10-epoch LoRA runner,
   pretrain checkpoint/vocabulary transfer, fixed validation/test evaluation.
   Enforce no held-out post observations in train/teacher/HVG/prototypes.
4. After dataset/model ready, supplement independent source-only GenePT vectors
   on server through approved Ark Keychain helper; preserve exact caches/rate cap.
5. When GPUs truly free, full-default forward/backward/DDP capacity and BF16
   parity/throughput audit; finish actual 2+10+10 epochs and result/performance
   receipts. CPU fixture completion must never count as formal training.
6. Continue tests/lint/build/secret scan, code checksum sync, commit/push and
   final requirement-by-requirement outcome audit. Goal remains active.
