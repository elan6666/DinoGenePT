# Active project state

## Controlling scope override: stop before formal training (2026-09-07)

- Latest formal budgets: pretraining30 epochs and LoRA fine-tuning30 epochs
  per dataset. These replace old2/10 defaults, NOT the preparation-only scope.
  Do not start those full runs until the user authorizes formal execution.

- Latest LR policy: scLong source-first epoch warmup/cosine/restarts, floor1e-6,
  launch peak5e-5, warmup5, initial cycle15, multiplier2, gamma0.9.
  See configs/cell/lr_sclong.json and docs/SCLONG_LR.md. The official README
  overrides the Python CLI's1e-4 fallback. No optimizer-step reinterpretation.

- The latest user request REPLACES the earlier full-epoch completion target:
  finish all pre-training preparation, including real-data training smoke tests,
  then complete the goal. Do NOT launch full pretraining or full fine-tuning.
- Retain the native default architecture and selected GeneCompass human500k
  corpus. Finish requested downloads, integrity/provenance/scale/gene audits,
  safe loading and vocabulary binding, frozen splits/configs and performance
  instrumentation. Do not bypass leakage gates or silently change datasets.
- Verify one real optimizer step on the selected pretraining data with the
  default model, and Adamson/Norman LoRA forward/backward/update smoke paths.
  Reuse valid existing checks where applicable; the old Census-vocabulary
  capacity probe does not alone prove GeneCompass data integration.
- Smoke checkpoints must be labeled non-formal and isolated from formal
  transfer eligibility. Never fake a completed two-epoch receipt to test LoRA.
- Final gate: finite losses/gradients, correct Teacher EMA/frozen LoRA backbone,
  masks/coverage, checkpoint IO, relevant tests/lint/build/CLI, verified
  local/server/GitHub synchronization, concise handoff and launch instructions.
  Then remove this goal's heartbeat and complete the user-revised goal.
- Downloads/data waits: monitor every20 minutes. Training: every1 hour.
  Confirm actual live jobs and correct progress before yielding. Do not poll
  on every automatic continuation. Goal pause/resume is not tool-exposed;
  disclose this rather than misusing complete/blocked as a pause.
- All full-epoch requirements and launch directions below are historical and
  superseded by this section; they are not current execution authority.

## Latest user override: GeneCompass human 500k (2026-09-07)

- Download recovery at 2026-09-07 15:14 UTC: original PID981950 absent and
  no matching downloader process remained; original exec58135 was not evidence
  of a live remote process. Preserved 2,540,699,648-byte500k partial and resumed
  with the same exclusive-lock downloader, PID1462303 / exec33494. HTTP resume
  offset accepted and file grew to2,542,796,800 bytes. No final integrity receipt
  yet; continue20-minute checks, do not duplicate the resumed process.

- Source audit now pinned at upstream59e5e48: see
  docs/GENECOMPASS_SOURCE_AUDIT.md. Example exports top2048 and only four
  fields (no study/donor/cell IDs); active median division +log2(1+x), not
  raw-count total normalization. Actual archive remains unverified; preserve
  selected source but do not bypass provenance/continuous-scale gates.

- Download human 50k/500k/5M archives on server; choose published human500k as
  current default, superseding Census. No automatic formal Census launch.
- Download PID981950 / exec58135 confirmed live with growing500k .part;
  sequential order500k,50k,5M. Exclusive .download.lock; resume only after
  terminal state. Read archive receipts before skipping completed files.
- Downloads now use a20-minute thread heartbeat rather than continuous active
  polling. Goal pause/resume API is unavailable; do not claim a status change.
- `configs/cell/default_data_selection.json` records this decision separately
  from durable AGENTS.md. Training/validation counts await content and metadata
  audit: 500k published cells cannot mean 500k train plus extra heldout cells.
- Audit actual expression values, top2048 truncation, gene identity, provenance,
  study/donor/downstream overlap before adapting model vocabulary and launching.
- Preserve existing Census artifacts; do not delete data. Old runtime snapshots
  below are historical, not current launch authority.
- Knowledge API job71852 completed all five sources (1777 union genes).
  Matching batched CUDA two-rank capacity1506 passed; formal epochs still zero.

## Outcome contract (2026-09-07, goal active)

- Native DinoGenePT: 12/768 single-direction hybrid KDA, no short convolution;
  DINO+iBOT+KoLeo+CellFM two reconstruction losses, independent knowledge locals.
- Choose audited real continuous-expression pretraining data for two RTX5090s;
  scGPT data route excluded. Complete 2 full pretraining epochs, then CellFM
  Adamson and Norman each 10 LoRA fine-tuning epochs. No ablation runs now.
- Do not preempt/share unrelated GPU jobs. Dataset selection and model building
  precede supplemental GenePT embedding API work. Ark Agent Plan existing
  Keychain helper/model/config must be reused secret-safely. The data/model gate
  is now satisfied and supplemental API generation is running.
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
- Current verification: server147 tests passed in42.29s; Ruff, native CLI,
  wheel and sdist pass. Local90 passed/11 skipped (torch/reference modules),
  Ruff/build pass. Secret-pattern scan returned no matching scoped files.
  Old factored-backend failure and new order-preserving pass are both retained.
- New native CellFM CSR data loader verifies frozen source/audit/mapping/row/gene
  identities, reads X/obs/var only (not uns/DE), and retains continuous values and
  zero-expression candidate genes. Training post/teacher rows are checked against
  train condition/context. Predict has no truth input. Server real IO audit passed:
  Adamson 30,779 train-post cells /3,872 bags/52 groups; Norman38,457/4,856/103.
  Every train condition plus all val/test groups was materialized. Receipt
  results/cellfm-training-io-v1.json sha efe771e14156773d66024ae4bd407098a3352903780abdc2335b551732a0934c.
- New knowledge bank checks exact corpus/vector/model/dimension/fingerprint;
  TextBase must cover all requested genes; optional locals require source-only
  provenance, missing combination targets omit the entire source.
- Native pretraining Student transfer pins checkpoint/completion/vocabulary;
  fixture weights and incomplete formal runs refused. Best may precede final
  epoch but overall run must finish. Exact weight-transfer tests passed.
- Native 10-epoch LoRA runner/evaluator implemented. CPU fixture checks full
  epoch/cell/bag accounting, frozen backbone and exact interrupted/resumed
  parameters/RNG; fixtures are NOT formal training. Validation chooses strict
  best txpert macro Pearson-delta; test runs only after ten epochs. See
  docs/CELLFM_LORA_PROTOCOL.md. Single-GPU jobs isolate the physical UUID before
  CUDA initialization; RNG snapshots touch only the current device.
- Frozen real evaluation states completed, not to be regenerated:
  data/perturbation/cellfm-evaluation-v1/adamson.json SHA
  e2cacc19a73db3f3eff56f9934c8ed446fa7fa9fc4524747ae596d9072eb15c5;
  norman.json SHA 2dd8f80b17bbd7a3290ffcd9eee5f5efc2fe25c94997231840e2f410e9314804.
  26/121 val+test conditions; DE19–20/18–20. Source-matched GraD-Pert method:
  300 ordered PCG64 replacement control draws, seed20260824, exact context
  sampling; t-test/non-dropout/first20-then-target-exclusion; equal-condition
  train+val Systema reference. This is METHOD parity, not canonical-data parity.

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
  Latest checked progress:58 shards /118,784 cells, PID482929 confirmed LIVE
  at elapsed1h22m. All58 shards independently array-hash verified. Final full
  corpus audit still due; manifest not yet present.
  Final manifest not yet ready.
  Prior PID479096/exec32196 terminated on modern SciPy 1D COO slicing; fixed to
  explicit 2D CSR row selection and added matrix/array tests before restarting.
  Do not restart a live materializer; inspect handle and shard receipts first.
- Both unrelated GPU jobs ended; live nvidia-smi at19:40 showed no compute
  processes. Default CUDA capacity probes now ran (see next section), but formal
  pretraining/fine-tuning epochs remain zero. Recheck GPU ownership each launch.
- Knowledge corpus completed at data/knowledge/cellfm-1777-v1/; union1777 output
  genes /176 targets,125 new exact GenePT summaries beyond the old master.
  TextBase1777 (1774 functional summaries +3 existing identity-only records),
  GO1444, Protein1765, Pathway1256, HPA1771. Optional sources are SOURCE-ONLY;
  missing annotations omitted, no aliases or fabricated functional text.
  Corpus receipt SHA e268fb8d23ca1bd1f93e6c51c780810d308ed1713d63030e96dddcfb484599cb.
- Ark generation LIVE PID506096 (original exec71852), one worker, batch10,
  interval8sec, no automatic retries. Original authorized Keychain helper
  injection, no persisted/printed credential. Original corpus miss count5909,
  2104 exact cache hits. Current audit: TextBase1777/1777 complete2048D;
  GO1444/1444 complete; Protein797/1765 cached,968 pending. Later sources
  sequential, do not duplicate. PID506096 live at elapsed31:52.
  Output data/embeddings/cellfm-1777-v1, checkpoint checkpoints/cellfm-1777-v1;
  final bundle.json only after all vectors/provenance pass KnowledgeBank audit.

## CUDA capacity and optimization (not epochs)

- New progress after6a9456a: native `batched_chunk` keeps original residual-before-
  solve order, only batches pair coefficients. Full12/768/60664-vocabulary,
  2048-token numerical gate PASSED without changing thresholds; BF16 forward
  CLS/genes bitwise equal in the audited fixture. Receipt
  results/kda-batched-full-cuda-numerics-v1.json SHA
  d4c2df7ad089477d4e264e94515e0d006eb531d42269be0b6e70f7e702025eaa.
- Same-card real upper-bound batch4/ten-update comparison: warm median10.098s
  original versus3.991s batched (~2.53x); finite updates, candidate22,635,177,472
  bytes peak allocated. Training trajectories are NOT bitwise equal. No formal
  epochs. See docs/DEFAULT_PRETRAINING_LAUNCH.md for evidence and limitations.
- Original `chunk` remains library default; proposed formal recipe explicitly
  selects `batched_chunk` but requires matching two-rank capacity before launch.
  Recipe500k/20k,2epochs,microbatch4/rank,accum16,effective128,AdamW peak1e-4,
  betas.9/.95,wd.01,clip1,10%warmup/cosine are declared native adaptations.
  scripts/resolve_pretraining_config.py refuses incomplete data, mismatched
  model/source/microbatch/world size or incomplete/failed numerical evidence.
- Latest live CPU/API check: raw133,120cells/65shards, no final manifest;
  PID482929 live elapsed1:42:06. TextBase1777,GO1444,Protein1765 complete2048D;
  Pathway725/1256cached,531pending; PID506096 live elapsed51:38. Do not duplicate.
  External GPU0 PID775988 still live (~4.3GB); no DinoGenePT GPU probe is live.
- Superseding last check: raw137,216cells/67shards, PID482929 live1:51:06;
  Pathway1256/1256 completed2048D, HPA377/1771cached,1394pending;
  PID506096 live1:00:38. GPU0 PID775988 still live32:42, not ours.
- DataLoader now explicitly uses spawn for workers>0, per installed PyTorch
  NCCL fork-safety warning. A spawned-loader complete-epoch/exact-resume fixture
  passes. Never inherit GPU contexts into forked data workers.

- scripts/probe_pretraining_capacity.py pins a completed real raw shard and
  full60664 vocabulary,12/768 backbone and normal-sized heads. Worst-bound
  globals2048/2048,locals820/820; no formal checkpoint or epoch claim.
- results/capacity-single-default-v1: two BF16 forward/backward/AdamW/EMA steps,
  microbatch2,119,039,863 Student parameters; step2=10.094s,12,409,045,504 peak
  allocated bytes. CUDA FP32 KDA recurrence/chunk value+gradient parity passed.
- results/capacity-ddp-default-v1: two complete two-rank steps, microbatch4/rank;
  step2=10.752s,0.744 cells/s total,22,912,157,696 peak allocated bytes/maxrank.
  All five losses/gradients finite. Two steps do not establish convergence.
- Native parallel_chunk candidate batches state-independent triangular solves;
  exact mechanism unchanged, original chunk remains default pending audits.
  CPU value/gradient/read-only/strong-decay parity tests pass. V3 CUDA DDP
  step2=3.288s,2.433cells/s,23,121,669,632bytes peak, but BF16 whole-backbone
  numerical audit FAILED on11/393216 gene outputs. Max abs0.09375; original
  chunk remains default. Long2049-token CUDA FP32 primitive and12/768 backbone
  FP32 parity passed. See docs/CUDA_CAPACITY_2026_09_07.md for exact receipts.
- First candidate DDP attempt failed SAFELY before allocation on rank0: torchrun
  creates separate process sessions, so an initialized peer was misidentified
  as unrelated. Verified installed torch Elastic start_new_session=True;
  guard now accepts only a verified common torchrun parent, not arbitrary
  shell siblings. Candidate v2 refused a distinct transient PID688199 (not
  attributed); v3 passed after free-GPU revalidation. Later external PID775988
  occupied GPU0, system Python/cwd /home/yilangliu; do NOT touch. Numerical
  audit used freeGPU1 only. No DinoGenePT GPU job remains live from these probes.

## Remaining critical path

1. Finish live raw-count extraction and independently verify frozen selection,
   donor/cell membership, counts and measurement metadata. Memory-mapped sparse
   storage implemented; whole-corpus readiness requires all 520,000 cells.
2. Full CellFM notebook-name equivalence remains unproven; report this run as
   released CellFM reduced-axis data plus native source-matched simulation split,
   not full GEARS/GraD-Pert data or a proven author checkpoint reproduction.
3. Condition/context sampler, real frozen CellFM IO, source-only vector bank,
   pretrain transfer and10-epoch LoRA runner/evaluator are implemented/tested.
   Resolve artifact-bound formal configs after full inputs exist. Do not redo the
   completed results/cellfm-training-io-v1.json audit or other frozen data jobs.
   Enforce no held-out post observations in train/teacher/HVG/prototypes.
4. Follow the LIVE independent source-only vector job; preserve exact caches,
   source receipts and rate cap, do not restart a live or completed source.
5. Full numerical and single-card batched_chunk probes now pass. When both
   GPUs are free, finish matching two-rank batched_chunk capacity. Once full
   data exists, resolve the formal recipe with source/hash gates; then
   run actual 2+10+10 epochs when full corpus and free GPUs are ready. Preserve
   original data/model/epoch scope. Produce final result/performance
   receipts. CPU fixture completion must never count as formal training.
6. Continue tests/lint/build/secret scan, code checksum sync, commit/push and
   final requirement-by-requirement outcome audit. Goal remains active.
