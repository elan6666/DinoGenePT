# Active project state

Latest monitor Sep8 08:14:45: no active goal; same downloader1565317 alive,
10638852096/25676724557 bytes, approximately0.259MB/s since resume. No exit
receipt or new429 in current log. GPU0 remains occupied by1456767; no500k
smoke/completion receipts. Continue same20min query quietly, no restart.

## Bounded download preparation (Sep8 04:10+ server time)

- Preparation verified locally/server (two pacing/failure tests, Ruff, CLI),
  synced/pushed c0fd426, goal completed. Controlled download started04:12:26,
  PID1565317, shell1565316. First exact8MiB appended:6880755712 total bytes;
  process alive. This verifies resumed progress, not final integrity/completion.
  Same heartbeat restoredACTIVE20min; no active goal.
- One1MiB probe at retained offset passed206, exact Content-Range and length.
  GPU0 remains occupied;500k training not launched. No partial file mutation.
- Implemented --bounded --workers1 --request-interval10: one8MiB request at
  a time, ten seconds between completed chunks; exact range/length gates,
  contiguous prefix, lock and final gzip/SHA checks retained. Any failure
  exits, no automatic retry. This is not evidence of sustained recovery yet.
- After preparation goal completion launch once in tmux
  dinogenept-genecompass-download; new .runtime/genecompass-download-paced.log
  and .runtime/genecompass-download-paced.exit. Keep earlier logs unchanged.
- Reactivate same dinogenept heartbeat20min after launch; inspect NEW log and
  actual bytes/PID. Completion requires final integrity receipt;40min no growth
  triggers inspection. On429 stop, respect Retry-After and lengthen cooldown;
  do not repeat this retry automatically. Duration hours-scale, refine only
  from observed rate. Overall50万1epoch and5M-download-only scope unchanged.

## Monitor observation (Sep8 03:50 server time)

- No active goal. GPU0 still occupied by1456767; no500k smoke/formal outputs.
- Failed single-stream exit code1 verified; partial still6872367104 bytes.
  One permitted cooldown probe returned206 with exact one-byte Content-Range
  and total25676724557. This does NOT prove sustained access: the preceding
  recovery also passed a one-byte probe before streaming returned429.
- No restart this check. Next scheduled check may use one bounded1MiB Range
  probe at the same offset (discard probe bytes); verify206/exact range/length.
  Only then consider a bounded, paced recovery design after stopping monitor.
  Keep20min query cadence and silence unchanged resource/download state.

## Download recovery handoff (Sep8 03:30 server time)

- Recovery preparation completed and synced as13ebe3d; controlled single-stream
  retryPID1551138 started03:30:08 but ALSO receivedHTTP429 before any append.
  Download is stopped, NOT progressing.6,872,367,104 bytes remain intact.
  New log .runtime/genecompass-download-single.log preserves failure; inspect
  matching .exit. Heartbeat `dinogenept` active20min with no active goal.
  Next query may make ONE bounded Range probe after cooldown. If429 persists,
  respect Retry-After if provided and extend cadence; do not relaunch. Only
  consider another prepared recovery after endpoint accepts sustained access.
- The four-worker downloader exited on HTTP429; session/PID gone. Preserved
  partial archive is6872367104 bytes. Exact one-byte HTTP206 Content-Range at
  that offset and total25676724557 verified after cooldown; no file changed.
- Recovery preparation: existing --workers1 uses one streaming HTTP request
  from the retained prefix, validates resume and final size/gzip/SHA. Completed
  500k/50k archives are verified and skipped, not redownloaded. No code change.
- Launch intent after this preparation goal completes: same download session
  name, --workers1, new .runtime/genecompass-download-single.log, shell exit
  code .runtime/genecompass-download-single.exit. Preserve the old error log.
  One controlled retry, not an unlimited restart loop. On another429 stop and
  reassess server cooldown; monitor partial bytes and tail this NEW log.
- Same heartbeat paused before recovery goal; reactivate20min after launch
  with verified process identity.40min without byte growth triggers inspection.
  GPU0 remains occupied by1456767;500k smoke/formal outputs remain absent.

## Authoritative preparation/monitor handoff (2026-09-08, latest)

- Preparation goal is now complete after verification and commit80ded9f was
  pushed/synchronized. Replacement heartbeat `dinogenept` was then created
  ACTIVE every20minutes (verified app result), with no active goal overlap.
  Current phase: resource wait plus retained download. No new training process
  was started. Next: disable heartbeat once both GPUs are available, then
  proceed with the documented GPU smoke and formal-epoch handoff.
- New Codex goal covers verified launch preparation AND the monitor design;
  it excludes waiting for GPU availability and the full epoch. This supersedes
  older goal descriptions below. Overall deliverable remains all500000 cells,
  no validation, continuous expression, one epoch, overlap unknown;5M download
  only. No perturbation fine-tuning or GraD-Pert operations.
- Safe launcher: scripts/launch_genecompass.py. Requires idle GPUs, exclusive
  lock and fresh output; formal mode requires successful matching smoke source,
  config, checkpoint, metrics and exit evidence. Twelve gate tests passed.
  Server smoke --check-only passed (NOT GPU authorization); train --check-only
  correctly refused missing real smoke. Real GPU smoke has NOT run.
- Monitoring design and launch intent: docs/GENECOMPASS_MONITOR_PLAN.md.
  User now explicitly authorizes replacement monitor after genuine goal
  completion. Resource/download20min, training1h, same heartbeat, quiet while
  unchanged. Disable before next goal. No active monitor during preparation.
- Current GPU0 is occupied by unrelated PID1456767; do not preempt or silently
  reduce world_size2.5M downloader1513852 (started Sep8 01:56:32 server time),
  tmux dinogenept-genecompass-download, four workers, observed6.067/25.677GB
  at approximately0.9MB/s. Keep it running; it does not block500k readiness.
- Native mmap conversion and IO comparison remain verified as documented;
  do not rerun them. No end-to-end GPU speedup claimed. Routine server tests
  MUST use CUDA_VISIBLE_DEVICES= to avoid allocating on occupied GPUs.
- Preparation verification: all180 server CPU-isolated tests passed; local
  twelve launcher tests passed; full server Ruff and scoped local Ruff passed;
  wheel/sdist build, CLI and git diff whitespace checks passed. Formal launch
  rejection without smoke was also checked directly on the server.

## Latest supervision override: mutually exclusive (2026-09-08)

- User explicitly requires no active heartbeat while goal mode is active.
  Design the next concise check at preparation completion; activate it only
  after the goal genuinely completes. Disable/remove it before the next goal.
  Server downloads may continue during active implementation. Supersedes
  earlier references to simultaneous goal and heartbeat supervision below.
- Current heartbeat is absent. Do not recreate it during this active goal.
  GPU0 still occupied by PID1456767 at this check; GPU smoke remains pending.

## Latest handoff status (2026-09-08)

- Performance changes committed/pushed as f4fb9ea and synchronized. Full
  GPU-isolated CPU suite168 tests passed; local wheel/sdist build and changed
  file Ruff checks passed. CLI confirms --smoke-one-step. IO comparison and
  prepared commands documented in docs/GENECOMPASS_PRETRAINING_HANDOFF.md.
- Mmap configs exist at .runtime/genecompass500k-mmap-smoke-config.json and
  .runtime/genecompass500k-mmap-one-epoch-config.json; neither has been launched.
- Still missing: real default dual-rank GPU smoke, GPU performance evidence,
  and an active scheduled handoff. GPU0 remains occupied by PID1456767; no
  preemption allowed. The 5M downloader PID1513852 remains live.
- Automation `dinogenept` was found deleted; update returned "does not exist".
  Do not claim an active heartbeat or silently recreate a user-deleted monitor.
  User requested concise checks; confirm recreation before registering anew.
- Existing preparation goal is unfinished, NOT complete. No full epoch has
  run. Keep current corpus and code; do not repeat conversion or IO benchmark.

## Current complex-task contract — authoritative (2026-09-08)

This section supersedes conflicting historical snapshots below. The active
goal tool cannot edit its objective; keep that unfinished goal intact and use
this user-approved performance extension alongside its original acceptance.

### Overall outcome

- Native DinoGenePT default model: train all500000 GeneCompass human cells for
  exactly1 complete pretraining epoch; no heldout validation, no5M training,
  no perturbation fine-tuning. Preserve continuous expression values and record
  downstream overlap as unknown; training loss is not generalization evidence.
- Optimize project performance without changing model/loss/crop/data semantics:
  data IO, batching/prefetch/CPU-GPU transfer, KDA/backbone execution, memory,
  mixed precision and DDP/accumulation. Profile first, retain only verified
  improvements, and do not guarantee100% utilization or a speedup without data.
- Keep requested human50k/500k/5M downloads;5M transfer runs independently and
  must not block work on the ready500k corpus. No changes/runs in GraD-Pert and
  no preemption or sharing of unrelated GPU jobs.

### Current phase: preparation, optimization, tested launch handoff

1. Finalize vocabulary, all-cell data format/loader and immutable1epoch config.
2. Verify performance changes with identical inputs and numerical/gradient
   checks. Report IO throughput separately from model/training throughput;
   record cache conditions, cells/s, tokens/s, data wait and peak GPU memory.
3. Run real-data default-model1step GPU smoke with checkpoint/EMA/mask checks,
   isolated from formal results. Recheck GPU ownership immediately before use.
4. Launch the authorized1epoch run and verify healthy progress; preserve job
   identity, logs, resolved config, manifest/checkpoint hashes and restart rules.
5. Test/lint/build/CLI as applicable and sync local/server/GitHub. Only after
   these actual phase deliverables hold, complete the existing phase goal and
   hand off to the same heartbeat. Do not wait for the whole epoch in this goal.

### Scheduled wait and next phase

- Download waits normally20min, training waits1h, adapt the SAME monitor
  `dinogenept` to concurrent needs. No repeated jobs or unchanged notifications.
- After verified training termination, create the next scoped acceptance goal:
  audit full500000-cell/1epoch coverage, finite losses, checkpoint and performance
  receipts; analyze limits and deliver. Retain monitoring for unfinished
  authorized downloads. Overall completion requires all retained obligations.

### Latest verified state and immediate work

- NPZ500k conversion/loader validation and no-validation fixture tests passed.
- Lossless NPY conversion completed:500000 cells/510 shards at
  data/pretraining/genecompass-human500k-mmap-v1, log.runtime/genecompass-mmap.log.
- IO comparison started in tmux `dinogenept-io-benchmark`; log
  .runtime/genecompass-io-benchmark.log and receipt
  results/genecompass-io-comparison-v1.json. Inspect actual result before claiming
  speedup; compare512 identical sampled cells/crops in both formats.
- Mmap code/tests and benchmark scripts remain pending final review/commit.
  Freeze NEW mmap smoke/formal configs; existing NPZ configs are not updated.
- Real-data GPU smoke and formal training have NOT started. Last GPU snapshot
  showed GPU0 occupied by unrelated PID1456767; refresh before any GPU action.

## Approved protocol: all500k, no validation, one epoch (2026-09-08)

- Conversion and full native loader verification COMPLETED:500000 cells,
  510 shards,23113 human vocabulary genes. Manifest SHA256
  92efa32e20eeb8e291f19bd6a9f196c326bf817a370c1453112eb74b3e684d31.
  Frozen smoke config:.runtime/genecompass500k-smoke-config.json; formal config
  resolver targets .runtime/genecompass500k-one-epoch-config.json. Neither is
  training completion. Confirm config files before reuse; do not overwrite.
- Nine local loader/crop tests passed; seven server loader+no-validation runner
  tests passed, including an actual tiny CPU-fixture epoch with null validation,
  no best.pt and exact cell coverage. This is NOT the real GPU smoke requirement.
  Current remaining gate: GPU availability, real default dual-rank1step smoke,
  then formal launch and healthy-process handoff. No formal run launched yet.

- Implementation progress: scripts/materialize_genecompass.py binds the pinned
  official token dictionary with opcode-only parsing and checksum validation;
  validates human IDs, PAD/value alignment, lengths and gene uniqueness.
  Background tmux `dinogenept-genecompass-materialize` writes
  data/pretraining/genecompass-human500k-v1; log.runtime/genecompass-materialize.log.
  Last observed282706 rows; no final manifest yet. Do not duplicate conversion.
- Native GeneCompassDataset and published continuous crop path implemented;
  train.py accepts the explicit one-epoch/no-validation protocol and records
  null validation metric, no best-validation checkpoint. Server19 existing/new
  runner+sampling tests passed. New loader-specific tests/full real-data checks
  and resolved config still required before launch. GPU0 observed occupied by
  PID1456767; do not preempt it or launch dual-GPU training while occupied.

- User accepted the source-aligned all-cells protocol: all500000 published
  cells participate in one complete pretraining epoch, no validation split.
  Missing study/donor/original-cell metadata is an acknowledged limitation,
  not a reason to fabricate provenance or block this approved pretraining run.
  Record downstream_overlap=unknown; do not present training loss as heldout
  performance or claim downstream contamination was ruled out.
- This supersedes the unresolved protocol decision below. Finish vocabulary
  binding and continuous-scale loader, isolated smoke, then launch on available
  server GPUs. No5M training and no perturbation fine-tuning. Download in parallel.
- Preserve published continuous expression without raw-count renormalization.
  Current selection config now freezes this protocol and one-epoch budget;
  launch_allowed stays false until integration/smoke checks pass.

## Latest execution authorization: human500k full one epoch (2026-09-08)

- User explicitly replaces the preparation-only restriction below: finish
  GeneCompass human500k preparation and train the default model for ONE complete
  pretraining epoch. Do not train human5M or run perturbation fine-tuning.
- Preserve the existing requested downloads; human5M download is not a
  prerequisite for human500k preparation/training. Update the same heartbeat
  from20-minute download checks to hourly checks when training is launched.
- Existing30-epoch defaults are historical campaign settings, not this run's
  budget. Implement a versioned one-epoch config and explicit validation;
  do not counterfeit a30-epoch completion or repurpose smoke receipts.
- The downloaded corpus lacks study/donor/original-cell IDs. Its source-level
  non-overlap audit is unresolved, not passed. User authorization to train does
  not by itself select a replacement provenance/split protocol. Obtain that
  decision before replacing the current audited-donor split contract.
- Existing raw-count loader rejects these fractional preprocessed values;
  native continuous-value loading must preserve the published scale and never
  relabel values raw_counts or normalize by the truncated top2048 sum.

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

- User-requested download acceleration: source supports exact HTTP206 ranges.
  Bounded probe: single8MiB in12.485s (671889 B/s); four8MiB ranges in34.720s
  (966437 B/s aggregate) while original downloader was still active. Added
  opt-in --workers4, bounded8MiB chunks, strict Content-Range/length checks and
  contiguous-prefix append. Old PID1486064 verified by cwd and interrupted with
  SIGINT; partial preserved. Replacement uses same tmux
  `dinogenept-genecompass-download`, now --workers4, log
  `.runtime/genecompass-download-parallel.log`. Final gzip/SHA gate unchanged.

- Content audit COMPLETED (read final JSON): exactly500000 rows across17 shards;
  all expression values finite/nonnegative with fractional float32 values.
  All shards have only input_ids/values/length/species. Both JSON sidecars were
  inspected and contain no study/donor/original-cell provenance. Therefore
  downstream non-overlap and study/donor split cannot currently be certified.
  Do not rerun completed content audit or mark formal_eligible true. Vocabulary
  binding remains pending; seek source provenance before any formal split claim.
  Human5M download PID1486064 remains live, latest logged1335885824 bytes of
  25676724557; preserve same20-minute monitor and existing checkpoint.

- Full archive listing passed:17 regular Arrow shards plus state.json and
  dataset_info.json, no other files observed. Read-only streaming content audit
  now runs in tmux `dinogenept-genecompass-content-audit`, PID1494327.
  Log `.runtime/genecompass-500k-content-audit.log`, output
  `results/genecompass-500k-content-audit.json`. Several shards already decoded
  with continuous fractional values and four fields; final totals and metadata
  require the completed JSON. No extraction or training performed.

- Latest check (2026-09-08 CST): human500k and human50k archives are final,
  with expected sizes and downloader gzip-CRC/SHA256 receipts. Human500k SHA:
  767e52d093ed7cf7e8355d6c7fb3a8b5a59d3b31bcc80658340fa2faec2d2a3f;
  human50k SHA:3d4ed25fa8dd48f616fd6267e9647fc55fcf1da2b0467d95e634a7db80c38f8b.
  These are local integrity hashes, not publisher checksums. Content/provenance
  audit remains pending. Archive listing exec65564 returned only its first
  member before remote process disappeared; do not claim full listing audited.
- Human5M downloader had exited with440401920 bytes preserved. Resumed under
  durable tmux `dinogenept-genecompass-download`, PID1486064; log
  `.runtime/genecompass-download-resume.log`. Resume offset accepted and file
  grew to441450496 bytes. Check this tmux/process, NOT obsolete exec33494 or
  PID981950; retain20-minute monitoring and do not duplicate the job.

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
