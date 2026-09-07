# Default pretraining recipe and launch prerequisites

> Superseded data choice (2026-09-07): the user selected GeneCompass human 500k.
> See `configs/cell/default_data_selection.json`. Do not launch the Census recipe
> below as the current default. GeneCompass archive/content/provenance audits and
> a new vocabulary-bound training configuration must pass first.

`configs/cell/census500k_default_recipe.json` is a **recipe, not a launchable
resolved config**. Full source/count/overlap audit and a matching two-rank CUDA
capacity receipt are mandatory. No partial-corpus, one-card or tiny-model probe
can authorize the full run. No ablations are scheduled.

## Declared optimizer/runtime adaptation

Keep12/768, single-direction9KDA+3globalMLA, no short convolution, Block AttnRes,
width1SiTU, two globals/two locals and all five pretraining losses. Continuous
counts receive full-library normalize/log1p; zero-expression positions filtered.

- Frozen500,000train /20,000donor-heldout validation cells,60,664-gene vocabulary.
- Two ranks, microbatch4/rank, accumulation16: nominal128cells/optimizer step.
  The last smaller accumulation window retains every cell with correct weights.
  Two full epochs:3,907steps/epoch,7,814total. No dropped or repeated padding cells.
- AdamW peak lr1e-4, betas(0.9,0.95), weight decay0.01, default eps1e-8,
  gradient clip1.0. Warmup10% (782steps), cosine to1% of peak. First lr is
  approximately1.28e-7. The warmup, peak LR, AdamW/decay and batch choice are
  **our declared adaptation**, not CellFM's published Adam lr1e-7 reproduction.
- BF16 autocast with FP32 parameters, optimizer, KDA state and sensitive losses;
  activation checkpointing; two spawn-based CPU loader workers/rank (NCCL is
  not fork-safe). Checkpoint every25
  optimizer steps plus epoch boundaries. Validation selects reconstruction best;
  downstream transfer still requires the complete two-epoch run.

The proposed recipe uses native `batched_chunk`: parallelize pair coefficients,
but retain residual-before-triangular-solve order. It is the same KDA mechanism,
not an architecture ablation. Library defaults still use `chunk` until matching
two-rank validation is available. Never substitute the failed factored
`parallel_chunk` candidate into this recipe.

## Current numerical/performance evidence

Both256-token and full2048-token/60,664-vocabulary backbone comparisons passed
the unchanged FP32/BF16 gate. BF16 CLS and gene outputs were bitwise equal in
these forward fixtures. This is **not** bitwise training-trajectory equivalence:
the matched real-shard10-update runs diverged slightly, including reconstruction
losses, as rounding/gradient accumulation propagated. Do not generalize the
forward fixture equality to every input, gradient or optimizer trajectory.

On the same free RTX5090, batch4 and upper-bound crops, the9post-warmup median
step time was10.0982s (`chunk`) versus3.9909s (`batched_chunk`), about2.53x. Both
finished ten updates with finite five-loss/gradient measurements. Candidate peak
allocated memory22,635,177,472bytes; reserved25,138,561,024bytes. Ten repeated-batch
capacity updates do not establish convergence or count as an epoch.

Server receipts (paths relative to `results/`):

| Receipt | SHA256 |
|---|---|
| kda-batched-cuda-numerics-v1.json | ce579decbc75b4a678e8ff8bf79908a3cbcff93dbe36d381cdff84ea3bc5a8df |
| kda-batched-full-cuda-numerics-v1.json | d4c2df7ad089477d4e264e94515e0d006eb531d42269be0b6e70f7e702025eaa |
| capacity-single-reference-matched-v1/completion.json | a339f4c26bfa52ae5fa46e42354b59452398db8712f66c12865c1f397f6636e7 |
| capacity-single-batched-default-v1/completion.json | dae31b0348e7d60feb0a21faa7205859163e4c130b8fbe410d87c92852b9bc4e |

## Next gate, not an already-issued job

After both GPUs are confirmed unoccupied, run the native capacity probe with
`torchrun --standalone --nproc_per_node=2`, `--microbatch 4 --steps 2
--kernel batched_chunk`, the same verified shard and a fresh output directory.
Once the full corpus is ready, call `scripts/resolve_pretraining_config.py`
with `--recipe`, that `--capacity` completion receipt, the full `--numerical`
receipt above and a new `--output` JSON path. It verifies source hashes and all
corpus arrays/identities, binds the artifacts, and reports `resolved_not_launched`.
Only then use native `dinogenept pretrain --config <resolved.json>` under two-rank
torchrun. An existing training output needs explicit `--resume`, never restart.

Corpus/API jobs remain independently live; do not duplicate them while waiting.
After completed pretraining and the full knowledge bundle, resolve the two
dataset-specific LoRA configs as described in `CELLFM_LORA_PROTOCOL.md`.
