# GeneCompass adaptive handoff

## Launch intent and phase boundary

After preparation acceptance, complete its Codex goal, verify it is inactive,
then enable one same-task heartbeat. If either GPU is occupied, hand off resource
waiting (do not launch or preempt). When both are free, disable the heartbeat
before the next active GPU-validation phase. Run the guarded smoke in unique
tmux `dinogenept-genecompass-smoke`, log `.runtime/genecompass-smoke.log`.
After successful smoke, prepare/verify the formal launch and hand off the full
epoch to `dinogenept-genecompass-train`, log `.runtime/genecompass-train.log`.
Commands and full-epoch acceptance are in GENECOMPASS_PRETRAINING_HANDOFF.md.
Never duplicate a live process, overwrite an output, or bypass the smoke gate.

## Queries (server project root, approved SSH skill)

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
tmux list-sessions
tail -n 3 .runtime/genecompass-download-parallel.log
```

For launched runs, inspect the corresponding log, `launch.json`, `exit.json`,
`metrics.jsonl`, and `smoke.json` or `completion.json` in
`results/pretraining/genecompass500k-mmap-{smoke,one-epoch}-v1`.
Record actual PID/start identity, config/source hashes and output when launched.
Missing PID/session alone is never successful completion.

## Cadence and terminal rules

- Download/resource wait: 20 minutes; no release-time estimate for occupied GPUs.
  The remaining5M download is hours-scale at the observed approximately0.9MB/s;
  recompute ETA from bytes, never treat it as a deadline. Two unchanged byte
  observations (40 minutes) or process failure trigger investigation, not restart.
- Full epoch: one hour. Estimate duration only from measured real optimizer
  steps after startup. No advancing metrics for two checks triggers investigation;
  explicit error/nonzero exit is actionable immediately. Do not invent a speed.
- Inspect 5M download alongside training on the same hourly heartbeat; do not
  keep a second monitor. Download success requires final size, gzip integrity
  and checksum receipt. Preserve data when a receipt is missing.
- Unreachable server: retry on cadence, notify after two consecutive failures.
- Unchanged/non-actionable state: remain quiet. Notify completion, failure,
  materially changed readiness, or required user action only.
- Completion/failure: record pending handoff, disable and verify this heartbeat
  before creating a validation/repair goal. Remove after both one-epoch result
  validation and the retained5M download obligation complete. Never train5M,
  fine-tune perturbations, or modify GraD-Pert in this campaign.

## Concise heartbeat prompt (prepared inside the goal)

Continue DinoGenePT only. Read .byte-os/STATE.md and
docs/GENECOMPASS_MONITOR_PLAN.md in /Users/elan/code/DinoGenePT. If a goal is
active, disable this monitor first. Using approved SSH, inspect existing500k
smoke/training and5M downloader, GPU ownership, logs and receipts; never duplicate
or preempt. Wait quietly unless readiness changes or an error/terminal event
occurs. Downloads/resource wait20min, training1h on this same heartbeat. On
readiness/terminal event record handoff and disable this monitor before starting
the next bounded goal. Overall scope: all500000 continuous-expression cells,
no validation, one pretraining epoch;5M download only; no GraD-Pert.
