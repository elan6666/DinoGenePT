# GeneCompass adaptive handoff

## Current contract — September 8, user override

Use ONE heartbeat, `dinogenept`, **every2hours**. Report progress on EVERY
check, including normal or unchanged progress. This supersedes all historical
20-minute/hourly/silent rules below; do not automatically change the interval.
No active goal and active heartbeat simultaneously.

- Training: user-authorized shared dualGPU; do not wait for exclusive cards or
  modify another job. tmux `dinogenept-genecompass-shared`, queue log/exit
  `.runtime/genecompass-shared-queue.{log,exit}`, output
  `results/pretraining/genecompass500k-mmap-one-epoch-v1`.
- Download: user explicitly authorized another5M resume. New tmux
  `dinogenept-genecompass-download-slow60`, log/exit
  `.runtime/genecompass-download-slow60.{log,exit}`; one worker, bounded8MiB
  requests,60seconds after each batch. Do not follow the old failed paced PID.
- Each report: timestamp; training steps/cells/percent and change since last
  check, finite loss/gradient, LR, recent throughput, checkpoint time; download
  bytes/total/percent and byte delta, speed, process/exit and any429/cooldown.
  State unavailable observations explicitly. ETA is measured, approximate,
  and only shown when meaningful. Keep the summary brief.
- Inspect both jobs, not only GPU utilization. Two consecutive no-progress
  checks warrant investigation; explicit errors or nonzero exits are actionable
  at the check that detects them. A2h schedule is not real-time alerting.
- On429 retain the prefix and structured `download_http_error` metadata.
  Respect Retry-After; no automatic repeat launch or changed rate. Report the
  blocker and arrange a bounded recovery phase if authorized. No extra probes
  while a downloader runs. Never bypass the archive lock.
- Full training success: exit0, epochs1, training_cells=cells_seen500000,
  validation_cells0, completed_steps=total_steps, and validlast.pt. Full download
  success: expected25,676,724,557bytes, fullgzipCRC andSHA receipt.5M is download
  only, not required by500k training. No GraD-Pert or perturbation fine-tuning.
- Before substantive repairs/validation, pause this monitor and verify that
  update succeeded, then create a bounded goal. Restore the SAME monitor after
  goal completion. Remove it only after both obligations are validated complete.

## Historical preparation plan (superseded by the current contract)

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
