# GeneCompass 5M download: controlled recovery proposal

Status: **proposal only; downloader stopped**, 2026-09-08 13:31 CST.
The independent 500k one-epoch training remains running and is not changed.

## Evidence

- Server archive prefix: `data/official/genecompass-human/randsel_500w_human.tar.gz.part`,
  11,771,314,176 of 25,676,724,557 bytes. No bytes were appended this check.
- Previous one-worker, bounded8MiB,10-second-between-batches run stopped with
  HTTP429 and exit1 at09:31. No downloader or corresponding tmux session alive.
- One bounded probe requested bytes11771314176-11772362751 and received
  HTTP206, exact `Content-Range: bytes 11771314176-11772362751/25676724557`,
  exactly1,048,576 bytes. Probe bytes were discarded. No Retry-After header.
- This tests only one small request. It does not show that an8MiB request or
  sustained requests are allowed; it is not proof that rate limits have cleared.

## Source-inspected behavior

`scripts/download_genecompass_human.py` owns a nonblocking archive-directory
lock. Bounded mode validates206, the complete Content-Range and body length
before appending. With one worker it retains only the contiguous verified
prefix. HTTP errors propagate and terminate the run: no internal retry loop.
Existing completed archives are size/hash checked and skipped. Final promotion
requires expected size, full gzip CRC read and a computed SHA receipt.

The current downloader does **not** persist Retry-After or rate-limit response
metadata, and8MiB is hard-coded. Its request interval is a delay AFTER each
batch, not an instantaneous network bandwidth cap. Slowing requests cannot
guarantee avoiding a server-side quota. No source changes were made this check.

## If the user authorizes another controlled attempt

1. Pause the hourly monitor before a bounded preparation/launch goal. Preserve
   all prior logs/exit receipts and the partial archive. Confirm no downloader
   and acquire the existing exclusive lock; never start a second concurrent run.
2. Add narrowly scoped, tested structured HTTP-error reporting (including
   Retry-After), without automatic retries. Keep range/length checks intact.
3. Proposed conservative setting: one worker, bounded8MiB requests,60-second
   delay between batches. This is a heuristic, not a provider-published limit.
   Do not revert to parallel requests, rotate identities or bypass rate limits.
4. Use a new log/exit lineage. Verify a few actual contiguous chunk appends
   before handoff; a submitted tmux command alone is not healthy progress.
5. On429, stop once, preserve prefix, honor recorded Retry-After and report.
   No automatic retry. On success require final size/gzip/SHA checks.
6. Complete preparation only after its deliverables are verified; then restore
   the same hourly monitor for training and downloading.5M is never a training
   prerequisite and is not authorized for training in this campaign.

Until that decision, the hourly monitor follows500k training only and keeps the
5M download marked awaiting a controlled retry, without repeated source probes.
