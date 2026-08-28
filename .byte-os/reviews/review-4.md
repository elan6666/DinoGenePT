---
created_at: 2026-08-28T22:45:00+08:00
verdict: ship
---

# Verdict

ship

# Findings

- P0/P1: none.
- P2, Research: the earlier downstream-training scope did not match the user's
  embedding-only request. All GraD-Pert processes and queued work were stopped;
  the final result uses only fixed embeddings and the prior GenePT GGI evaluator.
- P2, QA: SQLite row count was insufficient because stale text hashes can share
  gene keys. The final audit requires gene, text SHA-256, model, and dimension;
  all 10,870 GO rows now match.
- P3, Interpretation: GO gains are small and measured on one released split.
  Documentation reports absolute deltas and makes no general superiority claim.
- P2, Delivery: an unanchored rsync `results/` rule also excluded the tracked
  `docs/results/` receipt. Root anchoring fixes the source sync without exposing
  or deleting server-generated results.

# Role Notes

- Product/PM: final scope exactly matches the corrected two-condition request.
- Tech Lead: official texts remain immutable; the extension and GO augmentation
  are separate checksum-pinned artifacts.
- QA: both embeddings have 100% coverage and all benchmark fairness identities
  match.
- Research: GO-EXP improves all three requested metrics on this coordinate.
- Security: no secret is tracked or logged; tmux no longer contains the key.

# Required Changes

None.

# Suggested Changes

Use multiple splits or another GenePT task before making a broad GO claim.

# Verification Gaps

No multi-seed confidence interval and no independent task replication. Neither
is part of this selected experiment.

# Decision

Run final tests, synchronize Mac/server code, commit, push, and deliver.

# Current Verification

Local final gate passed 32 tests, Ruff, JSON validation, secret scan, and wheel/
sdist build after all result and iteration files were written. The final sync
gate additionally requires matching receipt hashes and an empty dry run.

That final gate passed: server tests/Ruff/build/CLI succeeded, Mac/server receipt
SHA-256 values match, and the post-apply checksum dry run is empty.

The parallel `.gitignore` rule was also root-anchored, so the compact receipt is
tracked while top-level generated results remain excluded.
