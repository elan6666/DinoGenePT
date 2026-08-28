---
created_at: 2026-08-28T19:44:00+08:00
evidence_source: API checkpoint, NPZ manifests, and vector audits
originating_review: iteration-4
---

# Evidence used

SQLite text hashes, model/dimension fields, full NPZ artifacts, and the fixed
GGI gene allowlist.

# Hypothesis and metric

Checkpointed generation should yield complete, single-model vectors without
silently accepting stale text rows. Metrics: 10,870 exact text-hash matches,
width 2,048, finite values, and 100% coverage.

# Diagnosis

The existing GO checkpoint contained stale rows for the same gene keys; row
count alone incorrectly looked complete. Exact text hashes showed only 2,331
initial matches.

# Changes made

Resumed Ark requests under tmux using text-hash-aware cache lookups, then
materialized new base and GO NPZ files. The key was removed from tmux after
process launch.

# Verification

Final GO cache match is 10,870/10,870. Both NPZ files contain 10,870 vectors of
width 2,048 and 100% selected coverage; no API error was logged.

# Remaining issues

None for vector completeness.

# Next decision

Run the matched GGI evaluation and compare full receipts.
