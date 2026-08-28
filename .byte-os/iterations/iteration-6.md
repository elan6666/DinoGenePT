---
created_at: 2026-08-28T19:48:00+08:00
evidence_source: matched GGI result JSON and cross-condition receipt audit
originating_review: iteration-5
---

# Evidence used

Both result JSON files, vector hashes, Gene2vec data receipt, gene-universe
receipt, classifier settings, dependency versions, and sample counts.

# Hypothesis and metric

GO-EXP should be evaluated as the only changed condition under the existing
GenePT GGI protocol. Metrics: equality of every fairness field and measured
Accuracy, AUROC, and average precision.

# Diagnosis

Console metrics alone would not prove that pair rows, preprocessing, seed, or
classifier remained fixed.

# Changes made

Added a compact machine-readable comparison receipt and evidence-bounded result
documentation with absolute deltas and an explicit no-GraD-Pert boundary. The
first sync dry run exposed that an unanchored `results/` exclusion also hid
`docs/results/`; root-anchored excludes now protect generated results while
allowing the tracked receipt to synchronize.

# Verification

All fairness fields are identical. Base is 0.73223/0.82099/0.81147; GO-EXP is
0.73528/0.82411/0.81541, for deltas +0.00305/+0.00313/+0.00394. Post-fix sync
must show the receipt on both Mac and server with the same SHA-256.

# Remaining issues

One released split cannot establish general improvement or uncertainty.

# Next decision

Run current ship review and final source synchronization.
