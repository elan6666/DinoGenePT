---
schema_version: 1
mode: auto
project_kind: existing_codebase
stage: reviewed
current_workflow: byte-review
next_workflow: byte-deliver
harness_status: ready
review_verdict: ship
iteration_count: 9
hard_blocked: false
updated_at: 2026-08-29T04:12:00+08:00
---

# Status

Goal: deliver three progressive GenePT-Seed+GO knowledge embeddings covering
all five GraD-Pert graph axes and targets, with matched GGI evaluation.

The 17,730-gene master universe, all three append-only corpora, and all three
aligned 2,048-wide Doubao artifacts are complete. Exact checkpoint audits are
17,730/17,730 with pending 0. Corpus and vector receipts prove complete graph
axis and target coverage across all five datasets.

The fixed 10,870-gene GGI comparison is complete. ProteinPathway is best at
0.74968 Accuracy, 0.83615 AUROC, and 0.82859 AP; all six conditions share all
15 fairness fields. No GraD-Pert model was trained or evaluated.

Review 5 verdict is `ship`. Local and server 40-test suites, Ruff, build, CLI,
JSON, credential scan, compact receipt hashes, and checksum sync pass. Remaining
handoff is the final tracked Git commit/push and monitor/goal closure.

Local source of truth: `/Users/elan/code/GenePT-Seed`.
Server execution mirror: `/data/yilangliu/GenePT-Seed`.
