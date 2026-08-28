---
schema_version: 1
mode: auto
project_kind: greenfield
stage: delivered
current_workflow: byte-deliver
next_workflow: byte-status
harness_status: not_required
review_verdict: ship
iteration_count: 3
hard_blocked: false
updated_at: 2026-08-28T20:45:00+08:00
---

# Status

Goal: Deliver GenePT-Seed as a reproducible research repository for comparing
Doubao embeddings against the latest official GenePT on selected paper
benchmarks, with server-ready execution, tests, review, iterations, and handoff.

Delivered result: 10,870/10,870 Doubao vectors at dimension 2,048 and a matched
GGI result of 0.73223 accuracy, 0.82099 AUROC, and 0.81147 average precision.

Latest verified server gate: 18 tests passed, Ruff passed, sdist/wheel built,
artifact and benchmark receipts passed, and source sync dry-run had no changes.

Local source of truth: `/Users/elan/code/GenePT-Seed`.
Server execution mirror: `/data/yilangliu/GenePT-Seed`.
