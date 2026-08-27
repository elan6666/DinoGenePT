---
schema_version: 1
mode: auto
project_kind: greenfield
stage: reviewed
current_workflow: byte-review
next_workflow: byte-build
harness_status: not_required
review_verdict: block
iteration_count: 3
last_updated: 2026-08-28
---

# Status

Goal: Deliver GenePT-Seed as a reproducible research repository for comparing
Doubao embeddings against the latest official GenePT on selected paper
benchmarks, with server-ready execution, tests, review, iterations, and handoff.

Open blocker: a rotated Agent Plan API key must be privately provisioned on the
server before live embedding generation. The exposed chat credential will not
be used.

Latest verified server gate: 15 tests passed, Ruff passed, sdist/wheel built,
all Gene2vec hashes passed, and source sync dry-run had no changes.

Local source of truth: `/Users/elan/code/GenePT-Seed`.
Server execution mirror: `/data/yilangliu/GenePT-Seed`.
