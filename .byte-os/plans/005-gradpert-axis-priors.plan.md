---
id: 005
title: GraD-Pert exact-axis Seed priors
status: complete
wave: 5
updated_at: 2026-08-28T22:20:00+08:00
owner_role: Research Engineer
depends_on: [004]
start_directory: src/dinogenept
agents_context_stack: [AGENTS.md]
subagent_policy: none
---

# Goal

Materialize complete 2,809-row Doubao priors for the frozen Nadig Jurkat graph
axis from NCBI+UniProt+identity text and the matched GO-EXP augmentation.

# Scope

Reuse the pinned corpora, exact case-sensitive axis, checkpointed Ark client,
2,048-dimensional output, manifests, and finite-value/hash audits.

# Non-Goals

No local embedding generation, no per-row model mixing, and no replacement of
the frozen official GenePT artifact.

# Acceptance Criteria

- Both artifacts contain the exact ordered 2,809-gene axis.
- Every vector is finite float32 with width 2,048.
- Corpus, gene-order, model, and output hashes are recorded.
- The API key exists only in the live server process environment.

# Verification

Run server receipt checks, NPZ structural audit, and source test/lint gates.

# Completion Evidence

Both exact-axis artifacts contain 2,809 finite float32 rows of width 2,048,
match the frozen case-sensitive axis exactly, and contain no zero rows. Server
tests, Ruff, package build, manifest checks, and key-removal checks passed.
