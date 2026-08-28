---
id: 006
title: Matched GraD-Pert prior comparison integration
status: complete
wave: 6
updated_at: 2026-08-28T22:35:00+08:00
owner_role: ML Engineer
depends_on: [005]
start_directory: /Users/elan/code/grad-pert
agents_context_stack: [/Users/elan/code/grad-pert/AGENTS.md]
subagent_policy: none
---

# Goal

Define three self-contained Nadig Jurkat configs that differ only in exact-axis
text prior: latest official GenePT, GenePT-Seed, or GenePT-Seed+GO-EXP.

# Scope

Use `genept_id_residual`, the same 2,809-node graph, split, seed 1, ten epochs,
losses, optimizer, evaluator, and metrics-only artifact policy.

# Non-Goals

No trainer fork, no new model family, no test-set tuning, and no fabricated
official embeddings for the official comparator's 60 zero-prior rows.

# Acceptance Criteria

Configs validate, exact NPZ order/hash checks fail closed, and local/GitHub/
server source identities match before launch.

# Verification

Focused config/text-prior tests, full repo gates, commit/push, server exact-
commit gates, and sync receipt.

# Completion Evidence

Three self-contained configs share common-contract SHA-256
`3aead9e02df0d0557f55668e378318045105db57a2e7f52ee09673065c32db75`.
GraD-Pert integration commit `adade44` and exact-axis full-graph repair commit
`8221ae0c372bb2eecaaa2ab1ac143f07797c0547` are on GitHub; the latter is
identical on Mac and server. Server gates passed 319 tests, Ruff, format, mypy
on 73 source files, and wheel/sdist build.
