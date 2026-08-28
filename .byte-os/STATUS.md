---
schema_version: 1
mode: auto
project_kind: existing_codebase
stage: delivered
current_workflow: byte-status
next_workflow: byte-status
harness_status: ready
review_verdict: ship
iteration_count: 6
hard_blocked: false
updated_at: 2026-08-28T20:25:00+08:00
---

# Status

Goal: compare GenePT-Seed from the completed NCBI+UniProt corpus against the
same corpus plus bounded GO-EXP using the matched GenePT GGI evaluation.

Execution is complete. Both server embeddings cover all 10,870 selected genes
at width 2,048. Their GGI receipts share data, universe, split, classifier,
normalization, pair operator, seed, dependency versions, and pair counts.
GO-EXP improves Accuracy/AUROC/AP by 0.00305/0.00313/0.00394 on this split.

No GraD-Pert model was trained or evaluated. Earlier exact-axis prior artifacts
remain prepared evidence only and are not part of the reported GGI score.

Delivery verification: local/server 32 tests, Ruff, package build, CLI, secret
scan, matching result-receipt hash, and empty checksum sync dry run passed.

Local source of truth: `/Users/elan/code/GenePT-Seed`.
Server execution mirror: `/data/yilangliu/GenePT-Seed`.
