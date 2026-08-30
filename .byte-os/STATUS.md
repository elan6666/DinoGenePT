---
schema_version: 1
mode: auto
project_kind: existing_codebase
stage: review
current_workflow: byte-review
next_workflow: git-handoff
harness_status: ready
review_verdict: ship
iteration_count: 13
hard_blocked: false
updated_at: 2026-08-30T19:28:00+08:00
---

# Status

Goal: deliver Reactome/SIGNOR decomposition and partner-identity controls,
strict gene-disjoint GGI, and four matched GenePT property-task replications.

All four corrected control corpora and 17,730 × 2,048 vectors are complete.
Reactome/SIGNOR decomposition, bounded-text leakage audits, masked/shuffled
partner controls, strict ten-seed 10,870-gene splits, and the four pinned
property tasks all pass final fairness audits.

Review 6 returns `ship`. Local and server tests, Ruff, build, CLI, JSON, script
syntax, secret scan, and checksum sync pass. Compact receipts and documentation
are ready for Git handoff. No GraD-Pert model was run or modified; user-owned
untracked `uv.lock` remains excluded.

Local source of truth: `/Users/elan/code/GenePT-Seed`.
Server execution mirror: `/data/yilangliu/GenePT-Seed`.
