---
id: 007
title: Completed-corpus Seed versus Seed+GO GGI delivery
status: complete
wave: 7
updated_at: 2026-08-28T19:50:00+08:00
owner_role: QA Engineer
depends_on: [006]
start_directory: .
agents_context_stack: [AGENTS.md]
subagent_policy: none
---

# Goal

Generate and compare GenePT-Seed embeddings from the completed NCBI+UniProt
corpus with and without bounded GO-EXP, then deliver matched GGI metrics.

# Scope

Server-only, checkpointed Ark embedding generation; exact 10,870-gene common
universe; the released Gene2vec GGI split; L2 normalization; pair-sum logistic
regression; three evidence-led review passes; compact receipts and final docs.

# Non-Goals

No GraD-Pert training or evaluation, no multi-seed significance claim, and no
claim that GO causes general improvement beyond this one released GGI split.

# Acceptance Criteria

Both 10,870-row embeddings are complete and finite; metric receipts share all
fairness identities; Accuracy/AUROC/AP and hashes are recorded; final review
says ship; local, GitHub, and server source are synchronized.

# Verification

Vector coverage checks, cross-condition receipt comparison, tests/lint/build,
secret scan, source-sync dry run, and current final review.

# Completion Evidence

Both conditions have 10,870 finite vectors of width 2,048 and 100% selected
coverage. The common GGI contract is identical. Base Accuracy/AUROC/AP is
0.73223/0.82099/0.81147; GO-EXP is 0.73528/0.82411/0.81541. No GraD-Pert run is
part of this result.
