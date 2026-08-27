---
id: 002
title: Official data and GenePT-Seed embeddings
status: in_progress
wave: 2
updated_at: 2026-08-28T20:20:00Z
owner_role: Backend Engineer
depends_on: [001]
start_directory: src/genept_seed
context_files: [AGENTS.md, .byte-os/TECH_SPEC.md]
agents_context_stack: [AGENTS.md]
subagent_policy: none
---

# Goal

Acquire verified official inputs and generate resumable Doubao embeddings.

# OKR Link

KR1, KR2, and KR4.

# Scope

Zenodo download/extraction, text audit, embedding client, checkpoint/finalize,
doctor command, and mocks.

# Non-Goals

Calling a live API before a rotated server credential is privately configured.

# Steps

## Step 1: Prepare official inputs safely

- Purpose: guarantee source identity.
- Actions: stream download, verify MD5, allowlist zip members, audit JSON keys.
- Files or modules: `data.py`, CLI, tests.
- Expected output: official text and baseline artifacts plus manifest.
- Step verification: fixture archive tests and server dry-run.
- Subagent: none.

## Step 2: Generate checkpointed embeddings

- Purpose: tolerate quota/rate/network interruption.
- Actions: batch text, POST OpenAI-compatible requests, retry retryable errors,
  append atomic JSONL checkpoints, and resume completed genes.
- Files or modules: `embedding.py`, CLI, tests.
- Expected output: resumable checkpoint and compressed final matrix.
- Step verification: mock HTTP batch/resume test.
- Subagent: none.

## Step 3: Finalize and audit vector sets

- Purpose: expose coverage and dimensions.
- Actions: load official pickle/ours NPZ, L2-normalize, write coverage manifest.
- Files or modules: `vectors.py`, CLI, tests.
- Expected output: common vector-set interface and audit JSON.
- Step verification: synthetic vector tests.
- Subagent: none.

# Dependencies

Plan 001.

# Scoped Commands

- Test: `python -m pytest tests/test_data.py tests/test_embedding.py tests/test_vectors.py`
- Lint: `python -m ruff check src/genept_seed tests`
- Typecheck: not required
- Build: `python -m build`

# AGENTS.md Context

- Root context: `AGENTS.md`.
- Module context: none.
- Scoped command source: this plan.
- Safe edit boundaries: data/embedding modules and tests.
- Missing or stale AGENTS.md notes: none.

# Subagent Plan

No subagents because checkpoint, client, and manifests share formats.

# Code Change Guardrails

No secrets in args/files; no unsafe pickle except trusted official artifact;
no broad zip extraction.

# Acceptance Criteria

Verified input preparation and mock-tested resumable embedding generation work.

# Verification

Unit/integration mocks plus credential-free server doctor.

# Experiment Or Measurement

Report text count, valid count, missing count, dimension, and checksum.

# Risks

Actual Agent Plan response shape may differ from OpenAI-compatible assumptions;
capture and adapt only after a safe live smoke call.
