---
id: 003
title: Matched gene-level benchmarks
status: in_progress
wave: 3
updated_at: 2026-08-28T20:28:00Z
owner_role: Research Engineer
depends_on: [002]
start_directory: src/genept_seed
context_files: [AGENTS.md, .byte-os/PRODUCT_SPEC.md, .byte-os/RESEARCH.md]
agents_context_stack: [AGENTS.md]
subagent_policy: none
---

# Goal

Compare official and GenePT-Seed vectors on matched gene-property and GGI tasks.

# OKR Link

KR3 and KR5.

# Scope

Generic binary CV kernel, fixed GGI split, exact intersections, random control,
results and run manifests.

# Non-Goals

PPI, cell-level tasks, perturbation prediction, or headline superiority claims.

# Steps

## Step 1: Implement matched evaluation kernels

- Purpose: remove notebook/path randomness.
- Actions: binary stratified CV with LR/RF; fixed split LR; deterministic seeds.
- Files or modules: `benchmarks.py`, tests.
- Expected output: reusable metric records with fold-level values.
- Step verification: synthetic separable/random datasets.
- Subagent: none.

## Step 2: Implement task adapters

- Purpose: preserve official task semantics.
- Actions: parse Geneformer labels and Gene2vec pairs; resolve symbols from
  supplied mappings; freeze common gene/pair intersections across models.
- Files or modules: `tasks.py`, fixtures, tests.
- Expected output: auditable benchmark datasets.
- Step verification: sample/count/checksum tests.
- Subagent: none.

## Step 3: Produce comparison artifacts

- Purpose: separate measured results from source claims.
- Actions: run selected tasks; write tidy CSV, JSON summary, and manifest.
- Files or modules: CLI, `reports.py`, tests.
- Expected output: comparison table with coverage and uncertainty.
- Step verification: end-to-end synthetic CLI test and server run when inputs
  are available.
- Subagent: none.

# Dependencies

Plan 002.

# Scoped Commands

- Test: `python -m pytest tests/test_benchmarks.py tests/test_tasks.py tests/test_workflow.py`
- Lint: `python -m ruff check src/genept_seed tests`
- Typecheck: not required
- Build: `python -m build`

# AGENTS.md Context

- Root context: `AGENTS.md`.
- Module context: none.
- Scoped command source: this plan.
- Safe edit boundaries: benchmark/task/report modules and tests.
- Missing or stale AGENTS.md notes: none.

# Subagent Plan

No subagents; common-intersection logic is a single scientific invariant.

# Code Change Guardrails

Every estimator receives a seed; no per-model sample differences; results must
carry provenance and never overwrite prior runs silently.

# Acceptance Criteria

Two benchmark families run through one matched comparison workflow and emit
machine-readable evidence.

# Verification

Synthetic deterministic tests, CLI integration, and server execution receipt.

# Experiment Or Measurement

ROC-AUC mean/SD, fold values, coverage, sample counts, model dimension, seed.

# Risks

Some Table 1 label sources require stable external downloads and gene-ID
mapping; unavailable tasks must be explicitly reported rather than fabricated.
