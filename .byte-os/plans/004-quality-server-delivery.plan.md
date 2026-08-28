---
id: 004
title: Quality, server synchronization, and delivery
status: complete
wave: 4
updated_at: 2026-08-28T20:40:00+08:00
owner_role: QA Engineer
depends_on: [003]
start_directory: .
context_files: [AGENTS.md, .byte-os/TECH_SPEC.md]
agents_context_stack: [AGENTS.md]
subagent_policy: none
---

# Goal

Verify the repository locally and on the server, document results, and publish.

# OKR Link

KR4 and KR5.

# Scope

Full checks, safe sync scripts/runbook, server environment, review repairs,
delivery docs, commit, and push.

# Non-Goals

Using the exposed key or claiming live results before a rotated key exists.

# Steps

## Step 1: Verify source locally

- Purpose: catch package and workflow defects before sync.
- Actions: run non-experimental source/diff/secret checks locally; run tests,
  lint, build, and CLI smoke on the server per the execution boundary.
- Files or modules: all tracked source/docs.
- Expected output: clean local verification receipt.
- Step verification: recorded command outputs.
- Subagent: none.

## Step 2: Synchronize and verify server mirror

- Purpose: enforce local-source/server-run boundary.
- Actions: rsync dry-run with exclusions, checksum sync, create isolated venv,
  install, run tests/doctor/dry-run, record environment and code hash.
- Files or modules: `scripts/`, runbook, build log.
- Expected output: server mirror matches local code and passes verification.
- Step verification: remote Git/source hash and test receipt.
- Subagent: none.

## Step 3: Package and publish

- Purpose: hand off a usable repository.
- Actions: current review, three iterations, delivery doc, commit, push, remote
  verification.
- Files or modules: `.byte-os/`, README, repository.
- Expected output: pushed main branch and delivery evidence.
- Step verification: commit hash, push output, clean status, remote hash.
- Subagent: none.

# Dependencies

Plan 003.

# Scoped Commands

- Test: `python -m pytest`
- Lint: `python -m ruff check .`
- Typecheck: not required
- Build: `python -m build`

# AGENTS.md Context

- Root context: `AGENTS.md`.
- Module context: none.
- Scoped command source: root guide and this plan.
- Safe edit boundaries: repository and explicit remote mirror only.
- Missing or stale AGENTS.md notes: none.

# Subagent Plan

No subagents due credential/server sensitivity and repository-wide verification.

# Code Change Guardrails

Dry-run before sync; exclude credentials/data/results; do not edit source on
server; do not publish unverified scientific claims.

# Acceptance Criteria

Local and server checks pass, delivery is documented, review says ship, and
GitHub points to the verified commit, or a precise credential blocker is logged.

# Verification

Tests, lint, build, CLI, secret scan, rsync receipt, remote tests, Git receipts.

# Experiment Or Measurement

Pass/fail evidence and runtime environment; live result availability is stated.

# Risks

Credential availability can block only the live embedding/result portion.

# Completion Evidence

Server Ruff, 18 tests, wheel/sdist build, doctor, and CLI smoke passed. Source
sync dry-run was empty after apply. Generated artifacts remained server-only,
the key was removed from tmux after use, and the final source was prepared for
GitHub delivery.
