---
id: 001
title: Package and safety foundation
status: complete
wave: 1
updated_at: 2026-08-28T20:20:00+08:00
owner_role: Tech Lead
depends_on: []
start_directory: .
context_files: [AGENTS.md, .byte-os/TECH_SPEC.md]
agents_context_stack: [AGENTS.md]
subagent_policy: none
---

# Goal

Create the minimal installable package, CLI shell, safety defaults, and tests.

# OKR Link

KR4 and KR5.

# Scope

Packaging, CLI, config, hashing/manifests, ignores, test harness, README shell.

# Non-Goals

Data download, API calls, or scientific benchmark execution.

# Steps

## Step 1: Scaffold package and commands

- Purpose: establish one stable entry point.
- Actions: add `pyproject.toml`, package modules, CLI routing, and version.
- Files or modules: `pyproject.toml`, `src/dinogenept/`, `tests/`.
- Expected output: `dinogenept --help` works after installation.
- Step verification: `python -m dinogenept --help`.
- Subagent: none.

## Step 2: Add safety and provenance primitives

- Purpose: prevent secrets/large artifacts and make outputs auditable.
- Actions: add hashing, atomic JSON, manifests, credential-presence checks,
  `.gitignore`, and tests.
- Files or modules: `src/dinogenept/provenance.py`, `.gitignore`, tests.
- Expected output: deterministic manifests without secret values.
- Step verification: `python -m pytest tests/test_provenance.py`.
- Subagent: none.

## Step 3: Document the first workflow

- Purpose: make the package usable without session context.
- Actions: write README setup, local/server boundary, and credential warning.
- Files or modules: `README.md`.
- Expected output: exact install and dry-run commands.
- Step verification: inspect links and commands; run CLI help.
- Subagent: none.

# Dependencies

None.

# Scoped Commands

- Test: `python -m pytest tests/test_provenance.py tests/test_cli.py`
- Lint: `python -m ruff check .`
- Typecheck: not required in v0
- Build: `python -m build`

# AGENTS.md Context

- Root context: `AGENTS.md`.
- Module context: none.
- Scoped command source: root guide and this plan.
- Safe edit boundaries: package/tests/docs only.
- Missing or stale AGENTS.md notes: none.

# Subagent Plan

No subagents; foundational files overlap later work.

# Code Change Guardrails

Standard library first; no generic framework or secret persistence.

# Acceptance Criteria

Installable CLI, tests, ignores, provenance helpers, and onboarding exist.

# Verification

Unit tests, Ruff, package build, CLI smoke.

# Experiment Or Measurement

No experiment; measure test and package success.

# Risks

Overbuilding command abstractions before workflows stabilize.
