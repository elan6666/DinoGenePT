---
created_at: 2026-08-28T20:45:00+08:00
verdict: ship
---

# Verdict

ship

# Findings

- P0/P1: none.
- P2, QA: the documented default batch of 32 contradicted the live Agent Plan
  limit. Fixed by changing the default to 10 and adding parser regression
  coverage; verified on the live full run.
- P3, Research: evidence covers one released GGI split, not every GenePT task.
  The result docs now state this limitation and make no general-superiority
  claim.

# Role Notes

- Product/PM: the controlled-replacement goal is complete and tightly scoped.
- UX: README now provides a runnable server workflow and tested API settings.
- Tech Lead: concurrency changes are optional, bounded, and keep SQLite writes
  on the main thread; no embedding or benchmark semantics changed.
- QA: server Ruff, 18 tests, package build, CLI smoke, artifact audit, and sync
  no-op all passed.
- Research: counts and receipts match across all three L2 conditions; native
  sensitivity supports the primary result.
- Security: no key is tracked or logged; the tmux variable was unset after use.

# Required Changes

None.

# Suggested Changes

Add more GenePT tasks only as separately scoped future experiments.

# Verification Gaps

No real-user feedback and no multi-task generalization evidence. Neither is a
delivery gate for this research repository.

# Engineering Rule Findings

Changes are limited to the embedding client, CLI, tests, and result/runbook
documentation. Failed 400/429 assumptions were converted into verified defaults
and regression evidence rather than hidden.

# Harness Findings

Root AGENTS, codebase map, noise exclusions, start directories, and scoped
commands remain current. No module guide is needed.

# Subagent Findings

Subagents stayed off because credentialed execution and shared research files
overlapped.

# Decision

Run `byte-deliver`, then commit, push, and verify GitHub main.
