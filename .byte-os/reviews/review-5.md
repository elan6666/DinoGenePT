---
created_at: 2026-08-29T04:12:00+08:00
verdict: ship
---

# Verdict

ship

# Findings

- P0/P1/P2: none.
- P3, Research: the knowledge ladder is non-monotonic. HPA is above Seed+GO
  but below ProteinPathway on all three metrics. The result table and discussion
  state this directly and do not convert one GGI split into a general claim.
- P3, Data identity: eight master labels collide under case folding. Exact
  graph labels are retained, and fixed GGI selection prefers exact labels before
  allowing only a unique case-fold fallback.

# Role Notes

- Product Director: the delivered artifact answers the requested embedding and
  corpus question without expanding into downstream GraD-Pert training.
- Product Manager: all plan 008 acceptance criteria have direct corpus, vector,
  benchmark, test, and delivery evidence.
- Tech Lead: source remains local-Git-first; large artifacts remain server-only;
  the finalizer is locked, resumable, and checkpoint-gated.
- QA Engineer: 40 tests pass locally and on the server; Ruff, build, CLI, JSON,
  secret, receipt-hash, and checksum-sync gates pass.
- Research: all six GGI rows share 15 fairness fields. ProteinPathway is the
  best tested condition at 0.74968 Accuracy, 0.83615 AUROC, and 0.82859 AP.
- Security: no credential, raw dataset, checkpoint, vector, or full log is
  tracked. Only compact JSON receipts are copied to Git.

# Required Changes

None.

# Suggested Changes

Before a biological superiority claim, test another GenePT task or multiple
GGI splits. Before a GraD-Pert claim, run a separately scoped downstream
ablation using these frozen priors.

# Verification Gaps

No multi-seed interval, independent task replication, or GraD-Pert evaluation.
These are explicitly outside the current embedding-only delivery.

# Engineering Rule Findings

No unrelated refactor or hidden model substitution. The implementation adds
small builders/auditors, preserves existing defaults, uses deterministic sorted
bounded sections, and records every new materialization gate.

# Harness Findings

The existing codebase map, harness, noisy-path exclusions, and server-only run
boundary remain accurate. No module-level guide is warranted for this package.

# Subagent Findings

Subagent mode remained off because source files overlapped and the live API run
was credential-sensitive. No unreviewed handoff exists.

# Decision

Commit the tracked source/docs/receipts, push `main`, verify the remote commit,
delete the completed heartbeat, and mark the goal complete.
