# AGENTS audit

- Reviewed: 2026-08-28
- Root guide is lean and points to the server-only execution boundary, test
  command, matched-evaluation invariant, and noisy/generated paths.
- No module-level guide is needed for this small package.
- Safe start directory is repository root locally and
  `/data/yilangliu/GenePT-Seed` on server.
- Subagents remain off because implementation and credential-sensitive server
  workflow overlap.
- Root `AGENTS.md` remains ready; `CLAUDE.md` now provides parity for the same
  source/server and secret boundaries.
- Scoped command coverage now includes corpus, GO, exact-axis, embedding, and
  benchmark modules through the codebase map and active plans.
- Current delivery explicitly excludes GraD-Pert training; server work is
  embedding generation and fixed GGI evaluation only.
- Next review: before another benchmark family or 2026-11-28.
