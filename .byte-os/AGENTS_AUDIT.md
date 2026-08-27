# AGENTS audit

- Reviewed: 2026-08-28
- Root guide is lean and points to the server-only execution boundary, test
  command, matched-evaluation invariant, and noisy/generated paths.
- No module-level guide is needed for this small package.
- Safe start directory is repository root locally and
  `/data/yilangliu/GenePT-Seed` on server.
- Subagents remain off because implementation and credential-sensitive server
  workflow overlap.
- No update required after the current review.
