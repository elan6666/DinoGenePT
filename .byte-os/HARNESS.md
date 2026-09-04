# Codebase harness

- Reviewed: 2026-08-28
- Claude support: ready through `CLAUDE.md`.
- Codex support: ready through `AGENTS.md` and `.byte-os/CODEBASE_MAP.md`.
- Active start directories: repository root for orchestration and
  `src/dinogenept` for corpus/embedding changes.
- Noise filters: `.gitignore` excludes virtual environments, caches, data,
  embeddings, checkpoints, results, distributions, and scientific binaries.
- Scoped checks: axis/GO tests from `CODEBASE_MAP.md`; full test, Ruff, and build
  commands from `AGENTS.md` and active plans.
- LSP: Pyright/Pylance recommended; `rg` is the reliable fallback.
- Subagents: unavailable and inappropriate for the credentialed run; planned
  tracks remain explicit in plans 005--007.
- AGENTS quality: ready, lean, and pointer-oriented.
- Known gap: no persistent module guide is warranted for this small package.
