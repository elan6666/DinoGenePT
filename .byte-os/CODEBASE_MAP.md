# Codebase map

- `src/genept_seed/data.py`: pinned server-side downloads and checksums.
- `src/genept_seed/embedding.py`: Ark client, text selection, SQLite resume,
  vector generation, and embedding manifests.
- `src/genept_seed/vectors.py`: official pickle gate and safe NPZ format.
- `src/genept_seed/tasks.py`: property and fixed-split GGI parsing.
- `src/genept_seed/benchmarks.py`: matched classifiers and metrics.
- `src/genept_seed/cli.py`: user-facing workflow.
- `tests/`: offline unit/regression suite; no live API calls.
- `scripts/sync_code_to_server.sh`: checksum mirror, dry-run by default.
- `docs/`: protocol and evidence-bounded baseline summary.

Generated server-only paths are `data/`, `results/`, `checkpoints/`, `.venv/`,
and build outputs. They are not source and must not be synchronized from Mac.
