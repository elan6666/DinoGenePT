# Codebase map

- `src/genept_seed/data.py`: pinned server-side downloads and checksums.
- `src/genept_seed/embedding.py`: Ark client, text selection, SQLite resume,
  vector generation, and embedding manifests.
- `src/genept_seed/axis_corpus.py`: exact graph-axis NCBI/UniProt/HGNC corpus.
- `src/genept_seed/go_corpus.py`: bounded GO-EXP augmentation and receipts.
- `src/genept_seed/axis_vectors.py`: official GenePT exact-axis alignment.
- `src/genept_seed/vectors.py`: official pickle gate and safe NPZ format.
- `src/genept_seed/tasks.py`: property and fixed-split GGI parsing.
- `src/genept_seed/benchmarks.py`: matched classifiers and metrics.
- `src/genept_seed/cli.py`: user-facing workflow.
- `tests/`: offline unit/regression suite; no live API calls.
- `scripts/sync_code_to_server.sh`: checksum mirror, dry-run by default.
- `docs/`: protocol and evidence-bounded baseline summary.

Generated server-only paths are `data/`, `results/`, `checkpoints/`, `.venv/`,
and build outputs. They are not source and must not be synchronized from Mac.

Stack: Python 3.11+, NumPy, standard-library HTTP/SQLite, pytest, Ruff, and
Hatchling. Use Pyright/Pylance for symbol navigation when available; otherwise
use `rg` from the repository root. Scoped checks for axis work are
`pytest -q tests/test_axis_corpus.py tests/test_axis_vectors.py tests/test_go_corpus.py`.
