# Codebase map

- `src/dinogenept/data.py`: pinned server-side downloads and checksums.
- `src/dinogenept/embedding.py`: Ark client, text selection, SQLite resume,
  vector generation, and embedding manifests.
- `src/dinogenept/axis_corpus.py`: exact graph-axis NCBI/UniProt/HGNC corpus.
- `src/dinogenept/go_corpus.py`: bounded GO-EXP augmentation and receipts.
- `src/dinogenept/axis_vectors.py`: official GenePT exact-axis alignment.
- `src/dinogenept/vectors.py`: official pickle gate and safe NPZ format.
- `src/dinogenept/tasks.py`: property and fixed-split GGI parsing.
- `src/dinogenept/benchmarks.py`: matched classifiers and metrics.
- `src/dinogenept/cli.py`: user-facing workflow.
- `tests/`: offline unit/regression suite; no live API calls.
- `scripts/sync_code_to_server.sh`: checksum mirror, dry-run by default.
- `docs/`: protocol and evidence-bounded baseline summary.

Generated server-only paths are `data/`, `results/`, `checkpoints/`, `.venv/`,
and build outputs. They are not source and must not be synchronized from Mac.

Stack: Python 3.11+, NumPy, standard-library HTTP/SQLite, pytest, Ruff, and
Hatchling. Use Pyright/Pylance for symbol navigation when available; otherwise
use `rg` from the repository root. Scoped checks for axis work are
`pytest -q tests/test_axis_corpus.py tests/test_axis_vectors.py tests/test_go_corpus.py`.
