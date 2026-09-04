# DinoGenePT Agent Guide

- Treat local Git as the source of truth; never edit source directly on the
  server.
- Run materializing downloads, embedding generation, and benchmarks only on
  `yilangliu@10.24.1.91` under `/data/yilangliu/DinoGenePT`.
- Never commit or print credentials, raw datasets, generated embeddings,
  checkpoints, or full run logs.
- Use `python -m pytest` for tests and `python -m dinogenept --help` for the
  CLI smoke check.
- Preserve matched inputs, intersections, seeds, and estimator settings across
  embedding models.
- Large/noisy paths: `.venv/`, `data/`, `results/`, `checkpoints/`, `*.npz`,
  `*.pickle`, `*.zip`, caches, and build artifacts.
