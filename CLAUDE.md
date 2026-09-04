# DinoGenePT Agent Guide

- Read `AGENTS.md` and `.byte-os/STATUS.md` first.
- Treat the Mac repository as source of truth; materializing work runs only at
  `/data/yilangliu/DinoGenePT`.
- Never persist or print credentials, raw data, embeddings, checkpoints, or
  full logs in Git.
- Preserve exact genes, order, splits, seeds, classifiers, and model settings
  across comparisons.
- Verify with `python -m pytest`, `python -m ruff check .`, package build, and
  server artifact receipts.
