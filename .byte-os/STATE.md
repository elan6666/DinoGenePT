# Active Byte Auto state

- Goal: finish Priority 1 pathway leakage controls and Priority 2 GenePT
  property tasks.
- Local Git remains the source of truth; server execution root is
  `/data/yilangliu/GenePT-Seed`.
- Reactome-only, SIGNOR-only, masked SIGNOR, and shuffled SIGNOR corpora and
  vectors are complete over the exact 17,730-label master universe. Every
  checkpoint is 17,730/17,730, pending 0, width 2,048.
- Fixed GGI, ten-seed strict gene-disjoint GGI, and all four property tasks for
  logistic regression and RF500 are complete. Final comparison auditors pass.
- The accepted gene-disjoint universe is exactly 10,870 genes, including 12
  isolated genes; invalid 11,299-gene preflights remain server-only and are
  excluded from final metrics.
- Compact receipts are under `docs/results/priority-1-2`; raw datasets,
  embeddings, checkpoints, benchmark rows, and logs remain server-only.
- Local and server tests, Ruff, build, CLI, JSON, script syntax, secret scan,
  and checksum sync pass. Review 6 verdict is `ship`.
- No GraD-Pert command was run or modified. User-owned untracked `uv.lock`
  remains excluded.
- Implementation/results commit `24e3bd07b1cc1be7fc498c628a691b0a3f2cdbd5`
  is pushed and verified on GitHub `main`. Only this metadata closeout remains
  before deleting the heartbeat and completing the goal.
